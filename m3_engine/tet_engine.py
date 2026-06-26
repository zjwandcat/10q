# Copyright 2026 zjwandcat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
M3 TET 风控外挂核心引擎

实现 4 个核心指标：
  - Trend-Score (TS): 4 大类趋势因子分层投票
  - Emotion-Index (EI): 12 个振荡器按 direction 归一化
  - Anchored-Trend-Score (ATS): Schmitt Trigger 穿轴锚定
  - Timing: ATS - EI，< sell_threshold 触发 SELL_TET

状态机规则：
  - 规则 A: 新入选股票 ATS = 当期 TS
  - 规则 B: SELL_TET 后立即 pop state
  - 规则 C: 现金池不结转
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 配置类
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class M3Config:
    """M3 引擎配置（从 YAML 加载）"""

    sell_threshold: float
    hys_band: float
    risk_free_rate_annual: float
    rf_source: str
    scheme: str
    debug_csv: bool
    momentum: list[str]
    ma_ratio_gt1: list[str]
    ma_ratio_gt_threshold: dict[str, float]
    crossover: list[list]
    strength: list[str]
    emotion_indicators: list[dict]

    def __post_init__(self) -> None:
        assert self.sell_threshold < 0, "sell_threshold 必须为负数"
        assert 0.0 <= self.hys_band <= 0.2, "hys_band 应在 [0.0, 0.2]"
        assert len(self.emotion_indicators) > 0, "必须配置至少一个情绪振荡器"
        assert self.rf_source in ("fixed", "shibor_dynamic"), "rf_source 非法"
        for item in self.emotion_indicators:
            assert "name" in item and "direction" in item, (
                f"emotion_indicators 每项必须含 name 和 direction: {item}"
            )
            assert item["direction"] in ("positive", "negative"), (
                f"direction 必须是 positive 或 negative: {item}"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> "M3Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)["tet_params"]
        return cls(
            sell_threshold=float(raw["sell_threshold"]),
            hys_band=float(raw["hys_band"]),
            risk_free_rate_annual=float(raw["risk_free_rate_annual"]),
            rf_source=str(raw["rf_source"]),
            scheme=str(raw["scheme"]),
            debug_csv=bool(raw["debug_csv"]),
            momentum=list(raw["trend_factors"]["momentum"]),
            ma_ratio_gt1=list(raw["trend_factors"]["ma_ratio_gt1"]),
            ma_ratio_gt_threshold=dict(raw["trend_factors"]["ma_ratio_gt_threshold"]),
            crossover=list(raw["trend_factors"]["crossover"]),
            strength=list(raw["trend_factors"]["strength"]),
            emotion_indicators=list(raw["emotion_indicators"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 状态机数据结构
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class StockState:
    """单只股票的状态机"""

    prev_sign: int  # 上一期 EI 有效符号：-1 或 +1
    ats: float      # 上一期锚定的 TS


# ─────────────────────────────────────────────────────────────────────────────
# 模块级纯函数（Schmitt Trigger）
# ─────────────────────────────────────────────────────────────────────────────
def _determine_emotion_sign(curr_ei: float, prev_sign: int, hys_band: float) -> int:
    """Schmitt Trigger 符号判定。

    prev_sign ∈ {-1, +1}，首期建仓时由调用方先转成 ±1。
    """
    if prev_sign == 1 and curr_ei < -hys_band:
        return -1
    elif prev_sign == -1 and curr_ei > hys_band:
        return 1
    return prev_sign  # 落在滞后带内或未突破时，符号不变


def _has_crossed_zero(
    curr_ei: float, prev_sign: int, hys_band: float
) -> tuple[bool, int]:
    """返回 (是否穿轴, 新符号)。"""
    next_sign = _determine_emotion_sign(curr_ei, prev_sign, hys_band)
    crossed = next_sign != prev_sign
    return crossed, next_sign


# ─────────────────────────────────────────────────────────────────────────────
# TET 引擎
# ─────────────────────────────────────────────────────────────────────────────
class TETEngine:
    """TET 状态机引擎"""

    def __init__(self, config: M3Config) -> None:
        self.config = config
        self.state_dict: dict[str, StockState] = {}
        self.last_month_sold_set: set[str] = set()
        self._m0_cols: set[str] = set()

    # ──────────────── 指标计算 ────────────────
    def _calc_trend_score(self, row: pd.Series) -> float:
        """4 大类分层投票：momentum / ma_ratio / crossover / strength"""
        category_scores: list[float] = []

        # A. 动量收益率类
        votes_a: list[float] = []
        for c in self.config.momentum:
            if c in self._m0_cols and pd.notna(row.get(c)):
                votes_a.append(1.0 if row[c] > 0 else -1.0)
        if votes_a:
            category_scores.append(float(np.mean(votes_a)))

        # B. 均线偏离类（gt1 + threshold）
        # NOTE: M0 数据经 Z-score 中性化，原始 >1.0/>threshold 判定
        #       退化为 >0（Z-score 正=原始值高于截面均值）
        votes_b: list[float] = []
        for c in self.config.ma_ratio_gt1:
            if c in self._m0_cols and pd.notna(row.get(c)):
                votes_b.append(1.0 if row[c] > 0.0 else -1.0)
        for c, thr in self.config.ma_ratio_gt_threshold.items():
            if c in self._m0_cols and pd.notna(row.get(c)):
                votes_b.append(1.0 if row[c] > 0.0 else -1.0)
        if votes_b:
            category_scores.append(float(np.mean(votes_b)))

        # C. 均线交叉类
        votes_c: list[float] = []
        for pair in self.config.crossover:
            a, b = pair[0], pair[1]
            if a in self._m0_cols and pd.notna(row.get(a)):
                if b == 0:
                    votes_c.append(1.0 if row[a] > 0 else -1.0)
                elif b in self._m0_cols and pd.notna(row.get(b)):
                    votes_c.append(1.0 if row[a] > row[b] else -1.0)
        if votes_c:
            category_scores.append(float(np.mean(votes_c)))

        # D. 动量强度类
        votes_d: list[float] = []
        for c in self.config.strength:
            if c in self._m0_cols and pd.notna(row.get(c)):
                votes_d.append(1.0 if row[c] > 0 else -1.0)
        if votes_d:
            category_scores.append(float(np.mean(votes_d)))

        if not category_scores:
            logger.error("所有趋势因子大类均缺失！TS 强制返回 0.0")
            return 0.0
        return float(np.mean(category_scores))

    def _calc_emotion_index(self, row: pd.Series) -> float:
        """带 direction 的振荡器归一化（适配 Z-score 中性化数据）"""
        emotions: list[float] = []
        for item in self.config.emotion_indicators:
            col, direction = item["name"], item["direction"]
            if col not in self._m0_cols:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            val = float(val)
            if direction == "positive":
                emotions.append(float(np.clip(val, -3.0, 3.0)) / 3.0)
            else:
                emotions.append(float(np.clip(-val, -3.0, 3.0)) / 3.0)
        if not emotions:
            logger.warning("当月所有情绪因子缺失，EI 返回 0.0")
            return 0.0
        return float(np.mean(emotions))

    def _assign_m3_action(self, df_month: pd.DataFrame) -> pd.DataFrame:
        """向量化生成 m3_action 标签"""
        in_sold = df_month["stock_code"].isin(self.last_month_sold_set)
        df_month["m3_action"] = np.where(
            df_month["action"] == "SELL_TET",
            "SELL_TET",
            np.where(
                df_month["action"] == "CASH_POOL",
                "CASH_POOL",
                np.where(in_sold, "REENTER", "HOLD"),
            ),
        )
        return df_month

    # ──────────────── 单月处理 ────────────────
    def process_month(
        self,
        pred_month: str,
        m2_month: pl.DataFrame,
        m0_month: pl.DataFrame,
        monthly_rf: float,
    ) -> pl.DataFrame:
        current_stocks: set[str] = set(m2_month["stock_code"].to_list())

        # 规则 A: 清理上月未被本月选中的股票
        self.state_dict = {
            k: v for k, v in self.state_dict.items() if k in current_stocks
        }

        # 记录 M0 可用列（去 trade_date 防止 merge 列名冲突）
        m0_keep_cols = [c for c in m0_month.columns if c != "trade_date"]
        self._m0_cols = set(m0_keep_cols)

        # INCONSIST-03: 缺失因子 WARNING 日志
        all_config_factors = (
            set(self.config.momentum)
            | set(self.config.ma_ratio_gt1)
            | set(self.config.ma_ratio_gt_threshold.keys())
            | {p[0] for p in self.config.crossover}
            | set(self.config.strength)
            | {item["name"] for item in self.config.emotion_indicators}
        )
        missing_factors = all_config_factors - self._m0_cols
        if missing_factors:
            logger.warning(
                f"pred_month={pred_month} 缺失因子列: {sorted(missing_factors)}"
            )

        # 转 pandas 处理：M2 为主表，merge M0 因子
        m2_pd = m2_month.to_pandas()
        m0_pd = m0_month.select(m0_keep_cols).to_pandas()
        # trade_date 提取一次（用 M2 的决策日）
        m2_trade_date = str(m2_pd["trade_date"].iloc[0])

        # BUG FIX: M2 宽表已包含因子列，merge 时会加 _x/_y 后缀
        # 只保留 M2 中不存在的列（避免 merge 后缀冲突）
        m2_cols_set = set(m2_pd.columns)
        m0_unique_cols = [c for c in m0_pd.columns if c not in m2_cols_set]
        if "stock_code" not in m0_unique_cols:
            m0_unique_cols = ["stock_code"] + m0_unique_cols
        df = m2_pd.merge(m0_pd[m0_unique_cols], on="stock_code", how="left")

        results: list[dict] = []
        sold_this_month: set[str] = set()

        for _, row in df.iterrows():
            stock = row["stock_code"]
            ts = self._calc_trend_score(row)
            ei = self._calc_emotion_index(row)

            # 状态机推进
            state = self.state_dict.get(stock)
            if state is None:
                # 规则 A: 新入选股票
                curr_sign = 1 if ei >= 0 else -1
                ats = ts
                self.state_dict[stock] = StockState(prev_sign=curr_sign, ats=ats)
            else:
                crossed, next_sign = _has_crossed_zero(
                    ei, state.prev_sign, self.config.hys_band
                )
                if crossed:
                    ats = ts
                    state.prev_sign = next_sign
                    state.ats = ats
                else:
                    ats = state.ats

            timing = ats - ei

            # 卖出判定（严格小于）
            if timing < self.config.sell_threshold:
                action = "SELL_TET"
                adj_weight = 0.0
                cash_weight = float(row["weight"])
                # 规则 B: 立即抹除状态
                self.state_dict.pop(stock, None)
                sold_this_month.add(stock)
            else:
                action = "HOLD"
                adj_weight = float(row["weight"])
                cash_weight = 0.0

            results.append(
                {
                    "trade_date": m2_trade_date,
                    "pred_month": pred_month,
                    "stock_code": stock,
                    "stock_name": row.get("stock_name", "") or "",
                    "orig_weight": float(row["weight"]),
                    "adj_weight": adj_weight,
                    "cash_weight": cash_weight,
                    "action": action,
                    "trend_score": ts,
                    "emotion_index": ei,
                    "anchored_trend": ats,
                    "timing": timing,
                    "Target_Return_1M": float(row.get("Target_Return_1M") or 0.0),
                    "rf_rate_monthly": monthly_rf,
                }
            )

        df_month = pd.DataFrame(results)
        df_month = self._assign_m3_action(df_month)

        # 追加 CASH_POOL 行
        total_cash = float(df_month["cash_weight"].sum())
        if total_cash > 0:
            cash_row = pd.DataFrame(
                [
                    {
                        "trade_date": df_month["trade_date"].iloc[0],
                        "pred_month": pred_month,
                        "stock_code": "CASH_POOL",
                        "stock_name": "M3风控现金池",
                        "orig_weight": 0.0,
                        "adj_weight": total_cash,
                        "cash_weight": total_cash,
                        "action": "CASH_POOL",
                        "m3_action": "CASH_POOL",
                        "trend_score": None,
                        "emotion_index": None,
                        "anchored_trend": None,
                        "timing": None,
                        "Target_Return_1M": monthly_rf,
                        "rf_rate_monthly": monthly_rf,
                    }
                ]
            )
            df_month = pd.concat([df_month, cash_row], ignore_index=True)

        # 跨月传递
        self.last_month_sold_set = sold_this_month

        return pl.from_pandas(df_month)

    # ──────────────── 主循环 ────────────────
    def run(
        self, m2_path: Path, m0_dir: Path, output_path: Path
    ) -> Path:
        """月度滚动主循环"""
        # 读取 M2
        m2_df = load_m2_holdings(m2_path)
        pred_months = clean_and_sort_pred_months(m2_df)

        # 预计算每月所需 M0 列
        needed_cols = (
            list(self.config.momentum)
            + list(self.config.ma_ratio_gt1)
            + list(self.config.ma_ratio_gt_threshold.keys())
            + [p[0] for p in self.config.crossover]
            + list(self.config.strength)
            + [item["name"] for item in self.config.emotion_indicators]
            + ["stock_code", "trade_date", "macro_shibor_1m"]
        )
        needed_cols = list(dict.fromkeys(needed_cols))  # 去重保持顺序

        output_path.parent.mkdir(parents=True, exist_ok=True)
        debug_dir = output_path.parent / "m3_debug"

        all_months: list[pl.DataFrame] = []
        total_sell = 0
        total_reenter = 0

        for pred_month in tqdm(pred_months, desc="M3 月度处理"):
            m2_month = m2_df.filter(pl.col("pred_month") == pred_month)
            if m2_month.height == 0:
                continue

            m2_trade_date = m2_month["trade_date"][0]

            m0_path = m0_dir / f"{pred_month}.parquet"
            if not m0_path.exists():
                logger.warning(f"M0 文件缺失 {m0_path}, 跳过当月")
                continue

            # 时序校验
            verify_timeline(m2_trade_date, m0_path)

            # 读取 M0（含 macro_shibor_1m，fallback 用 risk_free_rate_annual）
            m0_month, monthly_rf = load_m0_month(
                pred_month, needed_cols, m0_dir, self.config.risk_free_rate_annual
            )
            if m0_month is None:
                continue

            # 单月处理
            out = self.process_month(pred_month, m2_month, m0_month, monthly_rf)
            all_months.append(out)

            # 统计
            actions = out["action"].to_list()
            sold = sum(1 for a in actions if a == "SELL_TET")
            reentered = sum(
                1 for a in out["m3_action"].to_list() if a == "REENTER"
            )
            total_sell += sold
            total_reenter += reentered

            cash_total = float(
                out.filter(pl.col("stock_code") == "CASH_POOL")["adj_weight"].sum()
            )
            logger.info(
                f"Processing {pred_month}: {m2_month.height} stocks, "
                f"{sold} sold, cash={cash_total:.2f}"
            )

            # Debug CSV 落盘
            if self.config.debug_csv:
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"{pred_month}.csv"
                out.write_csv(debug_path)

        # 合并并写盘
        if all_months:
            final = pl.concat(all_months, how="vertical_relaxed")
            final.write_parquet(output_path)
        else:
            # 空表占位
            final = pl.DataFrame(schema=_OUTPUT_SCHEMA)
            final.write_parquet(output_path)

        logger.info(
            f"M3 完成：总月数={len(all_months)}, "
            f"累计 SELL_TET={total_sell}, 累计 REENTER={total_reenter}, "
            f"输出={output_path}"
        )
        return output_path


# ─────────────────────────────────────────────────────────────────────────────
# IO 函数（模块级）
# ─────────────────────────────────────────────────────────────────────────────
_OUTPUT_SCHEMA = {
    "trade_date": pl.Utf8,
    "pred_month": pl.Utf8,
    "stock_code": pl.Utf8,
    "stock_name": pl.Utf8,
    "orig_weight": pl.Float64,
    "adj_weight": pl.Float64,
    "cash_weight": pl.Float64,
    "action": pl.Utf8,
    "m3_action": pl.Utf8,
    "trend_score": pl.Float64,
    "emotion_index": pl.Float64,
    "anchored_trend": pl.Float64,
    "timing": pl.Float64,
    "Target_Return_1M": pl.Float64,
    "rf_rate_monthly": pl.Float64,
}


def load_m2_holdings(path: Path) -> pl.DataFrame:
    """读取 M2 持仓，过滤 is_holding == True，stock_name 缺失填空字符串。"""
    df = pl.read_parquet(path)
    if "is_holding" in df.columns:
        df = df.filter(pl.col("is_holding") == True)  # noqa: E712
    if "stock_name" not in df.columns:
        df = df.with_columns(pl.lit("").alias("stock_name"))
    else:
        df = df.with_columns(pl.col("stock_name").fill_null(""))
    return df


def clean_and_sort_pred_months(df: pl.DataFrame) -> list[str]:
    """提取 6 位纯数字 pred_month，按 int 排序。"""
    pattern = re.compile(r"^\d{6}$")
    raw = df["pred_month"].unique().to_list()
    cleaned: list[str] = []
    for v in raw:
        s = str(v).strip()
        assert pattern.match(s), f"pred_month 格式非法（非 6 位纯数字）: {v!r}"
        cleaned.append(s)
    return sorted(set(cleaned), key=int)


def verify_timeline(m2_trade_date: str, m0_path: Path) -> None:
    """前视偏差校验：m0_trade_date <= m2_trade_date。"""
    try:
        m0_trade_date = (
            pl.scan_parquet(m0_path)
            .select("trade_date")
            .head(1)
            .collect()
            .item()
        )
    except Exception as e:
        raise ValueError(f"读取 M0 trade_date 失败 {m0_path}: {e}") from e

    m0_dt = datetime.strptime(str(m0_trade_date).split()[0], "%Y-%m-%d")
    m2_dt = datetime.strptime(str(m2_trade_date).split()[0], "%Y-%m-%d")
    if m0_dt > m2_dt:
        raise ValueError(
            f"M0 数据存在前视偏差: m0_trade_date={m0_trade_date} > "
            f"m2_trade_date={m2_trade_date}"
        )


def load_m0_month(
    pred_month: str,
    needed_cols: list[str],
    m0_dir: Path,
    risk_free_rate_annual: float = 0.025,
) -> tuple[pl.DataFrame | None, float]:
    """读取 M0 月度因子，返回 (DataFrame, monthly_rf)。

    Args:
        pred_month: 预测月份字符串 "YYYYMM"
        needed_cols: 需要的因子列清单
        m0_dir: M0 parquet 所在目录
        risk_free_rate_annual: Shibor 缺失时的 fallback 年化无风险利率
    """
    m0_path = m0_dir / f"{pred_month}.parquet"
    if not m0_path.exists():
        logger.warning(f"M0 文件缺失 {m0_path}，跳过当月")
        return None, 0.0

    try:
        schema_names = pl.scan_parquet(m0_path).collect_schema().names()
    except Exception as e:
        logger.error(f"读取 M0 schema 失败 {m0_path}: {e}")
        return None, 0.0

    available = [c for c in needed_cols if c in schema_names]
    missing = [c for c in needed_cols if c not in schema_names]
    if missing:
        logger.warning(f"M0 缺失列 {missing}（已跳过）")

    df = pl.read_parquet(m0_path, columns=available)

    # BUG-02 修复：fallback 使用参数化风险利率
    fallback = risk_free_rate_annual / 12.0
    monthly_rf = fallback
    if "macro_shibor_1m" in df.columns:
        try:
            shibor_val = df["macro_shibor_1m"].drop_nulls()
            if shibor_val.len() > 0:
                monthly_rf = float(shibor_val[0]) / 100.0 / 12.0
        except Exception:
            monthly_rf = fallback

    return df, monthly_rf
