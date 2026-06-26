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
M4 回测报告生成器（iOS 26 风格）
================================

* 净值曲线：Chart.js + 时间区间滑块 + 鼠标 hover 提示
* 移除门槛检验（按需求 2026-06）
* 增加 Brinson / Five-Factor / Barra 归因
* 增加 SHAP 因子归因（Top10 持仓 × Top5 因子）
* 增加「月度持仓」交互页：选年-月即可看 10 只 + 13%/7% 分层 + 月度个股收益
* v5.4: 完全离线可用，Chart.js 内嵌 + 服务端 SVG 兜底
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics

logger = logging.getLogger("m4.report")

# ★ v5.4: 内嵌 Chart.js 用于离线生成
_STATIC_DIR = Path(__file__).parent / "static"
_CHARTJS_CACHE: Optional[str] = None


def _get_chartjs_inline() -> str:
    """读取本地 Chart.js, 内嵌到 HTML 里 (解决 CDN 拉不到的问题)"""
    global _CHARTJS_CACHE
    if _CHARTJS_CACHE is not None:
        return _CHARTJS_CACHE
    p = _STATIC_DIR / "chart.umd.min.js"
    if not p.exists():
        logger.warning(
            f"未找到本地 Chart.js: {p}, "
            "将回退到 CDN 模式")
        return ""
    try:
        content = p.read_text(encoding="utf-8")
        # ★ 修复: 检查文件内容是否有效
        if len(content) < 1000:
            logger.warning(f"Chart.js 文件内容过短: {len(content)} 字节")
            return ""
        _CHARTJS_CACHE = content
        logger.info(
            f"已加载本地 Chart.js: "
            f"{len(_CHARTJS_CACHE)/1024:.1f} KB")
        return _CHARTJS_CACHE
    except Exception as e:
        logger.warning(f"读本地 Chart.js 失败: {e}")
        return ""


# ──────────────────────────────────────────────────────
#  工具：色板（iOS 26 Liquid Glass 风格）
# ──────────────────────────────────────────────────────
PALETTE = {
    "port":        "#0A84FF",   # iOS systemBlue
    "port_grad":   "rgba(10, 132, 255, 0.18)",
    "bm":          "#FF9F0A",   # iOS systemOrange
    "bm_grad":     "rgba(255, 159, 10, 0.12)",
    "alpha":       "#30D158",   # systemGreen
    "alpha_grad":  "rgba(48, 209, 88, 0.18)",
    "warn":        "#FF453A",   # systemRed
    "warn_grad":   "rgba(255, 69, 58, 0.18)",
    "tint":        "#BF5AF2",   # systemPurple
    "tint_grad":   "rgba(191, 90, 242, 0.18)",
    "ink":         "#1C1C1E",   # label
    "ink2":        "#3A3A3C",
    "ink3":        "#8E8E93",
    "bg":          "#F2F2F7",   # systemGroupedBackground
    "card":        "rgba(255, 255, 255, 0.78)",
    "card_stroke": "rgba(255, 255, 255, 0.4)",
    "shadow":      "rgba(0, 0, 0, 0.06)",
}


def _safe_df(d) -> pd.DataFrame:
    if d is None:
        return pd.DataFrame()
    if isinstance(d, pd.DataFrame):
        return d.copy()
    return pd.DataFrame(d)


def _format_m3_sell_reason(row) -> str:
    action = row.get("m3_action", None)
    if action is None or (isinstance(action, float) and pd.isna(action)):
        return "N/A"
    action = str(action)
    if action == "SELL_TET":
        timing = _safe_float(row.get("timing", 0.0), 0.0)
        ts = _safe_float(row.get("trend_score", 0.0), 0.0)
        ei = _safe_float(row.get("emotion_index", 0.0), 0.0)
        ats = _safe_float(row.get("anchored_trend", 0.0), 0.0)
        sell_threshold = _safe_float(row.get("sell_threshold", 0.0), 0.0)
        return f"Timing={timing:.3f} < 阈值{sell_threshold:.1f}, TS={ts:.3f}, EI={ei:.3f}, ATS={ats:.3f}"
    elif action == "CASH_POOL":
        return "现金池"
    elif action == "REENTER":
        return "上月卖出本月重新入选"
    elif action == "HOLD":
        return "—"
    return "N/A"


def _safe_float(v, default: float = 0.0) -> float:
    """
    ★ v5.5 修复: 把 NaN / inf / None / 非数值 全部安全转 float.
    历史: float(NaN) 是真值, `or 0` 失效, json.dumps 输出非法字面量
    `NaN`/`Infinity`, 浏览器 JSON.parse 报错, 整个 report 挂掉.
    """
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    import math
    if not math.isfinite(f):
        return default
    return f


# ──────────────────────────────────────────────────────
#  ReportGenerator
# ──────────────────────────────────────────────────────
class ReportGenerator:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.pm = PerformanceMetrics()

    # ── 入口 ───────────────────────────────────────────
    def generate(
        self,
        all_portfolios: pd.DataFrame,
        output_path: str = "output/backtest_report.html",
        stats: Optional[Dict] = None,
        title: str = "TTHH量化回测报告",
        factor_df: Optional[pd.DataFrame] = None,
        shap_data: Optional[Dict] = None,
        m3_portfolios: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        生成 iOS 26 风格单页报告。

        参数:
            all_portfolios : M2 输出
            output_path    : HTML 落盘路径
            stats          : M2 stats dict
            title          : 报告标题
            factor_df      : M0 全量因子（带 industry/Barra），
                             用于 Brinson / Barra 归因
            shap_data      : {pred_month: (shap_values, shap_features, source)}
                             来自 run_m2 挂到 all_portfolios.attrs['shap_data']
                             （向后兼容：未传时尝试从 attrs 读）
        """
        p = Path(output_path)
        output_path = str(p)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # ★ 改造: 优先用调用方显式传的 shap_data；否则尝试从
        #   all_portfolios.attrs 读（M2 run_m2 挂上去的）
        if shap_data is None:
            shap_data = all_portfolios.attrs.get("shap_data", None)

        # ── 月度收益 / 28+ 指标 ──
        monthly = self.pm.compute_monthly_returns(all_portfolios)
        metrics = self.pm.calculate(monthly)
        cum_r   = (1 + monthly["portfolio_return"]).cumprod()
        cum_bm  = (1 + monthly["benchmark_return"]).cumprod()

        holdings_all = all_portfolios[
            all_portfolios["is_holding"] == True].copy()
        holdings_all = holdings_all.dropna(
            subset=["Target_Return_1M"])

        tier_high_ret = (
            holdings_all[holdings_all["tier"] == "High"]
            .groupby("pred_month")
            .apply(lambda g: pd.Series({
                "high_return": (
                    g["weight"] / g["weight"].sum() * g["Target_Return_1M"]
                ).sum() if g["weight"].sum() > 0 else 0.0
            }), include_groups=False)
            .reset_index()
        )
        tier_low_ret = (
            holdings_all[holdings_all["tier"] == "Low"]
            .groupby("pred_month")
            .apply(lambda g: pd.Series({
                "low_return": (
                    g["weight"] / g["weight"].sum() * g["Target_Return_1M"]
                ).sum() if g["weight"].sum() > 0 else 0.0
            }), include_groups=False)
            .reset_index()
        )
        tier_merged = monthly[["pred_month", "portfolio_return",
                                "benchmark_return"]].copy()
        tier_high_ret.attrs = {}
        tier_low_ret.attrs = {}
        tier_merged = tier_merged.merge(
            tier_high_ret, on="pred_month", how="left")
        tier_merged = tier_merged.merge(
            tier_low_ret, on="pred_month", how="left")
        tier_merged["high_return"] = tier_merged[
            "high_return"].fillna(0.0)
        tier_merged["low_return"] = tier_merged[
            "low_return"].fillna(0.0)
        cum_high = (1 + tier_merged["high_return"]).cumprod()
        cum_low  = (1 + tier_merged["low_return"]).cumprod()

        dd_port = ((cum_r / np.maximum.accumulate(cum_r)) - 1)
        dd_bm   = ((cum_bm / np.maximum.accumulate(cum_bm)) - 1)
        dd_high = ((cum_high / np.maximum.accumulate(cum_high)) - 1)
        dd_low  = ((cum_low / np.maximum.accumulate(cum_low)) - 1)

        months = monthly["pred_month"].astype(str).tolist()
        port_v = cum_r.round(6).tolist()
        bm_v   = cum_bm.round(6).tolist()
        cum_ex  = (1 + monthly["excess_return"]).cumprod()
        ex_v   = cum_ex.round(6).tolist()
        high_v = cum_high.round(6).tolist()
        low_v  = cum_low.round(6).tolist()

        dd_port_v = dd_port.round(6).tolist()
        dd_bm_v   = dd_bm.round(6).tolist()
        dd_high_v = dd_high.round(6).tolist()
        dd_low_v  = dd_low.round(6).tolist()

        monthly_ret = monthly["portfolio_return"].round(6).tolist()
        monthly_ex  = monthly["excess_return"].round(6).tolist()
        monthly_high_ret = tier_merged["high_return"].round(6).tolist()
        monthly_low_ret  = tier_merged["low_return"].round(6).tolist()

        attribution = self.pm.run_attribution(
            all_portfolios, monthly, factor_df=factor_df)

        summary_table = self._build_summary_table(stats)

        perf_rows = self._build_perf_rows(metrics)
        perf_data = self._perf_row_data(metrics)

        brinson_json = self._brinson_to_json(
            attribution.get("brinson", pd.DataFrame()))
        ff_json = self._ff_to_json(
            attribution.get("five_factor", pd.DataFrame()))
        barra_json = self._barra_to_json(
            attribution.get("barra", pd.DataFrame()))

        effective_shap = (shap_data
                          or attribution.get("shap_bundle")
                          or {})
        shap_data_json = self._shap_to_json(
            effective_shap,
            attribution.get("holdings"))

        holdings_df = attribution.get("holdings", pd.DataFrame())
        holdings_json = self._holdings_to_json(holdings_df, m3_data=m3_portfolios)

        month_options = self._month_options(months)

        nav_svg_fallback = self._render_nav_svg_fallback(
            months, port_v, bm_v)
        brinson_d = json.loads(brinson_json)
        ff_d = json.loads(ff_json)
        barra_d = json.loads(barra_json)
        brinson_fb = self._render_attrib_table_fallback(brinson_d)
        ff_fb = self._render_ff_table_fallback(ff_d)
        barra_fb = self._render_barra_table_fallback(barra_d)

        chartjs_inline = _get_chartjs_inline()
        use_inline = len(chartjs_inline) > 0
        logger.info(
            f"Chart.js 加载模式: "
            f"{'内嵌本地' if use_inline else 'CDN'}")

        summary_cards_m2 = summary_table
        summary_cards_m3 = ""
        m3_data_json = "null"
        m3_holdings_merge = None
        has_m3_data = False
        aligned_months_json = "[]"
        alignment_banner = ""

        if m3_portfolios is not None and not m3_portfolios.empty:
            has_m3_data = True
            m3_monthly = self._compute_m3_monthly_returns(m3_portfolios, monthly)
            m3_metrics = self.pm.calculate(m3_monthly)
            m3_cum_r = (1 + m3_monthly["portfolio_return"]).cumprod()
            m3_cum_bm = (1 + m3_monthly["benchmark_return"]).cumprod()

            m3_holdings_all = m3_portfolios[
                m3_portfolios["stock_code"] != "CASH_POOL"].copy()
            if "tier" not in m3_holdings_all.columns and "tier" in all_portfolios.columns:
                tier_map = all_portfolios[all_portfolios["is_holding"] == True][
                    ["pred_month", "stock_code", "tier"]
                ].drop_duplicates(subset=["pred_month", "stock_code"])
                m3_holdings_all = m3_holdings_all.merge(
                    tier_map, on=["pred_month", "stock_code"], how="left")
                m3_holdings_all["tier"] = m3_holdings_all["tier"].fillna("Low")
            m3_holdings_all = m3_holdings_all.dropna(
                subset=["Target_Return_1M"])
            m3_weight_col = "adj_weight" if "adj_weight" in m3_holdings_all.columns else "weight"

            m3_tier_high_ret = (
                m3_holdings_all[m3_holdings_all["tier"] == "High"]
                .groupby("pred_month")
                .apply(lambda g: pd.Series({
                    "high_return": (
                        g[m3_weight_col] / g[m3_weight_col].sum() * g["Target_Return_1M"]
                    ).sum() if g[m3_weight_col].sum() > 0 else 0.0
                }), include_groups=False)
                .reset_index()
            )
            m3_tier_low_ret = (
                m3_holdings_all[m3_holdings_all["tier"] == "Low"]
                .groupby("pred_month")
                .apply(lambda g: pd.Series({
                    "low_return": (
                        g[m3_weight_col] / g[m3_weight_col].sum() * g["Target_Return_1M"]
                    ).sum() if g[m3_weight_col].sum() > 0 else 0.0
                }), include_groups=False)
                .reset_index()
            )
            m3_tier_merged = m3_monthly[["pred_month", "portfolio_return",
                                          "benchmark_return"]].copy()
            m3_tier_high_ret.attrs = {}
            m3_tier_low_ret.attrs = {}
            m3_tier_merged = m3_tier_merged.merge(
                m3_tier_high_ret, on="pred_month", how="left")
            m3_tier_merged = m3_tier_merged.merge(
                m3_tier_low_ret, on="pred_month", how="left")
            m3_tier_merged["high_return"] = m3_tier_merged[
                "high_return"].fillna(0.0)
            m3_tier_merged["low_return"] = m3_tier_merged[
                "low_return"].fillna(0.0)
            m3_cum_high = (1 + m3_tier_merged["high_return"]).cumprod()
            m3_cum_low  = (1 + m3_tier_merged["low_return"]).cumprod()

            m3_dd_port = ((m3_cum_r / np.maximum.accumulate(m3_cum_r)) - 1)
            m3_dd_bm   = ((m3_cum_bm / np.maximum.accumulate(m3_cum_bm)) - 1)
            m3_dd_high = ((m3_cum_high / np.maximum.accumulate(m3_cum_high)) - 1)
            m3_dd_low  = ((m3_cum_low / np.maximum.accumulate(m3_cum_low)) - 1)

            m3_months = m3_monthly["pred_month"].astype(str).tolist()
            aligned, alignment_banner = self._align_months(months, m3_months)
            aligned_months_json = json.dumps(aligned, ensure_ascii=False)

            if aligned:
                align_set = set(aligned)
                m3_mask = m3_monthly["pred_month"].astype(str).isin(align_set)
                m3_monthly_aligned = m3_monthly[m3_mask].reset_index(drop=True)
                m3_cum_r_a = (1 + m3_monthly_aligned["portfolio_return"]).cumprod()
                m3_cum_bm_a = (1 + m3_monthly_aligned["benchmark_return"]).cumprod()

                m3_tier_a = m3_tier_merged[
                    m3_tier_merged["pred_month"].astype(str).isin(align_set)
                ].reset_index(drop=True)
                m3_cum_high_a = (1 + m3_tier_a["high_return"].fillna(0.0)).cumprod()
                m3_cum_low_a  = (1 + m3_tier_a["low_return"].fillna(0.0)).cumprod()

                m3_dd_port_a = ((m3_cum_r_a / np.maximum.accumulate(m3_cum_r_a)) - 1)
                m3_dd_bm_a   = ((m3_cum_bm_a / np.maximum.accumulate(m3_cum_bm_a)) - 1)
                m3_dd_high_a = ((m3_cum_high_a / np.maximum.accumulate(m3_cum_high_a)) - 1)
                m3_dd_low_a  = ((m3_cum_low_a / np.maximum.accumulate(m3_cum_low_a)) - 1)

                m3_cum_ex_a = (1 + m3_monthly_aligned["excess_return"]).cumprod()
                m3_port_v = m3_cum_r_a.round(6).tolist()
                m3_bm_v   = m3_cum_bm_a.round(6).tolist()
                m3_ex_v   = m3_cum_ex_a.round(6).tolist()
                m3_high_v = m3_cum_high_a.round(6).tolist()
                m3_low_v  = m3_cum_low_a.round(6).tolist()
                m3_dd_port_v = m3_dd_port_a.round(6).tolist()
                m3_dd_bm_v   = m3_dd_bm_a.round(6).tolist()
                m3_dd_high_v = m3_dd_high_a.round(6).tolist()
                m3_dd_low_v  = m3_dd_low_a.round(6).tolist()
                m3_monthly_ret = m3_monthly_aligned["portfolio_return"].round(6).tolist()
                m3_monthly_ex  = m3_monthly_aligned["excess_return"].round(6).tolist()
                m3_monthly_high_ret = m3_tier_a["high_return"].round(6).tolist()
                m3_monthly_low_ret  = m3_tier_a["low_return"].round(6).tolist()
                m3_months_aligned = m3_monthly_aligned["pred_month"].astype(str).tolist()
            else:
                m3_cum_ex = (1 + m3_monthly["excess_return"]).cumprod()
                m3_port_v = m3_cum_r.round(6).tolist()
                m3_bm_v   = m3_cum_bm.round(6).tolist()
                m3_ex_v   = m3_cum_ex.round(6).tolist()
                m3_high_v = m3_cum_high.round(6).tolist()
                m3_low_v  = m3_cum_low.round(6).tolist()
                m3_dd_port_v = m3_dd_port.round(6).tolist()
                m3_dd_bm_v   = m3_dd_bm.round(6).tolist()
                m3_dd_high_v = m3_dd_high.round(6).tolist()
                m3_dd_low_v  = m3_dd_low.round(6).tolist()
                m3_monthly_ret = m3_monthly["portfolio_return"].round(6).tolist()
                m3_monthly_ex  = m3_monthly["excess_return"].round(6).tolist()
                m3_monthly_high_ret = m3_tier_merged["high_return"].round(6).tolist()
                m3_monthly_low_ret  = m3_tier_merged["low_return"].round(6).tolist()
                m3_months_aligned = m3_months

            m3_perf_rows = self._build_perf_rows(m3_metrics)
            m3_perf_data = self._perf_row_data(m3_metrics)
            summary_cards_m3 = m3_perf_rows

            m3_brinson_json = brinson_json
            m3_ff_json = ff_json
            m3_barra_json = barra_json

            m3_holdings_json = holdings_json
            m3_holdings_merge = m3_portfolios

            m3_data_dict = {
                "months": m3_months_aligned,
                "port":   m3_port_v,
                "bm":     m3_bm_v,
                "ex":     m3_ex_v,
                "high":   m3_high_v,
                "low":    m3_low_v,
                "dd_port": m3_dd_port_v,
                "dd_bm":   m3_dd_bm_v,
                "dd_high": m3_dd_high_v,
                "dd_low":  m3_dd_low_v,
                "monthly_ret": m3_monthly_ret,
                "monthly_ex":  m3_monthly_ex,
                "monthly_high_ret": m3_monthly_high_ret,
                "monthly_low_ret":  m3_monthly_low_ret,
                "perf_rows": m3_perf_data,
                "brinson": json.loads(m3_brinson_json),
                "ff":      json.loads(m3_ff_json),
                "barra":   json.loads(m3_barra_json),
                "shap":    json.loads(shap_data_json),
                "holdings": json.loads(m3_holdings_json),
                "shap_degraded": False,
            }
            m3_data_json = json.dumps(m3_data_dict, ensure_ascii=False, allow_nan=False)

        html = self._render_html(
            title=title,
            months=months,
            port_v=port_v,
            bm_v=bm_v,
            ex_v=ex_v,
            high_v=high_v,
            low_v=low_v,
            dd_port_v=dd_port_v,
            dd_bm_v=dd_bm_v,
            dd_high_v=dd_high_v,
            dd_low_v=dd_low_v,
            monthly_ret=monthly_ret,
            monthly_ex=monthly_ex,
            monthly_high_ret=monthly_high_ret,
            monthly_low_ret=monthly_low_ret,
            summary_table=summary_table,
            perf_rows=perf_rows,
            perf_data=perf_data,
            brinson_json=brinson_json,
            ff_json=ff_json,
            barra_json=barra_json,
            shap_data_json=shap_data_json,
            holdings_json=holdings_json,
            month_options=month_options,
            nav_svg_fallback=nav_svg_fallback,
            brinson_fallback=brinson_fb,
            ff_fallback=ff_fb,
            barra_fallback=barra_fb,
            chartjs_inline=chartjs_inline,
            has_m3_data=has_m3_data,
            m3_data_json=m3_data_json,
            aligned_months_json=aligned_months_json,
            alignment_banner=alignment_banner,
            summary_cards_m2=summary_cards_m2,
            summary_cards_m3=summary_cards_m3,
        )

        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"iOS 26 报告已生成: {output_path}")

        result = {
            "report_path": str(output_path),
            "metrics":     metrics,
            "attribution": {
                k: (v.to_dict("records")
                    if isinstance(v, pd.DataFrame) else v)
                for k, v in attribution.items()
                if k not in ("holdings",)
            },
            "summary_cards_m2": summary_cards_m2,
            "summary_cards_m3": summary_cards_m3,
            "m3_metrics": m3_metrics if has_m3_data else None,
        }
        return result

    # ── 内部：构造各 section 的 HTML 片段 ────────────────
    def _compute_m3_monthly_returns(self, m3_portfolios: pd.DataFrame, m2_monthly: pd.DataFrame) -> pd.DataFrame:
        m3_returns_by_pm = {}
        for pm, g in m3_portfolios.groupby("pred_month"):
            non_cash = g[g["stock_code"] != "CASH_POOL"]
            cash = g[g["stock_code"] == "CASH_POOL"]
            stock_return = 0.0
            for _, r in non_cash.iterrows():
                aw = _safe_float(r.get("adj_weight", r.get("weight", 0)), 0.0)
                ret = _safe_float(r.get("Target_Return_1M", 0), 0.0)
                stock_return += aw * (1 + ret)
            cash_return = 0.0
            if not cash.empty:
                for _, r in cash.iterrows():
                    cw = _safe_float(r.get("adj_weight", r.get("weight", 0)), 0.0)
                    rf = _safe_float(r.get("rf_rate_monthly", float("nan")), 0.0)
                    if pd.isna(rf) or rf == 0.0:
                        rf = 0.025 / 12
                    cash_return += cw * (1 + rf)
            m3_returns_by_pm[pm] = stock_return + cash_return - 1.0
        result = m2_monthly[["pred_month", "portfolio_return", "benchmark_return"]].copy()
        result.rename(columns={"portfolio_return": "m2_return"}, inplace=True)
        result["portfolio_return"] = result["pred_month"].map(m3_returns_by_pm).fillna(result["m2_return"])
        result["excess_return"] = result["portfolio_return"] - result["benchmark_return"]
        result = result[["pred_month", "portfolio_return", "benchmark_return", "excess_return"]]
        return result.sort_values("pred_month").reset_index(drop=True)

    def _align_months(self, m2_months: list, m3_months: list):
        m2_set = set(m2_months)
        m3_set = set(m3_months)
        intersection = sorted(m2_set & m3_set, key=int)
        if m2_set == m3_set:
            return intersection, ""
        m2_only = sorted(m2_set - m3_set, key=int)
        m3_only = sorted(m3_set - m2_set, key=int)
        banner_parts = []
        if m2_only:
            banner_parts.append(f"M2独有月份: {', '.join(m2_only)}")
        if m3_only:
            banner_parts.append(f"M3独有月份: {', '.join(m3_only)}")
        banner = " | ".join(banner_parts) + f" | 已对齐至交集({len(intersection)}个月)"
        return intersection, banner

    def _preprocess_m3_for_attribution(self, holdings_df: pd.DataFrame) -> pd.DataFrame:
        df = holdings_df.copy()
        if "adj_weight" in df.columns and "weight" not in df.columns:
            df["weight"] = df["adj_weight"]
        cash_mask = df["stock_code"] == "CASH_POOL"
        df.loc[cash_mask, "industry"] = "CASH"
        expo_cols = [c for c in df.columns if c.startswith("expo_")]
        for c in expo_cols:
            df.loc[cash_mask, c] = 0.0
        factor_cols = [c for c in df.columns
                       if c not in ["pred_month", "stock_code", "stock_name",
                                    "industry", "tier", "weight", "adj_weight",
                                    "is_holding", "Target_Return_1M",
                                    "benchmark_return", "contribution", "score",
                                    "m3_action", "timing", "trend_score",
                                    "emotion_index", "anchored_trend", "sell_threshold",
                                    "rf_rate_monthly"]
                       and not c.startswith("expo_")]
        for c in factor_cols:
            if df[c].dtype in [np.float64, np.float32, float]:
                df[c] = df[c].fillna(0.0)
        return df

    def _build_summary_table(self, stats: Optional[Dict]) -> str:
        if not stats:
            return "<div class='kv-empty'>（未提供 M2 窗口统计）</div>"
        ic_list = stats.get("avg_val_ic", [])
        avg_ic  = float(np.mean(ic_list)) if ic_list else 0.0
        icir    = stats.get("avg_val_icir", "-")
        success = stats.get("success", 1)
        low     = stats.get("low_confidence_months", 0)
        low_pct = f"{low / success:.2%}" if success else "-"
        rows = [
            ("总窗口数",     stats.get("total_windows", "-")),
            ("成功窗口",     success),
            ("低置信度月份", f"{low} ({low_pct})"),
            ("平均 val_IC",  f"{avg_ic:+.4f}"),
            ("ICIR",         icir),
        ]
        return self._kv_grid(rows)

    def _build_perf_rows(self, m: Dict) -> str:
        rows = self._perf_row_data(m)
        return self._kv_grid(rows, ncol=5)

    def _perf_row_data(self, m: Dict) -> list:
        return [
            ("年化收益 CAGR",          f"{m['cagr']:.2%}"),
            ("年化超额",               f"{m['annual_excess']:.2%}"),
            ("最大回撤",               f"{m['max_drawdown']:.2%}"),
            ("年化波动率",             f"{m['volatility']:.2%}"),
            ("下行波动率",             f"{m['downside_volatility']:.2%}"),
            ("上行波动率",             f"{m['upside_volatility']:.2%}"),
            ("夏普比率",               f"{m['sharpe_ratio']:.3f}"),
            ("索提诺比率",             f"{m['sortino_ratio']:.3f}"),
            ("卡玛比率",               f"{m['calmar_ratio']:.3f}"),
            ("信息比率 IR",            f"{m['ir']:.3f}"),
            ("滚动6月IR (6M_IR)",      f"{m['rolling6m_ir']:.3f}"),
            ("Sterling",               f"{m['sterling_ratio']:.3f}"),
            ("Burke",                  f"{m['burke_ratio']:.3f}"),
            ("Martin",                 f"{m['martin_ratio']:.3f}"),
            ("Omega 比率",             f"{m['omega_ratio']:.3f}"),
            ("尾部比率",               f"{m['tail_ratio']:.3f}"),
            ("上行捕获",               f"{m['up_capture_ratio']:.3f}"),
            ("下行捕获",               f"{m['down_capture_ratio']:.3f}"),
            ("综合捕获",               f"{m['capture_ratio']:.3f}"),
            ("月度胜率",               f"{m['monthly_win_rate']:.2%}"),
            ("滚动6月超额胜率",        f"{m['rolling6m_win_rate']:.2%}"),
            ("VaR(95%)",               f"{m['var_95']:.2%}"),
            ("CVaR(95%)",              f"{m['cvar_95']:.2%}"),
            ("偏度",                   f"{m['skewness']:.3f}"),
            ("峰度",                   f"{m['kurtosis']:.3f}"),
            ("痛苦指数",               f"{m['pain_index']:.3f}"),
            ("溃疡指数",               f"{m['ulcer_index']:.3f}"),
            ("Jensen α (年化)",        f"{m['jensen_alpha']:.2%}"),
            ("Appraisal Ratio",        f"{m['appraisal_ratio']:.3f}"),
            ("β (CAPM)",               f"{m['beta']:.4f}"),
            ("SQN 系统质量指数",       f"{m['sqn']:.3f}"),
            ("月度换手成本",           f"{m['avg_monthly_turnover_cost']:.4%}"),
            ("年化换手成本",           f"{m['avg_annual_turnover_cost']:.2%}"),
            ("扣费后 CAGR",            f"{m['net_cagr_after_cost']:.2%}"),
            ("覆盖月份数",             str(m["n_months"])),
        ]

    def _kv_grid(self, rows, ncol: int = 4) -> str:
        out = ['<div class="kv-grid">']
        for label, val in rows:
            out.append(
                f'<div class="kv">'
                f'<div class="kv-label">{label}</div>'
                f'<div class="kv-value">{val}</div>'
                f'</div>')
        out.append('</div>')
        return "\n".join(out)

    def _month_options(self, months: list) -> str:
        opts = ['<option value="">选择月份…</option>']
        for m in months:
            opts.append(f'<option value="{m}">{m}</option>')
        return "\n".join(opts)

    # ── 服务端渲染 SVG 兜底 ──
    def _render_nav_svg_fallback(
        self, months: list, port_v: list, bm_v: list
    ) -> str:
        """
        ★ v5.4 新增: 服务端渲染纯 SVG 净值曲线 (Chart.js 加载失败时使用)
        输出一个 <svg> 字符串, 用户至少能看到数据
        """
        if not months or len(months) < 2:
            return ('<div class="kv-empty">'
                    '（无足够月度数据）</div>')
        # 归一化到起点 = 1
        try:
            port_base = float(port_v[0]) or 1.0
            bm_base = float(bm_v[0]) or 1.0
        except Exception:
            return ('<div class="kv-empty">'
                    '（净值数据异常）</div>')
        port_adj = [v / port_base for v in port_v]
        bm_adj = [v / bm_base for v in bm_v]
        n = len(months)
        W, H = 1200, 360
        PAD_L, PAD_R, PAD_T, PAD_B = 60, 30, 30, 50
        plot_w = W - PAD_L - PAD_R
        plot_h = H - PAD_T - PAD_B
        all_vals = port_adj + bm_adj
        vmin = min(all_vals)
        vmax = max(all_vals)
        if vmax - vmin < 1e-6:
            vmax, vmin = vmin + 0.01, vmin - 0.01
        # 留 5% padding
        vrange = vmax - vmin
        vmin -= vrange * 0.05
        vmax += vrange * 0.05
        vrange = vmax - vmin

        def x(i):
            return PAD_L + (i / (n - 1)) * plot_w

        def y(v):
            return PAD_T + (1 - (v - vmin) / vrange) * plot_h

        parts = [
            f'<svg viewBox="0 0 {W} {H}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'style="font-family:inherit;background:'
            f'rgba(255,255,255,0.4);border-radius:14px">'
        ]
        # Y 网格
        for k in range(5):
            yy = PAD_T + k * plot_h / 4
            vv = vmax - k * vrange / 4
            parts.append(
                f'<line x1="{PAD_L}" y1="{yy}" '
                f'x2="{PAD_L+plot_w}" y2="{yy}" '
                f'stroke="rgba(0,0,0,0.06)" stroke-width="1"/>'
                f'<text x="{PAD_L-6}" y="{yy+4}" '
                f'font-size="11" fill="#8E8E93" '
                f'text-anchor="end">{vv:.2f}x</text>'
            )
        # X 刻度 (首/1/4/2/4/3/4/末)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            i = min(n - 1, round(frac * (n - 1)))
            xx = x(i)
            parts.append(
                f'<text x="{xx}" y="{H-12}" '
                f'font-size="11" fill="#8E8E93" '
                f'text-anchor="middle">{months[i]}</text>'
            )

        def path_line(vals, color, width=2.5, dash=None):
            d = "M " + " L ".join(
                f"{x(i):.1f},{y(v):.1f}"
                for i, v in enumerate(vals)
            )
            # ★ v5.5 修复: 之前 f-string 拼接产生 `stroke-dasharray='"4,4"'`
            #   (单引号 + 双引号嵌套, 浏览器解析异常, 可能吞掉后续内容)
            #   改用变量构造, 确保输出标准的 ` stroke-dasharray="4,4"`
            dash_attr = ' stroke-dasharray="4,4"' if dash else ''
            return (f'<path d="{d}" stroke="{color}" '
                    f'stroke-width="{width}" fill="none" '
                    f'stroke-linejoin="round" '
                    f'stroke-linecap="round"'
                    f'{dash_attr}/>')

        # 面积 (策略) - 简单线性渐变
        area_d = (
            "M " + " L ".join(
                f"{x(i):.1f},{y(v):.1f}"
                for i, v in enumerate(port_adj)
            )
            + f" L {x(n-1):.1f},{PAD_T+plot_h} "
              f"L {x(0):.1f},{PAD_T+plot_h} Z"
        )
        parts.append(
            f'<defs><linearGradient id="gP" x1="0" x2="0" '
            f'y1="0" y2="1">'
            f'<stop offset="0%" stop-color="#0A84FF" '
            f'stop-opacity="0.25"/>'
            f'<stop offset="100%" stop-color="#0A84FF" '
            f'stop-opacity="0.02"/>'
            f'</linearGradient></defs>'
        )
        parts.append(f'<path d="{area_d}" fill="url(#gP)"/>')
        parts.append(path_line(port_adj, "#0A84FF", 2.5))
        parts.append(path_line(bm_adj, "#FF9F0A", 2.5, dash=True))
        # 末端标注
        last_port = port_adj[-1]
        last_bm = bm_adj[-1]
        parts.append(
            f'<text x="{x(n-1)-6}" y="{y(last_port):.1f}" '
            f'font-size="12" fill="#0A84FF" font-weight="600" '
            f'text-anchor="end">策略 {last_port:.3f}x</text>'
        )
        parts.append(
            f'<text x="{x(n-1)-6}" y="{y(last_bm)+16:.1f}" '
            f'font-size="12" fill="#FF9F0A" font-weight="600" '
            f'text-anchor="end">基准 {last_bm:.3f}x</text>'
        )
        # Legend
        parts.append(
            '<rect x="60" y="14" width="10" height="3" '
            'fill="#0A84FF"/>'
            '<text x="74" y="20" font-size="11" '
            'fill="#3A3A3C">策略</text>'
            '<rect x="120" y="14" width="10" height="3" '
            'fill="#FF9F0A"/>'
            '<text x="134" y="20" font-size="11" '
            'fill="#3A3A3C">基准</text>'
            '<text x="200" y="20" font-size="10" '
            'fill="#8E8E93">（Chart.js 不可用, 显示静态预览）'
            '</text>'
        )
        parts.append('</svg>')
        return "\n".join(parts)

    def _render_attrib_table_fallback(self, d: dict) -> str:
        """
        ★ v5.4 新增: Brinson 归因表格化 (Chart.js 不可用时)
        d = {'monthly': [...], 'by_industry': [...], 'summary': {...}}
        """
        if not d or not d.get("monthly"):
            return ('<div class="kv-empty">'
                    '（无 Brinson 归因数据）</div>')
        s = d.get("summary", {})
        m = d["monthly"]
        n = len(m)
        years = (n / 12) if n > 0 else 1

        def ann(v):
            return (v / years) if abs(v) >= 0.0001 else 0

        out = [
            '<div class="kv-grid">',
            f'<div class="kv"><div class="kv-label">'
            f'配置 α (累计)</div>'
            f'<div class="kv-value">{s.get("alloc", 0)*100:+.2f}%'
            f'</div>'
            f'<div class="kv-sub">年化 ≈ '
            f'{ann(s.get("alloc", 0))*100:+.2f}%</div></div>',
            f'<div class="kv"><div class="kv-label">'
            f'选择 α (累计)</div>'
            f'<div class="kv-value">{s.get("sel", 0)*100:+.2f}%'
            f'</div>'
            f'<div class="kv-sub">年化 ≈ '
            f'{ann(s.get("sel", 0))*100:+.2f}%</div></div>',
            f'<div class="kv"><div class="kv-label">'
            f'交互 (累计)</div>'
            f'<div class="kv-value">{s.get("inter", 0)*100:+.2f}%'
            f'</div>'
            f'<div class="kv-sub">年化 ≈ '
            f'{ann(s.get("inter", 0))*100:+.2f}%</div></div>',
            f'<div class="kv" style="background:'
            f'rgba(10,132,255,0.08);border:1px solid '
            f'rgba(10,132,255,0.2)">'
            f'<div class="kv-label">归因合计 ({n} 月)</div>'
            f'<div class="kv-value">'
            f'{s.get("total", 0)*100:+.2f}%</div>'
            f'<div class="kv-sub">年化 ≈ '
            f'{ann(s.get("total", 0))*100:+.2f}%</div></div>',
            '</div>',
            '<h3 style="font-size:14px;margin:18px 0 8px;'
            'color:var(--ink2)">行业贡献（Top 15）</h3>',
            '<div style="max-height:280px;overflow:auto">'
            '<table class="glass"><thead><tr>'
            '<th>行业</th><th class="num">配置 α</th>'
            '<th class="num">选择 α</th>'
            '<th class="num">交互</th>'
            '<th class="num">合计</th></tr></thead><tbody>',
        ]
        for r in d.get("by_industry", [])[:15]:
            out.append(
                f'<tr><td>{r["industry"]}</td>'
                f'<td class="num">{r["allocation"]*100:+.2f}%</td>'
                f'<td class="num">{r["selection"]*100:+.2f}%</td>'
                f'<td class="num">{r["interaction"]*100:+.2f}%</td>'
                f'<td class="num">{r["total"]*100:+.2f}%</td></tr>'
            )
        out.append('</tbody></table></div>')
        out.append(
            '<div style="color:#8E8E93;margin-top:8px;font-size:11px">'
            '（Chart.js 不可用, 显示静态表格）</div>'
        )
        return "\n".join(out)

    def _render_ff_table_fallback(self, d: dict) -> str:
        """★ v5.4 新增: 五因子归因表格化 (Chart.js 不可用时)"""
        if not d or not d.get("rows"):
            return ('<div class="kv-empty">'
                    '（无五因子归因数据）</div>')
        a = d.get("avg", {})
        out = ['<div class="kv-grid">']
        items = [
            ('α (年化)', a.get('alpha', 0), True),
            ('β·MKT', a.get('beta_mkt', 0), False),
            ('β·SMB', a.get('beta_smb', 0), False),
            ('β·HML', a.get('beta_hml', 0), False),
            ('β·RMW', a.get('beta_rmw', 0), False),
            ('β·CMA', a.get('beta_cma', 0), False),
            ('β·MOM', a.get('beta_mom', 0), False),
            ('R²', a.get('r2', 0), False),
        ]
        for k, v, is_alpha in items:
            if is_alpha:
                val_str = f"{v*100:+.2f}%"
            else:
                val_str = f"{v:.3f}"
            out.append(
                f'<div class="kv"><div class="kv-label">{k}</div>'
                f'<div class="kv-value">{val_str}</div></div>'
            )
        out.append('</div>')
        out.append(
            '<h3 style="font-size:14px;margin:18px 0 8px;'
            'color:var(--ink2)">月度 α (近 24 月)</h3>'
            '<div style="max-height:280px;overflow:auto">'
            '<table class="glass"><thead><tr>'
            '<th>月份</th><th class="num">α (年化)</th>'
            '<th class="num">β·MKT</th>'
            '<th class="num">β·SMB</th>'
            '<th class="num">β·HML</th>'
            '<th class="num">R²</th></tr></thead><tbody>'
        )
        for r in d["rows"][-24:]:
            out.append(
                f'<tr><td>{r["month"]}</td>'
                f'<td class="num">{r["alpha"]*100:+.2f}%</td>'
                f'<td class="num">{r["beta_mkt"]:.3f}</td>'
                f'<td class="num">{r["beta_smb"]:.3f}</td>'
                f'<td class="num">{r["beta_hml"]:.3f}</td>'
                f'<td class="num">{r["r2"]:.3f}</td></tr>'
            )
        out.append('</tbody></table></div>')
        out.append(
            '<div style="color:#8E8E93;margin-top:8px;font-size:11px">'
            '（Chart.js 不可用, 显示静态表格）</div>'
        )
        return "\n".join(out)

    def _render_barra_table_fallback(self, d: dict) -> str:
        """★ v5.4 新增: Barra 归因表格化"""
        if not d or not d.get("rows"):
            return ('<div class="kv-empty">'
                    '（无 Barra 归因数据）</div>')
        a = d.get("avg", {})
        factors = [k for k in a.keys() if k.startswith("ret_")]
        out = ['<div class="kv-grid">']
        for k in factors:
            name = k.replace("ret_", "").replace("barra_", "")
            out.append(
                f'<div class="kv"><div class="kv-label">{name}'
                f'</div>'
                f'<div class="kv-value">{a[k]*100:+.2f}%</div></div>'
            )
        out.append('</div>')
        out.append(
            '<h3 style="font-size:14px;margin:18px 0 8px;'
            'color:var(--ink2)">Barra 因子月收益 (近 24 月)</h3>'
            '<div style="max-height:280px;overflow:auto">'
            '<table class="glass"><thead><tr><th>月份</th>'
        )
        for k in factors[:8]:
            out.append(f'<th class="num">{k.replace("ret_", "").replace("barra_", "")}</th>')
        out.append('</tr></thead><tbody>')
        for r in d["rows"][-24:]:
            out.append(f'<tr><td>{r["month"]}</td>')
            for k in factors[:8]:
                out.append(f'<td class="num">{r.get(k, 0)*100:+.2f}%</td>')
            out.append('</tr>')
        out.append('</tbody></table></div>')
        out.append(
            '<div style="color:#8E8E93;margin-top:8px;font-size:11px">'
            '（Chart.js 不可用, 显示静态表格）</div>'
        )
        return "\n".join(out)

    # ── 序列化归因为 JSON ──────────────────────────────
    def _brinson_to_json(self, df: pd.DataFrame) -> str:
        df = _safe_df(df)
        if df.empty:
            return json.dumps({
                "monthly": [], "by_industry": [],
                "summary": {"alloc": 0, "sel": 0, "inter": 0,
                            "total": 0}}, ensure_ascii=False,
                allow_nan=False)
        # 月度合计
        m = (df.groupby("pred_month")[
            ["allocation", "selection", "interaction"]]
            .sum()
            .reset_index())
        monthly = [
            {"month": str(r["pred_month"]),
             "allocation": round(_safe_float(r["allocation"]), 6),
             "selection":  round(_safe_float(r["selection"]),  6),
             "interaction":round(_safe_float(r["interaction"]),6),
             "total":      round(_safe_float(
                 r["allocation"] + r["selection"] +
                 r["interaction"]), 6)}
            for _, r in m.iterrows()
        ]
        # 行业累计
        by_industry = (df.groupby("industry")[
            ["allocation", "selection", "interaction"]]
            .sum().reset_index())
        by_industry = by_industry.sort_values(
            "interaction", ascending=False)
        ind = [
            {"industry": str(r["industry"]),
             "allocation":  round(_safe_float(r["allocation"]),  6),
             "selection":   round(_safe_float(r["selection"]),   6),
             "interaction": round(_safe_float(r["interaction"]), 6),
             "total": round(_safe_float(
                 r["allocation"] + r["selection"] +
                 r["interaction"]), 6)}
            for _, r in by_industry.iterrows()
        ]
        s_alloc = _safe_float(m["allocation"].sum())
        s_sel   = _safe_float(m["selection"].sum())
        s_int   = _safe_float(m["interaction"].sum())
        summary = {
            "alloc": round(s_alloc, 6),
            "sel":   round(s_sel,   6),
            "inter": round(s_int,   6),
            "total": round(s_alloc + s_sel + s_int, 6),
        }
        return json.dumps({
            "monthly": monthly, "by_industry": ind,
            "summary": summary}, ensure_ascii=False,
            allow_nan=False)

    def _ff_to_json(self, df: pd.DataFrame) -> str:
        df = _safe_df(df)
        if df.empty:
            return json.dumps({"rows": [], "avg": {}},
                              ensure_ascii=False, allow_nan=False)
        rows = [
            {"month": str(r["pred_month"]),
             "alpha":   round(_safe_float(r.get("alpha", 0)), 6),
             "beta_mkt":round(_safe_float(r.get("beta_mkt", 0)), 6),
             "beta_smb":round(_safe_float(r.get("beta_smb", 0)), 6),
             "beta_hml":round(_safe_float(r.get("beta_hml", 0)), 6),
             "beta_rmw":round(_safe_float(r.get("beta_rmw", 0)), 6),
             "beta_cma":round(_safe_float(r.get("beta_cma", 0)), 6),
             "beta_mom":round(_safe_float(r.get("beta_mom", 0)), 6),
             "r2":      round(_safe_float(r.get("r_squared", 0)), 6)}
            for _, r in df.iterrows()
        ]
        if rows:
            avg = {k: round(_safe_float(
                float(np.mean([r[k] for r in rows]))), 6)
                for k in rows[0] if k != "month"}
        else:
            avg = {}
        return json.dumps({"rows": rows, "avg": avg},
                          ensure_ascii=False, allow_nan=False)

    def _barra_to_json(self, df: pd.DataFrame) -> str:
        df = _safe_df(df)
        if df.empty:
            return json.dumps({"rows": [], "avg": {}},
                              ensure_ascii=False, allow_nan=False)
        cols_risk = [c for c in df.columns if c.startswith("risk_")]
        cols_expo = [c for c in df.columns if c.startswith("expo_")]
        rows = [
            {"month": str(r["pred_month"]),
             **{c.replace("risk_", "ret_"): round(_safe_float(r[c]), 6)
                for c in cols_risk},
             **{c.replace("expo_", "expo_"): round(_safe_float(r[c]), 6)
                for c in cols_expo}}
            for _, r in df.iterrows()
        ]
        avg = {}
        for c in cols_risk:
            avg[c.replace("risk_", "ret_")] = round(
                _safe_float(df[c].mean()), 6)
        return json.dumps({"rows": rows, "avg": avg},
                          ensure_ascii=False, allow_nan=False)

    def _shap_to_json(
        self,
        shap_data: Dict,
        holdings: Optional[pd.DataFrame],
    ) -> str:
        """
        SHAP 深度分析（4 部分）：
          1. global_top5 : 全局 Top 5 核心因子（按所有月平均 |SHAP| 排序）
          2. correlation : Top 5 因子 SHAP 相关性矩阵（按月叠加所有股票）
          3. timeseries  : Top 5 因子影响力时间序列（每月 mean(|SHAP|)）
          4. monthly     : 每月 Top 10 因子（向后兼容旧版）
        shap_data: {pred_month: (np.ndarray[N×F], list[F]) 或 3-tuple}
        """
        empty = {
            "global_top5": {"factors": [], "mean_abs": [], "signed": []},
            "correlation": {"factors": [], "matrix": []},
            "timeseries":  {"months": [], "series": {}},
            "monthly":     {},
        }
        if not shap_data and (holdings is None or holdings.empty):
            return json.dumps(empty, ensure_ascii=False)

        # ① 解析 shap_data → {month: (sv[N×F], sf[list])}
        # ★ v5.3 修复: 不同时点的 FeatureStore 输出的 sf 长度可能不同
        #   (因 min_keep_factors 筛选不同). 旧逻辑要求"每个月 sf 都包含 top5"
        #   才能算相关性, 长尾下 164 窗口常因几个窗口 sf 缺 top5 导致空矩阵.
        #   改: 用 unified sf = 所有月 sf 的并集; 每个月的 sv 缺失列填 0
        parsed: Dict[str, tuple] = {}
        if shap_data:
            for m, item in shap_data.items():
                if isinstance(item, tuple) and len(item) == 3:
                    sv, sf, _src = item
                elif isinstance(item, tuple) and len(item) == 2:
                    sv, sf = item
                elif isinstance(item, np.ndarray) and item.ndim == 2:
                    sv = item
                    sf = [f"f_{i}" for i in range(item.shape[1])]
                else:
                    continue
                sv = np.asarray(sv)
                if sv.ndim != 2 or sv.size == 0:
                    continue
                parsed[str(m)] = (sv, list(sf))

        if not parsed:
            # 退化：仅用 shap_top1/2/3 列做粗略聚合
            if (holdings is not None
                    and "shap_top1_factor" in holdings.columns):
                fac_imp_abs: Dict[str, list] = {}
                fac_imp_sg:  Dict[str, list] = {}
                for _, r in holdings.iterrows():
                    for k in (1, 2, 3):
                        f = r.get(f"shap_top{k}_factor")
                        v = r.get(f"shap_top{k}_value")
                        if f is None or v is None:
                            continue
                        try:
                            f = str(f)
                            vf = float(v)
                        except (TypeError, ValueError):
                            continue
                        fac_imp_abs.setdefault(f, []).append(abs(vf))
                        fac_imp_sg.setdefault(f, []).append(vf)
                if fac_imp_abs:
                    ranked = sorted(
                        fac_imp_abs.items(),
                        key=lambda x: -float(np.mean(x[1]))
                    )[:5]
                    top5 = [f for f, _ in ranked]
                    empty["global_top5"] = {
                        "factors": top5,
                        "mean_abs": [round(float(np.mean(fac_imp_abs[f])), 6)
                                     for f in top5],
                        "signed":   [round(float(np.mean(fac_imp_sg[f])), 6)
                                     for f in top5],
                    }
                    empty["monthly"] = {
                        str(m): sorted(
                            [{"factor": f,
                              "mean_abs": round(float(np.mean(v)), 6),
                              "signed":   round(float(np.mean(
                                  fac_imp_sg.get(f, v))), 6)}
                             for f, v in fac_imp_abs.items()],
                            key=lambda x: -x["mean_abs"])[:10]
                        for m in holdings["pred_month"].astype(str).unique()
                    }
            return json.dumps(empty, ensure_ascii=False)

        # ② 全局平均 |SHAP| 聚合（用于排序）
        # ★ v5.3: 只统计"在 ≥30% 月份出现"的 factor, 避免偶发月份污染排序
        n_months_parsed = len(parsed)
        min_appear = max(3, int(n_months_parsed * 0.3))
        factor_abs_per_month:   Dict[str, list] = {}
        factor_signed_per_month: Dict[str, list] = {}
        factor_month_count:     Dict[str, int]   = {}
        for _m, (sv, sf) in parsed.items():
            mean_abs = np.abs(sv).mean(axis=0)
            signed   = sv.mean(axis=0)
            for i, f in enumerate(sf):
                f = str(f)
                factor_abs_per_month.setdefault(f, []).append(
                    float(mean_abs[i]))
                factor_signed_per_month.setdefault(f, []).append(
                    float(signed[i]))
                factor_month_count[f] = factor_month_count.get(f, 0) + 1
        # 只保留出现月份数 ≥ min_appear 的 factor
        qualified = {f for f, c in factor_month_count.items()
                     if c >= min_appear}
        factor_global_abs    = {f: float(np.mean(v))
                                for f, v in factor_abs_per_month.items()
                                if f in qualified}
        factor_global_signed = {f: float(np.mean(v))
                                for f, v in factor_signed_per_month.items()
                                if f in qualified}

        top5 = sorted(factor_global_abs.items(),
                      key=lambda x: -x[1])[:5]
        top5_factors = [f for f, _ in top5]

        empty["global_top5"] = {
            "factors":  top5_factors,
            "mean_abs": [round(factor_global_abs[f], 6)
                         for f in top5_factors],
            "signed":   [round(factor_global_signed[f], 6)
                         for f in top5_factors],
        }

        # ③ Top 5 因子 SHAP 相关性（拼接所有月所有股票）
        # ★ v5.3 改造: 用 unified sf = 所有月 sf 并集, 缺失列填 0,
        #   不再要求"每月 sf 都包含 top5"
        if top5_factors:
            all_shaps = []
            for _m, (sv, sf) in parsed.items():
                f2i_local = {f: i for i, f in enumerate(sf)}
                rows = sv  # (N, F_local)
                # 把每个 top5 factor 的列抽出来, 缺失列填 0
                cols = []
                for f in top5_factors:
                    if f in f2i_local:
                        cols.append(rows[:, f2i_local[f]])
                    else:
                        cols.append(np.zeros(rows.shape[0], dtype=rows.dtype))
                if rows.shape[0] >= 2:
                    all_shaps.append(np.column_stack(cols))
            if all_shaps:
                combined = np.vstack(all_shaps)  # [N_total × n_factors]
                n_fac = combined.shape[1]
                if (combined.shape[0] >= 2
                        and float(combined.std(axis=0).min()) > 1e-12):
                    corr = np.corrcoef(combined.T)
                    corr = np.nan_to_num(corr, nan=0.0)
                else:
                    corr = np.eye(n_fac)
                empty["correlation"] = {
                    "factors": top5_factors,
                    "matrix":  [[round(float(corr[i, j]), 4)
                                 for j in range(n_fac)]
                                for i in range(n_fac)],
                }

        # ④ 时间序列（每月 mean(|SHAP|)）
        months_sorted = sorted(parsed.keys())
        empty["timeseries"] = {
            "months": months_sorted,
            "series": {f: [] for f in top5_factors},
        }
        for m in months_sorted:
            sv, sf = parsed[m]
            mean_abs = np.abs(sv).mean(axis=0)
            f2i = {f: i for i, f in enumerate(sf)}
            for f in top5_factors:
                if f in f2i:
                    empty["timeseries"]["series"][f].append(
                        round(float(mean_abs[f2i[f]]), 6))
                else:
                    empty["timeseries"]["series"][f].append(0.0)

        # ⑤ 每月 Top 5 因子（向后兼容）
        for m, (sv, sf) in parsed.items():
            mean_abs = np.abs(sv).mean(axis=0)
            # ★ 改造: 只展示前 5 个因子
            order = np.argsort(-mean_abs)[:5]
            empty["monthly"][m] = [
                {"factor":  str(sf[i]),
                 "mean_abs": round(float(mean_abs[i]), 6),
                 "signed":   round(float(sv[:, i].mean()), 6)}
                for i in order
            ]

        return json.dumps(empty, ensure_ascii=False)

    def _holdings_to_json(
        self, holdings: pd.DataFrame, m3_data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        {pred_month: [{stock_code, stock_name, industry,
                       tier, weight, monthly_return, contribution}]}

        ★ Bug 修复: NaN 防御
          - 历史: float(NaN) 在 Python 里是真值,  `or 0` 不生效,
            json.dumps 会输出字面量 `NaN` (非标准 JSON),
            浏览器 JSON.parse 直接 SyntaxError, 整个 report 挂掉.
          - 现在: 用 pd.isna + math.isfinite 把 NaN/inf 都替成 0.
        """
        if holdings.empty:
            return json.dumps({}, ensure_ascii=False)
        m3_merge_cols = ["m3_action", "timing", "trend_score", "emotion_index", "anchored_trend"]
        m3_lookup = {}
        if m3_data is not None and not m3_data.empty:
            m3_cols_available = [c for c in m3_merge_cols if c in m3_data.columns]
            if m3_cols_available:
                m3_subset = m3_data[["pred_month", "stock_code"] + m3_cols_available].copy()
                for _, mr in m3_subset.iterrows():
                    key = (str(mr["pred_month"]), str(mr["stock_code"]))
                    m3_lookup[key] = {c: mr.get(c, None) for c in m3_cols_available}
        out = {}
        for m, g in holdings.groupby("pred_month"):
            rows = []
            for _, r in g.iterrows():
                w = _safe_float(r.get("weight", 0), 0.0)
                ret = _safe_float(r.get("Target_Return_1M", 0),
                                  0.0)
                c   = r.get("contribution", None)
                if c is None or pd.isna(c):
                    c = w * ret
                c = _safe_float(c, 0.0)
                row_dict = {
                    "stock_code": str(r.get("stock_code", "")),
                    "stock_name": str(r.get("stock_name", "")),
                    "industry":   str(r.get("industry", "")),
                    "tier":       str(r.get("tier", "")),
                    "weight":     round(w, 4),
                    "monthly_return": round(ret, 6),
                    "contribution":   round(c, 6),
                    "score":      round(_safe_float(
                        r.get("score", 0), 0.0), 4),
                    "is_holding": bool(r.get("is_holding", False)),
                }
                key = (str(m), str(r.get("stock_code", "")))
                m3_info = m3_lookup.get(key, {})
                is_sold_by_m3 = False
                m3_action_val = m3_info.get("m3_action", None)
                if m3_action_val is not None and not (isinstance(m3_action_val, float) and pd.isna(m3_action_val)):
                    if str(m3_action_val) == "SELL_TET":
                        is_sold_by_m3 = True
                row_dict["is_sold_by_m3"] = is_sold_by_m3
                sell_reason_row = dict(r)
                for mc in m3_merge_cols:
                    if mc in m3_info:
                        sell_reason_row[mc] = m3_info[mc]
                row_dict["m3_sell_reason"] = _format_m3_sell_reason(sell_reason_row)
                rows.append(row_dict)
            out[str(m)] = rows
        return json.dumps(out, ensure_ascii=False,
                          allow_nan=False)

    # ── HTML 渲染 ────────────────────────────────────────
    def _render_html(self, **kw) -> str:
        # 主色变量
        css = f"""
        :root {{
          --port: {PALETTE['port']};
          --bm:   {PALETTE['bm']};
          --alpha:{PALETTE['alpha']};
          --warn: {PALETTE['warn']};
          --tint: {PALETTE['tint']};
          --ink:  {PALETTE['ink']};
          --ink2: {PALETTE['ink2']};
          --ink3: {PALETTE['ink3']};
          --bg:   {PALETTE['bg']};
          --card: {PALETTE['card']};
          --card-stroke: {PALETTE['card_stroke']};
          --shadow: {PALETTE['shadow']};
        }}
        * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
        html, body {{
          margin: 0; padding: 0;
          font-family: -apple-system, BlinkMacSystemFont,
            "SF Pro Display", "SF Pro Text", "PingFang SC",
            "Microsoft YaHei", system-ui, sans-serif;
          background: var(--bg);
          color: var(--ink);
          letter-spacing: -0.01em;
        }}
        body {{
          background:
            radial-gradient(1200px 800px at 0% 0%,
              rgba(10,132,255,0.18), transparent 60%),
            radial-gradient(900px 700px at 100% 0%,
              rgba(191,90,242,0.16), transparent 60%),
            radial-gradient(1000px 800px at 50% 100%,
              rgba(48,209,88,0.14), transparent 60%),
            var(--bg);
          background-attachment: fixed;
          min-height: 100vh;
        }}
        .page {{
          max-width: 1480px;
          margin: 0 auto;
          padding: 28px 36px 60px;
        }}
        /* ── 顶部 Hero ── */
        .hero {{
          backdrop-filter: blur(28px) saturate(180%);
          -webkit-backdrop-filter: blur(28px) saturate(180%);
          background: var(--card);
          border: 1px solid var(--card-stroke);
          border-radius: 24px;
          padding: 28px 32px;
          box-shadow: 0 6px 30px var(--shadow);
          margin-bottom: 24px;
          display: flex; align-items: center; gap: 24px;
        }}
        .hero-icon {{
          width: 56px; height: 56px;
          background: linear-gradient(135deg, var(--port), var(--tint));
          border-radius: 16px;
          display: flex; align-items: center; justify-content: center;
          font-size: 28px; color: white;
          box-shadow: 0 4px 18px rgba(10,132,255,0.4);
        }}
        .hero-title {{ font-size: 28px; font-weight: 700; margin: 0; }}
        .hero-sub   {{ font-size: 14px; color: var(--ink3); margin-top: 4px; }}

        /* ── Glass Card ── */
        .card {{
          backdrop-filter: blur(20px) saturate(160%);
          -webkit-backdrop-filter: blur(20px) saturate(160%);
          background: var(--card);
          border: 1px solid var(--card-stroke);
          border-radius: 20px;
          padding: 22px 24px;
          box-shadow: 0 4px 24px var(--shadow);
          margin-bottom: 20px;
        }}
        .card h2 {{
          margin: 0 0 16px;
          font-size: 18px; font-weight: 600;
          display: flex; align-items: center; gap: 8px;
        }}
        .card h2 .badge {{
          font-size: 11px; padding: 2px 8px;
          border-radius: 10px;
          background: linear-gradient(135deg, var(--port), var(--tint));
          color: white; font-weight: 500;
        }}
        .card-sub {{
          font-size: 12px; color: var(--ink3);
          margin: -8px 0 14px;
        }}

        /* ── KV 网格 ── */
        .kv-grid {{
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 12px;
        }}
        .kv-grid:has(.kv:nth-child(4n+1):last-child) {{
          grid-template-columns: repeat(4, 1fr);
        }}
        .kv {{
          padding: 12px 14px;
          border-radius: 12px;
          background: rgba(255,255,255,0.5);
          border: 1px solid rgba(0,0,0,0.04);
        }}
        .kv-label {{
          font-size: 11px; color: var(--ink3);
          text-transform: uppercase; letter-spacing: 0.05em;
        }}
        .kv-value {{
          font-size: 18px; font-weight: 600;
          margin-top: 2px;
          font-variant-numeric: tabular-nums;
        }}
        /* v5.3: kv-sub 副标题 (年化估算等) */
        .kv-sub {{
          font-size: 11px; color: var(--ink3);
          margin-top: 4px;
          font-variant-numeric: tabular-nums;
        }}
        .kv-empty {{
          color: var(--ink3); font-size: 13px;
          padding: 12px 0;
        }}
        @media (max-width: 1100px) {{ .kv-grid {{ grid-template-columns: repeat(3,1fr); }} }}
        @media (max-width: 700px)  {{ .kv-grid {{ grid-template-columns: repeat(2,1fr); }} }}

        /* ── 净值曲线 ── */
        .chart-wrap {{
          position: relative; width: 100%;
          height: 420px;
        }}
        .chart-tooltip {{
          position: absolute;
          pointer-events: none;
          background: rgba(28,28,30,0.92);
          color: white; font-size: 12px;
          padding: 8px 12px; border-radius: 10px;
          backdrop-filter: blur(10px);
          transform: translate(-50%, -120%);
          opacity: 0; transition: opacity 0.12s;
          white-space: nowrap; z-index: 99;
        }}
        .chart-tooltip.show {{ opacity: 1; }}
        .chart-tooltip .tt-month {{ font-weight: 600; }}
        .chart-tooltip .tt-row {{
          display: flex; justify-content: space-between; gap: 14px;
          font-variant-numeric: tabular-nums;
          margin-top: 2px;
        }}
        .chart-tooltip .tt-pos {{ color: var(--alpha); }}
        .chart-tooltip .tt-neg {{ color: var(--warn); }}

        /* ── 滑块 ── */
        .range-row {{
          display: flex; align-items: center; gap: 12px;
          margin-top: 14px;
        }}
        .range-label {{
          font-size: 12px; color: var(--ink3);
          min-width: 64px;
        }}
        .range-value {{
          font-size: 12px; color: var(--ink);
          font-variant-numeric: tabular-nums;
          min-width: 110px; text-align: right;
        }}
        input[type="range"].ios-range {{
          -webkit-appearance: none; appearance: none;
          flex: 1; height: 4px; border-radius: 2px;
          background: linear-gradient(90deg,
            var(--port) 0%, var(--port) 50%,
            rgba(0,0,0,0.08) 50%, rgba(0,0,0,0.08) 100%);
          outline: none;
        }}
        input[type="range"].ios-range::-webkit-slider-thumb {{
          -webkit-appearance: none; appearance: none;
          width: 22px; height: 22px; border-radius: 50%;
          background: white;
          box-shadow: 0 2px 8px rgba(0,0,0,0.2),
            0 0 0 1px rgba(0,0,0,0.05);
          border: none; cursor: pointer;
        }}
        input[type="range"].ios-range::-moz-range-thumb {{
          width: 22px; height: 22px; border-radius: 50%;
          background: white;
          box-shadow: 0 2px 8px rgba(0,0,0,0.2);
          border: none; cursor: pointer;
        }}
        .range-pair {{ flex: 1; display: flex; gap: 10px; }}
        .range-pair input {{ flex: 1; }}

        /* ── 天天基金风格双滑块 ── */
        .ttjj-range {{
          margin-top: 18px;
          padding: 0 14px;
        }}
        .ttjj-track {{
          position: relative;
          height: 6px;
          background: rgba(0,0,0,0.08);
          border-radius: 3px;
          margin: 28px 10px 8px;
        }}
        .ttjj-fill {{
          position: absolute;
          top: 0; left: 0;
          height: 100%;
          background: linear-gradient(90deg, var(--port), var(--tint));
          border-radius: 3px;
          pointer-events: none;
        }}
        .ttjj-handle {{
          position: absolute;
          top: 50%;
          width: 22px; height: 22px;
          border-radius: 50%;
          background: white;
          box-shadow: 0 2px 8px rgba(0,0,0,0.18),
            0 0 0 1px rgba(0,0,0,0.05);
          transform: translate(-50%, -50%);
          cursor: grab;
          z-index: 2;
          transition: box-shadow 0.15s;
        }}
        .ttjj-handle:hover {{ box-shadow: 0 3px 12px rgba(10,132,255,0.35),
          0 0 0 1px rgba(10,132,255,0.3); }}
        .ttjj-handle:active {{ cursor: grabbing; }}
        .ttjj-ticks {{
          display: flex;
          justify-content: space-between;
          padding: 0 10px;
          margin-top: 4px;
          font-size: 10px;
          color: var(--ink3);
          font-variant-numeric: tabular-nums;
        }}
        .ttjj-ticks span {{ flex: 1; text-align: center; }}
        .ttjj-ticks span:first-child {{ text-align: left; }}
        .ttjj-ticks span:last-child  {{ text-align: right; }}
        .ttjj-row {{
          display: flex; justify-content: space-between;
          margin-top: 6px;
        }}
        .ttjj-label {{ display: flex; gap: 8px; align-items: baseline; }}
        .ttjj-cap   {{ font-size: 11px; color: var(--ink3); }}
        .ttjj-val   {{
          font-size: 13px; font-weight: 600;
          color: var(--ink);
          font-variant-numeric: tabular-nums;
        }}
        .ttjj-presets {{
          display: flex; gap: 6px; margin-top: 12px;
          flex-wrap: wrap;
        }}
        .ttjj-preset {{
          padding: 4px 12px;
          font-size: 12px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.08);
          background: rgba(255,255,255,0.6);
          color: var(--ink2);
          cursor: pointer;
          transition: all 0.15s;
        }}
        .ttjj-preset:hover {{
          background: rgba(10,132,255,0.06);
          border-color: rgba(10,132,255,0.3);
        }}
        .ttjj-preset.active {{
          background: var(--port);
          color: white;
          border-color: var(--port);
        }}

        /* ── 标签 / Pill ── */
        .pill {{
          display: inline-block;
          padding: 3px 10px; border-radius: 999px;
          font-size: 11px; font-weight: 500;
          background: rgba(10,132,255,0.12);
          color: var(--port);
        }}
        .pill.green {{ background: rgba(48,209,88,0.14); color: var(--alpha); }}
        .pill.red   {{ background: rgba(255,69,58,0.14);  color: var(--warn); }}
        .pill.amber {{ background: rgba(255,159,10,0.16); color: var(--bm); }}
        .pill.gray  {{ background: rgba(142,142,147,0.18); color: var(--ink2); }}

        /* ── 表格 ── */
        table.glass {{
          width: 100%; border-collapse: separate;
          border-spacing: 0;
          font-size: 13px;
          font-variant-numeric: tabular-nums;
        }}
        table.glass th {{
          text-align: left; font-weight: 600;
          color: var(--ink3); font-size: 11px;
          padding: 10px 12px;
          text-transform: uppercase; letter-spacing: 0.05em;
          border-bottom: 1px solid rgba(0,0,0,0.06);
        }}
        table.glass td {{
          padding: 10px 12px;
          border-bottom: 1px solid rgba(0,0,0,0.04);
        }}
        table.glass tr:last-child td {{ border-bottom: 0; }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .pos {{ color: var(--alpha); font-weight: 500; }}
        .neg {{ color: var(--warn);  font-weight: 500; }}

        /* ── Tab 切换 ── */
        .tabs {{
          display: flex; gap: 4px;
          background: rgba(0,0,0,0.04);
          padding: 4px;
          border-radius: 12px;
          margin-bottom: 16px;
          width: max-content;
        }}
        .tab-btn {{
          padding: 6px 14px; border-radius: 8px;
          font-size: 13px; font-weight: 500;
          border: none; background: transparent;
          color: var(--ink2); cursor: pointer;
          transition: all 0.18s;
        }}
        .tab-btn.active {{
          background: white;
          color: var(--ink);
          box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}
        .track-switcher {{
          display: flex; gap: 4px;
          background: rgba(0,0,0,0.04);
          padding: 4px;
          border-radius: 12px;
          margin-bottom: 12px;
          width: max-content;
        }}
        .track-btn {{
          padding: 6px 14px; border-radius: 8px;
          font-size: 13px; font-weight: 500;
          border: none; background: transparent;
          color: var(--ink2); cursor: pointer;
          transition: all 0.18s;
        }}
        .track-btn.active {{
          background: white;
          color: var(--ink);
          box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}
        .alignment-banner {{
          padding: 10px 14px;
          border-radius: 10px;
          background: rgba(255,159,10,0.10);
          border: 1px solid rgba(255,159,10,0.3);
          color: #FF9F0A;
          font-size: 12px;
          margin-bottom: 12px;
        }}
        /* v5.5: 修复 - tab-pane 默认全部可见, JS 未运行时也能看到数据 */
        .tab-pane {{ display: block; }}
        .tab-pane:not(.active) {{ display: none; }}
        .tab-pane.active {{ display: block; }}

        /* ── 月度持仓选择器 ── */
        .picker-row {{
          display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
        }}
        .picker-cap {{
          font-size: 12px; color: var(--ink3);
          font-weight: 500;
        }}
        .export-btn {{
          display: inline-flex; align-items: center; gap: 4px;
          padding: 7px 16px;
          font-size: 13px; font-weight: 500;
          border-radius: 12px;
          border: 1px solid rgba(48,209,88,0.4);
          background: linear-gradient(135deg,
            rgba(48,209,88,0.12), rgba(48,209,88,0.06));
          color: var(--alpha);
          cursor: pointer;
          transition: all 0.18s;
          margin-left: auto;
        }}
        .export-btn:hover {{
          background: linear-gradient(135deg,
            rgba(48,209,88,0.22), rgba(48,209,88,0.10));
          transform: translateY(-1px);
          box-shadow: 0 3px 10px rgba(48,209,88,0.2);
        }}
        .export-btn:active {{ transform: translateY(0); }}
        .export-icon {{
          font-size: 14px;
          display: inline-block;
          animation: bounce 1.5s ease-in-out infinite;
        }}
        @keyframes bounce {{
          0%, 100% {{ transform: translateY(0); }}
          50% {{ transform: translateY(2px); }}
        }}
        .ios-select {{
          appearance: none; -webkit-appearance: none;
          padding: 8px 36px 8px 14px;
          font-size: 14px;
          border-radius: 12px;
          border: 1px solid rgba(0,0,0,0.08);
          background: rgba(255,255,255,0.7)
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1 1l5 5 5-5' stroke='%238E8E93' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>")
            no-repeat right 12px center;
          color: var(--ink);
          outline: none; cursor: pointer;
        }}
        .ios-select:focus {{
          box-shadow: 0 0 0 3px rgba(10,132,255,0.25);
        }}

        /* ── 持仓卡片 ── */
        .hold-grid {{
          display: grid; gap: 10px;
          grid-template-columns: repeat(2, 1fr);
        }}
        @media (max-width: 900px) {{
          .hold-grid {{ grid-template-columns: 1fr; }}
        }}
        .hold-card {{
          padding: 12px 14px;
          background: rgba(255,255,255,0.6);
          border: 1px solid rgba(0,0,0,0.04);
          border-radius: 12px;
          display: flex; align-items: center; gap: 10px;
        }}
        .hold-rank {{
          width: 26px; height: 26px;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--port), var(--tint));
          color: white; font-weight: 600;
          display: flex; align-items: center; justify-content: center;
          font-size: 12px;
        }}
        .hold-rank.Low {{
          background: linear-gradient(135deg, var(--alpha), #5AC8FA);
        }}
        .hold-info {{ flex: 1; min-width: 0; }}
        .hold-name {{
          font-size: 13px; font-weight: 600;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}
        .hold-meta {{
          font-size: 11px; color: var(--ink3);
          margin-top: 2px;
        }}
        .hold-w {{
          font-size: 14px; font-weight: 600;
          font-variant-numeric: tabular-nums;
        }}
        .hold-ret {{ font-size: 12px; margin-top: 2px;
          font-variant-numeric: tabular-nums; }}

        /* ── SHAP 时间序列子图网格 ── */
        .ts-grid {{
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
        }}
        @media (max-width: 1100px) {{
          .ts-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        @media (max-width: 700px) {{
          .ts-grid {{ grid-template-columns: 1fr; }}
        }}
        .ts-card {{
          background: rgba(255,255,255,0.55);
          border: 1px solid rgba(0,0,0,0.05);
          border-radius: 14px;
          padding: 12px 14px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}
        .ts-card-title {{
          font-size: 13px; font-weight: 600;
          color: var(--ink);
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          margin-bottom: 4px;
          display: flex; align-items: center; gap: 6px;
        }}
        .ts-card-rank {{
          display: inline-flex; align-items: center; justify-content: center;
          width: 20px; height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--port), var(--tint));
          color: white; font-size: 10px; font-weight: 700;
          flex-shrink: 0;
        }}
        .ts-card-sub {{
          font-size: 10px; color: var(--ink3);
          margin-bottom: 6px;
        }}
        .ts-card-wrap {{ position: relative; height: 170px; }}

        /* ── 行内图表块 ── */
        .row {{ display: flex; gap: 20px; }}
        .col-2 {{ flex: 2; }}
        .col-1 {{ flex: 1; }}
        @media (max-width: 1100px) {{
          .row {{ flex-direction: column; }}
        }}

        .footer {{
          text-align: center; color: var(--ink3);
          font-size: 12px; padding: 30px 0 10px;
        }}
        """

        # ── JS 数据 ──
        # 注意：months 长度 0 / 1 时，slider 行为兜底
        js_data = f"""
        const DATA_M2 = {{
          months: {json.dumps(kw['months'])},
          port:   {json.dumps(kw['port_v'])},
          bm:     {json.dumps(kw['bm_v'])},
          ex:     {json.dumps(kw['ex_v'])},
          high:   {json.dumps(kw['high_v'])},
          low:    {json.dumps(kw['low_v'])},
          dd_port: {json.dumps(kw['dd_port_v'])},
          dd_bm:   {json.dumps(kw['dd_bm_v'])},
          dd_high: {json.dumps(kw['dd_high_v'])},
          dd_low:  {json.dumps(kw['dd_low_v'])},
          monthly_ret: {json.dumps(kw['monthly_ret'])},
          monthly_ex:  {json.dumps(kw['monthly_ex'])},
          monthly_high_ret: {json.dumps(kw['monthly_high_ret'])},
          monthly_low_ret:  {json.dumps(kw['monthly_low_ret'])},
          brinson: {kw['brinson_json']},
          ff:      {kw['ff_json']},
          barra:   {kw['barra_json']},
          shap:    {kw['shap_data_json']},
          holdings:{kw['holdings_json']},
          shap_degraded: false,
          perf_rows: {json.dumps(kw['perf_data'])},
        }};
        const DATA_M3 = {kw['m3_data_json']};
        const has_m3_data = {'true' if kw['has_m3_data'] else 'false'};
        const aligned_months = {kw['aligned_months_json']};
        const alignment_banner = {json.dumps(kw.get('alignment_banner', ''), ensure_ascii=False)};
        let active_track = has_m3_data ? "M2+M3" : "M2";
        let DATA = has_m3_data ? DATA_M3 : DATA_M2;
        // ★ v5.4: Chart.js 加载状态检测
        //   - 如果内嵌, window.Chart 一定存在
        //   - 如果是 CDN 模式, 加载失败后 __chartJsFailed = true
        //   - 任意情况: 立即判定, 不要等到 init 失败才补救
        window.__chartJsOk = !!window.Chart;
        window.__chartJsChecked = false;
        function checkChartJs() {{
          if (window.__chartJsChecked) return window.__chartJsOk;
          window.__chartJsChecked = true;
          if (window.__chartJsOk) return true;
          // 等待 1.5s 给 Chart.js 加载机会
          return new Promise((resolve) => {{
            let waited = 0;
            const t = setInterval(() => {{
              waited += 100;
              if (window.Chart || window.__chartJsFailed) {{
                clearInterval(t);
                window.__chartJsOk = !!window.Chart;
                resolve(window.__chartJsOk);
              }} else if (waited >= 1500) {{
                clearInterval(t);
                resolve(false);
              }}
            }}, 100);
          }});
        }}
        // ★ v5.4: 切换到静态兜底 (Chart.js 不可用时)
        function useFallback() {{
          // 净值曲线: 隐藏 canvas, 显示 SVG
          const cv = document.getElementById('navCurve');
          if (cv) cv.style.display = 'none';
          const tip = document.getElementById('chartTip');
          if (tip) tip.style.display = 'none';
          const svg = document.getElementById('navSvgFallback');
          if (svg) svg.style.display = 'block';
          // 归因 Tab: 隐藏 chart wrap, 显示 fallback 表格
          for (const [wrapId, fbId] of [
            ['brinsonChartWrap', 'brinsonFallback'],
            ['ffChartWrap',      'ffFallback'],
            ['barraChartWrap',   'barraFallback'],
          ]) {{
            const w = document.getElementById(wrapId);
            const f = document.getElementById(fbId);
            if (w) w.style.display = 'none';
            if (f) f.style.display = 'block';
          }}
          // SHAP Top5 / 时序 / 月度柱 都会失败, 提示用户
          const shapTop5 = document.getElementById('tab-shap-top5');
          if (shapTop5) {{
            shapTop5.innerHTML =
              '<div class="kv-empty">Chart.js 不可用, 请检查网络' +
              '或刷新页面 (SHAP Top 5/时序/月度详情需要 Chart.js)</div>';
          }}
          const holdChart = document.getElementById('holdContribChart');
          if (holdChart) {{
            // 隐藏 canvas, 显示文本提示
            const wrap = holdChart.closest('.chart-wrap');
            if (wrap) {{
              wrap.innerHTML =
                '<div class="kv-empty" style="padding:40px">Chart.js 不可用, 持仓贡献图无法显示' +
                '<br>但下方表格仍可正常查看</div>';
            }}
          }}
          // SHAP 时序 + 月度详情
          const tsGrid = document.getElementById('shapTsGrid');
          if (tsGrid) {{
            tsGrid.innerHTML =
              '<div class="kv-empty" style="grid-column:1/-1">' +
              'Chart.js 不可用, 时序图无法显示</div>';
          }}
          const shapMonthTabs = document.getElementById('shapMonthTabs');
          if (shapMonthTabs) {{
            shapMonthTabs.parentElement.innerHTML =
              '<div class="kv-empty">Chart.js 不可用, 月度详情图无法显示</div>';
          }}
        }}
        // ★ v5.4: 每个 init 单独 try/catch, 任何一个失败不阻断其他模块
        const _safe = (name, fn) => {{
          try {{ fn(); }} catch (e) {{
            console.error(`[M4 报告] ${{name}} 初始化失败:`, e);
            const main = document.querySelector('.page');
            if (main) {{
              const err = document.createElement('div');
              err.style.cssText = 'margin:12px;padding:12px;background:rgba(255,69,58,0.08);border:1px solid rgba(255,69,58,0.3);border-radius:10px;color:#FF453A;font-size:13px;';
              err.textContent = `[M4] ${{name}} 加载失败: ${{e.message}}`;
              main.appendChild(err);
            }}
          }}
        }};
        window.addEventListener('DOMContentLoaded', async () => {{
          // ★ v5.4: 等 Chart.js 加载完成 (内嵌模式立即通过)
          const ok = await checkChartJs();
          if (!ok) {{
            console.warn('[M4] Chart.js 不可用, 切换到静态兜底');
            useFallback();
            // SHAP Tab 切换 (只切到 SVG 热力图 + 表格可用的)
            _safe('月度持仓选择器',  initHoldingsPicker);
            _safe('SHAP 归因',      initShapTabsFallback);
            return;
          }}
          _safe('净值曲线 + 双滑块',  initNavCurve);
          _safe('回撤曲线 + 双滑块',  initDrawdownCurve);
          _safe('收益归因 (Brinson/FF/Barra)', initAttribution);
          _safe('月度持仓选择器',     initHoldingsPicker);
          _safe('SHAP 归因',         initShapTabs);
          _safe('月度持仓 Tab',      initHoldingsTab);
          _safe('双轨切换',          initTrackSwitcher);
        }});
        """

        # ── 完整 HTML 输出 ──
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{kw['title']} · {datetime.now().strftime('%Y-%m-%d')}</title>
<style>{css}</style>
<!-- ★ v5.4 全面修缮: 内嵌本地 Chart.js, 永不依赖网络 -->
{'<script>' + (kw['chartjs_inline'] or '') + '</script>' if kw.get('chartjs_inline') else ''}
</head>
<body>
<div class="page">

  <div class="hero">
    <div class="hero-icon">📈</div>
    <div>
      <div class="hero-title">{kw['title']}</div>
      <div class="hero-sub">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · M4 回测报告 · iOS 26 风格</div>
    </div>
  </div>

  <!-- 摘要 -->
  <div class="card">
    <h2>运行摘要 <span class="badge">M2 stats</span></h2>
    <div class="track-switcher" id="trackSwitcher" style="display:none">
      <button class="track-btn" data-track="M2" onclick="switchTrack('M2')">纯M2</button>
      <button class="track-btn active" data-track="M2+M3" onclick="switchTrack('M2+M3')">M2+M3</button>
    </div>
    <div class="alignment-banner" id="alignmentBanner" style="display:none"></div>
    {kw['summary_table']}
  </div>

  <!-- 净值曲线 + 天天基金风格双滑块 + hover -->
  <div class="card">
    <h2>净值曲线 <span class="badge">策略 vs 基准</span></h2>
    <div class="card-sub">鼠标 hover 看月度收益；下方双滑块拖动可调时间区间（起始/截止）</div>
    <div class="chart-wrap" id="chartWrap">
      <!-- ★ v5.5: 修复 - canvas 默认隐藏, SVG 兜底默认显示; Chart.js 启动后再切换 -->
      <canvas id="navCurve" style="display:none"></canvas>
      <div class="chart-tooltip" id="chartTip"></div>
      <div id="navSvgFallback" style="display:block; width:100%">{kw['nav_svg_fallback']}</div>
    </div>
    <!-- ★ 天天基金风格：左右两个滑块，联动区间选择 -->
    <div class="ttjj-range" id="ttjjRange">
      <div class="ttjj-track">
        <div class="ttjj-fill" id="ttjjFill"></div>
        <div class="ttjj-handle" id="ttjjHandleL"></div>
        <div class="ttjj-handle" id="ttjjHandleR"></div>
      </div>
      <div class="ttjj-ticks" id="ttjjTicks"></div>
      <div class="ttjj-row">
        <div class="ttjj-label">
          <span class="ttjj-cap">起始</span>
          <span class="ttjj-val" id="ttjjStartVal">—</span>
        </div>
        <div class="ttjj-label" style="text-align:right">
          <span class="ttjj-cap">截止</span>
          <span class="ttjj-val" id="ttjjEndVal">—</span>
        </div>
      </div>
    </div>
    <!-- v5.3: 移除"全部/近1年/近3年/近5年/今年"方块按钮, 区间完全由双滑块控制 -->
  </div>

  <!-- 回撤曲线 + 天天基金风格双滑块 + hover -->
  <div class="card">
    <h2>回撤曲线 <span class="badge">策略 vs 基准</span></h2>
    <div class="card-sub">策略 / 基准 / 13%×5 / 7%×5 的历史回撤；下方双滑块拖动可调时间区间（起始/截止）</div>
    <div class="chart-wrap" id="ddChartWrap">
      <canvas id="ddCurve"></canvas>
      <div class="chart-tooltip" id="ddTip"></div>
    </div>
    <div class="ttjj-range" id="ddTtjjRange">
      <div class="ttjj-track">
        <div class="ttjj-fill" id="ddTtjjFill"></div>
        <div class="ttjj-handle" id="ddTtjjHandleL"></div>
        <div class="ttjj-handle" id="ddTtjjHandleR"></div>
      </div>
      <div class="ttjj-ticks" id="ddTtjjTicks"></div>
      <div class="ttjj-row">
        <div class="ttjj-label">
          <span class="ttjj-cap">起始</span>
          <span class="ttjj-val" id="ddTtjjStartVal">—</span>
        </div>
        <div class="ttjj-label" style="text-align:right">
          <span class="ttjj-cap">截止</span>
          <span class="ttjj-val" id="ddTtjjEndVal">—</span>
        </div>
      </div>
    </div>
  </div>

  <!-- 28+ 绩效指标 -->
  <div class="card">
    <h2>核心绩效指标 <span class="badge">28+</span></h2>
    {kw['perf_rows']}
  </div>

  <!-- 归因模块（Tab 切换） -->
  <div class="card">
    <h2>收益归因 <span class="badge">3 维度</span></h2>
    <div class="card-sub">Brinson（五维度）/ 五因子回归 / Barra 风险</div>
    <div class="tabs" id="attrTabs">
      <button class="tab-btn active" data-tab="brinson">Brinson 归因</button>
      <button class="tab-btn" data-tab="ff">五因子回归</button>
      <button class="tab-btn" data-tab="barra">Barra 风险</button>
    </div>

    <!-- Brinson -->
    <div class="tab-pane active" id="tab-brinson">
      <!-- ★ v5.4: Chart.js 失败时, 显示静态表格 -->
      <div id="brinsonFallback">{kw['brinson_fallback']}</div>
      <div id="brinsonChartWrap">
      <div class="row">
        <div class="col-1">
          <h3 style="font-size:14px;margin:0 0 10px;color:var(--ink2)">累计归因</h3>
          <div id="brinsonSummary"></div>
        </div>
        <div class="col-2">
          <h3 style="font-size:14px;margin:0 0 10px;color:var(--ink2)">月度归因（堆叠柱）</h3>
          <div class="chart-wrap" style="height:280px">
            <canvas id="brinsonChart"></canvas>
          </div>
        </div>
      </div>
      <h3 style="font-size:14px;margin:18px 0 8px;color:var(--ink2)">行业贡献（累计）</h3>
      <div style="max-height:280px;overflow:auto">
        <table class="glass" id="brinsonIndTable">
          <thead><tr>
            <th>行业</th>
            <th class="num">配置 α</th>
            <th class="num">选择 α</th>
            <th class="num">交互</th>
            <th class="num">合计</th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
      </div>
    </div>

    <!-- 五因子 -->
    <div class="tab-pane" id="tab-ff">
      <div id="ffFallback">{kw['ff_fallback']}</div>
      <div id="ffChartWrap">
      <div class="kv-grid" id="ffAvg"></div>
      <h3 style="font-size:14px;margin:18px 0 8px;color:var(--ink2)">月度 α 与 β</h3>
      <div class="chart-wrap" style="height:320px">
        <canvas id="ffChart"></canvas>
      </div>
      </div>
    </div>

    <!-- Barra -->
    <div class="tab-pane" id="tab-barra">
      <div id="barraFallback">{kw['barra_fallback']}</div>
      <div id="barraChartWrap">
      <div class="kv-grid" id="barraAvg"></div>
      <h3 style="font-size:14px;margin:18px 0 8px;color:var(--ink2)">风险贡献（10 因子）</h3>
      <div class="chart-wrap" style="height:340px">
        <canvas id="barraChart"></canvas>
      </div>
      </div>
    </div>
  </div>

  <!-- SHAP 因子深度分析 -->
  <div class="card">
    <h2>SHAP 因子深度分析 <span class="badge">Top 5 + 相关 + 时序</span></h2>
    <div class="card-sub">全局 Top 5 核心因子 · SHAP 相关性热力图 · 影响力时间序列变化 · 月度详情；所有图表支持鼠标 hover 查看数据</div>
    <div class="tabs" id="shapMainTabs">
      <button class="tab-btn active" data-tab="top5">① Top 5 核心因子</button>
      <button class="tab-btn" data-tab="corr">② 因子相关性</button>
      <button class="tab-btn" data-tab="ts">③ 时间序列</button>
      <button class="tab-btn" data-tab="monthly">④ 月度详情</button>
    </div>

    <!-- Top 5 核心因子 -->
    <div class="tab-pane active" id="tab-shap-top5">
      <div class="chart-wrap" style="height:380px; position:relative">
        <canvas id="shapTop5Chart"></canvas>
        <div class="chart-tooltip" id="shapTop5Tip"></div>
      </div>
      <h3 style="font-size:14px;margin:14px 0 8px;color:var(--ink2)">因子中文说明</h3>
      <div class="kv-grid" id="shapTop5Desc"></div>
    </div>

    <!-- 因子相关性热力图 (v5.3: 改用纯 SVG, 移除 chartjs-chart-matrix 依赖) -->
    <div class="tab-pane" id="tab-shap-corr">
      <div class="card-sub" style="margin-top:0">基于所有月所有股票拼接后的 Top 5 因子 SHAP 相关性 · 鼠标 hover 查看数值</div>
      <div id="shapCorrSvgWrap" style="position:relative;padding:20px 12px 12px"></div>
    </div>

    <!-- 时间序列（5 个子图） -->
    <div class="tab-pane" id="tab-shap-ts">
      <div id="shapTsGrid" class="ts-grid"></div>
    </div>

    <!-- 月度详情（保留旧版） -->
    <div class="tab-pane" id="tab-shap-monthly">
      <div class="tabs" id="shapMonthTabs" style="margin-bottom:12px"></div>
      <div class="chart-wrap" style="height:380px; position:relative">
        <canvas id="shapChart"></canvas>
        <div class="chart-tooltip" id="shapMonthTip"></div>
      </div>
    </div>
  </div>

  <!-- 月度持仓选择器 -->
  <div class="card">
    <h2>月度持仓详情 <span class="badge">选月查看</span></h2>
    <div class="card-sub">选年份 + 月份 → 看 10 只持仓 (13% × 5 + 7% × 5) + 月度个股收益 + 简单归因；可一键导出 CSV</div>
    <div class="picker-row">
      <span class="picker-cap">年份</span>
      <select class="ios-select" id="holdYear"></select>
      <span class="picker-cap">月份</span>
      <select class="ios-select" id="holdMonth"></select>
      <span id="holdMeta" class="pill gray">—</span>
      <button class="export-btn" id="exportHoldingsCsv">
        <span class="export-icon">⬇</span>导出 CSV
      </button>
    </div>
    <div style="margin-top:14px">
      <h3 style="font-size:14px;margin:0 0 10px;color:var(--ink2)">前 10 大持仓（13% × 5 + 7% × 5）</h3>
      <div class="hold-grid" id="holdGrid"></div>
      <h3 style="font-size:14px;margin:18px 0 8px;color:var(--ink2)">本月收益归因（个股贡献）</h3>
      <div class="chart-wrap" style="height:260px">
        <canvas id="holdContribChart"></canvas>
      </div>
      <h3 style="font-size:14px;margin:18px 0 8px;color:var(--ink2)">月度持仓明细表</h3>
      <div style="max-height:340px;overflow:auto">
        <table class="glass" id="holdTable">
          <thead><tr>
            <th>#</th>
            <th>股票代码</th>
            <th>股票名称</th>
            <th>行业</th>
            <th>组别</th>
            <th class="num">权重</th>
            <th class="num">月收益率</th>
            <th class="num">贡献</th>
            <th class="m3-col" style="display:none">是否被M3卖出</th>
            <th class="m3-col" style="display:none">M3卖出原因</th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="footer">
    TTHH Quant System · M4 Report · iOS 26 风格 · 数据基于完整 M2 全量回测
  </div>
</div>

<script>
/* ──────────── 净值曲线 + 天天基金风格双滑块 + hover ──────────── */
let navChart = null;
let NAV_STATE = {{
  startIdx: 0, endIdx: 0, n: 0,
}};
function initNavCurve() {{
  // ★ v5.5: 修复 - Chart.js 已就绪, 显示 canvas, 隐藏 SVG 兜底
  const _cv = document.getElementById('navCurve');
  if (_cv) _cv.style.display = 'block';
  const _svg = document.getElementById('navSvgFallback');
  if (_svg) _svg.style.display = 'none';
  const ctx = document.getElementById('navCurve').getContext('2d');
  const n = DATA.months.length;
  if (n === 0) return;
  NAV_STATE.n = n;
  NAV_STATE.startIdx = 0;
  NAV_STATE.endIdx   = n - 1;

  // 渲染滑块刻度（取首/1/4/2/3/4/末 共 6 个刻度）
  renderTtjjTicks();

  const grad = ctx.createLinearGradient(0, 0, 0, 420);
  grad.addColorStop(0, 'rgba(10,132,255,0.28)');
  grad.addColorStop(1, 'rgba(10,132,255,0.02)');
  const grad2 = ctx.createLinearGradient(0, 0, 0, 420);
  grad2.addColorStop(0, 'rgba(255,159,10,0.18)');
  grad2.addColorStop(1, 'rgba(255,159,10,0.02)');

  navChart = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: DATA.months,
      datasets: [
        {{
          label: '策略净值',
          data: DATA.port,
          borderColor: '#0A84FF',
          backgroundColor: grad,
          fill: true, tension: 0.3, borderWidth: 2.5,
          pointRadius: 0, pointHoverRadius: 7,
          pointHoverBackgroundColor: '#0A84FF',
          pointHoverBorderColor: '#fff', pointHoverBorderWidth: 2,
        }},
        {{
          label: '基准净值',
          data: DATA.bm,
          borderColor: '#FF9F0A',
          backgroundColor: grad2,
          fill: true, tension: 0.3, borderWidth: 2.5,
          borderDash: [4, 4],
          pointRadius: 0, pointHoverRadius: 7,
          pointHoverBackgroundColor: '#FF9F0A',
          pointHoverBorderColor: '#fff', pointHoverBorderWidth: 2,
        }},
        {{
          label: '13%×5 组合',
          data: DATA.high,
          borderColor: '#30D158',
          backgroundColor: 'rgba(48,209,88,0.08)',
          fill: false, tension: 0.3, borderWidth: 2,
          borderDash: [6, 3],
          pointRadius: 0, pointHoverRadius: 6,
          pointHoverBackgroundColor: '#30D158',
          pointHoverBorderColor: '#fff', pointHoverBorderWidth: 2,
        }},
        {{
          label: '7%×5 组合',
          data: DATA.low,
          borderColor: '#BF5AF2',
          backgroundColor: 'rgba(191,90,242,0.08)',
          fill: false, tension: 0.3, borderWidth: 2,
          borderDash: [3, 3],
          pointRadius: 0, pointHoverRadius: 6,
          pointHoverBackgroundColor: '#BF5AF2',
          pointHoverBorderColor: '#fff', pointHoverBorderWidth: 2,
        }},
        {{
          label: '超额',
          data: DATA.ex,
          borderColor: 'rgba(48,209,88,0.7)',
          borderWidth: 1.5, fill: false, pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: '#30D158',
          yAxisID: 'yEx',
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          position: 'top', align: 'end',
          labels: {{ color: '#3A3A3C', font: {{ size: 12 }},
                    usePointStyle: true, padding: 16 }}
        }},
        tooltip: {{
          enabled: false,
        }}
      }},
      scales: {{
        x: {{
          grid: {{ color: 'rgba(0,0,0,0.04)' }},
          ticks: {{ color: '#8E8E93', font: {{ size: 11 }} }}
        }},
        y: {{
          grid: {{ color: 'rgba(0,0,0,0.04)' }},
          ticks: {{
            color: '#8E8E93', font: {{ size: 11 }},
            callback: v => v.toFixed(2) + 'x'
          }}
        }},
        yEx: {{
          position: 'right',
          grid: {{ drawOnChartArea: false }},
          ticks: {{
            color: 'rgba(48,209,88,0.6)', font: {{ size: 10 }},
            callback: v => v.toFixed(2) + 'x'
          }}
        }}
      }}
    }}
  }});

  // 自定义 hover tooltip
  const tip = document.getElementById('chartTip');
  const wrap = document.getElementById('chartWrap');
  navChart.canvas.addEventListener('mousemove', (e) => {{
    const points = navChart.getElementsAtEventForMode(
      e, 'index', {{ intersect: false }}, false);
    if (points.length === 0) {{
      tip.classList.remove('show');
      return;
    }}
    const i = points[0].index;
    const rect = wrap.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const port = DATA.port[i].toFixed(4);
    const bm   = DATA.bm[i].toFixed(4);
    const high = DATA.high[i].toFixed(4);
    const low  = DATA.low[i].toFixed(4);
    const ex   = (DATA.port[i] - DATA.bm[i]).toFixed(4);
    const exPct= (DATA.monthly_ex[i] * 100).toFixed(2);
    const mret = (DATA.monthly_ret[i] * 100).toFixed(2);
    const hret = (DATA.monthly_high_ret[i] * 100).toFixed(2);
    const lret = (DATA.monthly_low_ret[i] * 100).toFixed(2);
    const pos = DATA.monthly_ret[i] >= 0;
    tip.innerHTML = `
      <div class="tt-month">${{DATA.months[i]}}</div>
      <div class="tt-row"><span>策略收益</span><span class="${{pos?'tt-pos':'tt-neg'}}">${{mret}}%</span></div>
      <div class="tt-row"><span>策略净值</span><span>${{port}}</span></div>
      <div class="tt-row"><span>基准净值</span><span>${{bm}}</span></div>
      <div class="tt-row"><span>13%×5 净值</span><span>${{high}}</span></div>
      <div class="tt-row"><span>7%×5 净值</span><span>${{low}}</span></div>
      <div class="tt-row"><span>13%×5 收益</span><span class="${{hret>=0?'tt-pos':'tt-neg'}}">${{hret}}%</span></div>
      <div class="tt-row"><span>7%×5 收益</span><span class="${{lret>=0?'tt-pos':'tt-neg'}}">${{lret}}%</span></div>
      <div class="tt-row"><span>超额净值</span><span class="${{ex>=0?'tt-pos':'tt-neg'}}">${{ex}} (${{exPct}}%)</span></div>
    `;
    tip.style.left = cx + 'px';
    tip.style.top  = cy + 'px';
    tip.classList.add('show');
  }});
  navChart.canvas.addEventListener('mouseleave', () => {{
    tip.classList.remove('show');
  }});

  // ★ 天天基金风格双滑块交互
  initTtjjRange();
  // v5.3: 移除 initTtjjPresets() 因为方块按钮已删除
  // 初始渲染（全部区间）
  applyNavRange();
}}

function renderTtjjTicks() {{
  const ticks = document.getElementById('ttjjTicks');
  if (!ticks) return;
  const n = NAV_STATE.n;
  if (n === 0) return;
  // 取 6 个代表性刻度：首 / 1/4 / 2/4 / 3/4 / 末
  const idxs = [0, Math.floor(n*0.25), Math.floor(n*0.5),
                Math.floor(n*0.75), n - 1];
  const unique = [...new Set(idxs)];
  ticks.innerHTML = unique.map(i =>
    `<span>${{DATA.months[i]}}</span>`).join('');
}}

function initTtjjRange() {{
  const track = document.querySelector('.ttjj-track');
  const hL = document.getElementById('ttjjHandleL');
  const hR = document.getElementById('ttjjHandleR');
  if (!track) return;
  let dragging = null;  // 'L' | 'R' | null

  function getPct(idx) {{
    const n = NAV_STATE.n;
    return n <= 1 ? 0 : (idx / (n - 1)) * 100;
  }}
  function getIdxFromPct(pct) {{
    const n = NAV_STATE.n;
    return Math.round(pct * (n - 1) / 100);
  }}
  function setHandlePos(h, idx) {{
    h.style.left = getPct(idx) + '%';
  }}
  function setFill() {{
    const fill = document.getElementById('ttjjFill');
    const lPct = getPct(NAV_STATE.startIdx);
    const rPct = getPct(NAV_STATE.endIdx);
    fill.style.left = lPct + '%';
    fill.style.width = (rPct - lPct) + '%';
  }}

  function updateLabels() {{
    document.getElementById('ttjjStartVal').textContent =
      DATA.months[NAV_STATE.startIdx] || '—';
    document.getElementById('ttjjEndVal').textContent =
      DATA.months[NAV_STATE.endIdx] || '—';
  }}

  // 初始化位置
  setHandlePos(hL, NAV_STATE.startIdx);
  setHandlePos(hR, NAV_STATE.endIdx);
  setFill();
  updateLabels();
  // 暴露给外部调用
  NAV_STATE.setFill = setFill;
  NAV_STATE.setHandlePos = setHandlePos;
  NAV_STATE.updateLabels = updateLabels;

  function onPointerDown(e) {{
    const target = e.target;
    if (target === hL) dragging = 'L';
    else if (target === hR) dragging = 'R';
    else return;
    e.preventDefault();
    target.setPointerCapture(e.pointerId);
  }}
  function onPointerMove(e) {{
    if (!dragging) return;
    const rect = track.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100,
      ((e.clientX - rect.left) / rect.width) * 100));
    const idx = getIdxFromPct(pct);
    if (dragging === 'L') {{
      NAV_STATE.startIdx = Math.min(idx, NAV_STATE.endIdx);
      setHandlePos(hL, NAV_STATE.startIdx);
    }} else {{
      NAV_STATE.endIdx = Math.max(idx, NAV_STATE.startIdx);
      setHandlePos(hR, NAV_STATE.endIdx);
    }}
    setFill();
    updateLabels();
    applyNavRange();
    // 同步回撤曲线滑块
    DD_STATE.startIdx = NAV_STATE.startIdx;
    DD_STATE.endIdx = NAV_STATE.endIdx;
    if (DD_STATE.setHandlePos) {{
      const dhL = document.getElementById('ddTtjjHandleL');
      const dhR = document.getElementById('ddTtjjHandleR');
      DD_STATE.setHandlePos(dhL, DD_STATE.startIdx);
      DD_STATE.setHandlePos(dhR, DD_STATE.endIdx);
    }}
    if (DD_STATE.setFill) DD_STATE.setFill();
    if (DD_STATE.updateLabels) DD_STATE.updateLabels();
    applyDrawdownRange(DD_STATE.startIdx, DD_STATE.endIdx);
  }}
  function onPointerUp(e) {{
    if (dragging) {{
      try {{ (dragging === 'L' ? hL : hR).releasePointerCapture(e.pointerId); }} catch(_){{}}
      dragging = null;
    }}
  }}

  hL.addEventListener('pointerdown', onPointerDown);
  hR.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);
}}

function initTtjjPresets() {{
  const wrap = document.getElementById('ttjjPresets');
  if (!wrap) return;
  wrap.querySelectorAll('.ttjj-preset').forEach(btn => {{
    btn.addEventListener('click', () => {{
      wrap.querySelectorAll('.ttjj-preset').forEach(
        b => b.classList.remove('active'));
      btn.classList.add('active');
      const preset = btn.dataset.preset;
      const n = NAV_STATE.n;
      let s = 0, e = n - 1;
      const lastMonth = DATA.months[n - 1] || '';
      if (preset === '1y')      s = Math.max(0, n - 12);
      else if (preset === '3y') s = Math.max(0, n - 36);
      else if (preset === '5y') s = Math.max(0, n - 60);
      else if (preset === 'ytd') {{
        const yr = lastMonth.slice(0, 4);
        s = DATA.months.findIndex(m => m.startsWith(yr));
        if (s < 0) s = 0;
      }}
      NAV_STATE.startIdx = s;
      NAV_STATE.endIdx   = e;
      const hL = document.getElementById('ttjjHandleL');
      const hR = document.getElementById('ttjjHandleR');
      if (NAV_STATE.setHandlePos) {{
        NAV_STATE.setHandlePos(hL, s);
        NAV_STATE.setHandlePos(hR, e);
      }}
      if (NAV_STATE.setFill) NAV_STATE.setFill();
      if (NAV_STATE.updateLabels) NAV_STATE.updateLabels();
      applyNavRange();
    }});
  }});
}}

/* ★ 核心：调整区间时，重新计算 0 点（区间起点归一化为 1） */
function applyNavRange() {{
  if (!navChart) return;
  const s = NAV_STATE.startIdx;
  const e = NAV_STATE.endIdx;

  // 原始累计净值
  const portRaw = DATA.port;
  const bmRaw   = DATA.bm;
  const highRaw = DATA.high;
  const lowRaw  = DATA.low;

  // 区间切片 + 起点归一化（起点设为 1.0）
  const portSlice = portRaw.slice(s, e + 1);
  const bmSlice   = bmRaw.slice(s, e + 1);
  const highSlice = highRaw.slice(s, e + 1);
  const lowSlice  = lowRaw.slice(s, e + 1);
  const labels    = DATA.months.slice(s, e + 1);

  const portBase = portSlice[0] || 1.0;
  const bmBase   = bmSlice[0]   || 1.0;
  const highBase = highSlice[0] || 1.0;
  const lowBase  = lowSlice[0]  || 1.0;

  const portAdj  = portSlice.map(v => v / portBase);
  const bmAdj    = bmSlice.map(v => v / bmBase);
  const highAdj  = highSlice.map(v => v / highBase);
  const lowAdj   = lowSlice.map(v => v / lowBase);
  // 超额 = 策略/基准（在新的 0 点基础上）
  const exAdj    = portAdj.map((v, i) => v - bmAdj[i]);

  navChart.data.labels = labels;
  navChart.data.datasets[0].data = portAdj;
  navChart.data.datasets[1].data = bmAdj;
  navChart.data.datasets[2].data = highAdj;
  navChart.data.datasets[3].data = lowAdj;
  navChart.data.datasets[4].data = exAdj;
  navChart.update('none');

  // 同步更新回撤曲线
  applyDrawdownRange(s, e);
}}

/* ──────────── 回撤曲线 + 双滑块 ──────────── */
let ddChart = null;
let DD_STATE = {{
  startIdx: 0, endIdx: 0, n: 0,
}};

function initDrawdownCurve() {{
  const ctx = document.getElementById('ddCurve').getContext('2d');
  const n = DATA.months.length;
  if (n === 0) return;
  DD_STATE.n = n;
  DD_STATE.startIdx = 0;
  DD_STATE.endIdx   = n - 1;

  renderDdTtjjTicks();

  ddChart = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: DATA.months,
      datasets: [
        {{
          label: '策略回撤',
          data: DATA.dd_port,
          borderColor: '#0A84FF',
          backgroundColor: 'rgba(10,132,255,0.12)',
          fill: true, tension: 0.3, borderWidth: 2.5,
          pointRadius: 0, pointHoverRadius: 7,
          pointHoverBackgroundColor: '#0A84FF',
          pointHoverBorderColor: '#fff', pointHoverBorderWidth: 2,
        }},
        {{
          label: '基准回撤',
          data: DATA.dd_bm,
          borderColor: '#FF9F0A',
          backgroundColor: 'rgba(255,159,10,0.08)',
          fill: true, tension: 0.3, borderWidth: 2.5,
          borderDash: [4, 4],
          pointRadius: 0, pointHoverRadius: 7,
          pointHoverBackgroundColor: '#FF9F0A',
          pointHoverBorderColor: '#fff', pointHoverBorderWidth: 2,
        }},
        {{
          label: '13%×5 回撤',
          data: DATA.dd_high,
          borderColor: '#30D158',
          fill: false, tension: 0.3, borderWidth: 2,
          borderDash: [6, 3],
          pointRadius: 0, pointHoverRadius: 6,
          pointHoverBackgroundColor: '#30D158',
          pointHoverBorderColor: '#fff', pointHoverBorderWidth: 2,
        }},
        {{
          label: '7%×5 回撤',
          data: DATA.dd_low,
          borderColor: '#BF5AF2',
          fill: false, tension: 0.3, borderWidth: 2,
          borderDash: [3, 3],
          pointRadius: 0, pointHoverRadius: 6,
          pointHoverBackgroundColor: '#BF5AF2',
          pointHoverBorderColor: '#fff', pointHoverBorderWidth: 2,
        }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          position: 'top', align: 'end',
          labels: {{ color: '#3A3A3C', font: {{ size: 12 }},
                    usePointStyle: true, padding: 16 }}
        }},
        tooltip: {{ enabled: false }},
      }},
      scales: {{
        x: {{
          grid: {{ color: 'rgba(0,0,0,0.04)' }},
          ticks: {{ color: '#8E8E93', font: {{ size: 11 }} }}
        }},
        y: {{
          grid: {{ color: 'rgba(0,0,0,0.04)' }},
          ticks: {{
            color: '#8E8E93', font: {{ size: 11 }},
            callback: v => (v * 100).toFixed(1) + '%'
          }}
        }}
      }}
    }}
  }});

  // hover tooltip
  const tip = document.getElementById('ddTip');
  const wrap = document.getElementById('ddChartWrap');
  ddChart.canvas.addEventListener('mousemove', (e) => {{
    const points = ddChart.getElementsAtEventForMode(
      e, 'index', {{ intersect: false }}, false);
    if (points.length === 0) {{
      tip.classList.remove('show'); return;
    }}
    const i = points[0].index;
    const rect = wrap.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const dp = (DATA.dd_port[i] * 100).toFixed(2);
    const db = (DATA.dd_bm[i] * 100).toFixed(2);
    const dh = (DATA.dd_high[i] * 100).toFixed(2);
    const dl = (DATA.dd_low[i] * 100).toFixed(2);
    tip.innerHTML = `
      <div class="tt-month">${{DATA.months[i]}}</div>
      <div class="tt-row"><span>策略回撤</span><span class="tt-neg">${{dp}}%</span></div>
      <div class="tt-row"><span>基准回撤</span><span class="tt-neg">${{db}}%</span></div>
      <div class="tt-row"><span>13%×5 回撤</span><span class="tt-neg">${{dh}}%</span></div>
      <div class="tt-row"><span>7%×5 回撤</span><span class="tt-neg">${{dl}}%</span></div>
    `;
    tip.style.left = cx + 'px';
    tip.style.top  = cy + 'px';
    tip.classList.add('show');
  }});
  ddChart.canvas.addEventListener('mouseleave', () => {{
    tip.classList.remove('show');
  }});

  initDdRange();
  applyDrawdownRange(0, n - 1);
}}

function renderDdTtjjTicks() {{
  const ticks = document.getElementById('ddTtjjTicks');
  if (!ticks) return;
  const n = DD_STATE.n;
  if (n === 0) return;
  const idxs = [0, Math.floor(n*0.25), Math.floor(n*0.5),
                Math.floor(n*0.75), n - 1];
  const unique = [...new Set(idxs)];
  ticks.innerHTML = unique.map(i =>
    `<span>${{DATA.months[i]}}</span>`).join('');
}}

function initDdRange() {{
  const track = document.querySelector('#ddTtjjRange .ttjj-track');
  const hL = document.getElementById('ddTtjjHandleL');
  const hR = document.getElementById('ddTtjjHandleR');
  if (!track) return;
  let dragging = null;

  function getPct(idx) {{
    const n = DD_STATE.n;
    return n <= 1 ? 0 : (idx / (n - 1)) * 100;
  }}
  function getIdxFromPct(pct) {{
    const n = DD_STATE.n;
    return Math.round(pct * (n - 1) / 100);
  }}
  function setHandlePos(h, idx) {{
    h.style.left = getPct(idx) + '%';
  }}
  function setFill() {{
    const fill = document.getElementById('ddTtjjFill');
    const lPct = getPct(DD_STATE.startIdx);
    const rPct = getPct(DD_STATE.endIdx);
    fill.style.left = lPct + '%';
    fill.style.width = (rPct - lPct) + '%';
  }}
  function updateLabels() {{
    document.getElementById('ddTtjjStartVal').textContent =
      DATA.months[DD_STATE.startIdx] || '—';
    document.getElementById('ddTtjjEndVal').textContent =
      DATA.months[DD_STATE.endIdx] || '—';
  }}

  setHandlePos(hL, DD_STATE.startIdx);
  setHandlePos(hR, DD_STATE.endIdx);
  setFill();
  updateLabels();
  DD_STATE.setFill = setFill;
  DD_STATE.setHandlePos = setHandlePos;
  DD_STATE.updateLabels = updateLabels;

  function onPointerDown(e) {{
    const target = e.target;
    if (target === hL) dragging = 'L';
    else if (target === hR) dragging = 'R';
    else return;
    e.preventDefault();
    target.setPointerCapture(e.pointerId);
  }}
  function onPointerMove(e) {{
    if (!dragging) return;
    const rect = track.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100,
      ((e.clientX - rect.left) / rect.width) * 100));
    const idx = getIdxFromPct(pct);
    if (dragging === 'L') {{
      DD_STATE.startIdx = Math.min(idx, DD_STATE.endIdx);
      setHandlePos(hL, DD_STATE.startIdx);
    }} else {{
      DD_STATE.endIdx = Math.max(idx, DD_STATE.startIdx);
      setHandlePos(hR, DD_STATE.endIdx);
    }}
    setFill();
    updateLabels();
    applyDrawdownRange(DD_STATE.startIdx, DD_STATE.endIdx);
    // 同步净值曲线
    NAV_STATE.startIdx = DD_STATE.startIdx;
    NAV_STATE.endIdx = DD_STATE.endIdx;
    if (NAV_STATE.setHandlePos) {{
      const nhL = document.getElementById('ttjjHandleL');
      const nhR = document.getElementById('ttjjHandleR');
      NAV_STATE.setHandlePos(nhL, NAV_STATE.startIdx);
      NAV_STATE.setHandlePos(nhR, NAV_STATE.endIdx);
    }}
    if (NAV_STATE.setFill) NAV_STATE.setFill();
    if (NAV_STATE.updateLabels) NAV_STATE.updateLabels();
    applyNavRangeOnly();
  }}
  function onPointerUp(e) {{
    if (dragging) {{
      try {{ (dragging === 'L' ? hL : hR).releasePointerCapture(e.pointerId); }} catch(_){{}}
      dragging = null;
    }}
  }}

  hL.addEventListener('pointerdown', onPointerDown);
  hR.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);
}}

function applyDrawdownRange(s, e) {{
  if (!ddChart) return;
  const labels = DATA.months.slice(s, e + 1);
  ddChart.data.labels = labels;
  ddChart.data.datasets[0].data = DATA.dd_port.slice(s, e + 1);
  ddChart.data.datasets[1].data = DATA.dd_bm.slice(s, e + 1);
  ddChart.data.datasets[2].data = DATA.dd_high.slice(s, e + 1);
  ddChart.data.datasets[3].data = DATA.dd_low.slice(s, e + 1);
  ddChart.update('none');
}}

/* 仅更新净值曲线数据，不触发回撤同步（防止循环） */
function applyNavRangeOnly() {{
  if (!navChart) return;
  const s = NAV_STATE.startIdx;
  const e = NAV_STATE.endIdx;
  const portRaw = DATA.port;
  const bmRaw   = DATA.bm;
  const highRaw = DATA.high;
  const lowRaw  = DATA.low;
  const portSlice = portRaw.slice(s, e + 1);
  const bmSlice   = bmRaw.slice(s, e + 1);
  const highSlice = highRaw.slice(s, e + 1);
  const lowSlice  = lowRaw.slice(s, e + 1);
  const labels    = DATA.months.slice(s, e + 1);
  const portBase = portSlice[0] || 1.0;
  const bmBase   = bmSlice[0]   || 1.0;
  const highBase = highSlice[0] || 1.0;
  const lowBase  = lowSlice[0]  || 1.0;
  const portAdj  = portSlice.map(v => v / portBase);
  const bmAdj    = bmSlice.map(v => v / bmBase);
  const highAdj  = highSlice.map(v => v / highBase);
  const lowAdj   = lowSlice.map(v => v / lowBase);
  const exAdj    = portAdj.map((v, i) => v - bmAdj[i]);
  navChart.data.labels = labels;
  navChart.data.datasets[0].data = portAdj;
  navChart.data.datasets[1].data = bmAdj;
  navChart.data.datasets[2].data = highAdj;
  navChart.data.datasets[3].data = lowAdj;
  navChart.data.datasets[4].data = exAdj;
  navChart.update('none');
}}

/* ──────────── 归因 ──────────── */
function initAttribution() {{
  // ★ v5.5: 修复 - Chart.js 已就绪, 隐藏 fallback, 显示 chart wrap
  ['brinsonFallback', 'ffFallback', 'barraFallback'].forEach(function(id) {{
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  }});
  ['brinsonChartWrap', 'ffChartWrap', 'barraChartWrap'].forEach(function(id) {{
    const el = document.getElementById(id);
    if (el) el.style.display = 'block';
  }});
  document.querySelectorAll('#attrTabs .tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('#attrTabs .tab-btn')
        .forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane')
        .forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + btn.dataset.tab)
        .classList.add('active');
    }});
  }});
  renderBrinson();
  renderFiveFactor();
  renderBarra();
}}

function fmtPct(v) {{
  return (v * 100).toFixed(2) + '%';
}}
function signClass(v) {{
  if (v > 1e-6) return 'pos';
  if (v < -1e-6) return 'neg';
  return '';
}}

function renderBrinson() {{
  const d = DATA.brinson;
  if (!d || !d.monthly) return;
  // 累计
  const s = d.summary;
  // ★ v5.3: summary 是 14 年累计数字 (allocation/selection/interaction/total)
  //   显示时加 "累计" 前缀, 同时给年化值 (除以年数, 默认 12 年)
  const n = d.monthly.length;
  const years = n > 0 ? (n / 12) : 1;
  const ann = (v) => Math.abs(v) < 0.0001 ? 0 : v / years;
  document.getElementById('brinsonSummary').innerHTML = `
    <div class="kv" style="margin-bottom:10px">
      <div class="kv-label">行业配置 α (累计)</div>
      <div class="kv-value ${{signClass(s.alloc)}}">${{fmtPct(s.alloc)}}</div>
      <div class="kv-sub">年化 ≈ ${{fmtPct(ann(s.alloc))}}</div>
    </div>
    <div class="kv" style="margin-bottom:10px">
      <div class="kv-label">个股选择 α (累计)</div>
      <div class="kv-value ${{signClass(s.sel)}}">${{fmtPct(s.sel)}}</div>
      <div class="kv-sub">年化 ≈ ${{fmtPct(ann(s.sel))}}</div>
    </div>
    <div class="kv" style="margin-bottom:10px">
      <div class="kv-label">行业 × 个股 交互 (累计)</div>
      <div class="kv-value ${{signClass(s.inter)}}">${{fmtPct(s.inter)}}</div>
      <div class="kv-sub">年化 ≈ ${{fmtPct(ann(s.inter))}}</div>
    </div>
    <div class="kv" style="background:rgba(10,132,255,0.08);border:1px solid rgba(10,132,255,0.2)">
      <div class="kv-label">累计归因合计 (${{n}} 个月)</div>
      <div class="kv-value ${{signClass(s.total)}}" style="font-size:20px">${{fmtPct(s.total)}}</div>
      <div class="kv-sub">年化 ≈ ${{fmtPct(ann(s.total))}}</div>
    </div>
  `;
  // 堆叠柱
  const ctx = document.getElementById('brinsonChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: d.monthly.map(x => x.month),
      datasets: [
        {{ label: '配置', data: d.monthly.map(x => x.allocation),
           backgroundColor: 'rgba(10,132,255,0.7)' }},
        {{ label: '选择', data: d.monthly.map(x => x.selection),
           backgroundColor: 'rgba(48,209,88,0.7)' }},
        {{ label: '交互', data: d.monthly.map(x => x.interaction),
           backgroundColor: 'rgba(191,90,242,0.7)' }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{
        x: {{ stacked: true, ticks: {{ color: '#8E8E93', font: {{size:10}} }} }},
        y: {{ stacked: true,
             ticks: {{ callback: v => (v*100).toFixed(1)+'%' }},
             grid: {{ color: 'rgba(0,0,0,0.04)' }} }}
      }},
      plugins: {{ legend: {{ position: 'top', align: 'end' }} }}
    }}
  }});
  // 行业表
  const tbody = document.querySelector('#brinsonIndTable tbody');
  d.by_industry.slice(0, 30).forEach(r => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{r.industry}}</td>
      <td class="num ${{signClass(r.allocation)}}">${{fmtPct(r.allocation)}}</td>
      <td class="num ${{signClass(r.selection)}}">${{fmtPct(r.selection)}}</td>
      <td class="num ${{signClass(r.interaction)}}">${{fmtPct(r.interaction)}}</td>
      <td class="num ${{signClass(r.total)}}">${{fmtPct(r.total)}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

function renderFiveFactor() {{
  const d = DATA.ff;
  if (!d || !d.rows) return;
  const a = d.avg || {{}};
  const items = [
    ['α (年化)',      a.alpha || 0, true],
    ['β·MKT',         a.beta_mkt || 0, false],
    ['β·SMB',         a.beta_smb || 0, false],
    ['β·HML',         a.beta_hml || 0, false],
    ['β·RMW',         a.beta_rmw || 0, false],
    ['β·CMA',         a.beta_cma || 0, false],
    ['β·MOM',         a.beta_mom || 0, false],
    ['R²',            a.r2 || 0, false],
  ];
  document.getElementById('ffAvg').innerHTML = items.map(([k, v, isAlpha]) => `
    <div class="kv">
      <div class="kv-label">${{k}}</div>
      <div class="kv-value ${{isAlpha ? signClass(v) : ''}}">
        ${{isAlpha ? fmtPct(v) : (+v).toFixed(3)}}
      </div>
    </div>
  `).join('');
  const ctx = document.getElementById('ffChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: d.rows.map(r => r.month),
      datasets: [
        {{ label: 'α (年化)', data: d.rows.map(r => r.alpha),
           borderColor: '#30D158', backgroundColor: 'rgba(48,209,88,0.15)',
           fill: true, tension: 0.3, borderWidth: 2, pointRadius: 0 }},
        {{ label: 'β·MKT', data: d.rows.map(r => r.beta_mkt),
           borderColor: '#0A84FF', borderWidth: 1.5, pointRadius: 0 }},
        {{ label: 'β·SMB', data: d.rows.map(r => r.beta_smb),
           borderColor: '#FF9F0A', borderWidth: 1.5, pointRadius: 0 }},
        {{ label: 'β·HML', data: d.rows.map(r => r.beta_hml),
           borderColor: '#BF5AF2', borderWidth: 1.5, pointRadius: 0 }},
        {{ label: 'β·RMW', data: d.rows.map(r => r.beta_rmw),
           borderColor: '#FF453A', borderWidth: 1.5, pointRadius: 0 }},
        {{ label: 'β·CMA', data: d.rows.map(r => r.beta_cma),
           borderColor: '#5AC8FA', borderWidth: 1.5, pointRadius: 0 }},
        {{ label: 'β·MOM', data: d.rows.map(r => r.beta_mom),
           borderColor: '#FFD60A', borderWidth: 1.5, pointRadius: 0 }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{ legend: {{ position: 'top', align: 'end',
                              labels: {{ usePointStyle: true,
                                         font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8E8E93', font: {{size:10}} }} }},
        y: {{ grid: {{ color: 'rgba(0,0,0,0.04)' }} }}
      }}
    }}
  }});
}}

function renderBarra() {{
  const d = DATA.barra;
  if (!d || !d.rows) return;
  const a = d.avg || {{}};
  const factors = Object.keys(a).filter(k => k.startsWith('ret_'));
  const items = factors.map(k => `
    <div class="kv">
      <div class="kv-label">${{k.replace('ret_','').replace('barra_','')}}</div>
      <div class="kv-value ${{signClass(a[k])}}">${{fmtPct(a[k])}}</div>
    </div>
  `).join('');
  document.getElementById('barraAvg').innerHTML = items;
  const ctx = document.getElementById('barraChart').getContext('2d');
  const ds = factors.map((k, i) => {{
    const color = ['#0A84FF','#FF9F0A','#30D158','#FF453A',
                   '#BF5AF2','#5AC8FA','#FFD60A','#64D2FF',
                   '#FF6482','#AC8E68'][i % 10];
    return {{
      label: k.replace('ret_','').replace('barra_',''),
      data: d.rows.map(r => r[k] || 0),
      borderColor: color, backgroundColor: color + '33',
      borderWidth: 1.5, pointRadius: 0, tension: 0.3,
    }};
  }});
  new Chart(ctx, {{
    type: 'line',
    data: {{ labels: d.rows.map(r => r.month), datasets: ds }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', align: 'end',
                              labels: {{ usePointStyle: true,
                                         font: {{ size: 10 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8E8E93', font: {{size:10}} }} }},
        y: {{ ticks: {{ callback: v => (v*100).toFixed(2)+'%' }} }}
      }}
    }}
  }});
}}

/* ──────────── SHAP 深度分析（4 子 Tab） ──────────── */
const SHAP_COLORS = ['#0A84FF', '#FF9F0A', '#30D158', '#FF453A', '#BF5AF2'];
let shapTop5Chart = null;
let shapCorrChart = null;
let shapTsCharts  = [];
let shapChart     = null;

const FACTOR_DESC = {{
  'momentum_return_5d': '5日收益率',
  'momentum_return_10d': '10日收益率',
  'momentum_return_20d': '20日收益率（1个月）',
  'momentum_return_40d': '40日收益率',
  'momentum_return_60d': '60日收益率（3个月）',
  'momentum_return_90d': '90日收益率',
  'momentum_return_120d': '120日收益率（6个月）',
  'momentum_return_180d': '180日收益率',
  'momentum_ret_1m': '1个月月度收益',
  'momentum_ret_3m': '3个月月度收益',
  'momentum_ret_6m': '6个月月度收益',
  'momentum_ret_12m': '12个月月度收益',
  'momentum_reversal_5d': '5日短期反转',
  'momentum_reversal_10d': '10日短期反转',
  'momentum_medium_60d': '60日中期动量',
  'momentum_acceleration': '动量加速度',
  'momentum_long_240d': '240日长期动量',
  'momentum_mean_reversion': '长期均值回复',
  'momentum_sharpe_20d': '20日动量信息比率',
  'momentum_sharpe_60d': '60日动量信息比率',
  'momentum_sharpe_120d': '120日动量信息比率',
  'momentum_sharpe_240d': '240日动量信息比率',
  'momentum_rsi_6d': '6日RSI',
  'momentum_rsi_14d': '14日RSI',
  'momentum_rsi_20d': '20日RSI',
  'momentum_rsi_30d': '30日RSI',
  'momentum_price_position_10d': '10日价格位置',
  'momentum_price_position_20d': '20日价格位置',
  'momentum_price_position_60d': '60日价格位置',
  'momentum_price_position_120d': '120日价格位置',
  'momentum_score': '动量综合得分',
  'momentum_change_5d': '5日动量变化率',
  'momentum_change_10d': '10日动量变化率',
  'momentum_change_20d': '20日动量变化率',
  'momentum_change_60d': '60日动量变化率',
  'momentum_log_return_20d': '20日对数收益',
  'momentum_log_return_60d': '60日对数收益',
  'momentum_log_return_120d': '120日对数收益',
  'momentum_log_return_240d': '240日对数收益',
  'momentum_weighted_20d': '20日加权动量',
  'momentum_weighted_60d': '60日加权动量',
  'momentum_weighted_120d': '120日加权动量',
  'momentum_weighted_240d': '240日加权动量',
  'momentum_consistency_20d': '20日动量一致性',
  'momentum_consistency_60d': '60日动量一致性',
  'momentum_consistency_120d': '120日动量一致性',
  'momentum_consistency_240d': '240日动量一致性',
  'momentum_extreme_count_20d': '20日极端收益计数',
  'momentum_extreme_count_60d': '60日极端收益计数',
  'momentum_extreme_count_120d': '120日极端收益计数',
  'momentum_extreme_count_240d': '240日极端收益计数',
  'reversal_1m': '1个月反转',
  'reversal_3m': '3个月反转',
  'volatility_hist_5d': '5日历史波动率',
  'volatility_hist_10d': '10日历史波动率',
  'volatility_hist_20d': '20日历史波动率',
  'volatility_hist_40d': '40日历史波动率',
  'volatility_hist_60d': '60日历史波动率',
  'volatility_hist_90d': '90日历史波动率',
  'volatility_hist_120d': '120日历史波动率',
  'volatility_hist_240d': '240日历史波动率',
  'volatility_parkinson_20d': '20日Parkinson波动率',
  'volatility_parkinson_60d': '60日Parkinson波动率',
  'volatility_parkinson_120d': '120日Parkinson波动率',
  'volatility_parkinson_240d': '240日Parkinson波动率',
  'volatility_atr_14d': '14日ATR',
  'volatility_atr_20d': '20日ATR',
  'volatility_skew_60d': '60日波动率偏度',
  'volatility_kurt_60d': '60日波动率峰度',
  'volatility_downside_60d': '60日下行波动率',
  'volatility_gk_20d': '20日Garman-Klass波动率',
  'volatility_gk_60d': '60日Garman-Klass波动率',
  'volatility_rs_20d': '20日Rogers-Satchell波动率',
  'volatility_rs_60d': '60日Rogers-Satchell波动率',
  'volatility_yz_20d': '20日Yang-Zhang波动率',
  'volatility_yz_60d': '60日Yang-Zhang波动率',
  'volatility_change_20d': '20日波动率变化率',
  'volatility_change_60d': '60日波动率变化率',
  'volatility_mean_reversion': '波动率均值回复',
  'quality_roe': 'ROE净资产收益率',
  'quality_roa': 'ROA总资产收益率',
  'quality_gross_margin': '毛利率',
  'quality_debt_ratio': '资产负债率',
  'quality_current_ratio': '流动比率',
  'quality_quick_ratio': '速动比率',
  'quality_profit_growth': '净利润增长率',
  'quality_roe_ex_nonrecurring': '扣非ROE',
  'quality_roe_weighted': '加权ROE',
  'quality_inv_turnover': '存货周转率',
  'quality_ar_turnover': '应收账款周转率',
  'quality_ocf_to_revenue': '经营现金流/营收',
  'quality_ocf_to_profit': '盈余质量(经营CF/净利润)',
  'quality_debt_equity_ratio': '产权比率',
  'value_pe_ratio': '市盈率PE',
  'value_pb_ratio': '市净率PB',
  'value_ps_ratio': '市销率PS',
  'value_pcf_ratio': '市现率PCF',
  'value_ep_ratio': '盈利收益率EP(1/PE)',
  'value_bp_ratio': '账面市值比BP(1/PB)',
  'value_sp_ratio': '销售市值比SP(1/PS)',
  'value_cfp_ratio': '现金流市值比CFP(1/PCF)',
  'value_ep_bp_avg': 'EP与BP均值',
  'growth_revenue_yoy': '营收同比增长率',
  'growth_netprofit_yoy': '净利润同比增长率',
  'growth_netprofit_q_yoy': '单季净利润同比增长率',
  'growth_roe_trend': 'ROE趋势',
  'growth_rev_profit_diff': '营收与净利润增长差',
  'growth_eps_yoy': 'EPS增长率',
  'technical_ma_5d': '5日均线',
  'technical_ma_10d': '10日均线',
  'technical_ma_20d': '20日均线',
  'technical_ma_60d': '60日均线',
  'technical_ma_120d': '120日均线',
  'technical_ma_240d': '240日均线',
  'technical_price_ma_ratio_5d': '价格/5日均线比率',
  'technical_price_ma_ratio_10d': '价格/10日均线比率',
  'technical_price_ma_ratio_20d': '价格/20日均线比率',
  'technical_price_ma_ratio_60d': '价格/60日均线比率',
  'technical_ema_12d': '12日指数均线',
  'technical_ema_26d': '26日指数均线',
  'technical_ema_50d': '50日指数均线',
  'technical_ema_200d': '200日指数均线',
  'technical_macd_dif': 'MACD DIF线',
  'technical_macd_dea': 'MACD DEA线',
  'technical_macd_hist': 'MACD柱状图',
  'technical_kdj_rsv_9d': '9日KDJ RSV',
  'technical_kdj_rsv_14d': '14日KDJ RSV',
  'technical_kdj_rsv_20d': '20日KDJ RSV',
  'technical_kdj_rsv_30d': '30日KDJ RSV',
  'technical_boll_width_20d': '20日布林带宽度',
  'technical_boll_width_60d': '60日布林带宽度',
  'technical_boll_pos_20d': '20日布林带位置',
  'technical_boll_pos_60d': '60日布林带位置',
  'technical_obv': 'OBV能量潮',
  'technical_volume_ma_5d': '5日成交量均线',
  'technical_volume_ma_10d': '10日成交量均线',
  'technical_volume_ma_20d': '20日成交量均线',
  'technical_volume_ma_60d': '60日成交量均线',
  'technical_pv_corr_5d': '5日量价相关性',
  'technical_pv_corr_10d': '10日量价相关性',
  'technical_pv_corr_20d': '20日量价相关性',
  'technical_pv_corr_60d': '60日量价相关性',
  'technical_willr_10d': '10日Williams %R',
  'technical_willr_20d': '20日Williams %R',
  'technical_willr_60d': '60日Williams %R',
  'technical_willr_120d': '120日Williams %R',
  'technical_cci_10d': '10日CCI',
  'technical_cci_20d': '20日CCI',
  'technical_cci_60d': '60日CCI',
  'technical_cci_120d': '120日CCI',
  'technical_plus_di_14d': '14日+DI',
  'technical_plus_di_20d': '20日+DI',
  'technical_minus_di_14d': '14日-DI',
  'technical_minus_di_20d': '20日-DI',
  'technical_momentum_10d': '10日动量振荡器',
  'technical_momentum_20d': '20日动量振荡器',
  'technical_momentum_60d': '60日动量振荡器',
  'technical_momentum_120d': '120日动量振荡器',
  'technical_roc_10d': '10日变化率ROC',
  'technical_roc_20d': '20日变化率ROC',
  'technical_roc_60d': '60日变化率ROC',
  'technical_roc_120d': '120日变化率ROC',
  'liquidity_avg_volume_5d': '5日平均成交量',
  'liquidity_avg_volume_10d': '10日平均成交量',
  'liquidity_avg_volume_20d': '20日平均成交量',
  'liquidity_avg_turnover_5d': '5日平均换手率',
  'liquidity_avg_turnover_10d': '10日平均换手率',
  'liquidity_avg_turnover_20d': '20日平均换手率',
  'liquidity_volume_change_5d': '5日成交量变化率',
  'liquidity_volume_change_10d': '10日成交量变化率',
  'liquidity_volume_change_20d': '20日成交量变化率',
  'liquidity_avg_amount_5d': '5日平均成交额',
  'liquidity_avg_amount_10d': '10日平均成交额',
  'liquidity_avg_amount_20d': '20日平均成交额',
  'liquidity_amihud_5d': '5日Amihud非流动性',
  'liquidity_amihud_10d': '10日Amihud非流动性',
  'liquidity_amihud_20d': '20日Amihud非流动性',
  'liquidity_amihud_60d': '60日Amihud非流动性',
  'size_log_mcap': '对数总市值',
  'size_log_mcap_neg': '负对数总市值(小市值溢价)',
  'size_sqrt_mcap': '市值平方根(Barra Size非线性)',
  'size_log_mcap_cubed': '对数市值立方(Barra Size³)',
  'size_circ_ratio': '流通市值占比',
  'dividend_payout_ratio': '股息支付率',
  'dividend_yield': '股息率',
  'dividend_payout_stability': '股息支付稳定性',
  'dividend_yield_vs_market': '股息率vs市场均值',
  'dividend_growth_yoy': '股息同比增长',
  'barra_beta': 'Barra Beta',
  'barra_momentum': 'Barra动量',
  'barra_size': 'Barra规模',
  'barra_earnings_yield': 'Barra盈利收益率',
  'barra_value': 'Barra价值',
  'barra_volatility': 'Barra波动率',
  'barra_liquidity': 'Barra流动性',
  'barra_leverage': 'Barra杠杆',
  'barra_growth': 'Barra成长',
  'barra_quality': 'Barra质量',
  'aqr_momentum': 'AQR动量',
  'aqr_value': 'AQR价值',
  'aqr_quality': 'AQR质量',
  'aqr_size': 'AQR规模',
  'aqr_low_beta': 'AQR低Beta',
  'aqr_profit_growth': 'AQR利润增长',
  'jq_momentum_5d': '聚宽5日动量',
  'jq_momentum_10d': '聚宽10日动量',
  'jq_momentum_20d': '聚宽20日动量',
  'jq_momentum_60d': '聚宽60日动量',
  'jq_momentum_120d': '聚宽120日动量',
  'jq_volatility_5d': '聚宽5日波动率',
  'jq_volatility_10d': '聚宽10日波动率',
  'jq_volatility_20d': '聚宽20日波动率',
  'jq_volatility_60d': '聚宽60日波动率',
  'jq_volatility_120d': '聚宽120日波动率',
  'jq_turnover_5d': '聚宽5日换手率',
  'jq_turnover_10d': '聚宽10日换手率',
  'jq_turnover_20d': '聚宽20日换手率',
  'jq_turnover_60d': '聚宽60日换手率',
  'jq_turnover_120d': '聚宽120日换手率',
  'jq_rsi_20d': '聚宽20日RSI',
  'jq_price_position_20d': '聚宽20日价格位置',
  'jq_range_20d': '聚宽20日振幅',
  'jq_ma_deviation_20d': '聚宽20日均线偏离度',
  'jq_volume_ratio_20d': '聚宽20日量比',
  'deriv_mom_vol_ratio': '动量/波动比',
  'deriv_rsi_adjusted_mom': 'RSI调整动量',
  'deriv_vol_adjusted_return': '波动调整收益',
  'deriv_momentum_consistency': '动量一致性得分',
  'deriv_risk_adjusted_mom': '风险调整动量',
  'roll_max_20d': '20日滚动最高价',
  'roll_max_60d': '60日滚动最高价',
  'roll_min_20d': '20日滚动最低价',
  'roll_min_60d': '60日滚动最低价',
  'roll_mean_20d': '20日滚动均价',
  'roll_mean_60d': '60日滚动均价',
  'roll_ret_std_20d': '20日收益标准差',
  'roll_ret_std_60d': '60日收益标准差',
  'roll_vol_mean_20d': '20日成交量均值',
  'roll_vol_mean_60d': '60日成交量均值',
  'roll_vol_std_20d': '20日成交量标准差',
  'roll_vol_std_60d': '60日成交量标准差',
  'sentiment_daily_return': '日涨跌幅',
  'sentiment_amplitude': '日振幅',
  'sentiment_open_position': '开盘价相对位置',
  'sentiment_close_position': '收盘价相对位置',
  'sentiment_pv_divergence': '量价背离',
  'nm_gross_profitability': 'Novy-Marx毛利盈利能力',
  'ps_amihud_illiq': 'Amihud非流动性',
  'ps_volume_shock': '成交量冲击',
  'ps_price_impact': '价格冲击',
  'sup_mom_pv_5d': '5日量价动量',
  'sup_mom_pv_20d': '20日量价动量',
  'sup_mom_strength_20d': '20日动量强度',
  'sup_mom_strength_60d': '60日动量强度',
  'sup_mom_strength_120d': '120日动量强度',
  'sup_mom_jt_12_1': 'Jegadeesh-Titman 12-1动量',
  'sup_mom_relative_20d': '20日相对动量',
  'sup_mom_relative_60d': '60日相对动量',
  'sup_mom_accel': '动量加速度',
  'sup_tech_rsi_6': '6日RSI',
  'sup_tech_rsi_14': '14日RSI',
  'sup_tech_rsi_24': '24日RSI',
  'sup_tech_aroon_up_14': '14日Aroon上升指标',
  'sup_tech_aroon_down_14': '14日Aroon下降指标',
  'sup_tech_aroon_osc_14': '14日Aroon振荡器',
  'sup_tech_aroon_up_28': '28日Aroon上升指标',
  'sup_tech_aroon_down_28': '28日Aroon下降指标',
  'sup_tech_aroon_osc_28': '28日Aroon振荡器',
  'sup_tech_ma_cross_5_20': 'MA5/MA20金叉信号',
  'sup_tech_ma_cross_20_60': 'MA20/MA60金叉信号',
  'sup_val_pe_ttm': 'PE(TTM)',
  'sup_val_pb': 'PB',
  'sup_val_ps_ttm': 'PS(TTM)',
  'sup_val_pcf_ttm': 'PCF(TTM)',
  'sup_val_div_yield': '股息率',
  'sup_fin_roe_ttm': 'ROE(TTM)',
  'sup_fin_roa_ttm': 'ROA(TTM)',
  'sup_fin_gross_margin': '毛利率',
  'sup_fin_net_margin': '净利率',
  'sup_fin_asset_turnover': '总资产周转率',
  'sup_fin_current_ratio': '流动比率',
  'sup_fin_quick_ratio': '速动比率',
  'sup_fin_debt_ratio': '资产负债率',
  'sup_fin_interest_coverage': '利息覆盖率',
  'sup_fin_operating_cf': '经营现金流',
  'sup_risk_downside_20d': '20日下行风险',
  'sup_risk_downside_60d': '60日下行风险',
  'sup_risk_max_dd_60d': '60日最大回撤',
  'sup_risk_max_dd_120d': '120日最大回撤',
  'sup_risk_var_20d': '20日VaR(5%)',
  'sup_risk_var_60d': '60日VaR(5%)',
  'sup_risk_cvar_20d': '20日CVaR',
  'sup_risk_cvar_60d': '60日CVaR',
  'sup_liq_amount_shock_5d': '5日成交额冲击',
  'sup_liq_amount_shock_10d': '10日成交额冲击',
  'sup_liq_amount_shock_20d': '20日成交额冲击',
  'sup_liq_vol_std_5d': '5日成交量波动',
  'sup_liq_vol_std_10d': '10日成交量波动',
  'sup_liq_vol_std_20d': '20日成交量波动',
  'sup_liq_amihud_5d': '5日Amihud非流动性',
  'sup_liq_amihud_10d': '10日Amihud非流动性',
  'sup_liq_amihud_20d': '20日Amihud非流动性',
  'sup_liq_amihud_60d': '60日Amihud非流动性',
  'sup_price_pos_5d': '5日价格位置',
  'sup_price_pos_10d': '10日价格位置',
  'sup_price_pos_20d': '20日价格位置',
  'sup_price_pos_60d': '60日价格位置',
  'sup_ret_skew_20d': '20日收益偏度',
  'sup_ret_skew_60d': '60日收益偏度',
  'sup_ret_kurt_20d': '20日收益峰度',
  'sup_ret_kurt_60d': '60日收益峰度',
  'sup_vol_change': '波动率变化',
  'sup_sharpe_60d': '60日夏普比率',
  'sup_sortino_60d': '60日索提诺比率',
  'sup_calmar_120d': '120日卡玛比率',
  'macro_shibor_on': 'Shibor隔夜利率',
  'macro_shibor_1w': 'Shibor 1周利率',
  'macro_shibor_1m': 'Shibor 1个月利率',
  'macro_shibor_3m': 'Shibor 3个月利率',
  'macro_bond_yield_1y': '1年期国债收益率',
  'macro_bond_yield_10y': '10年期国债收益率',
  'macro_index_sh_close': '上证指数收盘价',
  'macro_index_sh_pct_chg': '上证指数涨跌幅',
  'macro_index_sz_close': '深证成指收盘价',
  'macro_index_sz_pct_chg': '深证成指涨跌幅',
  'macro_index_cyb_close': '创业板指收盘价',
  'macro_index_cyb_pct_chg': '创业板指涨跌幅',
  'ln_market_cap': '对数市值',
  'pe_ttm': 'PE(TTM)',
  'netprofit_yoy': '净利润同比增长率',
  'roe': 'ROE',
}};

function getFactorDesc(name) {{
  return FACTOR_DESC[name] || name;
}}

function shapHasData() {{
  const d = DATA.shap || {{}};
  return !!(d.global_top5 && d.global_top5.factors
            && d.global_top5.factors.length);
}}

function initShapTabs() {{
  // 主 Tab 切换
  document.querySelectorAll('#shapMainTabs .tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const t = btn.dataset.tab;
      document.querySelectorAll('#shapMainTabs .tab-btn')
        .forEach(b => b.classList.remove('active'));
      document.querySelectorAll('[id^="tab-shap-"]')
        .forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-shap-' + t)
        .classList.add('active');
      if (t === 'top5')        renderShapTop5();
      else if (t === 'corr')   renderShapCorr();
      else if (t === 'ts')     renderShapTs();
      else if (t === 'monthly') initShapMonthTabs();
    }});
  }});

  if (!shapHasData()) {{
    document.getElementById('shapMainTabs').innerHTML =
      '<div style="color:#8E8E93;padding:6px 12px">暂无 SHAP 数据（M5 调用时需开启 compute_shap=True）</div>';
    return;
  }}
  // 默认渲染 Top 5
  renderShapTop5();
}}

/* ★ v5.4: Chart.js 不可用时, SHAP Tab 切换器 (只渲染 SVG 热力图 + 静态文本) */
function initShapTabsFallback() {{
  if (!shapHasData()) {{
    document.getElementById('shapMainTabs').innerHTML =
      '<div style="color:#8E8E93;padding:6px 12px">暂无 SHAP 数据</div>';
    return;
  }}
  // ★ 静态渲染: Top5 核心因子用纯文本表格
  const d = DATA.shap;
  const factors = d.global_top5.factors || [];
  const meanAbs = d.global_top5.mean_abs || [];
  const signed  = d.global_top5.signed || [];
  let top5html = '<div class="kv-grid">';
  for (let i = 0; i < factors.length; i++) {{
    const cls = signed[i] >= 0 ? 'pos' : 'neg';
    top5html += `
      <div class="kv">
        <div class="kv-label">#${{i+1}} ${{factors[i]}}</div>
        <div class="kv-value ${{cls}}">
          ${{(signed[i] * 100).toFixed(2)}}%
        </div>
        <div class="kv-sub">|SHAP| 均值 ${{meanAbs[i].toFixed(4)}}</div>
      </div>
    `;
  }}
  top5html += '</div>';
  const top5El = document.getElementById('tab-shap-top5');
  if (top5El) top5El.innerHTML = top5html;

  // 相关性 Tab 仍然可用 (纯 SVG)
  const corrEl = document.getElementById('tab-shap-corr');
  if (corrEl) corrEl.style.display = 'block';

  // 时序 + 月度详情: 显示文本提示
  const tsEl = document.getElementById('tab-shap-ts');
  if (tsEl) tsEl.innerHTML =
    '<div class="kv-empty" style="padding:40px">Chart.js 不可用, 时序图无法显示</div>';
  const monthlyEl = document.getElementById('tab-shap-monthly');
  if (monthlyEl) monthlyEl.innerHTML =
    '<div class="kv-empty" style="padding:40px">Chart.js 不可用, 月度详情图无法显示</div>';

  // 主 Tab 切换
  document.querySelectorAll('#shapMainTabs .tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const t = btn.dataset.tab;
      document.querySelectorAll('#shapMainTabs .tab-btn')
        .forEach(b => b.classList.remove('active'));
      document.querySelectorAll('[id^="tab-shap-"]')
        .forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById('tab-shap-' + t);
      if (target) target.classList.add('active');
      // corr tab 默认渲染
      if (t === 'corr' && typeof renderShapCorr === 'function') {{
        renderShapCorr();
      }}
    }});
  }});
}}

/* ── ① Top 5 核心因子 ── */
function renderShapTop5() {{
  const d = DATA.shap;
  if (!d || !d.global_top5 || !d.global_top5.factors.length) return;
  if (shapTop5Chart) {{ shapTop5Chart.destroy(); shapTop5Chart = null; }}

  const factors = d.global_top5.factors;
  const meanAbs = d.global_top5.mean_abs;
  const signed  = d.global_top5.signed;

  // 颜色按 rank 分
  const colors = factors.map((_, i) => SHAP_COLORS[i % 5]);

  const wrap = document.getElementById('tab-shap-top5');
  const tip  = document.getElementById('shapTop5Tip');

  const ctx = document.getElementById('shapTop5Chart').getContext('2d');
  shapTop5Chart = new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: factors,
      datasets: [{{
        label: '平均 |SHAP|',
        data: meanAbs,
        backgroundColor: colors.map(c => c + 'C0'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 6,
        barPercentage: 0.75,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      animation: {{ duration: 600 }},
      onHover: (e, els) => {{
        if (els.length === 0) {{
          tip.classList.remove('show'); return;
        }}
        const i = els[0].index;
        const rect = wrap.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const sg = signed[i];
        const cls = sg >= 0 ? 'tt-pos' : 'tt-neg';
        tip.innerHTML = `
          <div class="tt-month">${{factors[i]}}</div>
          <div class="tt-row"><span>排名</span><span>#${{i+1}} / ${{factors.length}}</span></div>
          <div class="tt-row"><span>平均 |SHAP|</span><span>${{meanAbs[i].toFixed(6)}}</span></div>
          <div class="tt-row"><span>带符号均值</span><span class="${{cls}}">${{sg >= 0 ? '+' : ''}}${{sg.toFixed(6)}}</span></div>
          <div class="tt-row"><span>方向</span><span class="${{cls}}">${{sg >= 0 ? '正向贡献' : '负向贡献'}}</span></div>
        `;
        tip.style.left = cx + 'px';
        tip.style.top  = cy + 'px';
        tip.classList.add('show');
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ enabled: false }},
        title: {{
          display: true,
          text: 'Top 5 核心因子（按所有月平均 |SHAP| 排序）',
          color: '#3A3A3C', font: {{ size: 13, weight: '600' }},
          padding: {{ bottom: 10 }}
        }},
      }},
      scales: {{
        x: {{
          grid: {{ color: 'rgba(0,0,0,0.04)' }},
          ticks: {{ color: '#8E8E93', font: {{ size: 11 }},
                   callback: v => v.toFixed(4) }}
        }},
        y: {{ ticks: {{ font: {{ size: 12, weight: '500' }},
                        color: '#1C1C1E' }} }}
      }}
    }}
  }});

  wrap.addEventListener('mouseleave',
    () => tip.classList.remove('show'));

  // ★ 新增: 因子中文介绍
  const descWrap = document.getElementById('shapTop5Desc');
  if (descWrap) {{
    descWrap.innerHTML = factors.map((f, i) => {{
      const desc = getFactorDesc(f);
      const sg = signed[i];
      const cls = sg >= 0 ? 'pos' : 'neg';
      const dir = sg >= 0 ? '正向' : '负向';
      return `<div class="kv" style="border-left:3px solid ${{SHAP_COLORS[i % 5]}};padding-left:10px">
        <div class="kv-label" style="font-size:12px;font-weight:600;color:#1C1C1E">#${{i+1}} ${{f}}</div>
        <div style="font-size:12px;color:#3A3A3C;margin-top:2px">${{desc}}</div>
        <div style="font-size:11px;margin-top:2px"><span class="${{cls}}" style="font-weight:500">${{dir}}贡献</span> <span style="color:#8E8E93">|SHAP|=${{meanAbs[i].toFixed(4)}}</span></div>
      </div>`;
    }}).join('');
  }}
}}

/* ── ② 因子相关性热力图 (v5.3: 改用纯 SVG, 移除 chartjs-chart-matrix 依赖) ── */
function renderShapCorr() {{
  const d = DATA.shap;
  if (!d || !d.correlation || !d.correlation.factors.length) {{
    document.getElementById('shapCorrSvgWrap').innerHTML =
      '<div style="color:#8E8E93;padding:20px">相关性数据不足（至少需要 2 个月且有 SHAP 数据）</div>';
    return;
  }}
  const factors = d.correlation.factors;
  const matrix  = d.correlation.matrix;
  const n = factors.length;
  // ★ v5.3: 改用 SVG 自渲染热力图, 彻底脱离 chartjs-chart-matrix
  const wrap = document.getElementById('shapCorrSvgWrap');
  wrap.innerHTML = '';
  const W = wrap.clientWidth || 600;
  const labelW = 110, labelH = 96;
  const cell = Math.max(40, Math.min(110, Math.floor((W - labelW - 24) / n)));
  const H = labelH + n * cell + 16;
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('width', String(W));
  svg.setAttribute('height', String(H));
  svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
  svg.style.fontFamily = '-apple-system, system-ui, sans-serif';
  svg.style.fontSize = '11px';
  function cellColor(v) {{
    const a = Math.min(0.92, Math.abs(v) * 0.92);
    return v >= 0
      ? `rgba(10, 132, 255, ${{a.toFixed(3)}})`
      : `rgba(255, 69, 58, ${{a.toFixed(3)}})`;
  }}
  function textColor(v) {{ return Math.abs(v) > 0.55 ? '#fff' : '#1C1C1E'; }}
  // 列标签 (顶部, 旋转 -30°)
  factors.forEach((f, j) => {{
    const x = labelW + j * cell + cell / 2;
    const t = document.createElementNS(svgNS, 'text');
    t.setAttribute('x', String(x));
    t.setAttribute('y', String(labelH - 6));
    t.setAttribute('text-anchor', 'start');
    t.setAttribute('transform', `rotate(-30 ${{x}} ${{labelH - 6}})`);
    t.setAttribute('fill', '#1C1C1E');
    t.setAttribute('font-weight', '600');
    t.textContent = f.length > 14 ? f.slice(0, 13) + '…' : f;
    svg.appendChild(t);
  }});
  // 行 + 单元格
  factors.forEach((row, i) => {{
    const t = document.createElementNS(svgNS, 'text');
    t.setAttribute('x', String(labelW - 8));
    t.setAttribute('y', String(labelH + i * cell + cell / 2 + 4));
    t.setAttribute('text-anchor', 'end');
    t.setAttribute('fill', '#1C1C1E');
    t.setAttribute('font-weight', '600');
    t.textContent = row.length > 14 ? row.slice(0, 13) + '…' : row;
    svg.appendChild(t);
    factors.forEach((col, j) => {{
      const v = matrix[i][j];
      const x = labelW + j * cell;
      const y = labelH + i * cell;
      const rect = document.createElementNS(svgNS, 'rect');
      rect.setAttribute('x', String(x));
      rect.setAttribute('y', String(y));
      rect.setAttribute('width', String(cell));
      rect.setAttribute('height', String(cell));
      rect.setAttribute('rx', '6'); rect.setAttribute('ry', '6');
      rect.setAttribute('fill', cellColor(v));
      rect.setAttribute('stroke', 'rgba(255,255,255,0.85)');
      rect.setAttribute('stroke-width', '1.5');
      rect.addEventListener('mouseenter', (e) => {{
        const tip = document.getElementById('shapTop5Tip');
        const cls = v >= 0 ? 'tt-pos' : 'tt-neg';
        let strength = '无相关';
        const av = Math.abs(v);
        if      (av >= 0.8) strength = '极强';
        else if (av >= 0.6) strength = '强';
        else if (av >= 0.4) strength = '中等';
        else if (av >= 0.2) strength = '弱';
        else if (av >= 0.05) strength = '极弱';
        tip.innerHTML = `
          <div class="tt-month">${{row}} ↔ ${{col}}</div>
          <div class="tt-row"><span>相关系数</span><span class="${{cls}}">${{v >= 0 ? '+' : ''}}${{v.toFixed(4)}}</span></div>
          <div class="tt-row"><span>相关强度</span><span>${{strength}}</span></div>
          <div class="tt-row"><span>方向</span><span class="${{cls}}">${{v >= 0 ? '同向' : '反向'}}</span></div>
        `;
        const wRect = wrap.getBoundingClientRect();
        tip.style.left = (e.clientX - wRect.left) + 'px';
        tip.style.top  = (e.clientY - wRect.top) + 'px';
        tip.classList.add('show');
      }});
      rect.addEventListener('mousemove', (e) => {{
        const tip = document.getElementById('shapTop5Tip');
        const wRect = wrap.getBoundingClientRect();
        tip.style.left = (e.clientX - wRect.left) + 'px';
        tip.style.top  = (e.clientY - wRect.top) + 'px';
      }});
      rect.addEventListener('mouseleave', () => {{
        document.getElementById('shapTop5Tip').classList.remove('show');
      }});
      svg.appendChild(rect);
      // 单元格数值
      const txt = document.createElementNS(svgNS, 'text');
      txt.setAttribute('x', String(x + cell / 2));
      txt.setAttribute('y', String(y + cell / 2 + 4));
      txt.setAttribute('text-anchor', 'middle');
      txt.setAttribute('fill', textColor(v));
      txt.setAttribute('font-weight', '600');
      txt.textContent = v.toFixed(2);
      svg.appendChild(txt);
    }});
  }});
  wrap.appendChild(svg);
}}

/* ── ③ 时间序列（5 个子图） ── */
function renderShapTs() {{
  const d = DATA.shap;
  if (!d || !d.timeseries || !d.timeseries.months.length) {{
    document.getElementById('shapTsGrid').innerHTML =
      '<div style="color:#8E8E93;padding:20px">时间序列数据不足</div>';
    return;
  }}
  // 清理旧的 chart
  shapTsCharts.forEach(c => c && c.destroy && c.destroy());
  shapTsCharts = [];

  const months  = d.timeseries.months;
  const factors = (d.global_top5.factors || []).slice(0, 5);
  const series  = d.timeseries.series || {{}};

  const grid = document.getElementById('shapTsGrid');
  grid.innerHTML = '';

  factors.forEach((f, idx) => {{
    const card = document.createElement('div');
    card.className = 'ts-card';
    const color = SHAP_COLORS[idx % 5];
    const vals  = series[f] || [];
    const avg   = vals.length
      ? (vals.reduce((s, v) => s + v, 0) / vals.length) : 0;
    card.innerHTML = `
      <div class="ts-card-title">
        <span class="ts-card-rank">${{idx + 1}}</span>
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
              title="${{f}}">${{f}}</span>
      </div>
      <div class="ts-card-sub">
        均值 ${{avg.toFixed(5)}} · 最大 ${{vals.length ? Math.max(...vals).toFixed(5) : '—'}} · 最小 ${{vals.length ? Math.min(...vals).toFixed(5) : '—'}}
      </div>
      <div class="ts-card-wrap">
        <canvas id="shapTsChart_${{idx}}"></canvas>
        <div class="chart-tooltip" id="shapTsTip_${{idx}}"></div>
      </div>
    `;
    grid.appendChild(card);

    const cv = document.getElementById(`shapTsChart_${{idx}}`);
    const ctx = cv.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, 170);
    grad.addColorStop(0, color + '70');
    grad.addColorStop(1, color + '06');

    const chart = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: months,
        datasets: [{{
          label: f,
          data: vals,
          borderColor: color,
          backgroundColor: grad,
          borderWidth: 2,
          fill: true, tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: color,
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2,
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        animation: {{ duration: 600 }},
        interaction: {{ mode: 'index', intersect: false }},
        onHover: (e, els) => {{
          const tip = document.getElementById(`shapTsTip_${{idx}}`);
          if (els.length === 0) {{
            tip.classList.remove('show'); return;
          }}
          const i = els[0].index;
          const wrap = card.querySelector('.ts-card-wrap');
          const rect = wrap.getBoundingClientRect();
          const cx = e.clientX - rect.left;
          const cy = e.clientY - rect.top;
          const v = vals[i];
          tip.innerHTML = `
            <div class="tt-month">${{months[i]}}</div>
            <div class="tt-row"><span>${{f}}</span><span>${{v.toFixed(6)}}</span></div>
            <div class="tt-row"><span>排名</span><span>#${{idx + 1}}</span></div>
          `;
          tip.style.left = cx + 'px';
          tip.style.top  = cy + 'px';
          tip.classList.add('show');
        }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ enabled: false }},
        }},
        scales: {{
          x: {{
            ticks: {{
              font: {{ size: 9 }}, color: '#8E8E93',
              maxRotation: 0, autoSkip: true, maxTicksLimit: 6,
            }},
            grid: {{ display: false }},
          }},
          y: {{
            ticks: {{
              font: {{ size: 9 }}, color: '#8E8E93',
              callback: v => v.toFixed(3),
            }},
            grid: {{ color: 'rgba(0,0,0,0.04)' }},
          }}
        }}
      }}
    }});
    shapTsCharts.push(chart);

    // 离开隐藏 tooltip
    card.querySelector('.ts-card-wrap').addEventListener(
      'mouseleave', () => {{
        document.getElementById(`shapTsTip_${{idx}}`)
          .classList.remove('show');
      }});
  }});
}}

/* ── ④ 月度详情（保留旧逻辑） ── */
function initShapMonthTabs() {{
  const d = DATA.shap || {{}};
  const months = Object.keys(d.monthly || {{}}).sort();
  const wrap   = document.getElementById('shapMonthTabs');
  if (!months.length) {{
    wrap.innerHTML =
      '<div style="color:#8E8E93;padding:6px 12px">无月度 SHAP 数据</div>';
    return;
  }}
  wrap.innerHTML = '';
  // ★ 改造: 展示从开始到最后的全部月份（不再限制为最近 12 个）
  const allMonths = months;
  allMonths.forEach((m, i) => {{
    const b = document.createElement('button');
    b.className = 'tab-btn' + (i === allMonths.length - 1 ? ' active' : '');
    b.textContent = m;
    b.onclick = () => {{
      wrap.querySelectorAll('.tab-btn')
        .forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      renderShapMonth(m);
    }};
    wrap.appendChild(b);
  }});
  // 滚动到激活的 tab 位置
  const active = wrap.querySelector('.tab-btn.active');
  if (active && active.scrollIntoView) {{
    try {{ active.scrollIntoView({{inline: 'center', block: 'nearest'}}); }} catch(_) {{}}
  }}
  renderShapMonth(allMonths[allMonths.length - 1]);
}}

function renderShapMonth(month) {{
  const d = ((DATA.shap || {{}}).monthly || {{}})[month] || [];
  if (!d.length) return;
  const labels = d.map(x => x.factor);
  const absV   = d.map(x => x.mean_abs);
  const sgV    = d.map(x => x.signed);
  if (shapChart) shapChart.destroy();

  const wrap = document.getElementById('tab-shap-monthly');
  const tip  = document.getElementById('shapMonthTip');

  shapChart = new Chart(
    document.getElementById('shapChart').getContext('2d'),
    {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [
          {{ label: '|SHAP| (平均绝对)', data: absV,
             backgroundColor: 'rgba(10,132,255,0.78)',
             borderColor: '#0A84FF', borderWidth: 1.2,
             borderRadius: 4 }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        indexAxis: 'y',
        animation: {{ duration: 500 }},
        onHover: (e, els) => {{
          if (els.length === 0) {{
            tip.classList.remove('show'); return;
          }}
          const i = els[0].index;
          const rect = wrap.getBoundingClientRect();
          const sg = sgV[i];
          const cls = sg >= 0 ? 'tt-pos' : 'tt-neg';
          tip.innerHTML = `
            <div class="tt-month">${{labels[i]}}</div>
            <div class="tt-row"><span>月份</span><span>${{month}}</span></div>
            <div class="tt-row"><span>|SHAP|</span><span>${{absV[i].toFixed(6)}}</span></div>
            <div class="tt-row"><span>带符号均值</span><span class="${{cls}}">${{sg >= 0 ? '+' : ''}}${{sg.toFixed(6)}}</span></div>
          `;
          tip.style.left = (e.clientX - rect.left) + 'px';
          tip.style.top  = (e.clientY - rect.top) + 'px';
          tip.classList.add('show');
        }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ enabled: false }},
          title: {{
            display: true,
            text: `月度详情：${{month}} · Top 5 因子`,
            color: '#3A3A3C', font: {{ size: 13, weight: '600' }},
            padding: {{ bottom: 8 }}
          }},
        }},
        scales: {{
          x: {{ grid: {{ color: 'rgba(0,0,0,0.04)' }},
               ticks: {{ color: '#8E8E93', font: {{ size: 11 }} }} }},
          y: {{ ticks: {{ font: {{ size: 10 }} }} }}
        }}
      }}
    }}
  );
  wrap.addEventListener('mouseleave',
    () => tip.classList.remove('show'));
}}

/* ──────────── 月度持仓 ──────────── */
let holdContribChart = null;
let HOLD_STATE = {{
  year: '',
  month: '',
  ym: '',  // 完整 YYYYMM
  monthsAll: [],
  yearsAll:  [],
  monthsByYear: {{}},
}};

function initHoldingsPicker() {{
  const selY = document.getElementById('holdYear');
  const selM = document.getElementById('holdMonth');
  const months = Object.keys(DATA.holdings || {{}}).sort();
  HOLD_STATE.monthsAll = months;
  if (months.length === 0) {{
    selY.innerHTML = '<option>暂无</option>';
    selM.innerHTML = '<option>暂无</option>';
    return;
  }}
  // 提取所有年份与每月分组
  // ★ Bug 修复: 改回 Map, 不再混用 .set/.get 与对象
  const ymByYear = new Map();
  months.forEach(m => {{
    const y = m.slice(0, 4);
    if (!ymByYear.has(y)) ymByYear.set(y, []);
    ymByYear.get(y).push(m);
  }});
  const years = Array.from(ymByYear.keys()).sort();
  HOLD_STATE.yearsAll     = years;
  HOLD_STATE.monthsByYear = ymByYear;

  selY.innerHTML = years.map(y =>
    `<option value="${{y}}">${{y}}</option>`).join('');
  // 默认选最后一年
  const lastY = years[years.length - 1];
  HOLD_STATE.year = lastY;
  selY.value = lastY;

  function refreshMonths() {{
    // 用 HOLD_STATE.year（用户当前选中的年份）刷新月份列表
    // ★ Bug 修复: monthsByYear 现在是 Map, 需要用 .get()
    const cur = (HOLD_STATE.monthsByYear && HOLD_STATE.monthsByYear.get)
      ? (HOLD_STATE.monthsByYear.get(HOLD_STATE.year) || [])
      : (HOLD_STATE.monthsByYear[HOLD_STATE.year] || []);
    selM.innerHTML = cur.map(m =>
      `<option value="${{m}}">${{m}}</option>`).join('');
    if (cur.length) {{
      HOLD_STATE.month = cur[cur.length - 1];
      selM.value = HOLD_STATE.month;
    }} else {{
      HOLD_STATE.month = '';
      selM.value = '';
    }}
  }}
  refreshMonths();

  selY.addEventListener('change', () => {{
    HOLD_STATE.year = selY.value;
    refreshMonths();
    renderHoldings(HOLD_STATE.month);
  }});
  selM.addEventListener('change', () => {{
    HOLD_STATE.month = selM.value;
    renderHoldings(HOLD_STATE.month);
  }});

  renderHoldings(HOLD_STATE.month);

  // ★ 导出 CSV 按钮事件
  const expBtn = document.getElementById('exportHoldingsCsv');
  if (expBtn) {{
    expBtn.addEventListener('click', exportHoldingsCsv);
  }}
}}

function renderHoldings(m) {{
  HOLD_STATE.ym = m;
  const arr = (DATA.holdings || {{}})[m] || [];
  if (!arr.length) {{
    document.getElementById('holdGrid').innerHTML =
      '<div style="color:#8E8E93">该月无持仓数据</div>';
    document.getElementById('holdMeta').textContent = m;
    renderHoldTable([]);
    return;
  }}
  // 仅展示 is_holding=true (10 只)
  const top10 = arr.filter(x => x.is_holding).slice(0, 10);
  const totalRet = top10.reduce((s, x) => s + x.contribution, 0);
  const w13 = top10.filter(x => x.tier === 'High').length;
  const w7  = top10.filter(x => x.tier === 'Low').length;
  document.getElementById('holdMeta').innerHTML =
    `<span class="pill">${{m}}</span>
     <span class="pill green">13% × ${{w13}}</span>
     <span class="pill amber">7% × ${{w7}}</span>
     <span class="pill gray">组合收益: ${{fmtPct(totalRet)}}</span>`;
  const grid = document.getElementById('holdGrid');
  grid.innerHTML = top10.map((s, i) => {{
    const isPos = s.monthly_return >= 0;
    const tierLabel = s.tier === 'High' ? '13%'
                    : s.tier === 'Low'  ? '7%'
                    : (s.tier || '—');
    return `
      <div class="hold-card">
        <div class="hold-rank ${{s.tier}}">${{i+1}}</div>
        <div class="hold-info">
          <div class="hold-name">${{s.stock_name || s.stock_code}}</div>
          <div class="hold-meta">${{s.stock_code}} · ${{s.industry || '—'}} · ${{tierLabel}} (${{(s.weight*100).toFixed(1)}}%)</div>
        </div>
        <div style="text-align:right">
          <div class="hold-w ${{isPos?'pos':'neg'}}">${{fmtPct(s.monthly_return)}}</div>
          <div class="hold-ret" style="color:var(--ink3)">贡献 ${{fmtPct(s.contribution)}}</div>
        </div>
      </div>
    `;
  }}).join('');

  // 贡献柱图
  if (holdContribChart) holdContribChart.destroy();
  holdContribChart = new Chart(
    document.getElementById('holdContribChart').getContext('2d'),
    {{
      type: 'bar',
      data: {{
        labels: top10.map(s => s.stock_name || s.stock_code),
        datasets: [
          {{ label: '个股贡献', data: top10.map(s => s.contribution),
             backgroundColor: top10.map(s =>
               s.contribution >= 0
                 ? 'rgba(48,209,88,0.7)'
                 : 'rgba(255,69,58,0.7)') }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ font: {{ size: 10 }} }} }},
          y: {{ ticks: {{ callback: v => (v*100).toFixed(1)+'%' }} }}
        }}
      }}
    }}
  );

  // ★ 渲染明细表
  renderHoldTable(top10);
}}

function renderHoldTable(rows) {{
  const tbody = document.querySelector('#holdTable tbody');
  if (!tbody) return;
  if (!rows.length) {{
    tbody.innerHTML = '<tr><td colspan="10" style="color:#8E8E93">无数据</td></tr>';
    return;
  }}
  const showM3 = has_m3_data && active_track === 'M2+M3';
  tbody.innerHTML = rows.map((s, i) => {{
    const isPos = s.monthly_return >= 0;
    const tierLabel = s.tier === 'High' ? '13%'
                    : s.tier === 'Low'  ? '7%'
                    : (s.tier || '—');
    const tierPill  = s.tier === 'High' ? 'pill green'
                    : s.tier === 'Low'  ? 'pill amber'
                    : 'pill gray';
    let m3Cols = '';
    if (showM3) {{
      const soldFlag = s.is_sold_by_m3 ? '<span class="pill red">是</span>' : '<span class="pill gray">否</span>';
      const sellReason = s.m3_sell_reason || '—';
      m3Cols = `<td>${{soldFlag}}</td><td style="font-size:11px">${{sellReason}}</td>`;
    }}
    return `
      <tr>
        <td>${{i+1}}</td>
        <td>${{s.stock_code || ''}}</td>
        <td>${{s.stock_name || ''}}</td>
        <td>${{s.industry || '—'}}</td>
        <td><span class="${{tierPill}}">${{tierLabel}}</span></td>
        <td class="num">${{(s.weight*100).toFixed(2)}}%</td>
        <td class="num ${{isPos?'pos':'neg'}}">${{fmtPct(s.monthly_return)}}</td>
        <td class="num ${{s.contribution >= 0 ? 'pos' : 'neg'}}">${{fmtPct(s.contribution)}}</td>
        ${{m3Cols}}
      </tr>
    `;
  }}).join('');
}}

/* ★ 导出 CSV：所有月份的 Top10 持仓（含月收益率与 13%/7% 组别） */
function exportHoldingsCsv() {{
  const all = DATA.holdings || {{}};
  const months = Object.keys(all).sort();
  if (months.length === 0) {{
    alert('暂无持仓数据可导出');
    return;
  }}
  // CSV 表头
  const headers = [
    '月份', '排名', '股票代码', '股票名称', '行业',
    '组别', '权重(%)', '月收益率(%)', '贡献(%)',
  ];
  const showM3 = has_m3_data && active_track === 'M2+M3';
  if (showM3) {{
    headers.push('是否被M3卖出', 'M3卖出原因');
  }}
  const lines = [headers.join(',')];
  months.forEach(m => {{
    const arr = (all[m] || []).filter(x => x.is_holding).slice(0, 10);
    arr.forEach((s, i) => {{
      const tierLabel = s.tier === 'High' ? '13%'
                      : s.tier === 'Low'  ? '7%'
                      : (s.tier || '');
      const row = [
        m,
        i + 1,
        s.stock_code || '',
        '"' + (s.stock_name || '').replace(/"/g, '""') + '"',
        '"' + (s.industry || '').replace(/"/g, '""') + '"',
        tierLabel,
        (s.weight * 100).toFixed(4),
        (s.monthly_return * 100).toFixed(4),
        (s.contribution   * 100).toFixed(4),
      ];
      if (showM3) {{
        row.push(s.is_sold_by_m3 ? '是' : '否');
        row.push('"' + (s.m3_sell_reason || '').replace(/"/g, '""') + '"');
      }}
      lines.push(row.join(','));
    }});
  }});
  // 添加 BOM 防止 Excel 中文乱码。
  // 修复：用 \u005Cn 而不是 \u005Cn ，避免 Python f-string 把 换行符 当作换行符写入
  //   实际 JS 字符串里需要的是 2 字符序列 反斜杠n，而不是真正的换行
  const csv = '\\ufeff' + lines.join('\\n');
  const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8' }});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `月度持仓明细_${{
    months[0]}}_${{months[months.length-1]}}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}}
function switchTrack(track) {{
  active_track = track;
  DATA = track === 'M2+M3' ? DATA_M3 : DATA_M2;
  document.querySelectorAll('.track-btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.track === track);
  }});
  const showM3Cols = track === 'M2+M3';
  document.querySelectorAll('.m3-col').forEach(el => {{
    el.style.display = showM3Cols ? '' : 'none';
  }});
  if (navChart) {{
    navChart.destroy(); navChart = null;
    initNavCurve();
  }}
  if (ddChart) {{
    ddChart.destroy(); ddChart = null;
    initDrawdownCurve();
  }}
  renderPerfRows(track === 'M2+M3' ? DATA_M3 : DATA_M2);
  if (typeof renderBrinson === 'function') renderBrinson();
  if (typeof renderFiveFactor === 'function') renderFiveFactor();
  if (typeof renderBarra === 'function') renderBarra();
  if (DATA.shap_degraded) {{
    renderShapDegraded();
  }} else {{
    if (typeof renderShapTop5 === 'function') renderShapTop5();
  }}
  if (typeof renderHoldings === 'function' && HOLD_STATE.ym) {{
    renderHoldings(HOLD_STATE.ym);
  }}
}}

function renderShapDegraded() {{
  const top5El = document.getElementById('tab-shap-top5');
  if (top5El) {{
    const canvas = top5El.querySelector('canvas');
    if (canvas) canvas.style.display = 'none';
    let degradedDiv = top5El.querySelector('.shap-degraded-notice');
    if (!degradedDiv) {{
      degradedDiv = document.createElement('div');
      degradedDiv.className = 'shap-degraded-notice';
      degradedDiv.style.cssText = 'padding:40px;text-align:center;color:#8E8E93;background:rgba(142,142,147,0.08);border-radius:12px;margin-top:12px';
      degradedDiv.innerHTML = 'M3 轨道 SHAP 归因暂不可用（降级模式）<br><span style="font-size:11px">M3 轨道基于 M2 持仓+择时信号调整，SHAP 因子归因需原始模型输出</span>';
      top5El.appendChild(degradedDiv);
    }}
  }}
}}

function renderPerfRows(data) {{
  const cards = document.querySelectorAll('.card');
  let perfCard = null;
  for (const c of cards) {{
    const h2 = c.querySelector('h2');
    if (h2 && h2.textContent.includes('核心绩效指标')) {{ perfCard = c; break; }}
  }}
  if (!perfCard) return;
  const grid = perfCard.querySelector('.kv-grid');
  if (!grid) return;
  if (!data.perf_rows) return;
  const rows = data.perf_rows;
  let html = '';
  for (let i = 0; i < rows.length; i++) {{
    const [label, value] = rows[i];
    html += '<div class="kv"><div class="kv-label">' + label + '</div><div class="kv-value">' + value + '</div></div>';
  }}
  grid.innerHTML = html;
}}

function initTrackSwitcher() {{
  const switcher = document.getElementById('trackSwitcher');
  if (!switcher) return;
  if (has_m3_data) {{
    switcher.style.display = 'flex';
  }}
  const banner = document.getElementById('alignmentBanner');
  if (banner && alignment_banner) {{
    banner.textContent = alignment_banner;
    banner.style.display = 'block';
  }}
  if (has_m3_data && DATA.shap_degraded) {{
    renderShapDegraded();
  }}
}}

function initHoldingsTab() {{
  // 占位：如果未来需要 tab 切换展示
}}
</script>

<!-- ★ v5.4: DATA + 初始化 (放最后, 让函数定义先执行) -->
<script>
{js_data}
</script>
</body>
</html>"""


def generate_report(
    all_portfolios: pd.DataFrame,
    stats: Optional[Dict] = None,
    output_path: str = "output/backtest_report.html",
    suffix: str = "",
    title: str = "TTHH量化回测报告",
    factor_df: Optional[pd.DataFrame] = None,
    shap_data: Optional[Dict] = None,
    m3_portfolios: Optional[pd.DataFrame] = None,
) -> Dict:
    if suffix:
        p = Path(output_path)
        output_path = str(p.with_stem(p.stem + suffix))
    rg = ReportGenerator()
    return rg.generate(
        all_portfolios,
        output_path=output_path,
        stats=stats,
        title=title,
        factor_df=factor_df,
        shap_data=shap_data,
        m3_portfolios=m3_portfolios,
    )
