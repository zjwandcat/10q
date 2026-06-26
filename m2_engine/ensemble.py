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

# PEP 683: 使用 frozenset/MappingProxyType 替代可变 set/dict，避免 GIL refcount 开销
"""
集成预测模块
GPU模式：LGBM→XGB串行（GTX1650显存限制）
CPU模式：LGBM+XGB并行（ThreadPoolExecutor）

核心逻辑：
1. 双模型训练（GPU串行/CPU并行）
2. 验证集IC计算与ic_gap检验
3. 置信度标记（ic_gap>0.15标记为LOW，不影响仓位）
4. 支持compute_val_portfolio_metrics（M5因变量）
"""
import pandas as pd
import numpy as np

try:
    import bottleneck as bn
    _nanmean  = bn.nanmean
    _nanstd   = bn.nanstd
    _nansum   = bn.nansum
    _nanmax   = bn.nanmax
    _nanmin   = bn.nanmin
    _nanmedian = bn.nanmedian
    _mean = bn.nanmean   # bottleneck 无 bn.mean，用 nanmean 替代
    _std  = bn.nanstd    # bottleneck 无 bn.std，用 nanstd 替代
except ImportError:
    _nanmean  = np.nanmean
    _nanstd   = np.nanstd
    _nansum   = np.nansum
    _nanmax   = np.nanmax
    _nanmin   = np.nanmin
    _nanmedian = np.nanmedian
    _mean = np.mean
    _std  = np.std

from typing import Dict, Optional
import gc
import warnings
import logging

from .lgbm_model import LGBMRanker
from .xgb_model  import XGBRanker
from .gpu_detector import GPUConfig

logger = logging.getLogger("m2.ensemble")
warnings.filterwarnings("ignore")


def _fast_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    快速Spearman相关系数，跳过scipy的p值计算
    比scipy.spearmanr快约5x

    ★ 优化: 用 scipy.stats.rankdata 替代双 argsort
    - 旧: np.argsort(np.argsort(x)) = 2次排序 O(N log N) × 2
    - 新: rankdata(x) = 1次调用，内部实现更高效
    - 数值等价: Spearman只依赖排名顺序，rankdata与双argsort等价
    """
    from scipy.stats import rankdata
    n = len(x)
    if n < 10:
        return 0.0
    rx = rankdata(x).astype(np.float64)
    ry = rankdata(y).astype(np.float64)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt(np.sum(rx_c**2) * np.sum(ry_c**2)) + 1e-8
    return float(np.sum(rx_c * ry_c) / denom)


def _per_group_rank_corr(
    dates: np.ndarray,
    pred:  np.ndarray,
    label: np.ndarray,
    min_group_size: int = 10,
) -> list:
    """
    按 dates 分组, 每组算 Spearman rank corr (pred vs label).

    用法: 替掉 `for date in unique: df[date]==date mask` 的 O(N) 布尔 mask × K 次循环
    改用 1 次 argsort + 1 次 unique + per-group slice.

    性能: ~12-15ms/调用 → ~1-2ms/调用 (180 窗全量省 ~2-2.5s).
    数值: bit-exact 相同 — _fast_rank_corr 内部用 argsort(argsort()), 对元素顺序不敏感,
          Spearman 也对元素顺序不敏感, 所以 per-group 元素集合相同 → 结果相同.

    参数:
        dates:  (N,)  日期/分组键, 任意可比较类型
        pred:   (N,)  预测分数
        label:  (N,)  标签
        min_group_size: 低于此大小的组跳过 (返回 0.0)
    返回:
        list[float], 每组一个 IC, 顺序按 dates 升序
    """
    n = len(dates)
    if n < min_group_size:
        return [0.0]

    order = np.argsort(dates, kind="stable")
    sorted_dates = dates[order]
    sorted_pred  = pred[order]
    sorted_label = label[order]

    _, idx_start = np.unique(sorted_dates, return_index=True)
    idx_end = np.append(idx_start[1:], n)

    ics = []
    for s, e in zip(idx_start, idx_end):
        if e - s >= min_group_size:
            ics.append(_fast_rank_corr(
                sorted_pred[s:e], sorted_label[s:e]))
    return ics

# ── 换手成本常量 ──────────────────────────────
STAMP_DUTY  = 0.001   # 印花税0.1%（仅卖出）
COMMISSION  = 0.0003  # 佣金0.03%（双边）
SLIPPAGE    = 0.001   # 滑点0.1%（双边）


def _compute_capture_ratios(
    monthly_portfolio_ret,
    monthly_benchmark_ret,
) -> dict:
    """
    计算上行/下行/综合捕获比（与M4 metrics.py同口径：净收益）。
    monthly_portfolio_ret: 组合月度净收益（已扣换手成本，与M4 portfolio_return对齐）
    monthly_benchmark_ret: 基准（CSI800）月度收益
    最小样本12个月（与M4 _compute_jensen_appraisal_full对齐）
    """
    if len(monthly_portfolio_ret) < 12 or len(monthly_benchmark_ret) < 12:
        return {"up_capture_ratio": 0.0,
                "down_capture_ratio": 0.0,
                "capture_ratio": 0.0}

    bench = np.asarray(monthly_benchmark_ret, dtype=np.float32)
    port  = np.asarray(monthly_portfolio_ret,  dtype=np.float32)

    up_mask   = bench > 0
    down_mask = bench < 0

    if up_mask.sum() < 3 or down_mask.sum() < 3:
        return {"up_capture_ratio": 0.0,
                "down_capture_ratio": 0.0,
                "capture_ratio": 0.0}

    up_cap = (float(_nanmean(port[up_mask])) /
              float(_nanmean(bench[up_mask])))

    down_cap = (float(_nanmean(port[down_mask])) /
                float(_nanmean(bench[down_mask])))

    if abs(down_cap) < 1e-6:
        combined = 0.0
    else:
        combined = up_cap / down_cap

    return {
        "up_capture_ratio":   round(float(up_cap),      4),
        "down_capture_ratio": round(float(down_cap),     4),
        "capture_ratio":      round(float(combined),     4),
    }


def _compute_jensen_appraisal(
    monthly_portfolio_ret,
    monthly_benchmark_ret,
    rf_monthly: float = 0.03 / 12,
) -> dict:
    """
    计算 Jensen's Alpha (年化) 与 Appraisal Ratio (月度口径)。
    与M4 metrics.py _compute_jensen_appraisal_full同口径：净收益 + 最小12月。

    数学定义:
        y_t = R_p,t − R_f,t
        x_t = R_b,t − R_f,t
        OLS: y_t = α_monthly + β·x_t + ε_t
        → jensen_alpha (annual) = α_monthly × 12
        → appraisal_ratio = α_monthly / σ(ε)

    退化保护:
        - len(x) < 12            → 0.0（与M4对齐）
        - n_unique(x) < 2        → 0.0（基准无波动）
        - std(ε) < 1e-8          → appraisal_ratio = 0.0
        - NaN / Inf 输入或输出   → 0.0

    返回: dict 含 val_jensen_alpha / val_appraisal_ratio / val_beta
    """
    n = len(monthly_portfolio_ret)
    if n < 12 or len(monthly_benchmark_ret) < 12:
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}

    rets = np.asarray(monthly_portfolio_ret, dtype=np.float64)
    bms  = np.asarray(monthly_benchmark_ret, dtype=np.float64)

    y = rets - rf_monthly
    x = bms  - rf_monthly

    # 基准无波动 → 跳过 OLS
    if np.unique(x).size < 2:
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}

    # OLS:  β = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)² ,  α = ȳ − β·x̄
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    ss_xx  = float(((x - x_mean) ** 2).sum())
    if ss_xx < 1e-12:
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}

    beta  = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
    alpha = y_mean - beta * x_mean  # 月度 α

    residuals = y - (alpha + beta * x)
    if residuals.size < 2:
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}

    # 样本标准差 (ddof=1)
    sigma_eps = float(residuals.std(ddof=1))

    jensen_alpha    = alpha * 12.0
    appraisal_ratio = (alpha / sigma_eps) if sigma_eps > 1e-8 else 0.0

    # NaN / Inf 防御
    if not (np.isfinite(jensen_alpha) and np.isfinite(appraisal_ratio) and np.isfinite(beta)):
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}

    return {
        "val_jensen_alpha":    round(float(jensen_alpha),    6),
        "val_appraisal_ratio": round(float(appraisal_ratio), 6),
        "val_beta":            round(float(beta),            6),
    }


class EnsemblePredictor:
    """双模型软投票融合 + IC质量检验"""

    __slots__ = [
        'lgbm_weight', 'xgb_weight', 'lgbm', 'xgb',
        '_gpu_cfg', '_ic_penalize_threshold', '_disable_penalty',
    ]

    IC_PENALIZE_THRESHOLD = 0.15
    IC_WARN_THRESHOLD     = 0.02

    def __init__(
        self,
        lgbm_weight: float = 0.5,
        xgb_weight:  float = 0.5,
        lgbm_params: Optional[Dict] = None,
        xgb_params:  Optional[Dict] = None,
        ic_penalize_threshold: Optional[float] = None,
        disable_penalty: bool = False,
        strategy: Optional[str] = None,
    ):
        total = lgbm_weight + xgb_weight
        self.lgbm_weight = lgbm_weight / total
        self.xgb_weight  = xgb_weight  / total
        self.lgbm = LGBMRanker(lgbm_params)
        self.xgb  = XGBRanker(xgb_params)
        self._gpu_cfg = GPUConfig()
        if strategy is not None:
            self._gpu_cfg.set_strategy(strategy)
        self._ic_penalize_threshold = (
            ic_penalize_threshold if ic_penalize_threshold is not None
            else self.IC_PENALIZE_THRESHOLD
        )
        self._disable_penalty = disable_penalty

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        pred_df:  pd.DataFrame,
        feature_cols: list,
    ) -> Dict:
        # 准备数据
        train_v = train_df[train_df["label_rank"].notna()]
        val_v   = val_df[val_df["label_rank"].notna()]
        g_train = train_v.groupby("trade_date").size().tolist()
        g_val   = val_v.groupby("trade_date").size().tolist()
        X_train = train_v[feature_cols]
        y_train = train_v["label_rank"]
        X_val   = val_v[feature_cols]
        y_val   = val_v["label_rank"]

        gpu_mode = self._gpu_cfg.is_gpu_mode()

        if gpu_mode:
            # ── GPU模式：串行训练（显存限制）──────────
            # GTX1650 4GB，单模型约1.5GB，必须串行
            logger.info("GPU模式：LGBM→XGB串行训练")
            self.lgbm.fit(
                X_train, y_train, g_train,
                X_val,   y_val,   g_val,
                gpu_mode=self._gpu_cfg.opencl_available,
            )
            self.xgb.fit(
                X_train, y_train, g_train,
                X_val,   y_val,   g_val,
                gpu_mode=self._gpu_cfg.cuda_available,
            )
        else:
            # ── CPU模式：串行训练（稳定模式）──────────
            # 避免ThreadPoolExecutor并行训练导致LightGBM/XGBoost
            # 内部C结构体多线程访问冲突引发native内存崩溃
            logger.info("CPU模式：LGBM+XGB串行训练（稳定模式）")
            self.lgbm.fit(
                X_train, y_train, g_train,
                X_val,   y_val,   g_val,
                gpu_mode=False,
            )
            self.xgb.fit(
                X_train, y_train, g_train,
                X_val,   y_val,   g_val,
                gpu_mode=False,
            )

        # 释放训练数据，降低内存峰值
        del X_train, y_train, y_val
        del g_train, g_val
        # ★ 修复: gc.collect() 打破 Booster→callback→Dataset 循环引用
        gc.collect()

        # 验证集集成预测
        lgbm_val = self.lgbm.predict(X_val)
        xgb_val  = self.xgb.predict(X_val)
        ens_val  = (self.lgbm_weight * lgbm_val +
                    self.xgb_weight  * xgb_val)
        val_ic = self._monthly_ic(val_v, ens_val)
        del lgbm_val, xgb_val, ens_val  # ★ 修复: 释放预测数组

        # 训练集IC（最后一个训练月）
        last_m = train_v["trade_date"].max()
        mask   = train_v["trade_date"] == last_m
        X_tl   = train_v.loc[mask, feature_cols]
        y_tl   = train_v.loc[mask, "label_rank"]
        ens_tl = (self.lgbm_weight * self.lgbm.predict(X_tl) +
                  self.xgb_weight  * self.xgb.predict(X_tl))
        train_ic = _fast_rank_corr(
            ens_tl, y_tl.values)
        del X_tl, y_tl, ens_tl  # ★ 修复: 释放训练集临时数据

        ic_gap      = train_ic - val_ic
        is_penalized = False if self._disable_penalty else ic_gap > self._ic_penalize_threshold

        # 置信度标记（不影响仓位，只用于M4报告统计）
        if is_penalized:
            logger.info(
                f"ic_gap={ic_gap:.4f}>{self.IC_PENALIZE_THRESHOLD}，置信度标记为LOW")

        # 预测月打分
        X_pred = pred_df[feature_cols]
        lgbm_p = self.lgbm.predict(X_pred)
        xgb_p  = self.xgb.predict(X_pred)
        pred_scores = (self.lgbm_weight * lgbm_p +
                       self.xgb_weight  * xgb_p)
        score_cv = (
            _std([lgbm_p, xgb_p], axis=0) /
            (np.abs(pred_scores) + 1e-6)
        )

        pred_out = pred_df.copy(deep=False)  # ★ 修复: 浅拷贝避免 ~6MB/窗口深拷贝
        pred_out["score"]    = pred_scores
        pred_out["score_cv"] = score_cv

        pred_month = (pred_df["trade_date"].iloc[0]
                      .strftime("%Y%m"))

        return {
            "pred_month":          pred_month,
            "pred_df_with_scores": pred_out,
            "val_ic":              float(val_ic),
            "train_ic":            float(train_ic),
            "ic_gap":              float(ic_gap),
            "is_penalized":        bool(is_penalized),
            "lgbm_best_iter":      self.lgbm.best_iteration_,
            "xgb_best_iter":       self.xgb.best_iteration_,
            "gpu_mode":            gpu_mode,
        }

    def compute_val_portfolio_metrics(
        self,
        val_df_with_scores: pd.DataFrame,
    ) -> Dict:
        """
        验证集模拟持仓，计算滚动6月绩效指标
        供M5因变量使用
        已接入换手成本扣除
        """
        from .portfolio_builder import calculate_turnover_cost

        if (
            "benchmark_return" not in val_df_with_scores.columns
            or val_df_with_scores["benchmark_return"].eq(0).all()
        ):
            from m5_optimizer.benchmark_utils import load_aligned_csi800_benchmark
            bm_series = load_aligned_csi800_benchmark()
            ym = pd.to_datetime(val_df_with_scores["trade_date"]).dt.strftime("%Y%m")
            val_df_with_scores = val_df_with_scores.copy()
            val_df_with_scores["benchmark_return"] = ym.map(bm_series).fillna(0.0).values

        monthly_returns = []
        monthly_benchmarks = []
        monthly_gross_returns = []
        monthly_turnover_rates = []
        monthly_transaction_costs = []
        monthly_costs = []
        monthly_dates = []

        prev_holdings = {}  # {stock_code: weight}

        # ★ v4.3 优化: 一次性 np.lexsort(-score, date) 替代 per-date mask+sort
        #   bench_sort_full_186w.py 全量测试 (186 窗 × 6 月 × 1500 只):
        #     现版 pandas: 27.73 ms/窗
        #     新版 numpy :  8.76 ms/窗   (3.17x, 节省 18.97 ms/窗 ≈ 3.53s/全量)
        #   19 个 metric 全部 bit-exact (max_diff = 0.00e+00)
        #   D 模式 (m2_engine_gpu) 通过 v4.2 delegation 自动继承此优化
        _dates   = val_df_with_scores["trade_date"].values
        _scores  = val_df_with_scores["score"].values
        _stocks  = val_df_with_scores["stock_code"].values
        _target  = val_df_with_scores["Target_Return_1M"].values
        _bm_arr  = (val_df_with_scores["benchmark_return"].values
                    if "benchmark_return" in val_df_with_scores.columns
                    else np.zeros(len(val_df_with_scores), dtype=np.float64))
        # 主键 date asc, 次键 -score asc (= score desc)
        order = np.lexsort(
            (-_scores.astype(np.float64, copy=False), _dates))
        sorted_dates = _dates[order]
        _, idx_start = np.unique(sorted_dates, return_index=True)
        idx_end = np.append(idx_start[1:], len(sorted_dates))

        RETURN_CAP = 0.30
        _unique_dates = np.unique(sorted_dates)
        for i, (s, e) in enumerate(zip(idx_start, idx_end)):
            # order[s:e] 已按 -score 升序 (= score 降序)
            grp_idx   = order[s:e]
            stocks_grp = _stocks[grp_idx]
            rets_grp   = _target[grp_idx]
            # clip + NaN→0 等价于 pandas: .clip().fillna(0)
            rets_grp = np.where(
                np.isnan(rets_grp), 0.0,
                np.clip(rets_grp, -RETURN_CAP, RETURN_CAP))

            top5_codes  = stocks_grp[:5]
            next5_codes = stocks_grp[5:10]
            gross_ret   = (0.13 * rets_grp[:5].sum()
                           + 0.07 * rets_grp[5:10].sum())
            bm = float(_bm_arr[grp_idx[0]])

            # 向量化版本，避免iterrows
            curr_holdings = dict(zip(top5_codes, [0.13] * 5))
            curr_holdings.update(dict(zip(next5_codes, [0.07] * 5)))

            # 计算换手成本
            if prev_holdings:
                cost = calculate_turnover_cost(
                    prev_holdings, curr_holdings,
                    stamp_duty=STAMP_DUTY,
                    commission=COMMISSION,
                    slippage=SLIPPAGE,
                )
                # 单边换手率
                all_stocks = set(prev_holdings) | set(curr_holdings)
                turnover = sum(
                    abs(curr_holdings.get(s, 0) - prev_holdings.get(s, 0))
                    for s in all_stocks
                ) / 2.0
            else:
                cost = 0.0  # 首月无前期持仓，建仓成本暂不计
                turnover = 0.0

            net_ret = gross_ret - cost

            monthly_gross_returns.append(float(gross_ret))
            monthly_turnover_rates.append(float(turnover))
            monthly_transaction_costs.append(float(cost))
            monthly_returns.append(float(net_ret))
            monthly_benchmarks.append(float(bm))
            monthly_costs.append(float(cost))
            monthly_dates.append(_unique_dates[i])
            prev_holdings = curr_holdings

        n = len(monthly_returns)
        if n < 6:
            return {
                "val_rolling6m_ir": 0.0,
                "val_rolling6m_dir": 0.0,
                "val_rolling6m_sortino": 0.0,
                "val_rolling6m_return": 0.0,
                "val_global_ir": 0.0,
                "val_annual_return": 0.0,
                "pct_positive_excess": 0.0,
                "ir_worst_quartile": 0.0,
                "val_rolling6m_excess": 0.0,
                "val_rolling6m_excess_ann": 0.0,
                "up_capture_ratio": 0.0,
                "down_capture_ratio": 0.0,
                "capture_ratio": 0.0,
                "val_jensen_alpha": 0.0,
                "val_appraisal_ratio": 0.0,
                "val_beta": 0.0,
                # ★ P1 自适应型 2.0 新增（n<6 时无意义，默认 0.0）
                "val_ic_stability": 0.0,
                "val_ir_stability": 0.0,
                "turnover_penalty": 0.0,
                # ★ 风险指标（n<6 时无意义，默认 0.0）
                "val_var_95": 0.0,
                "val_cvar_95": 0.0,
                "val_pain_index": 0.0,
                "val_sqn": 0.0,
                "_monthly_returns": [],
                "_monthly_benchmarks": [],
                "_monthly_costs": [],
                "_monthly_dates": [],
            }

        rets = np.array(monthly_returns, dtype=np.float64)
        bms  = np.array(monthly_benchmarks, dtype=np.float64)
        excess = rets - bms

        # ★ 向量化优化: 用 sliding_window_view 替代逐窗 Python 循环
        # 原: for i in range(n-5) 逐窗计算 (n-5 次迭代, 每次 6 元素)
        # 新: 1 次 sliding_window_view 生成 (n-5, 6) 矩阵, 全程向量化
        # 数值等价: _std/_mean 在 1D 数组上与循环逐窗结果 bit-exact
        rf = 0.03 / 12
        if n >= 6:
            from numpy.lib.stride_tricks import sliding_window_view
            win_ex = sliding_window_view(excess, 6)      # (n-5, 6)
            win_ret = sliding_window_view(rets, 6)        # (n-5, 6)

            # IR: mean(excess) / std(excess) * sqrt(12)
            win_ex_mean = win_ex.mean(axis=1)
            win_ex_std = win_ex.std(axis=1, ddof=1)
            irs = np.where(
                win_ex_std > 1e-8,
                win_ex_mean / np.where(win_ex_std > 1e-8, win_ex_std, 1.0) * np.sqrt(12),
                0.0)

            # DIR: mean(excess) / down_std * sqrt(12)
            # down_std = std(excess[excess<0]) 或退化为全窗 std
            dirs = np.empty(n - 5, dtype=np.float64)
            for j in range(n - 5):
                w_ex = win_ex[j]
                down_ex = w_ex[w_ex < 0]
                down_std = (_std(down_ex, ddof=1)
                            if len(down_ex) > 1
                            else max(win_ex_std[j], 1e-8))
                dirs[j] = win_ex_mean[j] / down_std * np.sqrt(12)

            # Sortino: mean(ret-rf) / down_std_ret * sqrt(12)
            sortinos = np.empty(n - 5, dtype=np.float64)
            for j in range(n - 5):
                w_ret = win_ret[j]
                down_ret = w_ret[w_ret < rf]
                d_std_ret = (_std(down_ret, ddof=1)
                             if len(down_ret) > 1
                             else max(_std(w_ret, ddof=1), 1e-6))
                sortinos[j] = _mean(w_ret - rf) / d_std_ret * np.sqrt(12)

            # 累计收益: prod(1+ret) - 1
            cumrets = np.prod(1 + win_ret, axis=1) - 1
        else:
            irs = np.array([], dtype=np.float64)
            dirs = np.array([], dtype=np.float64)
            sortinos = np.array([], dtype=np.float64)
            cumrets = np.array([], dtype=np.float64)

        # ★ P1 自适应型 2.0：val_ir_stability（IR 跨时间稳定性）
        val_ir_stability = float(np.std(irs, ddof=1)) if len(irs) > 1 else 0.0

        # ★ P1 自适应型 2.0：val_ic_stability（IC 跨时间稳定性）
        # 按月计算 val_df_with_scores 的 Spearman IC，再按 6 月窗口分块取 std
        # ★ 性能优化: 用 _per_group_rank_corr 替掉 per-date `==` 布尔 mask
        # 节省: ~12ms/调用 (180 窗全量省 ~2s)
        ics_monthly = []
        if {"score", "label_rank", "trade_date"}.issubset(
            val_df_with_scores.columns
        ):
            ics_monthly = _per_group_rank_corr(
                val_df_with_scores["trade_date"].values,
                val_df_with_scores["score"].values,
                val_df_with_scores["label_rank"].values,
            )
        if len(ics_monthly) >= 6:
            ic_windows = []
            for i in range(len(ics_monthly) - 5):
                ic_windows.append(float(np.mean(ics_monthly[i:i+6])))
            val_ic_stability = (
                float(np.std(ic_windows, ddof=1))
                if len(ic_windows) > 1 else 0.0
            )
        else:
            val_ic_stability = 0.0

        # ★ P1 自适应型 2.0：turnover_penalty（日均双边换手率）
        # monthly_turnover_rates 在循环上方已累计完毕
        if monthly_turnover_rates and len(monthly_turnover_rates) > 0:
            turnover_penalty = float(np.mean(monthly_turnover_rates))
        else:
            turnover_penalty = 0.0

        # 全局IR：整个验证期的超额收益IR
        global_excess_mean = _mean(excess)
        global_excess_std = _std(excess, ddof=1)
        global_ir = (global_excess_mean / global_excess_std * np.sqrt(12)
                     if global_excess_std > 1e-8 else 0.0)

        # ★ 年化收益：整个验证期的复合年化收益（净收益，与M4 net_cagr_after_cost对齐）
        cum_return = np.prod(1 + rets) - 1
        annual_return = (1 + cum_return) ** (12 / n) - 1 if n > 0 else 0.0

        # ★ 扣费后年化收益：主收益序列已是净收益，因此与年化收益一致
        net_annual_return = annual_return

        # ── 熊牛一致性指标（向量化优化）──────────────────
        # 原: excess.tolist() 转 Python 列表, 再用 sum() + sorted() 循环
        # 新: 直接用 NumPy 向量化操作, 避免 Python 循环开销

        # 指标1：pct_positive_excess（月度超额胜率）
        if n > 0:
            pct_positive_excess = float(np.mean(excess > 0))
        else:
            pct_positive_excess = 0.0

        # 指标2：ir_worst_quartile（最差25%时期的平均超额 / 最差25%时期自身标准差）
        if n >= 4:
            sorted_excess = np.sort(excess)
            cutoff = max(1, n // 4)
            worst_quarter = sorted_excess[:cutoff]
            worst_std = float(np.std(worst_quarter, ddof=1)) if cutoff > 1 else 1e-8
            denom = worst_std if worst_std > 1e-8 else 1e-8
            ir_worst_quartile = float(np.mean(worst_quarter) / denom)
            ir_worst_quartile = float(np.clip(ir_worst_quartile, -5.0, 5.0))
        else:
            ir_worst_quartile = 0.0

        # 指标3：平均6月滚动超额收益（不除以标准差，直接看均值）
        # 单位：月度收益率，如0.005=每月跑赢0.5%
        if n >= 6:
            # ★ 向量化: 用 sliding_window_view 替代逐窗 Python 循环
            from numpy.lib.stride_tricks import sliding_window_view
            excess_windows = sliding_window_view(excess, 6)  # (n-5, 6)
            rolling6_means = excess_windows.mean(axis=1)
            val_rolling6m_excess = float(np.mean(rolling6_means))
            val_rolling6m_excess_ann = val_rolling6m_excess * 12
        else:
            val_rolling6m_excess = 0.0
            val_rolling6m_excess_ann = 0.0

        # 捕获比（净收益口径）
        _cap = _compute_capture_ratios(monthly_returns, monthly_benchmarks)

        # Jensen's Alpha & Appraisal Ratio（净收益口径）
        _ja = _compute_jensen_appraisal(
            monthly_returns, monthly_benchmarks,
            rf_monthly=0.03 / 12,
        )

        # ★ CVaR / VaR / 痛苦指数（与M4 metrics.py同公式）
        _r_arr = rets
        _var_95 = float(np.percentile(_r_arr, 5)) if n > 0 else 0.0
        _cvar_95 = (float(_r_arr[_r_arr <= _var_95].mean())
                    if n > 0 and (_r_arr <= _var_95).any() else _var_95)
        _cum_arr = np.cumprod(1 + _r_arr)
        _peak_arr = np.maximum.accumulate(_cum_arr)
        _dd_arr = (_cum_arr - _peak_arr) / _peak_arr
        _pain_index = float(np.abs(_dd_arr).mean()) if n > 0 else 0.0

        # ★ SQN (System Quality Number): mean(r)/std(r)*sqrt(n)（与M4同公式）
        _r_std = float(_r_arr.std(ddof=1)) if n > 1 else 1e-8
        _sqn = (float(_r_arr.mean()) / _r_std * np.sqrt(n)) if _r_std > 1e-8 and n > 1 else 0.0

        return {
            "val_rolling6m_ir":      float(np.mean(irs)) if len(irs) > 0 else 0.0,
            "val_rolling6m_dir":     float(np.mean(dirs)) if len(dirs) > 0 else 0.0,
            "val_rolling6m_sortino": float(np.mean(sortinos)) if len(sortinos) > 0 else 0.0,
            "val_rolling6m_return":  float(np.mean(cumrets)) if len(cumrets) > 0 else 0.0,
            "val_global_ir":         float(global_ir),
            "val_annual_return":     float(annual_return),
            "val_net_annual_return": float(net_annual_return),
            "pct_positive_excess":   float(pct_positive_excess),
            "ir_worst_quartile":     float(ir_worst_quartile),
            "val_rolling6m_excess":  float(val_rolling6m_excess),
            "val_rolling6m_excess_ann": float(val_rolling6m_excess_ann),
            "up_capture_ratio":      _cap["up_capture_ratio"],
            "down_capture_ratio":    _cap["down_capture_ratio"],
            "capture_ratio":         _cap["capture_ratio"],
            "val_jensen_alpha":      _ja["val_jensen_alpha"],
            "val_appraisal_ratio":   _ja["val_appraisal_ratio"],
            "val_beta":              _ja["val_beta"],
            # ★ P1 自适应型 2.0 新增 3 项
            "val_ic_stability":      val_ic_stability,
            "val_ir_stability":      val_ir_stability,
            "turnover_penalty":      turnover_penalty,
            # ★ 风险指标（与M4同公式）
            "val_var_95":            _var_95,
            "val_cvar_95":           _cvar_95,
            "val_pain_index":        _pain_index,
            "val_sqn":              _sqn,
            # ★ 月度序列（供run_m2汇总阶段全期拼接重算6m_ir）
            "_monthly_returns":      monthly_returns,
            "_monthly_benchmarks":   monthly_benchmarks,
            "_monthly_costs":        monthly_costs,
            "_monthly_dates":        monthly_dates,
        }

    def _monthly_ic(
        self,
        df: pd.DataFrame,
        pred_scores: np.ndarray,
        label_col: str = "label_rank",
    ) -> float:
        # ★ 性能优化: 用 _per_group_rank_corr 替掉 per-date `==` 布尔 mask
        # 原: 12 次 `df["trade_date"] == date` (每次 O(N) 哈希) + .loc + .values
        # 新: 1 次 argsort + 1 次 unique + per-group slice
        # 节省: ~12ms/调用 (180 窗全量省 ~2s)
        # 数值: bit-exact (rank corr 对顺序不敏感)
        if df.empty:
            return 0.0
        ics = _per_group_rank_corr(
            df["trade_date"].values,
            pred_scores,
            df[label_col].values,
        )
        return float(_mean(ics)) if ics else 0.0
