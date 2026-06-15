# PEP 683: frozenset/MappingProxyType 替代可变 set/dict
"""
集成预测模块 · v4.1 (D 方案专用: CPU train + GPU predict)

v4.1 简化:
  - 删除 v3.x 的 A/B/C/E 混合策略
  - 仅保留 D 策略 (CPU train + GPU predict) + 纯 CPU fallback
  - 与 m2_engine 纯 CPU 路径对比: max_diff ≈ 1.5e-7 (FP32 直方图噪声级)

关键优化 (v4.1):
  - LGBM (CPU) 与 XGB (CPU train) 在 2 个 Python 线程并行训练
  - XGB 预测走 device="cuda" + inplace_predict (0 分配)
  - 总时间 = max(LGBM_time, XGB_time) 而不是 LGBM_time + XGB_time
  - 预期节省: 30-50% (当两个模型耗时相近时)
"""
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import bottleneck as bn
    _nanmean  = bn.nanmean
    _nanstd   = bn.nanstd
    _nansum   = bn.nansum
    _nanmax   = bn.nanmax
    _nanmin   = bn.nanmin
    _nanmedian = bn.nanmedian
    _mean = bn.nanmean
    _std  = bn.nanstd
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
import warnings
import logging

from .lgbm_model import LGBMRanker
from .xgb_model  import XGBRanker
from .gpu_detector import GPUConfig

logger = logging.getLogger("m2.ensemble.v2")
warnings.filterwarnings("ignore")


def _fast_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 10:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float32)
    ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt(np.sum(rx_c**2) * np.sum(ry_c**2)) + 1e-8
    return float(np.sum(rx_c * ry_c) / denom)


# 换手成本常量
STAMP_DUTY  = 0.001
COMMISSION  = 0.0003
SLIPPAGE    = 0.001


def _compute_capture_ratios(monthly_portfolio_ret, monthly_benchmark_ret) -> dict:
    if len(monthly_portfolio_ret) < 6 or len(monthly_benchmark_ret) < 6:
        return {"up_capture_ratio": 0.0, "down_capture_ratio": 0.0, "capture_ratio": 0.0}
    bench = np.asarray(monthly_benchmark_ret, dtype=np.float32)
    port  = np.asarray(monthly_portfolio_ret,  dtype=np.float32)
    up_mask = bench > 0
    down_mask = bench < 0
    if up_mask.sum() < 3 or down_mask.sum() < 3:
        return {"up_capture_ratio": 0.0, "down_capture_ratio": 0.0, "capture_ratio": 0.0}
    up_cap = (float(_nanmean(port[up_mask])) / float(_nanmean(bench[up_mask])))
    down_cap = (float(_nanmean(port[down_mask])) / float(_nanmean(bench[down_mask])))
    combined = (up_cap / down_cap) if abs(down_cap) > 1e-6 else 0.0
    return {
        "up_capture_ratio":   round(float(up_cap), 4),
        "down_capture_ratio": round(float(down_cap), 4),
        "capture_ratio":      round(float(combined), 4),
    }


def _compute_jensen_appraisal(monthly_portfolio_ret, monthly_benchmark_ret, rf_monthly: float = 0.03/12) -> dict:
    n = len(monthly_portfolio_ret)
    if n < 6 or len(monthly_benchmark_ret) < 6:
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}
    rets = np.asarray(monthly_portfolio_ret, dtype=np.float64)
    bms  = np.asarray(monthly_benchmark_ret,  dtype=np.float64)
    y = rets - rf_monthly
    x = bms  - rf_monthly
    if np.unique(x).size < 2:
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    ss_xx  = float(((x - x_mean) ** 2).sum())
    if ss_xx < 1e-12:
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}
    beta  = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
    alpha = y_mean - beta * x_mean
    residuals = y - (alpha + beta * x)
    if residuals.size < 2:
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}
    sigma_eps = float(residuals.std(ddof=1))
    jensen_alpha    = alpha * 12.0
    appraisal_ratio = (alpha / sigma_eps) if sigma_eps > 1e-8 else 0.0
    if not (np.isfinite(jensen_alpha) and np.isfinite(appraisal_ratio) and np.isfinite(beta)):
        return {"val_jensen_alpha": 0.0, "val_appraisal_ratio": 0.0, "val_beta": 0.0}
    return {
        "val_jensen_alpha":    round(float(jensen_alpha),    6),
        "val_appraisal_ratio": round(float(appraisal_ratio), 6),
        "val_beta":            round(float(beta),            6),
    }


class EnsemblePredictor:
    """双模型软投票融合 v4.1 (D 方案: CPU train + GPU predict)"""

    __slots__ = [
        'lgbm_weight', 'xgb_weight', 'lgbm', 'xgb',
        '_gpu_cfg', '_ic_penalize_threshold', '_disable_penalty',
        '_strategy', '_xgb_predict_gpu',
        '_xgb_gpu_predict_only',   # v4.1 D 方案
        '_last_lgbm_time', '_last_xgb_time',
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
        self._strategy = self._gpu_cfg.strategy
        # v4.1: XGB 预测设备 (D 方案: GPU)
        self._xgb_predict_gpu = (self._gpu_cfg.get_xgb_predict_device() == "cuda")
        # v4.1: D 方案 gpu_predict_only=True
        self._xgb_gpu_predict_only = (self._strategy == "D" and self._xgb_predict_gpu)
        self._ic_penalize_threshold = (
            ic_penalize_threshold if ic_penalize_threshold is not None
            else self.IC_PENALIZE_THRESHOLD
        )
        self._disable_penalty = disable_penalty
        # 跟踪两个模型各自的墙钟时间
        self._last_lgbm_time = 0.0
        self._last_xgb_time  = 0.0

    def fit_predict(self, train_df, val_df, pred_df, feature_cols) -> Dict:
        # ── v3.8 优化: 一次性预取所有 numpy 数组, 避免 LGBM/XGB 内部 8+ 次 .values
        train_v = train_df[train_df["label_rank"].notna()]
        val_v   = val_df[val_df["label_rank"].notna()]
        g_train = train_v.groupby("trade_date").size().tolist()
        g_val   = val_v.groupby("trade_date").size().tolist()
        # 一次性预取 numpy 数组 (后续 LGBM/XGB/predict 都用)
        X_train = train_v[feature_cols].values
        y_train = train_v["label_rank"].values
        X_val   = val_v[feature_cols].values
        y_val   = val_v["label_rank"].values
        X_pred  = pred_df[feature_cols].values

        strategy = self._strategy
        xgb_train_dev = "CPU"        # v4.1 D 方案: 训练 CPU
        xgb_pred_dev  = "CUDA" if self._xgb_predict_gpu else "CPU"
        logger.info(
            f"[D方案] LGBM(CPU) + XGB(train={xgb_train_dev}, "
            f"predict={xgb_pred_dev}) 并行")

        # ── 关键优化 v4.1: CPU+CPU 训练并行 (与 v2.1 GPU 并行同样的原理) ──
        # LGBM (C++ backend) 和 XGB (C++ backend) 都释放 GIL
        # Python 线程可以让它们同时跑: CPU 跑一个 + CPU 跑一个
        t_wall_start = time.time()

        def _fit_lgbm_timed(
                X_tr, y_tr, g_tr, X_vl, y_vl, g_vl):
            t0 = time.time()
            self.lgbm.fit(X_tr, y_tr, g_tr, X_vl, y_vl, g_vl)
            return time.time() - t0

        def _fit_xgb_timed(
                X_tr, y_tr, g_tr, X_vl, y_vl, g_vl,
                gpu_predict_only):
            t0 = time.time()
            self.xgb.fit(X_tr, y_tr, g_tr, X_vl, y_vl, g_vl,
                         gpu_predict_only=gpu_predict_only)
            return time.time() - t0

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_lgbm = ex.submit(
                _fit_lgbm_timed,
                X_train, y_train, g_train, X_val, y_val, g_val)
            fut_xgb  = ex.submit(
                _fit_xgb_timed,
                X_train, y_train, g_train, X_val, y_val, g_val,
                self._xgb_gpu_predict_only)
            self._last_lgbm_time = fut_lgbm.result()
            self._last_xgb_time  = fut_xgb.result()

        t_wall = time.time() - t_wall_start
        sum_t = self._last_lgbm_time + self._last_xgb_time
        speedup_pct = (1 - t_wall / sum_t) * 100 if sum_t > 0 else 0
        logger.info(
            f"  并行诊断: LGBM={self._last_lgbm_time:.2f}s "
            f"XGB={self._last_xgb_time:.2f}s "
            f"和={sum_t:.2f}s "
            f"墙钟={t_wall:.2f}s "
            f"并行收益={speedup_pct:.1f}%")

        # 训练完立即释放原始数据（外层统一释放，避免重复 del 报错）
        try:
            del X_train, y_train, g_train
        except (NameError, UnboundLocalError):
            pass

        del y_val, g_val
        # ★ 修复: gc.collect() 打破 Booster→callback→Dataset 循环引用
        gc.collect()

        # 验证集集成预测
        # ★ v4.2: val_p 用 CPU predict (bit-exact 17 指标与 m2_engine 路径)
        #   GPU predict 在 val_p 上有 ~1.5e-7 噪声, 会让 top10 选股变化
        #   pred_p (用于 M5 选股打分) 仍用 GPU predict (速度优先)
        lgbm_val = self.lgbm.predict(X_val)
        xgb_val  = self.xgb.predict_cpu(X_val)   # v4.2: CPU 强制, bit-exact
        ens_val  = (self.lgbm_weight * lgbm_val +
                    self.xgb_weight  * xgb_val)
        val_ic = self._monthly_ic(val_v, ens_val)

        # 训练集 IC（最后一个训练月）
        last_m = train_v["trade_date"].max()
        mask   = (train_v["trade_date"] == last_m).values
        # ★ v3.8 优化: 用 numpy 切片, 避免 DataFrame .loc[]
        # ★ 关键: 必须用 train_v[feature_cols].values 而不是 train_v.values
        #   否则 pandas 会对所有列做 _interleave (8.45s 浪费)
        X_tl = train_v[feature_cols].values[mask]
        y_tl = train_v["label_rank"].values[mask]
        ens_tl = (self.lgbm_weight * self.lgbm.predict(X_tl) +
                  self.xgb_weight  * self.xgb.predict(X_tl))
        train_ic = _fast_rank_corr(ens_tl, y_tl)
        del X_tl, y_tl, ens_tl

        ic_gap       = train_ic - val_ic
        is_penalized = False if self._disable_penalty else ic_gap > self._ic_penalize_threshold

        if is_penalized:
            logger.info(
                f"ic_gap={ic_gap:.4f}>{self.IC_PENALIZE_THRESHOLD}，"
                f"置信度标记为LOW")

        # 预测月打分
        # ★ v3.8 优化: X_pred 已在前面预取, 直接复用
        lgbm_p = self.lgbm.predict(X_pred)
        xgb_p  = self.xgb.predict(X_pred)
        pred_scores = (self.lgbm_weight * lgbm_p +
                       self.xgb_weight  * xgb_p)
        score_cv = (
            _std([lgbm_p, xgb_p], axis=0) /
            (np.abs(pred_scores) + 1e-6)
        )

        # ★ v3.8 优化: pred_out 只加 score 列, 复用 pred_df
        # ★ 修复: 改用 copy(deep=False) 避免 ~6MB/窗口的深拷贝
        pred_out = pred_df.copy(deep=False)
        pred_out["score"]    = pred_scores
        pred_out["score_cv"] = score_cv

        # ★ 立即释放大数组
        del X_pred, lgbm_p, xgb_p, pred_scores, score_cv
        del ens_val, lgbm_val, xgb_val
        # gc.collect()  ← 移除：每窗 3 次 GC 占用 15% 时间（v2 GPU 测试）

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
            "gpu_mode":            self._gpu_cfg.is_gpu_mode(),
            "strategy":            strategy,
        }

    def compute_val_portfolio_metrics(
        self,
        val_df_with_scores: pd.DataFrame,
    ) -> Dict:
        """验证集模拟持仓指标（v4.2 改为复用 m2_engine 纯 CPU 实现, v4.3 进一步加速）

        v4.2 修复:
          旧: m2_engine_gpu 自带向量化实现, 与 m2_engine 纯 CPU 版本在
              NaN 处理 (np.nan_to_num vs fillna)、sort 算法 (lexsort vs
              sort_values) 上有细微差异, 导致 17 金融指标在 D 模式与纯 CPU
              模式间出现大 diff (e.g. capture_ratio: -6.95 vs 0.69).
          新: 直接 import m2_engine.EnsemblePredictor.compute_val_portfolio_metrics
              方法, 1:1 bit-exact. 实测 D vs CPU 17 指标全 OK.

        v4.3 加速 (D 模式自动继承):
          m2_engine.compute_val_portfolio_metrics 内部 per-date mask+sort
          改造为一次性 np.lexsort(-score, date). 全量 186 窗测试:
            现版 pandas: 27.73 ms/窗
            新版 numpy :  8.76 ms/窗   (3.17x, 节省 ~3.5s/全量)
        """
        from m2_engine.ensemble import EnsemblePredictor as _CPU_EP
        # m2_engine 的 compute_val_portfolio_metrics 是个无 self 依赖的方法
        # (不需要 lgbm/xgb 等训练好的模型, 只看 val_df_with_scores 的 score/date/...)
        return _CPU_EP.compute_val_portfolio_metrics(self, val_df_with_scores)


    def _monthly_ic(self, df, pred_scores, label_col="label_rank") -> float:
        """计算按月 IC 序列的均值

        v3.8.6 优化: 用 groupby indices 预取, 避免 N 次 mask 比较
        旧: 12 次 mask = df["trade_date"] == date + 12 次 .values 切片 + df.loc[mask,...].values
            → 每窗 ~50ms, 30 窗 ~1.5s
        新: 1 次 groupby('trade_date').indices 拿到 dict[date]→idx_array
            后续所有切片都用 np 数组索引, 0 pandas 开销
        """
        # ★ v3.8 优化: 一次性 groupby 拿到 indices, 后续纯 numpy
        # groupby().indices 返回 dict{date: array(idx)}
        gb = df["trade_date"].values  # 直接用 numpy
        # 排序后 group: 用 np.unique + return_index 模拟 groupby indices
        sort_idx = np.argsort(gb, kind='stable')
        sorted_dates = gb[sort_idx]
        unique_d, idx_start = np.unique(sorted_dates, return_index=True)
        idx_end = np.append(idx_start[1:], len(gb))
        # pred_scores 和 label 也要按 sort_idx 重排 (这样 group 切片与 pred_scores 对齐)
        sorted_pred = pred_scores[sort_idx]
        if hasattr(df[label_col], "values"):
            sorted_label = df[label_col].values[sort_idx]
        else:
            sorted_label = np.asarray(df[label_col])[sort_idx]

        ics = []
        for s, e in zip(idx_start, idx_end):
            n = e - s
            if n >= 10:
                ics.append(_fast_rank_corr(
                    sorted_pred[s:e],
                    sorted_label[s:e]))
        return float(_mean(ics)) if ics else 0.0
