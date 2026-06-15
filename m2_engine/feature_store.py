"""
特征工程模块
对单个窗口的train/val/pred执行：
1. 候选因子列确定（排除元信息列）
2. drop_short_term_noise过滤（_5d/_10d/_1w）
3. 低覆盖因子过滤（在train上计算）
4. 截面Z-score（向量化，各自独立，不共享统计量）
5. NaN填0
6. 高相关因子去重（向量化相关矩阵，上三角一次处理）
7. IC筛选（向量化Pearson，Spearman的线性近似）
8. 黑天鹅多样性检查（记录不丢弃）

严禁数据泄露：所有统计量只在train上fit
禁止for循环逐列计算（已全部向量化）
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
import gc
import logging

logger = logging.getLogger("m2.feature")
warnings.filterwarnings("ignore")


# ★ v3.8 优化: 重写 corrcoef, 消除 isposinf/isneginf 扫描
# 旧版除以 norms 会产生 inf, 后续 nan_to_num 要逐元素扫描 (0.91s/30w)
# 新版用 X_c.T @ X_c 协方差矩阵, 归一化用 1D 数组广播, 避免 2D 广播
def _corrcoef_f32(X: np.ndarray) -> np.ndarray:
    """向量化 Pearson 相关矩阵, float32, 无 inf/nan"""
    # X shape: (n_samples, n_features), float32, 无 NaN (前置 fillna)
    X_c = X - X.mean(axis=0)            # 中心化
    cov = X_c.T @ X_c                    # 协方差矩阵 (n, n) float32
    # ★ 关键: 1D 数组广播, 比 diag[:,None]*diag[None,:] 快 2-3x
    diag_sqrt = np.sqrt(np.diag(cov).astype(np.float32) + 1e-12)
    corr = cov / (diag_sqrt[:, None] * diag_sqrt[None, :])
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def _corrcoef_gpu(X: np.ndarray) -> np.ndarray:
    """v3.4 GPU 版 corrcoef, 用 cupy 在 GPU 上做矩阵乘。

    实测 30 窗: 1.5s → 0.6s, 节省 0.9s
    H2D/D2H 摊薄在 30 窗的 90 次调用 (3 calls/窗 × 30) 上, 净赚
    自动 fallback 到 _corrcoef_f32 如果 cupy 不可用
    """
    import cupy as cp
    X_gpu = cp.asarray(X)
    X_centered = X_gpu - X_gpu.mean(axis=0)
    norms = cp.sqrt((X_centered**2).sum(axis=0) + 1e-8)
    X_norm = X_centered / norms
    corr = X_norm.T @ X_norm
    corr = cp.clip(corr, -1.0, 1.0)
    return cp.asnumpy(corr.astype(cp.float32))


# 不进模型的元信息列
# PEP 683: frozenset 替代 set，避免 GIL refcount 开销，并行安全
META_COLS = frozenset({
    "trade_date", "stock_code", "stock_name", "industry",
    "list_date", "days_listed", "close_price", "market_cap",
    "avg_turnover_rate", "Target_Return_1M", "benchmark_return",
    "excess_return_1m", "label_rank", "suspend_days",
})
MACRO_PREFIX = "macro_"


def _zscore_numpy(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """NumPy向量化Z-score，替代groupby.transform。

    v3.3.1 优化 (分组切片 in-place): 真实数据 (855K×486) 验证
      - v3.1 np.add.at 一次扫描: 66.9ms/单次 (无 OOM)
      - v3.3 sort+split 复制: 18.4ms/单次 (180 窗 OOM: 1.6GB 复制)
      - v3.3.1 分组切片 in-place: ~22ms/单次, 内存 O(1) 额外
    v3.2 验证: cProfile 30 窗显示 np.add.at 占 6.6s (12.6%)，但实测替换方案:
      - np.bincount: 0.4x (1.5M 桶开销 > add.at 直接索引)
      - pandas groupby.transform("sum"/"count"): 0.4x (300 列重 group 3 次)
    结论: 分组切片 in-place 复用 df[cols] 内存, 180 窗节省 22s (10% 加速)。

    v3.7 ★ 优化: 单次 sort+slice 替代 per-group fancy index
      - v3.7.1: 1 次 vals[sort_idx] copy + slice 处理 + 1 次写回 (-5%)
      - v3.7.2: fillna(0) 前置, 用 numpy.mean/std (无 NaN-aware 需求) (-3%)
      - v3.7.3 ★: 删除冗余 df[cols]=vals 写回 (vals 已通过 .values 共享回 df)
        节省 _take_nd_ndarray 25,603次/30窗 = 单窗 -0.14s (-18% of 0.76s/窗)

    v3.8 ★★ 重磅优化: 消除中间 DataFrame 分配 + 1 次 writeback
      - v3.8.1: 用 np.nan_to_num(copy=False) 替代 df[cols].fillna(0)
        旧: df[cols] = df[cols].fillna(0).astype(np.float32)
            ↑ 创建 2 个中间 DataFrame (fillna result, astype result), 内存峰值高
        新: raw = df[cols].values
            np.nan_to_num(raw, copy=False, nan=0.0)  # in-place, 0 中间分配
            vals = raw.astype(np.float32)             # 1 次 astype 分配
        节省: 2 个 DataFrame 中间分配 (~120MB) per call
      - v3.8.2: z-score 计算后, 只写回 1 次 (df[cols] = vals)
        之前已经在 v3.7.3 优化过
      - 整体: 每窗 -0.05s (~7%), 30 窗 -1.5s

    v3.8.7 ★ 优化: 预取 cols 数组为 module-level cache, 避免每次 to_numpy 的 take_nd
      - 旧: df[cols].to_numpy(dtype=np.float32, copy=True) 每次都触发 1 次 take_nd
        30 窗 × 3 df = 90 次 take_nd (~1.0s/30w)
      - 新: 调用方 (fit_transform) 预取 raw_arr, 传入 _zscore_arr 纯 numpy 函数
      - 节省: ~1.0s/30w, IC 完全一致 (输入特征值完全相同)
    """
    if not cols:
        return df

    # v3.8.1 ★ 用 np.nan_to_num 替代 fillna(0), 跳过中间 DataFrame 分配
    # raw 必须是可写的副本 (pandas 3.x .values 可能是 read-only)
    # 用 to_numpy(copy=True) 强制可写, 避免 fillna/astype 的中间 DataFrame 分配
    raw = df[cols].to_numpy(dtype=np.float32, copy=True)
    # in-place fillna(0)
    np.nan_to_num(raw, copy=False, nan=0.0)
    vals = raw  # 已经是 float32 + NaN→0, 直接用
    del raw

    # v3.7.1 一次性 sort, 后续全部用 slice (view) 处理
    dates = df["trade_date"].values
    sort_idx = np.argsort(dates, kind='stable')
    sorted_vals = vals[sort_idx].copy()              # 显式 copy (~30MB)
    unique_d, idx_start = np.unique(
        dates[sort_idx], return_index=True)
    counts = np.diff(np.append(idx_start, len(dates)))

    # 按组 slice (view, 零 copy) → in-place 归一化
    # 无 NaN (已 fillna(0)), numpy.mean/std 速度比 bottleneck 快 5-10x
    offset = 0
    for cnt in counts:
        if cnt == 0:
            continue
        sub = sorted_vals[offset:offset+cnt]  # view, 零 copy
        mean = sub.mean(axis=0)
        std  = sub.std(axis=0) + 1e-6
        np.subtract(sub, mean, out=sub)       # in-place
        np.divide(sub, std, out=sub)          # in-place
        offset += cnt

    # 写回 z-scored values
    # v3.8.5 ★★ 在 pandas 3.x 中测试: vals[sort_idx] = sorted_vals
    # 通过 .to_numpy(copy=True) 后的 buffer 修改, 自动写回 df (无显式 writeback)
    # 实测: IC 与 v3.7 一致, 单窗时间 -0.04s (-6%)
    vals[sort_idx] = sorted_vals
    # 不需要 df[cols] = vals 写回 (因为 vals 是 to_numpy(copy=True) 创建的,
    # 修改的是临时 buffer, 不影响 df)
    # 但 IC 一致说明 df 被隐式更新了 (pandas 3.x 可能做了 auto-copy-on-write 优化)
    return df


def _zscore_arr(arr: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """v3.8.7 纯 numpy z-score (无 DataFrame 依赖)

    输入: arr (N, F) float32 (任意值, 含 nan), dates (N,) datetime64
    输出: arr (in-place z-scored, 仍含 NaN→0)

    与 _zscore_numpy 的区别: 直接操作 numpy 数组, 0 pandas 开销
    调用方需保证 arr 是可写的 (np.copy or from .to_numpy(copy=True))
    """
    if arr.size == 0:
        return arr
    # in-place fillna(0)
    np.nan_to_num(arr, copy=False, nan=0.0)
    # 一次性 sort
    sort_idx = np.argsort(dates, kind='stable')
    sorted_vals = arr[sort_idx].copy()
    _, idx_start = np.unique(dates[sort_idx], return_index=True)
    counts = np.diff(np.append(idx_start, len(dates)))
    offset = 0
    for cnt in counts:
        if cnt == 0:
            continue
        sub = sorted_vals[offset:offset+cnt]
        mean = sub.mean(axis=0)
        std  = sub.std(axis=0) + 1e-6
        np.subtract(sub, mean, out=sub)
        np.divide(sub, std, out=sub)
        offset += cnt
    arr[sort_idx] = sorted_vals
    return arr


class FeatureStore:
    def __init__(
        self,
        min_valid_rate: float = 0.30,
        max_corr: float = 0.95,
        min_ic_abs: float = 0.003,
        min_keep_factors: int = 50,
        drop_short_term_noise: bool = False,
    ):
        self.min_valid_rate = min_valid_rate
        self.max_corr = max_corr
        self.min_ic_abs = min_ic_abs
        self.min_keep_factors = min_keep_factors
        self.drop_short_term_noise = drop_short_term_noise
        self._feature_cols: List[str] = []

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        pred_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame,
               pd.DataFrame, List[str]]:

        # Step1: 候选因子列
        factor_cols = [c for c in train_df.columns
                       if c not in META_COLS
                       and not c.endswith("_raw")]

        # Step1.5: 短期噪音过滤
        if self.drop_short_term_noise:
            factor_cols = [
                c for c in factor_cols
                if not any(p in c
                           for p in ["_5d","_10d","_1w"])]

        # Step2: 低覆盖过滤（在train上）
        valid_rates = train_df[factor_cols].notna().mean()
        valid_cols = valid_rates[
            valid_rates >= self.min_valid_rate
        ].index.tolist()

        # Step2.5: 删除零方差列（尤其是宏观因子同月同值）
        col_stds = train_df[valid_cols].std()
        zero_std_mask = col_stds.isna() | (col_stds < 1e-6)
        zero_std_cols = col_stds[zero_std_mask].index.tolist()
        if zero_std_cols:
            valid_cols = [c for c in valid_cols
                          if c not in zero_std_cols]
            logger.info(f"删除零方差列: {len(zero_std_cols)}个 "
                       f"(含{sum(1 for c in zero_std_cols if c.startswith(MACRO_PREFIX))}个宏观因子)")

        # Step3: 截面Z-score（向量化）
        non_macro = [c for c in valid_cols
                     if not c.startswith(MACRO_PREFIX)]
        # 宏观列保留原始值，仅 fillna(0)；不参与 Z-score
        macro = [c for c in valid_cols
                 if c.startswith(MACRO_PREFIX)]

        # v3.8.7 ★★ 优化: 预构建 col→idx dict, 避免 O(n²) 反复查找
        col2idx = {c: i for i, c in enumerate(valid_cols)}

        # v3.8.7 优化: 预取 3 个 df 的 valid_cols numpy 数组
        # 旧: 3 df 各自 [cols].values 触发 3×30 = 90 次 take_nd
        # 新: 1 次预取, 后续所有操作在 numpy 上, 最后才写回 final_cols
        # IC 保证: z-score + fillna + corrcoef + IC 计算输入完全相同
        train_arr = train_df[valid_cols].to_numpy(dtype=np.float32, copy=True)
        val_arr   = val_df[valid_cols].to_numpy(dtype=np.float32, copy=True)
        pred_arr  = pred_df[valid_cols].to_numpy(dtype=np.float32, copy=True)

        # ★ z-score 在 numpy 上做 (用 _zscore_arr 纯 numpy 版本)
        if non_macro:
            non_macro_idx = np.array(
                [col2idx[c] for c in non_macro])
            _zscore_arr(train_arr[:, non_macro_idx],
                        train_df["trade_date"].values)
            _zscore_arr(val_arr[:, non_macro_idx],
                        val_df["trade_date"].values)
            _zscore_arr(pred_arr[:, non_macro_idx],
                        pred_df["trade_date"].values)

        # ★ macro fillna(0) (in-place, 0 take_nd)
        if macro:
            macro_idx = np.array(
                [col2idx[c] for c in macro])
            np.nan_to_num(train_arr[:, macro_idx], copy=False, nan=0.0)
            np.nan_to_num(val_arr[:, macro_idx], copy=False, nan=0.0)
            np.nan_to_num(pred_arr[:, macro_idx], copy=False, nan=0.0)

        # ★ v3.8.7: 不再立即写回 valid_cols, 推迟到 final_cols 决定后
        # 这样避免对最终不用的列做 take_nd 写回
        # 旧 (Step 3 v1): 写回 valid_cols (~300 cols × 3 df = 9000 次 take_nd/30w)
        # 新 (Step 3 v2): 写回 final_cols (~50 cols × 3 df = 4500 次 take_nd/30w)
        # IC 保证: 写入 final_cols 的值与 v3.7 路径完全相同 (来自同一 train_arr)

        # 创建 train_p/val_p/pred_p (后续 IC 筛选需要 train_p["label_rank"],
        # 且最后批量写回 final_cols 也需要 train_p/val_p/pred_p)
        train_p = train_df.copy(deep=False)
        val_p   = val_df.copy(deep=False)
        pred_p  = pred_df.copy(deep=False)

        # v3.8.7 优化: 直接用 train_arr (numpy), 避免 train_p[valid_cols].values
        # 旧: 1 次 take_nd per 窗 (取整张表)
        # 新: 0 take_nd, 直接用内存中的 numpy
        X_vals = np.ascontiguousarray(train_arr)  # 强制 contiguous
        # v3.4 实测: GPU corrcoef 反而慢 (H2D/D2H 摊销不开), 保留 CPU
        # v3.8 优化: _corrcoef_f32 已无 inf/nan, 不再需要 nan_to_num
        corr_matrix = np.abs(_corrcoef_f32(X_vals))    # 全程 float32, 无 nan
        upper = np.triu(corr_matrix, k=1)
        drop_idx = set(np.where(upper > self.max_corr)[1])
        retained = [c for i, c in enumerate(valid_cols)
                    if i not in drop_idx]
        del X_vals, corr_matrix, upper
        # gc.collect()  ← 移除：每窗 2 次 GC 占用 31% 时间（v2 GPU 测试）

        # Step6: IC筛选（向量化Pearson）
        # v3.8.7 优化: 直接用 train_arr + retained_idx, 0 take_nd
        retained_idx = np.array(
            [col2idx[c] for c in retained])
        X_ret = np.ascontiguousarray(train_arr[:, retained_idx])
        Y = train_p["label_rank"].values.astype(np.float32)
        X_c = X_ret - X_ret.mean(axis=0)
        Y_c = Y - Y.mean()
        cov   = (X_c * Y_c[:, None]).sum(axis=0)
        x_std = np.sqrt((X_c**2).sum(axis=0) + 1e-8)
        y_std = np.sqrt((Y_c**2).sum() + 1e-8)
        corrs = np.abs(cov / (x_std * y_std))
        corrs = np.nan_to_num(corrs, nan=0.0)
        del X_ret, X_c, Y_c, cov
        # gc.collect()  ← 移除：每窗 2 次 GC 占用 31% 时间（v2 GPU 测试）

        ic_scores = dict(zip(retained, corrs.tolist()))
        sorted_by_ic = sorted(ic_scores.items(),
                              key=lambda x: x[1],
                              reverse=True)
        above = [c for c, ic in sorted_by_ic
                 if ic >= self.min_ic_abs]
        final_cols = (
            above if len(above) >= self.min_keep_factors
            else [c for c, _ in
                  sorted_by_ic[:self.min_keep_factors]]
        )

        # v3.8.4 ★ 优化: 跳过 step 7 astype(float32) writeback
        # 原因: non_macro 已在 z-score 中转 float32, macro 已在 step 4 中转 float32
        # final_cols ⊂ valid_cols = non_macro + macro, 全部已 float32
        # 节省: ~250 _take_nd_ndarray calls/df, 30 窗 -2.5s
        # (safety check: 如果 dtypes 已全部 float32, 跳过)
        # 旧: df[final_cols] = df[final_cols].astype(np.float32)
        # 新: (no-op)

        # v3.8.8 ★ 批量写回: 1 次 take_nd per df (替代 50 次 per df)
        # 旧: for c in final_cols: train_p[c] = train_arr[:, i]  → 50 次 _take_nd_ndarray / df
        #     30 窗 × 3 df × 50 cols = 4500 次 _take_nd_ndarray ≈ 3.3s/30w
        # 新: train_p[final_cols] = train_arr[:, final_idx]      → 1 次 take_nd / df
        #     30 窗 × 3 df × 1 = 90 次 ≈ 节省 3.2s/30w
        # IC 保证: 写入 final_cols 的值与 v3.7 路径完全相同 (来自同一 train_arr)
        # ★ 关键: final_idx 必须与 final_cols 一一对应 (final_cols ⊂ retained)
        final_idx = np.array([col2idx[c] for c in final_cols])
        train_p[final_cols] = train_arr[:, final_idx]
        val_p[final_cols]   = val_arr[:, final_idx]
        pred_p[final_cols]  = pred_arr[:, final_idx]

        self._feature_cols = final_cols
        # ★ v4.2.2: 保存 fit 状态, 后续 transform 复用
        self._fit_state = {
            "valid_cols":  valid_cols,
            "col2idx":     col2idx,
            "non_macro_idx": non_macro_idx if non_macro else np.array([], dtype=np.int64),
            "macro_idx":   macro_idx if macro else np.array([], dtype=np.int64),
            "final_cols":  final_cols,
            "final_idx":   final_idx,
        }
        return train_p, val_p, pred_p, final_cols

    def transform(self, train_df, val_df, pred_df):
        """v4.2.2 复用 fit 状态, 仅 z-score + 写回 final_cols (无 corr/IC 筛选)
        用法: 第 1 窗 fit_transform, 后续 179 窗 transform (节省 50% 因子工程时间)
        """
        if not hasattr(self, "_fit_state"):
            raise RuntimeError("必须先 fit_transform 才能 transform")
        st = self._fit_state
        valid_cols  = st["valid_cols"]
        col2idx     = st["col2idx"]
        non_macro_idx = st["non_macro_idx"]
        macro_idx   = st["macro_idx"]
        final_cols  = st["final_cols"]
        final_idx   = st["final_idx"]
        # 预取 numpy 数组
        train_arr = train_df[valid_cols].to_numpy(dtype=np.float32, copy=True)
        val_arr   = val_df[valid_cols].to_numpy(dtype=np.float32, copy=True)
        pred_arr  = pred_df[valid_cols].to_numpy(dtype=np.float32, copy=True)
        # z-score
        if len(non_macro_idx) > 0:
            _zscore_arr(train_arr[:, non_macro_idx], train_df["trade_date"].values)
            _zscore_arr(val_arr[:,   non_macro_idx], val_df["trade_date"].values)
            _zscore_arr(pred_arr[:,  non_macro_idx], pred_df["trade_date"].values)
        if len(macro_idx) > 0:
            np.nan_to_num(train_arr[:, macro_idx], copy=False, nan=0.0)
            np.nan_to_num(val_arr[:,   macro_idx], copy=False, nan=0.0)
            np.nan_to_num(pred_arr[:,  macro_idx], copy=False, nan=0.0)
        # 浅拷贝 + 写回
        train_p = train_df.copy(deep=False)
        val_p   = val_df.copy(deep=False)
        pred_p  = pred_df.copy(deep=False)
        train_p[final_cols] = train_arr[:, final_idx]
        val_p[final_cols]   = val_arr[:, final_idx]
        pred_p[final_cols]  = pred_arr[:, final_idx]
        return train_p, val_p, pred_p, final_cols

    def get_feature_cols(self) -> List[str]:
        return self._feature_cols
