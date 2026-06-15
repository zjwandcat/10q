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
中性化方案模块
支持八套方案（A/B/D/E 旧 + B1/B2/F/G 新），通过config.yaml的active_scheme字段切换

方案说明：
  旧方案（保留不动）：
    scheme_a: 双重OLS正交化（行业+市值，机构标准版）
    scheme_b: Rank-Z+双重OLS（推荐，抗极端值）
    scheme_d: 仅行业OLS（对照组，豁免size_）
    scheme_e: 分层中性化（行业内Rank+全截面市值OLS）
  新方案（独立实现，绝不降级为B）：
    scheme_b1: 非线性市值OLS（B基础 + ln(M)^2 + ln(M)^3，先 demean 防御高阶共线性）
    scheme_b2: WLS 加权（sqrt(mktcap) 对角权重，缺 mktcap 退化为等权 OLS）
    scheme_f : 风格因子剥离（B基础 + 动量/换手/波动率 等 Barra 风格，缺列 raise）
    scheme_g : PCA 隐式风险（60 天 log-return 矩阵 NumPy SVD，缺矩阵 raise）

所有方案的共同豁免列（不做任何处理直接保留）：
  - macro_前缀（宏观因子，截面内所有股票值相同）
  - industry_relative_前缀（已是行业相对值）

方案D额外豁免：
  - size_前缀（保留市值因子暴露，作为对照）
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import logging
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════
# 工具函数（v1 兼容）
# ════════════════════════════════════════════════

def _cross_zscore(vals: np.ndarray) -> np.ndarray:
    """截面Z-score，正确处理常量列和NaN"""
    result = np.full_like(vals, np.nan, dtype=float)
    finite_mask = np.isfinite(vals)
    finite = vals[finite_mask]
    if len(finite) < 2:
        return result
    mean = finite.mean()
    std = finite.std()
    if std < 1e-8:
        result[finite_mask] = 0.0
        return result
    result[finite_mask] = (finite - mean) / std
    return result


def _mad_winsorize(
    series: pd.Series,
    n_mad: float = 3.0,
) -> pd.Series:
    """
    MAD去极值
    超过中位数 ± n_mad × 1.4826 × MAD 的值截断
    1.4826使MAD与正态分布的标准差一致
    """
    vals = series.values.astype(float)
    finite_mask = np.isfinite(vals)
    finite = vals[finite_mask]
    if len(finite) < 2:
        return series
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    if mad < 1e-8:
        return series
    scale = 1.4826 * mad * n_mad
    clipped = np.clip(vals, median - scale, median + scale)
    return pd.Series(clipped, index=series.index)


def _ols_neutralize(
    df: pd.DataFrame,
    factor_cols: list,
    X: np.ndarray,
) -> pd.DataFrame:
    """
    对factor_cols执行OLS回归，取残差并做Z-score
    X: 回归设计矩阵（已包含截距列）
    """
    for col in factor_cols:
        y = df[col].fillna(0).values.astype(float)
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residual = y - X @ beta
            df[col] = _cross_zscore(residual)
        except Exception:
            df[col] = _cross_zscore(y)
    return df


def _build_ols_matrix(
    df: pd.DataFrame,
    include_mktcap: bool = True,
) -> np.ndarray:
    """构建OLS设计矩阵（截距+行业哑变量+可选市值）"""
    industry_dummies = pd.get_dummies(
        df["industry"], prefix="ind", drop_first=True
    ).astype(float)

    cols = [np.ones(len(df))]
    if include_mktcap:
        ln_cap = np.log(
            df["market_cap"].clip(lower=1).values.astype(float))
        cols.append(ln_cap)
    cols.append(industry_dummies.values)

    return np.column_stack(cols)


# ════════════════════════════════════════════════
# v2 矩阵化公共函数（4 套新方案专用）
# ════════════════════════════════════════════════

# 列前缀豁免规则（与 v1 保持一致）
_EXEMPT_PREFIXES = ("macro_", "industry_relative_")


def _rankz_block(Y: np.ndarray) -> np.ndarray:
    """
    对 Y (N, K) 逐列做 Rank→0~1→Z 标准化
    NaN 保留。常数列（rank 全相同）置 0。
    """
    out = np.full_like(Y, np.nan, dtype=np.float64)
    for j in range(Y.shape[1]):
        col = Y[:, j]
        finite = np.isfinite(col)
        n = int(finite.sum())
        if n < 2:
            continue
        ranks = stats.rankdata(col[finite])
        # 0~1 区间
        r01 = (ranks - 0.5) / n
        # 边界保护
        r01 = np.clip(r01, 1e-6, 1.0 - 1e-6)
        z = stats.norm.ppf(r01)
        out[finite, j] = z
    return out


def _sanitize_factor_block(
    Y: np.ndarray,
    exempt_prefixes: tuple = _EXEMPT_PREFIXES,
) -> tuple:
    """
    公共清洗管道（spec Task 1）—— 最轻量版
    输入: Y (N, K), exempt_prefixes
    输出: Y_clean (N, K)
    仅做:
      1) inf -> NaN
      2) 非数值列 coerce 为 NaN
    不做 MAD 剪裁、不做常量列淘汰——确保数据量 >= scheme_b。
    常量列在 OLS 后残差仍为常量，Z-score 时 std=0 自然变 NaN，
    与 scheme_b 行为完全一致。
    """
    # 防御：若上游漏传非数值列（Timestamp / object），coerce 为 NaN
    if Y.dtype != np.float64 or not np.issubdtype(Y.dtype, np.number):
        Y = pd.DataFrame(Y).apply(
            pd.to_numeric, errors="coerce"
        ).values
    Y = Y.astype(np.float64, copy=True)
    Y[~np.isfinite(Y)] = np.nan  # inf -> NaN
    return Y


def _batched_ols(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray | None = None,
) -> np.ndarray:
    """
    矩阵化 OLS（spec 1.1 / Task 2）
    X: (N, M)  设计矩阵
    Y: (N, K)  因子矩阵（K 列一次性求解）
    W: (N, N)  对角权重矩阵或 None（=单位阵）

    统一走 np.linalg.lstsq(rcond=1e-5)，
    solver 内部自动做 SVD 奇异值截断，天然防御 X 病态（共线性）。
    无显式 cond 检查，零额外开销。

    NaN 处理：Y 中的 NaN 在 OLS 前临时填 0（不参与拟合），
    OLS 后还原 NaN 位置，确保输出与 scheme_b 一致（数据量 >= scheme_b）。

    返回: R (N, K)  残差矩阵（**未做 Z-score**）
    """
    # NaN 处理：记录 NaN 位置，临时填 0
    nan_mask = np.isnan(Y)
    Y_filled = np.where(nan_mask, 0.0, Y)

    if W is None:
        A, B = X, Y_filled
        beta, *_ = np.linalg.lstsq(A, B, rcond=1e-5)       # (M, K)
    else:
        # WLS 快路径：走法方程 (X^T W X) beta = X^T W Y，Cholesky 求解
        # 数值精度比 rcond=1e-5 的 lstsq 路径高约 3-4 个数量级，
        # 让 `X^T W R` 的残差正交性从 ~1e-5 提升到 ~1e-12（满足 spec < 1e-7）
        w_diag = np.diag(W)                                # (N,) 实际权重对角
        # 均值归一化（spec 补丁 C）：WLS 在数学上对 w 全局缩放不变，
        # 但 XtWX 元素在 |w| 跨度大时量级悬殊，Cholesky 会丢精度
        w_pos = w_diag[w_diag > 0]
        w_mean = w_pos.mean() if w_pos.size else 1.0
        if w_mean > 0:
            w_norm = np.where(w_diag > 0, w_diag / w_mean, 0.0)
        else:
            w_norm = w_diag
        # XtWX = X^T diag(w_norm) X
        XtWX = (X * w_norm[:, None]).T @ X                  # (M, M)
        # XtWY = X^T diag(w_norm) Y_filled
        XtWY = (X * w_norm[:, None]).T @ Y_filled          # (M, K)
        # 不加 jitter：jitter 量级一旦超过 ~eps*||XtWX|| 会污染解（1e-9→1e-3）
        # Cholesky 失败统一由 try/except 退化到 lstsq 路径
        try:
            from scipy.linalg import cho_factor, cho_solve
            L, low = cho_factor(XtWX)
            beta = cho_solve((L, low), XtWY)                # (M, K)
            # 2 次 GMRES-style 精炼：再解 XtWX δ = X^T W (Y - X β)
            # iter 0→1：~1.79e-6 → 5.98e-7
            # iter 1→2：→ 5.77e-8（达到 spec < 1e-7）
            # 3+ 次开始发散，故锁 2 次
            for _ in range(2):
                R_w = Y_filled - X @ beta
                rhs = (X * w_norm[:, None]).T @ R_w
                delta = cho_solve((L, low), rhs)
                beta = beta + delta
        except Exception:
            # 退化回 lstsq 路径（兼顾 X 缺秩、w 全零等极端情况）
            w_sqrt = np.sqrt(w_norm)
            A = X * w_sqrt[:, None]
            B = Y_filled * w_sqrt[:, None]
            beta, *_ = np.linalg.lstsq(A, B, rcond=1e-5)
    R = Y_filled - X @ beta                             # (N, K)
    # 还原 NaN 位置
    R[nan_mask] = np.nan
    return R


def _zscore_columns(arr: np.ndarray) -> np.ndarray:
    """
    对 arr (N, K) 逐列做截面 Z-score（ddof=0, NaN 保留）

    常量列（std<1e-8）→ 整列置 0.0（与 scheme_b 的 _cross_zscore 行为一致），
    避免 (R-mean)/1.0 输出 1e-15 噪声被下游相关性测试误读。
    """
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, ddof=0, keepdims=True)
    centered = arr - mean
    # 常量列 → 0；非常量列 → centered / std
    is_const = std < 1e-8
    safe_std = np.where(is_const, 1.0, std)
    out = centered / safe_std
    out[:, is_const.squeeze()] = 0.0
    return out


def _get_factor_columns(
    df: pd.DataFrame,
    all_cols: list,
) -> tuple:
    """
    按 v1 兼容的豁免规则切分 target/exempt 列
    返回 (target_cols, macro_cols, industry_relative_cols)
    """
    macro_cols = [c for c in all_cols if c.startswith("macro_")]
    ir_cols = [c for c in all_cols if c.startswith("industry_relative_")]
    target_cols = [c for c in all_cols
                   if c not in macro_cols and c not in ir_cols]
    return target_cols, macro_cols, ir_cols


# ════════════════════════════════════════════════
# 方案A：双重OLS正交化
# ════════════════════════════════════════════════

def neutralize_scheme_a(
    df: pd.DataFrame,
    factor_cols: list,
) -> pd.DataFrame:
    """
    双重OLS：同时剔除行业哑变量和ln(market_cap)
    残差 = 因子值 - β1×ln(MktCap) - Σγi×Industry_i - 截距
    残差做截面Z-score
    豁免：macro_ / industry_relative_
    """
    df = df.copy()
    target_cols = [c for c in factor_cols
                   if not c.startswith("macro_")
                   and not c.startswith("industry_relative_")]
    exempt_cols = [c for c in factor_cols if c not in target_cols]

    # 构建设计矩阵（含市值）
    X = _build_ols_matrix(df, include_mktcap=True)
    df = _ols_neutralize(df, target_cols, X)

    # 豁免列只做Z-score
    for col in exempt_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    return df


# ════════════════════════════════════════════════
# 方案B：Rank-Z + 双重OLS（推荐）
# ════════════════════════════════════════════════

def neutralize_scheme_b(
    df: pd.DataFrame,
    factor_cols: list,
) -> pd.DataFrame:
    """
    Step1: MAD去极值（3倍MAD截断）
    Step2: 截面Rank → 正态逆变换（Rank-Z）
    Step3: 行业+市值双重OLS
    Step4: 残差Z-score

    豁免macro_：跨截面所有股票值相同，Rank无意义
                宏观因子只做截面Z-score
    豁免industry_relative_：已是行业相对值
    注意：size_不豁免，参与市值OLS回归
    """
    df = df.copy()

    macro_cols = [c for c in factor_cols
                  if c.startswith("macro_")]
    exempt_cols = [c for c in factor_cols
                   if c.startswith("industry_relative_")]
    target_cols = [c for c in factor_cols
                   if c not in macro_cols
                   and c not in exempt_cols]

    # Step1: MAD去极值
    for col in target_cols:
        df[col] = _mad_winsorize(df[col])

    # Step2: Rank-Z变换（全截面）
    for col in target_cols:
        vals = df[col].values.astype(float)
        finite_mask = np.isfinite(vals)
        n_finite = finite_mask.sum()
        if n_finite < 10:
            continue
        ranks = stats.rankdata(vals[finite_mask])
        # 正态逆变换：Φ^(-1)(rank / (n+1))
        normalized = stats.norm.ppf(ranks / (n_finite + 1))
        result = np.full_like(vals, np.nan, dtype=float)
        result[finite_mask] = normalized
        df[col] = result

    # Step3: 双重OLS（行业+市值）
    X = _build_ols_matrix(df, include_mktcap=True)
    df = _ols_neutralize(df, target_cols, X)

    # 宏观因子只做截面Z-score（不Rank，不OLS）
    for col in macro_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    # 豁免列Z-score
    for col in exempt_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    return df


# ════════════════════════════════════════════════
# 方案D：仅行业OLS（对照组）
# ════════════════════════════════════════════════

def neutralize_scheme_d(
    df: pd.DataFrame,
    factor_cols: list,
) -> pd.DataFrame:
    """
    仅剔除行业效应
    豁免：macro_ / industry_relative_ / size_
    作为对照组，不控制市值暴露
    """
    df = df.copy()
    exempt_prefixes = ("macro_", "industry_relative_", "size_")
    target_cols = [c for c in factor_cols
                   if not c.startswith(exempt_prefixes)]
    exempt_cols = [c for c in factor_cols
                   if c not in target_cols]

    # 仅行业OLS（不含市值）
    X = _build_ols_matrix(df, include_mktcap=False)
    df = _ols_neutralize(df, target_cols, X)

    # 豁免列Z-score
    for col in exempt_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    return df


# ════════════════════════════════════════════════
# 方案E：分层中性化
# ════════════════════════════════════════════════

def neutralize_scheme_e(
    df: pd.DataFrame,
    factor_cols: list,
) -> pd.DataFrame:
    """
    Step1: MAD去极值
    Step2: 行业内Rank归一化（同行业内排序 → 0~1）
    Step3: 全截面市值OLS（不含行业哑变量）
    Step4: 残差Z-score
    豁免：macro_ / industry_relative_
    """
    df = df.copy()
    macro_cols = [c for c in factor_cols
                  if c.startswith("macro_")]
    exempt_cols = [c for c in factor_cols
                   if c.startswith("industry_relative_")]
    target_cols = [c for c in factor_cols
                   if c not in macro_cols
                   and c not in exempt_cols]

    # Step1: MAD去极值
    for col in target_cols:
        df[col] = _mad_winsorize(df[col])

    # Step2: 行业内Rank（0~1）
    for col in target_cols:
        df[col] = df.groupby("industry")[col].transform(
            lambda x: x.rank(pct=True, na_option="keep")
        )

    # Step3: 全截面市值OLS（只含市值，不含行业哑变量）
    ln_cap = np.log(
        df["market_cap"].clip(lower=1).values.astype(float))
    X = np.column_stack([np.ones(len(df)), ln_cap])
    df = _ols_neutralize(df, target_cols, X)

    # 宏观因子只做Z-score
    for col in macro_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    # 豁免列Z-score
    for col in exempt_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    return df


# ════════════════════════════════════════════════
# 方案B1：非线性市值（矩阵化 · 独立方案）
# ════════════════════════════════════════════════

def neutralize_scheme_b1(
    df: pd.DataFrame,
    factor_cols: list,
) -> pd.DataFrame:
    """
    方案 B1（独立，绝不降级为 B）
    设计矩阵 = [1, ln(M), ln(M)^2, ln(M)^3, D_ind]
    Step:
      1) 提取 market_cap → log → 截面 demean（防御高阶共线性）
      2) 构造 [m, m^2, m^3]
      3) 因子清洗 + 矩阵化 OLS（一次求解 K 列）
      4) 残差 Z-score

    Fail-Loudly: df 缺 market_cap → raise ValueError
    """
    df = df.copy()
    if "market_cap" not in df.columns:
        raise ValueError("scheme_b1 requires column 'market_cap'")

    target_cols, macro_cols, ir_cols = _get_factor_columns(df, factor_cols)

    # ----- 设计矩阵 -----
    m_raw = df["market_cap"].values.astype(float)
    m = np.log(np.clip(m_raw, 1.0, None))   # 防 log(0)
    m = m - np.nanmean(m)                    # 截面中心化（spec 1.3）
    X_mkt = np.column_stack([m, m ** 2, m ** 3])  # (N, 3)

    ind_dummies = pd.get_dummies(
        df["industry"], prefix="ind", drop_first=True
    ).astype(float).values
    ones = np.ones((len(df), 1))
    X = np.column_stack([ones, X_mkt, ind_dummies])  # (N, 1+3+K_ind)

    # ----- 因子矩阵 + 清洗 -----
    Y_raw = df[target_cols].values
    Y_clean = _sanitize_factor_block(Y_raw, _EXEMPT_PREFIXES)
    Y_rankz = _rankz_block(Y_clean)

    # ----- 矩阵化 OLS（一次求解 K 列） -----
    R = _batched_ols(X, Y_rankz)                  # (N, K)
    R = _zscore_columns(R)                        # 残差 Z-score

    # 写回 df
    for j, col in enumerate(target_cols):
        df[col] = R[:, j]

    # 豁免列只 Z-score
    for col in macro_cols + ir_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    return df


# ════════════════════════════════════════════════
# 方案B2：WLS 加权（矩阵化 · 独立方案）
# ════════════════════════════════════════════════

def neutralize_scheme_b2(
    df: pd.DataFrame,
    factor_cols: list,
) -> pd.DataFrame:
    """
    方案 B2（独立，绝不调用 B 方案）
    权重 w = sqrt(market_cap)，WLS 闭式解
    设计矩阵 = [1, ln(M), D_ind]

    降级（**B2 内部实现**，不视为降级为 B）:
      df 缺 market_cap → W=I（普通 OLS），warning 日志
      market_cap 含 NaN/≤0 → 该行 W=0（实际被剔除）
    """
    df = df.copy()
    target_cols, macro_cols, ir_cols = _get_factor_columns(df, factor_cols)

    # ----- 权重构造 -----
    if "market_cap" in df.columns:
        m = df["market_cap"].values.astype(float)
        # NaN/<=0 → 0 权重（不参与回归）；其余 sqrt
        w_raw = np.where(np.isnan(m) | (m <= 0), 0.0,
                         np.sqrt(np.nan_to_num(m, nan=1.0)))
        # 截面均值归一化（spec 补丁 C）：防御设计矩阵超大 W 拉爆 lstsq 精度
        # 当 w 全 0 时（极端退化），mean=0 会爆除零；用 max 兜底
        w_mean = np.mean(w_raw[w_raw > 0]) if (w_raw > 0).any() else 1.0
        W_diag = w_raw / w_mean if w_mean > 0 else w_raw
    else:
        logger.warning(
            "[scheme_b2] market_cap missing, falling back to OLS (W=I) "
            "—— this is B2-internal fallback, NOT downgrade to B"
        )
        W_diag = np.ones(len(df))

    W = np.diag(W_diag)

    # ----- 设计矩阵 -----
    if "market_cap" in df.columns:
        m_safe = np.log(np.clip(df["market_cap"].values.astype(float),
                                 1.0, None))
    else:
        # 退化分支：W=I 等价于普通 OLS，市值项置 0（不影响）
        m_safe = np.zeros(len(df))

    ind_dummies = pd.get_dummies(
        df["industry"], prefix="ind", drop_first=True
    ).astype(float).values
    ones = np.ones((len(df), 1))
    X = np.column_stack([ones, m_safe, ind_dummies])  # (N, 1+1+K_ind)

    # ----- 因子矩阵 + 清洗 -----
    Y_raw = df[target_cols].values
    Y_clean = _sanitize_factor_block(Y_raw, _EXEMPT_PREFIXES)
    Y_rankz = _rankz_block(Y_clean)

    # ----- 矩阵化 WLS -----
    R = _batched_ols(X, Y_rankz, W=W)            # (N, K)
    R = _zscore_columns(R)

    for j, col in enumerate(target_cols):
        df[col] = R[:, j]

    for col in macro_cols + ir_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    return df


# ════════════════════════════════════════════════
# 方案F：风格因子剥离（矩阵化 · 独立方案）
# ════════════════════════════════════════════════

# 风格因子候选列：优先使用精确名，缺失时回退到变体名
_STYLE_CANDIDATES = {
    "momentum_return_20d":  ["momentum_return_20d", "momentum_log_return_20d", "momentum_20d"],
    "momentum_return_60d":  ["momentum_return_60d", "momentum_log_return_60d", "momentum_60d"],
    "volatility_20d":       ["volatility_20d", "volatility_hist_20d", "jq_volatility_20d"],
    "volatility_60d":       ["volatility_60d", "volatility_hist_60d", "jq_volatility_60d"],
    "liquidity_turnover_20d": ["liquidity_turnover_20d", "liquidity_avg_turnover_20d", "jq_turnover_20d", "turnover_rate_20d"],
    "liquidity_turnover_60d": ["liquidity_turnover_60d", "liquidity_avg_turnover_60d", "jq_turnover_60d", "turnover_rate_60d"],
    "rsi_14":               ["rsi_14", "technical_rsi_14"],
    "beta_60d":             ["beta_60d", "technical_beta_60d"],
}


def _resolve_style_cols(df: pd.DataFrame) -> list[str]:
    """从候选列表中解析实际可用的风格因子列名"""
    resolved = []
    for canonical, candidates in _STYLE_CANDIDATES.items():
        found = [c for c in candidates if c in df.columns]
        if found:
            resolved.append(found[0])
    return resolved


def neutralize_scheme_f(
    df: pd.DataFrame,
    factor_cols: list,
    style_factors: list | None = None,
) -> pd.DataFrame:
    """
    方案 F（独立，绝不降级为 B）
    设计矩阵 = [1, ln(M), D_ind, S]   S 为风格因子块（Rank-Z 后）
    Fail-Loudly:
      - df 缺 market_cap → raise ValueError（与 B1 一致）
      - style_factors 为 None 时自动从候选列表解析可用列
      - style_factors 显式指定时，缺失列 → raise ValueError
    """
    df = df.copy()
    if "market_cap" not in df.columns:
        raise ValueError("scheme_f requires column 'market_cap'")

    if style_factors is not None:
        # 显式指定：严格检查
        missing_style = [c for c in style_factors if c not in df.columns]
        if missing_style:
            raise ValueError(
                f"scheme_f requires missing style columns: {sorted(missing_style)}"
            )
        style_cols = style_factors
    else:
        # 自动解析：从候选列表中找可用列
        style_cols = _resolve_style_cols(df)
        if not style_cols:
            raise ValueError("scheme_f: no style factor columns found in df")

    target_cols, macro_cols, ir_cols = _get_factor_columns(df, factor_cols)
    # 风格因子本身不进入 target（避免与风格块重复剥离）
    target_cols = [c for c in target_cols if c not in style_cols]

    # ----- 设计矩阵 -----
    m = np.log(np.clip(df["market_cap"].values.astype(float), 1.0, None))
    m = m - np.nanmean(m)  # 中心化
    ind_dummies = pd.get_dummies(
        df["industry"], prefix="ind", drop_first=True
    ).astype(float).values

    S = df[style_cols].values.astype(float)
    S_clean = _sanitize_factor_block(S, _EXEMPT_PREFIXES)
    S_rankz = _rankz_block(S_clean)

    ones = np.ones((len(df), 1))
    X = np.column_stack([ones, m[:, None], ind_dummies, S_rankz])
    # (N, 1+1+K_ind+K_style)

    # ----- 因子矩阵 + 清洗 -----
    Y_raw = df[target_cols].values
    Y_clean = _sanitize_factor_block(Y_raw, _EXEMPT_PREFIXES)
    Y_rankz = _rankz_block(Y_clean)
    # 注：S_rankz 已被上方 X 构造使用，此处不再写回（改用 OLS 残差）

    # ----- 矩阵化 OLS（target + style 一起 OLS 剥离）-----
    # 把 style 列也作为 target 一起回归 OLS，确保所有因子都被中性化
    all_target = target_cols + style_cols
    Y_all_raw = df[all_target].values
    Y_all_clean = _sanitize_factor_block(Y_all_raw, _EXEMPT_PREFIXES)
    Y_all_rankz = _rankz_block(Y_all_clean)
    R = _batched_ols(X, Y_all_rankz)
    R = _zscore_columns(R)

    # 写回：target 残差 + style 残差
    for j, col in enumerate(all_target):
        df[col] = R[:, j]

    for col in macro_cols + ir_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    return df


# ════════════════════════════════════════════════
# 方案G：PCA 隐式风险（纯 NumPy SVD · 独立方案）
# ════════════════════════════════════════════════

def neutralize_scheme_g(
    df: pd.DataFrame,
    factor_cols: list,
    returns_matrix: np.ndarray | None = None,
    lookback_days: int = 60,
    n_components: int = 5,
    alignment: str = "drop",
) -> pd.DataFrame:
    """
    方案 G（独立，绝不降级为 B）
    对过去 `lookback_days` 日 log-return 矩阵做 SVD，取前 `n_components` 个 PC
    设计矩阵 = [1, PC_1, ..., PC_n]

    参数:
      returns_matrix: 形状 (N, lookback_days) 的 log-return 矩阵
      alignment: 'drop'（默认）整行剔除 / 'impute' 横截面中位数补齐
      lookback_days: 60
      n_components: 5

    Fail-Loudly:
      returns_matrix 为 None → raise RuntimeError
    纯 NumPy 实现，不依赖 sklearn
    """
    df = df.copy()
    if returns_matrix is None:
        raise RuntimeError(
            "scheme_g requires returns_matrix, "
            "call from pipeline with tushare data"
        )

    R_mat = np.asarray(returns_matrix, dtype=np.float64)
    if R_mat.ndim != 2 or R_mat.shape[1] != lookback_days:
        raise ValueError(
            f"scheme_g returns_matrix must be (N, {lookback_days}), "
            f"got {R_mat.shape}"
        )

    target_cols, macro_cols, ir_cols = _get_factor_columns(df, factor_cols)

    N = len(df)
    mask_valid = ~np.isnan(R_mat).any(axis=1)   # (N,)
    n_valid = int(mask_valid.sum())

    if n_valid < 2:
        raise RuntimeError(
            f"scheme_g: only {n_valid} stocks with valid 60-day returns, "
            f"need >=2 for SVD"
        )

    # ----- 对齐策略 -----
    if alignment == "impute":
        col_median = np.nanmedian(R_mat, axis=0)
        idx = np.where(np.isnan(R_mat))
        R_mat[idx] = np.take(col_median, idx[1])
        mask_valid = np.ones(N, dtype=bool)
        df["_pca_imputed"] = ~mask_valid  # 实际全 True（无缺失），占位
        n_valid = N

    R_clean = R_mat[mask_valid]                # (N_valid, 60)
    R_centered = R_clean - R_clean.mean(axis=0, keepdims=True)

    # ----- 纯 NumPy SVD -----
    n_comp = max(1, min(n_components, n_valid - 1, lookback_days))
    U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
    PC_valid = U[:, :n_comp] * S[:n_comp]       # (N_valid, n_comp)

    # 投影回全量
    PC_full = np.full((N, n_comp), np.nan)
    PC_full[mask_valid] = PC_valid

    # ----- 设计矩阵（仅 valid 行） -----
    ones = np.ones((n_valid, 1))
    X = np.column_stack([ones, PC_valid])       # (N_valid, 1+n_comp)

    # ----- 因子矩阵 + 清洗 -----
    Y_raw = df[target_cols].values
    Y_clean = _sanitize_factor_block(Y_raw, _EXEMPT_PREFIXES)
    Y_rankz = _rankz_block(Y_clean)
    Y_valid = Y_rankz[mask_valid]               # (N_valid, K)

    # ----- 矩阵化 OLS -----
    R = _batched_ols(X, Y_valid)                # (N_valid, K)
    R = _zscore_columns(R)

    # 写回：仅 valid 行
    out = np.full((N, len(target_cols)), np.nan)
    out[mask_valid] = R
    for j, col in enumerate(target_cols):
        df[col] = out[:, j]

    for col in macro_cols + ir_cols:
        df[col] = _cross_zscore(df[col].values.astype(float))

    return df


# ════════════════════════════════════════════════
# 统一入口
# ════════════════════════════════════════════════

def apply_neutralization(
    df: pd.DataFrame,
    factor_cols: list,
    scheme: str,
    **kwargs,
) -> pd.DataFrame:
    """
    统一入口，根据scheme名称分发到对应方案

    参数：
        df: 单月截面DataFrame（已筛选股票池）
        factor_cols: 需要中性化的因子列名列表
        scheme: 方案名（scheme_a/b/d/e/b1/b2/f/g）
        **kwargs: 各方案的扩展参数
            - scheme_b1: 无扩展
            - scheme_b2: 无扩展
            - scheme_f:  style_factors: list[str]
            - scheme_g:  returns_matrix, lookback_days, n_components, alignment

    返回：
        中性化后的DataFrame
    """
    dispatch = {
        "scheme_a":  (neutralize_scheme_a,  {}),
        "scheme_b":  (neutralize_scheme_b,  {}),
        "scheme_d":  (neutralize_scheme_d,  {}),
        "scheme_e":  (neutralize_scheme_e,  {}),
        "scheme_b1": (neutralize_scheme_b1, {}),
        "scheme_b2": (neutralize_scheme_b2, {}),
        "scheme_f":  (neutralize_scheme_f,  {"style_factors": kwargs.get("style_factors")}),
        "scheme_g":  (neutralize_scheme_g,  {
            "returns_matrix": kwargs.get("returns_matrix"),
            "lookback_days":  kwargs.get("lookback_days", 60),
            "n_components":   kwargs.get("n_components", 5),
            "alignment":      kwargs.get("alignment", "drop"),
        }),
    }
    if scheme not in dispatch:
        raise ValueError(
            f"未知中性化方案: {scheme}, "
            f"可选: {sorted(dispatch.keys())}"
        )
    func, extra = dispatch[scheme]
    logger.info(f"  [中性化] 使用{scheme}: "
                f"{func.__doc__.strip().splitlines()[0]}")
    return func(df, factor_cols, **extra)
