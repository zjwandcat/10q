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
M4 归因模块
===========

提供三大类归因：

1. Brinson 归因（市场收益 / 行业配置 / 个股选择）
   把组合超额收益按行业维度拆解
       Allocation   = (w_p - w_b) × R_b,sector
       Selection    = w_b × (R_p,sector - R_b,sector)
       Interaction  = (w_p - w_b) × (R_p,sector - R_b,sector)

2. 五因子归因（Fama-French 5 + Carhart Momentum）
   用代理因子做截面回归：MKT / SMB / HML / RMW / CMA / MOM
       R_p - R_f = α + β_mkt·MKT + β_smb·SMB + ... + ε

3. Barra 风险归因
   用 M0 已算好的 barra_ 因子（10 个）做组合暴露 + 因子收益代理

输入：
    all_portfolios : M2 输出 DataFrame（至少含 stock_code/industry/
                     weight/Target_Return_1M/pred_month/barra_*）
    benchmark_df   : 中证 800 月度收益，列 date/close
    factor_df      : M0 输出的全量 factor_df（带 industry/Barra 等）

输出：
    dict{
        'brinson':     DataFrame[month × {allocation,selection,interaction}],
        'five_factor': DataFrame[month × {alpha,beta_mkt,beta_smb,...}],
        'barra':       DataFrame[month × {exposure_*,ret_*,risk_*}],
        'summary':     dict 月度累计归因贡献
    }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger("m4.attribution")


# ───────────────────────────────────────────────────────────
#  工具函数
# ───────────────────────────────────────────────────────────
def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _load_benchmark() -> pd.DataFrame:
    """加载中证 800 基准月度收益（点对点法）

    month m 的基准 = (month m+1 月末close / month m 月末close) - 1
    与 M0 pipeline.py 中 Target_Return_1M 严格对齐（month m 月末 → month m+1 月末）。
    原版 first_close/last_close 算法（month m 月内）会导致策略曲线比基准早一个月。
    """
    p = Path("data/benchmark_000906.parquet")
    if not p.exists():
        logger.warning("未找到 data/benchmark_000906.parquet，使用空基准")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df['date'] = df['date'].astype(str)
    df['year_month'] = df['date'].str[:6]
    monthly = (
        df.sort_values('date')
        .groupby('year_month')
        .agg(last_close=('close', 'last'))
    )
    monthly['next_last_close'] = monthly['last_close'].shift(-1)
    monthly['benchmark_return'] = (
        monthly['next_last_close'] / monthly['last_close'] - 1
    )
    return monthly[['benchmark_return']].reset_index()


def _build_industry_universe(
    holdings: pd.DataFrame,
    factor_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    用 factor_df 的 industry 列，构造每个月"全市场"行业截面权重基准。

    如果 factor_df 没传，退化用 holding 月份的 industry 分布作为基准。
    """
    if factor_df is not None and "industry" in factor_df.columns:
        sub = factor_df[["trade_date", "stock_code", "industry"]].copy()
        sub["pred_month"] = sub["trade_date"].dt.strftime("%Y%m")
        # 月度行业内股票数 = 该行业内月总股数
        ind_count = (
            sub.groupby(["pred_month", "industry"])
            .size()
            .reset_index(name="n_stock")
        )
        # 月度总股数
        month_total = (
            sub.groupby("pred_month")
            .size()
            .reset_index(name="n_total")
        )
        ind_count = ind_count.merge(month_total, on="pred_month")
        ind_count["w_benchmark"] = (
            ind_count["n_stock"] / ind_count["n_total"]
        )
        return ind_count[["pred_month", "industry",
                           "w_benchmark"]]
    # 退化方案：用 holdings 的 industry 分布
    if "industry" in holdings.columns:
        g = (
            holdings.groupby(["pred_month", "industry"])
            .size()
            .reset_index(name="n_stock")
        )
        m = g.groupby("pred_month")["n_stock"].sum().reset_index(
            name="n_total")
        g = g.merge(m, on="pred_month")
        g["w_benchmark"] = g["n_stock"] / g["n_total"]
        return g[["pred_month", "industry", "w_benchmark"]]
    # 完全没有行业信息
    return pd.DataFrame(columns=[
        "pred_month", "industry", "w_benchmark"])


# ───────────────────────────────────────────────────────────
#  1. Brinson 归因
# ───────────────────────────────────────────────────────────
def brinson_attribution(
    holdings: pd.DataFrame,
    factor_df: Optional[pd.DataFrame] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Brinson-Fachler 行业归因。
    返回每月三块归因 + 合计超额收益。
    """
    if "industry" not in holdings.columns:
        logger.warning("holdings 缺 industry 列，跳过 Brinson 归因")
        return pd.DataFrame()

    if benchmark_df is None:
        bm = _load_benchmark()
    else:
        bm = benchmark_df

    # 1) 组合端：每月每个行业的组合权重 + 行业组合收益
    h = holdings.copy()
    is_cash_h = h["industry"].str.upper().isin(("CASH", "CASH_POOL"))
    h.loc[is_cash_h, "Target_Return_1M"] = h.loc[is_cash_h, "Target_Return_1M"].fillna(0.0)
    ind_w = (
        h.groupby(["pred_month", "industry"])
        ["weight"].sum()
        .reset_index(name="w_port")
    )
    # 行业组合收益 = Σ weight_i × return_i / Σ weight_i
    h["w_r"] = h["weight"] * h["Target_Return_1M"]
    ind_r = (
        h.groupby(["pred_month", "industry"])
        .agg(w_r_sum=("w_r", "sum"),
             w_sum=("weight", "sum"))
        .reset_index()
    )
    ind_r["r_port"] = np.where(
        ind_r["w_sum"] > 1e-8,
        ind_r["w_r_sum"] / ind_r["w_sum"], 0.0
    )
    ind_r = ind_r[["pred_month", "industry", "r_port"]]

    port_ind = ind_w.merge(ind_r, on=["pred_month", "industry"])

    # 2) 基准端：每月每个行业的基准权重（用全市场行业市值占比代理）
    bm_universe = _build_industry_universe(holdings, factor_df)
    if bm_universe.empty:
        # 退化：所有行业 w_benchmark=1/n_industries_of_month
        nind = port_ind.groupby("pred_month")["industry"].transform("size")
        port_ind["w_benchmark"] = 1.0 / nind
    else:
        port_ind = port_ind.merge(
            bm_universe, on=["pred_month", "industry"], how="left")
        port_ind["w_benchmark"] = (
            port_ind["w_benchmark"].fillna(0.0))

    # 3) 基准收益：行业月度收益 = 当月所有该行业股票等权平均收益
    if factor_df is not None and "Target_Return_1M" in factor_df.columns:
        fac = factor_df[["trade_date", "stock_code",
                         "Target_Return_1M", "industry"]].copy()
        fac["pred_month"] = fac["trade_date"].dt.strftime("%Y%m")
        fac = fac.dropna(subset=["Target_Return_1M"])
        ind_bm = (
            fac.groupby(["pred_month", "industry"])
            ["Target_Return_1M"]
            .mean()
            .reset_index(name="r_benchmark")
        )
        port_ind = port_ind.merge(
            ind_bm, on=["pred_month", "industry"], how="left")
        port_ind["r_benchmark"] = (
            port_ind["r_benchmark"].fillna(0.0))
    else:
        port_ind["r_benchmark"] = 0.0

    # 4) 基准月度总收益（市场收益）
    if not bm.empty and "benchmark_return" in bm.columns:
        bm_ret = bm.rename(columns={
            "year_month": "pred_month",
            "benchmark_return": "r_market"})
        port_ind = port_ind.merge(
            bm_ret, on="pred_month", how="left")
        port_ind["r_market"] = (
            port_ind["r_market"].fillna(0.0))
    else:
        # 用所有股票当月均值当 r_market
        if factor_df is not None and "Target_Return_1M" in factor_df.columns:
            fac = factor_df[["trade_date",
                             "Target_Return_1M"]].copy()
            fac["pred_month"] = fac["trade_date"].dt.strftime("%Y%m")
            mr = fac.groupby("pred_month")[
                "Target_Return_1M"].mean().reset_index(
                name="r_market")
            port_ind = port_ind.merge(
                mr, on="pred_month", how="left")
            port_ind["r_market"] = (
                port_ind["r_market"].fillna(0.0))
        else:
            port_ind["r_market"] = 0.0

    # 5) Brinson 三块
    port_ind["allocation"] = (
        (port_ind["w_port"] - port_ind["w_benchmark"])
        * (port_ind["r_benchmark"] - port_ind["r_market"])
    )
    port_ind["selection"] = (
        port_ind["w_benchmark"]
        * (port_ind["r_port"] - port_ind["r_benchmark"])
    )
    port_ind["interaction"] = (
        (port_ind["w_port"] - port_ind["w_benchmark"])
        * (port_ind["r_port"] - port_ind["r_benchmark"])
    )

    is_cash = port_ind["industry"].str.upper().isin(("CASH", "CASH_POOL"))
    port_ind.loc[is_cash, "selection"] = 0.0
    port_ind.loc[is_cash, "interaction"] = 0.0

    return port_ind[[
        "pred_month", "industry",
        "w_port", "w_benchmark",
        "r_port", "r_benchmark", "r_market",
        "allocation", "selection", "interaction",
    ]]


# ───────────────────────────────────────────────────────────
#  2. 五因子归因
# ───────────────────────────────────────────────────────────
# 代理因子构造（不需要外部因子库）：
#   MKT  = 截面股票收益均值 - rf_monthly
#   SMB  = 小盘股组合收益 - 大盘股组合收益（按 ln_market_cap）
#   HML  = 高账面市值比组 - 低账面市值比组（用 EP 代理：1/pe_ttm）
#   RMW  = 高 ROE 组 - 低 ROE 组
#   CMA  = 投资增长低组 - 投资增长高组（用 netprofit_yoy 增速代理）
#   MOM  = 过去 12 月收益高组 - 过去 12 月收益低组（用 1m 收益近似）
FIVE_FACTOR_COLS = [
    "factor_mkt", "factor_smb", "factor_hml",
    "factor_rmw", "factor_cma", "factor_mom",
]


def _compute_proxy_factors(
    factor_df: pd.DataFrame,
    rf_annual: float = 0.03,
) -> pd.DataFrame:
    """
    构造每月 6 个代理因子的截面因子收益。
    返回 DataFrame[pred_month × 6 factor cols]
    """
    if factor_df is None or factor_df.empty:
        return pd.DataFrame(columns=["pred_month"] + FIVE_FACTOR_COLS)

    fac = factor_df.copy()
    if "trade_date" not in fac.columns:
        return pd.DataFrame(columns=["pred_month"] + FIVE_FACTOR_COLS)
    fac["pred_month"] = fac["trade_date"].dt.strftime("%Y%m")
    rf_m = rf_annual / 12.0

    rows = []
    for m, g in fac.groupby("pred_month"):
        row = {"pred_month": m}
        if "Target_Return_1M" not in g.columns or len(g) < 30:
            for c in FIVE_FACTOR_COLS:
                row[c] = 0.0
            rows.append(row)
            continue
        ret = g["Target_Return_1M"]
        row["factor_mkt"] = _safe_float(ret.mean() - rf_m)

        # SMB：按 ln_market_cap 中位数分大小盘
        # 兼容 M0 列名: size_log_mcap / ln_market_cap / log_mcap
        cap_col = None
        for c in ("size_log_mcap", "ln_market_cap", "log_mcap"):
            if c in g.columns:
                cap_col = c
                break
        if cap_col:
            cap = g[cap_col]
            valid_cap = cap.notna()
            if valid_cap.sum() >= 30:
                med = cap[valid_cap].median()
                small = ret[valid_cap][cap[valid_cap] <= med].mean()
                big   = ret[valid_cap][cap[valid_cap] >  med].mean()
                row["factor_smb"] = _safe_float(small - big)
            else:
                row["factor_smb"] = 0.0
        else:
            row["factor_smb"] = 0.0

        # HML：用 EP 代理（pe_ttm 倒数；缺则用 0）
        # 兼容 M0 列名: sup_val_pe_ttm / pe_ttm / val_pe_ttm
        pe_col = None
        for c in ("sup_val_pe_ttm", "pe_ttm", "val_pe_ttm"):
            if c in g.columns:
                pe_col = c
                break
        if pe_col:
            pe = g[pe_col]
            valid_pe = pe.notna() & (pe != 0)
            if valid_pe.sum() >= 30:
                ep = 1.0 / pe[valid_pe]
                med = ep.median()
                ret_valid = ret[valid_pe]
                high = ret_valid[ep >= med].mean()
                low  = ret_valid[ep <  med].mean()
                row["factor_hml"] = _safe_float(high - low)
            else:
                row["factor_hml"] = 0.0
        else:
            row["factor_hml"] = 0.0

        # RMW：ROE
        # 兼容 M0 列名: quality_roe / roe / quality_roe_dt
        roe_col = None
        for c in ("quality_roe", "roe", "quality_roe_dt"):
            if c in g.columns:
                roe_col = c
                break
        if roe_col:
            roe = g[roe_col]
            valid_roe = roe.notna()
            if valid_roe.sum() >= 30:
                med = roe[valid_roe].median()
                ret_valid = ret[valid_roe]
                high = ret_valid[roe[valid_roe] >= med].mean()
                low  = ret_valid[roe[valid_roe] <  med].mean()
                row["factor_rmw"] = _safe_float(high - low)
            else:
                row["factor_rmw"] = 0.0
        else:
            row["factor_rmw"] = 0.0

        # CMA：投资增长（netprofit_yoy）
        # 兼容 M0 列名: growth_netprofit_yoy / netprofit_yoy
        inv_col = None
        for c in ("growth_netprofit_yoy", "netprofit_yoy"):
            if c in g.columns:
                inv_col = c
                break
        if inv_col:
            inv = g[inv_col]
            valid_inv = inv.notna()
            if valid_inv.sum() >= 30:
                med = inv[valid_inv].median()
                ret_valid = ret[valid_inv]
                # CMA = 投资低 - 投资高
                low  = ret_valid[inv[valid_inv] <= med].mean()
                high = ret_valid[inv[valid_inv] >  med].mean()
                row["factor_cma"] = _safe_float(low - high)
            else:
                row["factor_cma"] = 0.0
        else:
            row["factor_cma"] = 0.0

        # MOM：1m 动量（用 Target_Return_1M 近似，无更长窗口时）
        med_r = ret.median()
        high  = ret[ret >= med_r].mean()
        low   = ret[ret <  med_r].mean()
        row["factor_mom"] = _safe_float(high - low)
        rows.append(row)

    return pd.DataFrame(rows)


def five_factor_attribution(
    monthly: pd.DataFrame,
    factor_df: Optional[pd.DataFrame] = None,
    rf_annual: float = 0.03,
) -> pd.DataFrame:
    """
    五因子回归归因。
    输入 monthly: PerformanceMetrics.compute_monthly_returns 的输出
        列至少含 pred_month/portfolio_return/benchmark_return
    返回每月 alpha + 6 个 beta + R²
    """
    if factor_df is None or factor_df.empty:
        return pd.DataFrame()

    factor_rets = _compute_proxy_factors(factor_df, rf_annual=rf_annual)
    if factor_rets.empty:
        return pd.DataFrame()

    df = monthly.merge(factor_rets, on="pred_month", how="left")
    for c in FIVE_FACTOR_COLS:
        df[c] = df[c].fillna(0.0)

    y = (df["portfolio_return"] - rf_annual / 12.0).values
    X = df[FIVE_FACTOR_COLS].values
    # 加常数项
    X1 = np.column_stack([np.ones(len(X)), X])

    rows = []
    for i, m in enumerate(df["pred_month"]):
        # ★ v5.3 改造: 用 expanding window (从第 6 月开始), 不再用 12 月 rolling
        #   原因: rolling 12 月在样本期前 6 个月会全部 0, 后续即使有数据
        #   也因窗口太小协方差矩阵病态 (det < 1e-10) 全 0
        #   expanding: 第 i 月用 [0..i] 全部历史, 样本越多回归越稳
        pass
    # 用 expanding window (最少 6 月) 做 OLS
    n = len(df)
    min_obs = 6
    for i in range(n):
        s = 0
        yi = y[s:i + 1]
        Xi = X1[s:i + 1]
        if len(yi) < min_obs:
            rows.append({
                "pred_month": df["pred_month"].iloc[i],
                "alpha": 0.0,
                "beta_mkt": 0.0, "beta_smb": 0.0, "beta_hml": 0.0,
                "beta_rmw": 0.0, "beta_cma": 0.0, "beta_mom": 0.0,
                "r_squared": 0.0,
            })
            continue
        # OLS β = (X'X)^-1 X'y
        # ★ v5.3 改造: 用 Ridge 正则化 + np.linalg.lstsq (SVD 自动处理奇异)
        #   原因: 多窗口下 SMB/HML/CMA 等代理因子可能因 valid 不足全为 0,
        #   X 矩阵严重奇异 (cond 1e+19), np.linalg.solve 抛 LinAlgError
        #   → 被 except 吞掉 → 全 0
        n_obs, n_feat = Xi.shape
        # Ridge: (X'X + λI)^-1 X'y, λ 自动按 n_feat 缩放
        lam = max(0.01, n_feat * 1e-4)
        XtX = Xi.T @ Xi + lam * np.eye(n_feat)
        try:
            coef = np.linalg.solve(XtX, Xi.T @ yi)
            # 检查解的合理性
            if not np.all(np.isfinite(coef)):
                coef = np.zeros(n_feat)
        except np.linalg.LinAlgError:
            # 退化: 用 lstsq 拿 SVD 最小二乘
            try:
                coef, _, _, _ = np.linalg.lstsq(Xi, yi, rcond=None)
                if not np.all(np.isfinite(coef)):
                    coef = np.zeros(n_feat)
            except Exception:
                coef = np.zeros(n_feat)

        # R²
        yhat = Xi @ coef
        ss_res = float(((yi - yhat) ** 2).sum())
        ss_tot = float(((yi - yi.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        rows.append({
            "pred_month": df["pred_month"].iloc[i],
            "alpha":      _safe_float(coef[0] * 12),  # 月度 α 年化
            "beta_mkt":   _safe_float(coef[1]),
            "beta_smb":   _safe_float(coef[2]),
            "beta_hml":   _safe_float(coef[3]),
            "beta_rmw":   _safe_float(coef[4]),
            "beta_cma":   _safe_float(coef[5]),
            "beta_mom":   _safe_float(coef[6]),
            "r_squared":  _safe_float(r2),
        })
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────
#  3. Barra 风险归因
# ───────────────────────────────────────────────────────────
BARRA_FACTORS = [
    "barra_beta", "barra_momentum", "barra_size",
    "barra_earnings_yield", "barra_value", "barra_volatility",
    "barra_liquidity", "barra_leverage", "barra_growth",
    "barra_quality",
]


def _monthly_barra_factor_returns(
    factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    构造每月 10 个 Barra 因子的截面因子收益。
    做法：每因子按当月截面分 5 组，求 Q5-Q1 收益差。
    """
    if factor_df is None or factor_df.empty:
        return pd.DataFrame(columns=["pred_month"] + BARRA_FACTORS)
    fac = factor_df.copy()
    if "trade_date" not in fac.columns:
        return pd.DataFrame(columns=["pred_month"] + BARRA_FACTORS)
    fac["pred_month"] = fac["trade_date"].dt.strftime("%Y%m")

    rows = []
    for m, g in fac.groupby("pred_month"):
        row = {"pred_month": m}
        if "Target_Return_1M" not in g.columns or len(g) < 50:
            for c in BARRA_FACTORS:
                row[c] = 0.0
            rows.append(row)
            continue
        for c in BARRA_FACTORS:
            if c not in g.columns:
                row[c] = 0.0
                continue
            v = g[c]
            valid = v.notna()
            if valid.sum() < 20:
                row[c] = 0.0
                continue
            try:
                qs = pd.qcut(v[valid], 5, labels=False,
                             duplicates="drop")
                if qs.nunique() < 2:
                    row[c] = 0.0
                    continue
                ret_v = g.loc[valid, "Target_Return_1M"]
                # Q5 - Q1
                top_q = qs.max()
                bot_q = qs.min()
                top = ret_v[qs == top_q].mean()
                bot = ret_v[qs == bot_q].mean()
                row[c] = _safe_float(top - bot)
            except Exception:
                row[c] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def barra_attribution(
    holdings: pd.DataFrame,
    factor_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Barra 风险归因。
    对每个月：
        exposure_k = Σ w_i × barra_k_i
        ret_k      = 截面 Q5-Q1 因子收益
        risk_k     = exposure_k × ret_k  (一阶近似)
    """
    if factor_df is None or factor_df.empty:
        return pd.DataFrame()
    if not any(c in holdings.columns for c in BARRA_FACTORS):
        fac = factor_df[
            ["trade_date", "stock_code"] +
            [c for c in BARRA_FACTORS if c in factor_df.columns]
        ].copy()
        fac["pred_month"] = fac["trade_date"].dt.strftime("%Y%m")
        h = holdings.merge(
            fac, on=["pred_month", "stock_code"], how="left")
    else:
        h = holdings.copy()

    is_cash_pool = h["stock_code"].astype(str).str.upper() == "CASH_POOL"
    for c in BARRA_FACTORS:
        if c in h.columns:
            h.loc[is_cash_pool, c] = 0.0

    factor_rets = _monthly_barra_factor_returns(factor_df)
    if factor_rets.empty:
        return pd.DataFrame()

    rows = []
    for m, g in h.groupby("pred_month"):
        row = {"pred_month": m}
        if g["weight"].sum() <= 0:
            for c in BARRA_FACTORS:
                row[f"expo_{c}"] = 0.0
                row[f"risk_{c}"] = 0.0
            rows.append(row)
            continue
        w = g["weight"] / g["weight"].sum()
        for c in BARRA_FACTORS:
            if c in g.columns:
                expo = _safe_float((w * g[c].fillna(0.0)).sum())
            else:
                expo = 0.0
            row[f"expo_{c}"] = expo
        # 因子收益（从截面分位 Q5-Q1 算的）
        frets = factor_rets[factor_rets["pred_month"] == m]
        if not frets.empty:
            frets_r = frets.iloc[0]
        else:
            frets_r = pd.Series(
                {c: 0.0 for c in BARRA_FACTORS})
        for c in BARRA_FACTORS:
            expo = row[f"expo_{c}"]
            fret = _safe_float(frets_r.get(c, 0.0))
            row[f"risk_{c}"] = expo * fret
        rows.append(row)
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────
#  对外主接口
# ───────────────────────────────────────────────────────────
def run_attribution(
    holdings: pd.DataFrame,
    monthly: pd.DataFrame,
    factor_df: Optional[pd.DataFrame] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
    rf_annual: float = 0.03,
) -> Dict[str, pd.DataFrame]:
    """
    一次性产出 Brinson / Five-Factor / Barra 三套归因。

    参数:
        holdings:  all_portfolios[is_holding==True]，
                   至少含 stock_code/pred_month/weight/
                   Target_Return_1M/industry
        monthly:   compute_monthly_returns 的输出
        factor_df: M0 全量因子数据（带 industry + Barra）
        benchmark_df: 中证 800 月度收益（可选）
        rf_annual: 无风险利率（年化）

    返回:
        dict{
            'brinson':     DataFrame,
            'five_factor': DataFrame,
            'barra':       DataFrame,
        }
    """
    logger.info(
        f"开始归因: holdings={len(holdings)}行 "
        f"monthly={len(monthly)}行 factor_df="
        f"{len(factor_df) if factor_df is not None else 0}行")

    out: Dict[str, pd.DataFrame] = {}

    # 1) Brinson
    try:
        out["brinson"] = brinson_attribution(
            holdings, factor_df=factor_df,
            benchmark_df=benchmark_df)
        logger.info(
            f"  Brinson 归因完成: {len(out['brinson'])}行")
    except Exception as e:
        logger.warning(f"Brinson 归因失败: {e}")
        out["brinson"] = pd.DataFrame()

    # 2) Five-Factor
    try:
        out["five_factor"] = five_factor_attribution(
            monthly, factor_df=factor_df, rf_annual=rf_annual)
        logger.info(
            f"  五因子归因完成: {len(out['five_factor'])}行")
    except Exception as e:
        logger.warning(f"五因子归因失败: {e}")
        out["five_factor"] = pd.DataFrame()

    # 3) Barra
    try:
        out["barra"] = barra_attribution(
            holdings, factor_df=factor_df)
        logger.info(
            f"  Barra 归因完成: {len(out['barra'])}行")
    except Exception as e:
        logger.warning(f"Barra 归因失败: {e}")
        out["barra"] = pd.DataFrame()

    return out
