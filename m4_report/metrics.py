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
绩效指标计算模块
输入：output/all_portfolios.parquet
输出：28+项绩效指标字典
"""
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger("m4.metrics")


def _load_config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_benchmark() -> pd.DataFrame:
    """加载中证800基准数据

    配套 _calc_benchmark_returns_point_to_point：
    month m 的基准 = month m 月末 close → month m+1 月末 close 的点对点收益。
    这样和 M0 中 Target_Return_1M 的"month m 月末 → month m+1 月末"语义一致，
    避免 pred_month=m 的策略（下月收益）和基准（本月收益）错位一个月。
    """
    p = Path("data/benchmark_000906.parquet")
    if not p.exists():
        logger.warning("未找到 data/benchmark_000906.parquet")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    logger.info(f"已加载基准数据: {len(df)}行")
    return df


def _calc_benchmark_returns_point_to_point(
    benchmark_df: pd.DataFrame,
) -> pd.Series:
    """
    点对点（point-to-point）月度收益计算。

    month m 的 return = (month m+1 月末close / month m 月末close) - 1

    与 M0 pipeline.py 里 Target_Return_1M 的定义完全一致：
    M0 用 yyyymm 月最后一个交易日 → yyyymm+1 月最后一个交易日的复权价收益
    作为 month yyyymm 截面对应的 forward 1M return。

    修复前 bug（first_close / last_close）：
        month m 的 return = (month m 月内last_close / month m 月内first_close) - 1
        也就是"month m 月内"收益，与策略"month m 月末 → month m+1 月末"语义不一致，
        导致净值曲线整体提前一个月（策略波峰对应上个月基准的波谷）。
    """
    if benchmark_df is None or benchmark_df.empty:
        return pd.Series(dtype=float)
    df = benchmark_df.copy()
    if "date" not in df.columns or "close" not in df.columns:
        logger.warning("benchmark_df 缺 date/close 列，返回空 Series")
        return pd.Series(dtype=float)
    df["date"] = df["date"].astype(str)
    df["year_month"] = df["date"].str[:6]

    # 每月最后一个交易日 close
    monthly_last = (
        df.sort_values("date")
        .groupby("year_month")
        .agg(last_close=("close", "last"))
    )
    # 下月最后交易日 close（最后一个月的下月为 NaN，return 也是 NaN）
    monthly_last["next_last_close"] = monthly_last["last_close"].shift(-1)
    monthly_last["return"] = (
        monthly_last["next_last_close"] / monthly_last["last_close"] - 1
    )
    return monthly_last["return"]


def _compute_jensen_appraisal_full(
    monthly: pd.DataFrame,
    rf_annual: float = 0.03,
) -> Dict[str, float]:
    """
    在完整月度收益序列上做 CAPM 回归，返回 Jensen's Alpha (年化) 与 Appraisal Ratio (月度口径)。

    数学定义:
        y_t = R_p,t − R_f,t
        x_t = R_b,t − R_f,t
        OLS: y_t = α_monthly + β·x_t + ε_t
        → jensen_alpha    = α_monthly × 12  (年化)
        → appraisal_ratio = α_monthly / std(ε, ddof=1)

    退化保护:
        - len(r) < 12            → 0.0
        - n_unique(x) < 2        → 0.0（基准无波动）
        - std(ε) < 1e-8          → appraisal_ratio = 0.0
        - NaN / Inf 输入或输出   → 0.0
    """
    if len(monthly) < 12:
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0, "beta": 0.0}
    if "portfolio_return" not in monthly.columns or \
       "benchmark_return" not in monthly.columns:
        logger.warning("monthly 缺少 portfolio_return / benchmark_return 列")
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0, "beta": 0.0}

    rf_monthly = rf_annual / 12.0
    rets = monthly["portfolio_return"].to_numpy(dtype=np.float64)
    bms  = monthly["benchmark_return"].to_numpy(dtype=np.float64)

    y = rets - rf_monthly
    x = bms  - rf_monthly

    if np.unique(x).size < 2:
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0, "beta": 0.0}

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    ss_xx  = float(((x - x_mean) ** 2).sum())
    if ss_xx < 1e-12:
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0, "beta": 0.0}

    beta  = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
    alpha = y_mean - beta * x_mean

    residuals = y - (alpha + beta * x)
    if residuals.size < 2:
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0, "beta": 0.0}

    sigma_eps = float(residuals.std(ddof=1))

    jensen_alpha    = alpha * 12.0
    appraisal_ratio = (alpha / sigma_eps) if sigma_eps > 1e-8 else 0.0

    if not (np.isfinite(jensen_alpha) and np.isfinite(appraisal_ratio) and np.isfinite(beta)):
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0, "beta": 0.0}

    return {
        "jensen_alpha":    round(float(jensen_alpha),    6),
        "appraisal_ratio": round(float(appraisal_ratio), 6),
        "beta":            round(float(beta),            6),
    }


class PerformanceMetrics:
    """
    绩效指标计算器
    用法：
        pm = PerformanceMetrics()
        monthly_returns = pm.compute_monthly_returns(all_portfolios)
        metrics = pm.calculate(monthly_returns)
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = _load_config()
        self._threshold_cfg = config.get("performance", {})
        perf = self._threshold_cfg
        self.rf = perf.get("risk_free_rate", 0.03)
        self.rf_monthly = self.rf / 12
        # 换手成本参数
        portfolio_cfg = config.get("portfolio", {})
        cost_cfg = portfolio_cfg.get("cost", {})
        self.stamp_duty = cost_cfg.get("stamp_duty", 0.001)
        self.commission = cost_cfg.get("commission", 0.0003)
        self.slippage = cost_cfg.get("slippage", 0.001)
        logger.info(
            f"PerformanceMetrics初始化: rf={self.rf:.2%}, "
            f"stamp_duty={self.stamp_duty}, commission={self.commission}, slippage={self.slippage}")

    def compute_monthly_returns(
        self,
        all_portfolios: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        从all_portfolios计算月度组合收益
        输入：run_m2输出的all_portfolios
        输出：含portfolio_return/benchmark_return/
              excess_return/turnover_cost列的月度DataFrame
        """
        logger.info("开始计算月度收益...")

        if "is_holding" in all_portfolios.columns:
            holdings = all_portfolios[
                all_portfolios["is_holding"] == True].copy()
        else:
            holdings = all_portfolios[
                all_portfolios["stock_code"] != "CASH_POOL"].copy()
        logger.info(f"  筛选持仓: {len(holdings)}行")

        # 处理收益缺失
        missing = holdings["Target_Return_1M"].isna().sum()
        if missing > 0:
            logger.warning(
                f"W002: {missing}个股票收益缺失，已忽略")
            holdings = holdings.dropna(
                subset=["Target_Return_1M"])

        # 按月计算加权组合收益
        monthly = (
            holdings.groupby("pred_month")
            .apply(lambda g: pd.Series({
                "portfolio_return": (
                    g["weight"] * g["Target_Return_1M"]
                ).sum(),
                "is_penalized": g["is_penalized"].iloc[0],
                "val_ic": g["val_ic"].iloc[0],
                "ic_gap": g["ic_gap"].iloc[0],
            }), include_groups=False)
            .reset_index()
        )

        # ★ 优化: 用 groupby 一次性构建所有月份持仓字典, 替代逐月循环过滤
        # 原: for m in months: holdings[holdings["pred_month"] == m] (O(N×M) 全表扫描)
        # 新: 1 次 groupby + dict 构建 (O(M) 一次扫描)
        from m2_engine.portfolio_builder import calculate_turnover_cost
        monthly_holdings = (
            holdings.groupby("pred_month")
            .apply(lambda g: dict(zip(g["stock_code"], g["weight"])))
            .to_dict()
        )

        # ★ 优化: 删除冗余 is_holding 检查
        # 原因: 第149行已确保所有记录 is_holding == True, 第187-192行的检查永远不会过滤掉任何记录
        prev_holdings = {}
        turnover_costs = []
        for m in sorted(monthly["pred_month"].tolist()):
            curr_holdings = monthly_holdings[m]  # O(1) 字典查找
            if prev_holdings:
                cost = calculate_turnover_cost(
                    prev_holdings, curr_holdings,
                    stamp_duty=self.stamp_duty,
                    commission=self.commission,
                    slippage=self.slippage,
                )
            else:
                cost = 0.0  # 首月无换手
            turnover_costs.append(cost)
            prev_holdings = curr_holdings

        monthly["turnover_cost"] = turnover_costs

        # 扣除换手成本后的净收益
        monthly["net_return"] = (
            monthly["portfolio_return"] - monthly["turnover_cost"]
        )

        # ★ 修复: 改用"点对点"算法计算基准月度收益
        #   month m 的基准 = (month m+1 月末close / month m 月末close) - 1
        #   与 M0 pipeline.py 中 Target_Return_1M 的定义严格对齐：
        #   M0 用 yyyymm 月最后交易日 → yyyymm+1 月最后交易日的复权价作为 month yyyymm
        #   截面对应的 forward 1M return。原 M4 用"月内首末 close"导致策略曲线早基准一个月。
        benchmark_df = _load_benchmark()
        if not benchmark_df.empty and len(monthly) > 0:
            bench_returns = _calc_benchmark_returns_point_to_point(
                benchmark_df)
            monthly['benchmark_return'] = (
                monthly['pred_month'].map(bench_returns).fillna(0.0)
            )
            logger.info("  使用真实基准数据（中证800，点对点法）")
        else:
            monthly['benchmark_return'] = 0.0
            logger.warning("  未找到基准数据，使用默认值0")

        monthly["excess_return"] = (
            monthly["portfolio_return"] -
            monthly["benchmark_return"]
        )

        monthly = monthly.sort_values("pred_month")
        logger.info(
            f"  月度收益计算完成: {len(monthly)}个月")
        return monthly

    def calculate(
        self,
        monthly: pd.DataFrame,
    ) -> Dict:
        """计算28+项绩效指标"""
        logger.info("开始计算绩效指标...")
        r = monthly["portfolio_return"].values
        bm = monthly["benchmark_return"].values
        ex = monthly["excess_return"].values
        n = len(r)

        # 累计净值
        cum = np.cumprod(1 + r)
        cum_bm = np.cumprod(1 + bm)

        # CAGR
        years = n / 12
        cagr = (cum[-1] ** (1 / years) - 1) if years > 0 else 0.0
        cagr_bm = (cum_bm[-1] ** (1 / years) - 1) if years > 0 else 0.0

        # 最大回撤
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(dd.min())

        # 月度胜率
        win_rate = float((r > 0).mean())

        # 波动率
        vol = float(np.std(r, ddof=1) * np.sqrt(12))
        down_r = r[r < self.rf_monthly]
        down_vol = float(np.std(down_r, ddof=1) * np.sqrt(12)) if len(down_r) > 1 else 1e-6
        up_r = r[r > self.rf_monthly]
        up_vol = float(np.std(up_r, ddof=1) * np.sqrt(12)) if len(up_r) > 1 else 1e-6

        # 夏普/索提诺/卡玛
        sharpe = (cagr - self.rf) / vol if vol > 1e-8 else 0.0
        sortino = (cagr - self.rf) / down_vol if down_vol > 1e-8 else 0.0
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

        # IR
        ex_std = float(np.std(ex, ddof=1))
        ir = float(np.mean(ex) / ex_std * np.sqrt(12)) if ex_std > 1e-8 else 0.0

        # VaR/CVaR (月度，95%)
        var_95 = float(np.percentile(r, 5))
        cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95

        # 偏度/峰度
        from scipy.stats import skew, kurtosis
        skewness = float(skew(r))
        kurt = float(kurtosis(r))

        # 痛苦指数/溃疡指数
        pain = float(np.abs(dd).mean())
        ulcer = float(np.sqrt(np.mean(dd**2)))

        # Omega比率
        gains = r[r > self.rf_monthly] - self.rf_monthly
        losses = self.rf_monthly - r[r <= self.rf_monthly]
        omega = float(gains.sum() / losses.sum()) if losses.sum() > 1e-8 else 0.0

        # 尾部比率
        p95 = float(np.percentile(r, 95))
        p05 = abs(float(np.percentile(r, 5)))
        tail = p95 / p05 if p05 > 1e-8 else 0.0

        # 捕获率
        up_months = bm > 0
        down_months = bm < 0
        up_cap = (float(r[up_months].mean() / bm[up_months].mean())
                  if up_months.any() and bm[up_months].mean() != 0 else 0.0)
        dn_cap = (float(r[down_months].mean() / bm[down_months].mean())
                  if down_months.any() and bm[down_months].mean() != 0 else 0.0)
        capture = up_cap / dn_cap if abs(dn_cap) > 1e-8 else 0.0

        # Sterling/Burke/Martin
        yearly_dd = [dd[i:i+12].min() for i in range(0, n-11, 12)]
        avg_yearly_dd = abs(np.mean(yearly_dd)) if yearly_dd else abs(max_dd)
        sterling = cagr / avg_yearly_dd if avg_yearly_dd > 1e-8 else 0.0
        burke = cagr / np.sqrt(np.sum(np.array(yearly_dd)**2)) if yearly_dd else 0.0
        martin = cagr / ulcer if ulcer > 1e-8 else 0.0

        # 滚动6月超额胜率
        roll6_win = 0.0
        roll6_excess = [ex[i:i+6].sum() for i in range(n-5)]
        if roll6_excess:
            roll6_win = float(np.mean(np.array(roll6_excess) > 0))

        # 滚动6月IR（与M2 ensemble.py同公式：6月窗口IR均值，ddof=1）
        rolling6m_ir = 0.0
        if n >= 6:
            from numpy.lib.stride_tricks import sliding_window_view
            win_ex = sliding_window_view(ex, 6)
            win_ex_mean = win_ex.mean(axis=1)
            win_ex_std = win_ex.std(axis=1, ddof=1)
            irs = np.where(
                win_ex_std > 1e-8,
                win_ex_mean / np.where(win_ex_std > 1e-8, win_ex_std, 1.0) * np.sqrt(12),
                0.0)
            rolling6m_ir = float(np.mean(irs))

        # ★ 换手成本指标
        turnover_costs = monthly["turnover_cost"].values if "turnover_cost" in monthly.columns else np.zeros(n)
        avg_monthly_turnover_cost = float(np.mean(turnover_costs))
        avg_annual_turnover_cost = avg_monthly_turnover_cost * 12

        # ★ 扣费后年化收益
        net_r = monthly["net_return"].values if "net_return" in monthly.columns else r
        cum_net = np.cumprod(1 + net_r)
        net_cagr = (cum_net[-1] ** (1 / years) - 1) if years > 0 else 0.0

        # ★ Jensen's Alpha & Appraisal Ratio（全期 OLS）
        _ja = _compute_jensen_appraisal_full(monthly, rf_annual=self.rf)

        # ★ SQN (System Quality Number): mean(r)/std(r)*sqrt(n)
        r_std = float(np.std(r, ddof=1))
        sqn = (float(np.mean(r)) / r_std * np.sqrt(n)) if r_std > 1e-8 and n > 1 else 0.0

        metrics = {
            "cagr": round(cagr, 6),
            "cagr_benchmark": round(cagr_bm, 6),
            "annual_excess": round(cagr - cagr_bm, 6),
            "max_drawdown": round(max_dd, 6),
            "monthly_win_rate": round(win_rate, 6),
            "volatility": round(vol, 6),
            "downside_volatility": round(down_vol, 6),
            "upside_volatility": round(up_vol, 6),
            "volatility_ratio": round(up_vol/down_vol if down_vol>1e-8 else 0, 6),
            "sharpe_ratio": round(sharpe, 6),
            "sortino_ratio": round(sortino, 6),
            "calmar_ratio": round(calmar, 6),
            "ir": round(ir, 6),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "skewness": round(skewness, 6),
            "kurtosis": round(kurt, 6),
            "pain_index": round(pain, 6),
            "ulcer_index": round(ulcer, 6),
            "omega_ratio": round(omega, 6),
            "tail_ratio": round(tail, 6),
            "up_capture_ratio": round(up_cap, 6),
            "down_capture_ratio": round(dn_cap, 6),
            "capture_ratio": round(capture, 6),
            "sterling_ratio": round(sterling, 6),
            "burke_ratio": round(burke, 6),
            "martin_ratio": round(martin, 6),
            "rolling6m_win_rate": round(roll6_win, 6),
            "rolling6m_ir": round(rolling6m_ir, 6),
            "n_months": n,
            "avg_monthly_turnover_cost": round(avg_monthly_turnover_cost, 6),
            "avg_annual_turnover_cost": round(avg_annual_turnover_cost, 6),
            "net_cagr_after_cost": round(net_cagr, 6),
            "jensen_alpha":    _ja["jensen_alpha"],
            "appraisal_ratio": _ja["appraisal_ratio"],
            "beta":            _ja["beta"],
            "sqn":             round(sqn, 6),
        }
        logger.info(
            f"  绩效指标计算完成: CAGR={cagr:.2%}, "
            f"夏普={sharpe:.2f}, IR={ir:.2f}")
        return metrics

    def check_thresholds(self, metrics: Dict) -> Dict:
        """9项硬性门槛检查，返回pass/fail"""
        cfg = self._threshold_cfg
        checks = {
            "ir_pass":       metrics["ir"] >= cfg.get("min_ir", 0.50),
            "calmar_pass":   metrics["calmar_ratio"] >= cfg.get("min_calmar", 1.00),
            "dd_pass":       metrics["max_drawdown"] >= -cfg.get("max_drawdown", 0.35),
            "sortino_pass":  metrics["sortino_ratio"] >= cfg.get("min_sortino", 1.20),
            "excess_pass":   metrics["annual_excess"] >= cfg.get("min_annual_excess", 0.05),
            "roll6_pass":    metrics["rolling6m_win_rate"] >= cfg.get("min_rolling6m_winrate", 0.60),
            "capture_pass":  metrics["capture_ratio"] >= cfg.get("min_capture_ratio", 1.2),
            "pain_pass":     metrics["pain_index"] <= cfg.get("max_pain_index", 0.10),
            "omega_pass":    metrics["omega_ratio"] >= cfg.get("min_omega_ratio", 1.2),
        }
        checks["all_pass"] = all(checks.values())
        return checks

    # ──────────────────────────────────────────────────
    #  归因（Brinson / 五因子 / Barra）
    # ──────────────────────────────────────────────────
    def run_attribution(
        self,
        all_portfolios: pd.DataFrame,
        monthly: pd.DataFrame,
        factor_df: pd.DataFrame = None,
    ) -> Dict:
        """
        对外统一入口：M4 报告需要的所有归因数据。
        返回:
            {
              'holdings':     DataFrame,  is_holding 子集 + 行业
              'brinson':      DataFrame,
              'five_factor':  DataFrame,
              'barra':        DataFrame,
              'shap_bundle':  dict,      {pred_month: (sv, sf, source)}
            }
        """
        from .attribution import run_attribution

        if "is_holding" in all_portfolios.columns:
            holdings = all_portfolios[
                all_portfolios["is_holding"] == True
            ].copy()
        else:
            holdings = all_portfolios[
                all_portfolios["stock_code"] != "CASH_POOL"
            ].copy()
        # ★ 改造: 优先从 all_portfolios.attrs 读完整 SHAP（M2 挂的）；
        #   退化用 shap_top1/2/3 列做粗略聚合
        shap_bundle: Dict = {}
        full_shap = all_portfolios.attrs.get("shap_data", None)
        if full_shap:
            shap_bundle = full_shap
        elif "shap_top1_factor" in holdings.columns:
            for m, g in holdings.groupby("pred_month"):
                cols = [c for c in [
                    "shap_top1_value", "shap_top2_value",
                    "shap_top3_value"] if c in g.columns]
                if cols:
                    shap_bundle[str(m)] = g[cols].to_numpy()

        attr = run_attribution(
            holdings=holdings,
            monthly=monthly,
            factor_df=factor_df,
            rf_annual=self.rf,
        )
        attr["holdings"]    = holdings
        attr["shap_bundle"] = shap_bundle
        return attr
