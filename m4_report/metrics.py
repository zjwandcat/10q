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
    """加载中证800基准数据"""
    p = Path("data/benchmark_000906.parquet")
    if not p.exists():
        logger.warning("未找到 data/benchmark_000906.parquet")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    logger.info(f"已加载基准数据: {len(df)}行")
    return df


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
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0}
    if "portfolio_return" not in monthly.columns or \
       "benchmark_return" not in monthly.columns:
        logger.warning("monthly 缺少 portfolio_return / benchmark_return 列")
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0}

    rf_monthly = rf_annual / 12.0
    rets = monthly["portfolio_return"].to_numpy(dtype=np.float64)
    bms  = monthly["benchmark_return"].to_numpy(dtype=np.float64)

    y = rets - rf_monthly
    x = bms  - rf_monthly

    if np.unique(x).size < 2:
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0}

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    ss_xx  = float(((x - x_mean) ** 2).sum())
    if ss_xx < 1e-12:
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0}

    beta  = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
    alpha = y_mean - beta * x_mean

    residuals = y - (alpha + beta * x)
    if residuals.size < 2:
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0}

    sigma_eps = float(residuals.std(ddof=1))

    jensen_alpha    = alpha * 12.0
    appraisal_ratio = (alpha / sigma_eps) if sigma_eps > 1e-8 else 0.0

    if not (np.isfinite(jensen_alpha) and np.isfinite(appraisal_ratio)):
        return {"jensen_alpha": 0.0, "appraisal_ratio": 0.0}

    return {
        "jensen_alpha":    round(float(jensen_alpha),    6),
        "appraisal_ratio": round(float(appraisal_ratio), 6),
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

        # 只取is_holding=True的持仓
        holdings = all_portfolios[
            all_portfolios["is_holding"] == True].copy()
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

        # 计算换手成本
        from m2_engine.portfolio_builder import calculate_turnover_cost
        prev_holdings = {}
        turnover_costs = []
        months_sorted = sorted(monthly["pred_month"].tolist())
        for m in months_sorted:
            month_holdings_df = holdings[holdings["pred_month"] == m]
            curr_holdings = dict(zip(
                month_holdings_df["stock_code"],
                month_holdings_df["weight"]
            ))
            # 只保留is_holding的
            curr_holdings = {
                s: w for s, w in curr_holdings.items()
                if month_holdings_df[
                    month_holdings_df["stock_code"] == s
                ]["is_holding"].iloc[0]
            }
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

        # 计算真实基准收益率（中证800）
        benchmark_df = _load_benchmark()
        if not benchmark_df.empty and len(monthly) > 0:
            benchmark_df['date'] = benchmark_df['date'].astype(str)
            monthly['benchmark_return'] = monthly['pred_month'].apply(
                lambda m: self._calc_benchmark_return(benchmark_df, m)
            )
            logger.info("  使用真实基准数据（中证800）")
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

    @staticmethod
    def _calc_benchmark_return(benchmark_df: pd.DataFrame, month: str) -> float:
        """计算某月的基准收益率"""
        try:
            year = int(str(month)[:4])
            month_num = int(str(month)[4:])
            
            # 找到该月第一个和最后一个交易日
            month_data = benchmark_df[
                (benchmark_df['date'].str[:6] == f"{year}{month_num:02d}")
            ].sort_values('date')
            
            if len(month_data) < 2:
                return 0.0
            
            first_close = month_data.iloc[0]['close']
            last_close = month_data.iloc[-1]['close']
            
            return (last_close - first_close) / first_close
        except Exception as e:
            logger.debug(f"计算基准收益失败 {month}: {e}")
            return 0.0

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
            "n_months": n,
            "avg_monthly_turnover_cost": round(avg_monthly_turnover_cost, 6),
            "avg_annual_turnover_cost": round(avg_annual_turnover_cost, 6),
            "net_cagr_after_cost": round(net_cagr, 6),
            "jensen_alpha":    _ja["jensen_alpha"],
            "appraisal_ratio": _ja["appraisal_ratio"],
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
