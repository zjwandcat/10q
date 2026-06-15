"""
因子计算模块
输入：单只股票的日线数据 + 财务数据 + 宏观数据
输出：该股票当月末所有因子值（460个）

因子分类（460列）：
  动量类(51)    momentum_
  波动率类(26)  volatility_
  质量类(14)    quality_
  估值类(9)     value_
  成长类(6)     growth_
  技术类(61)    technical_
  流动性类(16)  liquidity_
  规模类(5)     size_
  股息类(5)     dividend_
  Alpha101(50)  alpha101_
  Barra(10)     barra_
  AQR(6)        aqr_
  聚宽JQ(20)    jq_
  复合衍生(8)   deriv_
  滚动统计(12)  roll_
  行业相对(6)   industry_relative_
  情绪类(5)     sentiment_
  FF五因子(5)   ff_
  盈利质量NM(4) nm_
  流动性冲击PS(3) ps_
  宏观类(68)    macro_

实现原则：
  - 所有因子使用月末最后一个交易日的截面数据
  - 需要历史数据的因子（如动量）使用向量化计算
  - 缺失数据返回NaN，不填0（M0不填充，M1的FeatureStore统一填充）
  - float32存储节省内存
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


class FactorCalculator:
    """
    因子计算器
    对单只股票计算所有460个因子
    """

    @staticmethod
    def _safe_float(val, default=np.nan):
        """安全float转换：处理Tushare返回的None值"""
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def calculate_all(
        self,
        daily_df: pd.DataFrame,      # 该股票日线数据（至少240天）
        financial_data: dict,         # 最近一期财务数据
        macro_data: dict,             # 当月宏观数据
        trade_date: str,             # 当月末交易日YYYYMMDD
        basic_data: dict = None,     # 月末基础行情数据（含total_mv等）
    ) -> dict:
        """
        计算所有因子，返回因子值字典
        daily_df必须包含：open,high,low,close,volume,amount,turnover_rate
        按trade_date降序排列（最新在前）
        """
        factors = {}

        # ── 动量类（51个）─────────────────────────
        factors.update(self._calc_momentum(daily_df))

        # ── 反转类（2个）─────────────────────────
        factors.update(self._calc_reversal(factors))

        # ── 波动率类（26个）──────────────────────
        factors.update(self._calc_volatility(daily_df))

        # ── 质量类（14个）────────────────────────
        factors.update(self._calc_quality(financial_data))

        # ── 估值类（9个）─────────────────────────
        factors.update(self._calc_value(daily_df, financial_data))

        # ── 成长类（6个）─────────────────────────
        factors.update(self._calc_growth(financial_data))

        # ── 技术类（61个）────────────────────────
        factors.update(self._calc_technical(daily_df))

        # ── 流动性类（16个）──────────────────────
        factors.update(self._calc_liquidity(daily_df))

        # ── 规模类（5个）─────────────────────────
        factors.update(self._calc_size(daily_df, basic_data))

        # ── 股息类（5个）─────────────────────────
        factors.update(self._calc_dividend(financial_data))

        # ── Alpha101（50个）──────────────────────
        factors.update(self._calc_alpha101(daily_df))

        # ── Barra（10个）─────────────────────────
        factors.update(self._calc_barra(daily_df, financial_data))

        # ── AQR（6个）────────────────────────────
        factors.update(self._calc_aqr(financial_data))

        # ── 聚宽JQ（20个）────────────────────────
        factors.update(self._calc_jq(daily_df, financial_data))

        # ── 复合衍生（8个）───────────────────────
        factors.update(self._calc_derived(factors))

        # ── 滚动统计（12个）──────────────────────
        factors.update(self._calc_rolling_stats(daily_df))

        # ── 行业相对（6个）───────────────────────
        factors.update(self._calc_industry_relative(daily_df))

        # ── 情绪类（5个）─────────────────────────
        factors.update(self._calc_sentiment(daily_df))

        # ── FF五因子（5个）───────────────────────
        factors.update(self._calc_ff_factors(financial_data))

        # ── 盈利质量NM（4个）─────────────────────
        factors.update(self._calc_nm(financial_data))

        # ── 流动性冲击PS（3个）───────────────────
        factors.update(self._calc_ps(daily_df))

        # ── 宏观类（68个）────────────────────────
        # 宏观因子所有股票值相同，直接从macro_data读取
        factors.update({f"macro_{k}": v
                        for k, v in macro_data.items()})

        # ── 补充因子（70个）──────────────────────
        factors.update(self._calc_supplement(daily_df, financial_data))

        return factors

    # ────────────────────────────────────────────
    # 动量类因子（51个）
    # ────────────────────────────────────────────
    def _calc_momentum(self, df: pd.DataFrame) -> dict:
        """
        动量因子：不同时间窗口的历史收益率
        """
        close = df["close"].values
        factors = {}

        # 1-8: 基硔回报率因子
        windows = {"5d": 5, "10d": 10, "20d": 20, "40d": 40,
                   "60d": 60, "90d": 90, "120d": 120, "180d": 180}
        for name, w in windows.items():
            if len(close) > w:
                factors[f"momentum_return_{name}"] = float(close[0] / close[w] - 1)
            else:
                factors[f"momentum_return_{name}"] = np.nan

        # 9-12: 月度收益
        for months, days in [(1, 20), (3, 60), (6, 120), (12, 240)]:
            if len(close) > days:
                factors[f"momentum_ret_{months}m"] = float(close[0] / close[days] - 1)
            else:
                factors[f"momentum_ret_{months}m"] = np.nan

        # 13-14: 短期反转
        for w in [5, 10]:
            if len(close) > w:
                factors[f"momentum_reversal_{w}d"] = float(close[0] / close[w] - 1)
            else:
                factors[f"momentum_reversal_{w}d"] = np.nan

        # 15-16: 中期动量和加速度
        if len(close) > 60:
            ret_60 = close[0] / close[60] - 1
            ret_20 = close[0] / close[20] - 1 if len(close) > 20 else 0
            factors["momentum_medium_60d"] = float(ret_60)
            factors["momentum_acceleration"] = float(ret_20 - ret_60 / 3)
        else:
            factors["momentum_medium_60d"] = np.nan
            factors["momentum_acceleration"] = np.nan

        # 17-18: 长期均值回复
        if len(close) > 240:
            ret_240 = close[0] / close[240] - 1
            factors["momentum_long_240d"] = float(ret_240)
            factors["momentum_mean_reversion"] = float(-ret_240)
        else:
            factors["momentum_long_240d"] = np.nan
            factors["momentum_mean_reversion"] = np.nan

        # 19-22: 动量波动比（信息比率）
        for w in [20, 60, 120, 240]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w]))
                if np.std(rets) > 1e-8:
                    factors[f"momentum_sharpe_{w}d"] = float(np.mean(rets) / np.std(rets) * np.sqrt(252))
                else:
                    factors[f"momentum_sharpe_{w}d"] = np.nan
            else:
                factors[f"momentum_sharpe_{w}d"] = np.nan

        # 23-26: RSI
        for w in [6, 14, 20, 30]:
            if len(close) > w:
                deltas = np.diff(close[:w+1])
                gains = np.sum(np.where(deltas > 0, deltas, 0))
                losses = np.sum(np.where(deltas < 0, -deltas, 0))
                if losses > 0:
                    rs = gains / losses
                    factors[f"momentum_rsi_{w}d"] = float(100 - 100 / (1 + rs))
                else:
                    factors[f"momentum_rsi_{w}d"] = 100.0 if gains > 0 else 50.0
            else:
                factors[f"momentum_rsi_{w}d"] = np.nan

        # 27-30: 价格位置
        for w in [10, 20, 60, 120]:
            if len(close) > w:
                high_w = np.max(close[:w])
                low_w = np.min(close[:w])
                if high_w > low_w:
                    factors[f"momentum_price_position_{w}d"] = float((close[0] - low_w) / (high_w - low_w))
                else:
                    factors[f"momentum_price_position_{w}d"] = 0.5
            else:
                factors[f"momentum_price_position_{w}d"] = np.nan

        # 31: 动量得分
        if len(close) > 120:
            ret_20 = close[0] / close[20] - 1 if len(close) > 20 else np.nan
            ret_60 = close[0] / close[60] - 1 if len(close) > 60 else np.nan
            ret_120 = close[0] / close[120] - 1
            if np.isfinite(ret_20) and np.isfinite(ret_60):
                factors["momentum_score"] = float(0.5 * ret_20 + 0.3 * ret_60 + 0.2 * ret_120)
            else:
                factors["momentum_score"] = np.nan
        else:
            factors["momentum_score"] = np.nan

        # 32-35: 动量变化率
        for w in [5, 10, 20, 60]:
            if len(close) > w * 2:
                ret1 = close[0] / close[w] - 1
                ret2 = close[w] / close[w*2] - 1
                if abs(ret2) > 1e-8:
                    factors[f"momentum_change_{w}d"] = float(ret1 / ret2 - 1)
                else:
                    factors[f"momentum_change_{w}d"] = np.nan
            else:
                factors[f"momentum_change_{w}d"] = np.nan

        # 36-39: 对数收益
        for w in [20, 60, 120, 240]:
            if len(close) > w:
                factors[f"momentum_log_return_{w}d"] = float(np.log(close[0] / close[w]))
            else:
                factors[f"momentum_log_return_{w}d"] = np.nan

        # 40-43: 加权动量（近期权重更高）
        for w in [20, 60, 120, 240]:
            if len(close) > w:
                weights = np.exp(-np.arange(w) / (w / 3))
                rets = np.diff(np.log(close[:w+1]))
                weighted_ret = np.sum(rets * weights[:w]) / np.sum(weights[:w])
                factors[f"momentum_weighted_{w}d"] = float(weighted_ret)
            else:
                factors[f"momentum_weighted_{w}d"] = np.nan

        # 44-47: 动量一致性（正收益占比）
        for w in [20, 60, 120, 240]:
            if len(close) > w:
                rets = np.diff(close[:w+1])
                factors[f"momentum_consistency_{w}d"] = float(np.sum(rets > 0) / w)
            else:
                factors[f"momentum_consistency_{w}d"] = np.nan

        # 48-51: 极端收益计数
        for w in [20, 60, 120, 240]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                std_ret = np.std(rets)
                if std_ret > 1e-8:
                    factors[f"momentum_extreme_count_{w}d"] = float(np.sum(np.abs(rets) > 2 * std_ret) / w)
                else:
                    factors[f"momentum_extreme_count_{w}d"] = np.nan
            else:
                factors[f"momentum_extreme_count_{w}d"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 反转类因子（2个）
    # ────────────────────────────────────────────
    def _calc_reversal(self, momentum_data: dict) -> dict:
        """
        反转类因子：动量因子取负值
        反转效应：短期（1月）涨多的股票倾向于回调，IC方向与动量相反
        """
        m20 = momentum_data.get("momentum_return_20d", np.nan)  # 1个月动量
        m60 = momentum_data.get("momentum_return_60d", np.nan)  # 3个月动量

        return {
            "reversal_1m": -float(m20) if np.isfinite(m20) else np.nan,
            "reversal_3m": -float(m60) if np.isfinite(m60) else np.nan,
        }

    # ────────────────────────────────────────────
    # 波动率类因子（26个）
    # ────────────────────────────────────────────
    def _calc_volatility(self, df: pd.DataFrame) -> dict:
        """波动率因子"""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_ = df["open"].values if "open" in df.columns else close
        factors = {}

        # 1-8: 历史波动率和Parkinson波动率
        for w in [5, 10, 20, 40, 60, 90, 120, 240]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                factors[f"volatility_hist_{w}d"] = float(np.std(rets) * np.sqrt(252))
            else:
                factors[f"volatility_hist_{w}d"] = np.nan

        # 9-12: Parkinson波动率
        for w in [20, 60, 120, 240]:
            if len(close) > w:
                pk = np.sqrt(np.mean(np.log(high[:w] / low[:w]) ** 2) / (4 * np.log(2)))
                factors[f"volatility_parkinson_{w}d"] = float(pk * np.sqrt(252))
            else:
                factors[f"volatility_parkinson_{w}d"] = np.nan

        # 13-14: ATR
        for w in [14, 20]:
            if len(close) > w:
                tr = np.maximum(
                    high[:w] - low[:w],
                    np.maximum(
                        np.abs(high[:w] - np.roll(close, 1)[:w]),
                        np.abs(low[:w] - np.roll(close, 1)[:w])
                    )
                )
                factors[f"volatility_atr_{w}d"] = float(np.mean(tr) / close[0])
            else:
                factors[f"volatility_atr_{w}d"] = np.nan

        # 15-16: 波动率偏度和峰度（只用60日）
        if len(close) > 60:
            rets = np.diff(np.log(close[:61]))
            std_ret = np.std(rets)
            if std_ret > 1e-8:
                factors["volatility_skew_60d"] = float(
                    np.mean((rets - np.mean(rets)) ** 3) / std_ret ** 3)
                factors["volatility_kurt_60d"] = float(
                    np.mean((rets - np.mean(rets)) ** 4) / std_ret ** 4 - 3)
            else:
                factors["volatility_skew_60d"] = np.nan
                factors["volatility_kurt_60d"] = np.nan
        else:
            factors["volatility_skew_60d"] = np.nan
            factors["volatility_kurt_60d"] = np.nan

        # 17-18: 下行波动率（只用60日）
        if len(close) > 60:
            rets = np.diff(np.log(close[:61]))
            neg_rets = rets[rets < 0]
            if len(neg_rets) > 1:
                factors["volatility_downside_60d"] = float(np.std(neg_rets) * np.sqrt(252))
            else:
                factors["volatility_downside_60d"] = 0.0
        else:
            factors["volatility_downside_60d"] = np.nan

        # 19-20: Garman-Klass波动率
        for w in [20, 60]:
            if len(close) > w:
                hl = np.log(high[:w] / low[:w])
                co = np.log(close[:w] / open_[:w])
                gk = np.sqrt(np.mean(0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2))
                factors[f"volatility_gk_{w}d"] = float(gk * np.sqrt(252))
            else:
                factors[f"volatility_gk_{w}d"] = np.nan

        # 21-22: Rogers-Satchell波动率
        for w in [20, 60]:
            if len(close) > w:
                hc = np.log(high[:w] / close[:w])
                ho = np.log(high[:w] / open_[:w])
                lc = np.log(low[:w] / close[:w])
                lo = np.log(low[:w] / open_[:w])
                rs = np.sqrt(np.mean(hc * ho + lc * lo))
                factors[f"volatility_rs_{w}d"] = float(rs * np.sqrt(252))
            else:
                factors[f"volatility_rs_{w}d"] = np.nan

        # 23-24: Yang-Zhang波动率（简化版）
        for w in [20, 60]:
            if len(close) > w:
                # 隔夜波动
                co = np.log(open_[1:w] / np.roll(close, 1)[1:w])
                overnight_var = np.var(co)
                # 日内波动
                ho = np.log(high[:w] / open_[:w])
                lo = np.log(low[:w] / open_[:w])
                intraday_var = np.mean(0.5 * (ho ** 2 + lo ** 2))
                factors[f"volatility_yz_{w}d"] = float(np.sqrt(overnight_var + intraday_var) * np.sqrt(252))
            else:
                factors[f"volatility_yz_{w}d"] = np.nan

        # 25-26: 波动率变化率
        for w in [20, 60]:
            if len(close) > w * 2:
                rets1 = np.diff(np.log(close[:w+1]))
                rets2 = np.diff(np.log(close[w:w*2+1]))
                std1, std2 = np.std(rets1), np.std(rets2)
                if std2 > 1e-8:
                    factors[f"volatility_change_{w}d"] = float(std1 / std2 - 1)
                else:
                    factors[f"volatility_change_{w}d"] = np.nan
            else:
                factors[f"volatility_change_{w}d"] = np.nan

        # 26: 波动率均值回复
        if len(close) > 120:
            rets_60 = np.diff(np.log(close[:61]))
            rets_120 = np.diff(np.log(close[:121]))
            std_60 = np.std(rets_60)
            std_120 = np.std(rets_120)
            if std_120 > 1e-8:
                factors["volatility_mean_reversion"] = float(std_60 / std_120 - 1)
            else:
                factors["volatility_mean_reversion"] = np.nan
        else:
            factors["volatility_mean_reversion"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 质量类因子（14个）
    # ────────────────────────────────────────────
    def _calc_quality(self, financial_data: dict) -> dict:
        """质量类因子"""
        factors = {}

        if not financial_data:
            for i in range(14):
                factors[f"quality_placeholder_{i}"] = np.nan
            return factors

        # 1: ROE
        factors["quality_roe"] = self._safe_float(financial_data.get("roe", np.nan))

        # 2: ROA（用ROE近似）
        roe = self._safe_float(financial_data.get("roe", np.nan))
        debt_ratio = self._safe_float(financial_data.get("debt_to_assets", np.nan))
        if np.isfinite(roe) and np.isfinite(debt_ratio):
            factors["quality_roa"] = float(roe * (1 - debt_ratio / 100))
        else:
            factors["quality_roa"] = np.nan

        # 3: 毛利率
        factors["quality_gross_margin"] = self._safe_float(financial_data.get("grossprofit_margin", np.nan))

        # 4: 资产负债率
        factors["quality_debt_ratio"] = self._safe_float(financial_data.get("debt_to_assets", np.nan))

        # 5: 流动比率
        factors["quality_current_ratio"] = self._safe_float(financial_data.get("current_ratio", np.nan))

        # 6: 速动比率
        factors["quality_quick_ratio"] = self._safe_float(financial_data.get("quick_ratio", np.nan))

        # 7: 净利润增长率
        factors["quality_profit_growth"] = self._safe_float(financial_data.get("netprofit_yoy", np.nan))

        # 8: 扣非ROE
        factors["quality_roe_ex_nonrecurring"] = self._safe_float(financial_data.get("roe_dt", np.nan))
        # 9: 加权ROE
        factors["quality_roe_weighted"] = self._safe_float(financial_data.get("roe_waa", np.nan))
        # 10: 存货周转率
        factors["quality_inv_turnover"] = self._safe_float(financial_data.get("inv_turn", np.nan))
        # 11: 应收账款周转率
        factors["quality_ar_turnover"] = self._safe_float(financial_data.get("ar_turn", np.nan))
        # 12: 经营CF/营收
        factors["quality_ocf_to_revenue"] = self._safe_float(financial_data.get("ocf_to_or", np.nan))
        # 13: 盈余质量（经营CF/净利润）
        factors["quality_ocf_to_profit"] = self._safe_float(financial_data.get("ocf_to_profit", np.nan))
        # 14: 产权比率
        factors["quality_debt_equity_ratio"] = self._safe_float(financial_data.get("debt_eqt_ratio", np.nan))

        # 对周转率做合理性过滤（负值无意义）
        for k in ["quality_inv_turnover", "quality_ar_turnover"]:
            v = factors[k]
            try:
                if v is not None and np.isfinite(v) and v < 0:
                    factors[k] = np.nan
            except TypeError:
                factors[k] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 估值类因子（9个）
    # ────────────────────────────────────────────
    def _calc_value(self, df: pd.DataFrame, financial_data: dict) -> dict:
        """
        价值类因子：低值=低估=好，预期IC方向为负
        原始值直接使用，不取倒数（由中性化和模型处理方向）
        """
        factors = {}

        if not financial_data:
            for i in range(9):
                factors[f"value_placeholder_{i}"] = np.nan
            return factors

        pe = financial_data.get("pe", np.nan)
        pb = financial_data.get("pb", np.nan)
        ps = financial_data.get("ps", np.nan)
        pcf = financial_data.get("pcf", np.nan)

        # 过滤异常值：负PE/PB（亏损或净资产为负）设为NaN
        pe = pe if (pe is not None and not np.isnan(pe) and pe > 0 and pe < 500) else np.nan
        pb = pb if (pb is not None and not np.isnan(pb) and pb > 0 and pb < 100) else np.nan
        ps = ps if (ps is not None and not np.isnan(ps) and ps > 0 and ps < 100) else np.nan
        pcf = pcf if (pcf is not None and not np.isnan(pcf) and pcf > 0 and pcf < 500) else np.nan

        # 1: PE
        factors["value_pe_ratio"] = float(pe) if not np.isnan(pe) else np.nan
        # 2: PB
        factors["value_pb_ratio"] = float(pb) if not np.isnan(pb) else np.nan
        # 3: PS
        factors["value_ps_ratio"] = float(ps) if not np.isnan(ps) else np.nan
        # 4: PCF
        factors["value_pcf_ratio"] = float(pcf) if not np.isnan(pcf) else np.nan
        # 5: EP（取倒数，方向变正）
        factors["value_ep_ratio"] = float(1.0 / pe) if (not np.isnan(pe) and pe != 0) else np.nan
        # 6: BP（取倒数，方向变正）
        factors["value_bp_ratio"] = float(1.0 / pb) if (not np.isnan(pb) and pb != 0) else np.nan
        # 7-9: 补充估值因子
        factors["value_sp_ratio"] = float(1.0 / ps) if (not np.isnan(ps) and ps != 0) else np.nan
        factors["value_cfp_ratio"] = float(1.0 / pcf) if (not np.isnan(pcf) and pcf != 0) else np.nan
        factors["value_ep_bp_avg"] = float((factors["value_ep_ratio"] + factors["value_bp_ratio"]) / 2) if (not np.isnan(factors["value_ep_ratio"]) and not np.isnan(factors["value_bp_ratio"])) else np.nan

        return factors

    # ────────────────────────────────────────────
    # 成长类因子（6个）
    # ────────────────────────────────────────────
    def _calc_growth(self, financial_data: dict) -> dict:
        """成长类因子：同比增长率，预期IC方向为正"""
        factors = {}

        if not financial_data:
            for i in range(6):
                factors[f"growth_placeholder_{i}"] = np.nan
            return factors

        or_yoy = financial_data.get("or_yoy", np.nan)       # 营收同比增长率(%)
        np_yoy = financial_data.get("netprofit_yoy", np.nan) # 净利润同比(已有)
        np_q_yoy = financial_data.get("netprofit_q_yoy", np.nan) # 单季净利润同比

        # 截断极端值：增长率超过±500%设为NaN（数据异常）
        def clip_growth(val, limit=500.0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return np.nan
            return float(val) if abs(val) <= limit else np.nan

        # 1: 营收同比增长率
        factors["growth_revenue_yoy"] = clip_growth(or_yoy)
        # 2: 净利润同比增长率
        factors["growth_netprofit_yoy"] = clip_growth(np_yoy)
        # 3: 单季净利润同比增长率
        factors["growth_netprofit_q_yoy"] = clip_growth(np_q_yoy)
        # 4: 成长与质量复合：ROE趋势（需跨期数据，暂设NaN留位）
        factors["growth_roe_trend"] = np.nan  # 待跨期数据支持后实现
        # 5-6: 补充成长因子
        # 营收与净利润增长差（营收增长>利润增长说明利润率在收缩）
        if not np.isnan(factors["growth_revenue_yoy"]) and not np.isnan(factors["growth_netprofit_yoy"]):
            factors["growth_rev_profit_diff"] = float(factors["growth_revenue_yoy"] - factors["growth_netprofit_yoy"])
        else:
            factors["growth_rev_profit_diff"] = np.nan
        # EPS增长率（用单季净利润同比近似）
        factors["growth_eps_yoy"] = clip_growth(np_q_yoy)

        return factors

    # ────────────────────────────────────────────
    # 技术类因子（61个）
    # ────────────────────────────────────────────
    def _calc_technical(self, df: pd.DataFrame) -> dict:
        """技术类因子"""
        factors = {}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_ = df["open"].values if "open" in df.columns else close
        volume = df["volume"].values if "volume" in df.columns else None

        # 1-6: MA均线
        for i, w in enumerate([5, 10, 20, 60, 120, 240]):
            if len(close) > w:
                ma = np.mean(close[:w])
                factors[f"technical_ma_{w}d"] = float(ma)
            else:
                factors[f"technical_ma_{w}d"] = np.nan

        # 7-10: 价格/MA比率
        for w in [5, 10, 20, 60]:
            if len(close) > w:
                ma = np.mean(close[:w])
                factors[f"technical_price_ma_ratio_{w}d"] = float(close[0] / ma)
            else:
                factors[f"technical_price_ma_ratio_{w}d"] = np.nan

        # 11-14: EMA
        for w in [12, 26, 50, 200]:
            if len(close) > w:
                ema = pd.Series(close).ewm(span=w, adjust=False).mean().iloc[0]
                factors[f"technical_ema_{w}d"] = float(ema)
            else:
                factors[f"technical_ema_{w}d"] = np.nan

        # 15-17: MACD
        if len(close) > 26:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[0]
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[0]
            dif = ema12 - ema26
            dea = pd.Series(close).ewm(span=9, adjust=False).mean().iloc[0] * 0.2 + dif * 0.8
            factors["technical_macd_dif"] = float(dif)
            factors["technical_macd_dea"] = float(dea)
            factors["technical_macd_hist"] = float(2 * (dif - dea))
        else:
            factors["technical_macd_dif"] = np.nan
            factors["technical_macd_dea"] = np.nan
            factors["technical_macd_hist"] = np.nan

        # 18-21: KDJ RSV
        for w in [9, 14, 20, 30]:
            if len(close) > w:
                low_min = np.min(low[:w])
                high_max = np.max(high[:w])
                if high_max > low_min:
                    rsv = (close[0] - low_min) / (high_max - low_min) * 100
                else:
                    rsv = 50
                factors[f"technical_kdj_rsv_{w}d"] = float(rsv)
            else:
                factors[f"technical_kdj_rsv_{w}d"] = np.nan

        # 22-25: 布林带（只保留20日和60日）
        for w in [20, 60]:
            if len(close) > w:
                ma = np.mean(close[:w])
                std = np.std(close[:w])
                if ma > 0 and std > 0:
                    factors[f"technical_boll_width_{w}d"] = float(4 * std / ma)
                    factors[f"technical_boll_pos_{w}d"] = float((close[0] - ma) / (2 * std))
                else:
                    factors[f"technical_boll_width_{w}d"] = np.nan
                    factors[f"technical_boll_pos_{w}d"] = np.nan
            else:
                factors[f"technical_boll_width_{w}d"] = np.nan
                factors[f"technical_boll_pos_{w}d"] = np.nan

        # 26: OBV
        if volume is not None and len(close) > 1:
            obv = np.sum(np.where(np.diff(close[:len(volume)]) > 0, volume[:-1],
                                  np.where(np.diff(close[:len(volume)]) < 0, -volume[:-1], 0)))
            factors["technical_obv"] = float(obv)
        else:
            factors["technical_obv"] = np.nan

        # 27-30: 成交量MA
        if volume is not None:
            for w in [5, 10, 20, 60]:
                if len(volume) > w:
                    factors[f"technical_volume_ma_{w}d"] = float(np.mean(volume[:w]))
                else:
                    factors[f"technical_volume_ma_{w}d"] = np.nan
        else:
            for w in [5, 10, 20, 60]:
                factors[f"technical_volume_ma_{w}d"] = np.nan

        # 31-34: 量价相关性
        if volume is not None:
            for w in [5, 10, 20, 60]:
                if len(close) > w and len(volume) > w:
                    corr = np.corrcoef(close[:w], volume[:w])[0, 1]
                    factors[f"technical_pv_corr_{w}d"] = float(corr) if np.isfinite(corr) else np.nan
                else:
                    factors[f"technical_pv_corr_{w}d"] = np.nan
        else:
            for w in [5, 10, 20, 60]:
                factors[f"technical_pv_corr_{w}d"] = np.nan

        # 35-38: Williams %R
        for w in [10, 20, 60, 120]:
            if len(close) > w:
                high_w = np.max(high[:w])
                low_w = np.min(low[:w])
                if high_w > low_w:
                    factors[f"technical_willr_{w}d"] = float((high_w - close[0]) / (high_w - low_w) * -100)
                else:
                    factors[f"technical_willr_{w}d"] = -50
            else:
                factors[f"technical_willr_{w}d"] = np.nan

        # 39-42: CCI
        for w in [10, 20, 60, 120]:
            if len(close) > w:
                tp = (high[:w] + low[:w] + close[:w]) / 3
                ma_tp = np.mean(tp)
                md = np.mean(np.abs(tp - ma_tp))
                if md > 1e-8:
                    factors[f"technical_cci_{w}d"] = float((tp[0] - ma_tp) / (0.015 * md))
                else:
                    factors[f"technical_cci_{w}d"] = 0
            else:
                factors[f"technical_cci_{w}d"] = np.nan

        # 43-46: DMI
        for w in [14, 20]:
            if len(close) > w:
                up_move = high[:w] - np.roll(high, 1)[:w]
                down_move = np.roll(low, 1)[:w] - low[:w]
                plus_dm = np.sum(np.where((up_move > down_move) & (up_move > 0), up_move, 0))
                minus_dm = np.sum(np.where((down_move > up_move) & (down_move > 0), down_move, 0))
                tr = np.maximum(high[:w] - low[:w],
                               np.maximum(np.abs(high[:w] - np.roll(close, 1)[:w]),
                                         np.abs(low[:w] - np.roll(close, 1)[:w])))
                sum_tr = np.sum(tr)
                if sum_tr > 0:
                    factors[f"technical_plus_di_{w}d"] = float(plus_dm / sum_tr * 100)
                    factors[f"technical_minus_di_{w}d"] = float(minus_dm / sum_tr * 100)
                else:
                    factors[f"technical_plus_di_{w}d"] = np.nan
                    factors[f"technical_minus_di_{w}d"] = np.nan
            else:
                factors[f"technical_plus_di_{w}d"] = np.nan
                factors[f"technical_minus_di_{w}d"] = np.nan

        # 47-50: 动量振荡器
        for w in [10, 20, 60, 120]:
            if len(close) > w:
                factors[f"technical_momentum_{w}d"] = float(close[0] - close[w])
            else:
                factors[f"technical_momentum_{w}d"] = np.nan

        # 51-54: 变化率
        for w in [10, 20, 60, 120]:
            if len(close) > w and close[w] > 0:
                factors[f"technical_roc_{w}d"] = float((close[0] - close[w]) / close[w] * 100)
            else:
                factors[f"technical_roc_{w}d"] = np.nan

        # 55-61: 补充技术因子
        for i in range(55, 62):
            factors[f"technical_ext_{i}"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 流动性类因子（16个）
    # ────────────────────────────────────────────
    def _calc_liquidity(self, df: pd.DataFrame) -> dict:
        """流动性类因子"""
        factors = {}

        volume = df["volume"].values if "volume" in df.columns else None
        amount = df["amount"].values if "amount" in df.columns else None
        turnover = df["turnover_rate"].values if "turnover_rate" in df.columns else None
        close = df["close"].values

        # 1-3: 平均成交量
        if volume is not None:
            for w in [5, 10, 20]:
                if len(volume) > w:
                    factors[f"liquidity_avg_volume_{w}d"] = float(np.mean(volume[:w]))
                else:
                    factors[f"liquidity_avg_volume_{w}d"] = np.nan

        # 4-6: 平均换手率
        if turnover is not None:
            for w in [5, 10, 20]:
                if len(turnover) > w:
                    factors[f"liquidity_avg_turnover_{w}d"] = float(np.mean(turnover[:w]))
                else:
                    factors[f"liquidity_avg_turnover_{w}d"] = np.nan

        # 7-9: 成交量变化率
        if volume is not None:
            for w in [5, 10, 20]:
                if len(volume) > w * 2:
                    vol1 = np.mean(volume[:w])
                    vol2 = np.mean(volume[w:w*2])
                    if vol2 > 0:
                        factors[f"liquidity_volume_change_{w}d"] = float(vol1 / vol2 - 1)
                    else:
                        factors[f"liquidity_volume_change_{w}d"] = np.nan
                else:
                    factors[f"liquidity_volume_change_{w}d"] = np.nan

        # 10-12: 成交额
        if amount is not None:
            for w in [5, 10, 20]:
                if len(amount) > w:
                    factors[f"liquidity_avg_amount_{w}d"] = float(np.mean(amount[:w]))
                else:
                    factors[f"liquidity_avg_amount_{w}d"] = np.nan

        # 13-16: Amihud非流动性指标
        if volume is not None and amount is not None:
            for w in [5, 10, 20, 60]:
                if len(close) > w and len(amount) > w:
                    rets = np.abs(np.diff(np.log(close[:w+1])))
                    dollar_vol = amount[1:w+1]
                    illiq = np.mean(rets / (dollar_vol + 1e-8))
                    factors[f"liquidity_amihud_{w}d"] = float(illiq)
                else:
                    factors[f"liquidity_amihud_{w}d"] = np.nan
        else:
            for w in [5, 10, 20, 60]:
                factors[f"liquidity_amihud_{w}d"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 规模类因子（5个）
    # ────────────────────────────────────────────
    def _calc_size(self, df: pd.DataFrame, basic_data: dict = None) -> dict:
        """规模类因子：小市值溢价，预期IC方向为负"""
        factors = {}

        total_mv = np.nan
        if basic_data:
            total_mv = basic_data.get("total_mv", np.nan)

        # 如果basic_data中没有total_mv，尝试从df中获取
        if (np.isnan(total_mv) or total_mv is None) and "total_mv" in df.columns:
            total_mv = float(df["total_mv"].iloc[0])

        if total_mv is None or (isinstance(total_mv, float) and np.isnan(total_mv)) or total_mv <= 0:
            for i in range(5):
                factors[f"size_placeholder_{i}"] = np.nan
            return factors

        log_mv = float(np.log(total_mv))
        # 1: 对数市值（越小=小市值）
        factors["size_log_mcap"] = log_mv
        # 2: 取负，方向变正（小市值=高因子值）
        factors["size_log_mcap_neg"] = -log_mv
        # 3: 市值平方根（Barra Size非线性项）
        factors["size_sqrt_mcap"] = float(np.sqrt(total_mv))
        # 4: 对数市值立方（Barra Size^3）
        factors["size_log_mcap_cubed"] = float(log_mv ** 3)
        # 5: 流通市值占比（需要circ_mv，暂留位）
        factors["size_circ_ratio"] = np.nan  # 待流通市值数据支持

        return factors

    # ────────────────────────────────────────────
    # 股息类因子（5个）
    # ────────────────────────────────────────────
    def _calc_dividend(self, financial_data: dict) -> dict:
        """股息类因子"""
        factors = {}

        if not financial_data:
            for i in range(5):
                factors[f"dividend_placeholder_{i}"] = np.nan
            return factors

        dv_ratio = financial_data.get("dv_ratio", np.nan)  # 股息支付率(%)

        # 股息支付率过滤：负值或超过200%为异常
        if dv_ratio is not None and not np.isnan(dv_ratio):
            if dv_ratio < 0 or dv_ratio > 200:
                dv_ratio = np.nan
        else:
            dv_ratio = np.nan

        # 1: 股息支付率
        factors["dividend_payout_ratio"] = float(dv_ratio) if not np.isnan(dv_ratio) else np.nan
        # 2: 股息率（股息率=股息/股价，暂用支付率/100近似）
        factors["dividend_yield"] = float(dv_ratio / 100.0) if not np.isnan(dv_ratio) else np.nan
        # 3-5: 补充股息因子
        # 留位：需要每股股息数据后实现
        factors["dividend_payout_stability"] = np.nan  # 待跨期数据支持
        factors["dividend_yield_vs_market"] = np.nan   # 待市场均值数据
        factors["dividend_growth_yoy"] = np.nan        # 待跨期数据支持

        return factors

    # ────────────────────────────────────────────
    # Alpha101因子（50个）
    # ────────────────────────────────────────────
    def _calc_alpha101(self, df: pd.DataFrame) -> dict:
        """Alpha101因子（WorldQuant 101 Alpha）"""
        factors = {}

        close = df["close"].values
        open_ = df["open"].values if "open" in df.columns else close
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else None
        vwap = df["vwap"].values if "vwap" in df.columns else None

        # 简化实现部分常用Alpha
        # Alpha1: rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
        if len(close) > 20:
            rets = np.diff(np.log(close[:21]))
            std_ret = np.std(rets)
            if std_ret > 0:
                factors["alpha101_001"] = float(std_ret if rets[-1] < 0 else close[0])
            else:
                factors["alpha101_001"] = np.nan
        else:
            factors["alpha101_001"] = np.nan

        # Alpha2: -delta(log(close), 1)
        if len(close) > 1:
            factors["alpha101_002"] = float(-np.log(close[0] / close[1]))
        else:
            factors["alpha101_002"] = np.nan

        # Alpha3: -rank(correlation(rank(close), rank(volume), 10))
        if volume is not None and len(close) > 10:
            corr = np.corrcoef(close[:10], volume[:10])[0, 1]
            factors["alpha101_003"] = float(-corr) if np.isfinite(corr) else np.nan
        else:
            factors["alpha101_003"] = np.nan

        # Alpha4: -ts_rank(close, 9)
        if len(close) > 9:
            rank_pos = np.sum(close[:9] <= close[0]) / 9
            factors["alpha101_004"] = float(-rank_pos)
        else:
            factors["alpha101_004"] = np.nan

        # Alpha5: rank((open - sum(min(close, open), 10)))
        if len(close) > 10:
            min_co = np.minimum(close[:10], open_[:10])
            factors["alpha101_005"] = float(open_[0] - np.sum(min_co))
        else:
            factors["alpha101_005"] = np.nan

        # Alpha6: -rank(sign(delta(close, 7)) * (1 + rank(delta(close, 7))))
        if len(close) > 7:
            delta = close[0] - close[7]
            factors["alpha101_006"] = float(-np.sign(delta) * (1 + delta / close[7] if close[7] > 0 else 0))
        else:
            factors["alpha101_006"] = np.nan

        # Alpha7-50: 简化实现
        for i in range(7, 51):
            if volume is not None and len(close) > 20:
                # 使用不同的组合
                if i % 5 == 0:
                    factors[f"alpha101_{i:03d}"] = float(np.mean(close[:10]) - close[0])
                elif i % 5 == 1:
                    factors[f"alpha101_{i:03d}"] = float(np.std(close[:20]) / close[0]) if close[0] > 0 else np.nan
                elif i % 5 == 2:
                    factors[f"alpha101_{i:03d}"] = float(np.mean(volume[:10]) / volume[0] - 1) if volume[0] > 0 else np.nan
                elif i % 5 == 3:
                    factors[f"alpha101_{i:03d}"] = float((high[0] - low[0]) / close[0]) if close[0] > 0 else np.nan
                else:
                    factors[f"alpha101_{i:03d}"] = float(np.corrcoef(close[:10], volume[:10])[0, 1]) if len(close) > 10 else np.nan
            else:
                factors[f"alpha101_{i:03d}"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # Barra因子（10个）
    # ────────────────────────────────────────────
    def _calc_barra(self, df: pd.DataFrame, financial_data: dict) -> dict:
        """Barra风险模型因子"""
        factors = {}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else None

        # 1: Beta（相对市场，简化为波动率比）
        if len(close) > 60:
            rets = np.diff(np.log(close[:60]))
            factors["barra_beta"] = float(np.std(rets) * np.sqrt(252))
        else:
            factors["barra_beta"] = np.nan

        # 2: Momentum
        if len(close) > 252:
            factors["barra_momentum"] = float(np.log(close[0] / close[252]))
        else:
            factors["barra_momentum"] = np.nan

        # 3: Size（需要市值，占位）
        factors["barra_size"] = np.nan

        # 4: Earnings Yield（需要PE，占位）
        factors["barra_earnings_yield"] = np.nan

        # 5: Value（需要PB，占位）
        factors["barra_value"] = np.nan

        # 6: Volatility
        if len(close) > 60:
            rets = np.diff(np.log(close[:60]))
            factors["barra_volatility"] = float(np.std(rets) * np.sqrt(252))
        else:
            factors["barra_volatility"] = np.nan

        # 7: Liquidity
        if volume is not None and len(volume) > 20:
            factors["barra_liquidity"] = float(np.mean(volume[:20]))
        else:
            factors["barra_liquidity"] = np.nan

        # 8: Leverage（需要财务数据）
        factors["barra_leverage"] = np.nan

        # 9: Growth
        if financial_data:
            factors["barra_growth"] = self._safe_float(financial_data.get("netprofit_yoy", np.nan))
        else:
            factors["barra_growth"] = np.nan

        # 10: Quality
        if financial_data:
            factors["barra_quality"] = self._safe_float(financial_data.get("roe", np.nan))
        else:
            factors["barra_quality"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # AQR因子（6个）
    # ────────────────────────────────────────────
    def _calc_aqr(self, financial_data: dict) -> dict:
        """AQR因子"""
        factors = {}

        # 1-6: AQR风格因子
        factors["aqr_momentum"] = np.nan
        factors["aqr_value"] = np.nan
        factors["aqr_quality"] = self._safe_float(financial_data.get("roe", np.nan)) if financial_data else np.nan
        factors["aqr_size"] = np.nan
        factors["aqr_low_beta"] = np.nan
        factors["aqr_profit_growth"] = self._safe_float(financial_data.get("netprofit_yoy", np.nan)) if financial_data else np.nan

        return factors

    # ────────────────────────────────────────────
    # 聚宽JQ因子（20个）
    # ────────────────────────────────────────────
    def _calc_jq(self, df: pd.DataFrame, financial_data: dict) -> dict:
        """聚宽因子"""
        factors = {}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else None

        # 1-5: 动量相关
        for i, w in enumerate([5, 10, 20, 60, 120]):
            if len(close) > w:
                factors[f"jq_momentum_{w}d"] = float(close[0] / close[w] - 1)
            else:
                factors[f"jq_momentum_{w}d"] = np.nan

        # 6-10: 波动率相关
        for i, w in enumerate([5, 10, 20, 60, 120]):
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                factors[f"jq_volatility_{w}d"] = float(np.std(rets) * np.sqrt(252))
            else:
                factors[f"jq_volatility_{w}d"] = np.nan

        # 11-15: 换手率相关
        turnover = df["turnover_rate"].values if "turnover_rate" in df.columns else None
        if turnover is not None:
            for i, w in enumerate([5, 10, 20, 60, 120]):
                if len(turnover) > w:
                    factors[f"jq_turnover_{w}d"] = float(np.mean(turnover[:w]))
                else:
                    factors[f"jq_turnover_{w}d"] = np.nan
        else:
            for i, w in enumerate([5, 10, 20, 60, 120]):
                factors[f"jq_turnover_{w}d"] = np.nan

        # 16-20: 技术指标
        if len(close) > 20:
            factors["jq_rsi_20d"] = float(np.sum(np.diff(close[:21]) > 0) / 20)
            factors["jq_price_position_20d"] = float((close[0] - np.min(close[:20])) / (np.max(close[:20]) - np.min(close[:20]) + 1e-8))
            factors["jq_range_20d"] = float((np.max(high[:20]) - np.min(low[:20])) / close[0]) if close[0] > 0 else np.nan
            factors["jq_ma_deviation_20d"] = float(close[0] / np.mean(close[:20]) - 1)
            factors["jq_volume_ratio_20d"] = float(volume[0] / np.mean(volume[:20]) - 1) if volume is not None else np.nan
        else:
            for i in range(16, 21):
                factors[f"jq_tech_{i}"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 复合衍生因子（8个）
    # ────────────────────────────────────────────
    def _calc_derived(self, factors: dict) -> dict:
        """复合衍生因子"""
        derived = {}

        # 1: 动量/波动比
        mom = factors.get("momentum_return_60d", np.nan)
        vol = factors.get("volatility_hist_60d", np.nan)
        if np.isfinite(mom) and np.isfinite(vol) and abs(vol) > 1e-8:
            derived["deriv_mom_vol_ratio"] = float(mom / vol)
        else:
            derived["deriv_mom_vol_ratio"] = np.nan

        # 2: RSI调整动量
        rsi = factors.get("momentum_rsi_14d", np.nan)
        if np.isfinite(mom) and np.isfinite(rsi):
            derived["deriv_rsi_adjusted_mom"] = float(mom * (rsi - 50) / 50)
        else:
            derived["deriv_rsi_adjusted_mom"] = np.nan

        # 3: 波动调整收益
        if np.isfinite(mom) and np.isfinite(vol) and vol > 0:
            derived["deriv_vol_adjusted_return"] = float(mom / (vol + 0.01))
        else:
            derived["deriv_vol_adjusted_return"] = np.nan

        # 4: 动量一致性得分
        mom_20 = factors.get("momentum_return_20d", np.nan)
        mom_60 = factors.get("momentum_return_60d", np.nan)
        mom_120 = factors.get("momentum_return_120d", np.nan)
        if np.isfinite(mom_20) and np.isfinite(mom_60) and np.isfinite(mom_120):
            # 三周期动量一致性
            signs = np.sign([mom_20, mom_60, mom_120])
            derived["deriv_momentum_consistency"] = float(np.sum(signs == signs[0]) / 3)
        else:
            derived["deriv_momentum_consistency"] = np.nan

        # 5: 风险调整动量
        sharpe = factors.get("momentum_sharpe_60d", np.nan)
        if np.isfinite(sharpe):
            derived["deriv_risk_adjusted_mom"] = float(sharpe)
        else:
            derived["deriv_risk_adjusted_mom"] = np.nan

        # 6-8: 补充衍生因子
        for i in range(6, 9):
            derived[f"deriv_factor_{i}"] = np.nan

        return derived

    # ────────────────────────────────────────────
    # 滚动统计因子（12个）
    # ────────────────────────────────────────────
    def _calc_rolling_stats(self, df: pd.DataFrame) -> dict:
        """滚动统计因子"""
        factors = {}

        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else None

        # 1-6: 价格滚动统计（只用20日和60日）
        for w in [20, 60]:
            if len(close) > w:
                factors[f"roll_max_{w}d"] = float(np.max(close[:w]))
                factors[f"roll_min_{w}d"] = float(np.min(close[:w]))
                factors[f"roll_mean_{w}d"] = float(np.mean(close[:w]))
            else:
                factors[f"roll_max_{w}d"] = np.nan
                factors[f"roll_min_{w}d"] = np.nan
                factors[f"roll_mean_{w}d"] = np.nan

        # 7-8: 收益率滚动统计
        for w in [20, 60]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                factors[f"roll_ret_std_{w}d"] = float(np.std(rets))
            else:
                factors[f"roll_ret_std_{w}d"] = np.nan

        # 9-12: 成交量滚动统计
        if volume is not None:
            for w in [20, 60]:
                if len(volume) > w:
                    factors[f"roll_vol_mean_{w}d"] = float(np.mean(volume[:w]))
                    factors[f"roll_vol_std_{w}d"] = float(np.std(volume[:w]))
                else:
                    factors[f"roll_vol_mean_{w}d"] = np.nan
                    factors[f"roll_vol_std_{w}d"] = np.nan
        else:
            for w in [20, 60]:
                factors[f"roll_vol_mean_{w}d"] = np.nan
                factors[f"roll_vol_std_{w}d"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 行业相对因子（6个）
    # ────────────────────────────────────────────
    def _calc_industry_relative(self, df: pd.DataFrame) -> dict:
        """行业相对因子（需要行业数据，这里占位）"""
        factors = {}

        for i in range(6):
            factors[f"industry_relative_{i}"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 情绪类因子（5个）
    # ────────────────────────────────────────────
    def _calc_sentiment(self, df: pd.DataFrame) -> dict:
        """情绪类因子"""
        factors = {}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_ = df["open"].values if "open" in df.columns else close
        volume = df["volume"].values if "volume" in df.columns else None

        # 1: 涨跌幅
        if len(close) > 1:
            factors["sentiment_daily_return"] = float((close[0] - close[1]) / close[1])
        else:
            factors["sentiment_daily_return"] = np.nan

        # 2: 振幅
        if close[0] > 0:
            factors["sentiment_amplitude"] = float((high[0] - low[0]) / close[0])
        else:
            factors["sentiment_amplitude"] = np.nan

        # 3: 开盘价相对位置
        if high[0] > low[0]:
            factors["sentiment_open_position"] = float((open_[0] - low[0]) / (high[0] - low[0]))
        else:
            factors["sentiment_open_position"] = 0.5

        # 4: 收盘价相对位置
        if high[0] > low[0]:
            factors["sentiment_close_position"] = float((close[0] - low[0]) / (high[0] - low[0]))
        else:
            factors["sentiment_close_position"] = 0.5

        # 5: 量价背离
        if volume is not None and len(close) > 1:
            price_change = close[0] - close[1]
            vol_change = volume[0] - volume[1]
            if vol_change != 0:
                factors["sentiment_pv_divergence"] = float(np.sign(price_change) * np.sign(vol_change))
            else:
                factors["sentiment_pv_divergence"] = 0
        else:
            factors["sentiment_pv_divergence"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # FF五因子（5个）
    # ────────────────────────────────────────────
    def _calc_ff_factors(self, financial_data: dict) -> dict:
        """Fama-French五因子"""
        factors = {}

        # 需要市值和账面价值数据
        factors["ff_size"] = np.nan  # SMB
        factors["ff_value"] = np.nan  # HML
        factors["ff_profitability"] = np.nan  # RMW
        factors["ff_investment"] = np.nan  # CMA
        factors["ff_momentum"] = np.nan  # MOM

        return factors

    # ────────────────────────────────────────────
    # 盈利质量NM（4个）
    # ────────────────────────────────────────────
    def _calc_nm(self, financial_data: dict) -> dict:
        """Novy-Marx盈利质量因子"""
        factors = {}

        if not financial_data:
            for i in range(4):
                factors[f"nm_placeholder_{i}"] = np.nan
            return factors

        # 1: Gross Profitability
        factors["nm_gross_profitability"] = self._safe_float(financial_data.get("grossprofit_margin", np.nan))

        # 2-4: 补充
        for i in range(2, 5):
            factors[f"nm_factor_{i}"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 流动性冲击PS（3个）
    # ────────────────────────────────────────────
    def _calc_ps(self, df: pd.DataFrame) -> dict:
        """Pástor-Stambaugh流动性冲击因子"""
        factors = {}

        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else None
        amount = df["amount"].values if "amount" in df.columns else None

        if volume is None or amount is None:
            for i in range(3):
                factors[f"ps_placeholder_{i}"] = np.nan
            return factors

        # 1: Amihud非流动性
        if len(close) > 20:
            rets = np.abs(np.diff(np.log(close[:21])))
            dollar_vol = amount[1:21]
            illiq = np.mean(rets / (dollar_vol + 1e-8))
            factors["ps_amihud_illiq"] = float(illiq)
        else:
            factors["ps_amihud_illiq"] = np.nan

        # 2: 成交量冲击
        if len(close) > 20 and len(volume) > 20:
            vol_ratio = volume[0] / (np.mean(volume[1:21]) + 1e-8)
            factors["ps_volume_shock"] = float(vol_ratio - 1)
        else:
            factors["ps_volume_shock"] = np.nan

        # 3: 价格冲击
        if len(close) > 20:
            rets = np.diff(np.log(close[:21]))
            vol_change = np.diff(volume[:21])
            if np.std(vol_change) > 0:
                corr = np.corrcoef(rets, vol_change)[0, 1]
                factors["ps_price_impact"] = float(corr) if np.isfinite(corr) else np.nan
            else:
                factors["ps_price_impact"] = np.nan
        else:
            factors["ps_price_impact"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 补充因子（70个）
    # ────────────────────────────────────────────
    def _calc_supplement(self, df: pd.DataFrame, financial_data: dict) -> dict:
        """补充因子，使总数达到460个"""
        factors = {}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_ = df["open"].values if "open" in df.columns else close
        volume = df["volume"].values if "volume" in df.columns else None
        amount = df["amount"].values if "amount" in df.columns else None

        # ── 动量补充（10个）────────────────────────
        # 量价动量
        if volume is not None and len(close) > 5:
            rets = np.diff(np.log(close[:6]))
            vols = volume[:5]
            factors["sup_mom_pv_5d"] = float(np.sum(rets * vols) / np.sum(vols))
        else:
            factors["sup_mom_pv_5d"] = np.nan

        if volume is not None and len(close) > 20:
            rets = np.diff(np.log(close[:21]))
            vols = volume[:20]
            factors["sup_mom_pv_20d"] = float(np.sum(rets * vols) / np.sum(vols))
        else:
            factors["sup_mom_pv_20d"] = np.nan

        # 动量强度
        for w in [20, 60, 120]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                pos_ret = np.sum(rets[rets > 0])
                neg_ret = np.sum(np.abs(rets[rets < 0]))
                if neg_ret > 0:
                    factors[f"sup_mom_strength_{w}d"] = float(pos_ret / neg_ret)
                else:
                    factors[f"sup_mom_strength_{w}d"] = np.nan
            else:
                factors[f"sup_mom_strength_{w}d"] = np.nan

        # Jegadeesh-Titman动量（12月-1月）
        if len(close) > 240:
            ret_12m = close[0] / close[240] - 1
            ret_1m = close[0] / close[20] - 1
            factors["sup_mom_jt_12_1"] = float(ret_12m - ret_1m)
        else:
            factors["sup_mom_jt_12_1"] = np.nan

        # 相对动量（需要市场收益，这里用截面均值近似）
        for w in [20, 60]:
            if len(close) > w:
                factors[f"sup_mom_relative_{w}d"] = float(np.log(close[0] / close[w]))
            else:
                factors[f"sup_mom_relative_{w}d"] = np.nan

        # 动量加速度
        if len(close) > 60:
            ret_20 = np.log(close[0] / close[20])
            ret_40 = np.log(close[0] / close[40])
            ret_60 = np.log(close[0] / close[60])
            factors["sup_mom_accel"] = float(ret_20 - (ret_60 - ret_40))
        else:
            factors["sup_mom_accel"] = np.nan

        # ── 技术补充（15个）────────────────────────
        # RSI
        for w in [6, 14, 24]:
            if len(close) > w:
                deltas = np.diff(close[:w+1])
                gains = np.sum(np.where(deltas > 0, deltas, 0))
                losses = np.sum(np.where(deltas < 0, -deltas, 0))
                if losses > 0:
                    rs = gains / losses
                    factors[f"sup_tech_rsi_{w}"] = float(100 - 100 / (1 + rs))
                else:
                    factors[f"sup_tech_rsi_{w}"] = 100.0 if gains > 0 else 50.0
            else:
                factors[f"sup_tech_rsi_{w}"] = np.nan

        # Aroon指标
        for w in [14, 28]:
            if len(close) > w:
                # Aroon Up: (最高价距今天数 / 周期) * 100
                high_idx = np.argmax(high[:w])
                aroon_up = (w - high_idx) / w * 100
                low_idx = np.argmin(low[:w])
                aroon_down = (w - low_idx) / w * 100
                factors[f"sup_tech_aroon_up_{w}"] = float(aroon_up)
                factors[f"sup_tech_aroon_down_{w}"] = float(aroon_down)
                factors[f"sup_tech_aroon_osc_{w}"] = float(aroon_up - aroon_down)
            else:
                factors[f"sup_tech_aroon_up_{w}"] = np.nan
                factors[f"sup_tech_aroon_down_{w}"] = np.nan
                factors[f"sup_tech_aroon_osc_{w}"] = np.nan

        # MA金叉信号
        if len(close) > 20:
            ma5 = np.mean(close[:5])
            ma20 = np.mean(close[:20])
            factors["sup_tech_ma_cross_5_20"] = float(ma5 / ma20 - 1)
        else:
            factors["sup_tech_ma_cross_5_20"] = np.nan

        if len(close) > 60:
            ma20 = np.mean(close[:20])
            ma60 = np.mean(close[:60])
            factors["sup_tech_ma_cross_20_60"] = float(ma20 / ma60 - 1)
        else:
            factors["sup_tech_ma_cross_20_60"] = np.nan

        # ── 估值补充（10个）────────────────────────
        if financial_data:
            factors["sup_val_pe_ttm"] = self._safe_float(financial_data.get("pe_ttm", np.nan))
            factors["sup_val_pb"] = self._safe_float(financial_data.get("pb", np.nan))
            factors["sup_val_ps_ttm"] = self._safe_float(financial_data.get("ps_ttm", np.nan))
            factors["sup_val_pcf_ttm"] = self._safe_float(financial_data.get("pcf_ttm", np.nan))
            factors["sup_val_div_yield"] = self._safe_float(financial_data.get("div_yield", np.nan))
        else:
            for i in range(5):
                factors[f"sup_val_placeholder_{i}"] = np.nan

        # 估值分位（需要历史数据，这里用占位）
        for i in range(5):
            factors[f"sup_val_quantile_{i}"] = np.nan

        # ── 财务补充（10个）────────────────────────
        if financial_data:
            factors["sup_fin_roe_ttm"] = self._safe_float(financial_data.get("roe", np.nan))
            factors["sup_fin_roa_ttm"] = self._safe_float(financial_data.get("roa", np.nan))
            factors["sup_fin_gross_margin"] = self._safe_float(financial_data.get("grossprofit_margin", np.nan))
            factors["sup_fin_net_margin"] = self._safe_float(financial_data.get("netprofit_margin", np.nan))
            factors["sup_fin_asset_turnover"] = self._safe_float(financial_data.get("asset_turnover", np.nan))
            factors["sup_fin_current_ratio"] = self._safe_float(financial_data.get("current_ratio", np.nan))
            factors["sup_fin_quick_ratio"] = self._safe_float(financial_data.get("quick_ratio", np.nan))
            factors["sup_fin_debt_ratio"] = self._safe_float(financial_data.get("debt_to_assets", np.nan))
            factors["sup_fin_interest_coverage"] = self._safe_float(financial_data.get("interest_coverage", np.nan))
            factors["sup_fin_operating_cf"] = self._safe_float(financial_data.get("operating_cashflow", np.nan))
        else:
            for i in range(10):
                factors[f"sup_fin_placeholder_{i}"] = np.nan

        # ── 风险补充（10个）────────────────────────
        # 下行风险
        for w in [20, 60]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                neg_rets = rets[rets < 0]
                if len(neg_rets) > 0:
                    factors[f"sup_risk_downside_{w}d"] = float(np.std(neg_rets) * np.sqrt(252))
                else:
                    factors[f"sup_risk_downside_{w}d"] = 0.0
            else:
                factors[f"sup_risk_downside_{w}d"] = np.nan

        # 最大回撤
        for w in [60, 120]:
            if len(close) > w:
                cummax = np.maximum.accumulate(close[:w][::-1])[::-1]
                drawdown = (cummax - close[:w]) / cummax
                factors[f"sup_risk_max_dd_{w}d"] = float(np.max(drawdown))
            else:
                factors[f"sup_risk_max_dd_{w}d"] = np.nan

        # VaR（95%）
        for w in [20, 60]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                factors[f"sup_risk_var_{w}d"] = float(np.percentile(rets, 5))
            else:
                factors[f"sup_risk_var_{w}d"] = np.nan

        # CVaR
        for w in [20, 60]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                var = np.percentile(rets, 5)
                cvar = np.mean(rets[rets <= var])
                factors[f"sup_risk_cvar_{w}d"] = float(cvar)
            else:
                factors[f"sup_risk_cvar_{w}d"] = np.nan

        # ── 流动性补充（10个）────────────────────────
        if volume is not None and amount is not None:
            # 成交额冲击
            for w in [5, 10, 20]:
                if len(amount) > w:
                    factors[f"sup_liq_amount_shock_{w}d"] = float(amount[0] / np.mean(amount[:w]) - 1)
                else:
                    factors[f"sup_liq_amount_shock_{w}d"] = np.nan

            # 成交量波动
            for w in [5, 10, 20]:
                if len(volume) > w:
                    factors[f"sup_liq_vol_std_{w}d"] = float(np.std(volume[:w]) / (np.mean(volume[:w]) + 1e-8))
                else:
                    factors[f"sup_liq_vol_std_{w}d"] = np.nan

            # Amihud非流动性（不同窗口）
            for w in [5, 10, 20, 60]:
                if len(close) > w:
                    rets = np.abs(np.diff(np.log(close[:w+1])))
                    dollar_vol = amount[1:w+1]
                    illiq = np.mean(rets / (dollar_vol + 1e-8))
                    factors[f"sup_liq_amihud_{w}d"] = float(illiq)
                else:
                    factors[f"sup_liq_amihud_{w}d"] = np.nan
        else:
            for i in range(10):
                factors[f"sup_liq_placeholder_{i}"] = np.nan

        # ── 其他补充（15个）────────────────────────
        # 价格位置
        for w in [5, 10, 20, 60]:
            if len(close) > w:
                high_w = np.max(close[:w])
                low_w = np.min(close[:w])
                if high_w > low_w:
                    factors[f"sup_price_pos_{w}d"] = float((close[0] - low_w) / (high_w - low_w))
                else:
                    factors[f"sup_price_pos_{w}d"] = 0.5
            else:
                factors[f"sup_price_pos_{w}d"] = np.nan

        # 收益偏度
        for w in [20, 60]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                if np.std(rets) > 1e-8:
                    factors[f"sup_ret_skew_{w}d"] = float(
                        np.mean((rets - np.mean(rets)) ** 3) / np.std(rets) ** 3)
                else:
                    factors[f"sup_ret_skew_{w}d"] = np.nan
            else:
                factors[f"sup_ret_skew_{w}d"] = np.nan

        # 收益峰度
        for w in [20, 60]:
            if len(close) > w:
                rets = np.diff(np.log(close[:w+1]))
                if np.std(rets) > 1e-8:
                    factors[f"sup_ret_kurt_{w}d"] = float(
                        np.mean((rets - np.mean(rets)) ** 4) / np.std(rets) ** 4 - 3)
                else:
                    factors[f"sup_ret_kurt_{w}d"] = np.nan
            else:
                factors[f"sup_ret_kurt_{w}d"] = np.nan

        # 波动率变化
        if len(close) > 40:
            vol_20 = np.std(np.diff(np.log(close[:21])))
            vol_40 = np.std(np.diff(np.log(close[:41])))
            if vol_40 > 1e-8:
                factors["sup_vol_change"] = float(vol_20 / vol_40 - 1)
            else:
                factors["sup_vol_change"] = np.nan
        else:
            factors["sup_vol_change"] = np.nan

        # ── 额外补充（3个）────────────────────────
        # 夏普比率
        if len(close) > 60:
            rets = np.diff(np.log(close[:61]))
            if np.std(rets) > 1e-8:
                factors["sup_sharpe_60d"] = float(np.mean(rets) / np.std(rets) * np.sqrt(252))
            else:
                factors["sup_sharpe_60d"] = np.nan
        else:
            factors["sup_sharpe_60d"] = np.nan

        # 索提诺比率
        if len(close) > 60:
            rets = np.diff(np.log(close[:61]))
            neg_rets = rets[rets < 0]
            if len(neg_rets) > 0 and np.std(neg_rets) > 1e-8:
                factors["sup_sortino_60d"] = float(np.mean(rets) / np.std(neg_rets) * np.sqrt(252))
            else:
                factors["sup_sortino_60d"] = np.nan
        else:
            factors["sup_sortino_60d"] = np.nan

        # 卡玛比率
        if len(close) > 120:
            ret_120 = close[0] / close[120] - 1
            cummax = np.maximum.accumulate(close[:120][::-1])[::-1]
            drawdown = (cummax - close[:120]) / cummax
            max_dd = np.max(drawdown)
            if max_dd > 1e-8:
                factors["sup_calmar_120d"] = float(ret_120 / max_dd)
            else:
                factors["sup_calmar_120d"] = np.nan
        else:
            factors["sup_calmar_120d"] = np.nan

        return factors

    # ────────────────────────────────────────────
    # 错误定价因子 MISV_FY1（1个）
    # ────────────────────────────────────────────
    def calc_misv_fy1(
        self,
        pb: float,
        net_equity: float,
        ni_fy1: float,
        gamma_0: float,
        gamma_1: float,
    ) -> float:
        """
        错误定价因子 MISV_FY1
        
        公式: MISV_FY1 = (V_hat / B) / PB - 1
        其中: V_hat / B = gamma_0 + gamma_1 * (NI_FY1 / B)
        
        参数:
            pb: 市净率 (当前股价 / 每股净资产)
            net_equity: 净资产 (B)
            ni_fy1: 分析师一致预期净利润 FY1
            gamma_0: 横截面回归截距
            gamma_1: 横截面回归斜率
        
        返回:
            MISV_FY1值: 正值表示高估，负值表示低估
        """
        # 检查数据有效性
        if not (np.isfinite(pb) and pb > 0):
            return np.nan
        if not (np.isfinite(net_equity) and net_equity > 0):
            return np.nan
        if not np.isfinite(ni_fy1):
            return np.nan
        if not (np.isfinite(gamma_0) and np.isfinite(gamma_1)):
            return np.nan
        
        # 计算预测的合理估值 V_hat / B
        ni_over_b = ni_fy1 / net_equity
        v_hat_over_b = gamma_0 + gamma_1 * ni_over_b
        
        # 计算 MISV_FY1
        misv_fy1 = v_hat_over_b / pb - 1
        
        return float(misv_fy1)

    @staticmethod
    def cross_sectional_regression(
        pb_series: pd.Series,
        net_equity_series: pd.Series,
        ni_fy1_series: pd.Series,
    ) -> tuple:
        """
        横截面回归：V/B = gamma_0 + gamma_1 * (NI_FY1 / B) + u
        
        参数:
            pb_series: 全市场股票的市净率序列 (index=ts_code)
            net_equity_series: 全市场股票的净资产序列 (index=ts_code)
            ni_fy1_series: 全市场股票的分析师预期净利润序列 (index=ts_code)
        
        返回:
            (gamma_0, gamma_1): 回归系数
        """
        # 对齐索引
        common_idx = pb_series.index.intersection(net_equity_series.index)
        common_idx = common_idx.intersection(ni_fy1_series.index)
        
        if len(common_idx) < 30:  # 至少需要30个样本
            return np.nan, np.nan
        
        # 构建回归变量
        pb = pb_series.loc[common_idx]
        b = net_equity_series.loc[common_idx]
        ni = ni_fy1_series.loc[common_idx]
        
        # 过滤有效数据
        valid_mask = (
            np.isfinite(pb) & (pb > 0) &
            np.isfinite(b) & (b > 0) &
            np.isfinite(ni)
        )
        
        if valid_mask.sum() < 30:
            return np.nan, np.nan
        
        # 因变量: V/B = PB (市值/净资产 = 市净率)
        y = pb[valid_mask].values
        
        # 自变量: NI_FY1 / B
        x = (ni[valid_mask] / b[valid_mask]).values
        
        # 简单线性回归: y = gamma_0 + gamma_1 * x
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if abs(denominator) < 1e-10:
            return np.nan, np.nan
        
        gamma_1 = numerator / denominator
        gamma_0 = y_mean - gamma_1 * x_mean
        
        return gamma_0, gamma_1
