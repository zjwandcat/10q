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
Tushare数据拉取模块
拉取范围：2007-01 至 2025-12（228个月）
支持断点续跑（已缓存的月份跳过）
"""
import tushare as ts
import pandas as pd
import numpy as np
import yaml
import pickle
import time
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys

config_path = Path(__file__).parent.parent / "config" / "concurrency_config.py"
if config_path.exists():
    sys.path.insert(0, str(config_path.parent))
    from concurrency_config import M0_FETCH_MAX_WORKERS
    sys.path.pop(0)
else:
    M0_FETCH_MAX_WORKERS = 4

logger = logging.getLogger("m0.fetcher")


def load_config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TushareFetcher:
    """
    Tushare Pro数据拉取器
    包含限流控制（Tushare每分钟调用限制）
    包含断点续跑（原始数据缓存到data/raw_cache/）
    """

    def __init__(self):
        cfg = load_config()
        # Token 优先级：环境变量 TUSHARE_TOKEN > config.yaml（兼容旧配置，不推荐）
        token = os.environ.get("TUSHARE_TOKEN") or cfg.get("tushare", {}).get("token", "")
        if not token:
            raise RuntimeError(
                "Tushare token 未配置！\n"
                "请设置环境变量：\n"
                "  PowerShell: $env:TUSHARE_TOKEN = '你的token'\n"
                "  CMD:        set TUSHARE_TOKEN=你的token\n"
                "或在 config/config.yaml 的 tushare.token 字段填写（不推荐）"
            )
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache_dir = Path(cfg["data"]["raw_cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = Path(cfg["data"]["index_dir"])
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Tushare限流：每分钟200次，保守设置
        self._call_interval = 0.3  # 秒
        self._last_call = 0.0

    def _rate_limit(self):
        """限流控制"""
        elapsed = time.time() - self._last_call
        if elapsed < self._call_interval:
            time.sleep(self._call_interval - elapsed)
        self._last_call = time.time()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _load_cache(self, key: str):
        p = self._cache_path(key)
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, data):
        with open(self._cache_path(key), "wb") as f:
            pickle.dump(data, f)

    def get_trade_calendar(
        self,
        start_date: str,
        end_date: str,
    ) -> list:
        """获取交易日历，返回交易日列表（YYYYMMDD格式）"""
        key = f"calendar_{start_date}_{end_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self.pro.trade_cal(
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
            is_open=1,
        )
        result = sorted(df["cal_date"].tolist())
        self._save_cache(key, result)
        return result

    def get_stock_basic(self) -> pd.DataFrame:
        """获取全量股票基础信息（含上市日期、行业）"""
        key = "stock_basic_all"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,name,industry,list_date",
        )
        self._save_cache(key, df)
        return df

    def get_monthly_basic(self, trade_date: str) -> pd.DataFrame:
        """
        获取指定月末的股票基础行情
        包含：收盘价、总市值、换手率、停牌信息
        trade_date格式：YYYYMMDD（月末最后一个交易日）
        """
        key = f"monthly_basic_{trade_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self.pro.daily_basic(
            trade_date=trade_date,
            fields=(
                "ts_code,trade_date,close,total_mv,"
                "turnover_rate,volume_ratio"
            ),
        )
        self._save_cache(key, df)
        return df

    def get_adj_factor_map(
        self,
        trade_date: str,
    ) -> dict:
        """
        获取全市场复权因子映射
        trade_date: YYYYMMDD
        返回: {ts_code: adj_factor}
        后复权因子：close_adj = close * adj_factor
        """
        key = f"adj_factor_{trade_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = self.pro.adj_factor(
                trade_date=trade_date,
            )
            if df.empty:
                return {}
            result = dict(zip(
                df["ts_code"],
                df["adj_factor"]
            ))
            self._save_cache(key, result)
            return result
        except Exception as e:
            logger.warning(f"adj_factor获取失败{trade_date}: {e}")
            return {}

    def get_monthly_turnover(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> float:
        """
        计算近20交易日日均换手率
        """
        key = f"turnover_{ts_code}_{start_date}_{end_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,turnover_rate",
            )
            if df.empty:
                return 0.0
            result = float(df["turnover_rate"].mean() / 100)
            self._save_cache(key, result)
            return result
        except Exception:
            return 0.0

    def get_suspend_days(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """计算当月停牌天数"""
        key = f"suspend_{ts_code}_{start_date}_{end_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = self.pro.suspend_d(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            result = len(df) if df is not None else 0
            self._save_cache(key, result)
            return result
        except Exception:
            return 0

    def get_benchmark_return(
        self,
        trade_date_yyyymm: str,
    ) -> float:
        """
        获取000906.SH（中证800）当月收益率
        trade_date_yyyymm格式：YYYYMM
        """
        key = f"benchmark_{trade_date_yyyymm}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = self.pro.index_monthly(
                ts_code="000906.SH",
                start_date=trade_date_yyyymm + "01",
                end_date=trade_date_yyyymm + "31",
                fields="trade_date,pct_chg",
            )
            if df.empty:
                return 0.0
            result = float(df.iloc[0]["pct_chg"]) / 100
            self._save_cache(key, result)
            return result
        except Exception:
            return 0.0

    def get_financial_data(
        self,
        ts_code: str,
        period: str,
    ) -> dict:
        """
        获取财务数据（ROE/净利润增长/毛利率等）
        period格式：YYYYMMDD（季报日期）
        """
        key = f"financial_{ts_code}_{period}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = self.pro.fina_indicator(
                ts_code=ts_code,
                period=period,
                fields=(
                    "ts_code,period,"
                    # 已有字段
                    "roe,netprofit_yoy,grossprofit_margin,debt_to_assets,current_ratio,quick_ratio,"
                    # 新增价值类
                    "pe,pb,ps,pcf,"
                    # 新增成长类
                    "or_yoy,eps,netprofit_q_yoy,"
                    # 新增质量类
                    "roe_dt,roe_waa,inv_turn,ar_turn,ocf_to_or,ocf_to_profit,debt_eqt_ratio,turnover_days,"
                    # 新增股息类
                    "dv_ratio"
                ),
            )
            if df.empty:
                return {}
            result = df.iloc[0].to_dict()
            self._save_cache(key, result)
            return result
        except Exception:
            return {}

    def get_analyst_forecast(
        self,
        ts_code: str,
        period: str,
    ) -> dict:
        """
        获取分析师一致预期数据（净利润预测FY1/FY2）
        period格式：YYYYMMDD（报告期）
        返回：net_profit_fy1, net_profit_fy2 等
        """
        key = f"analyst_fcst_{ts_code}_{period}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            # Tushare分析师预测接口
            df = self.pro.ana_forecast(
                ts_code=ts_code,
                period=period,
                fields="ts_code,period,net_profit_fy1,net_profit_fy2,pe_fy1,pe_fy2",
            )
            if df.empty:
                return {}
            result = df.iloc[0].to_dict()
            self._save_cache(key, result)
            return result
        except Exception:
            return {}

    def get_daily_valuation(
        self,
        trade_date: str,
    ) -> pd.DataFrame:
        """
        获取全市场每日估值指标（PB/PE/PS等）
        trade_date格式：YYYYMMDD
        返回DataFrame包含：ts_code, pb, pe, ps, total_mv, circ_mv
        """
        key = f"daily_valuation_{trade_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = self.pro.daily_basic(
                trade_date=trade_date,
                fields="ts_code,trade_date,close,total_mv,circ_mv,pe,pb,ps,pcf",
            )
            self._save_cache(key, df)
            return df
        except Exception:
            return pd.DataFrame()

    def get_balance_sheet(
        self,
        ts_code: str,
        period: str,
    ) -> dict:
        """
        获取资产负债表数据（净资产等）
        period格式：YYYYMMDD（报告期）
        """
        key = f"balance_{ts_code}_{period}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = self.pro.balancesheet(
                ts_code=ts_code,
                period=period,
                fields="ts_code,period,total_hldr_eqy_exc_min_int,total_assets,total_liab",
            )
            if df.empty:
                return {}
            result = df.iloc[0].to_dict()
            self._save_cache(key, result)
            return result
        except Exception:
            return {}

    def get_daily_data(
        self,
        ts_code: str,
        end_date: str,
        n_days: int = 300,
    ) -> pd.DataFrame:
        """
        获取单只股票的历史日线数据（用于因子计算）
        
        数据源优先级：
        1. Tushare Pro API（主要）
        2. AKShare（备选）
        3. 本地缓存
        
        参数:
            ts_code: 股票代码（如'000001.SZ'）
            end_date: 结束日期（YYYYMMDD格式，通常为月末交易日）
            n_days: 需要的交易日数量（默认300天，约1.2年）
        
        返回:
            DataFrame包含：open, high, low, close, volume, amount, turnover_rate
            按日期降序排列（最新在前）
        """
        # 计算开始日期（向前推n_days*1.5个自然日以确保足够交易日）
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=int(n_days * 1.5))
        start_date = start_dt.strftime("%Y%m%d")
        
        key = f"daily_{ts_code}_{start_date}_{end_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        
        # 尝试方法1: Tushare
        df = self._get_daily_data_tushare(ts_code, start_date, end_date)
        
        # 如果Tushare失败，尝试方法2: AKShare
        if df.empty:
            df = self._get_daily_data_akshare(ts_code, start_date, end_date)
        
        if not df.empty:
            self._save_cache(key, df)
        
        return df

    def _get_daily_data_tushare(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """使用Tushare获取日线数据"""
        self._rate_limit()
        try:
            # 获取日线行情数据
            df_daily = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            )
            
            if df_daily.empty:
                return pd.DataFrame()
            
            # 获取每日基础数据（换手率等）
            self._rate_limit()
            df_basic = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,turnover_rate",
            )
            
            # 合并数据
            if not df_basic.empty:
                df = df_daily.merge(df_basic[["ts_code", "trade_date", "turnover_rate"]], 
                                    on=["ts_code", "trade_date"], how="left")
            else:
                df = df_daily.copy()
                df["turnover_rate"] = np.nan
            
            # 重命名列以匹配factor_calculator的期望
            df = df.rename(columns={"vol": "volume"})
            
            # ★ 新增：获取复权因子并调整close/open/high/low
            try:
                adj_df = self.pro.adj_factor(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not adj_df.empty:
                    # 取最新复权因子作为基准（后复权）
                    adj_df = adj_df.sort_values(
                        "trade_date", ascending=False)
                    adj_map = adj_df.set_index(
                        "trade_date")["adj_factor"]
                    
                    # 对价格列做后复权
                    latest_adj = adj_df["adj_factor"].iloc[0]
                    price_cols = ["open", "high", "low", "close"]
                    for col in price_cols:
                        if col in df.columns:
                            df[col] = (
                                df["trade_date"].map(adj_map) /
                                latest_adj *
                                df[col]
                            ).fillna(df[col])
            except Exception as e:
                logger.warning(
                    f"{ts_code} 日线复权失败，使用原始价格: {e}")
            
            # 按日期降序排列（最新在前）
            df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
            
            # 只保留需要的列
            required_cols = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]
            df = df[[c for c in required_cols if c in df.columns]]
            
            # 确保数值类型正确
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "turnover_rate" in df.columns:
                df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce") / 100  # 转为小数
            
            return df
            
        except Exception as e:
            logger.warning(f"Tushare获取日线数据失败 {ts_code}: {e}")
            return pd.DataFrame()

    def _get_daily_data_akshare(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """使用AKShare获取日线数据（备选数据源）"""
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare未安装，跳过备选数据源")
            return pd.DataFrame()
        
        try:
            # 转换股票代码格式: '000001.SZ' -> '000001'
            code = ts_code.split('.')[0]
            
            # 判断市场
            market = ts_code.split('.')[1]
            
            # AKShare获取日线数据
            if market == 'SH':
                df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                        start_date=start_date, end_date=end_date, adjust="")
            else:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                        start_date=start_date, end_date=end_date, adjust="")
            
            if df.empty:
                return pd.DataFrame()
            
            # 重命名列
            column_map = {
                '日期': 'trade_date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '换手率': 'turnover_rate',
            }
            df = df.rename(columns=column_map)
            
            # 转换日期格式
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
            
            # 按日期降序排列
            df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
            
            # 只保留需要的列
            required_cols = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]
            df = df[[c for c in required_cols if c in df.columns]]
            
            # 确保数值类型正确
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "turnover_rate" in df.columns:
                df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce") / 100  # 转为小数
            
            logger.info(f"AKShare获取日线数据成功 {ts_code}")
            return df
            
        except Exception as e:
            logger.warning(f"AKShare获取日线数据失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_daily_data_by_date(
        self,
        trade_date: str,
    ) -> pd.DataFrame:
        """
        按交易日获取全市场日线数据（批量方案）
        
        参数:
            trade_date: 交易日期（YYYYMMDD格式）
        
        返回:
            DataFrame包含：ts_code,trade_date,open,high,low,close,vol,amount
            按ts_code排序
        """
        key = f"daily_by_date_{trade_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        
        self._rate_limit()
        try:
            df = self.pro.daily(
                trade_date=trade_date,
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            )
            if not df.empty:
                df = df.sort_values("ts_code").reset_index(drop=True)
                self._save_cache(key, df)
            return df
        except Exception as e:
            logger.warning(f"按交易日获取日线数据失败 {trade_date}: {e}")
            return pd.DataFrame()

    def get_daily_data_batch(
        self,
        ts_codes: list,
        end_date: str,
        n_days: int = 300,
        daily_basic_df: pd.DataFrame = None,
    ) -> dict:
        """
        批量获取多只股票的日线数据（优化版：向量化拼装）

        参数:
            ts_codes: 股票代码列表
            end_date: 结束日期（YYYYMMDD格式）
            n_days: 需要的交易日数量（默认300天）
            daily_basic_df: 月末daily_basic数据（用于获取turnover_rate）

        返回:
            dict: {ts_code: DataFrame}
        """
        # 计算开始日期
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=int(n_days * 1.5))
        start_date = start_dt.strftime("%Y%m%d")

        # 获取交易日历
        trade_cal = self.get_trade_calendar(start_date, end_date)
        if not trade_cal:
            return {ts_code: pd.DataFrame() for ts_code in ts_codes}

        # 快速路径：检查是否所有股票都有单股票缓存
        # 如果全部命中，跳过逐日拼装
        all_cached = True
        results = {}
        for ts_code in ts_codes:
            cache_key = f"daily_{ts_code}_{start_date}_{end_date}"
            cached = self._load_cache(cache_key)
            if cached is not None:
                results[ts_code] = cached
            else:
                all_cached = False

        if all_cached:
            return results

        # 慢路径：按交易日获取全市场数据，然后向量化拼装
        daily_by_date = {}
        for trade_date in trade_cal:
            df = self.get_daily_data_by_date(trade_date)
            if not df.empty:
                daily_by_date[trade_date] = df

        # 向量化拼装：一次性 concat 全部交易日数据
        if daily_by_date:
            all_days = pd.concat(daily_by_date.values(), ignore_index=True)
            all_days = all_days.rename(columns={"vol": "volume"})
            # 只保留需要的列
            keep_cols = ["ts_code", "trade_date", "open", "high", "low",
                         "close", "volume", "amount"]
            all_days = all_days[[c for c in keep_cols if c in all_days.columns]]

            # 过滤目标股票
            ts_set = set(ts_codes)
            all_days = all_days[all_days["ts_code"].isin(ts_set)]

            # 按股票分组
            grouped = all_days.groupby("ts_code")

            # turnover_rate 映射
            turnover_map = {}
            if daily_basic_df is not None and not daily_basic_df.empty:
                for _, row in daily_basic_df.iterrows():
                    tr = row.get("turnover_rate", np.nan)
                    if not pd.isna(tr):
                        turnover_map[row["ts_code"]] = tr

            for ts_code in ts_codes:
                if ts_code in results:
                    continue  # 已有缓存
                if ts_code in grouped.groups:
                    df = grouped.get_group(ts_code).copy()
                    df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
                    df = df[[c for c in ["open","high","low","close","volume","amount"] if c in df.columns]]
                    df["turnover_rate"] = turnover_map.get(ts_code, np.nan)
                    cache_key = f"daily_{ts_code}_{start_date}_{end_date}"
                    self._save_cache(cache_key, df)
                    results[ts_code] = df
                else:
                    results[ts_code] = pd.DataFrame()
        else:
            for ts_code in ts_codes:
                if ts_code not in results:
                    results[ts_code] = pd.DataFrame()

        return results

    def get_macro_data(
        self,
        trade_date: str,
    ) -> dict:
        """
        获取宏观数据（用于宏观因子计算）
        
        数据源优先级：
        1. Tushare Pro API（主要）
        2. AKShare（备选）
        
        参数:
            trade_date: 交易日期（YYYYMMDD格式）
        
        返回:
            dict包含68个宏观变量
        """
        key = f"macro_{trade_date}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        
        macro_data = {}
        
        # 尝试Tushare获取宏观数据
        macro_data = self._get_macro_data_tushare(trade_date)
        
        # 如果Tushare数据不足，尝试AKShare补充
        if len(macro_data) < 10:
            macro_data.update(self._get_macro_data_akshare(trade_date))
        
        # 如果获取的宏观数据不足68个，不填充0（保持NaN避免污染截面Z-score）
        # 原因：宏观因子对所有股票同月取值相同，截面Z-score后方差=0、IC=0
        # 0填充比NaN更难过滤（valid_rate=100%但值无意义）
        # 让缺失的key不赋值，由feature_store.py的零方差过滤自动清除
        if len(macro_data) < 68:
            macro_data = {k: v for k, v in macro_data.items()
                          if v != 0.0 and v is not None and np.isfinite(v)
                          and abs(v) > 1e-10}
        
        self._save_cache(key, macro_data)
        return macro_data

    def _get_macro_data_tushare(self, trade_date: str) -> dict:
        """使用Tushare获取宏观数据"""
        macro_data = {}
        
        try:
            # 1. Shibor利率（上海银行间同业拆放利率）
            self._rate_limit()
            try:
                shibor_df = self.pro.shibor(
                    start_date=trade_date,
                    end_date=trade_date,
                )
                if not shibor_df.empty:
                    row = shibor_df.iloc[0]
                    macro_data["shibor_on"] = float(row.get("on", 0))  # 隔夜
                    macro_data["shibor_1w"] = float(row.get("w1", 0))  # 1周
                    macro_data["shibor_1m"] = float(row.get("m1", 0))  # 1个月
                    macro_data["shibor_3m"] = float(row.get("m3", 0))  # 3个月
            except Exception:
                pass
            
            # 2. 国债收益率
            self._rate_limit()
            try:
                bond_df = self.pro.bond_chy(
                    start_date=trade_date,
                    end_date=trade_date,
                )
                if not bond_df.empty:
                    # 提取不同期限国债收益率
                    for _, row in bond_df.iterrows():
                        bond_type = row.get("bond_type", "")
                        yield_val = row.get("yield", 0)
                        if "1年" in bond_type or "1Y" in bond_type:
                            macro_data["bond_yield_1y"] = float(yield_val)
                        elif "10年" in bond_type or "10Y" in bond_type:
                            macro_data["bond_yield_10y"] = float(yield_val)
            except Exception:
                pass
            
            # 3. 指数数据（上证指数、深证成指、创业板指）
            self._rate_limit()
            try:
                index_df = self.pro.index_daily(
                    ts_code="000001.SH",  # 上证指数
                    start_date=trade_date,
                    end_date=trade_date,
                )
                if not index_df.empty:
                    macro_data["index_sh_close"] = float(index_df.iloc[0].get("close", 0))
                    macro_data["index_sh_pct_chg"] = float(index_df.iloc[0].get("pct_chg", 0))
            except Exception:
                pass
            
            self._rate_limit()
            try:
                index_df = self.pro.index_daily(
                    ts_code="399001.SZ",  # 深证成指
                    start_date=trade_date,
                    end_date=trade_date,
                )
                if not index_df.empty:
                    macro_data["index_sz_close"] = float(index_df.iloc[0].get("close", 0))
                    macro_data["index_sz_pct_chg"] = float(index_df.iloc[0].get("pct_chg", 0))
            except Exception:
                pass
            
            self._rate_limit()
            try:
                index_df = self.pro.index_daily(
                    ts_code="399006.SZ",  # 创业板指
                    start_date=trade_date,
                    end_date=trade_date,
                )
                if not index_df.empty:
                    macro_data["index_cyb_close"] = float(index_df.iloc[0].get("close", 0))
                    macro_data["index_cyb_pct_chg"] = float(index_df.iloc[0].get("pct_chg", 0))
            except Exception:
                pass
            
        except Exception as e:
            logger.warning(f"Tushare获取宏观数据失败 {trade_date}: {e}")
        
        return macro_data

    def _get_macro_data_akshare(self, trade_date: str) -> dict:
        """使用AKShare获取宏观数据（备选数据源）"""
        macro_data = {}
        
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare未安装，跳过宏观数据备选数据源")
            return macro_data
        
        try:
            # 1. Shibor利率
            try:
                shibor_df = ak.rate_interbank(market="上海银行间同业拆放利率", 
                                              symbol="Shibor人民币", 
                                              date_range=trade_date[:4])
                if not shibor_df.empty:
                    # 提取最近一天的数据
                    shibor_df = shibor_df.sort_values('日期', ascending=False)
                    row = shibor_df.iloc[0]
                    macro_data["shibor_on_ak"] = float(row.get("隔夜", 0))
                    macro_data["shibor_1w_ak"] = float(row.get("1周", 0))
                    macro_data["shibor_1m_ak"] = float(row.get("1个月", 0))
                    macro_data["shibor_3m_ak"] = float(row.get("3个月", 0))
            except Exception:
                pass
            
            # 2. 上证指数
            try:
                index_df = ak.stock_zh_index_daily(symbol="sh000001")
                if not index_df.empty:
                    # 筛选指定日期
                    index_df['date'] = pd.to_datetime(index_df['date']).dt.strftime('%Y%m%d')
                    row = index_df[index_df['date'] == trade_date]
                    if not row.empty:
                        macro_data["index_sh_close_ak"] = float(row.iloc[0]['close'])
                        macro_data["index_sh_volume_ak"] = float(row.iloc[0].get('volume', 0))
            except Exception:
                pass
            
            # 3. 深证成指
            try:
                index_df = ak.stock_zh_index_daily(symbol="sz399001")
                if not index_df.empty:
                    index_df['date'] = pd.to_datetime(index_df['date']).dt.strftime('%Y%m%d')
                    row = index_df[index_df['date'] == trade_date]
                    if not row.empty:
                        macro_data["index_sz_close_ak"] = float(row.iloc[0]['close'])
            except Exception:
                pass
            
            # 4. 创业板指
            try:
                index_df = ak.stock_zh_index_daily(symbol="sz399006")
                if not index_df.empty:
                    index_df['date'] = pd.to_datetime(index_df['date']).dt.strftime('%Y%m%d')
                    row = index_df[index_df['date'] == trade_date]
                    if not row.empty:
                        macro_data["index_cyb_close_ak"] = float(row.iloc[0]['close'])
            except Exception:
                pass
            
            if len(macro_data) > 0:
                logger.info(f"AKShare获取宏观数据成功 {trade_date}")
                
        except Exception as e:
            logger.warning(f"AKShare获取宏观数据失败 {trade_date}: {e}")
        
        return macro_data
