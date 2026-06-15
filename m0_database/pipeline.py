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
M0主流程入口
支持：全量生成 / 指定月份生成 / 指定方案生成
支持断点续跑（已存在的parquet跳过）
"""
import pandas as pd
import numpy as np
import yaml
import logging
import gc
from pathlib import Path
from datetime import datetime

from m0_database.data_fetcher import TushareFetcher
from m0_database.factor_calculator import FactorCalculator
from m0_database.stock_filter import filter_stock_pool
from m0_database.neutralization import apply_neutralization

logging.basicConfig(
    level=logging.INFO,
    format="[M0][%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/m0/pipeline.log",
                           encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("m0.pipeline")


def load_config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_month_list(
    start: str = "200701",
    end:   str = "202512",
) -> list:
    """生成YYYYMM月份列表"""
    months = []
    current = datetime.strptime(start, "%Y%m")
    end_dt = datetime.strptime(end, "%Y%m")
    while current <= end_dt:
        months.append(current.strftime("%Y%m"))
        if current.month == 12:
            current = current.replace(
                year=current.year+1, month=1)
        else:
            current = current.replace(month=current.month+1)
    return months


# 元信息列（不进模型，不做中性化）
META_COLS = [
    "trade_date", "stock_code", "stock_name", "industry",
    "list_date", "days_listed", "close_price", "market_cap",
    "avg_turnover_rate", "Target_Return_1M", "benchmark_return",
    "excess_return_1m", "suspend_days",
]


def run_pipeline(
    schemes: list = None,     # 为None时使用config中active_scheme
    months:  list = None,     # 为None时生成全部228个月
    force_rebuild: bool = False,  # True时强制重建已存在文件
    phase: str = "full",      # "full"|"cache"|"neutralize"
):
    """
    M0主流程

    参数：
        schemes: 要生成的方案列表，如["scheme_b"]
        months: 要处理的月份列表，如["202001","202002"]
        force_rebuild: True时强制重建已存在文件
        phase:
            - "full": 计算+中性化+落盘（原行为）
            - "cache": 只计算+写 cache（不中性化），用于多方案共享前置工作
            - "neutralize": 跳过计算，从 cache 读，做中性化+落盘

    两阶段执行（regenerator 使用）：
        Phase 1 cache:      228 月 × 1 worker
                            跑 Tushare+因子计算，结果存 cache
                            内存：~1.5GB/worker
        Phase 2 neutralize: 912 任务 × 2-4 worker
                            读 cache + apply_neutralization
                            内存：~300MB/worker（极轻）
    """
    cfg = load_config()
    Path("logs/m0").mkdir(parents=True, exist_ok=True)

    # 确定要生成的方案
    if schemes is None:
        schemes = [cfg["data"]["neutralization"]["active_scheme"]]

    # 确定要生成的月份
    if months is None:
        months = get_month_list("200701", "202512")

    logger.info(f"M0开始运行: 方案={schemes}, 月份数={len(months)}")

    # 初始化工具
    fetcher = TushareFetcher()
    calculator = FactorCalculator()

    # 确保输出目录存在
    pool_dirs = cfg["data"]["pool_dirs"]
    for scheme in schemes:
        Path(pool_dirs[scheme]).mkdir(parents=True, exist_ok=True)

    # ── 主循环：按月处理 ──────────────────────────
    for month_idx, yyyymm in enumerate(months):

        cache_path = Path("data/cache_m0_factors") / f"{yyyymm}.parquet"
        Path("data/cache_m0_factors").mkdir(parents=True, exist_ok=True)

        # 检查是否需要跳过（断点续跑）
        if not force_rebuild:
            if phase == "cache":
                # cache 模式：只看 cache 文件是否存在
                if cache_path.exists():
                    logger.info(f"[{month_idx+1}/{len(months)}] "
                               f"{yyyymm} cache 已存在，跳过")
                    continue
            else:
                all_exist = all(
                    (Path(pool_dirs[s]) / f"{yyyymm}.parquet").exists()
                    for s in schemes
                )
                if all_exist:
                    logger.info(f"[{month_idx+1}/{len(months)}] "
                               f"{yyyymm} 已存在，跳过")
                    continue

        logger.info(f"[{month_idx+1}/{len(months)}] "
                   f"处理 {yyyymm}...")

        # ============ Phase: neutralize（直接读 cache） ============
        if phase == "neutralize":
            if not cache_path.exists():
                logger.error(
                    f"{yyyymm} cache 不存在 {cache_path}，无法 neutralize。"
                    f"请先跑 phase=cache。"
                )
                continue
            try:
                df = pd.read_parquet(cache_path)
                factor_cols = [c for c in df.columns if c not in META_COLS]

                # 用 scheme_b 补齐 cache 中全 NaN 的因子列
                # （确保新方案数据量 >= scheme_b）
                ref_fill_path = (Path(pool_dirs["scheme_b"]) /
                                 f"{yyyymm}.parquet")
                if ref_fill_path.exists():
                    ref_fill_df = pd.read_parquet(ref_fill_path)
                    for c in factor_cols:
                        if c in ref_fill_df.columns:
                            cache_nan = df[c].isna()
                            ref_has = ~ref_fill_df[c].isna()
                            fill_mask = cache_nan & ref_has
                            if fill_mask.any():
                                df[c] = df[c].fillna(ref_fill_df[c])

                for scheme in schemes:
                    output_path = (Path(pool_dirs[scheme]) /
                                   f"{yyyymm}.parquet")
                    extra_kwargs = {}
                    if scheme == "scheme_g":
                        # 优先从 cache 读 returns_matrix（不再拉 Tushare）
                        returns_npz = str(cache_path).replace(
                            ".parquet", "_returns.npz")
                        if Path(returns_npz).exists():
                            loaded = np.load(returns_npz)
                            returns_matrix = loaded["returns_matrix"]
                        else:
                            # 降级：cache 没有 returns_matrix，从 Tushare 拉
                            logger.warning(
                                f"  {yyyymm} cache 无 returns.npz，"
                                f"从 Tushare 拉日线（仅 G 方案）")
                            daily_data_dict = (
                                _get_daily_data_dict_for_neutralize(
                                    fetcher, df["stock_code"].tolist(),
                                    yyyymm))
                            returns_matrix = _build_returns_matrix(
                                df=df,
                                daily_data_dict=daily_data_dict,
                                lookback_days=60)
                            del daily_data_dict
                            gc.collect()
                        extra_kwargs["returns_matrix"] = returns_matrix
                        extra_kwargs["lookback_days"] = 60
                        extra_kwargs["alignment"] = (
                            cfg.get("data", {})
                               .get("neutralization", {})
                               .get("params", {})
                               .get("scheme_g", {})
                               .get("alignment", "drop")
                        )

                    df_scheme = apply_neutralization(
                        df.copy(), factor_cols, scheme, **extra_kwargs)
                    for col in factor_cols:
                        if col in df_scheme.columns:
                            df_scheme[col] = df_scheme[col].astype(
                                np.float32)
                    # 对齐 scheme_b 列结构：补齐缺失列、对齐列顺序
                    ref_path = Path(pool_dirs["scheme_b"]) / f"{yyyymm}.parquet"
                    if ref_path.exists():
                        ref_df = pd.read_parquet(ref_path, columns=[])
                        # 补齐 scheme_b 有但本方案缺失的列
                        for col in ref_df.columns:
                            if col not in df_scheme.columns:
                                if col in df.columns:
                                    df_scheme[col] = df[col].values
                                else:
                                    df_scheme[col] = np.nan
                        # 按 scheme_b 列顺序重排
                        ordered = [c for c in ref_df.columns if c in df_scheme.columns]
                        extra = [c for c in df_scheme.columns if c not in ref_df.columns]
                        df_scheme = df_scheme[ordered + extra]
                    df_scheme.to_parquet(output_path, index=False)
                    logger.info(
                        f"  [{scheme}] 已保存: {output_path.name} "
                        f"({len(df_scheme)}行×"
                        f"{len(df_scheme.columns)}列)")
                    del df_scheme
                    gc.collect()

                del df
                gc.collect()
            except Exception as e:
                import traceback
                logger.error(f"{yyyymm} neutralize 失败: {e}")
                logger.error(traceback.format_exc())
                continue
            continue

        try:
            # Step1: 获取当月末交易日
            cal = fetcher.get_trade_calendar(
                yyyymm + "01", yyyymm + "31")
            if not cal:
                logger.warning(f"{yyyymm} 无交易日，跳过")
                continue
            last_trade_date = cal[-1]  # 月末最后交易日

            # Step2: 获取股票基础信息
            stock_basic = fetcher.get_stock_basic()

            # Step3: 获取月末行情基础数据
            daily_basic = fetcher.get_monthly_basic(last_trade_date)
            if daily_basic.empty:
                logger.warning(f"{yyyymm} 行情数据为空，跳过")
                continue

            # Step4: 合并基础信息
            df = daily_basic.merge(
                stock_basic[["ts_code","name","industry",
                            "list_date"]],
                on="ts_code", how="left",
            )
            df = df.rename(columns={
                "ts_code": "stock_code",
                "name": "stock_name",
                "close": "close_price",
                "total_mv": "market_cap",
                "turnover_rate": "avg_turnover_rate",
            })

            # 计算上市天数
            df["list_date"] = pd.to_datetime(df["list_date"])
            df["trade_date_dt"] = pd.to_datetime(last_trade_date)
            df["days_listed"] = (
                df["trade_date_dt"] - df["list_date"]
            ).dt.days
            # 防止 Timestamp 列漏进 factor_cols
            df = df.drop(columns=["trade_date_dt", "list_date"])

            # market_cap单位转换：万元→亿元
            df["market_cap"] = df["market_cap"] / 10000

            # avg_turnover_rate转为小数（%→0~1）
            df["avg_turnover_rate"] = df["avg_turnover_rate"] / 100

            # 停牌天数（简化：从volume_ratio推断）
            # volume_ratio < 0.01视为停牌
            if "volume_ratio" in df.columns:
                # 精确方案：调用get_suspend_days
                # 简化方案：用volume_ratio估算（减少API调用）
                df["suspend_days"] = 0  # 简化版，精确版见注释

            # Step5: 股票池筛选
            df = filter_stock_pool(df, verbose=True)
            logger.info(f"  筛选后股票数: {len(df)}")

            if len(df) < 15:
                logger.warning(f"{yyyymm} 筛选后股票不足15只，跳过")
                continue

            # Step6: 计算460个因子
            # 使用真实Tushare数据计算因子
            logger.info("  计算因子中...")
            
            # 获取全市场估值数据（PE/PB/PS/PCF等）
            valuation_data = fetcher.get_daily_valuation(last_trade_date)
            
            # 获取宏观数据（当月所有股票共用）
            macro_data = fetcher.get_macro_data(last_trade_date)
            
            # 创建因子计算器
            calculator = FactorCalculator()
            
            # 确定财报期：取最近一期季报
            month_int = int(yyyymm[4:6])
            year_str = yyyymm[:4]
            if month_int <= 3:
                period = f"{int(year_str)-1}1231"
            elif month_int <= 6:
                period = f"{year_str}0331"
            elif month_int <= 9:
                period = f"{year_str}0630"
            else:
                period = f"{year_str}0930"
            
            # 为每只股票计算因子
            factor_rows = []
            
            # 批量获取所有股票的日线数据（方案D优化）
            ts_codes = df["stock_code"].tolist()
            logger.info(f"  批量获取{len(ts_codes)}只股票的日线数据...")
            daily_data_dict = fetcher.get_daily_data_batch(
                ts_codes, 
                last_trade_date, 
                n_days=300,
                daily_basic_df=daily_basic  # 传递月末数据用于获取turnover_rate
            )
            
            for idx, row in df.iterrows():
                ts_code = row["stock_code"]
                
                # 从批量结果中获取日线数据
                daily_df = daily_data_dict.get(ts_code, pd.DataFrame())
                
                # 如果日线数据获取失败或不足，跳过该股票
                if daily_df.empty or len(daily_df) < 60:
                    logger.warning(f"  {ts_code} 日线数据不足，跳过")
                    continue
                
                # 获取真实财务数据
                financial_data = fetcher.get_financial_data(ts_code, period)
                
                # 如果财务数据获取失败，使用空字典（因子计算器会返回NaN）
                if not financial_data:
                    financial_data = {}
                    logger.warning(f"  {ts_code} 财务数据获取失败")
                
                # 从估值数据中补充PE/PB/PS/PCF
                if not valuation_data.empty:
                    val_row = valuation_data[valuation_data["ts_code"] == ts_code]
                    if not val_row.empty:
                        for col in ["pe", "pb", "ps", "pcf"]:
                            val = val_row.iloc[0].get(col, np.nan)
                            if not pd.isna(val):
                                financial_data[col] = float(val)
                
                # 构建basic_data（含total_mv等）
                basic_data = {
                    "total_mv": row.get("market_cap", np.nan) * 10000 if not pd.isna(row.get("market_cap", np.nan)) else np.nan,  # 亿元→万元
                    "close": row.get("close_price", np.nan),
                }
                
                factors = calculator.calculate_all(daily_df, financial_data, macro_data, last_trade_date, basic_data)
                
                # 合并基础信息和因子
                factor_row = row.to_dict()
                factor_row.update(factors)
                factor_rows.append(factor_row)
            
            # 转换为DataFrame
            df = pd.DataFrame(factor_rows)
            logger.info(f"  因子计算完成: {len(df.columns)}列")

            # Step6.5: 计算MISV_FY1横截面因子
            if "value_pb_ratio" in df.columns:
                misv_result = calc_misv_fy1_cross_section(df, fetcher, period)
                if misv_result is not None:
                    df["value_misv_fy1"] = misv_result.values
                    logger.info(f"  MISV_FY1计算完成: 非NaN率={misv_result.notna().mean():.2%}")
                else:
                    df["value_misv_fy1"] = np.nan
                    logger.warning("  MISV_FY1计算失败，设为NaN")
            else:
                df["value_misv_fy1"] = np.nan

            # Step7: 获取下月收益率（复权版）
            next_month_cal = fetcher.get_trade_calendar(
                _next_month_start(yyyymm),
                _next_month_end(yyyymm),
            )
            if next_month_cal:
                next_last = next_month_cal[-1]
                next_basic = fetcher.get_monthly_basic(next_last)
                if not next_basic.empty:
                    # 获取两个时间点的复权因子
                    adj_curr = fetcher.get_adj_factor_map(last_trade_date)
                    adj_next = fetcher.get_adj_factor_map(next_last)
                    
                    next_price = next_basic.set_index("ts_code")["close"]
                    curr_price = df.set_index("stock_code")["close_price"]
                    
                    # 只处理两个时间点都有复权因子的股票
                    common = (curr_price.index
                              .intersection(next_price.index)
                              .intersection(pd.Index(adj_curr.keys()))
                              .intersection(pd.Index(adj_next.keys())))
                    
                    if len(common) > 0:
                        # 复权收益 = (close_next * adj_next) / (close_curr * adj_curr) - 1
                        adj_curr_s = pd.Series(adj_curr)[common]
                        adj_next_s = pd.Series(adj_next)[common]
                        
                        target_return = (
                            next_price[common] * adj_next_s /
                            (curr_price[common] * adj_curr_s) - 1
                        ).rename("Target_Return_1M")
                        
                        # 截断极端值（±50%，复权后仍可能有新股暴涨）
                        target_return = target_return.clip(-0.5, 0.5)
                    else:
                        # 降级：无复权因子时用原始价格+截断
                        target_return = (
                            next_price / curr_price - 1
                        ).rename("Target_Return_1M").clip(-0.5, 0.5)
                        target_return.index.name = "ts_code"
                        logger.warning(f"{yyyymm} adj_factor为空，降级使用原始价格")
                    
                    target_return.index.name = "ts_code"
                    target_return_df = target_return.reset_index().rename(
                        columns={"ts_code": "stock_code"})
                    df = df.merge(
                        target_return_df,
                        on="stock_code", how="left",
                    )

            # Step8: 获取基准收益（中证800，从本地Excel读取）
            from m0_database.benchmark_loader import (
                get_benchmark_return_for_month,
            )
            try:
                bm_ret = get_benchmark_return_for_month(yyyymm)
                df["benchmark_return"] = np.float32(bm_ret)
            except Exception as e:
                logger.warning(
                    f"中证800基准收益获取失败：{e}，"
                    f"降级为0.0"
                )
                df["benchmark_return"] = np.float32(0.0)
            df["excess_return_1m"] = (
                df.get("Target_Return_1M", np.nan) -
                df["benchmark_return"]
            )

            # Step9: 添加trade_date列
            df["trade_date"] = pd.to_datetime(last_trade_date)

            # Step10: 确定因子列
            factor_cols = [c for c in df.columns
                          if c not in META_COLS]

            # Step10.5: 为方案 G 构建 (N, lookback_days) log-return 矩阵
            # 窗口严格 [T-60, T-1]，T = last_trade_date
            # daily_data_dict[ts_code] 的 close 列按日期降序
            # 60 个 log_return 需 61 个收盘价
            g_lookback = int(
                cfg.get("data", {})
                   .get("neutralization", {})
                   .get("params", {})
                   .get("scheme_g", {})
                   .get("lookback_days", 60)
            )
            needs_g = "scheme_g" in schemes
            # Phase cache 时也构建 returns_matrix（给 Phase 2 用）
            if phase == "cache":
                needs_g = True
            returns_matrix = None
            if needs_g:
                returns_matrix = _build_returns_matrix(
                    df=df,
                    daily_data_dict=daily_data_dict,
                    lookback_days=g_lookback,
                )

            # Step10.8: 写 cache（仅 phase=cache 或 phase=full）
            #   cache 包含: 因子 DataFrame + returns_matrix (npz)
            #   Phase 2 neutralize 直接读 cache，不再拉日线
            if phase in ("cache", "full"):
                if not cache_path.exists() or force_rebuild:
                    # 先释放 daily_data_dict（cache 不需要它）
                    try:
                        del daily_data_dict
                    except NameError:
                        pass
                    gc.collect()
                    df.to_parquet(cache_path, index=False)
                    # returns_matrix 单独存 npz（轻量，~200KB/月）
                    if returns_matrix is not None:
                        np.savez_compressed(
                            str(cache_path).replace(".parquet", "_returns.npz"),
                            returns_matrix=returns_matrix,
                        )
                    logger.info(
                        f"  [cache] 已保存: {cache_path.name} "
                        f"({len(df)}行×{len(df.columns)}列)"
                        f"{' + returns_matrix' if returns_matrix is not None else ''}")
                else:
                    logger.info(
                        f"  [cache] 已存在: {cache_path.name}，跳过")
                # phase=cache 模式下，到此为止（不中性化）
                if phase == "cache":
                    del df, returns_matrix
                    gc.collect()
                    continue

            # Step11: 对每个方案执行中性化并保存
            for scheme in schemes:
                output_path = (Path(pool_dirs[scheme]) /
                              f"{yyyymm}.parquet")
                if output_path.exists() and not force_rebuild:
                    continue

                # 仅 G 方案传 returns_matrix，其它方案不传
                extra_kwargs = {}
                if scheme == "scheme_g":
                    extra_kwargs["returns_matrix"] = returns_matrix
                    extra_kwargs["lookback_days"] = g_lookback
                    # 读取对齐策略
                    extra_kwargs["alignment"] = (
                        cfg.get("data", {})
                           .get("neutralization", {})
                           .get("params", {})
                           .get("scheme_g", {})
                           .get("alignment", "drop")
                    )

                df_scheme = apply_neutralization(
                    df.copy(), factor_cols, scheme, **extra_kwargs)

                # 转float32节省内存
                for col in factor_cols:
                    if col in df_scheme.columns:
                        df_scheme[col] = (
                            df_scheme[col].astype(np.float32))

                df_scheme.to_parquet(output_path, index=False)
                logger.info(
                    f"  [{scheme}] 已保存: {output_path.name} "
                    f"({len(df_scheme)}行×"
                    f"{len(df_scheme.columns)}列)")

        except Exception as e:
            import traceback
            logger.error(f"{yyyymm} 处理失败: {e}")
            logger.error(traceback.format_exc())
            continue

        finally:
            gc.collect()

    logger.info("M0流程完成")


def _next_month_start(yyyymm: str) -> str:
    dt = datetime.strptime(yyyymm, "%Y%m")
    if dt.month == 12:
        return datetime(dt.year+1, 1, 1).strftime("%Y%m%d")
    return datetime(dt.year, dt.month+1, 1).strftime("%Y%m%d")


def _build_returns_matrix(
    df: pd.DataFrame,
    daily_data_dict: dict,
    lookback_days: int = 60,
) -> np.ndarray:
    """
    为方案 G 构建 (N, lookback_days) log-return 矩阵（纯 NumPy）。

    窗口定义（严格防前瞻偏差）：
      T = 当月最后交易日 last_trade_date
      窗口 = [T-60, T-1]，60 个交易日的 log-return
      log_return[i] = log(close[i] / close[i+1])  for i = 0..59

    参数:
        df: 单月截面 DataFrame，含 stock_code 列
        daily_data_dict: {ts_code: DataFrame}，
                         DataFrame 含 close 列，按日期降序
        lookback_days: 60（默认）

    返回:
        np.ndarray  shape=(N, lookback_days)
        缺失数据的股票对应行 = NaN（scheme_g 内部 drop/impute 决策）

    Fail-Loudly:
        df 缺 stock_code → raise KeyError
        daily_data_dict 缺某 ts_code → 该行全 NaN
    """
    if "stock_code" not in df.columns:
        raise KeyError("df missing 'stock_code' column")

    N = len(df)
    L = int(lookback_days)
    mat = np.full((N, L), np.nan, dtype=np.float64)

    for i, code in enumerate(df["stock_code"].values):
        ddf = daily_data_dict.get(code)
        if ddf is None or ddf.empty or "close" not in ddf.columns:
            continue
        closes = ddf["close"].values.astype(np.float64)
        # 需要 L+1 个收盘价
        if len(closes) < L + 1:
            continue
        # 头部 L+1 个 close（已按日期降序）
        head = closes[: L + 1]
        # 过滤非正值（停牌/异常），整行 NaN
        if (head <= 0).any() or not np.isfinite(head).all():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.log(head[:-1] / head[1:])   # (L,)
        if not np.isfinite(log_ret).all():
            continue
        mat[i, :] = log_ret

    return mat


def _get_daily_data_dict_for_neutralize(
    fetcher,
    ts_codes: list,
    yyyymm: str,
) -> dict:
    """
    Phase 2 (neutralize) 专用：仅当方案 G 需要 returns_matrix 时
    才从 Tushare 拉日线（其它方案不需要）

    内存优化：拉完就丢，传给 _build_returns_matrix 后即可 del
    """
    cal = fetcher.get_trade_calendar(yyyymm + "01", yyyymm + "31")
    if not cal:
        return {}
    last_trade_date = cal[-1]
    return fetcher.get_daily_data_batch(
        ts_codes,
        last_trade_date,
        n_days=300,
    )

def _next_month_end(yyyymm: str) -> str:
    dt = datetime.strptime(yyyymm, "%Y%m")
    if dt.month == 12:
        return datetime(dt.year+1, 1, 31).strftime("%Y%m%d")
    if dt.month < 11:
        next_month = dt.month + 1
        # 获取下月最后一天
        if next_month in [1,3,5,7,8,10,12]:
            last_day = 31
        elif next_month in [4,6,9,11]:
            last_day = 30
        else:
            last_day = 28  # 简化处理
        return datetime(dt.year, next_month, last_day).strftime("%Y%m%d")
    else:
        return datetime(dt.year, 12, 31).strftime("%Y%m%d")


def calc_misv_fy1_cross_section(
    month_df: pd.DataFrame,
    fetcher,
    period: str,
) -> pd.Series:
    """
    横截面计算MISV_FY1错误定价因子
    公式: MISV_FY1 = (gamma_0 + gamma_1 * NI_FY1/B) / PB - 1
    使用NI_FY1代理: 当期净利润 * (1 + 营收增长率/100)
    
    参数:
        month_df: 当月因子DataFrame，必须包含value_pb_ratio列
        fetcher: TushareFetcher实例
        period: 财报期（YYYYMMDD格式）
    返回:
        每只股票的MISV_FY1值（Series，index与month_df相同）
    """
    df = month_df.copy()
    
    # 获取净资产数据
    net_equity_map = {}
    for ts_code in df["stock_code"].values:
        try:
            bs = fetcher.get_balance_sheet(ts_code, period)
            if bs and "total_hldr_eqy_exc_min_int" in bs:
                net_equity_map[ts_code] = float(bs["total_hldr_eqy_exc_min_int"])
            else:
                net_equity_map[ts_code] = np.nan
        except Exception:
            net_equity_map[ts_code] = np.nan
    
    df["net_equity"] = df["stock_code"].map(net_equity_map)
    
    # 计算NI_FY1代理值：当期净利润 * (1 + 营收增长率/100)
    # 使用代理值（分析师预期数据不可得时的替代方案）
    # 需要从financial_data中获取净利润和营收增长率
    # 简化：用PB和ROE推算净利润 = ROE * 净资产
    df["ni_fy1_proxy"] = np.nan
    if "quality_roe" in df.columns and "net_equity" in df.columns:
        # NI = ROE(%) / 100 * B
        roe_vals = df["quality_roe"].values if "quality_roe" in df.columns else np.full(len(df), np.nan)
        b_vals = df["net_equity"].values
        # 营收增长率
        rev_yoy = df["growth_revenue_yoy"].values if "growth_revenue_yoy" in df.columns else np.full(len(df), np.nan)
        
        for i in range(len(df)):
            roe_i = roe_vals[i] if np.isfinite(roe_vals[i]) else np.nan
            b_i = b_vals[i] if np.isfinite(b_vals[i]) else np.nan
            rev_i = rev_yoy[i] if np.isfinite(rev_yoy[i]) else 0.0  # 默认0增长
            
            if np.isfinite(roe_i) and np.isfinite(b_i) and b_i > 0:
                ni_current = roe_i / 100.0 * b_i  # 当期净利润
                ni_fy1 = ni_current * (1.0 + rev_i / 100.0)  # FY1代理
                df.iloc[i, df.columns.get_loc("ni_fy1_proxy")] = ni_fy1
    
    # 准备回归数据
    # 因变量：PB（即V/B）
    y = df["value_pb_ratio"].values
    
    # 自变量：NI_FY1 / B
    df["ni_over_b"] = df["ni_fy1_proxy"] / df["net_equity"]
    
    # 过滤：去掉NaN和极端值
    mask = (
        pd.notna(df["value_pb_ratio"]) &
        pd.notna(df["ni_over_b"]) &
        (df["value_pb_ratio"] > 0) &
        (df["ni_over_b"].abs() < 10)  # 截断极端杠杆
    )
    df_reg = df[mask]
    
    if len(df_reg) < 30:  # 样本不足无法回归
        return pd.Series(np.nan, index=df.index)
    
    x = df_reg["ni_over_b"].values
    y_reg = df_reg["value_pb_ratio"].values
    
    # OLS横截面回归
    x_mean = np.mean(x)
    y_mean = np.mean(y_reg)
    numerator = np.sum((x - x_mean) * (y_reg - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if abs(denominator) < 1e-10:
        return pd.Series(np.nan, index=df.index)
    
    gamma_1 = numerator / denominator
    gamma_0 = y_mean - gamma_1 * x_mean
    
    # 计算R²
    y_hat = gamma_0 + gamma_1 * x
    ss_res = np.sum((y_reg - y_hat) ** 2)
    ss_tot = np.sum((y_reg - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    logger.info(f"  MISV_FY1回归: gamma_0={gamma_0:.4f}, gamma_1={gamma_1:.4f}, R²={r_squared:.4f}")
    
    # 计算每只股票的MISV_FY1
    result = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        pb = df.iloc[i].get("value_pb_ratio", np.nan)
        ni_b = df.iloc[i].get("ni_over_b", np.nan)
        if pd.isna(pb) or pd.isna(ni_b) or pb <= 0:
            continue
        v_hat_over_b = gamma_0 + gamma_1 * ni_b
        misv = v_hat_over_b / pb - 1.0
        # 截断极端值
        result.iloc[i] = float(np.clip(misv, -5.0, 5.0))
    
    return result


if __name__ == "__main__":
    # 全量重新生成（断点续跑，跳过已存在的文件）
    run_pipeline(
        schemes=["scheme_d", "scheme_a", "scheme_e", "scheme_b"],
        months=None,          # 全部228个月
        force_rebuild=False,  # 断点续跑（跳过已存在的文件）
    )
