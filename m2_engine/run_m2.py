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
M2主循环入口

★ 关键接口约束（禁止修改）：
第二参数必须是 xgbm_params，永远不能改成 xgb_params

GPU/CPU策略：
* 从GPUConfig单例读取当前模式
* GPU模式：窗口串行，模型GPU串行（显存限制）
* CPU模式：窗口threading并行，模型CPU并行

内存策略：
* 惰性切片：不预加载全量windows
* FeatureStore结果缓存（线程安全）
* 每窗口结束强制gc.collect()
"""
import gc
import json
import threading
import yaml
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
from pathlib import Path
from typing import Tuple, Dict, Optional
import time
import sys
import os
import traceback
import logging

# 自适应并发配置
config_path = (Path(__file__).parent.parent /
               "config" / "concurrency_config.py")
if config_path.exists():
    sys.path.insert(0, str(config_path.parent))
    from concurrency_config import GLOBAL_N_JOBS_OUTER
    sys.path.pop(0)
else:
    GLOBAL_N_JOBS_OUTER = max(
        (os.cpu_count() or 4) // 3, 1)

# 业务模块导入放在自适应并发配置之后
# (m1_engine.* 与 m2_engine.* 之间的相互依赖要求 sys.path 就绪)
from m1_engine.data_loader import DataLoader  # noqa: E402
from m1_engine.rolling_splitter import RollingSplitter  # noqa: E402
from m1_engine.label_maker import LabelMaker  # noqa: E402
from m2_engine.feature_store import FeatureStore  # noqa: E402
from m2_engine.ensemble import EnsemblePredictor  # noqa: E402
from m2_engine.portfolio_builder import PortfolioBuilder  # noqa: E402
from m2_engine.gpu_detector import GPUConfig  # noqa: E402

logger = logging.getLogger("m2.run")

# ── 线程安全FeatureStore缓存 ──────────────────────
_FEATURE_CACHE: Dict = {}
_FEATURE_CACHE_MAX_SIZE = 0        # 运行时动态设置（0=禁用）
_FEATURE_CACHE_LOCK = threading.Lock()
_FEATURE_CACHE_PARAMS_HASH: Optional[int] = None

# ★ v5.1: 首个窗口的失败异常 (供 run_m2 汇总时输出)
_FIRST_ERROR: Optional[str] = None


def _process_single_window(
    i:            int,
    window:       Dict,
    cfg:          Dict,
    feature_params: Optional[Dict],
    lgbm_params:  Optional[Dict],
    xgb_params:   Optional[Dict],
    lgbm_weight:  Optional[float],
    compute_shap: bool,
    date_return_map: Dict,
    compute_val_metrics: bool = True,
    disable_penalty: bool = False,
    stop_event=None,  # 窗口内可响应停止信号
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """处理单个窗口"""
    global _FEATURE_CACHE, _FEATURE_CACHE_PARAMS_HASH

    train_p = val_p = pred_p = predictor = result = None

    train_df = window["train_df"]
    val_df   = window["val_df"]
    pred_df  = window["pred_df"]
    pred_month = window["pred_month"]

    stats = {
        "success":     False,
        "val_ic":      0.0,
        "ic_gap":      0.0,
        "is_penalized":False,
        "val_metrics": {},
        "failed":      None,
    }

    try:
        # 窗口内 stop_event 检查：FeatureStore 之前
        if stop_event is not None and stop_event.is_set():
            stats["failed"] = pred_month
            return None, stats

        fp = feature_params or {}
        cache_key = (i, _FEATURE_CACHE_PARAMS_HASH)

        # 读缓存
        with _FEATURE_CACHE_LOCK:
            cached = _FEATURE_CACHE.get(cache_key)

        if cached is not None:
            train_p, val_p, pred_p, feature_cols = cached
        else:
            fs = FeatureStore(
                min_valid_rate=fp.get(
                    "min_valid_rate",
                    cfg["m2"]["feature_store"]["min_valid_rate"]),
                max_corr=fp.get(
                    "max_corr",
                    cfg["m2"]["feature_store"]["max_corr"]),
                min_ic_abs=fp.get(
                    "min_ic_abs",
                    cfg["m2"]["feature_store"]["min_ic_abs"]),
                min_keep_factors=fp.get(
                    "min_keep_factors",
                    cfg["m2"]["feature_store"].get(
                        "min_keep_factors", 50)),
                drop_short_term_noise=fp.get(
                    "drop_short_term_noise", False),
            )
            train_p, val_p, pred_p, feature_cols = (
                fs.fit_transform(train_df, val_df, pred_df))

            # 写缓存（线程安全）
            with _FEATURE_CACHE_LOCK:
                if len(_FEATURE_CACHE) < _FEATURE_CACHE_MAX_SIZE:
                    _FEATURE_CACHE[cache_key] = (
                        train_p, val_p, pred_p, feature_cols)

        # 窗口内 stop_event 检查：训练之前
        if stop_event is not None and stop_event.is_set():
            stats["failed"] = pred_month
            return None, stats

        eff_weight = (lgbm_weight if lgbm_weight is not None
                      else cfg["m2"]["ensemble"]["lgbm_weight"])

        predictor = EnsemblePredictor(
            lgbm_weight=eff_weight,
            xgb_weight=1.0 - eff_weight,
            lgbm_params=lgbm_params,
            xgb_params=xgb_params,
        )
        result = predictor.fit_predict(
            train_p, val_p, pred_p, feature_cols)

        # 窗口内 stop_event 检查：PortfolioBuilder 之前
        if stop_event is not None and stop_event.is_set():
            stats["failed"] = pred_month
            return None, stats

        pb = PortfolioBuilder()
        portfolio = pb.build(
            pred_df_with_scores=result["pred_df_with_scores"],
            val_ic=result["val_ic"],
            ic_gap=result["ic_gap"],
            is_penalized=result["is_penalized"],
            lgbm_model=predictor.lgbm,
            feature_cols=feature_cols,
            pred_month=pred_month,
            compute_shap=compute_shap,
        )

        if portfolio is None:
            stats["failed"] = pred_month
            logger.warning(f"窗口{pred_month}: 股票池不足15只，跳过")
            return None, stats

        # 回填收益
        # ★ v5.5 修复: 股票收益率获取失败的几个
        #   - 原因 1: pred_date 不在 date_return_map 里 → Target_Return_1M 列不存在
        #   - 原因 2: stock_code 不在 map 里 → 列为 NaN
        #   - 修复: 显式建列 (NaN), 并对未匹配的 stock_code 用相邻月 fallback.
        #     同时记 warning 让用户看见是哪些窗口.
        pred_date = pred_df["trade_date"].iloc[0]
        if pred_date in date_return_map:
            portfolio["Target_Return_1M"] = (
                portfolio["stock_code"].map(
                    date_return_map[pred_date]))
        else:
            # ★ Fallback 1: pred_date 不在 map 里, 但 portfolio 有 trade_date 列表
            #   - 用 pred_df 自带的 (它本身有 Target_Return_1M 列)
            #   - 若 pred_df 没有该列, 再尝试用 pred_df 上一月的 map
            if "Target_Return_1M" in pred_df.columns:
                portfolio["Target_Return_1M"] = (
                    portfolio["stock_code"].map(
                        pred_df.set_index("stock_code")[
                            "Target_Return_1M"].to_dict()))
                logger.warning(
                    f"窗口{pred_month}: pred_date={pred_date} 不在 "
                    f"date_return_map, 已用 pred_df 内置收益回填"
                    f"（{portfolio['Target_Return_1M'].notna().sum()}"
                    f"/{len(portfolio)} 匹配）")
            else:
                portfolio["Target_Return_1M"] = np.nan
                logger.warning(
                    f"窗口{pred_month}: pred_date={pred_date} 不在 "
                    f"date_return_map 且 pred_df 无 Target_Return_1M, "
                    f"该月收益全为 NaN")
        # ★ Fallback 2: 某些 stock_code 没匹配上 (NaN), 用 pred_df 的全量 map
        #   二次补充 (针对 stock-level 缺失)
        if (portfolio["Target_Return_1M"].isna().any()
                and "Target_Return_1M" in pred_df.columns):
            extra = pred_df.set_index("stock_code")[
                "Target_Return_1M"].to_dict()
            mask = portfolio["Target_Return_1M"].isna()
            if mask.any():
                portfolio.loc[mask, "Target_Return_1M"] = (
                    portfolio.loc[mask, "stock_code"].map(extra))
                still_na = portfolio["Target_Return_1M"].isna().sum()
                if still_na > 0:
                    logger.warning(
                        f"窗口{pred_month}: {still_na} 只股票二次 "
                        f"回填后仍无收益")

        # 计算滚动指标（compute_val_metrics=True时）
        # ★ 注意：fit_predict内部已过滤掉label_rank=NaN的行，
        #   result["val_pred_*"] 长度 = 非NaN行数。
        #   为严格保持原行为（val_scored包含所有行），
        #   这里仍然用直接.predict()对全行预测。
        if compute_val_metrics:
            try:
                val_scored = val_p.copy(deep=False)
                val_scored["score"] = (
                    predictor.lgbm_weight *
                    predictor.lgbm.predict(val_p[feature_cols]) +
                    predictor.xgb_weight *
                    predictor.xgb.predict(val_p[feature_cols])
                )
                stats["val_metrics"] = (
                    predictor.compute_val_portfolio_metrics(
                        val_scored))
            except Exception as e:
                logger.warning(f"滚动指标计算失败: {e}")

        stats.update({
            "success":      True,
            "val_ic":       result["val_ic"],
            "ic_gap":       result["ic_gap"],
            "is_penalized": result["is_penalized"],
        })
        return portfolio, stats

    except Exception as e:
        # ★ v5.1 修复: failed 字段存详细异常 (原版只存 pred_month，无法定位失败原因)
        import traceback as _tb
        _tb_str = _tb.format_exc()
        stats["failed"] = f"{pred_month}: {type(e).__name__}: {e}"
        global _FIRST_ERROR
        if _FIRST_ERROR is None:
            _FIRST_ERROR = (
                f"窗口{pred_month} 异常: {type(e).__name__}: {e}\n"
                f"{_tb_str}"
            )
        logger.error(
            f"窗口{pred_month}失败: {type(e).__name__}: {e}")
        logger.error(_tb_str)
        return None, stats

    finally:
        # ★ 显式置None + 显式del，比循环del更明确
        del train_df, val_df, pred_df
        del train_p, val_p, pred_p, predictor, result
        gc.collect()


def run_m2(
    lgbm_params:  Optional[Dict] = None,
    xgbm_params:  Optional[Dict] = None,  # ★ 必须是xgbm_params
    feature_params: Optional[Dict] = None,
    lgbm_weight:  Optional[float] = None,
    fast_mode:    bool = False,
    fast_window_count: int = 60,
    compute_val_metrics: bool = True,
    compute_shap: bool = True,
    verbose:      bool = True,
    preloaded_windows: list = None,
    date_return_map: Dict = None,
    stop_event = None,
    preloaded_factor_df: pd.DataFrame = None,
    gpu_mode:     Optional[bool] = None,  # ★ 新增：None=自动
    train_months: Optional[int] = None,   # ★ 新增：滚动训练窗口月数
    scheme:       Optional[str] = None,
    disable_penalty: bool = False,
    m5_optimize:  bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    M2主循环

    ★ xgbm_params：第二参数名永远不能改
    ★ gpu_mode：None=自动检测，True=强制GPU，False=强制CPU
    """
    start_time = time.time()

    # ★ v5.1 修复: 重置首错缓存
    global _FIRST_ERROR
    _FIRST_ERROR = None

    # ── GPU模式配置 ───────────────────────────────
    gpu_cfg = GPUConfig()
    if gpu_mode is not None:
        gpu_cfg.set_mode("gpu" if gpu_mode else "cpu")

    # ── 自适应缓存大小：快速模式才缓存，全量模式禁用防OOM ──
    global _FEATURE_CACHE, _FEATURE_CACHE_LOCK, _FEATURE_CACHE_MAX_SIZE, _FEATURE_CACHE_PARAMS_HASH
    if fast_mode and fast_window_count <= 20:
        _FEATURE_CACHE_MAX_SIZE = fast_window_count
    else:
        _FEATURE_CACHE_MAX_SIZE = 0  # 全量模式禁用缓存防OOM
    with _FEATURE_CACHE_LOCK:
        _FEATURE_CACHE.clear()

    if verbose:
        print(f"\n{'='*60}")
        print("M2 双引擎选股算法层")
        print(f"{'='*60}")
        print(f"运行模式: {gpu_cfg.summary()}")

    # ── FeatureStore缓存管理 ──────────────────────
    new_hash = (hash(frozenset((feature_params or {}).items()))
                if feature_params else 0)
    if new_hash != _FEATURE_CACHE_PARAMS_HASH:
        with _FEATURE_CACHE_LOCK:
            _FEATURE_CACHE.clear()
            _FEATURE_CACHE_PARAMS_HASH = new_hash

    # ── 加载配置 ──────────────────────────────────
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── 数据加载 ──────────────────────────────────
    if preloaded_windows is not None:
        windows = preloaded_windows
        if date_return_map is None:
            date_return_map = {}
        if verbose:
            print(f"使用预加载数据: {len(windows)}个窗口")
    else:
        if preloaded_factor_df is not None:
            factor_df = preloaded_factor_df
            if verbose:
                print(f"使用预加载factor_df: {len(factor_df):,}行")
        else:
            # ★ 内存优化：分步加载，及时释放中间变量
            loader = DataLoader(scheme=scheme)

            if verbose:
                print("开始加载数据...")

            factor_df = loader.load()

            if verbose:
                print(f"数据加载完成: {len(factor_df):,}行")

            # 立即制作标签（这一步可能产生新列）
            factor_df = LabelMaker().make_labels(factor_df)

            if verbose:
                print(f"标签制作完成，当前列数: {len(factor_df.columns)}")

            # ★ 内存优化：删除不必要的元信息列以减少内存占用
            #   ★ 改造（M4 归因需要）: 保留 stock_name/industry 到 pred_month→{code→industry}
            #     字典里，落盘时再回填。这样 M2 训练阶段内存不增加，但 M4 能拿到行业。
            cols_to_drop = [col for col in ["list_date"]
                           if col in factor_df.columns]
            if cols_to_drop:
                factor_df = factor_df.drop(columns=cols_to_drop)
                if verbose:
                    print(f"已删除冗余列: {cols_to_drop}")

            gc.collect()

        # 构建date_return_map
        if date_return_map is None:
            ret_df = factor_df[
                ["trade_date","stock_code","Target_Return_1M"]
            ].dropna(subset=["Target_Return_1M"])
            date_return_map = (
                ret_df.groupby("trade_date")
                .apply(lambda g: dict(zip(
                    g["stock_code"].values,
                    g["Target_Return_1M"].values)))
                .to_dict()
            )
            del ret_df
            gc.collect()

        splitter = RollingSplitter(train_months=train_months)

        # ★ fast_mode预切片：直接裁剪factor_df，splitter无需跳过任何窗口
        _cfg_train = train_months or 36
        WINDOW_SIZE = _cfg_train + 12 + 1  # train+valid+test，单个窗口所需月数

        if fast_mode:
            all_months = sorted(factor_df["trade_date"].unique())
            total_months_count = len(all_months)
            # N个窗口需要 WINDOW_SIZE + N - 1 个月
            months_needed = WINDOW_SIZE + fast_window_count - 1
            months_needed = min(months_needed, total_months_count)
            # 只保留最后months_needed个月
            keep_months = frozenset(all_months[-months_needed:])
            factor_df = factor_df[
                factor_df["trade_date"].isin(keep_months)
            ].copy()
            effective_count = min(fast_window_count,
                                  total_months_count - WINDOW_SIZE + 1)
            if verbose:
                print(f"快速模式: 预切片保留最后{months_needed}个月，"
                      f"预计{effective_count}个窗口")
        else:
            all_months = sorted(factor_df["trade_date"].unique())
            total_months_count = len(all_months)
            full_count = max(total_months_count - WINDOW_SIZE + 1, 0)
            # ★ 强制155窗口：只保留最后155个窗口对应的数据
            FIXED_WINDOW_COUNT = 155
            if full_count > FIXED_WINDOW_COUNT:
                months_needed = WINDOW_SIZE + FIXED_WINDOW_COUNT - 1
                keep_months = frozenset(all_months[-months_needed:])
                factor_df = factor_df[
                    factor_df["trade_date"].isin(keep_months)
                ].copy()
                effective_count = FIXED_WINDOW_COUNT
            else:
                effective_count = full_count
            if verbose:
                print(f"全量模式: 强制{FIXED_WINDOW_COUNT}窗口，"
                      f"预计{effective_count}个窗口")

    # ── 主循环：惰性窗口，串行处理 ──────────────────────
    all_records = []
    run_stats = {
        "total_windows":   effective_count,
        "success":         0,
        "failed":          [],
        "low_confidence_months": 0,  # 改名：低置信度月份
        "avg_val_ic":      [],
        "avg_ic_gap":      [],
        "val_metrics_list":[],
    }

    # ── 内存监控（仅用于进度打印）────────────────
    import psutil

    # ★ v5.1 修复: 全量模式渐进式 OOM 防御
    # 原因: train_months=50 全量回测时，内存可能持续增长直到 Windows 强杀进程
    # 方案: 每 10 个窗口检查一次可用内存，低于阈值时主动中止
    _OOM_AVAIL_LIMIT_GB = 0.8  # 系统可用内存低于此值时中止（留 0.3GB 给系统）

    def _get_mem_gb():
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3

    def _check_oom():
        """检查系统可用内存，低于阈值返回警告信息，否则返回 None"""
        try:
            _avail = psutil.virtual_memory().available / (1024**3)
            if _avail < _OOM_AVAIL_LIMIT_GB:
                return (
                    f"OOM_ABORT | 系统可用内存={_avail:.1f}GB < "
                    f"{_OOM_AVAIL_LIMIT_GB}GB，中止回测防止 Windows 强杀"
                )
        except Exception:
            pass
        return None

    # ★ B1 优化: BATCH_SIZE=1 + Parallel(threading) 等价裸 for,
    #   joblib 包装每窗徒增 ~3-8ms 调度开销。改用直接函数调用。
    #   顺带删 _safe_parallel_jobs / n_jobs / mem_now 降级逻辑（已死代码）
    results = []
    processed = 0

    for i, window in enumerate(splitter.split(factor_df, copy=False)):
        if stop_event and stop_event.is_set():
            break

        window_result = _process_single_window(
            processed + i, window, cfg, feature_params,
            lgbm_params, xgbm_params, lgbm_weight, compute_shap,
            date_return_map, compute_val_metrics,
            disable_penalty=disable_penalty,
            stop_event=stop_event,
        )
        results.append(window_result)
        processed += 1

        gc.collect()  # 释放窗口处理产生的内存碎片
        # 每个窗口都输出进度（旧版每 20 窗才输出，挂死时无法定位）
        if verbose:
            _w_status = "✓" if window_result[0] is not None else "✗"
            if processed % 10 == 0 or processed <= 2:
                print(f"  窗口{processed}/{effective_count} "
                      f"{_w_status} pred={window['pred_month']} "
                      f"内存={_get_mem_gb():.1f}GB")

        if processed % 10 == 0:
            try:
                from m5_optimizer.utils.rolling_logger import (
                    get_rolling_logger, _try_log)
                _try_log(get_rolling_logger().log_window_progress,
                         processed, effective_count, _get_mem_gb())
            except Exception:
                pass

        # ★ v5.1 修复: 每 10 窗检查系统可用内存，防止 OOM 导致 Windows 强杀
        if processed % 10 == 0:
            _oom_msg = _check_oom()
            if _oom_msg is not None:
                logger.critical(_oom_msg)
                if verbose:
                    print(f"  ⚠️ {_oom_msg}")
                break

        if processed >= effective_count:
            break

    # 释放factor_df
    try:
        del factor_df
        gc.collect()
    except NameError:
        pass

    # ── 汇总结果 ──────────────────────────────────
    shap_data: Dict[str, tuple] = {}   # ★ 改造: 收集每月完整 SHAP
    for portfolio, s in results:
        if s["success"]:
            # ★ 改造: 提取完整 SHAP（attrs 里的 numpy 数组）
            #   然后清空 attrs，避免 pd.concat 时比较 numpy 数组炸掉。
            #   top3 SHAP 列（shap_top1/2/3_factor/value）已在
            #   portfolio_builder.build() 里写入 DataFrame，足够 M4 用。
            if (compute_shap
                and hasattr(portfolio, "attrs")
                and "shap_values" in portfolio.attrs):
                m = str(portfolio["pred_month"].iloc[0]) \
                    if "pred_month" in portfolio.columns else "?"
                shap_data[m] = (
                    portfolio.attrs["shap_values"],
                    portfolio.attrs.get(
                        "shap_features", []),
                    portfolio.attrs.get(
                        "shap_source", "shap.TreeExplainer"),
                )
                portfolio.attrs.clear()
            all_records.append(portfolio)
            run_stats["success"] += 1
            run_stats["avg_val_ic"].append(s["val_ic"])
            run_stats["avg_ic_gap"].append(s["ic_gap"])
            if s["is_penalized"]:
                run_stats["low_confidence_months"] += 1
            if s.get("val_metrics"):
                run_stats["val_metrics_list"].append(
                    s["val_metrics"])
        else:
            run_stats["failed"].append(s["failed"])

    # ★ 修复: 汇总完立即释放 results 列表
    del results

    # ── 计算val_icir（IC信息比率 = IC均值 / IC标准差）──────────
    ic_list = run_stats["avg_val_ic"]
    if len(ic_list) >= 2:
        ic_arr = np.array(ic_list, dtype=np.float32)
        ic_mean = float(ic_arr.mean())
        ic_std = float(ic_arr.std())
        if ic_std < 1e-4 or len(ic_arr) < 2:
            run_stats["avg_val_icir"] = 0.0
        else:
            val_icir = ic_mean / ic_std
            val_icir = float(np.clip(val_icir, -5.0, 5.0))
            run_stats["avg_val_icir"] = val_icir
    else:
        run_stats["avg_val_icir"] = 0.0

    # ── 计算滚动指标均值 ─────────────────────────
    if compute_val_metrics and run_stats["val_metrics_list"]:
        vml = run_stats["val_metrics_list"]
        run_stats["avg_val_portfolio_metrics"] = {
            k: float(_mean([v.get(k, 0) for v in vml]))
            for k in ["val_rolling6m_ir",
                      "val_rolling6m_dir",
                      "val_rolling6m_sortino",
                      "val_rolling6m_return",
                      "val_global_ir",
                      "val_annual_return",
                      "val_net_annual_return",
                      "pct_positive_excess",
                      "ir_worst_quartile",
                      "val_rolling6m_excess",
                      "val_rolling6m_excess_ann",
                      "up_capture_ratio",
                      "down_capture_ratio",
                      "capture_ratio",
                      "val_jensen_alpha",
                      "val_appraisal_ratio",
                      "val_beta",
                      "val_var_95",
                      "val_cvar_95",
                      "val_pain_index",
                      "val_sqn",
                      "val_ic_stability",
                      "val_ir_stability",
                      "turnover_penalty",
                      ]
        }
        # ★ 全期拼接重算6m_ir（净收益口径，按日期去重，与M4 metrics.py一致）
        # 滚动窗口验证集有大量重叠月份，extend后930行但实际只有155个不重复月
        # 去重策略：同一日期保留最后窗口的值（最后窗口训练数据最多，预测最准）
        _all_rets = []
        _all_bms = []
        _all_costs = []
        _all_dates = []
        for v in vml:
            _all_rets.extend(v.get("_monthly_returns", []))
            _all_bms.extend(v.get("_monthly_benchmarks", []))
            _all_costs.extend(v.get("_monthly_costs", []))
            _all_dates.extend(v.get("_monthly_dates", []))
        if len(_all_rets) >= 6 and len(_all_dates) == len(_all_rets):
            # 按日期去重：后出现的覆盖先出现的（保留最后窗口）
            _dedup = {}
            for i, d in enumerate(_all_dates):
                _dedup[d] = i
            _idx = sorted(_dedup.values())
            _rets = np.array([_all_rets[i] for i in _idx], dtype=np.float64)
            _bms = np.array([_all_bms[i] for i in _idx], dtype=np.float64)
            _costs = np.array([_all_costs[i] for i in _idx], dtype=np.float64)
            _ex = _rets - _bms
            _n = len(_rets)
            _avm = run_stats["avg_val_portfolio_metrics"]

            # ★ 6M_IR（与M4 metrics.py同公式）
            from numpy.lib.stride_tricks import sliding_window_view
            _win_ex = sliding_window_view(_ex, 6)
            _win_ex_mean = _win_ex.mean(axis=1)
            _win_ex_std = win_ex_std = _win_ex.std(axis=1, ddof=1)
            _irs = np.where(
                _win_ex_std > 1e-8,
                _win_ex_mean / np.where(_win_ex_std > 1e-8, _win_ex_std, 1.0) * np.sqrt(12),
                0.0)
            _avm["val_rolling6m_ir"] = float(np.mean(_irs))

            # ★ 6M_DIR
            _dirs = np.empty(_n - 5, dtype=np.float64)
            for j in range(_n - 5):
                _w_ex = _win_ex[j]
                _down_ex = _w_ex[_w_ex < 0]
                _down_std = (float(np.std(_down_ex, ddof=1))
                             if len(_down_ex) > 1
                             else max(float(win_ex_std[j]), 1e-8))
                _dirs[j] = float(_win_ex_mean[j]) / _down_std * np.sqrt(12)
            _avm["val_rolling6m_dir"] = float(np.mean(_dirs))

            # ★ 6M_Sortino
            _rf = 0.03 / 12
            _win_ret = sliding_window_view(_rets, 6)
            _sortinos = np.empty(_n - 5, dtype=np.float64)
            for j in range(_n - 5):
                _w_ret = _win_ret[j]
                _down_ret = _w_ret[_w_ret < _rf]
                _d_std = (float(np.std(_down_ret, ddof=1))
                          if len(_down_ret) > 1
                          else max(float(np.std(_w_ret, ddof=1)), 1e-6))
                _sortinos[j] = float(np.mean(_w_ret - _rf)) / _d_std * np.sqrt(12)
            _avm["val_rolling6m_sortino"] = float(np.mean(_sortinos))

            # ★ 6M_Return
            _cumrets = np.prod(1 + _win_ret, axis=1) - 1
            _avm["val_rolling6m_return"] = float(np.mean(_cumrets))

            # ★ 全局IR
            _ex_std = float(np.std(_ex, ddof=1))
            _avm["val_global_ir"] = (float(np.mean(_ex)) / _ex_std * np.sqrt(12)
                                      if _ex_std > 1e-8 else 0.0)

            # ★ 年化收益（净）
            _cum_ret = np.prod(1 + _rets) - 1
            _avm["val_annual_return"] = ((1 + _cum_ret) ** (12 / _n) - 1
                                          if _n > 0 else 0.0)

            # ★ 扣费后年化收益（_rets 已是净收益，不再重复扣费）
            _net_rets = _rets
            _cum_net = np.prod(1 + _net_rets) - 1
            _avm["val_net_annual_return"] = ((1 + _cum_net) ** (12 / _n) - 1
                                              if _n > 0 else 0.0)

            # ★ 月度超额胜率
            _avm["pct_positive_excess"] = float(np.mean(_ex > 0))

            # ★ 最差四分位IR
            _sorted_ex = np.sort(_ex)
            _cutoff = max(1, _n // 4)
            _worst_q = _sorted_ex[:_cutoff]
            _worst_std = float(np.std(_worst_q, ddof=1))
            _avm["ir_worst_quartile"] = (float(np.mean(_worst_q) / _worst_std)
                                          if _worst_std > 1e-8 else 0.0)

            # ★ 6M滚动超额收益
            if _n >= 6:
                _excess_windows = sliding_window_view(_ex, 6)
                _rolling6_means = _excess_windows.mean(axis=1)
                _avm["val_rolling6m_excess"] = float(np.mean(_rolling6_means))
                _avm["val_rolling6m_excess_ann"] = _avm["val_rolling6m_excess"] * 12

            # ★ 捕获比（与M4同公式，净收益）
            _up_mask = _bms > 0
            _dn_mask = _bms < 0
            if _up_mask.sum() >= 3 and _dn_mask.sum() >= 3:
                _up_cap = float(_rets[_up_mask].mean() / _bms[_up_mask].mean())
                _dn_cap = float(_rets[_dn_mask].mean() / _bms[_dn_mask].mean())
                _avm["up_capture_ratio"] = round(_up_cap, 4)
                _avm["down_capture_ratio"] = round(_dn_cap, 4)
                _avm["capture_ratio"] = round(_up_cap / _dn_cap if abs(_dn_cap) > 1e-6 else 0.0, 4)

            # ★ Jensen Alpha & Appraisal Ratio & Beta（与M4同公式，净收益，最小12月）
            if _n >= 12:
                _rf_m = 0.03 / 12
                _y = _rets - _rf_m
                _x = _bms - _rf_m
                _x_mean = float(_x.mean())
                _y_mean = float(_y.mean())
                _ss_xx = float(((_x - _x_mean) ** 2).sum())
                if _ss_xx > 1e-12 and np.unique(_x).size >= 2:
                    _beta = float(((_x - _x_mean) * (_y - _y_mean)).sum() / _ss_xx)
                    _alpha = _y_mean - _beta * _x_mean
                    _resid = _y - (_alpha + _beta * _x)
                    _sigma = float(_resid.std(ddof=1))
                    _avm["val_jensen_alpha"] = round(_alpha * 12.0, 6)
                    _avm["val_appraisal_ratio"] = round(_alpha / _sigma if _sigma > 1e-8 else 0.0, 6)
                    _avm["val_beta"] = round(_beta, 6)

            # ★ VaR/CVaR/痛苦指数/SQN（与M4同公式）
            _avm["val_var_95"] = float(np.percentile(_rets, 5))
            _var_th = _avm["val_var_95"]
            _avm["val_cvar_95"] = (float(_rets[_rets <= _var_th].mean())
                                    if (_rets <= _var_th).any() else _var_th)
            _cum_arr = np.cumprod(1 + _rets)
            _peak_arr = np.maximum.accumulate(_cum_arr)
            _dd_arr = (_cum_arr - _peak_arr) / _peak_arr
            _avm["val_pain_index"] = float(np.abs(_dd_arr).mean())
            _r_std = float(_rets.std(ddof=1))
            _avm["val_sqn"] = (float(_rets.mean()) / _r_std * np.sqrt(_n)
                                if _r_std > 1e-8 and _n > 1 else 0.0)
    else:
        run_stats["avg_val_portfolio_metrics"] = {
            "val_rolling6m_ir":      0.0,
            "val_rolling6m_dir":     0.0,
            "val_rolling6m_sortino": 0.0,
            "val_rolling6m_return":  0.0,
            "val_global_ir":         0.0,
            "val_annual_return":     0.0,
            "val_net_annual_return": 0.0,
            "pct_positive_excess":   0.0,
            "ir_worst_quartile":     0.0,
            "val_rolling6m_excess":  0.0,
            "val_rolling6m_excess_ann": 0.0,
            "up_capture_ratio":      0.0,
            "down_capture_ratio":    0.0,
            "capture_ratio":         0.0,
            "val_jensen_alpha":      0.0,
            "val_appraisal_ratio":   0.0,
            "val_beta":              0.0,
            "val_var_95":            0.0,
            "val_cvar_95":           0.0,
            "val_pain_index":        0.0,
            "val_sqn":              0.0,
            "val_ic_stability":      0.0,
            "val_ir_stability":      0.0,
            "turnover_penalty":      0.0,
        }

    # ── 输出文件 ──────────────────────────────────
    if not all_records:
        # ★ v5.1 修复: 输出首个详细异常，旧版只说"所有XX窗口均失败"无任何线索
        _err = _FIRST_ERROR or "（无具体异常被捕获）"
        raise RuntimeError(
            f"所有{run_stats['total_windows']}个窗口均失败。"
            f"首个失败原因: {_err}"
        )

    Path("output").mkdir(exist_ok=True)
    all_portfolios = pd.concat(all_records, ignore_index=True)
    del all_records

    if not m5_optimize:
        all_portfolios.to_parquet(
            "output/all_portfolios.parquet", index=False)

        holdings = all_portfolios[all_portfolios["is_holding"]]
        holdings.to_csv(
            "output/all_portfolios.csv",
            index=False, encoding="utf-8-sig")
        del holdings

    # ── 摘要输出 ──────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    avg_ic  = _mean(run_stats["avg_val_ic"])
    avg_gap = _mean(run_stats["avg_ic_gap"])

    if verbose:
        print(f"\n{'='*60}")
        print("M2 回测完成")
        print(f"{'='*60}")
        print(f"运行模式:    {gpu_cfg.mode.upper()}")
        print(f"成功窗口:    {run_stats['success']}/{run_stats['total_windows']}")
        print(f"失败窗口:    {len(run_stats['failed'])}个")
        print(f"平均val_IC:  {avg_ic:.4f}")
        print(f"平均ic_gap:  {avg_gap:.4f}")
        low_conf_rate = run_stats['low_confidence_months'] / max(run_stats['success'], 1)
        print(f"低置信度月份: {run_stats['low_confidence_months']}个({low_conf_rate:.1%})")
        print(f"输出行数:    {len(all_portfolios)}")
        print(f"运行时间:    {elapsed:.1f}分钟")

        fingerprint = {
            "total_portfolios_rows": len(all_portfolios),
            "total_months_succeeded": run_stats["success"],
            "total_months_failed": len(run_stats["failed"]),
            "low_confidence_months": run_stats["low_confidence_months"],
            "low_confidence_rate": round(
                run_stats["low_confidence_months"] /
                max(run_stats["success"], 1), 4),
            "avg_val_ic":  round(float(avg_ic),  4),
            "avg_ic_gap":  round(float(avg_gap), 4),
            "elapsed_minutes": round(elapsed, 1),
            "gpu_mode": gpu_cfg.mode,
        }
        print("\n=== M2指纹 ===")
        print(json.dumps(fingerprint,
                        ensure_ascii=False, indent=2))

    gc.collect()  # 释放最终结果组装产生的内存碎片
    # ★ 改造: 把每月完整 SHAP 挂到 all_portfolios.attrs 上，
    #   供 M4 报告模块在同进程内读取（parquet 落盘后会丢，但
    #   top3 SHAP 列已写到 DataFrame，可重新构造简化版）。
    all_portfolios.attrs["shap_data"] = shap_data
    return all_portfolios, run_stats


if __name__ == "__main__":
    all_portfolios, stats = run_m2(
        fast_mode=True,
        fast_window_count=3,
        compute_shap=False,
        verbose=True,
    )
