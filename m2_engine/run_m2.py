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
        pred_date = pred_df["trade_date"].iloc[0]
        if pred_date in date_return_map:
            portfolio["Target_Return_1M"] = (
                portfolio["stock_code"].map(
                    date_return_map[pred_date]))

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
        stats["failed"] = pred_month
        logger.error(
            f"窗口{pred_month}失败: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
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
    scheme:       Optional[str] = None,   # ★ 新增：DataLoader scheme
    disable_penalty: bool = False,        # ★ 新增：禁用IC惩罚
) -> Tuple[pd.DataFrame, Dict]:
    """
    M2主循环

    ★ xgbm_params：第二参数名永远不能改
    ★ gpu_mode：None=自动检测，True=强制GPU，False=强制CPU
    """
    start_time = time.time()

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
            cols_to_drop = [col for col in ["stock_name", "industry", "list_date"]
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
            effective_count = max(len(all_months) - WINDOW_SIZE + 1, 0)
            if verbose:
                print(f"全量模式: 共{len(all_months)}个月份，"
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

    def _get_mem_gb():
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3

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
        )
        results.append(window_result)
        processed += 1

        gc.collect()  # 释放窗口处理产生的内存碎片
        if verbose and processed % 20 == 0:
            print(f"  进度: {processed}/{effective_count} "
                  f"内存={_get_mem_gb():.1f}GB")

        if processed % 20 == 0:
            try:
                from m5_optimizer.utils.rolling_logger import (
                    get_rolling_logger, _try_log)
                _try_log(get_rolling_logger().log_window_progress,
                         processed, effective_count, _get_mem_gb())
            except Exception:
                pass

        if processed >= effective_count:
            break

    # 释放factor_df
    try:
        del factor_df
        gc.collect()
    except NameError:
        pass

    # ── 汇总结果 ──────────────────────────────────
    for portfolio, s in results:
        if s["success"]:
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
                      "pct_positive_excess",
                      "ir_worst_quartile",
                      "val_rolling6m_excess",
                      "val_rolling6m_excess_ann",
                      "up_capture_ratio",
                      "down_capture_ratio",
                      "capture_ratio",
                      "val_jensen_alpha",
                      "val_appraisal_ratio",
                      "val_beta"]
        }
    else:
        run_stats["avg_val_portfolio_metrics"] = {
            "val_rolling6m_ir":      0.0,
            "val_rolling6m_dir":     0.0,
            "val_rolling6m_sortino": 0.0,
            "val_rolling6m_return":  0.0,
            "val_global_ir":         0.0,
            "val_annual_return":     0.0,
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
        }

    # ── 输出文件 ──────────────────────────────────
    if not all_records:
        raise RuntimeError(
            f"所有{run_stats['total_windows']}个窗口均失败，"
            f"请检查日志")

    Path("output").mkdir(exist_ok=True)
    all_portfolios = pd.concat(all_records, ignore_index=True)
    del all_records  # ★ 修复: concat 后立即释放列表，避免内存翻倍
    all_portfolios.to_parquet(
        "output/all_portfolios.parquet", index=False)

    holdings = all_portfolios[all_portfolios["is_holding"]]
    holdings.to_csv(
        "output/all_portfolios.csv",
        index=False, encoding="utf-8-sig")

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
    return all_portfolios, run_stats


if __name__ == "__main__":
    all_portfolios, stats = run_m2(
        fast_mode=True,
        fast_window_count=3,
        compute_shap=False,
        verbose=True,
    )
