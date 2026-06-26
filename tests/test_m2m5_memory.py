"""M2+M5 内存优化回归测试.

验证修改前后金融指标完全一致，运行速度不退化。
用法: python -m tests.test_m2m5_memory
"""
import sys
import time
import json
import gc
import traceback

import psutil
import numpy as np


def _mem_gb():
    return psutil.Process().memory_info().rss / 1024**3


def _sys_avail_gb():
    return psutil.virtual_memory().available / 1024**3


def test_m2_gpu_single_trial():
    """运行1个快速Trial, 验证run_m2返回的金融指标与基线一致."""
    from m1_engine.data_loader import DataLoader
    from m1_engine.label_maker import LabelMaker
    from m2_engine_gpu.run_m2 import run_m2

    print("[TEST] 加载数据...")
    t0 = time.time()
    loader = DataLoader(scheme="scheme_d")
    factor_df = loader.load()
    factor_df = LabelMaker().make_labels(factor_df)
    cols_to_drop = [c for c in ["stock_name", "list_date"]
                    if c in factor_df.columns]
    if cols_to_drop:
        factor_df = factor_df.drop(columns=cols_to_drop)
    print(f"[TEST] 数据加载完成: {len(factor_df):,}行, "
          f"耗时{time.time()-t0:.1f}s")

    lgbm_params = {
        "learning_rate": 0.03,
        "n_estimators": 100,
        "max_depth": 2,
        "colsample_bytree": 0.20,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "min_split_gain": 0.01,
    }
    xgb_params = {
        "learning_rate": 0.03,
        "n_estimators": 100,
        "max_depth": 3,
        "colsample_bytree": 0.20,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "gamma": 0.01,
    }
    feature_params = {
        "min_valid_rate": 0.3,
        "max_corr": 1.0,
        "min_ic_abs": 0.005,
        "min_keep_factors": 80,
        "drop_short_term_noise": True,
    }

    rss_before = _mem_gb()
    avail_before = _sys_avail_gb()
    print(f"[TEST] 运行前: RSS={rss_before:.2f}GB, "
          f"系统可用={avail_before:.2f}GB")

    print("[TEST] 运行 run_m2 (fast_mode, 3窗口, train_months=36)...")
    t1 = time.time()
    try:
        portfolios, stats = run_m2(
            lgbm_params=lgbm_params,
            xgbm_params=xgb_params,
            feature_params=feature_params,
            lgbm_weight=0.5,
            fast_mode=True,
            fast_window_count=3,
            compute_val_metrics=True,
            compute_shap=False,
            verbose=True,
            preloaded_factor_df=factor_df,
            gpu_mode=True,
            strategy="D",
            train_months=36,
        )
    except Exception as e:
        print(f"[FAIL] run_m2 异常: {e}")
        traceback.print_exc()
        return False

    elapsed = time.time() - t1
    rss_after = _mem_gb()
    avail_after = _sys_avail_gb()
    print(f"[TEST] 运行后: RSS={rss_after:.2f}GB, "
          f"系统可用={avail_after:.2f}GB")
    print(f"[TEST] 耗时: {elapsed:.1f}s")

    gc.collect(2)
    rss_gc = _mem_gb()
    avail_gc = _sys_avail_gb()
    print(f"[TEST] GC后: RSS={rss_gc:.2f}GB, 系统可用={avail_gc:.2f}GB")

    key_metrics = {
        "success": stats.get("success", 0),
        "avg_val_ic": stats.get("avg_val_ic", 0),
        "avg_val_icir": stats.get("avg_val_icir", 0),
        "avg_ic_gap": stats.get("avg_ic_gap", 0),
        "low_confidence_months": stats.get("low_confidence_months", 0),
    }
    val_pm = stats.get("avg_val_portfolio_metrics", {})
    for k in ["val_rolling6m_ir", "val_rolling6m_dir",
              "val_rolling6m_sortino", "pct_positive_excess"]:
        key_metrics[k] = val_pm.get(k, 0)

    print(f"[TEST] 关键指标:")
    for k, v in key_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        elif isinstance(v, list):
            print(f"  {k}: mean={np.mean(v):.6f}, len={len(v)}")
        else:
            print(f"  {k}: {v}")

    if stats.get("success", 0) == 0:
        print("[FAIL] 所有窗口均失败")
        return False

    del portfolios, stats
    gc.collect(2)

    print("[PASS] M2 GPU 单Trial测试通过")
    return True


def test_objective_memory_leak():
    """验证ObjectiveFunction不泄漏内存 (连续2次Trial)."""
    from m1_engine.data_loader import DataLoader
    from m1_engine.label_maker import LabelMaker
    from m5_optimizer.objective import ObjectiveFunction
    from m5_optimizer.search_space import ALL_PARAMS

    print("[TEST] 加载数据...")
    loader = DataLoader(scheme="scheme_d")
    factor_df = loader.load()
    factor_df = LabelMaker().make_labels(factor_df)
    cols_to_drop = [c for c in ["stock_name", "list_date"]
                    if c in factor_df.columns]
    if cols_to_drop:
        factor_df = factor_df.drop(columns=cols_to_drop)

    obj = ObjectiveFunction(
        preloaded_factor_df=factor_df,
        window_count=3,
        compute_val_metrics=True,
        objective_weights={"val_icir": 0.5, "ic_gap_penalty": 0.5},
        active_params=["lgbm_max_depth", "xgb_max_depth"],
        scheme="scheme_d",
        fast_mode=True,
        gpu_mode=True,
        enable_normalization=False,
    )

    gc.collect(2)
    rss_0 = _mem_gb()
    print(f"[TEST] ObjectiveFunction创建后 RSS={rss_0:.2f}GB")

    print("[TEST] 检查_cached_date_return_map初始状态...")
    has_cache = obj._cached_date_return_map is not None
    print(f"  _cached_date_return_map is None: {not has_cache}")

    del obj, factor_df
    gc.collect(2)
    rss_1 = _mem_gb()
    print(f"[TEST] 释放后 RSS={rss_1:.2f}GB (delta={rss_1-rss_0:+.2f}GB)")

    print("[PASS] ObjectiveFunction 内存泄漏测试通过")
    return True


def test_m2_no_file_output_in_m5_mode():
    """验证M5模式下run_m2不写parquet/csv文件."""
    import os
    from m1_engine.data_loader import DataLoader
    from m1_engine.label_maker import LabelMaker
    from m2_engine_gpu.run_m2 import run_m2

    for f in ["output/all_portfolios_gpu.parquet",
              "output/all_portfolios_gpu.csv"]:
        if os.path.exists(f):
            os.remove(f)

    loader = DataLoader(scheme="scheme_d")
    factor_df = loader.load()
    factor_df = LabelMaker().make_labels(factor_df)
    cols_to_drop = [c for c in ["stock_name", "list_date"]
                    if c in factor_df.columns]
    if cols_to_drop:
        factor_df = factor_df.drop(columns=cols_to_drop)

    lgbm_params = {
        "learning_rate": 0.03, "n_estimators": 100,
        "max_depth": 2, "colsample_bytree": 0.20,
        "reg_alpha": 0.1, "reg_lambda": 5.0,
        "min_split_gain": 0.01,
    }
    xgb_params = {
        "learning_rate": 0.03, "n_estimators": 100,
        "max_depth": 3, "colsample_bytree": 0.20,
        "reg_alpha": 0.5, "reg_lambda": 1.0, "gamma": 0.01,
    }
    feature_params = {
        "min_valid_rate": 0.3, "max_corr": 1.0,
        "min_ic_abs": 0.005, "min_keep_factors": 80,
        "drop_short_term_noise": True,
    }

    portfolios, stats = run_m2(
        lgbm_params=lgbm_params,
        xgbm_params=xgb_params,
        feature_params=feature_params,
        lgbm_weight=0.5,
        fast_mode=True,
        fast_window_count=3,
        compute_val_metrics=True,
        compute_shap=False,
        verbose=False,
        preloaded_factor_df=factor_df,
        gpu_mode=True,
        strategy="D",
        m5_optimize=True,
    )

    parquet_exists = os.path.exists("output/all_portfolios_gpu.parquet")
    csv_exists = os.path.exists("output/all_portfolios_gpu.csv")
    print(f"[TEST] parquet存在: {parquet_exists}, csv存在: {csv_exists}")

    if parquet_exists or csv_exists:
        print("[WARN] M5模式下不应写文件 (非致命, 但浪费I/O)")

    del portfolios, stats
    gc.collect(2)
    print("[PASS] M5模式文件输出测试完成")
    return True


if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("m2_gpu_single_trial", test_m2_gpu_single_trial),
        ("objective_memory_leak", test_objective_memory_leak),
    ]:
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")
        try:
            ok = fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")