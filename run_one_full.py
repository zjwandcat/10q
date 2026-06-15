"""
单策略全量运行 · v2.0
用法: python run_one_full.py <strategy> <n_windows>
   strategy: baseline | A | B | C
   n_windows: 60 | 186 | ...
"""
import sys
import os
import time
import json
import gc
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

import psutil
import numpy as np


def _mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def _vram_mb():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(m.used / 1024**2)
    except Exception:
        return 0


if len(sys.argv) < 3:
    print("Usage: python run_one_full.py <strategy> <n_windows>")
    print("  strategy: baseline | A | B | C")
    print("  n_windows: integer")
    sys.exit(1)

STRATEGY = sys.argv[1].upper()
N_WINDOWS = int(sys.argv[2])

# 显存优化参数
GPU_LGBM_PARAMS = {
    "max_bin":           63,
    "min_data_in_bin":   5,
    "gpu_use_dp":        False,
    "use_quantized_grad":True,
    "gpu_max_memory":    0.6,
}
GPU_XGB_PARAMS = {
    "max_bin":           128,
    "tree_method":       "hist",
}
COMMON_LGBM_PARAMS = {
    "max_depth":              4,
    "learning_rate":          0.05,
    "n_estimators":           200,
    "colsample_bytree":       0.3,
    "reg_alpha":              0.1,
    "reg_lambda":             1.0,
    "min_split_gain":         0.01,
    "early_stopping_rounds":  30,
    "lr_mode":                "fixed",
    "depth_mode":             "fixed",
}
COMMON_XGB_PARAMS = {
    "max_depth":              4,
    "learning_rate":          0.05,
    "n_estimators":           200,
    "colsample_bytree":       0.3,
    "reg_alpha":              0.1,
    "reg_lambda":             1.0,
    "gamma":                  0.01,
    "early_stopping_rounds":  30,
    "lr_mode":                "fixed",
}


def make_params(strategy):
    if strategy == "BASELINE":
        return dict(COMMON_LGBM_PARAMS), dict(COMMON_XGB_PARAMS)
    if strategy == "A":
        return {**COMMON_LGBM_PARAMS, **GPU_LGBM_PARAMS}, dict(COMMON_XGB_PARAMS)
    if strategy == "B":
        return dict(COMMON_LGBM_PARAMS), {**COMMON_XGB_PARAMS, **GPU_XGB_PARAMS}
    if strategy == "C":
        return {**COMMON_LGBM_PARAMS, **GPU_LGBM_PARAMS}, {**COMMON_XGB_PARAMS, **GPU_XGB_PARAMS}
    if strategy == "D":
        # O2: LGBM CPU + XGB CPU train (XGB 预测 GPU 在 EnsemblePredictor 内自动处理)
        return dict(COMMON_LGBM_PARAMS), dict(COMMON_XGB_PARAMS)
    raise ValueError(strategy)


print("=" * 70)
print(f"[run_one_full] strategy={STRATEGY} n_windows={N_WINDOWS}")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PID:   {os.getpid()}")
print(f"RSS:   {_mem_gb():.2f}GB VRAM: {_vram_mb()}MB")
print("=" * 70)

from m2_engine_gpu.gpu_detector import GPUConfig
gpu = GPUConfig()
print(gpu.summary())

if STRATEGY != "BASELINE":
    gpu.set_strategy(STRATEGY)

lgbm_params, xgb_params = make_params(STRATEGY)
print(f"\nLGBM: max_bin={lgbm_params.get('max_bin')}, "
      f"gpu_use_dp={lgbm_params.get('gpu_use_dp')}")
print(f"XGB:  max_bin={xgb_params.get('max_bin')}, "
      f"device={gpu.get_xgb_device() if STRATEGY != 'BASELINE' else 'cpu'}")

from m2_engine_gpu.run_m2 import run_m2

t0 = time.time()
rss0 = _mem_gb()
vram0 = _vram_mb()

try:
    all_pf, stats = run_m2(
        lgbm_params=lgbm_params,
        xgbm_params=xgb_params,
        lgbm_weight=0.5,
        fast_mode=(N_WINDOWS <= 186),
        fast_window_count=N_WINDOWS,
        compute_shap=False,
        compute_val_metrics=False,
        verbose=True,
        gpu_mode=(STRATEGY != "BASELINE"),
        strategy=STRATEGY if STRATEGY != "BASELINE" else None,
        n_window_workers=int(os.environ.get("N_WINDOW_WORKERS", 1)),
    )
    total_time = time.time() - t0
    rss1 = _mem_gb()
    vram1 = _vram_mb()

    ws = stats.get("window_stats", [])
    fit_times = [w["fit_time_s"] for w in ws]
    vrams = [w["vram_mb"] for w in ws]

    result = {
        "strategy":    STRATEGY.lower(),
        "status":      "OK",
        "n_windows":   stats["success"],
        "n_target":    N_WINDOWS,
        "total_time_s": total_time,
        "avg_window_time_s": float(np.mean(fit_times)) if fit_times else 0,
        "min_window_time_s": float(np.min(fit_times)) if fit_times else 0,
        "max_window_time_s": float(np.max(fit_times)) if fit_times else 0,
        "max_vram_mb":  int(np.max(vrams)) if vrams else vram1,
        "avg_vram_mb":  int(np.mean(vrams)) if vrams else vram1,
        "max_rss_gb":   rss1,
        "rss_delta_gb": rss1 - rss0,
        "avg_val_ic":   float(np.mean(stats.get("avg_val_ic", [0]))),
        "failed":       len(stats.get("failed", [])),
        "windows":      ws,
    }
    print(f"\n[OK] Strategy {STRATEGY} done: "
          f"{stats['success']}/{N_WINDOWS} windows, "
          f"{total_time/60:.1f} min, "
          f"Peak VRAM={result['max_vram_mb']}MB, "
          f"Peak RSS={rss1:.1f}GB, "
          f"Avg/Window={result['avg_window_time_s']:.2f}s")

    # Save result
    out = Path(f"output/benchmark/full_{STRATEGY.lower()}_{N_WINDOWS}w.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out}")

except Exception as e:
    elapsed = time.time() - t0
    print(f"\n[FAIL] Strategy {STRATEGY}: {type(e).__name__}: {e}")
    traceback.print_exc()
    out = Path(f"output/benchmark/full_{STRATEGY.lower()}_{N_WINDOWS}w_FAIL.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": STRATEGY.lower(),
            "status": "FAIL",
            "reason": f"{type(e).__name__}: {e}",
            "elapsed_s": elapsed,
        }, f, indent=2, ensure_ascii=False)
    sys.exit(1)
finally:
    gc.collect()
