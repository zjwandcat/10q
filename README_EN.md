# TTHH A-Share Quantitative Stock Selection System (m012345)

> Maintained by [zjwandcat](https://github.com/zjwandcat)
>
> Tushare Data + Six-Layer Pipeline + LightGBM/XGBoost Ensemble Learning + Optuna BO-TPE Bayesian Hyperparameter Optimization
>
> **Runs smoothly on 16 GB laptop** — Extreme memory engineering for full-scale backtesting on low-memory machines

🌐 **Language / Language**: [🇨🇳 中文](README.md) · [🇬🇧 English (current)](README_EN.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078d4.svg)](https://www.microsoft.com/windows/windows-11)
[![Version](https://img.shields.io/badge/Version-4.2-green.svg)](CHANGELOG.md)

**Copyright © 2026 zjwandcat. Licensed under the [Apache License, Version 2.0](LICENSE).**

---

## Table of Contents

- [1. System Overview](#1-system-overview)
- [2. Running Smoothly on Low-Memory Laptops](#2-running-smoothly-on-low-memory-laptops)
- [3. Core Algorithm Engine: LightGBM + XGBoost + BO-TPE](#3-core-algorithm-engine-lightgbm--xgboost--bo-tpe)
- [4. Python Engineering Capabilities](#4-python-engineering-capabilities)
- [5. Module Technical Details](#5-module-technical-details)
  - [5.1 M0 Database](#51-m0-database)
  - [5.2 M1 Rolling Split Engine](#52-m1-rolling-split-engine)
  - [5.3 M2 Dual-Engine Training & Ensemble](#53-m2-dual-engine-training--ensemble)
  - [5.4 M3 TET Risk Control](#54-m3-tet-risk-control)
  - [5.5 M4 Reporting & Attribution](#55-m4-reporting--attribution)
  - [5.6 M5 Bayesian Hyperparameter Optimization](#56-m5-bayesian-hyperparameter-optimization)
- [6. Entry-Point Quick Reference](#6-entry-point-quick-reference)
- [7. Directory Layout](#7-directory-layout)
- [8. Quick Start](#8-quick-start)
- [9. Performance Baseline](#9-performance-baseline)
- [10. Security Notes](#10-security-notes)
- [11. License](#11-license)
- [12. Acknowledgments](#12-acknowledgments)

---

## 1. System Overview

This system is a closed-loop pipeline for the A-share market implementing **six-layer modular rolling-window stock selection + ensemble learning + Bayesian hyperparameter optimization**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Tushare Pro API                                                         │
│        │                                                                 │
│        ▼                                                                 │
│  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  ┌─────┐│
│  │   M0   │─▶│   M1   │─▶│ M2 / GPU │─▶│   M3   │─▶│   M4   │─▶│ M5  ││
│  │Database│  │ Split  │  │Train/Ens.│  │  Risk  │  │Report  │  │Optim││
│  └────────┘  └────────┘  └──────────┘  └────────┘  └────────┘  └─────┘│
│  Parquet     Rolling Win   LGBM+XGB      TET State   28+ Metrics Optuna │
│  Storage    +Label Gen.   Soft Voting    Machine     HTML Report BO-TPE │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

| Module | Path | Core Responsibility |
|--------|------|---------------------|
| **M0** Data | `m0_database/` | Tushare data fetching, stock-pool filtering, factor calculation, industry/market-cap neutralization (Rank-Z + dual OLS), Parquet storage |
| **M1** Rolling Split | `m1_engine/` | Time-series rolling train/val/pred split, cross-sectional label generation, generator-mode memory saving |
| **M2** Model Training | `m2_engine/`, `m2_engine_gpu/` | 7-step feature engineering, LightGBM/XGBoost training, soft-voting ensemble, Top10 portfolio, SHAP attribution |
| **M3** Risk Control | `m3_engine/` | TET risk control module (Trend-Score / Emotion-Index / Anchored-Trend / Timing state machine) |
| **M4** Reporting | `m4_report/` | 28+ performance metrics, 9 threshold checks, Brinson/Five-Factor/Barra attribution, iOS 26-style HTML report |
| **M5** Optimizer | `m5_optimizer/` | Optuna TPE two-phase Bayesian optimization (30-D params × 37-D objectives), Gradio Web UI, project management |

---

## 2. Running Smoothly on Low-Memory Laptops

> **Core philosophy**: Don't solve problems by throwing hardware at them — use engineering optimization to run full 180-window backtests on a 16 GB laptop.

### 2.1 Memory Optimization Panorama

```
┌─────────────────────────────────────────────────────────────────┐
│                  Seven-Layer Memory Defense System               │
├─────────┬───────────────────────────────────────────────────────┤
│ Layer 1 │ Data Type Compression                                  │
│         │ float64→float32 (save 50%)  string→category (40-60%)  │
├─────────┼───────────────────────────────────────────────────────┤
│ Layer 2 │ Lazy Loading                                           │
│         │ Generator yield per window  No pre-expansion          │
├─────────┼───────────────────────────────────────────────────────┤
│ Layer 3 │ Incremental Append                                       │
│         │ Per-file read→instant optimize→append                 │
│         │ Avoids pd.concat memory spike                          │
├─────────┼───────────────────────────────────────────────────────┤
│ Layer 4 │ Explicit Release                                       │
│         │ del+gc.collect() breaks reference cycles              │
│         │ Forced reclaim after every window                      │
├─────────┼───────────────────────────────────────────────────────┤
│ Layer 5 │ OS-Level Return                                        │
│         │ SetProcessWorkingSetSize + msvcrt._heapmin()          │
│         │ Actively returns memory to Windows after each Trial   │
├─────────┼───────────────────────────────────────────────────────┤
│ Layer 6 │ OOM Sentinel                                           │
│         │ Checks available memory every 10 windows              │
│         │ <0.8GB → proactive abort to prevent Windows kill      │
│         │ RSS growth 60% + 2GB+ + available <1.5GB = leak       │
├─────────┼───────────────────────────────────────────────────────┤
│ Layer 7 │ Serial Strategy                                        │
│         │ Windows serial (not parallel)  Models serial           │
│         │ Sacrifices speed for stability                         │
└─────────┴───────────────────────────────────────────────────────┘
```

### 2.2 Key Optimization Techniques

#### (1) Ultra-Lightweight Data Loader (`lightweight_loader.py`)

```python
# Traditional: pd.concat all files at once → memory peak doubles
# factor_df = pd.concat([pd.read_parquet(f) for f in files])

# This system: per-file read → instant type optimization → incremental append
for file_path in files:
    df = pd.read_parquet(file_path)
    # Instant compression: float64→float32, int64→int32, drop redundant cols
    df = _optimize_dataframe(df)
    factor_df = pd.concat([factor_df, df], ignore_index=True)
    # gc.collect() every 20 files to release intermediates
```

#### (2) Windows Memory Forced Return (`win_memory.py`)

```python
def release_memory_to_os():
    gc.collect(2)                              # Full-gen GC (incl. old gen)
    kernel32.SetProcessWorkingSetSize(          # Force OS to reclaim unused pages
        kernel32.GetCurrentProcess(), -1, -1)
    msvcrt._heapmin()                          # Force CRT heap compaction
    # Effect: combined with gc.collect(2), typically drops 200-500MB RSS
```

#### (3) Adaptive Concurrency Configuration (`concurrency_config.py`)

```python
# Low-memory high-CPU mode — tailored for 16GB laptops
GLOBAL_N_JOBS_OUTER   = 1    # Windows serial (not parallel)
GLOBAL_NTHREAD_INNER  = 4    # Single model 4 threads
M5_NTHREAD_PER_MODEL  = 4    # M5 per model 4 threads
MEMORY_LIMIT_GB       = 6.0  # Memory red line 6GB (reserve for system)
# Theoretical total threads: 1×2×4 = 8 / logical cores
```

#### (4) Training Memory Defense Chain (`objective.py`)

```python
# Three-level memory guards
_OOM_RSS_LIMIT_GB    = 12.0   # Process RSS > 12GB → force GC + return
_OOM_AVAIL_LIMIT_GB  = 0.5    # System available < 0.5GB → skip Trial
_OOM_VRAM_LIMIT_MB   = 7000   # GPU VRAM > 7GB → warning

# Trial timeout guard: 30-minute hard limit prevents single Trial hang
_TRIAL_TIMEOUT_SEC = 1800
```

#### (5) FeatureStore Cache Reuse

```python
# M5 multi-Trial scenario: reuse _fit_state when feature_params match
# transform() skips corr/IC filtering, reuses fit-time column selection
# Effect: saves 50% feature engineering time
```

### 2.3 Measured Memory Data

| Scenario | Memory Peak | Notes |
|----------|-------------|-------|
| M2 CPU full 180 windows | 10-12 GB | Serial windows + instant release |
| M5 single Trial (60 windows) | 8-10 GB | fast_mode + cache |
| M5 continuous 50 Trials | Stable 10-12 GB | Memory returned after each Trial |
| GPU mode VRAM | ~3.5 GB | GTX 1650 4GB, no OOM |

---

## 3. Core Algorithm Engine: LightGBM + XGBoost + BO-TPE

### 3.1 Dual-Model Ensemble Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                 7-Step Feature Engineering (FeatureStore)            │
│                                                                     │
│  Candidate Factors → Short-term Noise Filter → Low Coverage Filter  │
│  → Zero Variance Filter → Cross-sectional Z-score                  │
│  → High Correlation Dedup → IC Screening                           │
│  (Pure numpy vectorized, no pandas groupby overhead)               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐     ┌──────────────────────┐             │
│  │   LightGBM Ranker    │     │   XGBoost Ranker     │             │
│  │                      │     │                      │             │
│  │  objective: MAE      │     │  objective: MAE      │             │
│  │  device: OpenCL/CPU  │     │  device: CUDA/CPU    │             │
│  │  Adaptive leaves     │     │  Adaptive depth      │             │
│  │  LR decay callback   │     │  LR scheduler        │             │
│  │  early_stopping=30   │     │  early_stopping=30   │             │
│  └──────────┬───────────┘     └──────────┬───────────┘             │
│             │                             │                         │
│             └──────────┬──────────────────┘                         │
│                        ▼                                            │
│              Soft Voting Ensemble                                    │
│         score = w_lgbm × pred_lgbm + w_xgb × pred_xgb              │
│              (w_lgbm ∈ [0.3, 0.7], optimized by M5)                │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Confidence Check: ic_gap = train_IC - val_IC                       │
│  ic_gap > 0.15 → flagged LOW (no position impact, report only)     │
│                                                                     │
│  Top20 → High(1-5)×13% + Low(6-10)×7% = 100% fully invested      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 LightGBM Technical Details

| Dimension | Implementation |
|-----------|----------------|
| **Objective** | `regression_l1` (MAE) — robust to outliers, stronger rank correlation |
| **GPU Acceleration** | `device_type="gpu"` (OpenCL), GTX 1650 measured ~1.5GB VRAM |
| **Adaptive Leaves** | `num_leaves = min(2^max_depth, train_size/200, 255)` |
| **LR Decay** | Callback `_make_lr_decay_callback`: × decay_factor every decay_every rounds |
| **Depth Mode** | `fixed` / `adaptive` (auto-adjusts based on training set size) |
| **Early-Stop Rollback** | `predict(num_iteration=best_iteration)` |
| **CPU Threads** | 2 threads (optimal for 8-core laptop) |
| **Defense** | `dict()` shallow copy params, prevents in-place modification; explicit `del Dataset; gc.collect()` breaks reference cycles |

**Why MAE instead of lambdarank?**
- MAE is more robust to outliers, and rank correlation is stronger
- Avoids lambdarank's native memory crash in multi-thread / multi-Trial iteration

### 3.3 XGBoost Technical Details

| Dimension | Implementation |
|-----------|----------------|
| **Objective** | `reg:absoluteerror` (MAE) — aligned with LGBM |
| **GPU Acceleration** | `device="cuda"` (XGBoost 2.0+), `tree_method="hist"` |
| **Custom Eval** | `eval_ic` — directly optimizes Spearman IC |
| **LR Schedule** | `_make_xgb_lr_schedule`: supports fixed / decay modes |
| **Early-Stop Rollback** | `predict(iteration_range=(0, best_iteration))` |
| **CPU Threads** | 3 threads (optimal for 8-core laptop) |
| **Gamma** | Min loss split gain (0.001-0.2), prevents overfitting |
| **Defense** | Same as LGBM: `dict()` shallow copy + explicit `del DMatrix; gc.collect()` |

### 3.4 Ensemble Strategy

```python
# Soft Voting — weighted average
score = lgbm_weight × lgbm_pred + (1 - lgbm_weight) × xgb_pred

# Confidence flag
ic_gap = train_IC - val_IC
is_penalized = (ic_gap > 0.15)  # Flag only, does not affect position

# Prediction confidence coefficient of variation
score_cv = std([lgbm_pred, xgb_pred]) / (|score| + 1e-6)
```

**GPU Serial / CPU Serial**: Due to GTX 1650 4GB limit, LGBM and XGB must train serially (not in parallel). CPU mode also serial (avoids native memory conflicts).

### 3.5 Optuna BO-TPE Bayesian Optimization Engine

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Optuna TPE (Tree-structured Parzen Estimator)     │
│                                                                     │
│  ┌─────────────────────┐         ┌─────────────────────┐           │
│  │  Phase1: Global      │         │  Phase2: Local       │           │
│  │  Exploration         │         │  Fine-tuning         │           │
│  │                     │         │                     │           │
│  │  TPESampler         │         │  Start from P1 best  │           │
│  │  n_startup=15       │────▶────│  Tightened range     │           │
│  │  n_ei_candidates=24 │         │  Trial reuse (P1→P2) │           │
│  │  Constraint func    │         │  Independent SQLite  │           │
│  │  50-200 Trials      │         │  30-100 Trials       │           │
│  └─────────────────────┘         └─────────────────────┘           │
│                                                                     │
│  30-D Search Space:                                                 │
│  ├── LGBM 12-D (lr/n_est/depth/colsample/regα/regλ/...)           │
│  ├── XGB  11-D (lr/n_est/depth/colsample/regα/regλ/gamma/...)     │
│  ├── Ensemble 1-D (lgbm_weight)                                    │
│  ├── Feature 5-D (min_valid_rate/max_corr/min_ic/keep/drop)        │
│  └── Window 1-D (train_months)                                     │
│                                                                     │
│  37-D Objective Space:                                              │
│  ├── M2 Validation 17 items (IC/ICIR/IR/Sortino/Jensenα/...)      │
│  ├── M2 Stress Test 3 items (2008/2015/2022, optional)             │
│  └── M4 Backtest 17 items (CAGR/WinRate/VaR/CVaR/Omega/Burke/...) │
│                                                                     │
│  Score Normalization (NORM_CONFIG):                                 │
│  ├── linear  → Rate metrics, clip to [-0.5, 0.5]                   │
│  ├── tanh    → IR/Sortino-like, compress to (-1, 1)                │
│  └── signed_log → Long-tail, sign(x)×log(1+|x|)                    │
│                                                                     │
│  Safety Constraints:                                                │
│  ├── lr × n_estimators ≤ 15.0 (prevent overfit + timeout)          │
│  ├── n_est ≤ 500, depth ≤ 8, lr ≥ 0.005 (hard caps)               │
│  ├── Trial timeout 30-minute hard limit                             │
│  └── IC_GAP penalty × 1.5 (overfit penalty amplified)              │
└─────────────────────────────────────────────────────────────────────┘
```

**TPE Core Principle**:

TPE uses two kernel density estimators to model the probability distributions of "good parameters" and "bad parameters" respectively, then selects the point that minimizes `l(x)/g(x)` as the next sample. Compared to grid search and random search, TPE explores high-dimensional hyperparameter spaces more efficiently.

```
EI(x) = ∫ max(f* - f(x), 0) p(f(x)|x) df
```

Where `f*` is the current best value, and `p(f(x)|x)` is given by TPE's two KDE models.

---

## 4. Python Engineering Capabilities

### 4.1 Python Version & Features

| Feature | Usage |
|---------|-------|
| **Python 3.14** | Used project-wide, PEP 745 free-threading optionally enabled |
| **match/case** | Extensively used in `search_space.py` / `objective.py` for structured pattern matching |
| **Immutable type optimization** | `frozenset` / `MappingProxyType` replace mutable set/dict, avoiding GIL refcount overhead |
| **dataclasses** | M3 TET engine config class `M3Config` uses `@dataclass(slots=True)` |
| **Type annotations** | Project-wide use of `typing` module (`Dict`, `List`, `Optional`, `Tuple`, `Any`) |
| **Generators** | M1 `RollingSplitter.split()` yields windows one at a time, lazy evaluation |
| **`__slots__`** | `EnsemblePredictor` uses `__slots__` to reduce instance memory overhead |

### 4.2 Core Technology Stack

```
Data & Scientific:   numpy ≥ 1.26  |  pandas ≥ 2.0  |  scipy ≥ 1.10  |  pyarrow ≥ 14.0
ML Models:           lightgbm ≥ 4.1  |  xgboost ≥ 2.0  |  shap ≥ 0.42
Bayesian Optim:      optuna ≥ 3.4  |  joblib ≥ 1.3
Data Sources:        tushare ≥ 1.4  |  akshare ≥ 1.12
Web UI:              gradio ≥ 4.0, < 5.0
Monitoring:          psutil ≥ 5.9
Performance:         bottleneck (nanmean/nanstd etc., optional)
```

### 4.3 Vectorized Programming Practices

```python
# 1. Pure numpy cross-sectional Z-score — no pandas groupby overhead
def _zscore_arr(arr, dates):
    """Vectorized cross-sectional Z-score: 1 argsort + 1 unique + group broadcast"""
    order = np.argsort(dates, kind='stable')
    sorted_dates = dates[order]
    _, idx_start = np.unique(sorted_dates, return_index=True)
    idx_end = np.append(idx_start[1:], len(sorted_dates))
    result = np.empty_like(arr)
    for s, e in zip(idx_start, idx_end):
        grp = arr[order[s:e]]
        mu, sigma = grp.mean(), grp.std()
        result[order[s:e]] = (grp - mu) / (sigma + 1e-8)
    return result

# 2. Vectorized Pearson correlation matrix — 1D broadcast replaces 2D
def _corrcoef_f32(X):
    X_c = X - X.mean(axis=0)
    cov = X_c.T @ X_c
    diag_sqrt = np.sqrt(np.diag(cov) + 1e-12)
    # 1D broadcast is 2-3x faster than diag[:,None]*diag[None,:]
    return cov / (diag_sqrt[:, None] * diag_sqrt[None, :] + 1e-12)

# 3. Vectorized IC computation — argsort+unique+slice replaces per-date boolean mask
def _per_group_rank_corr(dates, pred, label):
    order = np.argsort(dates, kind='stable')
    sorted_dates = dates[order]
    _, idx_start = np.unique(sorted_dates, return_index=True)
    idx_end = np.append(idx_start[1:], len(dates))
    return [_fast_rank_corr(pred[order[s:e]], label[order[s:e]])
            for s, e in zip(idx_start, idx_end)]
    # Performance: ~12ms/call → ~1ms/call (saves ~2s for 180-window full run)

# 4. sliding_window_view replaces Python loops
from numpy.lib.stride_tricks import sliding_window_view
win_ex = sliding_window_view(excess, 6)   # (n-5, 6) matrix
win_ex_mean = win_ex.mean(axis=1)          # Fully vectorized
```

### 4.4 Memory Management Engineering

```python
# 1. Data type compression
df[float64_cols] = df[float64_cols].astype(np.float32)  # Save 50%
df["stock_code"] = df["stock_code"].astype("category")   # Save 40-60%

# 2. Shallow copy replaces deep copy
pred_out = pred_df.copy(deep=False)  # Save ~6MB/window deep copy

# 3. Explicit cycle breaking
del X_train, y_train, lgb_train, lgb_val
gc.collect()  # Break Booster→callback→Dataset reference cycle

# 4. Thread-safe cache
_FEATURE_CACHE_LOCK = threading.Lock()
with _FEATURE_CACHE_LOCK:
    _FEATURE_CACHE[cache_key] = (train_p, val_p, pred_p, feature_cols)

# 5. Bottleneck acceleration (optional)
try:
    import bottleneck as bn
    _nanmean = bn.nanmean   # 3-5x faster than np.nanmean
    _nanstd  = bn.nanstd
except ImportError:
    _nanmean = np.nanmean
    _nanstd  = np.nanstd
```

### 4.5 Defensive Programming

```python
# 1. Parameter shallow copy — prevents in-place modification of caller's dict
def __init__(self, params=None):
    self.params = dict(params) if params else {}  # Shallow copy

# 2. Three-level OOM defense
if avail_gb < 0.5: return -999.0   # Skip Trial
if rss_gb > 12.0: release_memory() # Force return
if vram_mb > 7000: gc.collect()    # VRAM warning

# 3. Trial timeout guard
future = executor.submit(run_m2, **kwargs)
portfolios, stats = future.result(timeout=1800)  # 30-minute hard limit

# 4. Multi-level return backfill fallback
if pred_date in date_return_map:
    portfolio["Target_Return_1M"] = portfolio["stock_code"].map(...)
else:
    # Fallback 1: Use pred_df built-in returns
    # Fallback 2: Secondary stock-level fill

# 5. Memory leak detection (Scheme D)
# Three conditions must all be met: RSS growth >60% + absolute >2GB + available <1.5GB
check_rss_leak(trial.number, stop_now_event)
```

---

## 5. Module Technical Details

### 5.1 M0 Database

**Directory**: `m0_database/`
**Entry**: `python run_m0_full.py`

#### Core Components

| File | Responsibility |
|------|----------------|
| `pipeline.py` | Top-level pipeline, supports full/incremental/checkpoint resume |
| `data_fetcher.py` | Tushare data fetching (daily/financial/index), fallback to akshare |
| `stock_filter.py` | Stock pool filtering (ST/delisted/newly-listed) |
| `factor_calculator.py` | Factor calculation (technical/fundamental/Barra risk factors) |
| `neutralization.py` | Neutralization (Rank-Z + dual OLS industry/market-cap) |
| `format_validator.py` | Data format validation |
| `regenerator.py` | Incremental regeneration (recalculate specified months) |
| `_preflight_check.py` | Pre-launch environment check (Tushare token/disk space) |

#### 4 Neutralization Schemes

| Scheme | Description |
|--------|-------------|
| `scheme_a` | Raw factors (no neutralization) |
| `scheme_b` | Industry neutralization |
| `scheme_c` | Market-cap neutralization |
| `scheme_d` | Industry + market-cap dual neutralization (default) |

---

### 5.2 M1 Rolling Split Engine

**Directory**: `m1_engine/`
**Entry**: `python m1_engine/run_m1.py`

#### Core Components

| File | Responsibility |
|------|----------------|
| `data_loader.py` | Parallel read M0 Parquet pool, convert to float32/category, save memory |
| `label_maker.py` | Cross-sectional `label_rank ∈ [0,1]` (`pct=True` rank quantile) |
| `rolling_splitter.py` | Time-series rolling splitter (Python generator yield) |
| `run_m1.py` | Top-level entry |

#### Rolling Window Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `train_months` | **36** | Training window months |
| `valid_months` | **12** | Validation window months |
| `test_months` | **1** | Prediction window months |
| `step_months` | **1** | Slide forward months |

```
Window i:
  train: m_i ~ m_{i+35}     (36 months)
  val:   m_{i+36} ~ m_{i+47} (12 months)
  pred:  m_{i+48}            (1 month)
```

#### Key Design

- **Generator mode**: `yield` returns windows one at a time, no pre-generation into memory
- **Strict no data leakage**: train < val < pred time strictly monotonic
- **Parallel I/O**: `ThreadPoolExecutor` async Parquet reading

---

### 5.3 M2 Dual-Engine Training & Ensemble

**Directory**: `m2_engine/` (CPU), `m2_engine_gpu/` (GPU accelerated)
**Entry**: `python m2_engine/run_m2.py`

#### Pipeline

```
load window → FeatureStore.fit_transform → EnsemblePredictor.fit_predict
  → PortfolioBuilder.build → backfill returns → compute_val_metrics
```

#### Core Modules

| File | Responsibility | Latest Features |
|------|----------------|-----------------|
| `feature_store.py` | 7-step feature engineering | v3.8: Pure numpy vectorized, `transform()` reuses `_fit_state` |
| `lgbm_model.py` | LightGBM Ranker | Adaptive leaves + LR decay + OpenCL GPU |
| `xgb_model.py` | XGBoost Ranker | LR schedule + CUDA GPU + gamma regularization |
| `ensemble.py` | Soft voting + IC computation | Vectorized `_per_group_rank_corr` + `sliding_window_view` |
| `portfolio_builder.py` | Top10 holdings + turnover cost | SHAP attribution (Top3 factors written to DataFrame) |
| `gpu_detector.py` | GPU adaptive (singleton) | Strategy A/B/C/D/E runtime switching |
| `lightweight_loader.py` | Ultra-lightweight loader | Per-file incremental loading, prevents OOM |
| `smart_preprocessor.py` | Efficient preprocessor | Chunked reading + instant type optimization |
| `run_m2.py` | Top-level entry | v5.5: Multi-level return backfill + OOM sentinel |

#### Ensemble Predictor (`ensemble.py`)

- **Vectorized IC computation**: `_per_group_rank_corr` — 1 argsort + 1 unique + per-group slice, 12x faster than per-date boolean mask
- **`compute_val_portfolio_metrics`**: Simulates portfolio with 6-month rolling computation of 19 performance metrics (including P1 adaptive type 2.0 additions: `val_ic_stability` / `val_ir_stability` / `turnover_penalty`)
- **Capture ratios**: `_compute_capture_ratios` — upside/downside/combined capture
- **Jensen's Alpha**: `_compute_jensen_appraisal` — OLS regression + annualized α + Appraisal Ratio

#### Portfolio Construction (`portfolio_builder.py`)

| Tier | Rank | Weight |
|------|------|--------|
| High | 1-5 | 13% × 5 = 65% |
| Low | 6-10 | 7% × 5 = 35% |
| Reserve | 11-20 | 0% (backup) |

- **Always fully invested**: 100% allocation, no cash signal
- **Turnover cost**: Stamp tax 0.1% (sell only) + Commission 0.03% (both sides) + Slippage 0.1% (both sides)

---

### 5.4 M3 TET Risk Control

**Directory**: `m3_engine/`
**Entry**: `python m3_engine/run_m3.py`

#### 4 Core Indicators

| Indicator | Full Name | Description |
|-----------|-----------|-------------|
| **TS** | Trend-Score | 4 major trend factor tiers, layered voting |
| **EI** | Emotion-Index | 12 oscillators normalized by direction |
| **ATS** | Anchored-Trend-Score | Schmitt Trigger axis-crossing anchoring |
| **Timing** | ATS - EI | < sell_threshold triggers SELL_TET |

#### State Machine Rules

- **Rule A**: Newly admitted stock ATS = current period TS
- **Rule B**: Immediately pop state after SELL_TET
- **Rule C**: Cash pool does not carry over

#### Tech Stack

- Uses `polars` for high-performance data processing
- `@dataclass(slots=True)` for config classes
- `tqdm` progress bar

---

### 5.5 M4 Reporting & Attribution

**Directory**: `m4_report/`
**Entry**: `python m4_report/report_generator.py`

#### Performance Metrics (28+ items)

| Category | Metrics |
|----------|---------|
| Returns | CAGR, CAGR_benchmark, annual_excess, net_cagr_after_cost |
| Risk | max_drawdown, volatility, downside_volatility, upside_volatility, VaR(95%), CVaR(95%) |
| Ratios | sharpe, sortino, calmar, ir, sterling, burke, martin, omega, tail |
| Capture | up_capture, down_capture, capture_ratio |
| Distribution | skewness, kurtosis, pain_index, ulcer_index |
| Win Rate | monthly_win_rate, rolling6m_win_rate |
| α/β | jensen_alpha (annualized), appraisal_ratio |
| Cost | avg_monthly_turnover_cost, avg_annual_turnover_cost |

#### 9 Threshold Checks

| Check | Threshold |
|-------|-----------|
| IR | ≥ 0.50 |
| Calmar | ≥ 1.00 |
| MaxDD | ≥ -35% |
| Sortino | ≥ 1.20 |
| annual_excess | ≥ 5% |
| rolling6m_win_rate | ≥ 60% |
| capture_ratio | ≥ 1.20 |
| pain_index | ≤ 0.10 |
| omega_ratio | ≥ 1.20 |

#### Three Attribution Methods (`attribution.py`)

| Attribution | Method |
|-------------|--------|
| **Brinson** | Market return / industry allocation / stock selection decomposition |
| **Five-Factor** | Fama-French 5 + Carhart Momentum (MKT/SMB/HML/RMW/CMA/MOM) |
| **Barra** | 10 barra_ factor exposures + factor return proxies |

#### HTML Report Features

- iOS 26 style design
- Chart.js embedded (fully offline) + server-side SVG fallback
- Equity curve + time range slider + mouse hover tooltips
- Brinson / Five-Factor / Barra attribution charts
- SHAP factor attribution (Top10 holdings × Top5 factors)
- Monthly holdings interactive page (select year-month to view 10 stocks + tiers + individual returns)

---

### 5.6 M5 Bayesian Hyperparameter Optimization

**Directory**: `m5_optimizer/`
**Entry**: Double-click `m5_optimizer/启动M5优化器.bat` or `python m5_optimizer/app.py`
**Web UI**: http://127.0.0.1:7860

#### 30-D Search Space

| Group | Count | Key Parameters |
|-------|-------|----------------|
| **LGBM** | 12 | learning_rate (log), n_estimators, max_depth, colsample_bytree, reg_alpha/lambda, min_split_gain, lr_mode, decay_every/factor, depth_mode, early_stopping |
| **XGB** | 11 | learning_rate (log), n_estimators, max_depth, colsample_bytree, reg_alpha/lambda, gamma, lr_mode, decay_every/factor, early_stopping |
| **Ensemble** | 1 | lgbm_weight (0.3-0.7) |
| **Feature** | 5 | min_valid_rate, max_corr, min_ic_abs (log), min_keep_factors, drop_short_term_noise |
| **Window** | 1 | train_months (52-60) |

#### 37-D Objective Space

- **M2 Validation (17 items)**: val_ic, val_icir, val_rolling6m_ir, val_rolling6m_sortino, val_rolling6m_return, ic_gap_penalty, penalized_rate, val_global_ir, val_annual_return, pct_positive_excess, ir_worst_quartile, val_rolling6m_excess, val_rolling6m_excess_ann, val_jensen_alpha, val_appraisal_ratio, val_beta, val_ic_stability
- **M2 Stress Test (3 items, disabled)**: 2008/2015/2022
- **M4 Backtest (17 items)**: cagr, monthly_win_rate, downside_volatility, upside_volatility, volatility_ratio, var_95, cvar_95, skewness, kurtosis, pain_index, omega_ratio, burke_ratio, martin_ratio, tail_ratio, up_capture_ratio (enabled), down_capture_ratio (disabled), capture_ratio (disabled)

#### Two-Phase Optimization Strategy

| Phase | Strategy | Trials | Description |
|-------|----------|--------|-------------|
| **Phase1** | Global exploration | 50-200 | TPESampler + constraint function + checkpoint resume + warm-start prior |
| **Phase2** | Local fine-tuning | 30-100 | Start from P1 best + tightened range + Trial reuse + independent SQLite |

#### Gradio Web UI (5 Tabs)

| Tab | Function |
|-----|----------|
| **Tab1** Training Control | Select project → start P1/P2 → real-time progress bar → immediate stop / graceful stop |
| **Tab2** Result Analysis & Retro | P1 Trial statistics, dependent-variable slider filter, retro-analyze P2 search range |
| **Tab3** Top-5 Ranking | P1 + P2 joint ranking of top 5 Trial details |
| **Tab4** Parameter Heatmap | Factor-IC relationship, parameter-score sensitivity chart |
| **Tab5** Project Management | Create/clone/delete optimization projects, view history |

#### Utility Tools (`utils/`)

| File | Responsibility |
|------|----------------|
| `logger.py` | Unified logging facade |
| `rolling_logger.py` | Dual-channel rolling log (critical event sync flush + normal event async queue) |
| `memory_monitor.py` | Memory monitoring (psutil) |
| `win_memory.py` | Windows memory forced return (SetProcessWorkingSetSize + msvcrt._heapmin) |
| `trial_callback.py` | Optuna callback factory (dual stop mechanism) |
| `restart_check.py` | RSS growth rate detector (memory leak detection) |
| `retroactive_normalize.py` | Post-hoc normalization (P2 range retro-analysis) |

---

## 6. Entry-Point Quick Reference

| Program | Command | Function |
|---------|---------|----------|
| `run_m0_full.py` | `python run_m0_full.py` | M0 full data fetch → 4 neutralization schemes Parquet storage (checkpoint resume) |
| `run_m1.py` | `python m1_engine\run_m1.py` | M1 rolling split → window Parquet storage |
| `run_m2.py` | `python m2_engine\run_m2.py` | M2 dual-engine training → all_portfolios.parquet |
| `run_m2_m4.py` | `python run_m2_m4.py` | One-click M2 + M4 (CPU mode) |
| `run_one_full.py` | `python run_one_full.py baseline 60` | Single-strategy full benchmark |
| `run_m3.py` | `python m3_engine\run_m3.py` | M3 TET risk control |
| `report_generator.py` | `python m4_report\report_generator.py` | M4 HTML report generation |
| `启动M5优化器.bat` | Double-click | Launch Gradio Web UI (http://127.0.0.1:7860) |
| `app.py` | `python m5_optimizer\app.py` | Same as above (CLI equivalent) |

---

## 7. Directory Layout

```
10q-202604gpu/
├── config/                    # Configuration
│   ├── config.yaml            # Main config (neutralization, rolling, cost, 9 thresholds)
│   └── concurrency_config.py  # Adaptive threads/concurrency (low-memory high-CPU mode)
│
├── m0_database/               # M0 Data Module
│   ├── pipeline.py            # Top-level pipeline (full/incremental/checkpoint resume)
│   ├── data_fetcher.py        # Tushare data fetching
│   ├── stock_filter.py        # Stock pool filtering (ST/delisted/newly-listed)
│   ├── factor_calculator.py   # Factor calculation (technical/fundamental/Barra)
│   ├── neutralization.py      # Neutralization (Rank-Z + dual OLS)
│   ├── format_validator.py    # Data format validation
│   ├── regenerator.py         # Incremental regeneration
│   └── _preflight_check.py    # Pre-launch environment check
│
├── m1_engine/                 # M1 Rolling Split
│   ├── data_loader.py         # Parallel Parquet loading (float32/category optimization)
│   ├── label_maker.py         # Cross-sectional label_rank generation
│   ├── rolling_splitter.py    # Time-series rolling splitter (generator yield)
│   └── run_m1.py              # Top-level entry
│
├── m2_engine/                 # M2 CPU Training
│   ├── ensemble.py            # Soft voting + vectorized IC + 19 validation-set metrics
│   ├── feature_store.py       # 7-step feature engineering (pure numpy vectorized)
│   ├── lgbm_model.py          # LightGBM Ranker (CPU/OpenCL)
│   ├── xgb_model.py           # XGBoost Ranker (CPU/CUDA)
│   ├── portfolio_builder.py   # Top10 holdings + turnover cost + SHAP attribution
│   ├── preprocessor.py        # Data preloading
│   ├── smart_preprocessor.py  # Efficient preprocessor (chunked + instant optimization)
│   ├── lightweight_loader.py  # Ultra-lightweight loader (incremental append, OOM-proof)
│   ├── gpu_detector.py        # GPU/CPU adaptive (singleton, strategy A-E)
│   └── run_m2.py              # Top-level entry (v5.5: OOM sentinel + return Fallback)
│
├── m2_engine_gpu/             # M2 GPU Accelerated Training (parallel to m2_engine)
│   ├── ensemble.py            # GPU strategies A/B/C/D/E
│   ├── feature_store.py
│   ├── lgbm_model.py          # GPU LightGBM (OpenCL)
│   ├── xgb_model.py           # GPU XGBoost (CUDA)
│   ├── portfolio_builder.py
│   ├── gpu_detector.py
│   └── run_m2.py
│
├── m3_engine/                 # M3 TET Risk Control
│   ├── tet_engine.py          # TET core (TS/EI/ATS/Timing + state machine)
│   ├── run_m3.py              # Top-level entry
│   └── __main__.py
│
├── m4_report/                 # M4 Backtest Report
│   ├── metrics.py             # 28+ performance metrics + 9 thresholds
│   ├── attribution.py         # Three attribution methods (Brinson/Five-Factor/Barra)
│   └── report_generator.py    # iOS 26-style HTML report (Chart.js embedded)
│
├── m5_optimizer/              # M5 Bayesian Optimizer
│   ├── app.py                 # Gradio Web UI (5 Tabs)
│   ├── search_space.py        # 30-D params + 37-D objectives + NORM_CONFIG
│   ├── objective.py           # Optuna objective function (three-level OOM defense)
│   ├── phase1_global.py       # Phase1 global exploration (TPE + constraints)
│   ├── phase2_local.py        # Phase2 local fine-tuning (tightened range)
│   ├── config_manager.py      # M5 config read/write
│   ├── project_manager.py     # Project management
│   ├── result_analyzer.py     # P1 result analysis
│   ├── range_analyzer.py      # P2 range retro-analysis
│   ├── utils/                 # Utility tools
│   │   ├── rolling_logger.py  # Dual-channel rolling log (crash diagnosis)
│   │   ├── win_memory.py      # Windows memory forced return
│   │   ├── restart_check.py   # RSS leak detection
│   │   ├── trial_callback.py  # Optuna callback factory
│   │   └── ...
│   └── 启动M5优化器.bat       # One-click launcher
│
├── tests/                     # Tests
│   ├── conftest.py
│   ├── test_tet_engine.py     # M3 TET engine tests
│   └── verify_v42_fixes.py    # v4.2 fix verification (9 automated tests)
│
├── scripts/                   # Helper scripts
├── run_*.py                   # Top-level entry points
├── requirements.txt
├── CHANGELOG.md
├── README.md
└── .gitignore
```

> ⚠️ **Not in version control**: `data/`, `logs/`, `output/`, all `*.pkl / *.parquet / *.db` caches.
> See [.gitignore](.gitignore) for details.

---

## 8. Quick Start

### 8.1 Runtime Environment Requirements

> **All code in this repository has been validated in a single environment: Windows 11 + Python 3.14**

| Item | Requirement | Notes |
|------|-------------|-------|
| **Operating System** | Windows 11 (22H2 / 23H2 / 24H2) | ⚠️ Currently validated **only** on Win11 |
| **Python** | **3.14.x** (3.14.0+ recommended) | ⚠️ **Not tested on 3.11/3.12/3.13** |
| **Architecture** | x86_64 / ARM64 | Both tested |
| **RAM** | **≥ 16 GB** | System deeply optimized for 16GB laptops |
| **GPU** (optional) | NVIDIA GTX 1650 or above | GPU only accelerates M2, CPU mode also works |
| **CUDA** (optional) | 12.x | For GPU mode |
| **CPU** | 8 cores optimal | LGBM=2 threads + XGB=3 threads is optimal combo |
| **Disk** | ≥ 10 GB free | Dataset + Parquet cache |
| **Tushare Credits** | ≥ 5000 | Required for full A-share daily/factor data |

### 8.2 Install Dependencies

```powershell
# 1. Verify Python version
python --version   # Must be 3.14.x

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set Tushare Token (never write into config.yaml)
$env:TUSHARE_TOKEN = "your_token_here"

# 5. Verify GPU (if available)
python -c "import xgboost; print('CUDA:', xgboost.get_build_info())"
```

### 8.3 Run the Pipeline

```powershell
# M0 fetch data + compute factors (first time takes hours, checkpoint resume)
python run_m0_full.py

# M1 rolling split
python m1_engine\run_m1.py

# M2 training + portfolio construction
python m2_engine\run_m2.py

# M3 TET risk control (optional)
python m3_engine\run_m3.py

# M4 report
python m4_report\report_generator.py

# M5 launch optimizer Web UI
.\m5_optimizer\启动M5优化器.bat
# Open http://127.0.0.1:7860 in browser
```

---

## 9. Performance Baseline

| Item | Value | Notes |
|------|-------|-------|
| Rolling training window | 36 months train / 12 months val / 1 month test | Configurable |
| Total windows | ~180 (2007-01 ~ 2025-12) | 228 - 49 + 1 |
| Portfolio capacity | Top10 always fully invested (65% + 35%) | No cash signal |
| CPU mode memory peak | **10-12 GB** | Runs smoothly on 16GB laptop |
| GPU mode VRAM peak | ~3.5 GB | GTX 1650 4GB, no OOM |
| 8-core CPU single-window latency | ~0.76s | benchmark_v42 measured |
| GPU speedup | 1.5-2.5× | Depends on data size |
| M5 single Trial (60 windows) | ~5-8 minutes | fast_mode |
| M5 memory stability | No growth over 50 continuous Trials | Memory returned after each Trial |

---

## 10. Security Notes

- **Never commit the Tushare token**! The repo is configured to read it from the `TUSHARE_TOKEN` environment variable
- Historical commits have been cleaned of plaintext tokens
- If the token is leaked, reset it immediately at https://tushare.pro
- **No `*.db` / `*.parquet` / `*.pkl` files are in git**; see [.gitignore](.gitignore)

---

## 11. License

This project is released under the **Apache License 2.0**.

```
Copyright 2026 zjwandcat

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### What You Can Do

| Action | Allowed |
|--------|---------|
| Personal learning, research, source-code reading | ✅ |
| Forking / modifying for your own projects | ✅ |
| Commercial use, deployment to production | ✅ |
| Redistribution (with copyright and license preserved) | ✅ |
| Patent grants | ✅ |
| Proprietary use | ✅ |
| Publishing modifications under a different license | ✅ (original copyright must be preserved) |

### What You Must Do

- **Preserve copyright notice**: keep `Copyright 2026 zjwandcat` in all copies / derivative works
- **Mark modifications**: clearly state "modified" if you modified source files
- **Include LICENSE copy**: a copy of this LICENSE must accompany redistribution
- **NOTICE file**: must preserve attribution statement in [NOTICE](NOTICE)
- **Patent grant termination**: if you file patent litigation against any Contributor, all patent grants from that Contributor automatically terminate

### Third-Party Dependencies

This project depends on multiple third-party open-source libraries (pandas, numpy, lightgbm, xgboost, optuna, gradio, tushare, etc.). The complete list and licenses are in [NOTICE](NOTICE). These dependencies retain their original licenses and are **not constrained by this project's Apache 2.0**.

### Risk Disclaimer

This project is intended for quantitative research and learning only. **It does not constitute any investment advice**. The author **accepts no responsibility** for any investment loss arising from the use of this project's code. See LICENSE sections 7 and 8 (no warranty / limitation of liability).

---

## 12. Acknowledgments

Thanks to [Tushare Pro](https://tushare.pro) for providing high-quality A-share data APIs, and to the LightGBM, XGBoost, Optuna, Gradio, Polars open-source communities.
