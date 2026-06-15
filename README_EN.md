# TTHH A-Share Quantitative Stock Selection System (m01245)

> Maintained by [zjwandcat](https://github.com/zjwandcat)
>
> An A-share quantitative stock-selection, backtesting, and hyper-parameter optimization platform built on Tushare data, a multi-module pipeline, and GPU-accelerated XGBoost/LightGBM.

🌐 **Language**: [🇬🇧 English (current)](README_EN.md) · [🇨🇳 中文](README.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078d4.svg)](https://www.microsoft.com/windows/windows-11)

**Copyright © 2026 zjwandcat. Licensed under the [Apache License, Version 2.0](LICENSE).**

---

## 📋 Runtime Environment Requirements

> **All code in this repository has been validated in a single environment: Windows 11 + Python 3.14.**

| Item | Requirement | Notes |
|------|-------------|-------|
| **Operating System** | Windows 11 (22H2 / 23H2 / 24H2 all work) | ⚠️ Currently validated **only** on Win11 |
| **Python** | **3.14.x** (3.14.0+ recommended) | ⚠️ Validated **only** on Python 3.14 — **not tested on 3.11/3.12/3.13** |
| **Architecture** | x86_64 / ARM64 | Both tested |
| **GPU** (optional) | NVIDIA GTX 1650 or above | GPU only accelerates the M2 module; CPU mode works as well |
| **CUDA** (optional) | 12.x | Required for GPU mode |
| **RAM** | ≥ 16 GB | 32 GB recommended for full backtest |
| **Disk** | ≥ 10 GB free | Dataset + Parquet cache |
| **Tushare Credits** | ≥ 5000 | Required to pull full A-share daily/factor data |

### Python 3.14 Verification

```powershell
PS E:\10q\10q-202604gpu> python --version
Python 3.14.0

PS E:\10q\10q-202604gpu> python -c "import sys; print(sys.platform, sys.version_info)"
win32 (3, 14, 0, 'final', 0)
```

> ⚠️ **Important**: This project uses Python 3.14 (PEP 745 free-threading mode can be optionally enabled). Several third-party packages (LightGBM 4.6+, XGBoost 3.0+, Optuna 4.x) are compatible with 3.14. Any compatibility issues when running on older Python versions are **not considered a project bug**.

### Quick Environment Setup

```powershell
# 1. Verify Python version
python --version   # Must be 3.14.x

# 2. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt   # see below

# 4. Set Tushare Token
$env:TUSHARE_TOKEN = "your_token_here"

# 5. Verify GPU (if available)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---



## 1. System Overview

This system is a closed-loop pipeline for the A-share market implementing **multi-module rolling-window stock selection + ensemble learning + Bayesian hyper-parameter optimization**. It is organized into **five layers: M0 → M1 → M2 → M4 → M5**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  Tushare Pro API                                                      │
│        │                                                              │
│        ▼                                                              │
│  ┌────────┐    ┌────────┐    ┌──────────┐    ┌────────┐    ┌────────┐│
│  │   M0   │───▶│   M1   │───▶│  M2 / GPU│───▶│   M4   │    │   M5   ││
│  │ Database│   │ Split  │    │Train/Ens.│    │Report  │    │Optimize││
│  └────────┘    └────────┘    └──────────┘    └────────┘    └────────┘│
│  Parquet      Rolling Win.    LGBM+XGB         28+ Metrics  Optuna   │
│  Storage     + Label Gen.     Top10 Holdings   HTML Report  TPE Bayes│
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

| Module | Path | Core Responsibility |
|--------|------|---------------------|
| **M0** Data | `m0_database/` | Tushare data fetching, stock-pool filtering, factor calculation, industry/market-cap neutralization, Parquet storage |
| **M1** Rolling Split | `m1_engine/` | Time-series rolling train/val/pred split, cross-sectional label generation |
| **M2** Model Training | `m2_engine/`, `m2_engine_gpu/` | Feature engineering, LGBM/XGBoost training, soft-voting ensemble, Top10 portfolio construction |
| **M4** Reporting | `m4_report/` | Monthly returns, turnover cost, 28+ performance metrics, HTML report |
| **M5** Optimizer | `m5_optimizer/` | Optuna TPE two-phase Bayesian optimization, Gradio Web UI, project management |

---

## 2. Module Technical Details

### 2.1 M1 Rolling-Split Engine

**Directory**: `m1_engine/`
**Entry point**: `python m1_engine/run_m1.py` or `run_m1(force_rebuild=True)`

#### 2.1.1 Design Goals

- Split the "full-month cross-section" data produced by M0 into independently trainable rolling windows in chronological order.
- **Strict no data leakage**: For every window, train < val < pred times are strictly monotonic.
- **Memory-friendly**: Uses Python generators (`yield`) to slice windows one at a time; never expands everything into memory at once.

#### 2.1.2 Core Components

| File | Responsibility |
|------|----------------|
| `data_loader.py` | Parallel-read M0's Parquet pool, convert to float32, convert to Categorical, save memory |
| `label_maker.py` | Within each `trade_date` cross-section, ascending-rank `Target_Return_1M` to produce `label_rank ∈ [0,1]` |
| `rolling_splitter.py` | Time-series rolling splitter, yields a group of `{train_df, val_df, pred_df, window_idx, pred_month}` |
| `run_m1.py` | Top-level entry point, wires DataLoader → LabelMaker → RollingSplitter → storage |

#### 2.1.3 Rolling-Window Parameters

Defined in `config/config.yaml → rolling`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `train_months` | **36** | Training window length (months) |
| `valid_months` | **12** | Validation window length (months) |
| `test_months` | **1** | Prediction window length (months) — generates next-month portfolio |
| `step_months` | **1** | Months to slide forward at each step |

**Window boundaries** (`rolling_splitter.py` follows this strictly):

```
all_months = [m_0, m_1, ..., m_{T-1}]
Window i (0-indexed):
  train: m_i, m_{i+1}, ..., m_{i+35}      (36 months)
  val:   m_{i+36}, ..., m_{i+47}           (12 months)
  pred:  m_{i+48}                          ( 1 month)
```

#### 2.1.4 Key Implementation Details

1. **Parallel Parquet read** (`DataLoader.load()`): `ThreadPoolExecutor` async read, I/O and CPU decoupled; logs progress every 20 files.
2. **Memory optimization**:
   - Numeric columns uniformly `astype(np.float32)` — saves 50% vs float64
   - Stock-code columns `astype("category")` — saves 40-60% string memory, accelerates `groupby`
   - Explicit `df.copy()` to defragment
3. **Cross-sectional label generation** (`LabelMaker.make_labels()`):
   ```python
   df["label_rank"] = (
       df.groupby("trade_date")["Target_Return_1M"]
         .transform(lambda x: x.rank(method="average",
                                     ascending=True,
                                     pct=True,
                                     na_option="keep"))
   )
   ```
   - `pct=True` directly yields the 0~1 quantile
   - `na_option="keep"` keeps NaN as NaN, avoiding pollution of training
4. **Rolling-split generator** (`RollingSplitter.split()`): uses Python generator (yield) to return windows one at a time — **does not pre-generate all windows into memory**. Internal `.copy()` per window prevents view-sharing from polluting source data on later modifications.
5. **Window output format**: each window is saved as `window_{idx:03d}_{pred_month}.parquet`, with a `_split ∈ {train, val, pred}` column marking its origin.
6. **Window statistics**: `summary.json` + `window_stats.csv` record total window count, start/end months, average row/column counts, etc.

#### 2.1.5 Output

```
output/m1_windows/
├── summary.json                  # overall statistics fingerprint
├── window_stats.csv              # per-window row/column counts
├── window_000_201001.parquet     # window 0, predicting 2010-01
├── window_001_201002.parquet
└── ...
```

#### 2.1.6 Usage Example

```python
from m1_engine.data_loader import DataLoader
from m1_engine.label_maker import LabelMaker
from m1_engine.rolling_splitter import RollingSplitter

loader = DataLoader(scheme="scheme_d")         # choose neutralization scheme
factor_df = loader.load()                       # full pool
factor_df = LabelMaker().make_labels(factor_df) # generate label_rank
splitter = RollingSplitter()                    # default 36/12/1

for w in splitter.split(factor_df):
    train, val, pred = w["train_df"], w["val_df"], w["pred_df"]
    # ... training
```

---

### 2.2 M2 Dual-Engine Training & Ensemble

**Directory**: `m2_engine/` (CPU path), `m2_engine_gpu/` (GPU-accelerated path)
**Entry point**: `python m2_engine/run_m2.py` or `from m2_engine.run_m2 import run_m2`

#### 2.2.1 Design Goals

- Run "feature engineering → dual-model training → soft-voting ensemble → Top10 portfolio construction" on every rolling window from M1.
- Strictly enforce **no data leakage**: all statistics (z-score, correlation, IC) are fitted **only on train**; the same transformation is applied to val/pred.
- GPU memory friendly: under the GTX 1650 4GB constraint, run single-GPU serially to avoid OOM.

#### 2.2.2 Pipeline (`_process_single_window`)

```
load window
  │
  ▼
FeatureStore.fit_transform (train, val, pred)   ← candidate factors → low-coverage filter → z-score → de-correlate → IC screen
  │
  ▼
EnsemblePredictor.fit_predict
  ├─ LGBMRanker.fit  (objective=regression_l1, GPU: OpenCL)
  ├─ XGBRanker.fit   (objective=reg:absoluteerror, GPU: CUDA)
  ├─ val ensemble predict → compute val_IC (Spearman)
  ├─ predict on train's last month → compute train_IC
  ├─ ic_gap = train_IC - val_IC
  └─ pred ensemble predict → score, score_cv
  │
  ▼
PortfolioBuilder.build
  ├─ sort and take Top20
  ├─ split into High (1-5) / Low (6-10) / Reserve (11-20)
  └─ weights: High=13%×5=65%, Low=7%×5=35%
  │
  ▼
backfill Target_Return_1M
  │
  ▼
compute_val_portfolio_metrics (used as M5 dependent variable)
  └─ 6-month rolling IR / Sortino / Capture ratio / Jensen α / β ...
```

#### 2.2.3 Core Modules

##### (1) `feature_store.py` — Feature Engineering

**6-step pipeline** (`fit_transform`):

1. **Candidate factor columns**: remove `META_COLS` (trade_date, stock_code, industry, close_price, etc.) and raw columns ending in `_raw`.
2. **Short-term noise filter** (optional): remove `*_5d / *_10d / *_1w` short-term reversal factors.
3. **Low-coverage filter**: drop columns with `valid_rate < min_valid_rate (0.30)`.
4. **Zero-variance filter**: drop `std < 1e-6` (typical of month-fixed macro factors).
5. **Cross-sectional Z-score**: in each `trade_date` cross-section do `(x - μ) / σ`. Macro columns `macro_*` only `fillna(0)`, no z-score. **Pure numpy implementation** (`_zscore_arr`), no pandas groupby overhead.
6. **High-correlation de-dup**: among factor pairs with `|corr| > max_corr (0.95)`, keep one (vectorized `np.triu`).
7. **IC screen**: compute each factor's Pearson IC against `label_rank` on train, keep the top `min_keep_factors (60)` with `|IC| ≥ min_ic_abs (0.003)`.

**Performance optimization (v3.8)**:
- After prefetch `to_numpy(copy=True)`, the entire pipeline runs on numpy (saves `_take_nd` dispatch overhead).
- One bulk write-back (`train_p[final_cols] = train_arr[:, final_idx]`) replaces 50 single-column writes.
- `transform()` reuses `_fit_state` to skip the corr/IC screen — **saves 50% feature-engineering time in M5 multi-Trial scenarios**.

##### (2) `lgbm_model.py` / `xgb_model.py` — Dual Model

| Dimension | LGBMRanker | XGBRanker |
|-----------|------------|-----------|
| Objective | `regression_l1` (MAE) | `reg:absoluteerror` (MAE) |
| Eval metric | `mae` + custom `ic` | custom `eval_ic` |
| Eval callback | `lgb.early_stopping(30)` + `log_evaluation(100)` | `early_stopping_rounds=30` |
| GPU device | `device_type="gpu"` (OpenCL) | `device="cuda"` (XGBoost 2.0+) |
| CPU threads (8-core optimal) | `nthread=2` | `nthread=3` |
| Early-stop rollback | `predict(num_iteration=best_iteration)` | `predict(iteration_range=(0, best_iteration))` |
| Adaptive leaves | `num_leaves = min(2^max_depth, train_size/200, 255)` | — |
| LR decay | callback `_make_lr_decay_callback` | scheduler `_make_xgb_lr_schedule` |

**Why MAE instead of lambdarank?**
- MAE is more robust to outliers, and rank correlation is stronger
- Avoids lambdarank's native memory crash in multi-thread / multi-Trial iteration

**Key defenses**:
- `__init__` must `dict(params) if params else {}` (shallow copy), **never modify the caller's dict in place**
- `predict` must roll back to `best_iteration`, not the last iteration
- Explicit `del lgb_train/lgb_val; del dtrain/dval; gc.collect()` to break the `Booster→callback→Dataset` reference cycle

##### (3) `ensemble.py` — Ensemble & Confidence

- **Soft voting**: `score = lgbm_w × lgbm_pred + xgb_w × xgb_pred` (default 0.5 each)
- **Confidence flag**:
  ```python
  ic_gap = train_IC - val_IC
  is_penalized = (ic_gap > 0.15)  # marker only, does not affect position
  ```
- **`compute_val_portfolio_metrics`**: simulates portfolio with 6-month rolling computation of 18 performance metrics (`val_rolling6m_ir`, `val_global_ir`, `pct_positive_excess`, `capture_ratio`, `val_jensen_alpha`…), used as M5 objective dependent variables
- **GPU serial / CPU serial**: because of the GTX 1650 4GB limit, **LGBM and XGB must be trained serially** (not in parallel). CPU mode is also serial (to avoid native memory conflict)
- **Vectorized IC computation** (`_per_group_rank_corr`): 1 `argsort` + 1 `unique` + per-group slice, 12-15ms / call faster than per-date `==` boolean mask

##### (4) `portfolio_builder.py` — Portfolio Construction

- Take **Top20**, split into tiers:
  - **High (top 5)**: 13% × 5 = 65%
  - **Low (6-10)**: 7% × 5 = 35%
  - **Reserve (11-20)**: 0%
- **Always fully invested**: 100% allocation, no cash signal
- **`is_penalized` does not affect position**: only written to `confidence_flag` for M4 report statistics
- **Turnover cost** (`calculate_turnover_cost`):
  - Stamp tax 0.1% (sell only)
  - Commission 0.03% (both sides)
  - Slippage 0.1% (both sides)

##### (5) `gpu_detector.py` — GPU Adaptive

- Singleton pattern; at startup uses 100-row data to actually test whether XGBoost CUDA / LightGBM OpenCL is available
- If either is available → default `mode="gpu"`, otherwise `"cpu"`
- `set_strategy("A"|"B"|"C"|"D"|"E")`: A/B/C = GPU training + GPU predict, D/E = CPU training

#### 2.2.4 Output

```
output/
├── all_portfolios.parquet   # full backtest positions (with score / weight / is_holding / Target_Return_1M)
├── all_portfolios.csv       # same, in CSV format
└── benchmark/               # benchmark results (split by strategy)
```

---

### 2.3 M4 Report Generation

**Directory**: `m4_report/`
**Entry point**: `python m4_report/report_generator.py` or `from m4_report.report_generator import generate_report`

#### 2.3.1 Design Goals

- Accept M2's output `all_portfolios.parquet`
- Compute **28+ performance metrics** (CAGR, Sharpe, Sortino, Calmar, IR, VaR, CVaR, Omega, Burke, Martin, etc.)
- **Automatically deduct turnover cost** (stamp tax + commission + slippage)
- **9 hard-threshold checks** (IR ≥ 0.5, Calmar ≥ 1.0, MaxDD ≥ -35%, etc.)
- Generate an **HTML report** (with embedded Chart.js equity curve)

#### 2.3.2 Core Modules

##### (1) `metrics.py` — `PerformanceMetrics`

**`compute_monthly_returns(all_portfolios)`**:
- Take only positions with `is_holding=True`
- Group by `pred_month` and compute the weighted sum of `weight × Target_Return_1M`
- Compute monthly turnover cost (diff vs last month)
- Load CSI 800 (000906.SH) as benchmark
- Output `monthly: [portfolio_return, benchmark_return, excess_return, turnover_cost, net_return]`

**`calculate(monthly)` — 28+ metrics**:

| Category | Metrics |
|----------|---------|
| Returns | CAGR, CAGR_benchmark, annual_excess, net_cagr_after_cost |
| Risk | max_drawdown, volatility, downside_volatility, upside_volatility, VaR(95%), CVaR(95%) |
| Ratios | sharpe_ratio, sortino_ratio, calmar_ratio, ir, sterling_ratio, burke_ratio, martin_ratio, omega_ratio, tail_ratio |
| Capture | up_capture_ratio, down_capture_ratio, capture_ratio |
| Distribution | skewness, kurtosis, pain_index, ulcer_index |
| Win Rate | monthly_win_rate, rolling6m_win_rate |
| Cost | avg_monthly_turnover_cost, avg_annual_turnover_cost |
| α/β | jensen_alpha (annualized), appraisal_ratio |
| Other | n_months, volatility_ratio |

**`check_thresholds(metrics)` — 9 thresholds** (from `config/config.yaml → performance`):

| Check | Threshold |
|-------|-----------|
| `ir_pass` | IR ≥ 0.50 |
| `calmar_pass` | Calmar ≥ 1.00 |
| `dd_pass` | MaxDD ≥ -35% |
| `sortino_pass` | Sortino ≥ 1.20 |
| `excess_pass` | annual_excess ≥ 5% |
| `roll6_pass` | rolling6m_win_rate ≥ 60% |
| `capture_pass` | capture_ratio ≥ 1.20 |
| `pain_pass` | pain_index ≤ 0.10 |
| `omega_pass` | omega_ratio ≥ 1.20 |

##### (2) `report_generator.py` — HTML Report

- Generates `output/backtest_report.html` with embedded Chart.js equity curve
- Report includes: parameter summary, equity-curve chart, 25+ core metrics, 9 threshold checks (✅ / ❌)
- Overall rating: `all_pass=True` → "✅ All Pass", otherwise "❌ Some Not Met"

#### 2.3.3 Output

```
output/backtest_report.html
```

---

### 2.4 M5 Bayesian Hyper-Parameter Optimization

**Directory**: `m5_optimizer/`
**Entry point**: Double-click `m5_optimizer/启动M5优化器.bat` (launches Gradio Web UI)
       or `python m5_optimizer/app.py`
**Web UI**: Open http://127.0.0.1:7860 in a browser

#### 2.4.1 Design Goals

- Automatically search M2's hyper-parameter space (30-D) and performance-objective space (37-D) via **Optuna TPE**
- **Two-phase strategy**: Phase1 global exploration → Phase2 local fine-tuning
- **Project-oriented**: each optimization project has its own SQLite database, supports checkpoint/resume and warm-start
- **Gradio visualization**: real-time progress, parameter retro-analysis, Top-5 Trial ranking

#### 2.4.2 Core Modules

##### (1) `search_space.py` — 30-D Search Space

| Group | Count | Key Parameters |
|-------|-------|----------------|
| **LGBM** | 12 | learning_rate (log, 0.01-0.1), n_estimators (100-300), max_depth (2-4), colsample_bytree (0.1-0.3), reg_alpha/lambda, min_split_gain, lr_mode (fixed/decay), decay_every, decay_factor, depth_mode, early_stopping_rounds |
| **XGBoost** | 11 | learning_rate, n_estimators, max_depth, colsample_bytree, reg_alpha/lambda, gamma, lr_mode, decay_every, decay_factor, early_stopping_rounds |
| **Ensemble** | 1 | lgbm_weight (0.3-0.7) |
| **Feature** | 5 | min_valid_rate (0.2-0.5), max_corr (0.8-0.97), min_ic_abs (log, 0.002-0.02), min_keep_factors (50-120), drop_short_term_noise |
| **Window** | 1 | train_months (52-60) |

**`OBJECTIVE_VARS` — 37-D dependent variables** (grouped by source):

- **M2 validation-set (17)**: val_ic, val_icir, val_rolling6m_ir, val_rolling6m_sortino, val_rolling6m_return, ic_gap_penalty, penalized_rate, val_global_ir, val_annual_return, pct_positive_excess, ir_worst_quartile, val_rolling6m_excess, val_rolling6m_excess_ann, val_jensen_alpha, val_appraisal_ratio, val_beta, val_ic_stability
- **M2 stress-test (3, disabled)**: 2008 / 2015 / 2022
- **M4 backtest (17)**: cagr, monthly_win_rate, downside_volatility, upside_volatility, volatility_ratio, var_95, cvar_95, skewness, kurtosis, pain_index, omega_ratio, burke_ratio, martin_ratio, tail_ratio, up_capture_ratio, down_capture_ratio (disabled), capture_ratio (disabled)

**`NORM_CONFIG` — Score Normalization** (3 methods):

| Method | Use Case | Formula |
|--------|----------|---------|
| `linear` | Rate metrics with bounded range | `(clip(x, lo, hi) - lo) / (hi - lo) - 0.5` → [-0.5, 0.5] |
| `tanh` | IR / Sortino-like, clustered around 0 | `tanh(x / scale)` → (-1, 1) |
| `signed_log` | Long-tailed / high-volatility | `sign(x) × log(1 + \|x\|)` |

##### (2) `objective.py` — Optuna Objective Function

**`ObjectiveFunction.__call__(trial)`** main flow:

1. Sample 30 hyper-parameters (`_sample_param` supports custom intervals + active_params subset)
2. `assemble_params` splits them by group back into `lgbm_params / xgbm_params / feature_params / lgbm_weight / window_params`
3. Constraint check: `lr × n_estimators ≤ 15.0`, to avoid too-slow overfitting
4. Call `run_m2(**_run_m2_kwargs)`, supports GPU path (when `gpu_mode=True` uses `m2_engine_gpu.run_m2`)
5. Extract 18 M2 dependent variables + 3 estimated M4 metrics
6. Normalize (optional) → weighted sum → `score`
7. `trial.set_user_attr(metric, value)` to persist each metric
8. Return `-score` (Optuna's minimization direction)

**`IC_GAP_PENALTY_MULTIPLIER = 1.5`**: the IC overfit penalty term is amplified 1.5×

##### (3) `phase1_global.py` — Global Exploration

- **Sampler**: `TPESampler(seed=42, n_startup_trials=15, n_ei_candidates=24, constraints_func=...)`
- **Constraint function**: returns `lr × n_estimators - 15.0` (≤ 0 means satisfied)
- **Checkpoint resume**: uses `optuna.load_study(..., storage="sqlite:///...")` to auto-recover
- **Warm-start prior**: `study.enqueue_trial(warm_start)` injects 1 default Trial
- **Trial callback**: `make_trial_callback` writes to `RollingLogger`, supports "immediate stop" and "graceful stop" dual mechanisms

##### (4) `phase2_local.py` — Local Fine-Tuning

- **Start point**: best Trial's parameters from Phase1
- **Tightened range**: adjusts P2 search interval based on P1 retro-analysis (`_build_p2_distributions`)
- **Trial reuse**: injects P1 Trials that fall within P2's range into the P2 study as COMPLETE
- **Independent SQLite**: stored separately from P1, to avoid pollution

##### (5) `app.py` — Gradio Web UI (5 Tabs)

| Tab | Function |
|-----|----------|
| **Tab1** Training Control | Select project → start P1/P2 → real-time progress bar → immediate stop / graceful stop |
| **Tab2** Result Analysis & Retro | Display P1 Trial statistics, filter by dependent-variable slider, retro-analyze P2 search range |
| **Tab3** Top-5 Ranking | Show details of top 5 Trials after joint P1+P2 ranking |
| **Tab4** Parameter Heatmap | Factor-IC relationship, parameter-score sensitivity chart |
| **Tab5** Project Management | Create / clone / delete optimization projects, view history |

##### (6) `utils/` — Helper Tools

| File | Responsibility |
|------|----------------|
| `logger.py` | Unified logging facade |
| `rolling_logger.py` | Rolling log (by Trial / system state / error, 3 categories) |
| `memory_monitor.py` | Memory monitoring (psutil) |
| `win_memory.py` | Windows `release_memory_to_os()` (forced reclaim) |
| `trial_callback.py` | Optuna callback factory |
| `retroactive_normalize.py` | Post-hoc normalization (used for retro-analyzing P2 ranges) |

#### 2.4.3 Optimization Workflow (Typical)

```
1. Create project (Tab5)
   → automatically generate projects/{name}/p1.db
2. Choose active_params (subset of 30-D, decided in Tab2)
3. Set objective_weights (which dependent variables participate in scoring + weights)
4. Start Phase1 (Tab1)
   → 50-200 TPE Trials
   → SQLite persists each Trial's 30-D params + 37-D metrics
5. Retro-analyze P2 range (Tab2)
   → slider filter for "effective range"
6. Start Phase2 (Tab1)
   → 30-100 Trials, search range tightened
7. View Top-5 (Tab3)
```

#### 2.4.4 Output

```
projects/{project_name}/
├── p1/
│   ├── p1.db               # SQLite, Optuna Study
│   ├── p1_config.json      # P1 config (active_params / weights / param_ranges)
│   └── p1_summary.json     # written after P1 completion
└── p2/
    ├── p2.db
    └── p2_config.json
```

---

## 3. Entry-Point Quick Reference

| Program | Command | Function |
|---------|---------|----------|
| `run_m0_full.py` | `python run_m0_full.py` | Pull full Tushare data → 4 neutralization schemes Parquet storage (with checkpoint resume) |
| `m1_engine/run_m1.py` | `python m1_engine\run_m1.py` | M1 rolling split → window Parquet storage |
| `m2_engine/run_m2.py` | `python m2_engine\run_m2.py` | M2 dual-engine training → all_portfolios.parquet |
| `run_m2_m4.py` | `python run_m2_m4.py` | One-click M2 + M4 (CPU mode) |
| `run_one_full.py` | `python run_one_full.py baseline 60` | Single-strategy full benchmark (strategy: baseline/A/B/C/D, n_windows: 60/186) |
| `m4_report/report_generator.py` | `python m4_report\report_generator.py` | M4 HTML report generation |
| `m5_optimizer/启动M5优化器.bat` | Double-click | Launch Gradio Web UI (http://127.0.0.1:7860) |
| `m5_optimizer/app.py` | `python m5_optimizer\app.py` | Same as above (command-line equivalent) |

---

## 4. Directory Layout

```
10q-202604gpu/
├── config/                    # configuration (yaml + thread constants)
│   ├── config.yaml            # main config (neutralization, rolling, cost, 9 thresholds)
│   └── concurrency_config.py  # adaptive threads / concurrency
│
├── m0_database/               # M0 data module
│   ├── pipeline.py            # top-level pipeline
│   ├── data_fetcher.py        # Tushare data fetching
│   ├── stock_filter.py        # stock pool filtering (ST/delisted/newly-listed)
│   ├── factor_calculator.py   # factor calculation
│   ├── neutralization.py      # neutralization (Rank-Z + dual OLS)
│   ├── format_validator.py    # data format validation
│   ├── regenerator.py         # incremental regeneration
│   └── _preflight_check.py    # pre-launch environment check
│
├── m1_engine/                 # M1 rolling split
│   ├── data_loader.py         # M0 data loading (parallel parquet → float32 DataFrame)
│   ├── label_maker.py         # cross-section label_rank generation
│   ├── rolling_splitter.py    # time-series rolling splitter (generator)
│   └── run_m1.py              # top-level entry
│
├── m2_engine/                 # M2 CPU training
│   ├── ensemble.py            # ensemble + IC computation + 18 validation-set metrics
│   ├── feature_store.py       # feature engineering pipeline
│   ├── lgbm_model.py          # LightGBM Ranker (CPU/OpenCL)
│   ├── xgb_model.py           # XGBoost Ranker (CPU/CUDA)
│   ├── portfolio_builder.py   # portfolio construction + turnover cost
│   ├── preprocessor.py        # data preloading
│   ├── smart_preprocessor.py  # efficient preprocessor
│   ├── lightweight_loader.py  # ultra-lightweight loader (avoid OOM)
│   ├── gpu_detector.py        # GPU/CPU adaptive
│   └── run_m2.py              # top-level entry
│
├── m2_engine_gpu/             # M2 GPU-accelerated training (parallel to m2_engine)
│   ├── ensemble.py            # GPU strategies A/B/C/D/E
│   ├── feature_store.py
│   ├── lgbm_model.py          # GPU LightGBM
│   ├── xgb_model.py           # GPU XGBoost
│   ├── portfolio_builder.py
│   ├── gpu_detector.py
│   └── run_m2.py
│
├── m4_report/                 # M4 backtest report
│   ├── metrics.py             # 28+ performance metrics + 9 thresholds
│   └── report_generator.py    # HTML report (Chart.js equity curve)
│
├── m5_optimizer/              # M5 Bayesian optimizer
│   ├── app.py                 # Gradio Web UI (5 tabs)
│   ├── search_space.py        # 30-D params + 37-D dependent variables + NORM_CONFIG
│   ├── objective.py           # Optuna objective function
│   ├── phase1_global.py       # Phase1 global exploration
│   ├── phase2_local.py        # Phase2 local fine-tuning
│   ├── config_manager.py      # M5 config read/write
│   ├── project_manager.py     # project management
│   ├── result_analyzer.py     # P1 result analysis
│   ├── range_analyzer.py      # P2 range retro-analysis
│   ├── utils/                 # helper tools
│   └── 启动M5优化器.bat       # one-click launcher
│
├── run_*.py                   # top-level entry points
├── bench_*.py / analyze_*.py  # benchmark / analysis scripts
├── analyze_m5_p1.py           # P1 report generation
├── p1_report_generator.py     # P1 HTML report
├── requirements.txt
├── README.md                  # Chinese (primary)
├── README_EN.md               # English
└── .gitignore
```

> ⚠️ **Not in version control**: `data/`, `logs/`, `output/`, all `*.pkl / *.parquet / *.db` caches.
> See [.gitignore](.gitignore) for details.

---

## 5. Quick Start

### 5.1 Environment Requirements

> ⚠️ **Identical to the "📋 Runtime Environment Requirements" section: validated only on Windows 11 + Python 3.14.**

- **Python** **3.14.x** (≥ 3.14.0 required)
- **OS**: **Windows 11** (22H2 / 23H2 / 24H2 all work; **other systems not tested**)
- **RAM**: 16 GB+ (CPU mode M2 full ≈ 12-14 GB)
- **GPU** (recommended): NVIDIA GPU, CUDA 12.x, 4GB+ VRAM
  - Validated: GTX 1650 4GB serial training, no OOM
- **CPU**: 8 cores optimal (measured LGBM=2 threads + XGB=3 threads is the optimal combo)

### 5.2 Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5.3 Configure Tushare Token

**Must** be set via environment variable (**never** write it into yaml):

```powershell
$env:TUSHARE_TOKEN = "your_tushare_pro_token"
```

Register at https://tushare.pro and obtain it from your personal center. **Under no circumstances write the token into config.yaml**.

### 5.4 Run the Pipeline

```powershell
# M0 pull data + compute factors (takes hours the first time, cached in data/raw_cache/)
python run_m0_full.py

# M1 rolling split
python m1_engine\run_m1.py

# M2 training + portfolio construction
python m2_engine\run_m2.py

# M4 report
python m4_report\report_generator.py

# M5 launch optimizer Web UI
.\m5_optimizer\启动M5优化器.bat
# Open http://127.0.0.1:7860 in your browser
```

---

## 6. Performance Constraints & Baseline

| Item | Value |
|------|-------|
| Rolling training window | 36 months train / 12 months val / 1 month test |
| Default step | 1 month |
| Total windows (2007-01 ~ 2025-12) | 228 - 49 + 1 = 180 windows |
| Portfolio capacity | Top10 always fully invested (65% + 35%) |
| CPU-mode memory peak | 12-14 GB |
| GPU-mode VRAM peak | ~3.5 GB (GTX 1650) |
| 8-core CPU single-window latency | ~0.76s (benchmark_v42 measured) |
| GPU speedup | 1.5-2.5× (depends on data size) |

---

## 7. Security Notes

- **Never commit the Tushare token**! The repo is configured to read it from the `TUSHARE_TOKEN` environment variable
- Historical commits have been cleaned of plaintext tokens
- If the token is leaked, reset it immediately at https://tushare.pro
- **No `*.db` / `*.parquet` / `*.pkl` files are in git**; see [.gitignore](.gitignore)

---

## 8. License

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

### 8.1 What You Can Do

| Action | Allowed |
|--------|---------|
| Personal learning, research, source-code reading | ✅ |
| Forking / modifying for your own projects | ✅ |
| Commercial use, deployment to production | ✅ |
| Redistribution (with copyright and license preserved) | ✅ |
| Patent grants | ✅ |
| Proprietary use | ✅ |
| Publishing modifications under a different license | ✅ (original copyright must be preserved) |

### 8.2 What You Must Do

- **Preserve copyright notice**: keep `Copyright 2026 zjwandcat` in all copies / derivative works
- **Mark modifications**: clearly state "modified" if you modified source files
- **Include LICENSE copy**: a copy of this LICENSE must accompany redistribution
- **NOTICE file** (if modified): attribution notices in [NOTICE](NOTICE) must be preserved
- **Patent grant termination**: if you file patent litigation against any Contributor, all patent grants from that Contributor automatically terminate

### 8.3 Third-Party Dependencies

This project depends on multiple third-party open-source libraries (pandas, numpy, lightgbm, xgboost, optuna, gradio, tushare, etc.). The complete list and licenses are in [NOTICE](NOTICE). These dependencies retain their original licenses and are **not constrained by this project's Apache 2.0**.

### 8.4 Risk Disclaimer

This project is intended for quantitative research and learning only. **It does not constitute any investment advice**. The author **accepts no responsibility** for any investment loss arising from the use of this project's code. See LICENSE sections 7 and 8 (no warranty / limitation of liability).

---

## 9. Acknowledgments

Thanks to [Tushare Pro](https://tushare.pro) for providing high-quality A-share data APIs, and to the LightGBM, XGBoost, Optuna, and Gradio open-source communities.
