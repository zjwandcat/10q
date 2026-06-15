# TTHH A 股量化选股系统 (m01245)

> 私有仓库 · 由 [zjwandcat](https://github.com/zjwandcat) 维护
>
> 基于 Tushare 数据 + 多模块流水线 + GPU 加速 XGBoost/LightGBM 的 A 股量化选股回测与超参优化系统

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

**Copyright © 2026 zjwandcat. Licensed under the [Apache License, Version 2.0](LICENSE).**

---

## 一、系统总览

本系统是一套面向 A 股市场的 **多模块滚动窗口选股 + 集成学习 + 贝叶斯超参优化** 闭环流水线，整体分为 **M0 → M1 → M2 → M4 → M5** 五层：

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  Tushare Pro API                                                      │
│        │                                                              │
│        ▼                                                              │
│  ┌────────┐    ┌────────┐    ┌──────────┐    ┌────────┐    ┌────────┐│
│  │   M0   │───▶│   M1   │───▶│  M2 / GPU│───▶│   M4   │    │   M5   ││
│  │ 数据库 │    │ 切分   │    │ 训练/集成 │    │ 回测   │    │ 优化器 ││
│  └────────┘    └────────┘    └──────────┘    └────────┘    └────────┘│
│  Parquet      滚动窗口      LGBM+XGB        28+项绩效      Optuna     │
│  落盘        + 标签生成      Top10 持仓      HTML 报告      TPE 贝叶斯 │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

| 模块 | 路径 | 核心职责 |
|------|------|----------|
| **M0** 数据 | `m0_database/` | Tushare 拉数、股票池筛选、因子计算、行业/市值中性化、Parquet 落盘 |
| **M1** 滚动切分 | `m1_engine/` | 时序滚动 train/val/pred 切分、截面 Label 生成 |
| **M2** 模型训练 | `m2_engine/`, `m2_engine_gpu/` | 特征工程、LGBM/XGBoost 训练、软投票集成、Top10 持仓构建 |
| **M4** 报告 | `m4_report/` | 月度收益、换手成本、28+ 项绩效指标、HTML 报告 |
| **M5** 优化器 | `m5_optimizer/` | Optuna TPE 双阶段贝叶斯优化、Gradio Web UI、项目管理 |

---

## 二、模块技术细节

### 2.1 M1 滚动切分引擎

**目录**：`m1_engine/`
**入口**：`python m1_engine/run_m1.py` 或 `run_m1(force_rebuild=True)`

#### 2.1.1 设计目标

- 把 M0 输出的"全月截面"数据，按时间顺序切分为可独立训练的滚动窗口
- **严格无数据泄露**：每个窗口的 train < val < pred 时间严格单调
- **内存友好**：使用 Python 生成器（`yield`）逐窗口切片，不一次性展开到内存

#### 2.1.2 核心组件

| 文件 | 职责 |
|------|------|
| `data_loader.py` | 并行读取 M0 的 Parquet 池，转 float32，转 Categorical，省内存 |
| `label_maker.py` | 在每个 `trade_date` 截面内对 `Target_Return_1M` 做升序排名，生成 `label_rank ∈ [0,1]` |
| `rolling_splitter.py` | 时序滚动切分器，yield 一组 `{train_df, val_df, pred_df, window_idx, pred_month}` |
| `run_m1.py` | 顶层入口，串起 DataLoader → LabelMaker → RollingSplitter → 落盘 |

#### 2.1.3 滚动窗口参数

`config/config.yaml → rolling` 节定义：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `train_months` | **36** | 训练窗口月数 |
| `valid_months` | **12** | 验证窗口月数 |
| `test_months` | **1** | 预测窗口月数（生成下月持仓） |
| `step_months` | **1** | 每次向前滑动的月数 |

**窗口边界**（`rolling_splitter.py` 严格按此划分）：

```
all_months = [m_0, m_1, ..., m_{T-1}]
窗口 i（0-indexed）:
  train: m_i, m_{i+1}, ..., m_{i+35}      (共 36 个月)
  val:   m_{i+36}, ..., m_{i+47}           (共 12 个月)
  pred:  m_{i+48}                          (共  1 个月)
```

#### 2.1.4 关键实现要点

1. **并行 Parquet 读取**（`DataLoader.load()`）：`ThreadPoolExecutor` 异步读取，I/O 与 CPU 解耦；每读完 20 个文件打一次进度日志。
2. **内存优化**：
   - 数值列统一 `astype(np.float32)`，相比 float64 省 50%
   - 股票代码列 `astype("category")`，省 40-60% 字符串列内存，加速 `groupby`
   - `df.copy()` 显式去碎片
3. **截面 Label 生成**（`LabelMaker.make_labels()`）：
   ```python
   df["label_rank"] = (
       df.groupby("trade_date")["Target_Return_1M"]
         .transform(lambda x: x.rank(method="average",
                                     ascending=True,
                                     pct=True,
                                     na_option="keep"))
   )
   ```
   - `pct=True` 直接得到 0~1 分位数
   - `na_option="keep"` 让 NaN 保持 NaN，不污染训练
4. **滚动切分生成器**（`RollingSplitter.split()`）：用 Python 生成器（yield）逐窗口返回，**不预先生成所有窗口到内存**。每个窗口内部切片后 `.copy()` 防 View 共享导致后续修改污染源数据。
5. **窗口输出格式**：每个窗口保存为一个 `window_{idx:03d}_{pred_month}.parquet`，并用 `_split ∈ {train, val, pred}` 列标记来源。
6. **窗口统计**：`summary.json` + `window_stats.csv` 记录总窗口数、起止月份、平均行列数等。

#### 2.1.5 输出

```
output/m1_windows/
├── summary.json                  # 整体统计指纹
├── window_stats.csv              # 每窗口的行列数
├── window_000_201001.parquet     # 窗口 0，预测 2010-01
├── window_001_201002.parquet
└── ...
```

#### 2.1.6 调用样例

```python
from m1_engine.data_loader import DataLoader
from m1_engine.label_maker import LabelMaker
from m1_engine.rolling_splitter import RollingSplitter

loader = DataLoader(scheme="scheme_d")         # 选中性化方案
factor_df = loader.load()                       # 全量池
factor_df = LabelMaker().make_labels(factor_df) # 生成 label_rank
splitter = RollingSplitter()                    # 默认 36/12/1

for w in splitter.split(factor_df):
    train, val, pred = w["train_df"], w["val_df"], w["pred_df"]
    # ... 训练
```

---

### 2.2 M2 双引擎训练与集成

**目录**：`m2_engine/`（CPU 路径），`m2_engine_gpu/`（GPU 加速路径）
**入口**：`python m2_engine/run_m2.py` 或 `from m2_engine.run_m2 import run_m2`

#### 2.2.1 设计目标

- 在 M1 输出的每个滚动窗口上做"特征工程 → 双模型训练 → 软投票集成 → Top10 持仓构建"
- 严格遵守**无数据泄露**：所有统计量（z-score、corr、IC）**只在 train 上 fit**，val/pred 复用同一变换
- GPU 显存友好：GTX 1650 4GB 限制下做单卡串行，避免 OOM

#### 2.2.2 流水线（`_process_single_window`）

```
load window
  │
  ▼
FeatureStore.fit_transform (train, val, pred)   ← 候选因子→低覆盖过滤→z-score→去相关→IC筛选
  │
  ▼
EnsemblePredictor.fit_predict
  ├─ LGBMRanker.fit  (objective=regression_l1, GPU: OpenCL)
  ├─ XGBRanker.fit   (objective=reg:absoluteerror, GPU: CUDA)
  ├─ val 集成预测 → 算 val_IC（Spearman）
  ├─ train 最后1月预测 → 算 train_IC
  ├─ ic_gap = train_IC - val_IC
  └─ pred 集成预测 → score, score_cv
  │
  ▼
PortfolioBuilder.build
  ├─ 排序取 Top20
  ├─ 分层 High (1-5) / Low (6-10) / Reserve (11-20)
  └─ 仓位：High=13%×5=65%, Low=7%×5=35%
  │
  ▼
backfill Target_Return_1M
  │
  ▼
compute_val_portfolio_metrics (M5 因变量用)
  └─ 6月滚动 IR / Sortino / 捕获比 / Jensen α / β ...
```

#### 2.2.3 核心模块

##### (1) `feature_store.py` — 特征工程

**6 步管线**（`fit_transform`）：

1. **候选因子列**：剔除 `META_COLS`（trade_date、stock_code、industry、close_price 等元信息列）和以 `_raw` 结尾的原始列
2. **短期噪音过滤**（可选）：剔除 `*_5d / *_10d / *_1w` 短期反转类因子
3. **低覆盖过滤**：`valid_rate < min_valid_rate (0.30)` 的列丢弃
4. **零方差列过滤**：`std < 1e-6`（常见于同月同值的宏观因子）丢弃
5. **截面 Z-score**：在每个 `trade_date` 截面内做 `(x - μ) / σ`，宏观列 `macro_*` 仅 fillna(0) 不做 z-score。**纯 numpy 实现**（`_zscore_arr`），无 pandas groupby 开销
6. **高相关去重**：`|corr| > max_corr (0.95)` 的两两冗余因子中保留一个（用向量化 `np.triu`）
7. **IC 筛选**：在 train 上计算每个因子与 `label_rank` 的 Pearson，取 `|IC| ≥ min_ic_abs (0.003)` 的前 `min_keep_factors (60)` 个

**性能优化（v3.8）**：
- 预取 `to_numpy(copy=True)` 后全程在 numpy 上做（省 `_take_nd` 调度）
- 1 次批量写回（`train_p[final_cols] = train_arr[:, final_idx]`）替代 50 次单列写
- `transform()` 复用 `_fit_state` 跳过 corr/IC 筛选，**M5 多 Trial 场景下省 50% 因子工程时间**

##### (2) `lgbm_model.py` / `xgb_model.py` — 双模型

| 维度 | LGBMRanker | XGBRanker |
|------|------------|-----------|
| 目标函数 | `regression_l1`（MAE） | `reg:absoluteerror`（MAE） |
| 评测指标 | `mae` + 自定义 `ic` | 自定义 `eval_ic` |
| 评估回调 | `lgb.early_stopping(30)` + `log_evaluation(100)` | `early_stopping_rounds=30` |
| GPU 设备 | `device_type="gpu"`（OpenCL） | `device="cuda"`（XGBoost 2.0+） |
| CPU 线程（8 核最优） | `nthread=2` | `nthread=3` |
| 早停回滚 | `predict(num_iteration=best_iteration)` | `predict(iteration_range=(0, best_iteration))` |
| 自适应叶子数 | `num_leaves = min(2^max_depth, train_size/200, 255)` | — |
| 学习率衰减 | 回调 `_make_lr_decay_callback` | 调度 `_make_xgb_lr_schedule` |

**为什么用 MAE 而非 lambdarank？**
- MAE 对异常值更鲁棒，rank 相关性更强
- 避免 lambdarank 在多线程 / 多 Trial 迭代中的 native 内存崩溃

**关键防御**：
- `__init__` 必须 `dict(params) if params else {}` 浅拷贝，**禁止原地修改调用方 dict**
- `predict` 必须用 `best_iteration` 回滚，不用最后一轮
- 显式 `del lgb_train/lgb_val; del dtrain/dval; gc.collect()` 打破 `Booster→callback→Dataset` 循环引用

##### (3) `ensemble.py` — 集成与置信度

- **软投票**：`score = lgbm_w × lgbm_pred + xgb_w × xgb_pred`（默认各 0.5）
- **置信度标记**：
  ```python
  ic_gap = train_IC - val_IC
  is_penalized = (ic_gap > 0.15)  # 仅作标记，不影响仓位
  ```
- **`compute_val_portfolio_metrics`**：模拟持仓 6 月滚动计算 18 项绩效指标（`val_rolling6m_ir`、`val_global_ir`、`pct_positive_excess`、`capture_ratio`、`val_jensen_alpha`…），供 M5 目标函数作为因变量
- **GPU 串行 / CPU 串行**：因 GTX 1650 4GB 限制，**LGBM 与 XGB 必须串行训练**（不并行），CPU 模式同样串行（避免 native 内存冲突）
- **向量化 IC 计算**（`_per_group_rank_corr`）：1 次 `argsort` + 1 次 `unique` + per-group slice，比 per-date `==` 布尔 mask 快 12-15ms / 调用

##### (4) `portfolio_builder.py` — 持仓构建

- **取 Top20**，分层：
  - **High（前 5）**：权重 13% × 5 = 65%
  - **Low（6-10）**：权重 7% × 5 = 35%
  - **Reserve（11-20）**：权重 0%
- **永远满仓**：100% 仓位，无空仓信号
- **`is_penalized` 不影响仓位**：只写到 `confidence_flag` 用于 M4 报告统计
- **换手成本**（`calculate_turnover_cost`）：
  - 印花税 0.1%（仅卖出）
  - 佣金 0.03%（双边）
  - 滑点 0.1%（双边）

##### (5) `gpu_detector.py` — GPU 自适应

- 单例模式，启动时用 100 行小数据实测 XGBoost CUDA / LightGBM OpenCL 是否可用
- 任意一个可用 → 默认 `mode="gpu"`，否则 `"cpu"`
- `set_strategy("A"|"B"|"C"|"D"|"E")`：A/B/C = GPU 训练 + GPU 预测，D/E = CPU 训练

#### 2.2.4 输出

```
output/
├── all_portfolios.parquet   # 全量回测持仓（含 score / weight / is_holding / Target_Return_1M）
├── all_portfolios.csv       # 同上，CSV 格式
└── benchmark/               # benchmark 结果（按策略切分）
```

---

### 2.3 M4 报告生成

**目录**：`m4_report/`
**入口**：`python m4_report/report_generator.py` 或 `from m4_report.report_generator import generate_report`

#### 2.3.1 设计目标

- 接收 M2 输出的 `all_portfolios.parquet`
- 计算 **28+ 项绩效指标**（CAGR、夏普、Sortino、Calmar、IR、VaR、CVaR、Omega、Burke、Martin…）
- **自动扣除换手成本**（印花税 + 佣金 + 滑点）
- **9 项硬性门槛检查**（IR ≥ 0.5、Calmar ≥ 1.0、MaxDD ≥ -35%…）
- 生成 **HTML 报告**（内嵌 Chart.js 净值曲线）

#### 2.3.2 核心模块

##### (1) `metrics.py` — `PerformanceMetrics`

**`compute_monthly_returns(all_portfolios)`**：
- 仅取 `is_holding=True` 的持仓
- 按 `pred_month` 分组做 `weight × Target_Return_1M` 的加权和
- 计算每月的换手成本（与上月持仓 diff）
- 加载中证 800（CSI800，000906.SH）作为基准
- 输出 `monthly: [portfolio_return, benchmark_return, excess_return, turnover_cost, net_return]`

**`calculate(monthly)` — 28+ 项指标**：

| 类别 | 指标 |
|------|------|
| 收益 | CAGR、CAGR_benchmark、annual_excess、net_cagr_after_cost |
| 风险 | max_drawdown、volatility、downside_volatility、upside_volatility、VaR(95%)、CVaR(95%) |
| 比率 | sharpe_ratio、sortino_ratio、calmar_ratio、ir、sterling_ratio、burke_ratio、martin_ratio、omega_ratio、tail_ratio |
| 捕获 | up_capture_ratio、down_capture_ratio、capture_ratio |
| 分布 | skewness、kurtosis、pain_index、ulcer_index |
| 胜率 | monthly_win_rate、rolling6m_win_rate |
| 成本 | avg_monthly_turnover_cost、avg_annual_turnover_cost |
| α/β | jensen_alpha（年化）、appraisal_ratio |
| 其他 | n_months、volatility_ratio |

**`check_thresholds(metrics)` — 9 项门槛**（`config/config.yaml → performance`）：

| 检验项 | 阈值 |
|--------|------|
| `ir_pass` | IR ≥ 0.50 |
| `calmar_pass` | Calmar ≥ 1.00 |
| `dd_pass` | MaxDD ≥ -35% |
| `sortino_pass` | Sortino ≥ 1.20 |
| `excess_pass` | annual_excess ≥ 5% |
| `roll6_pass` | rolling6m_win_rate ≥ 60% |
| `capture_pass` | capture_ratio ≥ 1.20 |
| `pain_pass` | pain_index ≤ 0.10 |
| `omega_pass` | omega_ratio ≥ 1.20 |

##### (2) `report_generator.py` — HTML 报告

- 生成 `output/backtest_report.html`，内嵌 Chart.js 净值曲线
- 报告包含：参数摘要、净值曲线图、25+ 项核心指标、9 项门槛检验（✅ / ❌）
- 综合评级：`all_pass=True` → "✅ 全部通过"，否则 "❌ 存在未达标项"

#### 2.3.3 输出

```
output/backtest_report.html
```

---

### 2.4 M5 贝叶斯超参优化

**目录**：`m5_optimizer/`
**入口**：双击 `m5_optimizer/启动M5优化器.bat`（启动 Gradio Web UI）
       或 `python m5_optimizer/app.py`
**Web UI**：浏览器打开 http://127.0.0.1:7860

#### 2.4.1 设计目标

- 把 M2 的超参数空间（30 维）和绩效目标空间（37 维）通过 **Optuna TPE** 自动搜索
- **双阶段策略**：Phase1 全局探索 → Phase2 局部精调
- **项目化**：每个优化项目独立 SQLite 数据库，支持断点续跑、热启动
- **Gradio 可视化**：实时进度、参数反推、Top-5 Trial 排名

#### 2.4.2 核心模块

##### (1) `search_space.py` — 30 维搜索空间

| 组 | 数量 | 关键参数 |
|----|------|----------|
| **LGBM** | 12 | learning_rate (log, 0.01-0.1)、n_estimators (100-300)、max_depth (2-4)、colsample_bytree (0.1-0.3)、reg_alpha/lambda、min_split_gain、lr_mode (fixed/decay)、decay_every、decay_factor、depth_mode、early_stopping_rounds |
| **XGBoost** | 11 | learning_rate、n_estimators、max_depth、colsample_bytree、reg_alpha/lambda、gamma、lr_mode、decay_every、decay_factor、early_stopping_rounds |
| **Ensemble** | 1 | lgbm_weight (0.3-0.7) |
| **Feature** | 5 | min_valid_rate (0.2-0.5)、max_corr (0.8-0.97)、min_ic_abs (log, 0.002-0.02)、min_keep_factors (50-120)、drop_short_term_noise |
| **Window** | 1 | train_months (52-60) |

**`OBJECTIVE_VARS` — 37 维因变量**（按 source 划分）：

- **M2 验证集类**（17 项）：val_ic、val_icir、val_rolling6m_ir、val_rolling6m_sortino、val_rolling6m_return、ic_gap_penalty、penalized_rate、val_global_ir、val_annual_return、pct_positive_excess、ir_worst_quartile、val_rolling6m_excess、val_rolling6m_excess_ann、val_jensen_alpha、val_appraisal_ratio、val_beta、val_ic_stability
- **M2 压力测试**（3 项，disabled）：2008/2015/2022
- **M4 回测类**（17 项）：cagr、monthly_win_rate、downside_volatility、upside_volatility、volatility_ratio、var_95、cvar_95、skewness、kurtosis、pain_index、omega_ratio、burke_ratio、martin_ratio、tail_ratio、up_capture_ratio、down_capture_ratio（disabled）、capture_ratio（disabled）

**`NORM_CONFIG` — 评分归一化**（3 种方法）：

| 方法 | 适用场景 | 公式 |
|------|----------|------|
| `linear` | 值域有边界的率指标 | `(clip(x, lo, hi) - lo) / (hi - lo) - 0.5` → [-0.5, 0.5] |
| `tanh` | IR/Sortino 类，集中 0 附近 | `tanh(x / scale)` → (-1, 1) |
| `signed_log` | 长尾 / 剧烈波动 | `sign(x) × log(1 + |x|)` |

##### (2) `objective.py` — Optuna 目标函数

**`ObjectiveFunction.__call__(trial)`** 主流程：

1. 采样 30 个超参数（`_sample_param` 支持自定义区间 + active_params 子集）
2. `assemble_params` 按 group 拆装回 `lgbm_params / xgbm_params / feature_params / lgbm_weight / window_params`
3. 约束检查：`lr × n_estimators ≤ 15.0`，避免过慢过拟合
4. 调用 `run_m2(**_run_m2_kwargs)`，支持 GPU 路径（`gpu_mode=True` 时走 `m2_engine_gpu.run_m2`）
5. 提取 18 项 M2 因变量 + 3 项 M4 估算指标
6. 归一化（可选）→ 加权求和 → `score`
7. `trial.set_user_attr(metric, value)` 持久化每个指标
8. 返回 `-score`（Optuna 最小化方向）

**`IC_GAP_PENALTY_MULTIPLIER = 1.5`**：IC 过拟合惩罚项权重放大 1.5 倍

##### (3) `phase1_global.py` — 全局探索

- **Sampler**：`TPESampler(seed=42, n_startup_trials=15, n_ei_candidates=24, constraints_func=...)`
- **约束函数**：返回 `lr × n_estimators - 15.0`（≤ 0 表示满足）
- **断点续跑**：用 `optuna.load_study(..., storage="sqlite:///...")` 自动恢复
- **热启动先验**：`study.enqueue_trial(warm_start)` 注入 1 个默认 Trial
- **Trial 回调**：`make_trial_callback` 写入 `RollingLogger`，支持「立即停止」和「优雅停止」双机制

##### (4) `phase2_local.py` — 局部精调

- **起点**：Phase1 最优 Trial 的参数
- **范围收紧**：根据 P1 反推结果调整 P2 搜索区间（`_build_p2_distributions`）
- **Trial 复用**：将 P1 中落在 P2 区间内的 Trial 以 COMPLETE 状态注入 P2 study
- **独立 SQLite**：与 P1 独立存储，避免污染

##### (5) `app.py` — Gradio Web UI（5 Tab）

| Tab | 功能 |
|-----|------|
| **Tab1** 训练控制 | 选项目 → 启动 P1/P2 → 实时进度条 → 立即停止 / 优雅停止 |
| **Tab2** 结果分析与反推 | 显示 P1 Trial 统计、按因变量滑块过滤、反推 P2 搜索区间 |
| **Tab3** Top-5 排名 | 展示 P1 + P2 联合排序后的最优 5 个 Trial 详情 |
| **Tab4** 参数热力图 | 因子-IC 关系、参数-分数敏感度图 |
| **Tab5** 项目管理 | 创建/克隆/删除优化项目，查看历史记录 |

##### (6) `utils/` — 辅助工具

| 文件 | 职责 |
|------|------|
| `logger.py` | 统一日志门面 |
| `rolling_logger.py` | 滚动日志（按 Trial / 系统状态 / 错误 3 类） |
| `memory_monitor.py` | 内存监控（psutil） |
| `win_memory.py` | Windows 平台 `release_memory_to_os()`（强制回收） |
| `trial_callback.py` | Optuna 回调工厂 |
| `retroactive_normalize.py` | 事后归一化（用于反推 P2 区间） |

#### 2.4.3 优化流程（典型）

```
1. 创建项目（Tab5）
   → 自动生成 projects/{name}/p1.db
2. 选 active_params（30 维的子集，Tab2 决定哪些超参要搜索）
3. 设置 objective_weights（哪些因变量参与评分 + 权重）
4. 启动 Phase1（Tab1）
   → 50-200 个 TPE Trial
   → SQLite 持久化每个 Trial 的 30 维参数 + 37 维指标
5. 反推 P2 区间（Tab2）
   → 滑块过滤出"有效区间"
6. 启动 Phase2（Tab1）
   → 30-100 个 Trial，搜索范围收紧
7. 查看 Top-5（Tab3）
```

#### 2.4.4 输出

```
projects/{project_name}/
├── p1/
│   ├── p1.db               # SQLite，Optuna Study
│   ├── p1_config.json      # P1 配置（active_params / weights / param_ranges）
│   └── p1_summary.json     # P1 完成后写入
└── p2/
    ├── p2.db
    └── p2_config.json
```

---

## 三、入口程序速查

| 程序 | 命令 | 作用 |
|------|------|------|
| `run_m0_full.py` | `python run_m0_full.py` | 拉 Tushare 全量数据 → 4 套中性化方案 Parquet 落盘（断点续跑） |
| `m1_engine/run_m1.py` | `python m1_engine\run_m1.py` | M1 滚动切分 → 窗口 Parquet 落盘 |
| `m2_engine/run_m2.py` | `python m2_engine\run_m2.py` | M2 双引擎训练 → all_portfolios.parquet |
| `run_m2_m4.py` | `python run_m2_m4.py` | 一键跑 M2 + M4（CPU 模式） |
| `run_one_full.py` | `python run_one_full.py baseline 60` | 单策略全量 benchmark（strategy: baseline/A/B/C/D，n_windows: 60/186） |
| `m4_report/report_generator.py` | `python m4_report\report_generator.py` | M4 HTML 报告生成 |
| `m5_optimizer/启动M5优化器.bat` | 双击 | 启动 Gradio Web UI（http://127.0.0.1:7860） |
| `m5_optimizer/app.py` | `python m5_optimizer\app.py` | 同上（命令行等价） |

---

## 四、目录约定

```
10q-202604gpu/
├── config/                    # 配置（yaml + 线程常量）
│   ├── config.yaml            # 主配置（中性化方案、滚动参数、换手成本、9 项门槛）
│   └── concurrency_config.py  # 自适应线程/并发数
│
├── m0_database/               # M0 数据模块
│   ├── pipeline.py            # 顶层 pipeline
│   ├── data_fetcher.py        # Tushare 拉数
│   ├── stock_filter.py        # 股票池筛选（ST/退市/次新过滤）
│   ├── factor_calculator.py   # 因子计算
│   ├── neutralization.py      # 中性化（Rank-Z + 双重 OLS）
│   ├── format_validator.py    # 数据格式校验
│   ├── regenerator.py         # 增量重生
│   └── _preflight_check.py    # 启动前环境检查
│
├── m1_engine/                 # M1 滚动切分
│   ├── data_loader.py         # M0 数据加载（并行 parquet → float32 DataFrame）
│   ├── label_maker.py         # 截面 label_rank 生成
│   ├── rolling_splitter.py    # 时序滚动切分器（生成器）
│   └── run_m1.py              # 顶层入口
│
├── m2_engine/                 # M2 CPU 训练
│   ├── ensemble.py            # 集成 + IC 计算 + 18 项验证集绩效
│   ├── feature_store.py       # 特征工程管线
│   ├── lgbm_model.py          # LightGBM Ranker（CPU/OpenCL）
│   ├── xgb_model.py           # XGBoost Ranker（CPU/CUDA）
│   ├── portfolio_builder.py   # 持仓构建 + 换手成本
│   ├── preprocessor.py        # 数据预加载
│   ├── smart_preprocessor.py  # 高效预处理器
│   ├── lightweight_loader.py  # 超轻量级加载器（避免 OOM）
│   ├── gpu_detector.py        # GPU/CPU 自适应
│   └── run_m2.py              # 顶层入口
│
├── m2_engine_gpu/             # M2 GPU 加速训练（与 m2_engine 平行）
│   ├── ensemble.py            # GPU 策略 A/B/C/D/E
│   ├── feature_store.py
│   ├── lgbm_model.py          # GPU LightGBM
│   ├── xgb_model.py           # GPU XGBoost
│   ├── portfolio_builder.py
│   ├── gpu_detector.py
│   └── run_m2.py
│
├── m4_report/                 # M4 回测报告
│   ├── metrics.py             # 28+ 项绩效指标 + 9 项门槛
│   └── report_generator.py    # HTML 报告（Chart.js 净值曲线）
│
├── m5_optimizer/              # M5 贝叶斯优化器
│   ├── app.py                 # Gradio Web UI（5 Tab）
│   ├── search_space.py        # 30 维参数 + 37 维因变量 + NORM_CONFIG
│   ├── objective.py           # Optuna 目标函数
│   ├── phase1_global.py       # Phase1 全局探索
│   ├── phase2_local.py        # Phase2 局部精调
│   ├── config_manager.py      # M5 配置读写
│   ├── project_manager.py     # 项目管理
│   ├── result_analyzer.py     # P1 结果分析
│   ├── range_analyzer.py      # P2 区间反推
│   ├── utils/                 # 辅助工具
│   └── 启动M5优化器.bat       # 一键启动
│
├── run_*.py                   # 顶层入口
├── bench_*.py / analyze_*.py  # benchmark / 分析脚本
├── analyze_m5_p1.py           # P1 报告生成
├── p1_report_generator.py     # P1 HTML 报告
├── requirements.txt
├── README.md
└── .gitignore
```

> ⚠️ **不进入版本控制**：`data/`、`logs/`、`output/`、所有 `*.pkl / *.parquet / *.db` 缓存。
> 详见 [.gitignore](.gitignore)。

---

## 五、快速开始

### 5.1 环境要求

- **Python** 3.11
- **OS**：Windows 10/11（Linux/Mac 亦可，路径分隔符自动适配）
- **RAM**：16 GB+（CPU 模式 M2 全量约 12-14 GB）
- **GPU**（推荐）：NVIDIA 显卡，CUDA 12.x，4GB+ VRAM
  - 已验证：GTX 1650 4GB 串行训练无 OOM
- **CPU**：8 核最佳（实测 LGBM=2 线程 + XGB=3 线程为最优组合）

### 5.2 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5.3 配置 Tushare Token

**必须**通过环境变量设置（**不要**写进 yaml）：

```powershell
$env:TUSHARE_TOKEN = "你的tushare_pro_token"
```

去 https://tushare.pro 注册并在个人中心获取。**任何情况都不要把 token 写入 config.yaml**。

### 5.4 跑通流程

```powershell
# M0 拉数据 + 计算因子（首次需要数小时，缓存到 data/raw_cache/）
python run_m0_full.py

# M1 滚动切分
python m1_engine\run_m1.py

# M2 训练 + 组合构建
python m2_engine\run_m2.py

# M4 报告
python m4_report\report_generator.py

# M5 启动优化器 Web UI
.\m5_optimizer\启动M5优化器.bat
# 浏览器打开 http://127.0.0.1:7860
```

---

## 六、性能约束与基线

| 项 | 数值 |
|----|------|
| 滚动训练窗口 | 36 个月训练 / 12 个月验证 / 1 个月测试 |
| 默认步长 | 1 月 |
| 全量窗口数（2007-01 ~ 2025-12） | 228 - 49 + 1 = 180 个窗口 |
| 持仓容量 | Top10 永远满仓（65% + 35%） |
| CPU 模式内存峰值 | 12-14 GB |
| GPU 模式显存峰值 | ~3.5 GB（GTX 1650） |
| 8 核 CPU 单窗时延 | ~0.76s（benchmark_v42 实测） |
| GPU 加速比 | 1.5-2.5×（视数据规模） |

---

## 七、安全提示

- **Tushare token 严禁提交**！仓库已配置为从环境变量 `TUSHARE_TOKEN` 读取
- 历史 commit 已剔除明文 token
- 如发现 token 泄露，请立即在 https://tushare.pro 重置
- **任何 *db / *parquet / *pkl 都不进 git**，见 [.gitignore](.gitignore)

---

## 八、许可证

本项目采用 **Apache License 2.0** 开源许可证。

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

### 8.1 你可以做什么

| 行为 | 是否允许 |
|------|----------|
| 个人学习、研究、阅读源码 | ✅ |
| 在自有项目中 fork / 修改 | ✅ |
| 商用、部署到生产环境 | ✅ |
| 再分发（需保留版权与许可证声明） | ✅ |
| 申请专利 | ✅ |
| 闭源使用 | ✅ |
| 修改后以不同许可证发布 | ✅（须保留原作者版权） |

### 8.2 你必须遵守什么

- **保留版权声明**：在所有副本 / 衍生作品中保留 `Copyright 2026 zjwandcat`
- **标注修改**：若修改了源文件，必须明确标注"已修改"
- **包含 LICENSE 副本**：再分发时必须附带本 LICENSE 文件
- **NOTICE 文件**（如有修改）：必须保留 [NOTICE](NOTICE) 中的归属声明
- **专利授权终止条款**：若对任何 Contributor 发起专利诉讼，则该 Contributor 授予你的所有专利授权自动终止

### 8.3 第三方依赖

本项目依赖多个第三方开源库（pandas、numpy、lightgbm、xgboost、optuna、gradio、tushare 等），完整列表及许可证见 [NOTICE](NOTICE) 文件。这些依赖保留各自原始许可证，**不受本项目 Apache 2.0 约束**。

### 8.4 风险声明

本项目仅供量化研究学习用途，**不构成任何投资建议**。因使用本项目代码产生的任何投资损失，**作者不承担任何责任**。详见 LICENSE 第 7、8 条（无担保 / 责任限制）。

---

## 九、致谢

感谢 [Tushare Pro](https://tushare.pro) 提供高质量的 A 股数据接口，感谢 LightGBM、XGBoost、Optuna、Gradio 等开源社区。
