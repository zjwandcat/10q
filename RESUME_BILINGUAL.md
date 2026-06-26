# TTHH A 股量化选股系统 | A-Share Quantitative Stock Selection System

> 六层模块化滚动窗口选股 + 集成学习 + 贝叶斯超参优化平台，16GB 笔记本流畅运行
>
> A six-layer modular rolling-window stock selection + ensemble learning + Bayesian hyperparameter optimization platform, running smoothly on 16GB laptops.

---

## 项目亮点 | Project Highlights

### 中文

- **六层流水线**：M0 数据库 → M1 滚动切分 → M2 双引擎训练 → M3 风控 → M4 报告 → M5 贝叶斯优化
- **双模型集成**：LightGBM + XGBoost 软投票，自适应权重
- **BO-TPE 优化**：Optuna 树结构 Parzen 估计器，30 维超参 × 37 维目标空间
- **16GB 笔记本优化**：七层内存防御体系，低内存机器跑通 180 窗口全量回测
- **28+ 绩效指标**：9 项门槛检查 + 三大归因（Brinson / 五因子 / Barra）
- **Python 3.14**：match/case 结构化模式匹配、`@dataclass(slots=True)`、完整类型注解、生成器惰性求值

### English

- **Six-Layer Pipeline**: M0 Database → M1 Rolling Split → M2 Dual-Engine Training → M3 Risk Control → M4 Reporting → M5 Bayesian Optimization
- **Dual-Model Ensemble**: LightGBM + XGBoost soft voting with adaptive weighting
- **BO-TPE Optimization**: Optuna Tree-structured Parzen Estimator for 30-D hyperparameter × 37-D objective space
- **16GB Laptop Optimized**: Seven-layer memory defense system enables full 180-window backtest on low-memory machines
- **28+ Performance Metrics**: 9 threshold checks + three attribution methods (Brinson / Five-Factor / Barra)
- **Python 3.14**: match/case structured pattern matching, `@dataclass(slots=True)`, comprehensive type annotations, generator-based lazy evaluation

---

## 核心技术 | Core Technologies

| 类别 / Category | 技术 / Technologies |
|----------------|---------------------|
| **ML 模型 / Models** | LightGBM (MAE, OpenCL GPU), XGBoost (MAE, CUDA GPU) |
| **贝叶斯优化 / Bayesian Opt** | Optuna TPE + 约束函数 + 双阶段策略（全局探索 + 局部精调）|
| **数据处理 / Data Processing** | numpy 向量化, pandas, pyarrow Parquet, polars |
| **Web UI** | Gradio 5-Tab 界面（训练控制 / 结果分析 / 参数热力图）|
| **内存工程 / Memory Eng** | float32 压缩, 生成器惰性加载, 显式 GC, Windows 内存归还 API |
| **风控 / Risk Control** | TET 状态机（趋势得分 / 情绪指数 / 锚定趋势 / 择时）|

---

## 技术成就 | Technical Achievements

### 16GB 笔记本内存优化 | Memory Optimization for 16GB Laptops

**中文**
- **七层防御**：数据类型压缩（float64→float32 省 50%）、惰性加载、增量追加、显式释放、OS 级归还、OOM 哨兵、串行策略
- **实测数据**：CPU 模式峰值 10-12GB，GPU 模式显存 ~3.5GB，连续 50 个 M5 Trial 内存稳定
- **Windows 专用**：`SetProcessWorkingSetSize` + `msvcrt._heapmin()` 强制归还内存给操作系统

**English**
- **Seven-Layer Defense**: Data type compression (float64→float32 saves 50%), lazy loading, incremental append, explicit release, OS-level return, OOM sentinel, serial strategy
- **Measured Results**: CPU mode peak 10-12GB, GPU mode VRAM ~3.5GB, stable over 50 continuous M5 Trials
- **Windows-Specific**: `SetProcessWorkingSetSize` + `msvcrt._heapmin()` for forced memory return to OS

### 向量化特征工程 | Vectorized Feature Engineering

**中文**
- **7 步管线**：候选因子 → 噪音过滤 → 覆盖率过滤 → 零方差过滤 → 截面 Z-score → 高相关去重 → IC 筛选
- **纯 numpy 实现**：argsort+unique+slice 比逐日布尔 mask 快 12 倍
- **FeatureStore 缓存**：M5 多 Trial 场景复用 `_fit_state`，节省 50% 特征工程时间

**English**
- **7-Step Pipeline**: Candidate factors → noise filter → coverage filter → zero-variance filter → cross-sectional Z-score → correlation dedup → IC screening
- **Pure numpy Implementation**: 12x faster IC computation via argsort+unique+slice vs per-date boolean mask
- **FeatureStore Cache**: Reuses `_fit_state` in M5 multi-Trial scenarios, saves 50% feature engineering time

### 贝叶斯超参优化 | Bayesian Hyperparameter Optimization

**中文**
- **30 维搜索空间**：LGBM (12 维), XGB (11 维), Ensemble (1 维), Feature (5 维), Window (1 维)
- **37 维目标空间**：M2 验证集 (17 项), M2 压力测试 (3 项), M4 回测 (17 项)
- **分数归一化**：三种方法（linear / tanh / signed_log）适配不同指标分布
- **安全约束**：lr × n_estimators ≤ 15.0，Trial 超时 30 分钟，IC_GAP 惩罚 × 1.5

**English**
- **30-D Search Space**: LGBM (12-D), XGB (11-D), Ensemble (1-D), Feature (5-D), Window (1-D)
- **37-D Objective Space**: M2 validation (17 items), M2 stress test (3 items), M4 backtest (17 items)
- **Score Normalization**: Three methods (linear / tanh / signed_log) for different metric distributions
- **Safety Constraints**: lr × n_estimators ≤ 15.0, Trial timeout 30min, IC_GAP penalty × 1.5

### 集成学习架构 | Ensemble Learning Architecture

**中文**
- **软投票**：`score = w_lgbm × pred_lgbm + w_xgb × pred_xgb`（w_lgbm ∈ [0.3, 0.7]）
- **置信度标记**：`ic_gap = train_IC - val_IC > 0.15` → 标记（不影响仓位）
- **MAE 目标**：比 lambdarank 更鲁棒，避免原生内存崩溃

**English**
- **Soft Voting**: `score = w_lgbm × pred_lgbm + w_xgb × pred_xgb` (w_lgbm ∈ [0.3, 0.7])
- **Confidence Flag**: `ic_gap = train_IC - val_IC > 0.15` → flagged (no position impact)
- **MAE Objective**: More robust to outliers than lambdarank, avoids native memory crashes

---

## Python 工程实践 | Python Engineering Practices

**中文**
- **现代 Python 3.14**：match/case 模式匹配、不可变类型优化（frozenset/MappingProxyType）、dataclasses with slots
- **生成器模式**：M1 RollingSplitter 惰性 yield 窗口，防止内存爆炸
- **类型注解**：全项目使用 typing 模块（Dict, List, Optional, Tuple, Any）
- **防御性编程**：参数浅拷贝、三级 OOM 防御、多级 Fallback
- **线程安全**：Lock 保护的特征缓存、del + gc.collect() 显式打破循环引用

**English**
- **Modern Python 3.14**: match/case pattern matching, immutable type optimization (frozenset/MappingProxyType), dataclasses with slots
- **Generator Pattern**: M1 RollingSplitter yields windows lazily, prevents memory explosion
- **Type Annotations**: Full project uses typing module (Dict, List, Optional, Tuple, Any)
- **Defensive Programming**: Parameter shallow copy, three-level OOM defense, multi-level fallback
- **Thread Safety**: Lock-protected feature cache, explicit cycle breaking with del + gc.collect()

---

## 性能指标 | Performance Metrics

| 指标 / Metric | 数值 / Value |
|--------------|-------------|
| 滚动窗口 / Rolling Windows | ~180（36 月训练 / 12 月验证 / 1 月测试）|
| 持仓容量 / Portfolio Capacity | Top10 永远满仓（65% High + 35% Low）|
| CPU 内存峰值 / CPU Memory Peak | 10-12 GB（16GB 笔记本友好）|
| GPU 显存峰值 / GPU VRAM Peak | ~3.5 GB（GTX 1650 4GB，无 OOM）|
| 单窗口时延 / Single Window Latency | ~0.76s（8 核 CPU）|
| GPU 加速比 / GPU Speedup | 1.5-2.5× |
| M5 单 Trial / M5 Single Trial | ~5-8 分钟（60 窗口，fast_mode）|
| M5 内存稳定性 / M5 Memory Stability | 连续 50 Trial 无增长 |

---

## 核心模块 | Key Modules

| 模块 / Module | 职责 / Responsibility |
|--------------|----------------------|
| **M0 数据库 / Database** | Tushare 拉数、股票池筛选、因子计算、行业/市值中性化（Rank-Z + 双重 OLS）|
| **M1 滚动切分 / Rolling Split** | 时序滚动 train/val/pred 切分、截面 Label 生成、生成器模式省内存 |
| **M2 训练 / Training** | 7 步特征工程、LightGBM/XGBoost 训练、软投票集成、Top10 持仓、SHAP 归因 |
| **M3 风控 / Risk Control** | TET 状态机（4 指标：TS/EI/ATS/Timing）、Schmitt Trigger 锚定 |
| **M4 报告 / Reporting** | 28+ 绩效指标、9 项门槛检查、Brinson/五因子/Barra 归因、iOS 26 风格 HTML 报告 |
| **M5 优化器 / Optimizer** | Optuna TPE 双阶段贝叶斯优化、Gradio Web UI、项目管理、30 维参数 × 37 维目标 |

---

## 项目结构 | Project Structure

```
10q-202604gpu/
├── m0_database/          # 数据拉取、因子计算、中性化 / Data fetching, factor calculation, neutralization
├── m1_engine/            # 滚动窗口切分、标签生成 / Rolling window split, label generation
├── m2_engine/            # CPU 训练（LightGBM + XGBoost 集成）/ CPU training (ensemble)
├── m2_engine_gpu/        # GPU 加速训练（OpenCL + CUDA）/ GPU accelerated training
├── m3_engine/            # TET 风控状态机 / TET risk control state machine
├── m4_report/            # 绩效指标、归因、HTML 报告 / Metrics, attribution, HTML report
├── m5_optimizer/         # 贝叶斯超参优化、Gradio UI / Bayesian optimization, Gradio UI
├── config/               # 配置（YAML + 并发）/ Configuration (YAML + concurrency)
└── tests/                # 自动化测试 / Automated tests
```

---

## 运行环境 | Environment

- **OS**: Windows 11 (22H2 / 23H2 / 24H2)
- **Python**: 3.14.x
- **RAM**: ≥ 16 GB（针对 16GB 笔记本优化 / optimized for 16GB laptops）
- **GPU**（可选 / optional）: NVIDIA GTX 1650+ with CUDA 12.x
- **CPU**: 8 核最佳 / 8 cores optimal（LGBM=2 线程 / threads + XGB=3 线程 / threads）

---

## 许可证 | License

Apache License 2.0 — Copyright © 2026 zjwandcat

完整项目详情 / Full project details: [README.md](README.md) (中文 / Chinese) | [README_EN.md](README_EN.md) (English)
