# TTHH A 股量化选股系统 (m012345)

> 由 [zjwandcat](https://github.com/zjwandcat) 维护
>
> Tushare 数据 + 六层流水线 + LightGBM/XGBoost 集成学习 + Optuna BO-TPE 贝叶斯超参优化
>
> **16 GB 笔记本流畅运行** — 极致内存工程，小内存也能跑完全量回测

🌐 **语言 / Language**: [🇨🇳 中文 (当前)](README.md) · [🇬🇧 English](README_EN.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078d4.svg)](https://www.microsoft.com/windows/windows-11)
[![Version](https://img.shields.io/badge/Version-4.2-green.svg)](CHANGELOG.md)

**Copyright © 2026 zjwandcat. Licensed under the [Apache License, Version 2.0](LICENSE).**

---

## 目录

- [一、系统总览](#一系统总览)
- [二、系统规模与领域全景](#二系统规模与领域全景)
- [三、小内存笔记本流畅运行之道](#三小内存笔记本流畅运行之道)
- [四、核心算法引擎：LightGBM + XGBoost + BO-TPE](#四核心算法引擎lightgbm--xgboost--bo-tpe)
- [五、Python 工程能力展示](#五python-工程能力展示)
- [六、模块技术细节](#六模块技术细节)
  - [6.1 M0 数据库](#61-m0-数据库)
  - [6.2 M1 滚动切分引擎](#62-m1-滚动切分引擎)
  - [6.3 M2 双引擎训练与集成](#63-m2-双引擎训练与集成)
  - [6.4 M3 TET 风控外挂](#64-m3-tet-风控外挂)
  - [6.5 M4 报告与归因](#65-m4-报告与归因)
  - [6.6 M5 贝叶斯超参优化](#66-m5-贝叶斯超参优化)
- [七、入口程序速查](#七入口程序速查)
- [八、目录约定](#八目录约定)
- [九、快速开始](#九快速开始)
- [十、性能基线](#十性能基线)
- [十一、安全提示](#十一安全提示)
- [十二、许可证](#十二许可证)
- [十三、致谢](#十三致谢)

---

## 一、系统总览

本系统是一套面向 A 股市场的 **六层模块化滚动窗口选股 + 集成学习 + 贝叶斯超参优化** 闭环流水线：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Tushare Pro API                                                         │
│        │                                                                 │
│        ▼                                                                 │
│  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  ┌─────┐│
│  │   M0   │─▶│   M1   │─▶│ M2 / GPU │─▶│   M3   │─▶│   M4   │─▶│ M5  ││
│  │ 数据库 │  │ 切分   │  │训练/集成 │  │ 风控   │  │ 回测   │  │优化 ││
│  └────────┘  └────────┘  └──────────┘  └────────┘  └────────┘  └─────┘│
│  Parquet     滚动窗口    LGBM+XGB      TET状态机   28+项绩效   Optuna  │
│  落盘       +标签生成    软投票集成    趋势/情绪    HTML报告    BO-TPE  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

| 模块 | 路径 | 核心职责 |
|------|------|----------|
| **M0** 数据 | `m0_database/` | Tushare 拉数、股票池筛选、**23 大类 460 列因子**计算、**8 套中性化方案**（Rank-Z + 双重 OLS / WLS / PCA / 风格因子剥离）、Parquet 落盘 |
| **M1** 滚动切分 | `m1_engine/` | 时序滚动 train/val/pred 切分、截面 Label 生成、生成器模式省内存 |
| **M2** 模型训练 | `m2_engine/`, `m2_engine_gpu/` | 特征工程 7 步管线、LightGBM/XGBoost 训练、软投票集成、Top10 持仓、SHAP 归因 |
| **M3** 风控 | `m3_engine/` | TET 风控外挂（Trend-Score / Emotion-Index / Anchored-Trend / Timing 状态机） |
| **M4** 报告 | `m4_report/` | 28+ 项绩效指标、9 项门槛检查、Brinson/五因子/Barra 三大归因、iOS 26 风格 HTML 报告 |
| **M5** 优化器 | `m5_optimizer/` | Optuna TPE 双阶段贝叶斯优化（**30 维参数 × 37 维目标**）、Gradio Web UI、项目管理 |

---

## 二、系统规模与领域全景

> m012345 不是单一脚本的回测玩具，而是一个横跨 **金融工程、机器学习、高性能计算、操作系统内核、Web 工程、数据工程** 六大技术领域的工业级闭环系统。

### 2.1 数字概览

| 维度 | 规模 |
|------|------|
| 核心模块 | **6 层** (M0→M1→M2→M3→M4→M5) |
| Python 源文件 | **50+** |
| 因子体系 | **23 大类 / 460 列**（动量 51 + 技术 61 + 宏观 68 + 补充 70 + …） |
| 中性化方案 | **8 套**（A/B/B1/B2/D/E/F/G，含非线性市值 OLS、WLS、PCA、风格因子剥离） |
| 搜索空间 | **30 维参数 × 37 维目标**（67 维联合空间，Optuna TPE 双阶段优化） |
| 绩效指标 | **28+ 项**（CAGR → Omega → Burke → Martin → Pain → Ulcer → Tail → CVaR → …） |
| 门槛检查 | **9 项**（IR≥0.50 / Calmar≥1.00 / MaxDD≥-35% / Sortino≥1.20 / …） |
| 归因体系 | **3 大类**（Brinson 行业拆解 / Fama-French 五因子+Carhart / Barra 10 因子暴露） |
| 回测窗口 | **~180 个**（2007-01 ~ 2025-12，滚动 43+12+1 月） |
| 内存防御 | **7 层**（类型压缩 → 惰性加载 → 增量追加 → 显式释放 → OS 归还 → OOM 哨兵 → 串行策略） |
| 数据池 | **8 套 Parquet**（每套中性化方案独立数据池） |
| 测试覆盖 | **9 项自动化**（v4.2 修复验证，集成 pre-commit hook） |

### 2.2 涉及技术领域

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  金融工程                    机器学习                  高性能计算            │
│  ├── A 股滚动窗口选股        ├── LightGBM Ranker       ├── GPU 策略 D       │
│  ├── 460 列因子工程          ├── XGBoost Ranker        │   (CPU train +     │
│  ├── 8 套中性化方案          ├── MAE 目标函数          │    GPU predict)    │
│  ├── Top10 分层持仓          ├── 软投票集成            ├── OpenCL / CUDA    │
│  ├── Brinson 归因            ├── SHAP 归因             ├── 纯 numpy 向量化  │
│  ├── Fama-French 五因子      ├── Optuna TPE            ├── cupy 矩阵乘     │
│  ├── Barra 10 因子暴露       ├── 30×37 贝叶斯优化      ├── polars 流式      │
│  ├── 换手成本建模            ├── early stopping        ├── bottleneck 加速  │
│  └── 9 项门槛检查            └── 学习率衰减/调度       └── 生成器惰性求值   │
│                                                                             │
│  操作系统内核                Web 工程                  数据工程              │
│  ├── Windows 内存强制归还    ├── Gradio 5-Tab UI       ├── Tushare Pro API  │
│  │   (SetProcessWorking      ├── iOS 26 风格 HTML      ├── akshare 备用源   │
│  │    SetSize + _heapmin)    ├── Chart.js 离线内嵌     ├── Parquet 列式存储 │
│  ├── RSS 泄漏三条件检测      ├── SVG 服务端兜底        ├── 断点续跑 pipeline │
│  ├── OOM 三级防御哨兵        ├── Puppeteer 渲染验证    ├── 增量重生          │
│  ├── CRT 堆整理归还          ├── JSDOM DOM 解析        ├── 7 步特征管线     │
│  └── GPU 显存监控            └── SQLite 持久化         └── 截面 Z-score     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 因子体系全图（23 大类 / 460 列）

| 类别 | 前缀 | 列数 | 典型因子 |
|------|------|------|----------|
| 动量 | `momentum_` | 51 | 1M/3M/6M/12M 反转、行业动量、加速度 |
| 反转 | `reversal_` | 2 | 短期反转因子 |
| 波动率 | `volatility_` | 26 | 20D/60D/120D 滚动标准差、Parkinson/Garman-Klass 估计 |
| 质量 | `quality_` | 14 | ROE/ROA/毛利率稳定性、应计利润 |
| 估值 | `value_` | 9 | EP/BP/SP/CFP、相对行业估值 |
| 成长 | `growth_` | 6 | 营收/利润同比增长、PEG |
| 技术 | `technical_` | 61 | MACD/RSI/KDJ/布林带/OBV/ATR/CCI/DMI/… |
| 流动性 | `liquidity_` | 16 | 换手率/Amihud 非流动性/买卖价差 |
| 规模 | `size_` | 5 | 总市值/流通市值/对数市值 |
| 股息 | `dividend_` | 5 | 股息率/分红支付比 |
| Alpha101 | `alpha101_` | 50 | WorldQuant Alpha#1~#101 精选 |
| Barra 风险 | `barra_` | 10 | 市值/贝塔/动量/残差波动/非线性市值/… |
| AQR 风格 | `aqr_` | 6 | 价值/动量/盈利/投资/安全/质量 |
| 聚宽 JQ | `jq_` | 20 | 聚宽因子库精选 |
| 复合衍生 | `deriv_` | 8 | 多因子交叉衍生 |
| 滚动统计 | `roll_` | 12 | 滚动均值/偏度/峰度/分位数 |
| 行业相对 | `industry_relative_` | 6 | 行业内排名分位数 |
| 情绪 | `sentiment_` | 5 | 涨跌停/换手异常/资金流 |
| FF 五因子 | `ff_` | 5 | MKT/SMB/HML/RMW/CMA 代理 |
| 盈利质量 NM | `nm_` | 4 | Novy-Marx 盈利质量 |
| 流动性冲击 PS | `ps_` | 3 | Pastor-Stambaugh 流动性 |
| 宏观 | `macro_` | 68 | CPI/PPI/M1/M2/PMI/利率/汇率/… |
| 补充 | `supp_` | 70 | 交叉验证补充因子 |

### 2.4 中性化方案矩阵（8 套）

| 方案 | 核心方法 | 特殊处理 | 豁免列 |
|------|----------|----------|--------|
| `scheme_a` | 双重 OLS 正交化（行业+市值，机构标准版） | — | `macro_*`, `industry_relative_*` |
| `scheme_b` | Rank-Z + 双重 OLS（**推荐**，抗极端值） | 先 Rank→Z-score 消除量级差异 | 同上 |
| `scheme_b1` | 非线性市值 OLS | B 基础 + ln(M)² + ln(M)³，先 demean 防御高阶共线性 | 同上 |
| `scheme_b2` | WLS 加权回归 | sqrt(mktcap) 对角权重，缺 mktcap 退化为等权 OLS | 同上 |
| `scheme_d` | 仅行业 OLS（对照组） | 豁免 size_ 前缀，保留市值因子暴露 | + `size_*` |
| `scheme_e` | 分层中性化 | 行业内 Rank + 全截面市值 OLS | 同 scheme_a |
| `scheme_f` | 风格因子剥离 | B 基础 + 动量/换手/波动率等 Barra 风格，缺列 raise | 同 scheme_a |
| `scheme_g` | PCA 隐式风险 | 60 天 log-return 矩阵 NumPy SVD，缺矩阵 raise | 同 scheme_a |

---

## 三、小内存笔记本流畅运行之道

> **核心理念**：不靠堆硬件解决问题，靠工程优化让 16 GB 笔记本跑完全量 180 窗口回测。

### 3.1 内存优化全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                    内存优化七层防御体系                            │
├─────────┬───────────────────────────────────────────────────────┤
│ 第1层   │ 数据类型压缩                                           │
│         │ float64→float32 (省50%)  string→category (省40-60%)   │
├─────────┼───────────────────────────────────────────────────────┤
│ 第2层   │ 惰性加载                                               │
│         │ 生成器yield逐窗口  不预展开  按需读取Parquet           │
├─────────┼───────────────────────────────────────────────────────┤
│ 第3层   │ 增量追加                                               │
│         │ 逐文件read→即时优化→append  避免pd.concat内存峰值      │
├─────────┼───────────────────────────────────────────────────────┤
│ 第4层   │ 显式释放                                               │
│         │ del+gc.collect()打破循环引用  每窗口结束强制回收        │
├─────────┼───────────────────────────────────────────────────────┤
│ 第5层   │ OS级归还                                               │
│         │ SetProcessWorkingSetSize + msvcrt._heapmin()          │
│         │ 每个Trial结束后主动归还内存给Windows                   │
├─────────┼───────────────────────────────────────────────────────┤
│ 第6层   │ OOM哨兵                                                │
│         │ 每10窗口检查系统可用内存 <0.8GB主动中止防Windows强杀   │
│         │ RSS增长60%+2GB+可用<1.5GB 三条件判定内存泄漏          │
├─────────┼───────────────────────────────────────────────────────┤
│ 第7层   │ 串行策略                                               │
│         │ 窗口串行(非并行)  模型串行(非并行)  牺牲速度换稳定性   │
└─────────┴───────────────────────────────────────────────────────┘
```

### 3.2 关键优化技术详解

#### (1) 超轻量级数据加载器 (`lightweight_loader.py`)

```python
# 传统方式：一次pd.concat所有文件 → 内存峰值翻倍
# factor_df = pd.concat([pd.read_parquet(f) for f in files])

# 本系统：逐文件读取 → 即时类型优化 → 增量追加
for file_path in files:
    df = pd.read_parquet(file_path)
    # 即时压缩：float64→float32, int64→int32, 删除冗余列
    df = _optimize_dataframe(df)
    factor_df = pd.concat([factor_df, df], ignore_index=True)
    # 每20个文件gc.collect()释放中间变量
```

#### (2) Windows 内存强制归还 (`win_memory.py`)

```python
def release_memory_to_os():
    gc.collect(2)                              # 全代GC（含老年代）
    kernel32.SetProcessWorkingSetSize(          # 强制OS回收未使用内存页
        kernel32.GetCurrentProcess(), -1, -1)
    msvcrt._heapmin()                          # 强制CRT堆整理归还Windows
    # 效果：配合gc.collect(2)通常能再降200-500MB RSS
```

#### (3) 自适应并发配置 (`concurrency_config.py`)

```python
# 低内存高CPU模式 — 为16GB笔记本量身定制
GLOBAL_N_JOBS_OUTER   = 1    # 窗口串行（不并行）
GLOBAL_NTHREAD_INNER  = 4    # 单模型4线程
M5_NTHREAD_PER_MODEL  = 4    # M5每模型4线程
MEMORY_LIMIT_GB       = 6.0  # 内存红线6GB（留余量给系统）
# 理论总线程：1×2×4 = 8 / 逻辑核
```

#### (4) 训练内存防御链 (`objective.py`)

```python
# 三层内存守卫
_OOM_RSS_LIMIT_GB    = 12.0   # 进程RSS超12GB → 强制GC+归还
_OOM_AVAIL_LIMIT_GB  = 0.5    # 系统可用<0.5GB → 跳过Trial
_OOM_VRAM_LIMIT_MB   = 7000   # GPU显存超7GB → 警告

# Trial超时守卫：30分钟硬限，防止单Trial挂死拖垮进程
_TRIAL_TIMEOUT_SEC = 1800
```

#### (5) FeatureStore 缓存复用

```python
# M5多Trial场景：相同feature_params时复用_fit_state
# transform()跳过corr/IC筛选，直接复用fit时的列选择结果
# 效果：省50%因子工程时间
```

### 3.3 实测内存数据

| 场景 | 内存峰值 | 说明 |
|------|----------|------|
| M2 CPU 全量 180 窗口 | 10-12 GB | 串行窗口 + 即时释放 |
| M5 单 Trial (60 窗口) | 8-10 GB | fast_mode + 缓存 |
| M5 连续 50 Trial | 稳定 10-12 GB | 每 Trial 后归还内存 |
| GPU 模式显存 | ~3.5 GB | GTX 1650 4GB 无 OOM |

---

## 四、核心算法引擎：LightGBM + XGBoost + BO-TPE

### 4.1 双模型集成学习架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     特征工程 7 步管线 (FeatureStore)                  │
│                                                                     │
│  候选因子 → 短期噪音过滤 → 低覆盖过滤 → 零方差过滤                  │
│           → 截面Z-score → 高相关去重 → IC筛选                       │
│           (纯numpy向量化, 无pandas groupby开销)                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐     ┌──────────────────────┐             │
│  │   LightGBM Ranker    │     │   XGBoost Ranker     │             │
│  │                      │     │                      │             │
│  │  objective: MAE      │     │  objective: MAE      │             │
│  │  device: OpenCL/CPU  │     │  device: CUDA/CPU    │             │
│  │  自适应叶子数         │     │  自适应深度           │             │
│  │  学习率衰减回调       │     │  学习率调度器         │             │
│  │  early_stopping=30   │     │  early_stopping=30   │             │
│  └──────────┬───────────┘     └──────────┬───────────┘             │
│             │                             │                         │
│             └──────────┬──────────────────┘                         │
│                        ▼                                            │
│              软投票集成 (Soft Voting)                                │
│         score = w_lgbm × pred_lgbm + w_xgb × pred_xgb              │
│              (w_lgbm ∈ [0.3, 0.7], 由M5优化)                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  置信度检验: ic_gap = train_IC - val_IC                             │
│  ic_gap > 0.15 → 标记LOW (不影响仓位, 仅报告统计)                  │
│                                                                     │
│  Top20 → High(1-5)×13% + Low(6-10)×7% = 100%满仓                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 LightGBM 技术细节

| 维度 | 实现 |
|------|------|
| **目标函数** | `regression_l1` (MAE) — 对异常值鲁棒，rank 相关性更强 |
| **GPU 加速** | `device_type="gpu"` (OpenCL)，GTX 1650 实测 ~1.5GB VRAM |
| **自适应叶子** | `num_leaves = min(2^max_depth, train_size/200, 255)` |
| **学习率衰减** | 回调 `_make_lr_decay_callback`：每 decay_every 轮 × decay_factor |
| **深度模式** | `fixed` / `adaptive`（根据训练集大小自动调整） |
| **早停回滚** | `predict(num_iteration=best_iteration)` |
| **CPU 线程** | 2 线程（8 核笔记本最优） |
| **防御** | `dict()` 浅拷贝 params，禁止原地修改；显式 `del Dataset; gc.collect()` 打破循环引用 |

**为什么用 MAE 而非 lambdarank？**
- MAE 对异常值更鲁棒，rank 相关性更强
- 避免 lambdarank 在多线程 / 多 Trial 迭代中的 native 内存崩溃

### 4.3 XGBoost 技术细节

| 维度 | 实现 |
|------|------|
| **目标函数** | `reg:absoluteerror` (MAE) — 与 LGBM 对齐 |
| **GPU 加速** | `device="cuda"` (XGBoost 2.0+)，`tree_method="hist"` |
| **自定义评测** | `eval_ic` — 直接优化 Spearman IC |
| **学习率调度** | `_make_xgb_lr_schedule`：支持 fixed / decay 两种模式 |
| **早停回滚** | `predict(iteration_range=(0, best_iteration))` |
| **CPU 线程** | 3 线程（8 核笔记本最优） |
| **gamma** | 最小损失分裂增益 (0.001-0.2)，防止过拟合 |
| **防御** | 同 LGBM，`dict()` 浅拷贝 + 显式 `del DMatrix; gc.collect()` |

### 4.4 集成策略

```python
# 软投票集成 — 加权平均
score = lgbm_weight × lgbm_pred + (1 - lgbm_weight) × xgb_pred

# 置信度标记
ic_gap = train_IC - val_IC
is_penalized = (ic_gap > 0.15)  # 仅标记, 不影响仓位

# 预测置信度变异系数
score_cv = std([lgbm_pred, xgb_pred]) / (|score| + 1e-6)
```

**GPU 串行 / CPU 串行**：因 GTX 1650 4GB 限制，LGBM 与 XGB 必须串行训练（不并行），CPU 模式同样串行（避免 native 内存冲突）。

### 4.5 Optuna BO-TPE 贝叶斯优化引擎

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Optuna TPE (Tree-structured Parzen Estimator)     │
│                                                                     │
│  ┌─────────────────────┐         ┌─────────────────────┐           │
│  │  Phase1: 全局探索    │         │  Phase2: 局部精调    │           │
│  │                     │         │                     │           │
│  │  TPESampler         │         │  从P1最优出发        │           │
│  │  n_startup=15       │────▶────│  搜索范围收紧        │           │
│  │  n_ei_candidates=24 │         │  Trial复用(P1注入P2) │           │
│  │  约束函数           │         │  独立SQLite存储      │           │
│  │  50-200 Trial       │         │  30-100 Trial        │           │
│  └─────────────────────┘         └─────────────────────┘           │
│                                                                     │
│  30维搜索空间:                                                      │
│  ├── LGBM 12维 (lr/n_est/depth/colsample/regα/regλ/...)           │
│  ├── XGB  11维 (lr/n_est/depth/colsample/regα/regλ/gamma/...)     │
│  ├── Ensemble 1维 (lgbm_weight)                                    │
│  ├── Feature 5维 (min_valid_rate/max_corr/min_ic/keep/drop)        │
│  └── Window 1维 (train_months)                                     │
│                                                                     │
│  37维目标空间:                                                      │
│  ├── M2验证集 17项 (IC/ICIR/IR/Sortino/Jensenα/...)               │
│  ├── M2压力测试 3项 (2008/2015/2022, 可选)                         │
│  └── M4回测 17项 (CAGR/胜率/VaR/CVaR/Omega/Burke/...)             │
│                                                                     │
│  评分归一化 (NORM_CONFIG):                                          │
│  ├── linear  → 率指标, 截断映射到[-0.5, 0.5]                       │
│  ├── tanh    → IR/Sortino类, 压缩到(-1, 1)                         │
│  └── signed_log → 长尾分布, sign(x)×log(1+|x|)                     │
│                                                                     │
│  安全约束:                                                          │
│  ├── lr × n_estimators ≤ 15.0 (防过拟合+超时)                      │
│  ├── n_est ≤ 500, depth ≤ 8, lr ≥ 0.005 (硬上限)                  │
│  ├── Trial超时 30分钟硬限                                           │
│  └── IC_GAP惩罚 × 1.5 (过拟合惩罚放大)                             │
└─────────────────────────────────────────────────────────────────────┘
```

**TPE 核心原理**：

TPE 用两个核密度估计器分别建模"好参数"和"坏参数"的概率分布，然后选择使 `l(x)/g(x)` 最小的点作为下一个采样点。相比网格搜索和随机搜索，TPE 能更高效地探索高维超参空间。

```
EI(x) = ∫ max(f* - f(x), 0) p(f(x)|x) df
```

其中 `f*` 是当前最优值，`p(f(x)|x)` 由 TPE 的两个 KDE 模型给出。

---

## 五、Python 工程能力展示

### 5.1 Python 版本与特性

| 特性 | 使用情况 |
|------|----------|
| **Python 3.14** | 全项目使用，PEP 745 free-threading 可选启用 |
| **match/case** | `search_space.py` / `objective.py` 中大量使用结构化模式匹配 |
| **不可变类型优化** | `frozenset` / `MappingProxyType` 替代可变 set/dict，避免 GIL refcount 开销 |
| **dataclasses** | M3 TET 引擎配置类 `M3Config` 使用 `@dataclass(slots=True)` |
| **类型注解** | 全项目使用 `typing` 模块（`Dict`, `List`, `Optional`, `Tuple`, `Any`） |
| **生成器** | M1 `RollingSplitter.split()` yield 逐窗口返回，惰性求值 |
| **`__slots__`** | `EnsemblePredictor` 使用 `__slots__` 减少实例内存开销 |

### 5.2 核心技术栈

```
数据与科学计算:    numpy ≥ 1.26  |  pandas ≥ 2.0  |  scipy ≥ 1.10  |  pyarrow ≥ 14.0  |  polars
机器学习模型:      lightgbm ≥ 4.1  |  xgboost ≥ 2.0  |  shap ≥ 0.42
贝叶斯优化:        optuna ≥ 3.4  |  joblib ≥ 1.3
数据源:            tushare ≥ 1.4  |  akshare ≥ 1.12
Web UI:            gradio ≥ 4.0, < 5.0
监控:              psutil ≥ 5.9
性能加速:          bottleneck (nanmean/nanstd等, 可选)  |  cupy (GPU矩阵乘, 可选)
```

### 5.3 向量化编程实践

```python
# 1. 纯numpy截面Z-score — 无pandas groupby开销
def _zscore_arr(arr, dates):
    """向量化截面Z-score: 1次argsort + 1次unique + 分组广播"""
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

# 2. 向量化Pearson相关矩阵 — 1D广播替代2D广播
def _corrcoef_f32(X):
    X_c = X - X.mean(axis=0)
    cov = X_c.T @ X_c
    diag_sqrt = np.sqrt(np.diag(cov) + 1e-12)
    # 1D广播比diag[:,None]*diag[None,:]快2-3x
    return cov / (diag_sqrt[:, None] * diag_sqrt[None, :] + 1e-12)

# 3. 向量化IC计算 — argsort+unique+slice替代per-date布尔mask
def _per_group_rank_corr(dates, pred, label):
    order = np.argsort(dates, kind='stable')
    sorted_dates = dates[order]
    _, idx_start = np.unique(sorted_dates, return_index=True)
    idx_end = np.append(idx_start[1:], len(dates))
    return [_fast_rank_corr(pred[order[s:e]], label[order[s:e]])
            for s, e in zip(idx_start, idx_end)]
    # 性能: ~12ms/调用 → ~1ms/调用 (180窗全量省~2s)

# 4. sliding_window_view替代Python循环
from numpy.lib.stride_tricks import sliding_window_view
win_ex = sliding_window_view(excess, 6)   # (n-5, 6) 矩阵
win_ex_mean = win_ex.mean(axis=1)          # 全程向量化
```

### 5.4 内存管理工程

```python
# 1. 数据类型压缩
df[float64_cols] = df[float64_cols].astype(np.float32)  # 省50%
df["stock_code"] = df["stock_code"].astype("category")   # 省40-60%

# 2. 浅拷贝替代深拷贝
pred_out = pred_df.copy(deep=False)  # 省~6MB/窗口深拷贝

# 3. 显式打破循环引用
del X_train, y_train, lgb_train, lgb_val
gc.collect()  # 打破 Booster→callback→Dataset 循环引用

# 4. 线程安全缓存
_FEATURE_CACHE_LOCK = threading.Lock()
with _FEATURE_CACHE_LOCK:
    _FEATURE_CACHE[cache_key] = (train_p, val_p, pred_p, feature_cols)

# 5. Bottleneck加速（可选）
try:
    import bottleneck as bn
    _nanmean = bn.nanmean   # 比np.nanmean快3-5x
    _nanstd  = bn.nanstd
except ImportError:
    _nanmean = np.nanmean
    _nanstd  = np.nanstd
```

### 5.5 工程防御性编程

```python
# 1. 参数浅拷贝 — 禁止原地修改调用方dict
def __init__(self, params=None):
    self.params = dict(params) if params else {}  # 浅拷贝

# 2. OOM三级防御
if avail_gb < 0.5: return -999.0   # 跳过Trial
if rss_gb > 12.0: release_memory() # 强制归还
if vram_mb > 7000: gc.collect()    # 显存警告

# 3. Trial超时守卫
future = executor.submit(run_m2, **kwargs)
portfolios, stats = future.result(timeout=1800)  # 30分钟硬限

# 4. 收益回填多级Fallback
if pred_date in date_return_map:
    portfolio["Target_Return_1M"] = portfolio["stock_code"].map(...)
else:
    # Fallback 1: 用pred_df内置收益
    # Fallback 2: 二次补充stock-level缺失

# 5. 内存泄漏检测 (方案D)
# 三条件同时满足才判定: RSS增长>60% + 绝对>2GB + 可用<1.5GB
check_rss_leak(trial.number, stop_now_event)
```

---

## 六、模块技术细节

### 6.1 M0 数据库

**目录**：`m0_database/`
**入口**：`python run_m0_full.py`

#### 核心组件

| 文件 | 职责 |
|------|------|
| `pipeline.py` | 顶层 pipeline，支持全量/增量/断点续跑 |
| `data_fetcher.py` | Tushare 拉数（日线/财务/指数），备用 akshare |
| `stock_filter.py` | 股票池筛选（ST/退市/次新/停牌/市值/换手率过滤） |
| `factor_calculator.py` | 因子计算（**23 大类 460 列因子**，含 Alpha101/Barra/AQR/JQ/宏观） |
| `neutralization.py` | 中性化（**8 套方案**：A/B/B1/B2/D/E/F/G，含 WLS/PCA/风格剥离） |
| `format_validator.py` | 数据格式校验 |
| `regenerator.py` | 增量重生（指定月份重新计算） |
| `_preflight_check.py` | 启动前环境检查（Tushare token/磁盘空间等） |
| `benchmark_loader.py` | Benchmark 数据加载 |
| `1_查看进度.bat` ~ `5_全量重建.bat` | Windows 批处理启动器 |

#### 8 套中性化方案

| 方案 | 说明 |
|------|------|
| `scheme_a` | 双重 OLS 正交化（行业+市值，机构标准版） |
| `scheme_b` | Rank-Z + 双重 OLS（**推荐**，抗极端值） |
| `scheme_b1` | 非线性市值 OLS（B 基础 + ln(M)² + ln(M)³，demean 防御高阶共线性） |
| `scheme_b2` | WLS 加权（sqrt(mktcap) 对角权重，缺 mktcap 退化为等权 OLS） |
| `scheme_d` | 仅行业 OLS（对照组，豁免 size_ 前缀） |
| `scheme_e` | 分层中性化（行业内 Rank + 全截面市值 OLS） |
| `scheme_f` | 风格因子剥离（B 基础 + 动量/换手/波动率等 Barra 风格） |
| `scheme_g` | PCA 隐式风险（60 天 log-return 矩阵 NumPy SVD） |

---

### 6.2 M1 滚动切分引擎

**目录**：`m1_engine/`
**入口**：`python m1_engine/run_m1.py`

#### 核心组件

| 文件 | 职责 |
|------|------|
| `data_loader.py` | 并行读取 M0 Parquet 池，转 float32/category，省内存 |
| `label_maker.py` | 截面 `label_rank ∈ [0,1]`（`pct=True` 排名分位数） |
| `rolling_splitter.py` | 时序滚动切分器（Python 生成器 yield） |
| `run_m1.py` | 顶层入口 |

#### 滚动窗口参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `train_months` | **43** | 训练窗口月数 |
| `valid_months` | **12** | 验证窗口月数 |
| `test_months` | **1** | 预测窗口月数 |
| `step_months` | **1** | 每次向前滑动月数 |

```
窗口 i:
  train: m_i ~ m_{i+42}     (43个月)
  val:   m_{i+43} ~ m_{i+54} (12个月)
  pred:  m_{i+55}            (1个月)
```

#### 关键设计

- **生成器模式**：`yield` 逐窗口返回，不预生成所有窗口到内存
- **严格无数据泄露**：train < val < pred 时间严格单调
- **并行 I/O**：`ThreadPoolExecutor` 异步读取 Parquet

---

### 6.3 M2 双引擎训练与集成

**目录**：`m2_engine/`（CPU），`m2_engine_gpu/`（GPU 加速）
**入口**：`python m2_engine/run_m2.py`

#### 流水线

```
load window → FeatureStore.fit_transform → EnsemblePredictor.fit_predict
  → PortfolioBuilder.build → backfill收益 → compute_val_metrics
```

#### 核心模块

| 文件 | 职责 | 最新版本特性 |
|------|------|-------------|
| `feature_store.py` | 特征工程 7 步管线 | v3.8: 纯 numpy 向量化，`transform()` 复用 `_fit_state` |
| `lgbm_model.py` | LightGBM Ranker | 自适应叶子数 + 学习率衰减 + OpenCL GPU |
| `xgb_model.py` | XGBoost Ranker | 学习率调度 + CUDA GPU + gamma 正则化 |
| `ensemble.py` | 软投票集成 + IC 计算 | 向量化 `_per_group_rank_corr` + `sliding_window_view` |
| `portfolio_builder.py` | Top10 持仓 + 换手成本 | SHAP 归因（Top3 因子写入 DataFrame） |
| `gpu_detector.py` | GPU 自适应（单例） | 策略 D（CPU train + GPU predict，bit-exact） |
| `lightweight_loader.py` | 超轻量级加载器 | 逐文件增量加载，避免 OOM |
| `smart_preprocessor.py` | 高效预处理器 | 分块读取 + 即时类型优化 |
| `run_m2.py` | 顶层入口 | v5.5: 收益回填多级 Fallback + OOM 哨兵 |

#### GPU 策略 D（v4.1 唯一保留策略）

```
策略D: LGBM CPU train + XGB CPU train → XGB GPU predict
  └── 与纯CPU路径 max_diff ~1.5e-7（噪声级差异）
  └── 删除策略 A/B/C/E，确保结果可复现
```

#### 集成预测器 (`ensemble.py`)

- **向量化 IC 计算**：`_per_group_rank_corr` — 1 次 argsort + 1 次 unique + per-group slice，比 per-date 布尔 mask 快 12x
- **`compute_val_portfolio_metrics`**：模拟持仓 6 月滚动计算 19 项绩效指标（含 `val_ic_stability` / `val_ir_stability` / `turnover_penalty`）
- **捕获比**：`_compute_capture_ratios` — 上行/下行/综合捕获比
- **Jensen's Alpha**：`_compute_jensen_appraisal` — OLS 回归 + 年化 α + Appraisal Ratio

#### 持仓构建 (`portfolio_builder.py`)

| 分层 | 排名 | 权重 |
|------|------|------|
| High | 1-5 | 13% × 5 = 65% |
| Low | 6-10 | 7% × 5 = 35% |
| Reserve | 11-20 | 0%（备选） |

- **永远满仓**：100% 仓位，无空仓信号
- **换手成本**：印花税 0.1%（仅卖出）+ 佣金 0.03%（双边）+ 滑点 0.1%（双边）

---

### 6.4 M3 TET 风控外挂

**目录**：`m3_engine/`
**入口**：`python m3_engine/run_m3.py`

#### 4 个核心指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **TS** | Trend-Score | 4 大类趋势因子分层投票 |
| **EI** | Emotion-Index | 12 个振荡器按 direction 归一化 |
| **ATS** | Anchored-Trend-Score | Schmitt Trigger 穿轴锚定 |
| **Timing** | ATS - EI | < sell_threshold 触发 SELL_TET |

#### 状态机规则

- **规则 A**：新入选股票 ATS = 当期 TS
- **规则 B**：SELL_TET 后立即 pop state
- **规则 C**：现金池不结转

#### 技术栈

- 使用 `polars` 做高性能数据处理
- `@dataclass(slots=True)` 定义配置类 `M3Config`
- `tqdm` 进度条

---

### 6.5 M4 报告与归因

**目录**：`m4_report/`
**入口**：`python m4_report/report_generator.py`

#### 绩效指标 (28+ 项)

| 类别 | 指标 |
|------|------|
| 收益 | CAGR、CAGR_benchmark、annual_excess、net_cagr_after_cost |
| 风险 | max_drawdown、volatility、downside_volatility、upside_volatility、VaR(95%)、CVaR(95%) |
| 比率 | sharpe、sortino、calmar、ir、sterling、burke、martin、omega、tail |
| 捕获 | up_capture、down_capture、capture_ratio |
| 分布 | skewness、kurtosis、pain_index、ulcer_index |
| 胜率 | monthly_win_rate、rolling6m_win_rate |
| α/β | jensen_alpha（年化）、appraisal_ratio |
| 成本 | avg_monthly_turnover_cost、avg_annual_turnover_cost |

#### 9 项门槛检查

| 检验项 | 阈值 |
|--------|------|
| IR | ≥ 0.50 |
| Calmar | ≥ 1.00 |
| MaxDD | ≥ -35% |
| Sortino | ≥ 1.20 |
| annual_excess | ≥ 5% |
| rolling6m_win_rate | ≥ 60% |
| capture_ratio | ≥ 1.20 |
| pain_index | ≤ 0.10 |
| omega_ratio | ≥ 1.20 |

#### 三大归因 (`attribution.py`)

| 归因 | 方法 |
|------|------|
| **Brinson** | 市场收益 / 行业配置 / 个股选择拆解 |
| **五因子** | Fama-French 5 + Carhart Momentum (MKT/SMB/HML/RMW/CMA/MOM) |
| **Barra** | 10 个 barra_ 因子暴露 + 因子收益代理 |

#### HTML 报告特性

- iOS 26 风格设计
- Chart.js 内嵌（完全离线可用）+ 服务端 SVG 兜底
- 净值曲线 + 时间区间滑块 + 鼠标 hover 提示
- Brinson / Five-Factor / Barra 归因图表
- SHAP 因子归因（Top10 持仓 × Top5 因子）
- 月度持仓交互页（选年-月看 10 只 + 分层 + 个股收益）

---

### 6.6 M5 贝叶斯超参优化

**目录**：`m5_optimizer/`
**入口**：双击 `m5_optimizer/启动M5优化器.bat` 或 `python m5_optimizer/app.py`
**Web UI**：http://127.0.0.1:7860

#### 30 维搜索空间

| 组 | 数量 | 关键参数 |
|----|------|----------|
| **LGBM** | 12 | learning_rate (log)、n_estimators、max_depth、colsample_bytree、reg_alpha/lambda、min_split_gain、lr_mode、decay_every/factor、depth_mode、early_stopping |
| **XGB** | 11 | learning_rate (log)、n_estimators、max_depth、colsample_bytree、reg_alpha/lambda、gamma、lr_mode、decay_every/factor、early_stopping |
| **Ensemble** | 1 | lgbm_weight (0.3-0.7) |
| **Feature** | 5 | min_valid_rate、max_corr、min_ic_abs (log)、min_keep_factors、drop_short_term_noise |
| **Window** | 1 | train_months (52-60) |

#### 37 维目标空间

- **M2 验证集类**（17 项）：val_ic、val_icir、val_rolling6m_ir、val_rolling6m_sortino、val_rolling6m_return、ic_gap_penalty、penalized_rate、val_global_ir、val_annual_return、pct_positive_excess、ir_worst_quartile、val_rolling6m_excess、val_rolling6m_excess_ann、val_jensen_alpha、val_appraisal_ratio、val_beta、val_ic_stability
- **M2 压力测试**（3 项，disabled）：2008/2015/2022
- **M4 回测类**（17 项）：cagr、monthly_win_rate、downside_volatility、upside_volatility、volatility_ratio、var_95、cvar_95、skewness、kurtosis、pain_index、omega_ratio、burke_ratio、martin_ratio、tail_ratio、up_capture_ratio（enabled）、down_capture_ratio（disabled）、capture_ratio（disabled）

#### 评分归一化 (NORM_CONFIG)

| 方法 | 适用指标 | 映射 |
|------|----------|------|
| `linear` | 率指标（val_ic, monthly_win_rate, …） | 截断映射到 [-0.5, 0.5] |
| `tanh` | IR/Sortino/Jensen's α（集中 0 附近） | 压缩到 (-1, 1) |
| `signed_log` | 长尾分布（capture_ratio, annual_return, …） | sign(x)×log(1+\|x\|) |

#### 双阶段优化策略

| 阶段 | 策略 | Trial 数 | 说明 |
|------|------|----------|------|
| **Phase1** | 全局探索 | 50-200 | TPESampler + 约束函数 + 断点续跑 + 热启动先验 |
| **Phase2** | 局部精调 | 30-100 | 从 P1 最优出发 + 范围收紧 + Trial 复用 + 独立 SQLite |

#### Gradio Web UI (5 Tab)

| Tab | 功能 |
|-----|------|
| **Tab1** 训练控制 | 选项目 → 启动 P1/P2 → 实时进度条 → 立即停止 / 优雅停止 |
| **Tab2** 结果分析与反推 | P1 Trial 统计、按因变量滑块过滤、反推 P2 搜索区间 |
| **Tab3** Top-5 排名 | P1 + P2 联合排序最优 5 个 Trial 详情 |
| **Tab4** 参数热力图 | 因子-IC 关系、参数-分数敏感度图 |
| **Tab5** 项目管理 | 创建/克隆/删除优化项目，查看历史记录 |

#### 辅助工具 (`utils/`)

| 文件 | 职责 |
|------|------|
| `logger.py` | 统一日志门面 |
| `rolling_logger.py` | 双通道滚动日志（关键事件同步 flush + 普通事件异步队列） |
| `memory_monitor.py` | 内存监控（psutil） |
| `win_memory.py` | Windows 内存强制归还（SetProcessWorkingSetSize + msvcrt._heapmin） |
| `trial_callback.py` | Optuna 回调工厂（双停止机制） |
| `restart_check.py` | RSS 增长率检测器（内存泄漏判定） |
| `retroactive_normalize.py` | 事后归一化（P2 区间反推） |

---

## 七、入口程序速查

| 程序 | 命令 | 作用 |
|------|------|------|
| `run_m0_full.py` | `python run_m0_full.py` | M0 全量数据拉取 → 8 套中性化方案 Parquet 落盘（断点续跑） |
| `run_m1.py` | `python m1_engine\run_m1.py` | M1 滚动切分 → 窗口 Parquet 落盘 |
| `run_m2.py` | `python m2_engine\run_m2.py` | M2 双引擎训练 → all_portfolios.parquet |
| `run_m2_m4.py` | `python run_m2_m4.py` | 一键跑 M2 + M4（CPU 模式） |
| `run_one_full.py` | `python run_one_full.py baseline 60` | 单策略全量 benchmark |
| `run_m3.py` | `python m3_engine\run_m3.py` | M3 TET 风控外挂 |
| `report_generator.py` | `python m4_report\report_generator.py` | M4 HTML 报告生成 |
| `启动M5优化器.bat` | 双击 | 启动 Gradio Web UI (http://127.0.0.1:7860) |
| `app.py` | `python m5_optimizer\app.py` | 同上（命令行等价） |

---

## 八、目录约定

```
10q-202604gpu/
├── config/                    # 配置
│   ├── config.yaml            # 主配置（8套中性化方案、滚动参数、换手成本、9项门槛）
│   ├── config_m3.yaml         # M3 TET 风控配置
│   └── concurrency_config.py  # 自适应线程/并发数（低内存高CPU模式）
│
├── m0_database/               # M0 数据模块
│   ├── pipeline.py            # 顶层 pipeline（全量/增量/断点续跑）
│   ├── data_fetcher.py        # Tushare 拉数 + akshare 备用
│   ├── stock_filter.py        # 股票池筛选（ST/退市/次新/停牌/市值/换手率）
│   ├── factor_calculator.py   # 因子计算（23大类460列）
│   ├── neutralization.py      # 中性化（8套方案: A/B/B1/B2/D/E/F/G）
│   ├── format_validator.py    # 数据格式校验
│   ├── regenerator.py         # 增量重生
│   ├── benchmark_loader.py    # Benchmark 数据加载
│   ├── _preflight_check.py    # 启动前环境检查
│   └── 1_查看进度.bat ~ 5_全量重建.bat  # Windows 批处理启动器
│
├── m1_engine/                 # M1 滚动切分
│   ├── data_loader.py         # 并行 Parquet 加载（float32/category 优化）
│   ├── label_maker.py         # 截面 label_rank 生成
│   ├── rolling_splitter.py    # 时序滚动切分器（生成器 yield）
│   └── run_m1.py              # 顶层入口
│
├── m2_engine/                 # M2 CPU 训练
│   ├── ensemble.py            # 软投票集成 + 向量化IC + 19项验证集绩效
│   ├── feature_store.py       # 特征工程 7 步管线（纯numpy向量化）
│   ├── lgbm_model.py          # LightGBM Ranker（CPU/OpenCL）
│   ├── xgb_model.py           # XGBoost Ranker（CPU/CUDA）
│   ├── portfolio_builder.py   # Top10持仓 + 换手成本 + SHAP归因
│   ├── preprocessor.py        # 数据预加载
│   ├── smart_preprocessor.py  # 高效预处理器（分块+即时优化）
│   ├── lightweight_loader.py  # 超轻量级加载器（增量追加防OOM）
│   ├── gpu_detector.py        # GPU/CPU 自适应（单例，策略D）
│   └── run_m2.py              # 顶层入口（v5.5: OOM哨兵+收益Fallback）
│
├── m2_engine_gpu/             # M2 GPU 加速训练（与 m2_engine 平行）
│   ├── ensemble.py            # GPU 策略 D
│   ├── feature_store.py
│   ├── lgbm_model.py          # GPU LightGBM (OpenCL)
│   ├── xgb_model.py           # GPU XGBoost (CUDA)
│   ├── portfolio_builder.py
│   ├── gpu_detector.py
│   └── run_m2.py
│
├── m3_engine/                 # M3 TET 风控外挂
│   ├── tet_engine.py          # TET核心（TS/EI/ATS/Timing + 状态机）
│   ├── m3_runner.py           # M3 运行器
│   ├── run_m3.py              # 顶层入口
│   └── __main__.py            # 模块入口
│
├── m4_report/                 # M4 回测报告
│   ├── metrics.py             # 28+项绩效指标 + 9项门槛
│   ├── attribution.py         # 三大归因（Brinson/五因子/Barra）
│   ├── report_generator.py    # iOS 26风格HTML报告（Chart.js内嵌）
│   └── static/chart.umd.min.js # Chart.js 离线资源
│
├── m5_optimizer/              # M5 贝叶斯优化器
│   ├── app.py                 # Gradio Web UI（5 Tab）
│   ├── search_space.py        # 30维参数 + 37维目标 + NORM_CONFIG
│   ├── objective.py           # Optuna目标函数（三级OOM防御+Trial超时守卫）
│   ├── phase1_global.py       # Phase1 全局探索（TPE + 约束）
│   ├── phase2_local.py        # Phase2 局部精调（范围收紧）
│   ├── config_manager.py      # M5 配置读写
│   ├── project_manager.py     # 项目管理
│   ├── result_analyzer.py     # P1 结果分析
│   ├── range_analyzer.py      # P2 区间反推
│   ├── _launcher.py           # 启动器
│   ├── utils/                 # 辅助工具
│   │   ├── logger.py          # 统一日志门面
│   │   ├── rolling_logger.py  # 双通道滚动日志（崩溃定位）
│   │   ├── memory_monitor.py  # 内存监控
│   │   ├── win_memory.py      # Windows内存强制归还
│   │   ├── restart_check.py   # RSS泄漏检测
│   │   ├── trial_callback.py  # Optuna回调工厂
│   │   └── retroactive_normalize.py  # 事后归一化
│   ├── docs/                  # 设计文档
│   │   ├── 评分归一化设计方案.md
│   │   ├── 评分归一化实现说明.md
│   │   ├── M5_归一化与TPE汇报.md
│   │   └── M5_中性化策略说明.md
│   └── 启动M5优化器.bat       # 一键启动
│
├── diagnostics/               # 诊断脚本与报告
│   ├── test_full_regression.py  # 全量回归测试
│   ├── test_m5_trial_memory.py  # M5 Trial 内存测试
│   ├── test_memory_profile.py   # 内存分析
│   ├── test_speed_profile.py    # 速度分析
│   ├── test_vram_leak.py        # GPU 显存泄漏检测
│   ├── codebase_audit_report.md # 代码审计报告
│   ├── HH-报告B-rolling6m_ir_negative_rootcause.md  # IR负值根因分析
│   └── TT-报告A-m5_rolling6m_ir_diagnosis.md        # M5 IR诊断
│
├── tests/                     # 测试
│   ├── conftest.py
│   ├── test_tet_engine.py     # M3 TET 引擎测试
│   └── verify_v42_fixes.py    # v4.2 修复验证（9项自动化测试）
│
├── data/                      # 数据目录（不入 git）
│   ├── pool_v2_scheme_a/ ~ pool_v2_scheme_g/  # 8套中性化方案 Parquet 数据池
│   ├── cache_m0_factors/      # M0 因子缓存
│   ├── raw_cache/             # 原始数据缓存
│   └── benchmark_000906.parquet  # 中证800基准
│
├── output/                    # 输出目录（不入 git）
│   ├── backtest_report_*.html  # 回测HTML报告
│   ├── m1_windows/             # M1 窗口数据
│   ├── m5/                     # M5 优化结果
│   └── all_portfolios*.parquet # 组合数据
│
├── scripts/                   # 辅助脚本
├── run_*.py                   # 顶层入口
├── requirements.txt
├── CHANGELOG.md
├── README.md
└── .gitignore
```

> ⚠️ **不进入版本控制**：`data/`、`logs/`、`output/`、所有 `*.pkl / *.parquet / *.db` 缓存。
> 详见 [.gitignore](.gitignore)。

---

## 九、快速开始

### 9.1 运行环境要求

> **本仓库全部代码均经过单一环境验证：Windows 11 + Python 3.14**

| 项目 | 要求 | 备注 |
|------|------|------|
| **操作系统** | Windows 11 (22H2 / 23H2 / 24H2) | ⚠️ 当前**仅**在 Win11 上验证 |
| **Python** | **3.14.x** (推荐 3.14.0+) | ⚠️ **未测试 3.11/3.12/3.13** |
| **架构** | x86_64 / ARM64 | 两种均测试通过 |
| **内存** | **≥ 16 GB** | 本系统针对 16GB 笔记本深度优化 |
| **GPU**（可选） | NVIDIA GTX 1650 及以上 | GPU 仅加速 M2，CPU 模式同样可运行 |
| **CUDA**（可选） | 12.x | 配合 GPU 模式 |
| **CPU** | 8 核最佳 | LGBM=2线程 + XGB=3线程 为最优组合 |
| **磁盘** | ≥ 10 GB 可用 | 数据集 + Parquet 缓存 |
| **Tushare 积分** | ≥ 5000 | 拉取全 A 股日线/因子数据 |

### 9.2 安装依赖

```powershell
# 1. 确认 Python 版本
python --version   # 必须为 3.14.x

# 2. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 设置 Tushare Token（严禁写入config.yaml）
$env:TUSHARE_TOKEN = "your_token_here"

# 5. 验证 GPU（如有）
python -c "import xgboost; print('CUDA:', xgboost.get_build_info())"
```

### 9.3 跑通流程

```powershell
# M0 拉数据 + 计算因子（首次需要数小时，断点续跑）
python run_m0_full.py

# M1 滚动切分
python m1_engine\run_m1.py

# M2 训练 + 组合构建
python m2_engine\run_m2.py

# M3 TET 风控（可选）
python m3_engine\run_m3.py

# M4 报告
python m4_report\report_generator.py

# M5 启动优化器 Web UI
.\m5_optimizer\启动M5优化器.bat
# 浏览器打开 http://127.0.0.1:7860
```

---

## 十、性能基线

| 项 | 数值 | 说明 |
|----|------|------|
| 滚动训练窗口 | 43 月训练 / 12 月验证 / 1 月测试 | 可配置 |
| 全量窗口数 | ~180 个（2007-01 ~ 2025-12） | 228 - 56 + 1 |
| 持仓容量 | Top10 永远满仓（65% + 35%） | 无空仓信号 |
| CPU 模式内存峰值 | **10-12 GB** | 16GB 笔记本流畅运行 |
| GPU 模式显存峰值 | ~3.5 GB | GTX 1650 4GB 无 OOM |
| 8 核 CPU 单窗时延 | ~0.76s | benchmark_v42 实测 |
| GPU 加速比 | 1.5-2.5× | 视数据规模 |
| M5 单 Trial (60窗) | ~5-8 分钟 | fast_mode |
| M5 内存稳定性 | 连续 50 Trial 不增长 | 每 Trial 后归还内存 |

---

## 十一、安全提示

- **Tushare token 严禁提交**！仓库已配置为从环境变量 `TUSHARE_TOKEN` 读取
- 历史 commit 已剔除明文 token
- 如发现 token 泄露，请立即在 https://tushare.pro 重置
- **任何 *db / *parquet / *pkl 都不进 git**，见 [.gitignore](.gitignore)

---

## 十二、许可证

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

### 你可以做什么

| 行为 | 是否允许 |
|------|----------|
| 个人学习、研究、阅读源码 | ✅ |
| 在自有项目中 fork / 修改 | ✅ |
| 商用、部署到生产环境 | ✅ |
| 再分发（需保留版权与许可证声明） | ✅ |
| 申请专利 | ✅ |
| 闭源使用 | ✅ |
| 修改后以不同许可证发布 | ✅（须保留原作者版权） |

### 你必须遵守什么

- **保留版权声明**：在所有副本 / 衍生作品中保留 `Copyright 2026 zjwandcat`
- **标注修改**：若修改了源文件，必须明确标注"已修改"
- **包含 LICENSE 副本**：再分发时必须附带本 LICENSE 文件
- **NOTICE 文件**：必须保留 [NOTICE](NOTICE) 中的归属声明
- **专利授权终止条款**：若对任何 Contributor 发起专利诉讼，则该 Contributor 授予你的所有专利授权自动终止

### 第三方依赖

本项目依赖多个第三方开源库（pandas、numpy、lightgbm、xgboost、optuna、gradio、tushare、polars 等），完整列表及许可证见 [NOTICE](NOTICE) 文件。这些依赖保留各自原始许可证，**不受本项目 Apache 2.0 约束**。

### 风险声明

本项目仅供量化研究学习用途，**不构成任何投资建议**。因使用本项目代码产生的任何投资损失，**作者不承担任何责任**。详见 LICENSE 第 7、8 条（无担保 / 责任限制）。

---

## 十三、致谢

感谢 [Tushare Pro](https://tushare.pro) 提供高质量的 A 股数据接口，感谢 LightGBM、XGBoost、Optuna、Gradio、Polars、Chart.js 等开源社区。
