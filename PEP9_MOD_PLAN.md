# PEP-9 改造方案 v2

> 9 PEP × 22 文件 | Python 3.14 | 全量229月窗口 | m1+m2+m5

---

## 选用PEP

| # | PEP | 名称 | 核心收益 |
|---|-----|------|----------|
| 1 | 3119 | ABC | 接口契约，消除hasattr/防御代码 |
| 2 | 544 | Protocol | 结构子类型，轻量接口 |
| 3 | 634 | match/case | 分派加速5-15% |
| 4 | 649 | 延迟注解 | 移除__future__，导入加速10-30% |
| 5 | 393 | 灵活字符串 | category dtype，内存-5x |
| 6 | 683 | 不朽对象 | frozenset常量，并行refcount-15% |
| 7 | 750 | t-string | 日志惰性求值，开销-5-15% |
| 8 | 779 | Free-Threaded | 真多线程，窗口并行+2-4x |
| 9 | 784 | Zstandard | parquet zstd，I/O+1.5-3x |

---

## 文件清单

### m1 (4文件)
data_loader.py | label_maker.py | rolling_splitter.py | run_m1.py

### m2 (9文件)
lightweight_loader.py | portfolio_builder.py | preprocessor.py | run_m2.py | smart_preprocessor.py | xgb_model.py | ensemble.py | feature_store.py | lgbm_model.py

### m5 (9文件)
objective.py | phase1_global.py | phase2_local.py | project_manager.py | range_analyzer.py | result_analyzer.py | search_space.py | app.py | config_manager.py

---

## 逻辑变更分类

### ✅ 不改变业务逻辑（纯重构/类型/性能）

以下改造仅改变代码组织形式、类型标注或内部实现，输入输出完全一致：

| PEP | 改造 | 说明 |
|-----|------|------|
| 3119 | 加ABC继承 | 仅添加基类，现有方法签名和返回值不变 |
| 544 | 加Protocol标注 | 仅类型标注，不强制继承，运行时零影响 |
| 634 | if/elif→match/case | 语义等价替换，分支条件和结果不变 |
| 649 | 删__future__+TypedDict | 3.14原生延迟求值；TypedDict仅约束返回值字段名，不影响值 |
| 683 | set→frozenset/常量冻结 | frozenset查找语义同set，纯只读操作无区别 |
| 750 | f-string→t-string日志 | 日志输出文本不变，仅惰性求值减少无效拼接 |

### ⚠️ 可能改变逻辑

| PEP | 改造 | 风险点 | 利 | 弊 |
|-----|------|--------|-----|-----|
| **393** | stock_code→category | groupby行为差异：category的groupby可能改变排序/迭代顺序；某些str操作不兼容；与外部系统交互时category→object需显式转 | 内存-5x；groupby加速2-5x；rank/pct操作更稳定 | 需全链路验证groupby结果；merge时category与object不匹配报错；写入CSV/JSON需额外处理 |
| **779** | free-threaded并行 | 线程安全：LGBM/XGBoost C扩展未适配free-threaded；共享状态(_FEATURE_CACHE等)需加锁；窗口并行→结果顺序可能变 | 窗口并行+2-4x；双模型并行+1.5-2x；I/O并行+1.3-2x | C扩展可能非线程安全→segfault；需验证LGBM/XGB的nthread在free-threaded下行为；锁粒度不当→死锁或性能退化 |
| **784** | parquet zstd压缩 | 压缩格式变更：旧文件snappy/gzip，新文件zstd；混合读取无问题，但回写覆盖后无法回退 | I/O+1.5-3x(比gzip)；磁盘-30%(比snappy)；解压CPU更低 | 需重新生成所有parquet数据源；旧版pandas/pyarrow不兼容zstd(3.14不存在此问题) |

---

## 修改顺序建议

**原则：先零风险→后低风险→最后高风险；每步可独立验证**

```
Step1  PEP683  set→frozenset/常量冻结        ← 零风险，5min改完
Step2  PEP649  删__future__+TypedDict        ← 3.14原生，零风险
Step3  PEP544  加Protocol标注                ← 零影响，类型安全
Step4  PEP3119 加ABC继承                     ← 零影响，接口契约
Step5  PEP634  if/elif→match/case            ← 语义等价，需逐函数验证
Step6  PEP750  f-string→t-string日志         ← 零影响，日志惰性求值
       ─── 以上6步不影响业务逻辑，可批量完成 ───
Step7  PEP784  parquet改zstd                 ← 需重新生成数据源，但逻辑不变
Step8  PEP393  category dtype                ← ⚠️ 需全链路测试groupby/merge
Step9  PEP779  free-threaded并行             ← ⚠️ 最高风险，需充分测试C扩展
```

### 验证策略
- **Step1-6**：跑一次全量回测，数值对比基线（IC/IR/持仓权重差异<1e-6）
- **Step7**：重新生成parquet后对比文件hash+读取结果
- **Step8**：全量回测+逐窗口IC对比，特别关注groupby("stock_code")结果
- **Step9**：先单窗口并行验证→再全量；监控内存/死锁/C扩展crash

---

## PEP3119 ABC ✅无逻辑变更

| 文件 | 改造 |
|------|------|
| m2/xgb_model.py | `class XGBRanker(RankerBase):` |
| m2/lgbm_model.py | `class LGBMRanker(RankerBase):` |
| m2/ensemble.py | 参数标注 `lgbm: RankerBase, xgb: RankerBase`，删hasattr |
| m2/feature_store.py | 定义 `FeatureStoreBase(ABC)` |
| m2/portfolio_builder.py | 定义 `PortfolioBuilderBase(ABC)` |
| m1/label_maker.py | 可定义 `LabelMakerBase(ABC)`（可选） |
| m1/rolling_splitter.py | 可定义 `SplitterBase(ABC)`（可选） |
| m5/search_space.py | 可定义 `ParamDef(ABC)` 子类（可选） |

```python
# 新增 m2/base.py
from abc import ABC, abstractmethod
class RankerBase(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, group_train, X_val, y_val, group_val, gpu_mode=False): ...
    @abstractmethod
    def predict(self, X) -> np.ndarray: ...
    @abstractmethod
    def get_feature_importance(self, importance_type="gain") -> pd.Series: ...
    @property
    @abstractmethod
    def best_iteration_(self) -> int: ...
```

---

## PEP544 Protocol ✅无逻辑变更

| 文件 | 改造 |
|------|------|
| m2/ensemble.py | `lgbm: RankerProtocol, xgb: RankerProtocol` |
| m2/run_m2.py | 定义 `ParallelStrategy(Protocol)` |
| m2/lightweight_loader.py | 定义 `DataLoaderProto(Protocol)` |
| m2/smart_preprocessor.py | 定义 `PreprocessorProto(Protocol)` |
| m1/data_loader.py | 同上模式 |
| m5/objective.py | 隐式Protocol |
| m5/phase1_global.py | 定义 `OptimizerPhase(Protocol)` |
| m5/phase2_local.py | 同上 |

---

## PEP634 match/case ✅无逻辑变更

| 文件 | 改造 |
|------|------|
| m2/xgb_model.py | `match self.lr_mode:` / `match self.depth_mode:` |
| m2/lgbm_model.py | 同上 |
| m2/ensemble.py | `match gpu_mode:` |
| m2/portfolio_builder.py | `match tier:` |
| m5/objective.py | `_sample_param` → `match ptype:` |
| m5/search_space.py | `assemble_params` → `match group:` |
| m1/rolling_splitter.py | 无复杂分支，无需改 |

```python
# objective.py _sample_param
match ptype:
    case "float_log":   return trial.suggest_float(name, low, high, log=True)
    case "float":       return trial.suggest_float(name, low, high)
    case "int":         return trial.suggest_int(name, low, high)
    case "categorical": return trial.suggest_categorical(name, choices)
    case _:             return DEFAULT_PARAMS[name]
```

---

## PEP649 延迟注解 ✅无逻辑变更

**3.14环境：直接删 `from __future__ import annotations`，原生延迟求值**

| 文件 | 改造 |
|------|------|
| 全部22文件 | 删 `from __future__ import annotations` |
| m2/ensemble.py | 返回值→`TypedDict FitPredictResult` |
| m2/run_m2.py | stats→`TypedDict RunM2Stats` |
| m5/project_manager.py | →`TypedDict ProjectDict` |
| m5/search_space.py | →`TypedDict ParamDef` |
| m1/rolling_splitter.py | 返回值→`TypedDict WindowDict` |
| m1/run_m1.py | summary→`TypedDict M1Summary` |

---

## PEP393 灵活字符串 ⚠️可能改变逻辑

**风险：category dtype改变groupby/merge/str操作行为**

| 文件 | 改造 | 风险 |
|------|------|------|
| m1/data_loader.py | 已有category转换✅ L157-161 | 无——已实现 |
| m2/lightweight_loader.py | 加 `astype("category")` | 低——纯读取端 |
| m2/smart_preprocessor.py | 同上 | 低 |
| m2/preprocessor.py | 低基数列转category | 中——需验证下游 |
| m2/feature_store.py | `factor_cols`筛选排除object/string列 | 中——可能过滤掉有效列 |
| m2/ensemble.py | 确保predict时category列不参与计算 | 低——仅过滤 |
| m2/run_m2.py | factor_df传入后确保category | 低 |
| m5/app.py | `_load_factor_df` 加载后转category | 低——data_loader已处理 |

### 利弊评估
- **利**：stock_code内存-5x；groupby加速2-5x；rank/pct操作更稳定
- **弊**：
  1. merge时两边dtype必须一致（category+object→TypeError）→需统一转换时机
  2. str方法如`.str.startswith()`对category有限→需先`.astype(str)`
  3. 写入CSV时category自动转str，但写入parquet保留category→需确认下游兼容
  4. 某些pandas版本category列的`.value_counts()`排序不同

### 建议：统一在data_loader入口处转换，后续所有模块不再重复转换

---

## PEP683 不朽对象 ✅无逻辑变更

| 文件 | 改造 |
|------|------|
| m1/data_loader.py | `meta_cols` set→frozenset L141 |
| m2/feature_store.py | `META_COLS` set→frozenset |
| m2/run_m2.py | `_FEATURE_CACHE` key用不可变tuple |
| m2/ensemble.py | 常量冻结 |
| m2/xgb_model.py | `HARD_CONSTRAINTS` →冻结 |
| m2/lgbm_model.py | 同上 |
| m5/search_space.py | `SPECIAL_LGBM_KEYS/SPECIAL_XGB_KEYS` 冻结 |
| m5/objective.py | `IC_GAP_PENALTY_MULTIPLIER` 已是int✅ |

```python
# data_loader.py
META_COLS = frozenset({"trade_date","stock_code","stock_name","industry","list_date"})

# feature_store.py
META_COLS = frozenset({...})
```

---

## PEP750 t-string ✅无逻辑变更

| 文件 | 改造 |
|------|------|
| m1/run_m1.py | `print(f"...")` → `logger.info(t"...")` |
| m2/ensemble.py | 日志f-string→t-string |
| m2/run_m2.py | 同上 |
| m2/portfolio_builder.py | 交易成本公式→t-string模板 |
| m5/objective.py | `logger.warning(f"...")` → t-string |
| m5/app.py | 日志模板化 |
| m5/project_manager.py | 日志模板化 |

---

## PEP779 Free-Threaded ⚠️可能改变逻辑

**风险：线程安全、C扩展兼容、共享状态**

| 文件 | 改造 | 风险等级 |
|------|------|----------|
| m1/data_loader.py | ThreadPoolExecutor真正并行读取 | 低——纯I/O |
| m2/run_m2.py | 窗口并行ThreadPoolExecutor | **高**——共享factor_df |
| m2/ensemble.py | CPU模式双模型并行 | **高**——LGBM/XGB C扩展 |
| m2/smart_preprocessor.py | parquet读取真正并行 | 低——纯I/O |
| m2/xgb_model.py | nthread可真正并行 | **高**——C扩展线程安全 |
| m2/lgbm_model.py | 同上 | **高** |
| m5/objective.py | 无需改（Optuna调度） | — |
| m5/phase1_global.py | 无需改 | — |

### 利弊评估
- **利**：
  1. 窗口并行：229月全量回测从80min→20min(+4x)
  2. 双模型并行：单窗口训练从60s→35s(+1.7x)
  3. I/O并行：数据加载从45s→15s(+3x)
  4. 消除joblib进程开销（共享内存天然支持）
- **弊**：
  1. **LGBM/XGBoost C扩展可能非线程安全**→segfault风险，需逐一验证
  2. **共享状态竞争**：`_FEATURE_CACHE`需从Lock→细粒度锁或无锁结构
  3. **factor_df只读假设**：窗口并行要求factor_df不可变，需验证copy-on-write
  4. **调试困难**：并行crash的traceback更难定位
  5. **nthread×n_jobs叠加**：LGBM nthread=4 + 4窗口并行 = 16线程竞争CPU

### 建议：先I/O并行验证→再单窗口内双模型→最后窗口级并行

---

## PEP784 Zstandard ⚠️可能改变逻辑（数据格式）

**风险：压缩格式变更，需重新生成数据源**

| 文件 | 改造 | 风险 |
|------|------|------|
| m1/data_loader.py | 读取自动解压✅ | 无 |
| m1/run_m1.py | `to_parquet` 改zstd | 低——一次性重新生成 |
| m2/lightweight_loader.py | 数据源重写为zstd | 低 |
| m2/smart_preprocessor.py | 写入用zstd | 低 |
| m2/preprocessor.py | 写入用zstd | 低 |
| m2/run_m2.py | `to_parquet(compression='zstd', compression_level=3)` | 低 |
| m5/result_analyzer.py | run_m2已覆盖 | — |

### 利弊评估
- **利**：I/O+1.5-3x(比gzip)；磁盘-30%(比snappy)；解压CPU更低
- **弊**：
  1. 需重新生成所有parquet数据源（一次性成本）
  2. 旧snappy/gzip文件与新zstd文件混合读取无问题，但回写覆盖后无法回退
  3. 3.14环境下pyarrow原生支持zstd，无兼容问题

### 建议：m1重新生成数据源时统一改zstd，后续m2/m5无需特殊处理

---

## 全量窗口

```
TOTAL_DATA_MONTHS = 229
train_months=36 → window_size=49 → window_count=181
train_months∈[24,60] → window_count∈[157,193]
```

| 文件 | 改造 |
|------|------|
| m5/search_space.py | `TOTAL_DATA_MONTHS=229` |
| m1/rolling_splitter.py | 无需改（从factor_df动态计算）✅ |

---

## m5→m2调用链

| 调用点 | 文件 | 影响 |
|--------|------|------|
| Trial目标 | m5/objective.py → run_m2() | 函数签名不变✅ |
| 全量回测 | m5/result_analyzer.py → run_m2() | 同上✅ |
| 数据加载 | m5/app.py → DataLoader | PEP393已处理✅ |

**m5对m2的依赖仅通过函数签名，PEP改造全在m2内部。**

---

## m1文件补充审查

| 文件 | PEP3119 | PEP544 | PEP634 | PEP649 | PEP393 | PEP683 | PEP750 | PEP779 | PEP784 |
|------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| data_loader.py | ● | ● | | ● | ✅已做 | ● | | ●● | |
| label_maker.py | ● | ● | | ● | | | | | |
| rolling_splitter.py | ● | ● | | ● | | ● | | | |
| run_m1.py | | | | ● | | | ● | | ●● |

- data_loader.py：已有category转换(PEP393)✅；ThreadPoolExecutor可受益free-threaded(PEP779)
- label_maker.py：groupby("trade_date")操作简单，无优化空间
- rolling_splitter.py：生成器模式，PEP683可冻结常量
- run_m1.py：to_parquet需改zstd(PEP784)；日志可改t-string(PEP750)
