# M5 优化器自动关闭 Bug — 完整调查与修复方案

> 生成日期：2026-05-15
> 状态：待审批（未修改任何代码）

---

## 一、问题描述

用户在浏览器未关闭的情况下，M5 优化器 Gradio 服务在 5/15 中午运行期间**自行关闭**，
无任何异常日志记录，程序进程直接消失。

---

## 二、时间线还原

| 时间 | 事件 | 日志文件 |
|------|------|----------|
| 5/15 00:20:54 | 凌晨运行静默中断（最后一条 WINDOW_PROGRESS） | m5_rolling.log |
| 5/15 13:56:44 | **中午重启，两个线程同时加载数据**（日志双份） | m5_run.log |
| 5/15 15:18:40 | **OOM 崩溃**：Unable to allocate 177 MiB，内存飙到 14.1 GB | m5_run.log |
| 5/15 15:42:53 | 再次重启加载数据，已用 9.1 GB | m5_run.log |
| 5/15 16:03:35 | Trial 78 OOM：Unable to allocate 44.8 MiB | m5_error.log |
| 5/15 16:04:19 | Trial 79 OOM：Unable to allocate 42.7 MiB | m5_error.log |
| 5/15 16:04:22 | **最后一条日志**，之后完全静默 | m5_run.log |

关键证据 — 5/15 13:56:44 的日志（**双份加载**）：
```
可用内存: 8.0 GB                     ← 线程A检查
可用内存: 7.9 GB                     ← 线程B检查（内存已被线程A占用一部分）
数据加载完成(scheme=scheme_e)，已用内存: 9.6 GB  ← 线程A完成
数据加载完成(scheme=scheme_e)，已用内存: 9.6 GB  ← 线程B完成
```
这证明：**两个线程同时加载了同一份数据**，各占 ~9.6 GB，总内存 ~19 GB，
远超系统 16 GB 物理内存 → Windows OOM Killer 强制终止 Python 进程。

---

## 三、根因分析

### Bug #1（P0）：`_load_factor_df` 无线程锁 — 竞态条件

- **文件**：`m5_optimizer/app.py`
- **行号**：L112-L152
- **问题**：`_factor_df_cache` 是一个全局 dict，但没有任何 `threading.Lock` 保护。
  当两个线程同时调用此函数时，都会通过 `if cache_key in _factor_df_cache` 的检查（都 miss），
  各自独立加载一份数据到内存。

### Bug #2（P0）：`start_phase1` / `start_phase2` 无防重复启动保护

- **文件**：`m5_optimizer/app.py`
- **行号**：L1267-L1269（Phase1）、L1935（Phase2）
- **问题**：点击"开始"按钮时，没有检查上一次运行的线程是否还在运行。
  `_phase1_thread.is_alive()` 仅在停止按钮中使用，启动时完全没检查。

### Bug #3（P1）：`_factor_df_cache` 永不清理

- **文件**：`m5_optimizer/app.py`
- **行号**：L46、L144
- **问题**：数据加载到缓存后，整个进程生命周期都不会被清理。
  如果切换了 scheme（如从 scheme_e 切到 scheme_d），两份数据同时驻留内存。

### Bug #4（P1）：`rolling_splitter.py` 的 `.copy()` 在内存紧张时反复触发 OOM

- **文件**：`m1_engine/rolling_splitter.py`
- **行号**：L111-L121
- **问题**：每个滚动窗口都要对 factor_df 做 `.copy()`，pandas 内部的
  `_consolidate_inplace()` → `_merge_blocks()` 需要分配临时内存。
  当系统内存紧张时，即使只需 40-70 MiB 也会失败。

### Bug #5（P2）：`_memory_watchdog` 只打印日志不采取任何缓解措施

- **文件**：`m5_optimizer/app.py`
- **行号**：L2032-L2049
- **问题**：内存超 13 GB 时只打印 `logger.critical`，不做 GC 之外的任何操作。
  无法阻止进程被 Windows 杀死。

---

## 四、涉及的文件清单

| # | 文件路径 | 改动类型 | 优先级 |
|---|----------|----------|--------|
| 1 | `m5_optimizer/app.py` | **核心改动**：线程锁 + 防重复启动 + 缓存清理 | P0 |
| 2 | `m1_engine/rolling_splitter.py` | 内存优化：.copy() 前检查内存 | P1 |
| 3 | `m5_optimizer/objective.py` | 内存释放优化 | P1 |
| 4 | `m2_engine/run_m2.py` | 内存优化：降低阈值 | P1 |
| 5 | `m5_optimizer/phase1_global.py` | 不变（仅审查确认） | — |
| 6 | `m5_optimizer/phase2_local.py` | 不变（仅审查确认） | — |

---

## 五、修复方案

### 方案 A：`m5_optimizer/app.py` 核心修复（4 处改动）

#### A-1：添加全局线程锁

```python
# 在 L46 之后新增
_factor_df_load_lock = threading.Lock()
```

#### A-2：改造 `_load_factor_df`，用锁保护整个加载流程

```python
# 修改 L112-L152 的完整函数
def _load_factor_df(scheme: str = None):
    global _factor_df_cache
    cache_key = scheme or "default"

    # 第一次快速检查（无锁，避免不必要的等待）
    if cache_key in _factor_df_cache:
        return _factor_df_cache[cache_key]

    # 加锁：确保同时只有一个线程加载数据
    with _factor_df_load_lock:
        # ★ 二次检查（防止等待期间其他线程已加载完成）
        if cache_key in _factor_df_cache:
            return _factor_df_cache[cache_key]

        try:
            from m1_engine.data_loader import DataLoader
            from m1_engine.label_maker import LabelMaker

            # 检查可用内存
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / 1e9
            logger.info(f"可用内存: {available_gb:.1f} GB")

            if available_gb < 2.0:
                raise MemoryError(
                    f"可用内存不足 ({available_gb:.1f} GB < 2 GB)，"
                    f"请关闭其他程序后重试"
                )

            loader = DataLoader(scheme=scheme)
            factor_df = loader.load()
            factor_df = LabelMaker().make_labels(factor_df)

            mem_after = psutil.virtual_memory()
            used_gb = (mem.total - mem_after.available) / 1e9
            logger.info(f"数据加载完成(scheme={cache_key})，已用内存: {used_gb:.1f} GB")

            _factor_df_cache[cache_key] = factor_df
            return factor_df

        except MemoryError as e:
            logger.error(f"内存不足: {e}")
            raise
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
```

#### A-3：`start_phase1` 添加防重复启动保护

```python
# 在 start_phase1() 函数的 L1166 之后，在 try 之前添加
def start_phase1(project, n_trials, scheme, enable_timer_val, timer_hours_val, *all_values):
    if project is None:
        return "❌ 请先加载项目", ""

    # ★ 新增：检查 Phase1 是否已在运行
    global _phase1_thread
    if _phase1_thread is not None and _phase1_thread.is_alive():
        return "⚠️ Phase1 正在运行中，请勿重复启动。等待当前完成或点击停止后再启动。", ""

    # ... 后续代码保持不变 ...
```

同理，对 `start_phase2` 做相同处理：
```python
# 在 start_phase2() 函数的 L1808 之后添加
def start_phase2(project, n_trials, fast_mode, window_count, from_best, enable_timer_val, timer_hours_val, *all_values):
    if project is None:
        return "❌ 请先加载项目", ""

    # ★ 新增：检查 Phase2 是否已在运行
    global _phase2_thread
    if _phase2_thread is not None and _phase2_thread.is_alive():
        return "⚠️ Phase2 正在运行中，请勿重复启动。等待当前完成或点击停止后再启动。", ""

    # ... 后续代码保持不变 ...
```

#### A-4：添加缓存清理函数 + 在适当时机调用

```python
# 新增函数（放在 cancel_graceful_timer 之后）
def clear_factor_df_cache(scheme: str = None):
    """清理指定 scheme 或全部 factor_df 缓存"""
    global _factor_df_cache
    if scheme:
        cache_key = scheme or "default"
        if cache_key in _factor_df_cache:
            del _factor_df_cache[cache_key]
            logger.info(f"已清理 factor_df 缓存: {cache_key}")
    else:
        _factor_df_cache.clear()
        import gc
        gc.collect()
        logger.info("已清理全部 factor_df 缓存")
```

调用时机（在 `stop_now()` 和 `stop_graceful()` 中）：
```python
# 在 stop_now() 函数中（L1286），_stop_now_event.set() 之后
def stop_now(project):
    _stop_now_event.set()
    cancel_graceful_timer()
    clear_factor_df_cache()  # ★ 新增：停止时释放缓存
    ...

# 在 stop_graceful() 函数中（L1317），_stop_graceful_event.set() 之后
def stop_graceful():
    _stop_graceful_event.set()
    cancel_graceful_timer()
    clear_factor_df_cache()  # ★ 新增
    ...
```

对 Phase2 的 `stop_p2_now()` 和 `stop_p2_graceful()` 同样添加。

---

### 方案 B：`m1_engine/rolling_splitter.py` 内存优化

#### B-1：在 `.copy()` 前检查可用内存，不足时先 gc.collect()

```python
# 在 L110 之前添加内存检查逻辑
import gc
import psutil as _psutil

# 切片并copy（防止内存泄漏）
# ★ 内存检查：如果可用内存低于 3 GB，先尝试释放
if _psutil.virtual_memory().available < 3e9:
    gc.collect()

train_df = factor_df[
    factor_df["trade_date"].isin(train_months_list)
].copy()

# ... val_df, pred_df 同理 ...
```

**注意**：此改动为保守优化，不改变逻辑，只是在内存紧张时多做一次 `gc.collect()`。

---

### 方案 C：`m5_optimizer/objective.py` 内存释放优化

#### C-1：在 Trial 异常时也执行 gc.collect()

```python
# 修改 L301-L306 的 except 块
except Exception as e:
    logger.error(f"Trial异常: {e}\n{traceback.format_exc()}")
    rl = get_rolling_logger()
    _try_log(rl.log_error, type(e).__name__, str(e),
             traceback.format_exc())
    import gc
    gc.collect()  # ★ 新增：异常时也释放内存
    return -999.0
```

---

### 方案 D：`m2_engine/run_m2.py` 内存阈值调整

#### D-1：降低触发降级为串行的内存阈值

```python
# 修改 L393：从 10.0 GB 降为 8.0 GB
if mem_now > 8.0 and BATCH_SIZE > 1:  # 原来是 10.0
```

**理由**：系统只有 16 GB 物理内存，10 GB 已经太高了。
在 8 GB 时降级可以更早释放内存压力。

---

## 六、改动影响评估

| 改动 | 影响范围 | 风险等级 | 是否影响 Optuna DB | 是否影响已有 Trial |
|------|----------|----------|-------------------|-------------------|
| A-1 线程锁 | 仅 `_load_factor_df` | 低 | 否 | 否 |
| A-2 锁保护加载 | 仅 `_load_factor_df` | 低 | 否 | 否 |
| A-3 防重复启动 | `start_phase1`, `start_phase2` | 极低 | 否 | 否 |
| A-4 缓存清理 | `stop_*` 系列函数 | 低 | 否 | 否 |
| B-1 内存检查 | `rolling_splitter.split()` | 极低 | 否 | 否 |
| C-1 异常 GC | `objective.__call__` | 极低 | 否 | 否 |
| D-1 阈值调整 | `run_m2` 并行决策 | 低 | 否 | 否 |

**总结**：所有改动都是**非破坏性的**，不修改 Optuna 数据库结构，
不影响已有的 Trial 数据，不改变优化算法逻辑。
唯一的行为变化是：
1. 重复点击"开始"时会被拒绝（而非创建第二个线程）
2. 数据加载变为线程安全（而非竞态条件）
3. 停止运行时会释放缓存内存

---

## 七、实施顺序

| 步骤 | 改动 | 文件 | 预计影响 |
|------|------|------|----------|
| 1 | A-1 + A-2（线程锁） | `app.py` | **彻底解决竞态条件** |
| 2 | A-3（防重复启动） | `app.py` | 防止用户误操作 |
| 3 | A-4（缓存清理） | `app.py` | 释放不必要内存 |
| 4 | C-1（异常 GC） | `objective.py` | 降低 OOM 频率 |
| 5 | B-1（内存检查） | `rolling_splitter.py` | 降低 copy 失败率 |
| 6 | D-1（阈值调整） | `run_m2.py` | 更早降级为串行 |

**步骤 1-3 即可完全解决"程序自己关闭"的问题**。
步骤 4-6 是锦上添花的内存优化。

---

## 八、测试建议

修复完成后建议的测试流程：

1. **竞态条件测试**：快速连续点击"开始 Phase1"按钮两次，第二次应显示"正在运行中"警告
2. **内存压力测试**：运行 Phase1 期间打开多个浏览器标签页，观察内存是否稳定
3. **长运行测试**：运行 >20 个 Trial，确认无内存泄漏累积
4. **停止恢复测试**：点击停止后，等待内存释放，再次启动确认缓存已重建
5. **Phase1 → Phase2 切换测试**：完成 Phase1 后立即启动 Phase2，确认不会内存叠加

---

## 九、风险与免责声明

- 所有改动**不涉及** Optuna 数据库、算法逻辑、因子计算、模型训练
- 已有 Trial 数据完全不受影响
- 线程锁仅在数据加载时持有（~30-60 秒），不会阻塞正常的 Trial 执行
- 缓存清理在停止时执行，不会影响正在运行的 Trial

---

*报告结束*
