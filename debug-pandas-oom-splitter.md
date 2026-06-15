# Debug: numpy OOM in rolling_splitter.split

**Session ID**: `pandas-oom-splitter`
**Status**: `[OPEN]`
**Reporter**: User (log review)
**First seen**: 2026-06-15 09:45:03 (Trial 52)
**Reproduced**: Yes — at least 2 distinct trials (52, 56) before Phase1 crash

## Symptom

```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 45.4 MiB
for an array with shape (483, 24625) and data type float32
```

Stack anchor:
```
m1_engine/rolling_splitter.py:125  →  factor_df[factor_df["trade_date"].isin(train_months_list)]
pandas/core/internals/blocks.py:1015 → algos.take_nd → np.empty(out_shape)
```

## Environment evidence (from `logs/m5/m5_run.log`)

| Timestamp        | Line   | Event                                                  |
|------------------|--------|--------------------------------------------------------|
| 2026-06-15 07:26 | 122-125| 可用内存 5.9 GB → 数据加载 11.3 GB (第1次, scheme_b1) |
| 2026-06-15 09:35 | 164-168| 可用内存 7.9 GB → 数据加载 9.4 GB (第2次, scheme_b1)  |
| 2026-06-15 09:45 | 169    | **Trial 52 OOM**: shape (483, 24625) → 45.4 MiB         |
| 2026-06-15 10:09 | 242    | **Trial 56 OOM**: shape (483, 9654)  → 17.8 MiB         |
| 2026-06-15 10:14 | 303,331| 二次错误: `KeyError: 'Record does not exist.'` (optuna) |
| 2026-06-15 10:14 | 330,358| `Phase1运行异常: cannot access local variable 'updated_state'` |

## 实际/预期

- 实际: Trial 在 `splitter.split()` 第一次进入时就 OOM
- 预期: 顺利完成窗口切片，OOM 不应发生在第一次 yield 之前

## Hypotheses (3 falsifiable)

### H1 — 系统级内存压力 + Windows 堆碎片
数据加载 9.4 GB / 可用 7.9 GB → 进程已经**超出物理内存** (在用 swap)。
`np.empty` 需要**连续**虚拟地址空间；即使 45.4 MiB 看似很小，碎片化时
也分配不到。

> 验证点: 在 `splitter.split` 入口打印 `psutil.Process(os.getpid()).memory_info().rss`
> 与 `psutil.virtual_memory().available`；读取 trial 0 vs trial 52 的差值。

### H2 — pandas 布尔索引对 wide DF 不友好
`factor_df[factor_df["trade_date"].isin(train_months_list)]` 在 pandas 内部走
`take_nd → np.empty(out_shape, dtype=arr.dtype)`，**每个 block 都会分配一份
完整副本**。当 `factor_df` 有 24,625 列时,这一步至少分配
`483 × 24625 × 4B ≈ 47.5 MiB`(且**每个数值 dtype block 都要分配一次**,
dtypes 混杂时倍数更大)。`min_keep_factors=113` 实际上把列数放大到 24K+。

> 验证点: `factor_df.dtypes.value_counts()` 与
> `factor_df.memory_usage(deep=True).sum()`；记录 train_df/val_df/pred_df
> 的 `memory_usage(deep=True)`。

### H3 — `train_months=52` 拉宽 483 行训练集 + 12+1 验证/测试集,3 次 .copy()
trial 55 的 `train_months=52`(默认 36)。`splitter.split(..., copy=True)` 对
每个窗口都做 `train_df/val_df/pred_df` 三次 `.copy()`。在 `run_m2.py:407`
外层 `for i, window in enumerate(...)` 内,3 个 DataFrame 的总内存放大
~3 倍。

> 验证点: 同一 Trial 的 `train_df.shape`、`val_df.shape`、`pred_df.shape`、
> `__sizeof__`；对比 `copy=False` 时的 `train_df.values.base is factor_df.values`。

## Status

`[OPEN]`

## Instrumentation Applied (Step 3-4)

| File | Lines | Point ID | Captures |
|------|-------|----------|----------|
| `m1_engine/rolling_splitter.py` | ~85-114 | `H1H2:split-entry` | `shape`, `total_mb`, `avail_gb`, `rss_gb`, `dtypes` 分布 |
| `m1_engine/rolling_splitter.py` | ~155-176 | `H3:per-window-pre-slice` | `window_idx`, `train_months_param`, 三段月份数 |

**Zero behavior change**: only 2 `urllib.request.urlopen` POSTs per `split()` call (1 at entry + 1 per window pre-slice). Both wrapped in `try/except` with `timeout=1`; server unreachable ⇒ silent skip.

## Debug Server

- **URL**: http://127.0.0.1:7777/event
- **Session ID**: `pandas-oom-splitter`
- **Env file**: `.dbg/pandas-oom-splitter.env` ✅ (auto-written by server)
- **Log file**: `.dbg/trae-debug-log-pandas-oom-splitter.ndjson`
- **Idle timeout**: 3600s

## Reproduction Acceleration (Step 6, user-authorized)

修改 `m5_optimizer/search_space.py` 2 个参数加速 OOM 复现:
- `train_months`: `[24, 60]` → `[52, 60]` (默认 36 → 56)
- `min_keep_factors`: `[50, 120]` → `[100, 120]` (默认 60 → 110)

**清理承诺**: 修复完成 / 用户中止时,务必把这两处改回原值再走 Step 11 cleanup。



## Open Questions (需用户决策)

- 是否接受"调试期间插桩的日志污染"?(TRAE-debugger 协议要求)
- 修复路径倾向: **A** 限 `min_keep_factors` 上限 / **B** 改 splitter 用索引 /
  **C** 在 objective 层加内存预算短路 / **D** 中止调试。
