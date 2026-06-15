# Debug Session: m2-eval-oom (v4.1 D 策略下 OOM)

**Session ID**: `m2-eval-oom`
**Status**: [OPEN]
**Date**: 2026-06-11
**Reporter**: USER
**Symptom**: 进程突然死机，OOM (Out of Memory)
**Context**: 刚完成 v4.1 改造 — `m2_engine_gpu` 中删除 B/E 策略，仅保留 D + 纯 CPU fallback

---

## Hypotheses (按可能性排序)

### H1: GPU 显存 OOM (VRAM 4GB GTX 1650 极限)
- D 方案中 XGBoost `device="cuda"` 预测路径在数据量很大时会持有 GPU tensor 不释放
- GTX 1650 4GB 显存, 单 trial 叠加 N windows × 大量 feature 可能撑爆
- **可证伪点**: `nvidia-smi dmon` 中显存单 trial 增长曲线; `xgb.Booster.predict(device="cuda")` 是否调用 `inplace_predict`

### H2: 进程 RSS OOM (DMatrix + Dataset 未释放)
- XGBoost DMatrix 训练后 `del dtrain` 立即释放
- LGBM Dataset 在 windows 多窗口训练时, 若 fit() 退出前未置 None 会持续累积
- **可证伪点**: `tracemalloc` 在 fit() 前后快照; `psutil.Process().memory_info().rss` 在每 trial 增长

### H3: Optuna Study 累积 (加载所有 trials 到内存)
- `study.trials` 列表在 P1 跑到几十个 trial 后, pickle + 反序列化 17 个 user_attrs 每个会增长
- `_get_metric_actual_range` 在 UI Tab2 刷新时遍历全部 trials
- **可证伪点**: `len(study.trials) * avg_trial_size` 对比 RSS

### H4: 我刚改的 v4.1 代码有 bug 导致策略退化
- 比如 `set_strategy("B")` 实际没 fallback 到 D, 直接用 None 走默认路径
- `get_xgb_predict_device()` 返回 None 传给 xgb 触发异常
- **可证伪点**: 在 `m2_engine_gpu/gpu_detector.py` 单元测试中跑 B/E/A/C → D 的 fallback, 看 log

### H5: 18BB 数据集过大触发 OOM
- 18BB 项目 P1 配置 30 个 trial, 每 trial 训练 180+ windows, 数据是 30 维 × 数十万行
- 单 Optuna trial 估算: 17 指标 × 200 windows × 8D feature × 30d ≈ 几十 MB, 不应该 OOM
- **可证伪点**: data_loader 输出的 X_train.shape × n_windows × dtype_size

---

## Information Need From User

**在插桩之前, 需要你确认 3 件事**:

1. **OOM 发生在哪个进程?** (M5 UI Streamlit / M2 直接命令行调用 / 后台 trial worker)
2. **OS 报告的 OOM 类型?** (Windows 弹窗"内存不足" / Python MemoryError / CUDA OOM / 整个系统卡死蓝屏)
3. **触发 OOM 的具体操作?** (启动 M5 / 跑 P1 trial / 打开 Tab2 看历史 / 切换到 P2)
4. **运行到第几个 trial / 哪个 window 崩溃?** (log 文件最后一行)
5. **系统总内存和已用内存?** (任务管理器截图或 `wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value`)

---

## Plan

- **Step 1** (本轮): 创建本文件 + 列假设 + 等用户回复
- **Step 2**: 在 `m2_engine/xgb_model.py.fit` / `m2_engine_gpu/xgb_model.py.fit` 入口加 RSS+VRAM 探针
- **Step 3**: 在 `m5_optimizer/objective.py` trial 入口加 RSS 探针
- **Step 4**: 跑一次短 trial (5 windows × 3 trials), 收集证据
- **Step 5**: 根据证据定位根因 → 最小修复
- **Step 6**: 跑对照 → 验证
- **Step 7**: 用户确认 → 清理

## Next Action
等待 USER 回复 H1-H5 确认 + 上述 5 个信息点
