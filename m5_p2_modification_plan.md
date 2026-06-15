# M5 P2 五项升级 — 实施记录

## 已完成修改（按tthh0602.md执行）

### 修改1：上分位数%默认100
- **文件**：`m5_optimizer/app.py` Tab2
- **改动**：`pct_high = gr.Slider(70, 100, value=100, step=1, label="上分位数%")`
- **原因**：上分位数<100%会丢弃最高分部分Trial的参数值，上限100%保留全部

### 修改2：Trial数上限300，默认60
- **文件**：`m5_optimizer/app.py` Tab3
- **改动**：`p2_trials = gr.Slider(5, 300, step=5, value=60, label="Trial数")`

### 修改3：删除快速模式
- **3-A**：删除`p2_fast_cb`和`p2_windows` UI组件
- **3-B**：`start_phase2()`签名移除`fast_mode, window_count`，函数体内用字面量`False`/`173`
- **3-C**：`btn_start_p2.click` inputs删除`p2_fast_cb, p2_windows`
- **3-D**：`phase2_local.py` `fast_mode: bool = False, window_count: int = 173`（默认值已是全量）

### 修改4：数值参数添加勾选框
- **4-A UI**：每个数值参数行 = `[✓ checkbox] [**name**] [下限] [上限] [固定值(隐藏)]`
  - `p2_param_widgets[name] = {"checkbox": cb2, "low": lo2, "high": hi2, "default_val": dv2}`
  - 勾选→显示low/high，隐藏default_val；取消→隐藏low/high，显示default_val
- **4-B toggle**：`_make_toggle_fn()`闭包绑定checkbox.change事件
- **4-C start_phase2解包**：
  - `all_values` = weights + checkboxes(26) + lows(26) + highs(26) + defaults(26)
  - 未勾选参数→`param_ranges[name] = {"adjusted_default": val}` + 加入`excluded_names`
  - 构建`active_params_final`：从ALL_PARAMS或p2_cfg_active中排除excluded_names
  - `run_phase2(active_params=active_params_final, param_ranges={**cfg_ranges, **ui_ranges})`
- **4-D click inputs**：`5(fixed) + 4(cat) + 31(weights) + 26×4(numeric) = 144`
- **on_save_p2_config**：同步更新解包逻辑和inputs

### 修改5：on_load_t3_project回填数值参数
- **5-A**：从p2_config.json读取param_ranges和active_params，按分组填充：
  - `cb_updates`(26) + `lo_updates`(26) + `hi_updates`(26) + `dv_updates`(26)
  - active参数→checkbox=True, low/high可见, default_val隐藏
  - 固定参数→checkbox=False, low/high隐藏, default_val可见
- **5-B outputs**：`4(原有) + 4(cat) + 26×4(numeric) = 112`

## P2热启动先验机制

1. p2_config.json `init_from_p1_best: true` → 启用
2. `phase2_local.py`运行时自动从P1 DB读取`best_trial.params`
3. `study.enqueue_trial(init_params)` 注入为首个Trial起点
4. TPESampler在n_startup_trials(=5)内优先使用enqueue参数，之后基于历史采样
5. 本质是"以P1最优为起点搜索"，非验证

## 验证结果

- `app.py` 语法检查 ✅
- `phase2_local.py` 语法检查 ✅
- on_load_t3_project outputs = 112 ✅（与tthh0602.md一致）
- btn_start_p2 inputs = 144 ✅（5+4+31+104）
