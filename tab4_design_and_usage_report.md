# M5 优化器 Tab4（结果与部署）设计与实现报告

> **目的**：本报告完整描述 M5 优化器 Gradio Web 界面中 Tab4（"结果与部署"）当前的功能、UI 布局、事件处理逻辑与业务流程。读者无需查看 `app.py` 源码，仅凭本报告即可完整理解 Tab4 的设计理念与操作方式。

---

## 目录

1. [Tab4 的定位与设计目标](#1-tab4-的定位与设计目标)
2. [Tab4 整体 UI 布局](#2-tab4-整体-ui-布局)
3. [数据源与作用域](#3-数据源与作用域)
4. [核心辅助函数](#4-核心辅助函数)
5. [核心事件处理函数](#5-核心事件处理函数)
6. [业务流程详解](#6-业务流程详解)
7. [视觉与样式设计](#7-视觉与样式设计)
8. [边界情况与保护机制](#8-边界情况与保护机制)
9. [与其他 Tab 的协作关系](#9-与其他-tab-的协作关系)
10. [关键文件位置速查](#10-关键文件位置速查)
11. [近期迭代历史](#11-近期迭代历史)

---

## 1. Tab4 的定位与设计目标

### 1.1 定位

Tab4 是 M5 优化器的"**结果分析与最终部署**"环节，对应 Phase 2（P2）精调研究的成果展示与落地。它的核心使命是：

- **只关注 P2**：Tab4 显式只分析 P2 数据库（`phase2_local`），不读 P1（Phase 1 全局探索），避免与 Tab2/Result Analyzer 重复职责。
- **从研究到生产**：用户在 Tab4 中挑选一个中意的 Trial，将其参数写回 `config/config.yaml`（静态配置），或者直接触发 M2+M4 完整重跑（动态验证），完成从研究结果到生产部署的闭环。

### 1.2 设计目标

| 目标 | 实现方式 |
| --- | --- |
| **Top-N 排名可视化** | 用户可在 2~20 范围内自由选择展示多少个最优 Trial，每个 Trial 卡片显示排名、评分与因变量指标 |
| **参数收敛性诊断** | 借鉴 P1 报告生成器的判定算法，对 P2 的数值型/类别型参数进行收敛性分析 |
| **Trial 可选性** | 摒弃"读取最优"的硬编码方式，用户从 Top-N 排名区下拉选择任意一个 Trial 进行部署 |
| **深色模式兼容** | 全部使用 CSS 变量 `var(--border-color-primary, ...)` 而非硬编码颜色，Gradio 浅色/深色主题均能正常显示 |
| **内存安全** | 重跑前进行可用内存检查（≥3GB），并在每个 Trial 后主动释放内存，避免 OOM |
| **错误容忍** | 所有事件均捕获异常并以人类可读字符串形式返回到日志框 |

---

## 2. Tab4 整体 UI 布局

Tab4 由上到下分为 **6 个逻辑区段**：

### 2.1 区段 ①：项目加载区

```
[输入框：项目文件夹路径] [📂 加载 按钮]
[项目状态卡片：显示名称、状态、P2 DB 路径、最优评分等]
[数据源标记：P2 · scheme=xxx · 有效N✅ 异常M⚠️ 失败K❌ · 最优=xxx]
```

- 用户输入或粘贴项目文件夹绝对路径，点击"📂 加载"。
- 项目状态卡片用 4px 左侧色块表示状态：
  - 🟡 **fresh**（黄色）— 项目刚创建/未开始
  - 🔵 **in_progress**（蓝色）— P2 正在运行
  - 🟢 **completed**（绿色）— 全部 Trial 完成
  - ⚠️ **conflict**（橙色）— 加载失败/错误
  - ⬜ **unconfigured**（灰色）— 未加载

### 2.2 区段 ②：Top-N Trial 排名区

```
### 🏆 Top-N Trial 排名
[Slider: 展示最优的N个Trial  最小2 / 最大20 / 默认5 / step=1]
[🔄 刷新排名 按钮]
[HTML 渲染区：排名卡片列表]
```

- **Slider 范围 2~20**，默认 5，用户可自由调整。
- 每个 Trial 卡片包含：
  - **排名奖牌**：🥇/🥈/🥉 或 `#4`/`#5`……
  - **Trial 编号**
  - **评分**（`score = -trial.value`，数值越大越好；Optuna 内部为最小化）
  - **因变量指标行**（小字）：IC、ICIR、6M_IR、全局IR、正超额%、最差Q_IR、捕获比、降权率
  - **左侧色块**：前三名分别为金/银/铜色，其余为蓝色

### 2.3 区段 ③：参数收敛分析区

```
### 📊 参数收敛分析
[HTML 渲染区：收敛摘要 + 逐参数表格]
```

- 显示整体**收敛率**（`已收敛数 / 总参数数`），并用色块指示：
  - **绿色**（≥70%）：整体收敛良好
  - **橙色**（40%~70%）：部分收敛
  - **红色**（<40%）：整体未收敛
- 下方是逐参数表格，每行显示：
  - 参数名
  - 统计量（数值型：mean/std/CV；类别型：mode 频率）
  - 范围占比（数值型 Top 区搜索范围占原始搜索范围的比）
  - 状态（✓ 收敛 / → 继续搜索 + 原因）

### 2.4 区段 ④：选择 Trial 部署区

```
### 🚀 部署所选Trial
[Dropdown: 选择Trial编号（含评分标签） scale=3]
[📋 读取参数 按钮                                       scale=1]
[DataFrame: 所选Trial参数（参数名/值/下限/上限/组）]
[Markdown:  所选Trial指标摘要（评分/IC/ICIR/6M_IR/全局IR/正超额）]

[💾 写回config.yaml 按钮]   [🔄 触发M2+M4完整重跑 按钮]
```

- **Dropdown 选项**由 Top-N 排名区动态生成，格式为 `Trial {n} (评分{x.xxxx})`。
- **DataFrame** 显示该 Trial 的全部参数（含范围上下限、所属组）。
- **Markdown 摘要**显示该 Trial 的关键指标。

### 2.5 区段 ⑤：执行日志区

```
[Textbox: 执行日志（10行，自动滚动）]
```

- 写回 config 与完整重跑的运行结果均输出到此。
- 异步日志通过 `gr.Timer`（2 秒刷新一次）持续推送到此框。

### 2.6 区段 ⑥：报告路径与打开区

```
[Textbox: 报告路径（只读）]
[🌐 打开回测报告 按钮]
```

- 完整重跑完成后，回填报告路径；用户点击按钮可在默认浏览器中打开 HTML 报告。

---

## 3. 数据源与作用域

### 3.1 输入：项目文件夹

Tab4 接收**项目文件夹路径**作为唯一外部输入，必须由用户粘贴（项目文件夹的格式约定见下文）。

### 3.2 项目文件夹约定的内部结构

`init_project(path)` 解析后的项目字典（`project`）中 Tab4 用到的键：

| 键 | 用途 |
| --- | --- |
| `project_name` | 显示在卡片中 |
| `project_root` | 显示在卡片中 |
| `p2_db_path` | P2 SQLite 数据库绝对路径，**唯一数据源** |
| `p2_study_name` | P2 study 名称（默认 `phase2_local`） |

### 3.3 P2 数据库约定

- 路径：`<project_root>/p2/phase2_local.db`
- Study 名：`phase2_local`
- 内部使用 `optuna.load_study(study_name, storage=f"sqlite:///{p2_db_path}")` 加载。

### 3.4 P2 config 文件

- 路径：`<project_root>/p2_config.json`
- Tab4 通过 `load_p2_config(project)` 读取，其中 `config.scheme` 是中性化方案标识（`scheme_d` / `scheme_b` / `scheme_a` / `scheme_e`），用于：
  - 项目状态卡片上展示 `P2 · scheme=xxx`
  - `on_write_config` 时同步到 `config/config.yaml` 的 `data.neutralization.active_scheme`

---

## 4. 核心辅助函数

Tab4 内部定义了 3 个核心辅助函数。理解它们即可理解 Tab4 的核心算法。

### 4.1 `_get_p2_study(project) -> optuna.Study | None`

**职责**：安全加载 P2 study。

**逻辑**：
1. `project is None` → 返回 `None`
2. `p2_db_path` 不存在 → 返回 `None`
3. 否则 `optuna.load_study(study_name=p2_study_name, storage=...)`

**调用方**：所有需要读 Trial 的事件处理器。

### 4.2 `_render_ranking_html(study, top_n) -> (str, gr.update)`

**职责**：渲染 Top-N 排名卡片 HTML 与 Dropdown 更新。

**算法**：
1. 过滤有效 Trial：`state == COMPLETE and value is not None and value > -999`
2. 按 `value` 升序排序（Optuna 内部为最小化，越小越好 → `score = -value` 越大越好）
3. 取前 `min(top_n, len(completed))` 个
4. 对每个 Trial 构建卡片：
   - 排名奖牌 🥇🥈🥉 / `#N`
   - 评分 `score = -trial.value`
   - 从 `trial.user_attrs` 读取 8 个关键因变量指标：
     - `val_ic`（IC）
     - `val_icir`（ICIR）
     - `val_rolling6m_ir`（6M_IR）
     - `val_global_ir`（全局IR）
     - `pct_positive_excess`（正超额%）
     - `ir_worst_quartile`（最差Q_IR）
     - `capture_ratio`（捕获比）
     - `penalized_rate`（降权率）
5. 数值自动按量级格式化（`<0.01` 用 4 位小数；`<1` 用 3 位；否则 2 位）
6. 返回 `(html, gr.update(choices=[...], value=choices[0]))`

### 4.3 `_render_convergence_html(study) -> str`

**职责**：对 P2 全部参数进行收敛性诊断。

**判定规则**（借鉴 P1 报告生成器 S4/S6/S7 区）：

1. 过滤有效 Trial（条件同 4.2）
2. 若有效 Trial **不足 5 个** → 直接返回"⚠️ 有效Trial不足5个，无法进行收敛分析"
3. 按 `value` 升序排序，取**上半区**（前 50%）和**下半区**（后 50%）
4. 遍历 P2 全部出现过的参数名（取并集 `all_param_names`）：
   - **类别型 / 布尔型 / 字符串型**：
     - `mode_val = Counter(top_vals).most_common(1)[0][0]`
     - `mode_freq = mode 出现次数 / len(top_vals)`
     - **收敛条件**：`mode_freq > 0.80`
   - **数值型（float/int）**：
     - `top_arr = np.array(top_vals)`
     - `top_mean, top_std, cv = top_std / (abs(top_mean) + 1e-8)`
     - `std_ratio = top_std / bot_arr.std()`（下半区分散度）
       - 若下半区 std ≤ 0 或样本不足，则 `std_ratio = +∞`（必不收敛）
     - **收敛条件**：`std_ratio < 0.7 AND cv < 0.30`
     - 计算 `range_ratio = (top_arr.max() - top_arr.min()) / original_range` 作为展示
5. 汇总：
   - `converged_count / total_count = 收敛率`
   - 收敛率 ≥ 70% → 🟢；40%~70% → 🟠；< 40% → 🔴
6. 返回 HTML，包含整体摘要 + 逐参数表格（已收敛与未收敛分块列出）

**与 P1 报告生成器的一致性**：判定算法（`std_ratio < 0.7` + `CV < 30%` + 类别型 `mode > 80%`）完全沿用 `p1_report_generator.py` 的 S4/S6/S7 区域。

---

## 5. 核心事件处理函数

Tab4 共有 **6 个核心事件**，按触发顺序整理如下。

### 5.1 `on_load_t4_project(path)` — 📂 加载按钮

**触发**：`btn_t4_load.click`

**输入**：`t4_project_path`（项目文件夹路径）

**输出（8 个组件）**：
1. `t4_project_state`（State，存 `project` 字典）
2. `db_status`（Markdown，数据源摘要）
3. `t4_project_card`（HTML，状态卡片）
4. `t2_project_path`（同步给 Tab2）
5. `t3_project_path`（同步给 Tab3）
6. `t4_ranking_html`（HTML，Top-5 排名）
7. `t4_convergence_html`（HTML，收敛分析）
8. `t4_trial_selector`（Dropdown，初始 choices 为 Top-5）

**逻辑**：
1. 调用 `init_project(path.strip())` → 得到 `result = {"status": ..., "project": ..., "message": ...}`
2. 若 status ∉ {`loaded`, `created`} → 返回空状态卡片 + 提示
3. 解包 `project = result["project"]`
4. 判断 P2 DB 是否存在（`os.path.exists(p2_db_path)`）：
   - **存在** → 调用 `_get_p2_study` 加载 study，调用 `get_study_stats` 拿统计，调用 `load_p2_config` 拿 scheme，渲染完整 6 个输出
   - **不存在** → 渲染黄色 `fresh` 卡片 + 提示语，排名/收敛区显示灰色占位

### 5.2 `on_t4_refresh(project, top_n)` — 🔄 刷新排名按钮

**触发**：`btn_t4_refresh.click`

**输入**：`t4_project_state`、`t4_top_n`（Slider 值）

**输出（3 个）**：
1. `t4_ranking_html`
2. `t4_convergence_html`
3. `t4_trial_selector`（Dropdown choices 重置为 Top-N）

**逻辑**：
- 用最新 top_n 重新渲染排名与收敛，刷新 Dropdown 选项。

### 5.3 `on_t4_load_trial(project, selection)` — 📋 读取参数按钮

**触发**：`btn_t4_load_trial.click`

**输入**：`t4_project_state`、`t4_trial_selector`（DropDown 选中的 `Trial {n} (评分{x.xxxx})`）

**输出（2 个）**：
1. `t4_selected_params_df`（DataFrame）
2. `t4_selected_metrics`（Markdown）

**逻辑**：
1. 解析 `selection`：取 `(` 前的部分，去掉 `Trial`，得到 Trial number
2. `trial = study.trials[trial_num]`
3. 遍历 `trial.params.items()`，从 `ALL_PARAMS` 取每个参数的 low/high/group，构建 DataFrame 行
4. 拼接指标摘要 Markdown：`评分/IC/ICIR/6M_IR/全局IR/正超额`

### 5.4 `on_write_config(project, selection)` — 💾 写回 config.yaml 按钮

**触发**：`btn_write_config.click`

**输入**：`t4_project_state`、`t4_trial_selector`

**输出**：`log_box_deploy`（执行日志）

**逻辑**：
1. 同 5.3 解析 selection 取 trial
2. `best_params = dict(trial.params)`
3. 调用 `assemble_params(best_params)` → 解包为 `(lgbm_params, xgbm_params, feature_params, lgbm_weight, window_params)`
4. **备份** `config/config.yaml` → `config/config.yaml.bak`
5. 读取 `config.yaml` 字典
6. 分组更新：
   - `cfg.m2.lgbm.update(lgbm_params)`
   - `cfg.m2.xgb.update(xgbm_params)`
   - `cfg.m2.feature_store.update(feature_params)`
   - `cfg.m2.ensemble.lgbm_weight = lgbm_weight`
   - `cfg.rolling.train_months = window_params.train_months`
   - `cfg.data.neutralization.active_scheme = load_p2_config(project).config.scheme`
7. `yaml.dump` 写回
8. 返回成功摘要：参数数量、train_months、scheme

### 5.5 `on_run_full(project, selection)` — 🔄 触发 M2+M4 完整重跑按钮

**触发**：`btn_run_full.click`（主绑定）

**输入**：`t4_project_state`、`t4_trial_selector`

**输出**：`log_box_deploy`、`btn_run_full`（按钮交互态置灰）

**逻辑**：
1. **内存检查**：`psutil.virtual_memory().available < 3GB` → 直接拒绝并恢复按钮
2. 设置 `_t4_running["value"] = True`
3. 启动 daemon 线程 `_run()`：
   - 解析 selection 取 trial params
   - 拿 scheme（`project.get("scheme", "scheme_b")`）
   - 调用 `run_full_backtest(best_params, scheme, include_no_penalty=False, progress_callback=...)`
   - 进度通过 `_t4_log_queue` 异步推送
   - 完成 / 异常后向队列推最终消息
4. 主线程立即返回 `"⏳ 回测已启动，请等待..."` + 按钮置灰

**定时器联动**（同一 `btn_run_full.click` 的第二个绑定）：

```
btn_run_full.click(fn=lambda: gr.update(active=True), outputs=[t4_timer])
```

激活 2 秒间隔的 `t4_timer`，由 `on_t4_timer_tick` 持续从 `_t4_log_queue` 取消息刷到 `log_box_deploy`。

> 重复绑定的修复：早期版本中定时器区块的 `btn_run_full.click` 重复触发了 `on_run_full`，会导致点击一次执行两次。修复后定时器区块只保留 lambda 激活。

### 5.6 `on_open_report(report_path)` — 🌐 打开回测报告按钮

**触发**：`btn_open_report.click`

**输入**：`report_path_box`

**输出**：`log_box_deploy`

**逻辑**：
- 文件存在 → `webbrowser.open(f"file:///{abspath}")`
- 文件不存在 → 提示

---

## 6. 业务流程详解

### 6.1 标准操作流程

```
┌──────────────────────────────────────────────────────────┐
│ 1. Tab3 跑完 P2 精调（产生 phase2_local.db）              │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 2. 切到 Tab4，输入项目路径 → 点击「📂 加载」             │
│    触发 on_load_t4_project：                              │
│    - 解析 project 字典                                    │
│    - 加载 P2 study                                        │
│    - 渲染状态卡片 + Top-5 排名 + 收敛分析 + Dropdown       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 3. （可选）拖动 Slider 调整 Top-N → 「🔄 刷新排名」       │
│    触发 on_t4_refresh：                                    │
│    - 用新的 top_n 重新渲染排名与收敛                       │
│    - Dropdown choices 同步更新                            │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 4. 从 Dropdown 中挑选目标 Trial → 「📋 读取参数」         │
│    触发 on_t4_load_trial：                                │
│    - DataFrame 显示该 Trial 的全部参数与范围               │
│    - Markdown 显示该 Trial 的关键指标                      │
└──────────────────────────────────────────────────────────┘
                          ↓
              ┌───────────┴────────────┐
              ↓                        ↓
   ┌──────────────────┐    ┌──────────────────────┐
   │ A. 「💾 写回     │    │ B. 「🔄 触发 M2+M4   │
   │    config.yaml」 │    │    完整重跑」         │
   │  on_write_config │    │  on_run_full          │
   │  → 静态写配置   │    │  → 异步回测           │
   │  → 含 .bak 备份  │    │  → 日志框滚动         │
   └──────────────────┘    └──────────────────────┘
```

### 6.2 Dropdown 选择语义

Dropdown 文本格式：`Trial {n} (评分{x.xxxx})`

- `n` = `trial.number`（Optuna Trial 编号，0-indexed）
- `x.xxxx` = `-trial.value`（最大化为正分）

事件处理器从 selection 解析 Trial 编号的代码：

```python
trial_num = int(selection.split("(")[0].replace("Trial", "").strip())
```

### 6.3 写回 config 的粒度

| 写入字段 | 来源 |
| --- | --- |
| `m2.lgbm.*` | `assemble_params` 返回的 `lgbm_params` |
| `m2.xgb.*` | `assemble_params` 返回的 `xgbm_params` |
| `m2.feature_store.*` | `assemble_params` 返回的 `feature_params` |
| `m2.ensemble.lgbm_weight` | `assemble_params` 返回的 `lgbm_weight` |
| `rolling.train_months` | `assemble_params` 返回的 `window_params.train_months` |
| `data.neutralization.active_scheme` | `load_p2_config(project).config.scheme` |

**未触达**：`search_space` 中除上述分组外的其他参数（如 `objective_weights` 等）不会被本次 Trial 覆盖 — 这与 P1 报告生成器的设计一致。

---

## 7. 视觉与样式设计

### 7.1 色块语义

Tab4 大量使用 `border-left: 4px solid <color>` 的"色块"表达状态：

| 颜色 | 含义 |
| --- | --- |
| 🟡 `#ffc107` | 项目刚创建（fresh） |
| 🔵 `#2196f3` | 进行中（in_progress） |
| 🟢 `#4caf50` | 已完成（completed） |
| ⚠️ `#ff9800` | 冲突/错误（conflict） |
| ⬜ `#9e9e9e` | 未配置（unconfigured，opacity 0.7） |
| 🥇 `#ffd700` | Top-1 排名 |
| 🥈 `#c0c0c0` | Top-2 排名 |
| 🥉 `#cd7f32` | Top-3 排名 |
| 🔵 `#2196f3` | Top-4+ 排名 |

### 7.2 深色模式兼容

所有色块基础边框使用 CSS 变量：

```css
border: 1px solid var(--border-color-primary, #e0e0e0);
```

`var(--border-color-primary)` 是 Gradio 主题内置变量，浅色模式下为浅灰、深色模式下为深灰，因此卡片在两种主题下都正常显示（与 Tab3 保持一致）。

### 7.3 收敛率颜色阈值

| 收敛率 | 颜色 |
| --- | --- |
| ≥ 70% | 🟢 `#4caf50` |
| 40% ~ 70% | 🟠 `#ff9800` |
| < 40% | 🔴 `#f44336` |

---

## 8. 边界情况与保护机制

| 场景 | 处理 |
| --- | --- |
| 项目路径不存在 | `_card("conflict", message=...)` + 错误日志 |
| P2 DB 不存在 | 黄色 `fresh` 卡片 + 排名/收敛区显示灰色占位 + 提示"请先在Tab3运行Phase2" |
| 有效 Trial < 5 | 收敛分析区显示"⚠️ 有效Trial不足5个，无法进行收敛分析" |
| 排名区无有效 Trial | 橙色卡片"⚠️ P2尚无有效Trial" + Dropdown choices=[] |
| 未选择 Trial 就点写回/重跑 | 返回提示"请先加载项目并选择Trial" |
| `trial_num` 解析失败 | 异常被 `try/except` 捕获，返回"读取失败：{e}" |
| 内存 < 3GB | 拒绝启动回测，返回"❌ 可用内存不足（x.xGB < 3GB）" |
| Gradio 版本不支持 `gr.Timer` | `try/except (AttributeError, TypeError)` 跳过定时器绑定 |
| 回测过程异常 | daemon 线程通过 `_t4_log_queue` 推 `❌ 回测失败：{e}`，主线程不阻塞 |

---

## 9. 与其他 Tab 的协作关系

```
┌────────────┐  写 p2_config.json   ┌────────────┐
│   Tab1     │ ──────────────────▶ │   Tab3     │
│  (P1 全局) │                     │  (P2 精调) │
└────────────┘                     └────────────┘
                                          │
                                          │ 写 phase2_local.db
                                          ↓
                                    ┌────────────┐
                                    │   Tab4     │
                                    │ (结果部署) │
                                    └────────────┘
                                          │
                       ┌──────────────────┴──────────────────┐
                       ↓                                     ↓
              写 config/config.yaml              触发 M2+M4 完整重跑
                       │                                     │
                       ↓                                     ↓
                 生产环境静态配置                      生成回测报告
```

- **Tab1 → Tab3**：P1 全局搜索结果可作为 P2 局部精调的"热启动先验"。
- **Tab3 → Tab4**：P2 跑完后，Tab4 加载同一项目即可看到所有 P2 Trials。
- **Tab4 → config**：写回的 config 由用户后续手动启动生产回测/模拟。
- **Tab4 ↔ Tab2/Tab3**：项目路径通过 `on_load_t4_project` 的输出 4/5 同步给 Tab2/Tab3 输入框，避免重新输入。

> 备注：Tab2（结果分析与反推）已被设计为分析 P1 的反推工具；Tab4 是 P2 的展示与部署工具。两者分工明确，**不重叠**。

---

## 10. 关键文件位置速查

| 文件 | 用途 |
| --- | --- |
| `m5_optimizer/app.py` | Tab4 主代码（UI + 事件），约 L1045~L3205 |
| `m5_optimizer/phase2_local.py` | P2 精调入口（被 Tab3 调用，被 Tab4 间接消费其产出的 DB） |
| `m5_optimizer/search_space.py` | `ALL_PARAMS`、`calc_window_count()` 字典（参数范围/类型/分组的真理之源） |
| `m5_optimizer/project_manager.py` | `init_project`、`load_p2_config` 等项目工具 |
| `m5_optimizer/result_analyzer.py` | `get_study_stats` 等 P2 study 统计函数 |
| `p1_report_generator.py` | 收敛判定算法的参考实现（与 Tab4 共享同一套判定规则） |
| `config/config.yaml` | 写回配置的目标文件（写回前会备份为 `config.yaml.bak`） |

### Tab4 关键代码段位置（行号会随重构变化，仅作参考）

- `_get_p2_study` → 约 L2703
- `_render_ranking_html` → 约 L2716
- `_render_convergence_html` → 约 L2778
- `on_load_t4_project` → 约 L2902
- `on_t4_refresh` → 约 L2967
- `on_t4_load_trial` → 约 L2987
- `on_write_config` → 约 L3036
- `on_run_full` → 约 L3114
- `on_open_report` → 约 L3190
- Tab4 UI 定义 → 约 L1045~L1108

---

## 11. 近期迭代历史

### v1（基线）— 项目加载不刷新、显示"名称：- 方案：未知"
**问题**：`init_project()` 返回 `{"status":"loaded","project":{...}}` 但原代码把整个 result 当 project 用，导致所有字段读取失败。
**修复**：正确解包 `result["project"]`；scheme 从 `load_p2_config()` 而非 `project.get("scheme")` 读取。

### v2 — Top-N 排名 + 收敛分析
**新增**：
- `_render_ranking_html` 渲染 Top-N Trial 卡片（评分 + 因变量指标 + 排名奖牌）
- `_render_convergence_html` 渲染收敛分析（沿用 P1 报告生成器的判定规则）
- Slider（2~20，默认 5）控制 Top-N 大小
- Dropdown 动态生成 Top-N 选项

**修复**：
- Tab4 不再使用 `get_best_study_from_project()`，统一改为 `_get_p2_study`
- 替换 `top_n = gr.Slider(minimum=5, maximum=20)` 为 2~20
- `from collections import Counter` 移至文件顶部

### v3 — Trial 可选化部署
**新增**：
- Dropdown `t4_trial_selector` + 按钮「📋 读取参数」展示所选 Trial 详情
- `on_write_config` 改为基于所选 Trial 写回 config
- `on_run_full` 改为基于所选 Trial 触发 M2+M4 重跑
- 定时器 (`gr.Timer`) 2 秒刷新 `log_box_deploy` 异步日志
- 内存预警（< 3GB 拒绝启动回测）
- 修复 `btn_run_full.click` 重复绑定（删除定时器区块的第二次 `on_run_full` 绑定）

### v4 — 深色模式兼容
- 全部色块基础边框从硬编码 `#e0e0e0` 改为 `var(--border-color-primary, #e0e0e0)`
- 收敛率颜色阈值与色块颜色不变（语义化色值在两种主题下均可读）

### v5 — 内存稳健性
- `phase2_local.py` 开头与每个 Trial 后增加 `release_memory_to_os() + gc.collect()`
- `objective.py` 成功 return 前也主动 `release_memory_to_os()`
- Tab4 启动回跑前 `psutil` 检查可用内存 < 3GB 直接拒绝

---

## 附录 A：术语表

| 术语 | 含义 |
| --- | --- |
| **Trial** | Optuna 中的一次参数采样与评估，对应 P2 数据库中的一行 |
| **study** | Optuna 中一组 Trial 的容器，对应一个 P2 数据库文件 |
| **`trial.value`** | Optuna 内部为最小化目标函数，所以 `score = -trial.value` 是越大越好 |
| **`trial.user_attrs`** | 用户自定义属性字典，P2 中存了 val_ic / val_icir / 等因变量指标 |
| **`ALL_PARAMS`** | `search_space.py` 中定义的全部参数元信息（含 low/high/type/group） |
| **`assemble_params`** | 将 trial.params 扁平字典按组拆分为 lgbm/xgb/feature/lgbm_weight/window 的工具函数 |
| **`run_full_backtest`** | 触发 M2+M4 完整回测的主入口（被 Tab4 的「完整重跑」按钮调用） |
| **scheme** | 中性化方案标识符：`scheme_d` / `scheme_b` / `scheme_a` / `scheme_e` |
| **std_ratio** | 上半区 Trial 参数标准差 / 下半区 Trial 参数标准差，反映"好 Trial 是否比差 Trial 更聚焦" |
| **CV** | Coefficient of Variation，变异系数 = std / |mean|，反映参数分散度 |
| **range_ratio** | Top 区搜索范围 / 原始搜索空间，反映搜索是否在向某个子区间收缩 |

## 附录 B：用户视角的"做什么/看什么"速查

| 我想... | 应该点哪里？ |
| --- | --- |
| 看看 P2 跑得怎么样 | Tab4 区段 ① + 区段 ② |
| 知道 P2 是不是搜得差不多了 | Tab4 区段 ③（参数收敛分析） |
| 选出最优的 N 个 | Tab4 区段 ② 的 Slider + 「🔄 刷新排名」 |
| 看某个 Trial 的参数 | Tab4 区段 ② 选 Trial → 区段 ④「📋 读取参数」 |
| 把它写进生产 config | Tab4 区段 ④「💾 写回 config.yaml」 |
| 实际跑一次验证 | Tab4 区段 ④「🔄 触发 M2+M4 完整重跑」 |
| 看回测报告 | Tab4 区段 ⑥ 路径框 + 「🌐 打开回测报告」 |

---

*文档生成时间：基于当前 `m5_optimizer/app.py` 实际代码（Tab4 部分已重写完成并通过 `py_compile` 语法验证）*
