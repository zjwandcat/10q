# TTHH量化基金项目 · Claude交接文档 v6.0  
  
> 适用模型：Claude Sonnet 4.6  
> 最后更新：2026-05-14  
> 工作目录：E:\10q\10q-202604gpu    
> 启动入口：双击 启动M5优化器.bat → http://localhost:7860  
  
-----  
  
## 0. 你的角色（AA主管）  
  
你是本项目的AI架构师（AA），负责：  
  
- 审阅tthh（AI编码代理）提交的代码和报告  
- 生成精确的提示词交给用户转发给tthh执行  
- 根据M5 Trial报告做参数决策  
- 不直接运行代码，通过提示词驱动tthh工作  
  
用户工作流：AA生成提示词 → 用户发给tthh → tthh执行并返回结果 → 用户转发给AA分析  
  
-----  
  
## 1. 硬件环境  
  
- Win11 / Python 3.14 / ROG幻X13笔记本  
- 16GB RAM（系统常驻~8GB，Python可用~8GB）  
- AMD CPU 8核16线程  
- NVIDIA GTX 1650 4GB VRAM  
- NVMe SSD  
  
-----  
  
## 2. 全局不可修改规则（铁律）  
  
1. `run_m2` 第二参数必须是 `xgbm_params`，禁止写成 `xgb_params`  
1. M5的 `load_if_exists=True` 禁止修改  
1. M0/M1/M4主体逻辑禁止修改  
1. 禁止 `list(splitter.split())` 全量预加载windows（OOM）  
1. 单进程内存峰值不超过8GB  
1. 模型`__init__`必须先 `dict(params)` 浅拷贝再 `.pop()`  
1. 外层joblib必须用 `backend='threading'`，禁止 `loky`  
1. feature_store所有矩阵运算必须float32，禁止float64  
  
-----  
  
## 3. 模块状态总览  
  
|模块   |状态      |核心输出                                       |  
|-----|--------|-------------------------------------------|  
|M0   |✅完成（已复权）|data/pool_v2_scheme_[a/b/d/e]/ 各228个parquet|  
|M1   |✅完成     |output/m1_windows/ 180个滚动窗口                |  
|M2   |✅运行中    |ensemble+portfolio，已接入换手成本、复权收益            |  
|M3   |❌未开发    |熔断机制（熊市切换现金）                               |  
|M4   |✅完成     |output/backtest_report.html                |  
|M5   |✅运行中    |Gradio 4-Tab UI，正在跑4套scheme的P1             |  
|M6/实盘|❌未开发    |实盘综合程序                                     |  
  
-----  
  
## 4. 数据规范  
  
### 股票池筛选（不可更改）  
  
- 剔除ST/*ST/PT  
- 上市满252日  
- 换手率1~15%  
- 市值100~3000亿  
- 停牌≤5日  
- 股价≥2元  
  
### 四套中性化方案  
  
- scheme_b：Rank-Z+双重OLS  
- scheme_a：双重OLS正交化  
- scheme_d：仅行业OLS（对照组）  
- scheme_e：分层中性化  
  
### 数据关键修复（已完成）  
  
- ✅ Target_Return_1M已改用复权价格（adj_factor）  
- ✅ 日线数据已改后复权（hfq）  
- ✅ 月度超额收益截断±50%（clip）  
- ✅ M2 ensemble层额外截断±30%  
  
### 持仓权重规范（不可更改）  
  
- Top5（排名1-5）：每只13%，合计65%  
- Next5（排名6-10）：每只7%，合计35%  
- 月度满仓，is_penalized仅为置信度标记不减仓  
  
### 换手成本  
  
- 印花税0.1%（卖出）+ 佣金0.03%（双边）+ 滑点0.1%（双边）  
- 完全换仓成本约0.36%/月  
  
### 基准数据  
  
- 中证800（CSI 800）  
- 文件：data/000906perf.xlsx  
  
-----  
  
## 5. M2核心架构（已稳定）  
  
### 模型配置  
  
- LightGBM：objective=regression_l1（MAE）  
- XGBoost：objective=reg:absoluteerror（MAE）  
- 均已启用early_stopping（rounds=30~50）  
- max_bin=128（提速）  
- nthread=cpu_count-2（主进程独占）  
  
### FeatureStore流程  
  
1. 候选因子列确定  
1. drop_short_term_noise过滤（可选）  
1. 低覆盖过滤（min_valid_rate）  
1. 零方差列过滤（含宏观因子）  
1. 截面Z-score（np.add.at向量化）  
1. NaN填0  
1. 高相关去重（max_corr）  
1. IC筛选（min_ic_abs，min_keep_factors兜底）  
  
### 已计算的M5因变量  
  
|指标                      |含义        |方向  |  
|------------------------|----------|----|  
|val_ic                  |验证集IC均值   |越大越好|  
|val_icir                |IC信息比率    |越大越好|  
|val_rolling6m_ir        |滚动6月IR（年化）|越大越好|  
|val_rolling6m_excess_ann|滚动6月平均超额年化|越大越好|  
|pct_positive_excess     |月度超额胜率    |越大越好|  
|ir_worst_quartile       |最差25%期IR  |越大越好|  
|val_global_ir           |全局IR      |越大越好|  
|penalized_rate          |低置信度月比例   |越小越好|  
|ic_gap                  |训练-验证IC差  |越小越好|  
  
### 关键已修复Bug  
  
|Bug                       |位置               |状态                        |  
|--------------------------|-----------------|--------------------------|  
|penalized_rate键名不匹配       |objective.py L219|✅已修复→low_confidence_months|  
|rolling6m_ir未年化           |ensemble.py      |✅已修复→×√12                 |  
|val_global_ir硬编码0         |objective.py     |✅已修复                      |  
|val_icir数据流断裂             |run_m2.py        |✅已修复                      |  
|compute_val_metrics默认False|run_m2.py        |✅已修复→True                 |  
|Target_Return_1M未复权       |pipeline.py      |✅已修复                      |  
|XGBoost无效set_group        |xgb_model.py     |✅已修复                      |  
|exec(f”del {name}”)无效     |run_m2.py        |✅已修复                      |  
  
-----  
  
## 6. M5详细规格  
  
### 搜索空间（~30个参数）  
  
LightGBM（12）/ XGBoost（11）/ Ensemble（1）/ FeatureStore（5）  
  
### 已收窄的参数范围  
  
- max_depth：2（14B实验中收敛到2）  
- n_estimators：100~300  
- lgbm_reg_lambda：推荐5~10（高正则更好）  
- drop_short_term_noise：固定True  
- train_months：38~48  
  
### 目标函数权重（全自适应型预设）  
  
```  
val_icir:                 0.20  
ic_gap_penalty:           0.20  
val_rolling6m_excess_ann: 0.25  ← 主核心指标  
pct_positive_excess:      0.20  
val_rolling6m_ir:         0.05  
ir_worst_quartile:        0.05  
val_ic:                   0.03  
penalized_rate:           0.02  
```  
  
### 贝叶斯优化改进（已实施）  
  
- 对数变换：lgbm/xgb learning_rate、reg_lambda（log=True）  
- 事前约束：LR×n_est≤15  
- 热启动先验：从最优Trial的params注入  
  
### 并发配置  
  
```python  
GLOBAL_N_JOBS_OUTER   = 2  
GLOBAL_NTHREAD_INNER  = 2  
M5_NTHREAD_PER_MODEL  = cpu_count - 2  
DATA_LOADER_MAX_WORKERS = 8  
MEMORY_LIMIT_GB  = 9.5  
```  
  
-----  
  
## 7. 当前M5运行状态（2026-05-14）  
  
### 正在跑的4个项目  
  
|项目 |Scheme|Trial数|最优得分  |val_rolling6m_ir>0%|状态  |  
|---|------|------|------|-------------------|----|  
|14B|待确认   |168   |0.8204|99.4%              |✅优秀 |  
|14D|待确认   |105   |0.7384|54.3%              |🔄继续跑|  
|14A|待确认   |待报告   |-     |-                  |🔄运行中|  
|14E|待确认   |待报告   |-     |-                  |🔄运行中|  
  
### 14B关键发现（最优方案）  
  
- val_rolling6m_ir 均值0.1935，99.4%转正 ← 复权修复生效  
- pct_positive_excess均值50.3%（刚过50%门槛）  
- 参数已高度收敛：max_depth=2，drop_short_term_noise=True，train_months≈42  
  
### 14B TOP3热启动先验（Trial 122/121/123）  
  
```json  
Trial 122（score=0.8204）关键参数：  
  lgbm_learning_rate: 0.025, lgbm_n_estimators: 103  
  lgbm_max_depth: 2, lgbm_reg_lambda: 8.875  
  xgb_learning_rate: 0.053, xgb_n_estimators: 200  
  xgb_max_depth: 2, xgb_reg_lambda: 1.348  
  lgbm_weight: 0.699, min_ic_abs: 0.002  
  min_keep_factors: 112, max_corr: 0.908  
  drop_short_term_noise: true, train_months: 42  
```  
  
### 单Trial耗时  
  
- 当前：14-16分钟/Trial（正常，180窗口×478因子的合理结果）  
- 已确认不是代码效率问题，是数据规模合理耗时  
  
-----  
  
## 8. P1→P2决策流程  
  
### 何时进入P2  
  
每个scheme跑满150个Trial后：  
  
1. 用M5_P1_Reporter脚本生成分析报告  
1. 发给AA（Claude）分析  
1. AA判断是否收敛，给出P2范围  
  
### 报告生成方法  
  
使用以下脚本（已有skill文件）：  
  
```python  
# 运行：python generate_p1_report.py  
# 输出：output/{项目名}/p1/p1_analysis_report.md  
```  
  
### AA分析的关键维度  
  
1. val_rolling6m_ir >0占比是否>80%  
1. 参数收敛（TOP半区std/均值<30%且TOP-BOT差>0.3σ）  
1. 固定参数：收敛✅的参数在P2固定值  
1. 缩窄范围：TOP半区均值±1.5σ  
  
### 14B已可进P2的参数范围  
  
```  
lgbm_max_depth: 固定=2  
xgb_max_depth: 固定=2  
drop_short_term_noise: 固定=True  
train_months: 38~48  
min_keep_factors: 100~120  
max_corr: 0.87~0.93  
min_ic_abs: 0.001~0.007  
lgbm_reg_lambda: 5.0~10.0  
lgbm_n_estimators: 80~180  
lgbm_weight: 0.55~0.75  
xgb_reg_lambda: 0.8~2.5  
xgb_n_estimators: 160~240  
```  
  
-----  
  
## 9. 待完成任务清单  
  
### 紧急（当前进行中）  
  
- [ ] 14A/14E跑满150 Trial → 生成报告发给AA  
- [ ] 14D继续跑到150 Trial  
- [ ] 修复启动M5优化器.bat（见Section 10）  
  
### P1完成后  
  
- [ ] AA分析4套scheme报告 → 决定哪套进P2/重跑P1  
- [ ] 进入P2局部精调（缩窄搜索范围+热启动）  
- [ ] P2完成后提取最优参数组合  
  
### 后续大项目  
  
- [ ] **M3熔断机制**：熊市识别（2008/2015/2018/2022）→ 切换现金持仓  
- [ ] **实盘综合程序（M6）**：月度选股报告 + 调仓指令生成  
  
### 长期待优化  
  
- [ ] Target_Return_1M改用真正复权价（当前是adj_factor×close，已足够）  
- [ ] 分析师预期因子接入（get_analyst_forecast已实现但未调用）  
- [ ] 行业内rank因子批量生成  
- [ ] M0因子扩充至460+（当前478-489列）  
  
-----  
    
## 11. 日志系统（已规划，待实施）  
  
### 目标  
  
终端崩溃后发给AA最后1小时的log即可诊断原因  
  
### 已规划的rolling_logger.py  
  
- 位置：m5_optimizer/utils/rolling_logger.py  
- 每小时滚动，保留1个备份  
- 异步QueueHandler，不阻塞M2  
- 记录：TRIAL_START/TRIAL_END/WINDOW/SYS_STATS/ERROR  
  
### 崩溃诊断方法  
  
- 最后行是WINDOW → OOM或GPU崩溃  
- 最后行是TRIAL_START无END → 进程被强杀  
- sys_mem_avail < 0.5GB → OOM Killer触发  
- 无日志超30分钟 → Windows更新或死锁  
  
-----  
  
## 12. M3熔断机制（未开发，设计构想）  
  
### 核心逻辑  
  
- 识别熊市信号（中证800连续N月跌幅超X%）  
- 触发熔断 → 组合100%切换现金  
- 熔断解除条件 → 恢复正常选股持仓  
   
  
### 与M2的接口  
  
- M3在portfolio_builder.py之后介入  
- 修改月度持仓：熊市月→现金，非熊市月→M2选股结果  
  
-----  
  
## 13. 实盘程序（M6，未开发，设计构想）  
  
### 功能需求  
  
1. 月度运行（每月最后一个交易日）  
1. 读取最新行情数据（Tushare）  
1. 调用M2最优参数模型打分  
1. 输出：当月推荐持仓10只股票 + 权重  
1. 对比上月持仓 → 生成调仓指令  
1. 生成月报PDF/HTML  
  
### 技术依赖  
  
- 依赖M5 P2完成后的最优参数  
- 数据源：Tushare Pro API（已有token）  
- 报告格式：参考M4的report_builder.py  
  
-----  
  
## 14. M5 Tab2 已知问题与修复状态  
  
### 初始化滑块0匹配Bug  
  
**根因**：filter_conditions包含Trial user_attrs中不存在的指标键（如新加的pct_positive_excess等），count_matched遇到None返回matched=False排除所有Trial。  
  
**修复方案（已提供提示词，待tthh实施）**：  
  
1. range_analyzer.py：`if value is None: continue`（改为skip而非排除）  
1. app.py：filter_conditions只包含existing_metrics中存在的键  
  
-----  
  
## 15. 常见陷阱备忘  
  
1. `xgbm_params`写成`xgb_params` → run_m2静默用默认参数  
1. `list(splitter.split())`预加载 → OOM  
1. `compute_val_metrics=False` → 所有IR指标全为0  
1. categorical参数astype(float) → range_analyzer崩溃  
1. slider value超出minimum/maximum → Gradio TypeError  
1. `_stop_now_event.clear()`在定时器后执行 → 清除定时器信号  
1. `val_global_ir`硬编码0.0 → Tab2滑块无效（已修复）  
1. `penalized_months`键名 → 应为`low_confidence_months`（已修复）  
1. HARD_CONSTRAINTS如含objective会覆盖用户传入值（需检查）  
1. 终端莫名关闭 → Windows更新/OOM，日志系统待部署  
  
-----  
  
## 16. 向新AA说明的工作节奏  
  
当用户带来M5 P1报告时，你需要：  
  
**分析维度**（按此顺序）：  
  
1. `val_rolling6m_ir >0占比` → <50%需继续跑或重设自变量范围  
1. `pct_positive_excess` → 是否过50%  
1. `参数收敛分析` → ✅标记的参数可进P2固定  
1. `TOP均值±标准差` → P2搜索范围  
1. `14B对比` → 14B是当前最优基准（score 0.82，IR转正99.4%）  
  
**决策输出**：  
  
- 继续跑P1：Trial数不足或指标未收敛  
- 重跑P1（调参）：指标持续差或参数分布异常  
- 进入P2：参数收敛，指标良好  
  
-----  
  
*文档生成时间：2026-05-14 | 版本：v6.0 | 用于新Claude会话交接*  
