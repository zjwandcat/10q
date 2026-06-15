【批次1】search\_space.py

你是TTHH量化系统M5构建Agent。  
工作目录：quant\_fund/  
本批任务：生成 m5\_optimizer/search\_space.py

硬约束（全项目通用，每批均需遵守）：

* run\_m2第二参数永远是xgbm\_params
* load\_if\_exists=True禁止修改
* 禁止list(splitter.split())预加载
* 内存峰值<8GB，joblib必须backend='threading'
* 模型\_\_init\_\_必须dict()浅拷贝params再.pop()

文件职责：定义29个搜索参数和30个因变量。

ALL\_PARAMS结构：  
{  
"参数名": {  
"type": "float\_log"|"float"|"int"|"categorical",  
"low":..., "high":...,   # float/int/float\_log用  
"choices":\[...],          # categorical用  
"default":...,  
"group":"lgbm"|"xgb"|"ensemble"|"feature"  
}  
}

完整29个参数：

LightGBM（12个）：  
lgbm\_learning\_rate: float\_log, 0.01\~0.1, default=0.05  
lgbm\_n\_estimators: int, 100\~400, default=200  
lgbm\_max\_depth: int, 2\~8, default=4  
lgbm\_colsample\_bytree: float, 0.1\~0.4, default=0.3  
lgbm\_reg\_alpha: float\_log, 0.01\~3.0, default=0.1  
lgbm\_reg\_lambda: float\_log, 0.1\~10.0, default=1.0  
lgbm\_min\_split\_gain: float\_log, 0.001\~0.1, default=0.01  
lgbm\_lr\_mode: categorical, \[fixed,decay], default=fixed  
lgbm\_decay\_every: int, 30\~80, default=50  
lgbm\_decay\_factor: float, 0.6\~0.92, default=0.8  
lgbm\_depth\_mode: categorical, \[fixed,adaptive], default=fixed  
lgbm\_early\_stopping\_rounds: int, 20\~80, default=30

XGBoost（11个）：  
xgb\_learning\_rate: float\_log, 0.01\~0.1, default=0.05  
xgb\_n\_estimators: int, 100\~400, default=200  
xgb\_max\_depth: int, 2\~5, default=4  
xgb\_colsample\_bytree: float, 0.1\~0.4, default=0.3  
xgb\_reg\_alpha: float\_log, 0.01\~3.0, default=0.1  
xgb\_reg\_lambda: float\_log, 0.1\~10.0, default=1.0  
xgb\_gamma: float\_log, 0.001\~0.2, default=0.01  
xgb\_lr\_mode: categorical, \[fixed,decay], default=fixed  
xgb\_decay\_every: int, 30\~80, default=50  
xgb\_decay\_factor: float, 0.6\~0.92, default=0.8  
xgb\_early\_stopping\_rounds: int, 20\~80, default=30

Ensemble（1个）：  
lgbm\_weight: float, 0.3\~0.7, default=0.5

FeatureStore（5个）：  
min\_valid\_rate: float, 0.20\~0.50, default=0.30  
max\_corr: float, 0.80\~0.97, default=0.95  
min\_ic\_abs: float\_log, 0.002\~0.02, default=0.005  
min\_keep\_factors: int, 30\~120, default=50  
drop\_short\_term\_noise: categorical, \[True,False], default=False

OBJECTIVE\_VARS列表结构：  
\[{"name":..., "source":"m2"|"m4", "direction":"max"|"min", "enabled":True/False}]

m2来源13个（enabled=True）：  
val\_ic(max), val\_icir(max), val\_rolling6m\_ir(max), val\_rolling6m\_dir(max),  
val\_rolling6m\_sortino(max), val\_rolling6m\_return(max),  
ic\_gap\_penalty(min), penalized\_rate(min), val\_global\_ir(max),  
val\_annual\_return(max)

m2来源stress 3个（enabled=False）：  
stress\_2008\_excess(max), stress\_2015\_mdd(min), stress\_2022\_excess(max)

m4来源17个（enabled=True）：  
cagr(max), monthly\_win\_rate(max), downside\_volatility(min),  
upside\_volatility(max), volatility\_ratio(max), var\_95(min), cvar\_95(min),  
skewness(max), kurtosis(min), pain\_index(min), ulcer\_index(min),  
omega\_ratio(max), sterling\_ratio(max), burke\_ratio(max), martin\_ratio(max),  
tail\_ratio(max), up\_capture\_ratio(max), down\_capture\_ratio(min), capture\_ratio(max)

参数透传注释（写在参数定义旁）：  
lgbm\_lr\_mode→lgbm\_params\["lr\_mode"]  
lgbm\_decay\_every→lgbm\_params\["decay\_every"]  
lgbm\_decay\_factor→lgbm\_params\["decay\_factor"]  
lgbm\_depth\_mode→lgbm\_params\["depth\_mode"]  
lgbm\_early\_stopping\_rounds→lgbm\_params\["early\_stopping\_rounds"]  
xgb\_\*同理→xgbm\_params\["..."]（去掉xgb\_前缀）

导出：ALL\_PARAMS, OBJECTIVE\_VARS, DEFAULT\_PARAMS

末尾必须执行：  
assert len(ALL\_PARAMS)==29, f"参数数量错误:{len(ALL\_PARAMS)}"  
assert len(OBJECTIVE\_VARS)==30, f"因变量数量错误:{len(OBJECTIVE\_VARS)}"  
print("search\_space验证通过")

输出完整search\_space.py，末尾执行验证。



【批次2】objective.py

你是TTHH量化系统M5构建Agent。  
工作目录：quant\_fund/  
本批任务：生成 m5\_optimizer/objective.py  
前置：search\_space.py已完成。

硬约束：

* run\_m2第二参数永远是xgbm\_params（禁止写xgb\_params）
* 禁止list(splitter.split())预加载
* load\_if\_exists=True禁止修改
* 内存峰值<8GB

文件职责：Optuna Trial目标函数。

关键常量：  
IC\_GAP\_PENALTY\_MULTIPLIER = 1.5  # ★ ic\_gap\_penalty额外×1.5

模块级缓存：  
\_GLOBAL\_FACTOR\_DF = None  # ★ 避免每Trial重载

class ObjectiveFunction：  
def **init**(self,  
preloaded\_factor\_df,  
preloaded\_windows=None,  # ★ 永远传None，惰性切片  
window\_count=60,  
compute\_val\_metrics=True,  # ★ 必须True  
objective\_weights=None,  
active\_params=None,  # None=全部29个  
scheme="scheme\_b",  
fast\_mode=True,  
stop\_event=None,  
):

def **call**(self, trial) -> float:  
try:  
# 1. 采样参数（active\_params中的用suggest，其余用default）  
# 2. 组装lgbm\_params/xgbm\_params/feature\_params  
# 3. 调用run\_m2  
# 4. 提取指标，计算score  
# 5. 所有因变量写入trial.set\_user\_attr  
# 6. 返回-score（optuna minimize）  
except Exception as e:  
logger.error(f"Trial异常: {e}\\n{traceback.format\_exc()}")  
return -999.0  # ★ 异常返回-999，不抛出

参数采样逻辑（按type）：  
float\_log → trial.suggest\_float(name, low, high, log=True)  
float → trial.suggest\_float(name, low, high)  
int → trial.suggest\_int(name, low, high)  
categorical → trial.suggest\_categorical(name, choices)  
非active参数 → 直接用DEFAULT\_PARAMS\[name]

参数分组组装：

* group=="lgbm"：去掉"lgbm\_"前缀放入lgbm\_params  
注意special keys直接透传：lr\_mode/decay\_every/decay\_factor/depth\_mode/early\_stopping\_rounds
* group=="xgb"：去掉"xgb\_"前缀放入xgbm\_params（同上）
* group=="ensemble"：lgbm\_weight单独提取
* group=="feature"：组成feature\_params字典

run\_m2调用（必须完整）：  
from m2\_engine.run\_m2 import run\_m2  
portfolios, stats = run\_m2(  
lgbm\_params=lgbm\_params,  
xgbm\_params=xgbm\_params,      # ★ 必须xgbm\_params  
feature\_params=feature\_params,  
lgbm\_weight=lgbm\_weight,  
fast\_mode=self.fast\_mode,  
fast\_window\_count=self.window\_count,  
compute\_val\_metrics=True,      # ★  
compute\_shap=False,  
verbose=False,  
preloaded\_windows=None,        # ★ 惰性  
preloaded\_factor\_df=self.factor\_df,  
gpu\_mode=False,  
stop\_event=self.stop\_event,  
)

加权得分计算：  
score = 0.0  
for metric, weight in self.objective\_weights.items():  
value = float(stats.get(metric, 0) or 0)  
multiplier = IC\_GAP\_PENALTY\_MULTIPLIER if metric=="ic\_gap\_penalty" else 1.0  
direction = {v\["name"]:v\["direction"] for v in OBJECTIVE\_VARS}\[metric]  
if direction == "max":  
score += weight \* multiplier \* value  
else:  
score -= weight \* multiplier \* value  
return -score  # optuna minimize

内存检查（超限只警告不中断）：  
import psutil  
mem\_gb = psutil.Process().memory\_info().rss / 1e9  
if mem\_gb > 9.5:  
logger.warning(f"内存超限:{mem\_gb:.1f}GB，Trial继续")

输出完整objective.py。



【批次3】phase1\_global.py + phase2\_local.py

你是TTHH量化系统M5构建Agent。  
工作目录：quant\_fund/  
本批任务：生成 m5\_optimizer/phase1\_global.py 和 phase2\_local.py  
前置：search\_space.py、objective.py已完成。

硬约束：

* load\_if\_exists=True禁止修改
* preloaded\_windows=None（惰性切片）
* compute\_val\_metrics=True
* run\_m2第二参数永远是xgbm\_params

━━━ phase1\_global.py ━━━

职责：TPE全局探索，断点续跑，支持双停止机制。  
study存储：output/m5/phase1.db

函数签名：  
def run\_phase1(  
factor\_df,  
n\_trials: int = 50,  
fast\_mode: bool = True,  
window\_count: int = 60,  
objective\_weights: dict = None,  
active\_params: list = None,  
scheme: str = "scheme\_b",  
stop\_now\_event = None,    # ⚡立即停止Event  
stop\_graceful\_event = None,  # 🏁优雅停止Event  
progress\_callback = None,  # callable(trial\_num, total, best\_score, user\_attrs)  
) -> optuna.Study:

核心实现：  
import os, optuna  
os.makedirs("output/m5", exist\_ok=True)

study = optuna.create\_study(  
study\_name="phase1\_global",  
storage="sqlite:///output/m5/phase1.db",  
load\_if\_exists=True,   # ★ 禁止修改  
direction="minimize",  
sampler=optuna.samplers.TPESampler(seed=42),  
)

objective = ObjectiveFunction(  
preloaded\_factor\_df=factor\_df,  
preloaded\_windows=None,    # ★  
window\_count=window\_count,  
compute\_val\_metrics=True,  # ★  
objective\_weights=objective\_weights,  
active\_params=active\_params,  
scheme=scheme,  
fast\_mode=fast\_mode,  
stop\_event=stop\_now\_event,  
)

def trial\_callback(study, trial):  
if stop\_graceful\_event and stop\_graceful\_event.is\_set():  
study.stop()  
return  
if progress\_callback:  
attrs = trial.user\_attrs if trial.state.is\_finished() else {}  
progress\_callback(  
len(study.trials), n\_trials,  
study.best\_value if study.best\_trial else None,  
attrs  
)

study.optimize(  
objective,  
n\_trials=n\_trials,  
callbacks=\[trial\_callback],  
catch=(Exception,),  # ★ 捕获异常，Trial标为FAIL不中断  
)  
return study

━━━ phase2\_local.py ━━━

职责：局部精化，从phase1最优参数出发，独立study。

函数签名：  
def run\_phase2(  
factor\_df,  
n\_trials: int = 30,  
fast\_mode: bool = False,   # Phase2推荐全量  
window\_count: int = 173,  
objective\_weights: dict = None,  
active\_params: list = None,  
init\_params: dict = None,  # phase1最优参数，作为起点  
param\_ranges: dict = None, # Tab2反推的范围，覆盖默认搜索范围  
scheme: str = "scheme\_b",  
stop\_now\_event = None,  
stop\_graceful\_event = None,  
progress\_callback = None,  
) -> optuna.Study:

核心实现：  
study = optuna.create\_study(  
study\_name="phase2\_local",  
storage="sqlite:///output/m5/phase2.db",  
load\_if\_exists=True,   # ★  
direction="minimize",  
sampler=optuna.samplers.TPESampler(seed=42, n\_startup\_trials=5),  
)

# 若有初始参数，入队作为第一个Trial

if init\_params:  
study.enqueue\_trial(init\_params)

# 若有param\_ranges，创建带缩小范围的ObjectiveFunction子类

# 通过覆盖ALL\_PARAMS的low/high实现（传入objective的custom\_ranges参数）

objective = ObjectiveFunction(  
preloaded\_factor\_df=factor\_df,  
preloaded\_windows=None,  
window\_count=window\_count,  
compute\_val\_metrics=True,  
objective\_weights=objective\_weights,  
active\_params=active\_params,  
scheme=scheme,  
fast\_mode=fast\_mode,  
stop\_event=stop\_now\_event,  
# custom\_ranges=param\_ranges,  # ObjectiveFunction需支持此参数  
)

# trial\_callback同phase1

study.optimize(objective, n\_trials=n\_trials, callbacks=\[trial\_callback], catch=(Exception,))  
return study

注意：ObjectiveFunction需增加custom\_ranges参数支持，用于覆盖suggest的low/high范围。

输出两个完整文件。



【批次4】range\_analyzer.py + result\_analyzer.py + config\_manager.py

你是TTHH量化系统M5构建Agent。  
工作目录：quant\_fund/  
本批任务：生成以下3个文件：

* m5\_optimizer/range\_analyzer.py
* m5\_optimizer/result\_analyzer.py
* m5\_optimizer/config\_manager.py

硬约束：

* run\_m2第二参数永远是xgbm\_params
* 内存峰值<8GB

━━━ range\_analyzer.py ━━━

职责：从study的Trial中，按因变量过滤条件反推参数范围。

关键规则：

1. ★ 跳过categorical参数（无法astype float）
2. ★ 整数参数：p\_low=int(floor(percentile(vals,pct\_low)))，p\_high=int(ceil(percentile(vals,pct\_high)))

函数：  
def analyze\_ranges(  
study: "optuna.Study",  
filter\_conditions: dict,  # {metric\_name: (min\_val, max\_val)}，-999表示不限  
percentile\_low: float = 10,  
percentile\_high: float = 90,  
) -> dict:  
# 返回：  
# {  
#   "n\_matched": int,  
#   "n\_total": int,  
#   "ranges": {param\_name: {"low":..., "high":..., "type":..., "default":...}}  
# }

逻辑：

1. 遍历study.trials，state==COMPLETE的才参与
2. 按filter\_conditions过滤：trial.user\_attrs\[metric]在(min\_val,max\_val)内
3. 收集满足条件trial的params
4. 对每个非categorical参数计算percentile范围
5. 整数参数取整，float保留4位小数

def count\_matched(study, filter\_conditions) -> tuple\[int,int]:  
# 快速统计满足条件的trial数，供Tab2实时更新

━━━ result\_analyzer.py ━━━

职责：读最优Trial，触发M2+M4完整重跑。

from m2\_engine.run\_m2 import run\_m2  
from m4\_report.report\_generator import generate\_report  # 按实际路径调整

def get\_best\_params(study) -> dict:  
return dict(study.best\_trial.params)

def get\_best\_study(phase2\_path="output/m5/phase2.db",  
phase1\_path="output/m5/phase1.db") -> "optuna.Study":  
# 优先读phase2，不存在则读phase1  
import os, optuna  
for path, name in \[(phase2\_path,"phase2\_local"),(phase1\_path,"phase1\_global")]:  
if os.path.exists(path):  
return optuna.load\_study(study\_name=name, storage=f"sqlite:///{path}")  
raise FileNotFoundError("未找到任何study数据库")

def run\_full\_backtest(  
best\_params: dict,  
scheme: str = "scheme\_b",  
include\_no\_penalty: bool = True,  
progress\_callback=None,  
) -> dict:  
# 1. 从best\_params分解lgbm\_params/xgbm\_params/feature\_params（同objective.py逻辑）  
# 2. run\_m2全量（fast\_mode=False）  
# 3. 生成M4报告  
# 4. 若include\_no\_penalty，额外跑关闭降权的对比版本  
# 返回{"report\_path":str, "metrics":dict, "compare\_report\_path":str}

━━━ config\_manager.py ━━━

import yaml, json, os, shutil

def load\_config(path="config/config.yaml") -> dict:  
with open(path, encoding="utf-8") as f:  
return yaml.safe\_load(f)

def save\_config(config:dict, path="config/config.yaml"):  
# 先备份原文件（.bak）  
shutil.copy(path, path+".bak")  
with open(path, "w", encoding="utf-8") as f:  
yaml.dump(config, f, allow\_unicode=True, default\_flow\_style=False)

def write\_best\_params\_to\_config(best\_params:dict, path="config/config.yaml"):  
# 读取现有config，更新lgbm/xgb/feature节，保留其他节不变  
config = load\_config(path)  
# 按group分配（逻辑同objective.py的参数分组）  
# config\["lgbm"] = lgbm\_params  
# config\["xgb"] = xgbm\_params  （key用xgbm以匹配run\_m2）  
# config\["feature"] = feature\_params  
save\_config(config, path)

def save\_ranges\_json(ranges:dict, path="output/m5/p2\_ranges.json"):  
os.makedirs(os.path.dirname(path), exist\_ok=True)  
with open(path, "w", encoding="utf-8") as f:  
json.dump(ranges, f, ensure\_ascii=False, indent=2)

def load\_ranges\_json(path="output/m5/p2\_ranges.json") -> dict:  
with open(path, encoding="utf-8") as f:  
return json.load(f)

输出3个完整文件。



【批次5】app.py Tab1 + Tab2

你是TTHH量化系统M5构建Agent。  
工作目录：quant\_fund/  
本批任务：生成 m5\_optimizer/app.py 的 Tab1 和 Tab2 部分  
前置：search\_space/objective/phase1/phase2/range\_analyzer/result\_analyzer/config\_manager 均已完成。

框架：Gradio 4.x，gr.Blocks()

全局状态（模块级）：  
import threading  
\_stop\_now\_event = threading.Event()  
\_stop\_graceful = threading.Event()  
\_phase1\_thread = None  
\_factor\_df = None  # ★ 全局缓存factor\_df

预设模板（4套，用于按钮快速填充）：  
PRESET\_TEMPLATES = {  
"稳健型": {  
"weights":{"val\_icir":0.20,"ic\_gap\_penalty":0.25,"penalized\_rate":0.15,"val\_ic":0.15,"val\_rolling6m\_sortino":0.10},  
"active":\["lgbm\_max\_depth","xgb\_max\_depth","lgbm\_colsample\_bytree","xgb\_colsample\_bytree","lgbm\_reg\_lambda","xgb\_reg\_lambda","min\_keep\_factors"]  
},  
"激进型": {  
"weights":{"val\_ic":0.30,"val\_rolling6m\_ir":0.25,"val\_rolling6m\_return":0.25,"ic\_gap\_penalty":0.10,"penalized\_rate":0.10},  
"active":\["lgbm\_learning\_rate","lgbm\_n\_estimators","xgb\_learning\_rate","xgb\_n\_estimators","lgbm\_weight","min\_ic\_abs"]  
},  
"防御型": {  
"weights":{"ic\_gap\_penalty":0.15,"val\_icir":0.15},  
"active":\["lgbm\_max\_depth","xgb\_max\_depth","lgbm\_colsample\_bytree","xgb\_colsample\_bytree","lgbm\_reg\_alpha","lgbm\_reg\_lambda","drop\_short\_term\_noise"]  
},  
"全自适应型": {  
"weights":{"val\_icir":0.25,"ic\_gap\_penalty":0.25,"penalized\_rate":0.20,"val\_rolling6m\_ir":0.20,"val\_ic":0.10},  
"active":"ALL\_29"  
}  
}

━━━ Tab1：Phase1全局探索 ━━━

with gr.Tab("Phase1 全局探索"):

# 1\. 预设按钮行

with gr.Row():  
btn\_stable = gr.Button("稳健型")  
btn\_aggressive = gr.Button("激进型")  
btn\_defensive = gr.Button("防御型")  
btn\_adaptive = gr.Button("全自适应型")

# 2\. 中性化方案

scheme\_radio = gr.Radio(  
choices=\[("方案B：Rank-Z+双重OLS（推荐）","scheme\_b"),  
("方案A：双重OLS正交化","scheme\_a"),  
("方案E：分层中性化","scheme\_e"),  
("方案D：仅行业OLS（对照组）","scheme\_d")],  
value="scheme\_b", label="中性化方案"  
)

# 3\. 历史Trial橙色警告（启动时检测phase1.db）

history\_warn = gr.Markdown("", elem\_id="warn\_box")

# CSS：#warn\_box {background:#fff3cd;border-left:4px solid #ff9800;padding:8px;display:none}

# 若phase1.db存在且trials>0，设display:block并填充文字

# 4\. 参数配置（Accordion）

with gr.Accordion("参数搜索范围（勾选=激活）", open=True):  
param\_widgets = {}  # {name: {"checkbox":..., "low":..., "high":..., "choices":...}}  
for name, pdef in ALL\_PARAMS.items():  
with gr.Row():  
cb = gr.Checkbox(label=name, value=True)  
if pdef\["type"] == "categorical":  
sel = gr.CheckboxGroup(choices=pdef\["choices"], value=pdef\["choices"], label="可选值", visible=True)  
param\_widgets\[name] = {"checkbox":cb, "choices":sel}  
else:  
low\_box = gr.Number(label="下限", value=pdef\["low"], visible=True)  
high\_box = gr.Number(label="上限", value=pdef\["high"], visible=True)  
param\_widgets\[name] = {"checkbox":cb, "low":low\_box, "high":high\_box}  
cb.change(fn=lambda v,lo=low\_box if pdef\["type"]!="categorical" else sel: gr.update(visible=v),  
inputs=cb, outputs=lo)

# 5\. 因变量权重（Accordion，只渲染enabled=True且非stress的）

with gr.Accordion("目标函数权重（合计应=1）", open=True):  
weight\_sliders = {}  
for var in OBJECTIVE\_VARS:  
if var\["enabled"] and not var\["name"].startswith("stress\_"):  
weight\_sliders\[var\["name"]] = gr.Slider(0, 1, step=0.01, label=var\["name"], value=0)  
weight\_sum\_md = gr.Markdown("权重合计：0.00")  
# 任意滑块变化时更新合计，合计≠1显示红色

# 6\. 运行配置

with gr.Row():  
n\_trials\_slider = gr.Slider(10, 200, step=10, value=50, label="Trial数")  
fast\_mode\_cb = gr.Checkbox(label="快速模式(60窗口)", value=True)

# 7\. 操作按钮

with gr.Row():  
btn\_start\_p1 = gr.Button("▶ 开始Phase1", variant="primary")  
btn\_stop\_now = gr.Button("⚡ 立即停止", variant="stop")  
btn\_stop\_graceful = gr.Button("🏁 优雅停止")

# 8\. 进度

log\_box\_p1 = gr.Textbox(label="运行日志", lines=8, autoscroll=True)  
with gr.Row():  
cur\_trial\_num = gr.Number(label="当前Trial", value=0)  
best\_score\_num = gr.Number(label="最优得分", value=0)

# 事件绑定：

# btn\_start\_p1.click → 新线程run\_phase1，progress\_callback更新log\_box/cur\_trial\_num/best\_score\_num

# btn\_stop\_now.click → \_stop\_now\_event.set()

# btn\_stop\_graceful.click → \_stop\_graceful.set()

# 预设按钮 → 填充weight\_sliders和param\_widgets的激活状态

━━━ Tab2：结果分析与反推 ━━━

with gr.Tab("结果分析与反推"):

gr.Markdown("基于Phase1结果，按因变量过滤Trial，反推参数搜索范围。")

# 因变量过滤滑块（m2来源，enabled=True，非stress）

filter\_sliders = {}  
for var in OBJECTIVE\_VARS:  
if var\["source"]=="m2" and var\["enabled"] and not var\["name"].startswith("stress\_"):  
# ★ 双端滑块，默认-2到2  
filter\_sliders\[var\["name"]] = gr.Slider(-2, 2, step=0.01,  
value=\[-2,2], label=var\["name"])  # gr.Slider with range

# 实时Trial计数

matched\_md = gr.Markdown("满足条件Trial：- / -")

# 分位数设置

with gr.Row():  
pct\_low = gr.Slider(5, 30, value=10, step=1, label="下分位数%")  
pct\_high = gr.Slider(70, 95, value=90, step=1, label="上分位数%")

# 操作

with gr.Row():  
btn\_analyze = gr.Button("📊 分析参数范围")  
btn\_export = gr.Button("💾 导出p2\_ranges.json")

range\_df = gr.DataFrame(label="反推参数范围（整数已取整）")  
status\_box = gr.Textbox(label="状态")

# 事件绑定：

# 每个filter\_slider.change → count\_matched → 更新matched\_md

# btn\_analyze.click → analyze\_ranges → 更新range\_df

# btn\_export.click → save\_ranges\_json → 更新status\_box

输出app.py中Tab1和Tab2的完整代码块（含import和全局变量，不含Tab3/Tab4）。  
下一批继续Tab3/Tab4。



【批次6】app.py Tab3+Tab4 + utils + 启动文件

你是TTHH量化系统M5构建Agent。  
工作目录：quant\_fund/  
本批任务：

1. 续写app.py的Tab3和Tab4（与上一批Tab1/Tab2代码合并）
2. 生成 m5\_optimizer/utils/logger.py
3. 生成 m5\_optimizer/utils/memory\_monitor.py
4. 生成 启动M5优化器.bat
5. 生成 m5\_optimizer/**init**.py（空文件或基本导入）

━━━ Tab3：Phase2独立精调 ━━━

with gr.Tab("Phase2 独立精调"):

gr.Markdown("建议先完成Tab2并导出p2\_ranges.json，再执行Phase2精调。")

# 导入反推范围

btn\_import\_ranges = gr.Button("📂 从p2\_ranges.json导入范围")  
import\_status = gr.Textbox(label="导入状态", interactive=False)

# 参数范围配置（导入后自动填充）

with gr.Accordion("Phase2搜索范围", open=True):  
p2\_param\_widgets = {}  # 结构同Tab1 param\_widgets  
for name, pdef in ALL\_PARAMS.items():  
if pdef\["type"] != "categorical":  
with gr.Row():  
lo2 = gr.Number(label=f"{name} 下限", value=pdef\["low"])  
hi2 = gr.Number(label=f"{name} 上限", value=pdef\["high"])  
p2\_param\_widgets\[name] = {"low":lo2, "high":hi2}

# 独立权重（同Tab1权重区）

with gr.Accordion("Phase2目标权重", open=False):  
p2\_weight\_sliders = {}  
for var in OBJECTIVE\_VARS:  
if var\["enabled"] and not var\["name"].startswith("stress\_"):  
p2\_weight\_sliders\[var\["name"]] = gr.Slider(0,1,step=0.01,label=var\["name"],value=0)

# 运行配置

with gr.Row():  
p2\_trials = gr.Slider(5, 100, step=5, value=30, label="Trial数")  
p2\_fast\_cb = gr.Checkbox(label="快速模式", value=False)  
p2\_windows = gr.Number(label="窗口数", value=173)

p2\_from\_best = gr.Checkbox(label="以Phase1最优参数为起点", value=True)

with gr.Row():  
btn\_start\_p2 = gr.Button("▶ 开始Phase2", variant="primary")  
btn\_stop\_p2\_now = gr.Button("⚡ 立即停止", variant="stop")  
btn\_stop\_p2\_graceful = gr.Button("🏁 优雅停止")

log\_box\_p2 = gr.Textbox(label="运行日志", lines=8, autoscroll=True)  
with gr.Row():  
p2\_trial\_num = gr.Number(label="当前Trial", value=0)  
p2\_best\_score = gr.Number(label="最优得分", value=0)

# 事件绑定：

# btn\_import\_ranges.click → load\_ranges\_json → 填充p2\_param\_widgets

# btn\_start\_p2.click → 新线程run\_phase2（init\_params从phase1.db读取）

# 停止按钮同Tab1逻辑

━━━ Tab4：结果与部署 ━━━

with gr.Tab("结果与部署"):

# 数据库状态

db\_status = gr.Markdown("数据源：检测中...")

# 启动时检测phase2.db/phase1.db，更新db\_status

btn\_load\_best = gr.Button("📋 读取最优参数")  
best\_params\_df = gr.DataFrame(label="最优参数（点击读取后显示）")

with gr.Row():  
btn\_write\_config = gr.Button("💾 写回config.yaml", variant="primary")  
btn\_run\_full = gr.Button("🔄 触发M2+M4完整重跑", variant="primary")  
no\_penalty\_cb = gr.Checkbox(label="同时运行无降权对比", value=True)

log\_box\_deploy = gr.Textbox(label="执行日志", lines=10, autoscroll=True)  
report\_path\_box = gr.Textbox(label="报告路径", interactive=False)  
btn\_open\_report = gr.Button("🌐 打开回测报告")

# 事件绑定：

# btn\_load\_best.click → get\_best\_study → get\_best\_params → 更新best\_params\_df

# btn\_write\_config.click → write\_best\_params\_to\_config → 更新log\_box\_deploy

# btn\_run\_full.click → 新线程run\_full\_backtest → 实时更新log\_box\_deploy/report\_path\_box

# btn\_open\_report.click → os.startfile(report\_path) 或 webbrowser.open

━━━ app.py 末尾 ━━━

def build\_app():  
with gr.Blocks(title="TTHH M5优化器", theme=gr.themes.Soft()) as demo:  
gr.Markdown("# TTHH量化系统 · M5贝叶斯超参数优化器")  
# Tab1 Tab2 Tab3 Tab4...  
return demo

if **name** == "**main**":  
import yaml  
cfg = yaml.safe\_load(open("config/config.yaml",encoding="utf-8"))  
app = build\_app()  
app.launch(  
server\_name=cfg.get("m5",{}).get("gradio",{}).get("server\_name","0.0.0.0"),  
server\_port=cfg.get("m5",{}).get("gradio",{}).get("server\_port",7860),  
)

━━━ utils/logger.py ━━━

import logging, os  
def get\_logger(name="m5") -> logging.Logger:  
logger = logging.getLogger(name)  
if not logger.handlers:  
os.makedirs("logs/m5", exist\_ok=True)  
fh = logging.FileHandler("logs/m5/m5.log", encoding="utf-8")  
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))  
logger.addHandler(fh)  
sh = logging.StreamHandler()  
sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))  
logger.addHandler(sh)  
logger.setLevel(logging.INFO)  
return logger

━━━ utils/memory\_monitor.py ━━━

import psutil, os  
def get\_memory\_gb() -> float:  
return psutil.Process(os.getpid()).memory\_info().rss / 1e9

def check\_memory(limit\_gb=9.5, logger=None) -> bool:  
mem = get\_memory\_gb()  
if mem > limit\_gb:  
if logger: logger.warning(f"内存超限:{mem:.2f}GB>{limit\_gb}GB")  
return False  
return True

━━━ 启动M5优化器.bat ━━━

@echo off  
chcp 65001 >nul  
cd /d %\~dp0  
cd ..  
echo 正在启动TTHH M5优化器...  
python -m m5\_optimizer.app  
if %errorlevel% neq 0 (  
echo 启动失败，请检查依赖  
pause  
)

━━━ 最终验证（全部文件完成后执行）━━━

python -c "  
from m5\_optimizer.search\_space import ALL\_PARAMS, OBJECTIVE\_VARS  
from m5\_optimizer.objective import ObjectiveFunction, IC\_GAP\_PENALTY\_MULTIPLIER  
from m5\_optimizer.phase1\_global import run\_phase1  
from m5\_optimizer.phase2\_local import run\_phase2  
from m5\_optimizer.range\_analyzer import analyze\_ranges  
from m5\_optimizer.result\_analyzer import get\_best\_study  
from m5\_optimizer.config\_manager import load\_config  
assert len(ALL\_PARAMS)==29  
assert len(OBJECTIVE\_VARS)==30  
assert IC\_GAP\_PENALTY\_MULTIPLIER==1.5  
print('所有模块导入验证通过')  
print('参数数量:',len(ALL\_PARAMS))  
print('因变量数量:',len(OBJECTIVE\_VARS))  
"

输出所有文件完整代码。



使用顺序提示： 批次1→2→3→4→5→6，每批发送前确认上一批验证通过。批次3中的 custom\_ranges 参数需在批次2的 objective.py 中补充支持，可在发送批次3时附带说明让TTHH回头补丁。​​​​​​​​​​​​​​​​

