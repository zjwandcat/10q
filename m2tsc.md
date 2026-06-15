M2完整任务提示词  
【项目规则 - 必读】  
项目：TTHH A股量化选股系统 / 工作目录：quant\_fund/  
Python 3.11 / Windows / RTX 1650 4GB + 16GB RAM  
这是全新构建M2，不改动M0/M1任何文件  
参数命名硬约束：run\_m2第二参数必须是xgbm\_params（不是xgb\_params）  
修改前先输出目录结构确认 / 只做任务要求的内容  
有验证脚本必须实际执行并回传结果

M2 选股算法引擎 · 全新构建任务 v3.0  
任务目标  
实现双模型（LightGBM+XGBoost）月频选股引擎，  
支持GPU/CPU自适应切换，输出186个月的持仓结果。  
目录结构

quant\_fund/  
└── m2\_engine/  
├── **init**.py  
├── gpu\_detector.py      ← GPU检测与配置  
├── feature\_store.py     ← 特征工程  
├── lgbm\_model.py        ← LightGBM排序模型  
├── xgb\_model.py         ← XGBoost排序模型  
├── ensemble.py          ← 集成预测（GPU/CPU自适应）  
├── portfolio\_builder.py ← 持仓构建  
└── run\_m2.py            ← 主循环入口



第一步：创建 m2\_engine/**init**.py

"""M2 选股算法引擎"""  
from .run\_m2 import run\_m2  
from .gpu\_detector import GPUConfig

**all** = \["run\_m2", "GPUConfig"]



第二步：创建 m2\_engine/gpu\_detector.py

"""  
GPU检测与配置模块  
自适应检测CUDA/OpenCL可用性  
支持运行时切换GPU/CPU模式

GTX 1650 4GB VRAM限制：

* 单模型训练约1.5GB VRAM
* LGBM和XGB必须串行使用GPU（并行会OOM）
* CPU模式下LGBM+XGB并行（ThreadPoolExecutor）  
"""  
import os  
import logging

logger = logging.getLogger("m2.gpu")



class GPUConfig:  
"""  
GPU配置管理器  
单例模式，全局统一管理GPU/CPU模式  
"""  
\_instance = None  
\_mode = None          # "gpu" or "cpu"  
\_cuda\_available = None  
\_opencl\_available = None

&#x20;   def \_\_new\_\_(cls):  
        if cls.\_instance is None:  
            cls.\_instance = super().\_\_new\_\_(cls)  
            cls.\_instance.\_detect()  
        return cls.\_instance  
  
    def \_detect(self):  
        """检测GPU可用性"""  
        # 检测CUDA（XGBoost使用）  
        self.\_cuda\_available = False  
        try:  
            import xgboost as xgb  
            # 用小数据测试CUDA是否真正可用  
            import numpy as np  
            X = np.random.randn(100, 10).astype(np.float32)  
            y = np.random.randn(100).astype(np.float32)  
            dm = xgb.DMatrix(X, label=y)  
            params = {  
                "tree\_method": "hist",  
                "device": "cuda",  
                "verbosity": 0,  
            }  
            xgb.train(params, dm, num\_boost\_round=1,  
                     verbose\_eval=False)  
            self.\_cuda\_available = True  
            logger.info("✅ CUDA可用（XGBoost GPU模式）")  
        except Exception as e:  
            logger.info(f"CUDA不可用: {e}")  
  
        # 检测OpenCL（LightGBM使用）  
        self.\_opencl\_available = False  
        try:  
            import lightgbm as lgb  
            import numpy as np  
            X = np.random.randn(100, 10).astype(np.float32)  
            y = np.random.randn(100).astype(np.float32)  
            ds = lgb.Dataset(X, label=y, free\_raw\_data=False)  
            params = {  
                "device\_type": "gpu",  
                "verbosity": -1,  
                "num\_leaves": 4,  
            }  
            lgb.train(params, ds, num\_boost\_round=1)  
            self.\_opencl\_available = True  
            logger.info("✅ OpenCL可用（LightGBM GPU模式）")  
        except Exception as e:  
            logger.info(f"OpenCL不可用: {e}")  
  
        # 自动选择默认模式  
        if self.\_cuda\_available or self.\_opencl\_available:  
            self.\_mode = "gpu"  
            logger.info("默认模式: GPU")  
        else:  
            self.\_mode = "cpu"  
            logger.info("默认模式: CPU（无可用GPU）")  
  
    @property  
    def mode(self) -> str:  
        return self.\_mode  
  
    @property  
    def cuda\_available(self) -> bool:  
        return self.\_cuda\_available  
  
    @property  
    def opencl\_available(self) -> bool:  
        return self.\_opencl\_available  
  
    def set\_mode(self, mode: str):  
        """手动设置模式（供M5 UI调用）"""  
        if mode not in ("gpu", "cpu"):  
            raise ValueError("mode必须是'gpu'或'cpu'")  
        if mode == "gpu" and not (  
                self.\_cuda\_available or self.\_opencl\_available):  
            logger.warning("GPU不可用，强制使用CPU模式")  
            self.\_mode = "cpu"  
            return  
        self.\_mode = mode  
        logger.info(f"运行模式已切换: {mode}")  
  
    def get\_lgbm\_device(self) -> str:  
        """获取LightGBM设备类型"""  
        if self.\_mode == "gpu" and self.\_opencl\_available:  
            return "gpu"  
        return "cpu"  
  
    def get\_xgb\_device(self) -> str:  
        """获取XGBoost设备类型"""  
        if self.\_mode == "gpu" and self.\_cuda\_available:  
            return "cuda"  
        return "cpu"  
  
    def is\_gpu\_mode(self) -> bool:  
        return self.\_mode == "gpu"  
  
    def summary(self) -> str:  
        return (  
            f"GPU配置: mode={self.\_mode}, "  
            f"CUDA={self.\_cuda\_available}, "  
            f"OpenCL={self.\_opencl\_available}"  
        )  
  
  



第三步：创建 m2\_engine/feature\_store.py

"""  
特征工程模块  
对单个窗口的train/val/pred执行：

1. 候选因子列确定（排除元信息列）
2. drop\_short\_term\_noise过滤（\_5d/\_10d/\_1w）
3. 低覆盖因子过滤（在train上计算）
4. 截面Z-score（向量化，各自独立，不共享统计量）
5. NaN填0
6. 高相关因子去重（向量化相关矩阵，上三角一次处理）
7. IC筛选（向量化Pearson，Spearman的线性近似）
8. 黑天鹅多样性检查（记录不丢弃）

严禁数据泄露：所有统计量只在train上fit  
禁止for循环逐列计算（已全部向量化）  
"""  
import pandas as pd  
import numpy as np  
from scipy.stats import spearmanr  
from typing import Tuple, List, Dict  
import warnings  
import gc  
import logging

logger = logging.getLogger("m2.feature")  
warnings.filterwarnings("ignore")



# 不进模型的元信息列

META\_COLS = {  
"trade\_date", "stock\_code", "stock\_name", "industry",  
"list\_date", "days\_listed", "close\_price", "market\_cap",  
"avg\_turnover\_rate", "Target\_Return\_1M", "benchmark\_return",  
"excess\_return\_1m", "label\_rank", "suspend\_days",  
}  
MACRO\_PREFIX = "macro\_"



class FeatureStore:  
def **init**(  
self,  
min\_valid\_rate: float = 0.30,  
max\_corr: float = 0.95,  
min\_ic\_abs: float = 0.005,  
min\_keep\_factors: int = 50,  
drop\_short\_term\_noise: bool = False,  
):  
self.min\_valid\_rate = min\_valid\_rate  
self.max\_corr = max\_corr  
self.min\_ic\_abs = min\_ic\_abs  
self.min\_keep\_factors = min\_keep\_factors  
self.drop\_short\_term\_noise = drop\_short\_term\_noise  
self.\_feature\_cols: List\[str] = \[]

&#x20;   def fit\_transform(  
        self,  
        train\_df: pd.DataFrame,  
        val\_df: pd.DataFrame,  
        pred\_df: pd.DataFrame,  
    ) -> Tuple\[pd.DataFrame, pd.DataFrame,  
               pd.DataFrame, List\[str]]:  
  
        # Step1: 候选因子列  
        factor\_cols = \[c for c in train\_df.columns  
                       if c not in META\_COLS  
                       and not c.endswith("\_raw")]  
  
        # Step1.5: 短期噪音过滤  
        if self.drop\_short\_term\_noise:  
            factor\_cols = \[  
                c for c in factor\_cols  
                if not any(p in c  
                           for p in \["\_5d","\_10d","\_1w"])]  
  
        # Step2: 低覆盖过滤（在train上）  
        valid\_rates = train\_df\[factor\_cols].notna().mean()  
        valid\_cols = valid\_rates\[  
            valid\_rates >= self.min\_valid\_rate  
        ].index.tolist()  
  
        # Step3: 截面Z-score（向量化）  
        non\_macro = \[c for c in valid\_cols  
                     if not c.startswith(MACRO\_PREFIX)]  
        macro\_cols = \[c for c in valid\_cols  
                      if c.startswith(MACRO\_PREFIX)]  
  
        train\_p = train\_df.copy(deep=False)  
        val\_p   = val\_df.copy(deep=False)  
        pred\_p  = pred\_df.copy(deep=False)  
  
        def \_zscore(df, cols):  
            if not cols:  
                return df  
            grp = df.groupby("trade\_date")\[cols]  
            means = grp.transform("mean")  
            stds  = (grp.transform("std")  
                        .fillna(1.0)  
                        .replace(0, 1.0))  
            stds\[stds < 1e-6] = 1.0  
            df\[cols] = (df\[cols] - means) / stds  
            return df  
  
        train\_p = \_zscore(train\_p, non\_macro)  
        val\_p   = \_zscore(val\_p,   non\_macro)  
        pred\_p  = \_zscore(pred\_p,  non\_macro)  
  
        # Step4: NaN填0  
        for df in \[train\_p, val\_p, pred\_p]:  
            df\[valid\_cols] = df\[valid\_cols].fillna(0)  
  
        # Step5: 高相关去重（向量化上三角）  
        X\_vals = (train\_p\[valid\_cols].values  
                  .astype(np.float32))  
        corr\_matrix = np.abs(  
            np.corrcoef(X\_vals, rowvar=False))  
        corr\_matrix = np.nan\_to\_num(corr\_matrix, nan=0.0)  
        upper = np.triu(corr\_matrix, k=1)  
        drop\_idx = set(np.where(upper > self.max\_corr)\[1])  
        retained = \[c for i, c in enumerate(valid\_cols)  
                    if i not in drop\_idx]  
        del X\_vals, corr\_matrix, upper  
        gc.collect()  
  
        # Step6: IC筛选（向量化Pearson）  
        X\_ret = (train\_p\[retained].values  
                 .astype(np.float32))  
        Y = train\_p\["label\_rank"].values.astype(np.float32)  
        X\_c = X\_ret - X\_ret.mean(axis=0)  
        Y\_c = Y - Y.mean()  
        cov   = (X\_c \* Y\_c\[:, None]).sum(axis=0)  
        x\_std = np.sqrt((X\_c\*\*2).sum(axis=0) + 1e-8)  
        y\_std = np.sqrt((Y\_c\*\*2).sum() + 1e-8)  
        corrs = np.abs(cov / (x\_std \* y\_std))  
        corrs = np.nan\_to\_num(corrs, nan=0.0)  
        del X\_ret, X\_c, Y\_c, cov  
        gc.collect()  
  
        ic\_scores = dict(zip(retained, corrs.tolist()))  
        sorted\_by\_ic = sorted(ic\_scores.items(),  
                              key=lambda x: x\[1],  
                              reverse=True)  
        above = \[c for c, ic in sorted\_by\_ic  
                 if ic >= self.min\_ic\_abs]  
        final\_cols = (  
            above if len(above) >= self.min\_keep\_factors  
            else \[c for c, \_ in  
                  sorted\_by\_ic\[:self.min\_keep\_factors]]  
        )  
  
        # Step7: float32转换  
        for df in \[train\_p, val\_p, pred\_p]:  
            df\[final\_cols] = df\[final\_cols].astype(np.float32)  
  
        self.\_feature\_cols = final\_cols  
        return train\_p, val\_p, pred\_p, final\_cols  
  
    def get\_feature\_cols(self) -> List\[str]:  
        return self.\_feature\_cols  
  
  



第四步：创建 m2\_engine/lgbm\_model.py

"""  
LightGBM排序模型  
支持GPU（OpenCL）/ CPU自适应  
支持学习率衰减/自适应深度/可配置早停

关键规则：  
★ \_\_init\_\_必须先dict()拷贝params再.pop()  
★ predict必须使用best\_iteration（早停回滚）  
★ GPU模式下不与XGBoost并行（共用GTX1650显存）  
"""  
import lightgbm as lgb  
import pandas as pd  
import numpy as np  
from scipy.stats import spearmanr  
from typing import Dict, Optional  
import warnings  
import os  
import sys  
import logging  
from pathlib import Path

logger = logging.getLogger("m2.lgbm")  
warnings.filterwarnings("ignore")

# 自适应线程配置

config\_path = (Path(**file**).parent.parent /  
"config" / "concurrency\_config.py")  
if config\_path.exists():  
sys.path.insert(0, str(config\_path.parent))  
from concurrency\_config import (  
M5\_NTHREAD\_PER\_MODEL, GLOBAL\_NTHREAD\_INNER)  
sys.path.pop(0)  
else:  
\_cores = os.cpu\_count() or 8  
M5\_NTHREAD\_PER\_MODEL = max(\_cores // 4, 2)  
GLOBAL\_NTHREAD\_INNER = max(\_cores // 4, 2)



def \_make\_lr\_decay\_callback(  
init\_lr: float,  
decay\_every: int = 50,  
decay\_factor: float = 0.8,  
):  
"""LightGBM学习率阶梯衰减回调"""  
def callback(env):  
if (env.iteration > 0 and  
env.iteration % decay\_every == 0):  
stage = env.iteration // decay\_every  
new\_lr = init\_lr \* (decay\_factor \*\* stage)  
new\_lr = max(new\_lr, init\_lr \* 0.1)  
env.model.reset\_parameter(  
{"learning\_rate": new\_lr})  
callback.order = 10  
return callback



class LGBMRanker:  
"""LightGBM排序模型（GPU/CPU自适应）"""

&#x20;   HARD\_CONSTRAINTS = {  
        "subsample": 0.8,  
        "min\_child\_samples": 20,  
        "boosting\_type": "gbdt",  
    }  
    TUNABLE\_DEFAULTS = {  
        "max\_depth": 4,  
        "colsample\_bytree": 0.3,  
        "learning\_rate": 0.05,  
        "n\_estimators": 200,  
        "reg\_alpha": 0.1,  
        "reg\_lambda": 1.0,  
        "min\_split\_gain": 0.01,  
        "verbose": -1,  
    }  
  
    def \_\_init\_\_(self, params: Optional\[Dict] = None):  
        # ★ 必须先浅拷贝，禁止原地修改调用方的dict  
        params = dict(params) if params else {}  
  
        # ★ pop提取自定义字段（这些key不能传给LightGBM API）  
        self.lr\_mode = params.pop("lr\_mode", "fixed")  
        self.decay\_every = params.pop("decay\_every", 50)  
        self.decay\_factor = params.pop("decay\_factor", 0.8)  
        self.depth\_mode = params.pop("depth\_mode", "fixed")  
        self.early\_stopping\_rounds = params.pop(  
            "early\_stopping\_rounds", 30)  
  
        # 合并参数：默认值 < 用户传入 < 硬约束  
        self.params = self.TUNABLE\_DEFAULTS.copy()  
        self.params.update(params)  
        self.params.update(self.HARD\_CONSTRAINTS)  
  
        self.model\_: Optional\[lgb.Booster] = None  
        self.best\_iteration\_: int = 0  
        self.train\_ic\_: float = 0.0  
        self.val\_ic\_:   float = 0.0  
  
    def fit(  
        self,  
        X\_train: pd.DataFrame,  
        y\_train: pd.Series,  
        group\_train: list,  
        X\_val: pd.DataFrame,  
        y\_val: pd.Series,  
        group\_val: list,  
        gpu\_mode: bool = False,  
    ) -> None:  
        """  
        训练模型  
        gpu\_mode: True时使用OpenCL GPU加速  
        """  
        # GPU/CPU设备配置  
        device\_type = "gpu" if gpu\_mode else "cpu"  
  
        num\_boost\_round = self.params.get("n\_estimators", 200)  
        learning\_rate   = self.params.get("learning\_rate", 0.05)  
        max\_depth       = self.params.get("max\_depth", 4)  
        train\_size      = len(X\_train)  
  
        # 自适应深度（num\_leaves）  
        if self.depth\_mode == "adaptive":  
            num\_leaves = min(  
                int(2 \*\* max\_depth),  
                max(15, train\_size // 200),  
                255,  
            )  
        else:  
            num\_leaves = min(int(2 \*\* max\_depth), 63)  
  
        lgb\_params = {  
            "objective":        "regression",  
            "metric":           "rmse",  
            "learning\_rate":    learning\_rate,  
            "max\_depth":        max\_depth,  
            "num\_leaves":       num\_leaves,  
            "colsample\_bytree": self.params.get(  
                "colsample\_bytree", 0.3),  
            "subsample":        self.params\["subsample"],  
            "min\_child\_samples":self.params\["min\_child\_samples"],  
            "reg\_alpha":        self.params\["reg\_alpha"],  
            "reg\_lambda":       self.params\["reg\_lambda"],  
            "min\_split\_gain":   self.params\["min\_split\_gain"],  
            "verbosity":        -1,  
            "device\_type":      device\_type,  
            "histogram\_pool\_size": 2048,  
        }  
  
        # GPU模式不设nthread（GPU自管理线程）  
        # CPU模式设nthread  
        if not gpu\_mode:  
            lgb\_params\["nthread"] = M5\_NTHREAD\_PER\_MODEL  
  
        # 自定义IC评估回调  
        def ic\_metric(y\_pred, dataset):  
            label = dataset.get\_label()  
            pred\_rank = np.argsort(  
                np.argsort(y\_pred)).astype(np.float32)  
            ic = np.corrcoef(pred\_rank, label)\[0, 1]  
            return "ic", (ic if np.isfinite(ic) else 0.0), True  
  
        # 构建callbacks  
        callbacks = \[  
            lgb.early\_stopping(  
                stopping\_rounds=self.early\_stopping\_rounds,  
                verbose=False),  
            lgb.log\_evaluation(period=100),  
        ]  
        if self.lr\_mode == "decay":  
            callbacks.append(\_make\_lr\_decay\_callback(  
                learning\_rate,  
                self.decay\_every,  
                self.decay\_factor))  
  
        lgb\_train = lgb.Dataset(  
            X\_train.values, label=y\_train.values,  
            group=group\_train)  
        lgb\_val = lgb.Dataset(  
            X\_val.values, label=y\_val.values,  
            group=group\_val, reference=lgb\_train)  
  
        self.model\_ = lgb.train(  
            lgb\_params, lgb\_train,  
            num\_boost\_round=num\_boost\_round,  
            valid\_sets=\[lgb\_train, lgb\_val],  
            valid\_names=\["train", "valid"],  
            feval=ic\_metric,  
            callbacks=callbacks,  
        )  
  
        # ★ 早停回滚：使用best\_iteration  
        self.best\_iteration\_ = self.model\_.best\_iteration  
        logger.info(  
            f"LGBM \[{device\_type.upper()}] "  
            f"best\_iter={self.best\_iteration\_}/"  
            f"{num\_boost\_round} "  
            f"num\_leaves={num\_leaves}")  
  
        # 计算IC  
        train\_pred = self.model\_.predict(  
            X\_train.values,  
            num\_iteration=self.best\_iteration\_)  
        val\_pred = self.model\_.predict(  
            X\_val.values,  
            num\_iteration=self.best\_iteration\_)  
        self.train\_ic\_ = spearmanr(  
            train\_pred, y\_train.values).correlation  
        self.val\_ic\_ = spearmanr(  
            val\_pred, y\_val.values).correlation  
  
    def predict(self, X: pd.DataFrame) -> np.ndarray:  
        """★ 使用best\_iteration预测，不用默认的最后一轮"""  
        if self.model\_ is None:  
            raise ValueError("模型未训练")  
        return self.model\_.predict(  
            X.values,  
            num\_iteration=self.best\_iteration\_)  
  
    def get\_feature\_importance(  
        self,  
        importance\_type: str = "gain",  
    ) -> pd.Series:  
        importance = self.model\_.feature\_importance(  
            importance\_type=importance\_type)  
        return pd.Series(  
            importance,  
            index=self.model\_.feature\_name())  
  
  



第五步：创建 m2\_engine/xgb\_model.py

"""  
XGBoost排序模型  
支持CUDA GPU / CPU自适应  
支持学习率衰减/可配置早停

★ \_*init\_\_必须先dict()拷贝params再.pop()  
★ predict必须使用iteration\_range=(0, best\_iteration*)  
★ GPU模式使用device='cuda'（XGBoost 2.0+）  
"""  
import xgboost as xgb  
import pandas as pd  
import numpy as np  
from scipy.stats import spearmanr  
from typing import Dict, Optional, List  
import warnings  
import os  
import sys  
import logging  
from pathlib import Path

logger = logging.getLogger("m2.xgb")  
warnings.filterwarnings("ignore")

config\_path = (Path(**file**).parent.parent /  
"config" / "concurrency\_config.py")  
if config\_path.exists():  
sys.path.insert(0, str(config\_path.parent))  
from concurrency\_config import M5\_NTHREAD\_PER\_MODEL  
sys.path.pop(0)  
else:  
M5\_NTHREAD\_PER\_MODEL = max((os.cpu\_count() or 8) // 4, 2)



def \_make\_xgb\_lr\_schedule(  
init\_lr: float,  
n\_estimators: int,  
decay\_every: int = 50,  
decay\_factor: float = 0.8,  
) -> List\[float]:  
"""XGBoost学习率阶梯衰减调度"""  
lrs = \[]  
for i in range(n\_estimators):  
stage = i // decay\_every  
lr = init\_lr \* (decay\_factor \*\* stage)  
lrs.append(max(lr, init\_lr \* 0.1))  
return lrs



class XGBRanker:  
"""XGBoost排序模型（CUDA GPU/CPU自适应）"""

&#x20;   HARD\_CONSTRAINTS = {"min\_child\_weight": 20}  
    TUNABLE\_DEFAULTS = {  
        "max\_depth":        4,  
        "colsample\_bytree": 0.3,  
        "learning\_rate":    0.05,  
        "n\_estimators":     200,  
        "reg\_alpha":        0.1,  
        "reg\_lambda":       1.0,  
        "gamma":            0.01,  
        "subsample":        0.8,  
        "verbosity":        0,  
        "tree\_method":      "hist",  
    }  
  
    def \_\_init\_\_(self, params: Optional\[Dict] = None):  
        # ★ 必须先浅拷贝  
        params = dict(params) if params else {}  
  
        # ★ pop提取自定义字段  
        self.lr\_mode = params.pop("lr\_mode", "fixed")  
        self.decay\_every = params.pop("decay\_every", 50)  
        self.decay\_factor = params.pop("decay\_factor", 0.8)  
        self.early\_stopping\_rounds = params.pop(  
            "early\_stopping\_rounds", 30)  
  
        self.params = self.TUNABLE\_DEFAULTS.copy()  
        self.params.update(params)  
        self.params.update(self.HARD\_CONSTRAINTS)  
  
        self.model\_: Optional\[xgb.Booster] = None  
        self.best\_iteration\_: int = 0  
        self.train\_ic\_: float = 0.0  
        self.val\_ic\_:   float = 0.0  
  
    def fit(  
        self,  
        X\_train: pd.DataFrame,  
        y\_train: pd.Series,  
        group\_train: list,  
        X\_val: pd.DataFrame,  
        y\_val: pd.Series,  
        group\_val: list,  
        gpu\_mode: bool = False,  
    ) -> None:  
        n\_estimators  = self.params\["n\_estimators"]  
        learning\_rate = self.params\["learning\_rate"]  
  
        xgb\_params = {  
            "objective":        "reg:squarederror",  
            "learning\_rate":    learning\_rate,  
            "max\_depth":        self.params\["max\_depth"],  
            "colsample\_bytree": self.params\["colsample\_bytree"],  
            "subsample":        self.params\["subsample"],  
            "min\_child\_weight": self.params\["min\_child\_weight"],  
            "reg\_alpha":        self.params\["reg\_alpha"],  
            "reg\_lambda":       self.params\["reg\_lambda"],  
            "gamma":            self.params\["gamma"],  
            "tree\_method":      "hist",  
            "verbosity":        0,  
        }  
  
        if gpu\_mode:  
            xgb\_params\["device"] = "cuda"  
        else:  
            xgb\_params\["nthread"] = M5\_NTHREAD\_PER\_MODEL  
  
        # 自定义IC评估函数  
        def eval\_ic(preds, dtrain):  
            labels = dtrain.get\_label()  
            pred\_rank = np.argsort(  
                np.argsort(preds)).astype(np.float32)  
            ic = np.corrcoef(pred\_rank, labels)\[0, 1]  
            return "ic", ic if np.isfinite(ic) else 0.0  
  
        dtrain = xgb.DMatrix(X\_train.values,  
                             label=y\_train.values)  
        dtrain.set\_group(group\_train)  
        dval   = xgb.DMatrix(X\_val.values,  
                             label=y\_val.values)  
        dval.set\_group(group\_val)  
  
        lr\_schedule = None  
        if self.lr\_mode == "decay":  
            lr\_schedule = \_make\_xgb\_lr\_schedule(  
                learning\_rate, n\_estimators,  
                self.decay\_every, self.decay\_factor)  
  
        evals\_result = {}  
        self.model\_ = xgb.train(  
            xgb\_params, dtrain,  
            num\_boost\_round=n\_estimators,  
            evals=\[(dtrain,"train"),(dval,"valid")],  
            custom\_metric=eval\_ic,  
            evals\_result=evals\_result,  
            early\_stopping\_rounds=self.early\_stopping\_rounds,  
            verbose\_eval=False,  
            learning\_rates=lr\_schedule,  
        )  
  
        # ★ 早停回滚  
        self.best\_iteration\_ = self.model\_.best\_iteration  
        device\_str = "GPU(CUDA)" if gpu\_mode else "CPU"  
        logger.info(  
            f"XGB \[{device\_str}] "  
            f"best\_iter={self.best\_iteration\_}/"  
            f"{n\_estimators}")  
  
        train\_pred = self.model\_.predict(  
            dtrain,  
            iteration\_range=(0, self.best\_iteration\_))  
        val\_pred = self.model\_.predict(  
            dval,  
            iteration\_range=(0, self.best\_iteration\_))  
        self.train\_ic\_ = spearmanr(  
            train\_pred, y\_train.values).correlation  
        self.val\_ic\_   = spearmanr(  
            val\_pred,   y\_val.values).correlation  
  
    def predict(self, X: pd.DataFrame) -> np.ndarray:  
        """★ 使用iteration\_range，不用默认的最后一轮"""  
        if self.model\_ is None:  
            raise ValueError("模型未训练")  
        dpred = xgb.DMatrix(X.values)  
        return self.model\_.predict(  
            dpred,  
            iteration\_range=(0, self.best\_iteration\_))  
  
    def get\_feature\_importance(  
        self,  
        importance\_type: str = "gain",  
    ) -> pd.Series:  
        scores = self.model\_.get\_score(  
            importance\_type=importance\_type)  
        n\_features = len(scores)  
        return pd.Series(  
            list(scores.values()),  
            index=list(scores.keys()))  
  
  



第六步：创建 m2\_engine/ensemble.py

"""  
集成预测模块  
GPU模式：LGBM→XGB串行（GTX1650显存限制）  
CPU模式：LGBM+XGB并行（ThreadPoolExecutor）

核心逻辑：

1. 双模型训练（GPU串行/CPU并行）
2. 验证集IC计算与ic\_gap检验
3. 降权决策（ic\_gap>0.15触发）
4. 支持compute\_val\_portfolio\_metrics（M5因变量）  
"""  
import pandas as pd  
import numpy as np  
from scipy.stats import spearmanr  
from concurrent.futures import ThreadPoolExecutor  
from typing import Dict, Optional  
import gc  
import warnings  
import logging

from .lgbm\_model import LGBMRanker  
from .xgb\_model  import XGBRanker  
from .gpu\_detector import GPUConfig

logger = logging.getLogger("m2.ensemble")  
warnings.filterwarnings("ignore")



class EnsemblePredictor:  
"""双模型软投票融合 + IC质量检验"""

&#x20;   IC\_PENALIZE\_THRESHOLD = 0.15  
    IC\_WARN\_THRESHOLD     = 0.02  
  
    def \_\_init\_\_(  
        self,  
        lgbm\_weight: float = 0.5,  
        xgb\_weight:  float = 0.5,  
        lgbm\_params: Optional\[Dict] = None,  
        xgb\_params:  Optional\[Dict] = None,  
    ):  
        total = lgbm\_weight + xgb\_weight  
        self.lgbm\_weight = lgbm\_weight / total  
        self.xgb\_weight  = xgb\_weight  / total  
        self.lgbm = LGBMRanker(lgbm\_params)  
        self.xgb  = XGBRanker(xgb\_params)  
        self.\_gpu\_cfg = GPUConfig()  
  
    def fit\_predict(  
        self,  
        train\_df: pd.DataFrame,  
        val\_df:   pd.DataFrame,  
        pred\_df:  pd.DataFrame,  
        feature\_cols: list,  
    ) -> Dict:  
        # 准备数据  
        train\_v = train\_df\[train\_df\["label\_rank"].notna()]  
        val\_v   = val\_df\[val\_df\["label\_rank"].notna()]  
        g\_train = train\_v.groupby("trade\_date").size().tolist()  
        g\_val   = val\_v.groupby("trade\_date").size().tolist()  
        X\_train = train\_v\[feature\_cols]  
        y\_train = train\_v\["label\_rank"]  
        X\_val   = val\_v\[feature\_cols]  
        y\_val   = val\_v\["label\_rank"]  
  
        gpu\_mode = self.\_gpu\_cfg.is\_gpu\_mode()  
  
        if gpu\_mode:  
            # ── GPU模式：串行训练（显存限制）──────────  
            # GTX1650 4GB，单模型约1.5GB，必须串行  
            logger.info("GPU模式：LGBM→XGB串行训练")  
            self.lgbm.fit(  
                X\_train, y\_train, g\_train,  
                X\_val,   y\_val,   g\_val,  
                gpu\_mode=self.\_gpu\_cfg.opencl\_available,  
            )  
            self.xgb.fit(  
                X\_train, y\_train, g\_train,  
                X\_val,   y\_val,   g\_val,  
                gpu\_mode=self.\_gpu\_cfg.cuda\_available,  
            )  
        else:  
            # ── CPU模式：并行训练（ThreadPoolExecutor）  
            logger.info("CPU模式：LGBM+XGB并行训练")  
            def \_fit\_lgbm():  
                self.lgbm.fit(  
                    X\_train, y\_train, g\_train,  
                    X\_val,   y\_val,   g\_val,  
                    gpu\_mode=False)  
            def \_fit\_xgb():  
                self.xgb.fit(  
                    X\_train, y\_train, g\_train,  
                    X\_val,   y\_val,   g\_val,  
                    gpu\_mode=False)  
            with ThreadPoolExecutor(max\_workers=2) as exe:  
                f1 = exe.submit(\_fit\_lgbm)  
                f2 = exe.submit(\_fit\_xgb)  
                f1.result()  
                f2.result()  
  
        # 验证集集成预测  
        lgbm\_val = self.lgbm.predict(X\_val)  
        xgb\_val  = self.xgb.predict(X\_val)  
        ens\_val  = (self.lgbm\_weight \* lgbm\_val +  
                    self.xgb\_weight  \* xgb\_val)  
        val\_ic = self.\_monthly\_ic(val\_v, ens\_val)  
  
        # 训练集IC（最后一个训练月）  
        last\_m = train\_v\["trade\_date"].max()  
        mask   = train\_v\["trade\_date"] == last\_m  
        X\_tl   = train\_v.loc\[mask, feature\_cols]  
        y\_tl   = train\_v.loc\[mask, "label\_rank"]  
        ens\_tl = (self.lgbm\_weight \* self.lgbm.predict(X\_tl) +  
                  self.xgb\_weight  \* self.xgb.predict(X\_tl))  
        train\_ic = spearmanr(  
            ens\_tl, y\_tl.values).correlation  
  
        ic\_gap      = train\_ic - val\_ic  
        is\_penalized = ic\_gap > self.IC\_PENALIZE\_THRESHOLD  
  
        if is\_penalized:  
            logger.warning(  
                f"ic\_gap={ic\_gap:.4f}>{self.IC\_PENALIZE\_THRESHOLD}"  
                f"，触发降权")  
  
        # 预测月打分  
        X\_pred = pred\_df\[feature\_cols]  
        lgbm\_p = self.lgbm.predict(X\_pred)  
        xgb\_p  = self.xgb.predict(X\_pred)  
        pred\_scores = (self.lgbm\_weight \* lgbm\_p +  
                       self.xgb\_weight  \* xgb\_p)  
        score\_cv = (  
            np.std(\[lgbm\_p, xgb\_p], axis=0) /  
            (np.abs(pred\_scores) + 1e-6)  
        )  
  
        pred\_out = pred\_df.copy()  
        pred\_out\["score"]    = pred\_scores  
        pred\_out\["score\_cv"] = score\_cv  
  
        pred\_month = (pred\_df\["trade\_date"].iloc\[0]  
                      .strftime("%Y%m"))  
  
        return {  
            "pred\_month":          pred\_month,  
            "pred\_df\_with\_scores": pred\_out,  
            "val\_ic":              float(val\_ic),  
            "train\_ic":            float(train\_ic),  
            "ic\_gap":              float(ic\_gap),  
            "is\_penalized":        bool(is\_penalized),  
            "lgbm\_best\_iter":      self.lgbm.best\_iteration\_,  
            "xgb\_best\_iter":       self.xgb.best\_iteration\_,  
            "gpu\_mode":            gpu\_mode,  
        }  
  
    def compute\_val\_portfolio\_metrics(  
        self,  
        val\_df\_with\_scores: pd.DataFrame,  
    ) -> Dict:  
        """  
        验证集模拟持仓，计算滚动6月绩效指标  
        供M5因变量使用  
        """  
        monthly\_returns = \[]  
        monthly\_benchmarks = \[]  
  
        for date in sorted(  
                val\_df\_with\_scores\["trade\_date"].unique()):  
            mdf = val\_df\_with\_scores\[  
                val\_df\_with\_scores\["trade\_date"] == date  
            ].sort\_values("score", ascending=False)  
  
            top5  = mdf.head(5)  
            next5 = mdf.iloc\[5:10]  
  
            ret = (  
                0.13 \* top5\["Target\_Return\_1M"].fillna(0).sum() +  
                0.07 \* next5\["Target\_Return\_1M"].fillna(0).sum()  
            )  
            bm = (mdf\["benchmark\_return"].iloc\[0]  
                  if "benchmark\_return" in mdf.columns  
                  else 0.0)  
            monthly\_returns.append(float(ret))  
            monthly\_benchmarks.append(float(bm))  
  
        n = len(monthly\_returns)  
        if n < 6:  
            return {  
                "val\_rolling6m\_ir": 0.0,  
                "val\_rolling6m\_dir": 0.0,  
                "val\_rolling6m\_sortino": 0.0,  
                "val\_rolling6m\_return": 0.0,  
            }  
  
        rets = np.array(monthly\_returns)  
        bms  = np.array(monthly\_benchmarks)  
        excess = rets - bms  
  
        irs, dirs, sortinos, cumrets = \[], \[], \[], \[]  
        for i in range(n - 5):  
            w\_ex = excess\[i:i+6]  
            w\_ret = rets\[i:i+6]  
  
            std\_ex = np.std(w\_ex, ddof=0)  
            ir = np.mean(w\_ex) / std\_ex if std\_ex > 1e-8 else 0.0  
            irs.append(ir)  
  
            down\_ex = w\_ex\[w\_ex < 0]  
            down\_std = np.std(down\_ex) if len(down\_ex) > 0 else 1e-6  
            dirs.append(  
                np.mean(w\_ex) / down\_std \* np.sqrt(12))  
  
            rf = 0.03 / 12  
            down\_ret = w\_ret\[w\_ret < rf]  
            d\_std\_ret = (np.std(down\_ret)  
                         if len(down\_ret) > 0 else 1e-6)  
            sortinos.append(  
                np.mean(w\_ret - rf) / d\_std\_ret \* np.sqrt(12))  
  
            cumrets.append(np.prod(1 + w\_ret) - 1)  
  
        return {  
            "val\_rolling6m\_ir":      float(np.mean(irs)),  
            "val\_rolling6m\_dir":     float(np.mean(dirs)),  
            "val\_rolling6m\_sortino": float(np.mean(sortinos)),  
            "val\_rolling6m\_return":  float(np.mean(cumrets)),  
        }  
  
    def \_monthly\_ic(  
        self,  
        df: pd.DataFrame,  
        pred\_scores: np.ndarray,  
        label\_col: str = "label\_rank",  
    ) -> float:  
        ics = \[]  
        for date in df\["trade\_date"].unique():  
            mask = df\["trade\_date"] == date  
            if mask.sum() >= 10:  
                ic = spearmanr(  
                    pred\_scores\[mask.values],  
                    df.loc\[mask, label\_col].values,  
                ).correlation  
                ics.append(ic)  
        return float(np.mean(ics)) if ics else 0.0  
  
  



第七步：创建 m2\_engine/portfolio\_builder.py

"""  
持仓构建模块  
按score降序选Top20，标记is\_holding前10只  
仓位分配：正常High(13%)/Low(7%)，降权High(10%)/Low(5%)  
"""  
import pandas as pd  
import numpy as np  
from typing import Optional, List  
import logging

logger = logging.getLogger("m2.portfolio")



class PortfolioBuilder:  
"""月度持仓构建器"""

&#x20;   def build(  
        self,  
        pred\_df\_with\_scores: pd.DataFrame,  
        val\_ic:   float,  
        ic\_gap:   float,  
        is\_penalized: bool,  
        lgbm\_model,  
        feature\_cols: List\[str],  
        pred\_month: str,  
        compute\_shap: bool = False,  
    ) -> pd.DataFrame:  
        """  
        构建当月持仓  
  
        输出列：  
            stock\_code, stock\_name, industry, score, score\_cv,  
            tier(High/Low/Reserve), weight, is\_holding,  
            val\_ic, ic\_gap, is\_penalized, pred\_month, trade\_date  
        """  
        df = pred\_df\_with\_scores.copy()  
        df = df.sort\_values("score", ascending=False)  
  
        # 选Top20（前20只）  
        top20 = df.head(20).copy()  
        top20\["pred\_month"] = pred\_month  
        top20\["val\_ic"]     = val\_ic  
        top20\["ic\_gap"]     = ic\_gap  
        top20\["is\_penalized"] = is\_penalized  
  
        # 分层  
        top20\["tier"] = "Reserve"  
        top20.iloc\[:5,  top20.columns.get\_loc("tier")] = "High"  
        top20.iloc\[5:10,top20.columns.get\_loc("tier")] = "Low"  
  
        # 仓位  
        weight\_map = {  
            ("High", False): 0.13,  
            ("High", True):  0.10,  
            ("Low",  False): 0.07,  
            ("Low",  True):  0.05,  
            ("Reserve", False): 0.0,  
            ("Reserve", True):  0.0,  
        }  
        top20\["weight"] = top20.apply(  
            lambda r: weight\_map.get(  
                (r\["tier"], is\_penalized), 0.0),  
            axis=1,  
        )  
        top20\["is\_holding"] = top20\["tier"].isin(  
            \["High", "Low"])  
  
        # SHAP归因（M5调用时跳过）  
        if compute\_shap and lgbm\_model is not None:  
            try:  
                import shap  
                X\_holding = pred\_df\_with\_scores.loc\[  
                    top20.index, feature\_cols]  
                explainer = shap.TreeExplainer(  
                    lgbm\_model.model\_)  
                shap\_vals = explainer.shap\_values(X\_holding)  
                if isinstance(shap\_vals, list):  
                    shap\_vals = shap\_vals\[0]  
                for k in range(min(3, shap\_vals.shape\[1])):  
                    top20\[f"shap\_top{k+1}\_factor"] = (  
                        feature\_cols\[k])  
                    top20\[f"shap\_top{k+1}\_value"] = (  
                        shap\_vals\[:, k])  
            except Exception as e:  
                logger.warning(f"SHAP计算失败: {e}")  
  
        return top20  
  
  



第八步：创建 m2\_engine/run\_m2.py

"""  
M2主循环入口

★ 关键接口约束（禁止修改）：  
第二参数必须是 xgbm\_params，永远不能改成 xgb\_params

GPU/CPU策略：

* 从GPUConfig单例读取当前模式
* GPU模式：窗口串行，模型GPU串行（显存限制）
* CPU模式：窗口threading并行，模型CPU并行

内存策略：

* 惰性切片：不预加载全量windows
* FeatureStore结果缓存（线程安全）
* 每窗口结束强制gc.collect()  
"""  
import gc  
import json  
import threading  
import yaml  
import pandas as pd  
import numpy as np  
from pathlib import Path  
from typing import Tuple, Dict, Optional  
import time  
import sys  
import os  
import traceback  
import logging

# 自适应并发配置

config\_path = (Path(**file**).parent.parent /  
"config" / "concurrency\_config.py")  
if config\_path.exists():  
sys.path.insert(0, str(config\_path.parent))  
from concurrency\_config import GLOBAL\_N\_JOBS\_OUTER  
sys.path.pop(0)  
else:  
GLOBAL\_N\_JOBS\_OUTER = max(  
(os.cpu\_count() or 4) // 3, 1)

from m1\_engine.data\_loader import DataLoader  
from m1\_engine.rolling\_splitter import RollingSplitter  
from m1\_engine.label\_maker import LabelMaker  
from m2\_engine.feature\_store import FeatureStore  
from m2\_engine.ensemble import EnsemblePredictor  
from m2\_engine.portfolio\_builder import PortfolioBuilder  
from m2\_engine.gpu\_detector import GPUConfig

logger = logging.getLogger("m2.run")

# ── 线程安全FeatureStore缓存 ──────────────────────

\_FEATURE\_CACHE: Dict = {}  
\_FEATURE\_CACHE\_MAX\_SIZE = 99       # 缓存一半窗口  
\_FEATURE\_CACHE\_LOCK = threading.Lock()  
\_FEATURE\_CACHE\_PARAMS\_HASH: Optional\[int] = None



def \_process\_single\_window(  
i:            int,  
window:       Dict,  
cfg:          Dict,  
feature\_params: Optional\[Dict],  
lgbm\_params:  Optional\[Dict],  
xgb\_params:   Optional\[Dict],  
lgbm\_weight:  Optional\[float],  
compute\_shap: bool,  
date\_return\_map: Dict,  
compute\_val\_metrics: bool = False,  
) -> Tuple\[Optional\[pd.DataFrame], Dict]:  
"""处理单个窗口"""  
global \_FEATURE\_CACHE, \_FEATURE\_CACHE\_PARAMS\_HASH

&#x20;   train\_df = window\["train\_df"]  
    val\_df   = window\["val\_df"]  
    pred\_df  = window\["pred\_df"]  
    pred\_month = window\["pred\_month"]  
  
    stats = {  
        "success":     False,  
        "val\_ic":      0.0,  
        "ic\_gap":      0.0,  
        "is\_penalized":False,  
        "val\_metrics": {},  
        "failed":      None,  
    }  
  
    try:  
        fp = feature\_params or {}  
        cache\_key = (i, \_FEATURE\_CACHE\_PARAMS\_HASH)  
  
        # 读缓存  
        with \_FEATURE\_CACHE\_LOCK:  
            cached = \_FEATURE\_CACHE.get(cache\_key)  
  
        if cached is not None:  
            train\_p, val\_p, pred\_p, feature\_cols = cached  
        else:  
            fs = FeatureStore(  
                min\_valid\_rate=fp.get(  
                    "min\_valid\_rate",  
                    cfg\["m2"]\["feature\_store"]\["min\_valid\_rate"]),  
                max\_corr=fp.get(  
                    "max\_corr",  
                    cfg\["m2"]\["feature\_store"]\["max\_corr"]),  
                min\_ic\_abs=fp.get(  
                    "min\_ic\_abs",  
                    cfg\["m2"]\["feature\_store"]\["min\_ic\_abs"]),  
                min\_keep\_factors=fp.get(  
                    "min\_keep\_factors",  
                    cfg\["m2"]\["feature\_store"].get(  
                        "min\_keep\_factors", 50)),  
                drop\_short\_term\_noise=fp.get(  
                    "drop\_short\_term\_noise", False),  
            )  
            train\_p, val\_p, pred\_p, feature\_cols = (  
                fs.fit\_transform(train\_df, val\_df, pred\_df))  
  
            # 写缓存（线程安全）  
            with \_FEATURE\_CACHE\_LOCK:  
                if len(\_FEATURE\_CACHE) < \_FEATURE\_CACHE\_MAX\_SIZE:  
                    \_FEATURE\_CACHE\[cache\_key] = (  
                        train\_p, val\_p, pred\_p, feature\_cols)  
  
        eff\_weight = (lgbm\_weight if lgbm\_weight is not None  
                      else cfg\["m2"]\["ensemble"]\["lgbm\_weight"])  
  
        predictor = EnsemblePredictor(  
            lgbm\_weight=eff\_weight,  
            xgb\_weight=1.0 - eff\_weight,  
            lgbm\_params=lgbm\_params,  
            xgb\_params=xgb\_params,  
        )  
        result = predictor.fit\_predict(  
            train\_p, val\_p, pred\_p, feature\_cols)  
  
        pb = PortfolioBuilder()  
        portfolio = pb.build(  
            pred\_df\_with\_scores=result\["pred\_df\_with\_scores"],  
            val\_ic=result\["val\_ic"],  
            ic\_gap=result\["ic\_gap"],  
            is\_penalized=result\["is\_penalized"],  
            lgbm\_model=predictor.lgbm,  
            feature\_cols=feature\_cols,  
            pred\_month=pred\_month,  
            compute\_shap=compute\_shap,  
        )  
  
        # 回填收益  
        pred\_date = pred\_df\["trade\_date"].iloc\[0]  
        if pred\_date in date\_return\_map:  
            portfolio\["Target\_Return\_1M"] = (  
                portfolio\["stock\_code"].map(  
                    date\_return\_map\[pred\_date]))  
  
        # 计算滚动指标（compute\_val\_metrics=True时）  
        if compute\_val\_metrics:  
            try:  
                val\_scored = val\_p.copy(deep=False)  
                val\_scored\["score"] = (  
                    predictor.lgbm\_weight \*  
                    predictor.lgbm.predict(val\_p\[feature\_cols]) +  
                    predictor.xgb\_weight \*  
                    predictor.xgb.predict(val\_p\[feature\_cols])  
                )  
                stats\["val\_metrics"] = (  
                    predictor.compute\_val\_portfolio\_metrics(  
                        val\_scored))  
            except Exception as e:  
                logger.warning(f"滚动指标计算失败: {e}")  
  
        stats.update({  
            "success":      True,  
            "val\_ic":       result\["val\_ic"],  
            "ic\_gap":       result\["ic\_gap"],  
            "is\_penalized": result\["is\_penalized"],  
        })  
        return portfolio, stats  
  
    except Exception as e:  
        stats\["failed"] = pred\_month  
        logger.error(  
            f"窗口{pred\_month}失败: {type(e).\_\_name\_\_}: {e}")  
        logger.error(traceback.format\_exc())  
        return None, stats  
  
    finally:  
        del train\_df, val\_df, pred\_df  
        for name in \["train\_p","val\_p","pred\_p",  
                     "predictor","result"]:  
            try:  
                exec(f"del {name}")  
            except Exception:  
                pass  
        gc.collect()  
  
  



def run\_m2(  
lgbm\_params:  Optional\[Dict] = None,  
xgbm\_params:  Optional\[Dict] = None,  # ★ 必须是xgbm\_params  
feature\_params: Optional\[Dict] = None,  
lgbm\_weight:  Optional\[float] = None,  
fast\_mode:    bool = False,  
fast\_window\_count: int = 60,  
compute\_val\_metrics: bool = False,  
compute\_shap: bool = True,  
verbose:      bool = True,  
preloaded\_windows: list = None,  
date\_return\_map: Dict = None,  
stop\_event = None,  
preloaded\_factor\_df: pd.DataFrame = None,  
gpu\_mode:     Optional\[bool] = None,  # ★ 新增：None=自动  
) -> Tuple\[pd.DataFrame, Dict]:  
"""  
M2主循环

&#x20;   ★ xgbm\_params：第二参数名永远不能改  
    ★ gpu\_mode：None=自动检测，True=强制GPU，False=强制CPU  
    """  
    start\_time = time.time()  
  
    # ── GPU模式配置 ───────────────────────────────  
    gpu\_cfg = GPUConfig()  
    if gpu\_mode is not None:  
        gpu\_cfg.set\_mode("gpu" if gpu\_mode else "cpu")  
  
    if verbose:  
        print(f"\\n{'='\*60}")  
        print(f"M2 双引擎选股算法层")  
        print(f"{'='\*60}")  
        print(f"运行模式: {gpu\_cfg.summary()}")  
  
    # ── FeatureStore缓存管理 ──────────────────────  
    global \_FEATURE\_CACHE, \_FEATURE\_CACHE\_PARAMS\_HASH  
    new\_hash = (hash(frozenset((feature\_params or {}).items()))  
                if feature\_params else 0)  
    if new\_hash != \_FEATURE\_CACHE\_PARAMS\_HASH:  
        with \_FEATURE\_CACHE\_LOCK:  
            \_FEATURE\_CACHE.clear()  
            \_FEATURE\_CACHE\_PARAMS\_HASH = new\_hash  
  
    # ── 加载配置 ──────────────────────────────────  
    with open("config/config.yaml", encoding="utf-8") as f:  
        cfg = yaml.safe\_load(f)  
  
    # ── 数据加载 ──────────────────────────────────  
    if preloaded\_windows is not None:  
        windows = preloaded\_windows  
        if date\_return\_map is None:  
            date\_return\_map = {}  
        if verbose:  
            print(f"使用预加载数据: {len(windows)}个窗口")  
    else:  
        if preloaded\_factor\_df is not None:  
            factor\_df = preloaded\_factor\_df  
        else:  
            loader = DataLoader()  
            factor\_df = loader.load()  
            factor\_df = LabelMaker().make\_labels(factor\_df)  
  
        # 构建date\_return\_map  
        if date\_return\_map is None:  
            ret\_df = factor\_df\[  
                \["trade\_date","stock\_code","Target\_Return\_1M"]  
            ].dropna(subset=\["Target\_Return\_1M"]).copy()  
            date\_return\_map = (  
                ret\_df.groupby("trade\_date")  
                .apply(lambda x: dict(zip(  
                    x\["stock\_code"],  
                    x\["Target\_Return\_1M"])))  
                .to\_dict()  
            )  
            del ret\_df  
            gc.collect()  
  
        splitter = RollingSplitter()  
        all\_windows = list(splitter.split(factor\_df))  
  
        if fast\_mode:  
            windows = all\_windows\[-fast\_window\_count:]  
            if verbose:  
                print(f"快速模式: 使用最近{fast\_window\_count}个窗口")  
        else:  
            windows = all\_windows  
  
        del factor\_df  
        gc.collect()  
  
    # ── 主循环 ────────────────────────────────────  
    all\_records = \[]  
    run\_stats = {  
        "total\_windows":   len(windows),  
        "success":         0,  
        "failed":          \[],  
        "penalized\_months":0,  
        "avg\_val\_ic":      \[],  
        "avg\_ic\_gap":      \[],  
        "val\_metrics\_list":\[],  
    }  
  
    # GPU模式强制串行（显存限制）  
    # CPU模式使用threading并行  
    use\_parallel = (not gpu\_cfg.is\_gpu\_mode() and  
                    GLOBAL\_N\_JOBS\_OUTER > 1)  
  
    if use\_parallel:  
        from joblib import Parallel, delayed  
        if verbose:  
            print(f"CPU并行模式: {GLOBAL\_N\_JOBS\_OUTER}窗口并行")  
  
        results = Parallel(  
            n\_jobs=GLOBAL\_N\_JOBS\_OUTER,  
            backend="threading",  
            prefer="threads",  
        )(  
            delayed(\_process\_single\_window)(  
                i, w, cfg, feature\_params,  
                lgbm\_params, xgbm\_params,  
                lgbm\_weight, compute\_shap,  
                date\_return\_map, compute\_val\_metrics,  
            )  
            for i, w in enumerate(windows)  
            if not (stop\_event and stop\_event.is\_set())  
        )  
    else:  
        # GPU串行或单线程  
        mode\_str = "GPU串行" if gpu\_cfg.is\_gpu\_mode() else "CPU串行"  
        if verbose:  
            print(f"{mode\_str}模式")  
        results = \[]  
        for i, w in enumerate(windows):  
            if stop\_event and stop\_event.is\_set():  
                if verbose:  
                    print(f"检测到停止信号，已处理{i}个窗口")  
                break  
            r = \_process\_single\_window(  
                i, w, cfg, feature\_params,  
                lgbm\_params, xgbm\_params,  
                lgbm\_weight, compute\_shap,  
                date\_return\_map, compute\_val\_metrics,  
            )  
            results.append(r)  
            if verbose and (i+1) % 10 == 0:  
                print(f"  进度: {i+1}/{len(windows)}")  
  
    # ── 汇总结果 ──────────────────────────────────  
    for portfolio, s in results:  
        if s\["success"]:  
            all\_records.append(portfolio)  
            run\_stats\["success"] += 1  
            run\_stats\["avg\_val\_ic"].append(s\["val\_ic"])  
            run\_stats\["avg\_ic\_gap"].append(s\["ic\_gap"])  
            if s\["is\_penalized"]:  
                run\_stats\["penalized\_months"] += 1  
            if s.get("val\_metrics"):  
                run\_stats\["val\_metrics\_list"].append(  
                    s\["val\_metrics"])  
        else:  
            run\_stats\["failed"].append(s\["failed"])  
  
    # ── 计算滚动指标均值 ─────────────────────────  
    if compute\_val\_metrics and run\_stats\["val\_metrics\_list"]:  
        vml = run\_stats\["val\_metrics\_list"]  
        run\_stats\["avg\_val\_portfolio\_metrics"] = {  
            k: float(np.mean(\[v.get(k, 0) for v in vml]))  
            for k in \["val\_rolling6m\_ir",  
                      "val\_rolling6m\_dir",  
                      "val\_rolling6m\_sortino",  
                      "val\_rolling6m\_return"]  
        }  
    else:  
        run\_stats\["avg\_val\_portfolio\_metrics"] = {  
            "val\_rolling6m\_ir":      0.0,  
            "val\_rolling6m\_dir":     0.0,  
            "val\_rolling6m\_sortino": 0.0,  
            "val\_rolling6m\_return":  0.0,  
        }  
  
    # ── 输出文件 ──────────────────────────────────  
    if not all\_records:  
        raise RuntimeError(  
            f"所有{run\_stats\['total\_windows']}个窗口均失败，"  
            f"请检查日志")  
  
    Path("output").mkdir(exist\_ok=True)  
    all\_portfolios = pd.concat(all\_records, ignore\_index=True)  
    all\_portfolios.to\_parquet(  
        "output/all\_portfolios.parquet", index=False)  
  
    holdings = all\_portfolios\[  
        all\_portfolios\["is\_holding"] == True]  
    holdings.to\_csv(  
        "output/all\_portfolios.csv",  
        index=False, encoding="utf-8-sig")  
  
    # ── 摘要输出 ──────────────────────────────────  
    elapsed = (time.time() - start\_time) / 60  
    avg\_ic  = np.mean(run\_stats\["avg\_val\_ic"])  
    avg\_gap = np.mean(run\_stats\["avg\_ic\_gap"])  
  
    if verbose:  
        print(f"\\n{'='\*60}")  
        print("M2 回测完成")  
        print(f"{'='\*60}")  
        print(f"运行模式:    {gpu\_cfg.mode.upper()}")  
        print(f"成功窗口:    {run\_stats\['success']}/"  
              f"{run\_stats\['total\_windows']}")  
        print(f"失败窗口:    {len(run\_stats\['failed'])}个")  
        print(f"平均val\_IC:  {avg\_ic:.4f}")  
        print(f"平均ic\_gap:  {avg\_gap:.4f}")  
        print(f"降权月份:    {run\_stats\['penalized\_months']}个"  
              f"({run\_stats\['penalized\_months']/"  
              f"max(run\_stats\['success'],1):.1%})")  
        print(f"输出行数:    {len(all\_portfolios)}")  
        print(f"运行时间:    {elapsed:.1f}分钟")  
  
        fingerprint = {  
            "total\_portfolios\_rows": len(all\_portfolios),  
            "total\_months\_succeeded": run\_stats\["success"],  
            "total\_months\_failed": len(run\_stats\["failed"]),  
            "penalized\_months": run\_stats\["penalized\_months"],  
            "penalized\_rate": round(  
                run\_stats\["penalized\_months"] /  
                max(run\_stats\["success"], 1), 4),  
            "avg\_val\_ic":  round(float(avg\_ic),  4),  
            "avg\_ic\_gap":  round(float(avg\_gap), 4),  
            "elapsed\_minutes": round(elapsed, 1),  
            "gpu\_mode": gpu\_cfg.mode,  
        }  
        print(f"\\n=== M2指纹 ===")  
        print(json.dumps(fingerprint,  
                        ensure\_ascii=False, indent=2))  
  
    return all\_portfolios, run\_stats  
  
  



if **name** == "**main**":  
all\_portfolios, stats = run\_m2(  
fast\_mode=True,  
fast\_window\_count=3,  
compute\_shap=False,  
verbose=True,  
)



第九步：验证脚本

# test\_m2.py 放在 quant\_fund/ 根目录

"""M2全面验证脚本"""  
import sys, json  
sys.path.insert(0, ".")

print("=" \* 60)  
print("M2 模块验证")  
print("=" \* 60)

# ── 验证1：GPU检测 ─────────────────────────────

print("\\n【验证1】GPU检测")  
from m2\_engine.gpu\_detector import GPUConfig  
gpu = GPUConfig()  
print(f"  {gpu.summary()}")  
print(f"  CUDA可用: {'✅' if gpu.cuda\_available else '❌'}")  
print(f"  OpenCL可用: {'✅' if gpu.opencl\_available else '❌'}")  
print(f"  默认模式: {gpu.mode.upper()}")

# ── 验证2：参数命名保护 ───────────────────────

print("\\n【验证2】参数命名保护（dict浅拷贝）")  
from m2\_engine.lgbm\_model import LGBMRanker  
from m2\_engine.xgb\_model  import XGBRanker

params\_test = {  
"learning\_rate": 0.05,  
"lr\_mode": "decay",  
"depth\_mode": "adaptive",  
"early\_stopping\_rounds": 30,  
}  
original\_keys = set(params\_test.keys())

lgbm = LGBMRanker(params\_test)  
assert set(params\_test.keys()) == original\_keys, \\  
"❌ LGBMRanker修改了原始dict！"  
print("  ✅ LGBMRanker不修改原始dict")

xgb\_p = {"learning\_rate": 0.05, "lr\_mode": "fixed",  
"early\_stopping\_rounds": 30}  
orig\_xgb = set(xgb\_p.keys())  
xgb = XGBRanker(xgb\_p)  
assert set(xgb\_p.keys()) == orig\_xgb, \\  
"❌ XGBRanker修改了原始dict！"  
print("  ✅ XGBRanker不修改原始dict")

# ── 验证3：快速运行3个窗口 ────────────────────

print("\\n【验证3】快速运行3个窗口（CPU模式）")  
from m2\_engine.run\_m2 import run\_m2  
from m2\_engine.gpu\_detector import GPUConfig

GPUConfig().set\_mode("cpu")  # 先用CPU测试

import time  
t0 = time.time()  
portfolios, stats = run\_m2(  
fast\_mode=True,  
fast\_window\_count=3,  
compute\_val\_metrics=True,  
compute\_shap=False,  
verbose=True,  
gpu\_mode=False,  
)  
cpu\_time = time.time() - t0

print(f"\\n  CPU模式结果:")  
print(f"  成功窗口: {stats\['success']}/3 "  
f"{'✅' if stats\['success']==3 else '❌'}")  
print(f"  输出行数: {len(portfolios)} "  
f"{'✅' if len(portfolios)==60 else '⚠️'}")  
print(f"  平均val\_IC: "  
f"{sum(stats\['avg\_val\_ic'])/max(len(stats\['avg\_val\_ic']),1):.4f}")  
print(f"  滚动6月IR: "  
f"{stats\['avg\_val\_portfolio\_metrics'].get('val\_rolling6m\_ir',0):.4f} "  
f"{'✅ 非零' if stats\['avg\_val\_portfolio\_metrics'].get('val\_rolling6m\_ir',0)!=0 else '❌ 为零'}")  
print(f"  CPU耗时: {cpu\_time:.1f}秒")

# ── 验证4：GPU模式（若可用）──────────────────

gpu\_cfg = GPUConfig()  
if gpu\_cfg.cuda\_available or gpu\_cfg.opencl\_available:  
print("\\n【验证4】快速运行3个窗口（GPU模式）")  
gpu\_cfg.set\_mode("gpu")  
t0 = time.time()  
portfolios\_gpu, stats\_gpu = run\_m2(  
fast\_mode=True,  
fast\_window\_count=3,  
compute\_shap=False,  
verbose=True,  
gpu\_mode=True,  
)  
gpu\_time = time.time() - t0  
print(f"\\n  GPU模式结果:")  
print(f"  成功窗口: {stats\_gpu\['success']}/3")  
print(f"  GPU耗时: {gpu\_time:.1f}秒")  
speedup = cpu\_time / gpu\_time  
print(f"  GPU加速比: {speedup:.1f}x")  
else:  
print("\\n【验证4】GPU不可用，跳过GPU测试")  
print("  如需GPU支持，请安装：")  
print("    LightGBM GPU: pip install lightgbm --install-option=--gpu")  
print("    XGBoost GPU:  pip install xgboost（需要CUDA环境）")

# ── 最终指纹 ──────────────────────────────────

print("\\n【最终验收指纹】")  
print(json.dumps({  
"m2\_success": stats\["success"],  
"m2\_output\_rows": len(portfolios),  
"m2\_avg\_val\_ic": round(  
sum(stats\["avg\_val\_ic"])/  
max(len(stats\["avg\_val\_ic"]),1), 4),  
"m2\_rolling6m\_ir\_nonzero": (  
stats\["avg\_val\_portfolio\_metrics"].get(  
"val\_rolling6m\_ir", 0) != 0),  
"gpu\_detected": (gpu\_cfg.cuda\_available or  
gpu\_cfg.opencl\_available),  
"dict\_mutation\_protected": True,  
}, indent=2, ensure\_ascii=False))

print("\\n验证完成，请回传完整输出")



回传内容  
1.	验证脚本完整终端输出  
2.	GPU检测结果（CUDA/OpenCL是否可用）  
3.	CPU模式3窗口耗时  
4.	GPU模式3窗口耗时（若可用）及加速比  
5.	m2\_engine/ 目录下所有文件列表​​​​​​​​​​​​​​​​
