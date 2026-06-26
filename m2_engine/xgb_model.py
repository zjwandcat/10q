# Copyright 2026 zjwandcat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PEP 683: 使用 frozenset/MappingProxyType 替代可变 set/dict，避免 GIL refcount 开销
"""
XGBoost排序模型
支持CUDA GPU / CPU自适应
支持学习率衰减/可配置早停

★ __init__必须先dict()拷贝params再.pop()
★ predict必须使用iteration_range=(0, best_iteration)
★ GPU模式使用device='cuda'（XGBoost 2.0+）
"""
import gc
import types

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import warnings
import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger("m2.xgb")
warnings.filterwarnings("ignore")


def _get_seed() -> int:
    try:
        import yaml as _yaml
        _cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if _cfg_path.exists():
            with open(_cfg_path, encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f)
            return int(_cfg.get("m5", {}).get("optimization", {}).get("seed", 42))
    except Exception:
        pass
    return 42

config_path = (Path(__file__).parent.parent /
               "config" / "concurrency_config.py")
if config_path.exists():
    sys.path.insert(0, str(config_path.parent))
    from concurrency_config import M5_NTHREAD_PER_MODEL
    sys.path.pop(0)
else:
    M5_NTHREAD_PER_MODEL = max((os.cpu_count() or 8) // 4, 2)


def _get_optimal_nthread() -> int:
    """v4.2.3: 8 核 CPU 纯 CPU 最优 = 3 线程 (与 LGBM=2 配合, 总 5 线程)
    实测 benchmark_v42: L2+X3 (LGBM=2, XGB=3) 是纯 CPU 28 组合中最优
    旧: v4.2.1 固定 1 线程 (bit-exact 优先, 但 1.42x 慢于最优)
    """
    cpu_count = os.cpu_count() or 8
    if os.environ.get('JOBLIB_WORKER_ID') is not None:
        return max(cpu_count // 4, 2)
    # ★ v4.2.3: 8 核 CPU 纯 CPU 最优 = 3 线程 (benchmark_v42 实测)
    if cpu_count >= 8:
        return 3
    return 1


def _make_xgb_lr_schedule(
    init_lr: float,
    n_estimators: int,
    decay_every: int = 50,
    decay_factor: float = 0.8,
) -> List[float]:
    """XGBoost学习率阶梯衰减调度"""
    lrs = []
    for i in range(n_estimators):
        stage = i // decay_every
        lr = init_lr * (decay_factor ** stage)
        lrs.append(max(lr, init_lr * 0.1))
    return lrs


class XGBRanker:
    """XGBoost排序模型（CUDA GPU/CPU自适应）"""

    # PEP 683: MappingProxyType 替代可变 dict，避免 GIL refcount 开销
    HARD_CONSTRAINTS = types.MappingProxyType({"min_child_weight": 20})
    TUNABLE_DEFAULTS = types.MappingProxyType({
        "max_depth":        4,
        "colsample_bytree": 0.3,
        "learning_rate":    0.05,
        "n_estimators":     200,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "gamma":            0.01,
        "subsample":        0.8,
        "verbosity":        0,
        "tree_method":      "hist",
    })

    def __init__(self, params: Optional[Dict] = None):
        # ★ 必须先浅拷贝
        params = dict(params) if params else {}

        # ★ pop提取自定义字段
        self.lr_mode = params.pop("lr_mode", "fixed")
        self.decay_every = params.pop("decay_every", 50)
        self.decay_factor = params.pop("decay_factor", 0.8)
        self.early_stopping_rounds = params.pop(
            "early_stopping_rounds", 30)

        self.params = self.TUNABLE_DEFAULTS.copy()
        self.params.update(params)
        self.params.update(self.HARD_CONSTRAINTS)

        self.model_: Optional[xgb.Booster] = None
        self.best_iteration_: int = 0
        self.train_ic_: float = 0.0
        self.val_ic_:   float = 0.0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        group_train: list,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        group_val: list,
        gpu_mode: bool = False,
    ) -> None:
        n_estimators  = self.params["n_estimators"]
        learning_rate = self.params["learning_rate"]

        # ★ 使用reg:absoluteerror（MAE）替代reg:squarederror
        # MAE对异常值更鲁棒，rank相关性更强
        # y_train/y_val保持原始label_rank浮点数（0~1）

        xgb_params = {
            "objective":        "reg:absoluteerror",
            "learning_rate":    learning_rate,
            "max_depth":        self.params["max_depth"],
            "colsample_bytree": self.params["colsample_bytree"],
            "subsample":        self.params["subsample"],
            "min_child_weight": self.params["min_child_weight"],
            "reg_alpha":        self.params["reg_alpha"],
            "reg_lambda":       self.params["reg_lambda"],
            "gamma":            self.params["gamma"],
            "tree_method":      "hist",
            "verbosity":        0,
            "max_bin":          int(self.params.get("max_bin", 128)),
            "seed":             _get_seed(),
        }

        if gpu_mode:
            xgb_params["device"] = "cuda"
        else:
            xgb_params["device"] = "cpu"  # ★ v3.8: 显式 device=cpu (与 m2_engine_gpu 路径一致)
            xgb_params["single_precision_training"] = True  # ★ v3.8: 单精度 (与 m2_engine_gpu D 方案一致)
        # ★ v4.2.2 关键修复: nthread 必须在 if 块外设置 (gpu_mode 也要设)
        # ★ v4.2.2: 优先用用户传入
        xgb_params["nthread"] = xgb_params.get("nthread") or _get_optimal_nthread()

        # 自定义IC评估函数
        def eval_ic(preds, dtrain):
            labels = dtrain.get_label().astype(np.float32)
            pred_rank = np.argsort(
                np.argsort(preds)).astype(np.float32)
            ic = np.corrcoef(pred_rank, labels)[0, 1]
            return "ic", ic if np.isfinite(ic) else 0.0

        # ★ 兼容 numpy ndarray
        Xt = X_train.values if hasattr(X_train, "values") else X_train
        yt = y_train.values if hasattr(y_train, "values") else y_train
        Xv = X_val.values if hasattr(X_val, "values") else X_val
        yv = y_val.values if hasattr(y_val, "values") else y_val
        dtrain = xgb.DMatrix(Xt,
                             label=yt)  # ★ 浮点数label_rank
        # set_group已删除：reg:absoluteerror(MAE回归)不使用group信息
        dval   = xgb.DMatrix(Xv,
                             label=yv)  # ★ 浮点数label_rank

        # 注意：XGBoost新版本不支持learning_rates参数
        # 学习率衰减通过callbacks实现（如果需要）
        evals_result = {}
        self.model_ = xgb.train(
            xgb_params, dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain,"train"),(dval,"valid")],
            custom_metric=eval_ic,
            evals_result=evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )

        # ★ 早停回滚
        self.best_iteration_ = self.model_.best_iteration
        device_str = "GPU(CUDA)" if gpu_mode else "CPU"
        # ★ v3.8: 与 m2_engine_gpu 路径对齐 log
        logger.info(
            f"XGB-v2 [{device_str}] "
            f"best_iter={self.best_iteration_}/{n_estimators} "
            f"max_bin={xgb_params['max_bin']} "
            f"sp={xgb_params.get('single_precision_training', False)}")

        # IC已在ensemble.py的_monthly_ic中计算，此处无需重复
        self.train_ic_ = 0.0
        self.val_ic_   = 0.0

        # ★ 修复: 显式删除 DMatrix + gc.collect() 打破循环引用
        try:
            if dval is not dtrain:
                del dval
            del dtrain
        except (NameError, UnboundLocalError):
            pass
        gc.collect()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """★ 使用iteration_range，不用默认的最后一轮"""
        if self.model_ is None:
            raise ValueError("模型未训练")
        # 兼容 numpy ndarray
        Xv = X.values if hasattr(X, "values") else X
        dpred = xgb.DMatrix(Xv)
        try:
            return self.model_.predict(
                dpred,
                iteration_range=(0, self.best_iteration_))
        finally:
            del dpred

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.Series:
        scores = self.model_.get_score(
            importance_type=importance_type)
        return pd.Series(
            scores.values(),
            index=scores.keys())
