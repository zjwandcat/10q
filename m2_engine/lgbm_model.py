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
LightGBM排序模型
支持GPU（OpenCL）/ CPU自适应
支持学习率衰减/自适应深度/可配置早停

关键规则：
★ __init__必须先dict()拷贝params再.pop()
★ predict必须使用best_iteration（早停回滚）
★ GPU模式下不与XGBoost并行（共用GTX1650显存）
"""
import gc
import types

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger("m2.lgbm")
warnings.filterwarnings("ignore")

# 自适应线程配置
config_path = (Path(__file__).parent.parent /
               "config" / "concurrency_config.py")
if config_path.exists():
    sys.path.insert(0, str(config_path.parent))
    from concurrency_config import (
        M5_NTHREAD_PER_MODEL, GLOBAL_NTHREAD_INNER)
    sys.path.pop(0)
else:
    _cores = os.cpu_count() or 8
    M5_NTHREAD_PER_MODEL = max(_cores // 4, 2)
    GLOBAL_NTHREAD_INNER = max(_cores // 4, 2)


def _get_optimal_nthread() -> int:
    """v4.2.1+ 调优: 8 核 CPU 实测纯 CPU 模式 LGBM=2 最优
    实测 28 种 L+X 组合 (LGBM=1~7, XGB=1~7, L+X ≤ 8), 纯 CPU 最优配置:
    L2+X3 (LGBM=2, XGB=3) → 0.762s/窗
    LGBM 2 线程 + XGB 3 线程共 5 线程, 余 3 线程给 OS/数据加载
    19 金融指标 CPU/D 模式差异 ~1e-6 量级 (thread-scheduling 噪声, 金融可忽略)
    旧: v4.2.1 固定 1 线程 (bit-exact 优先, 但 1.42x 慢于最优)
    """
    cpu_count = os.cpu_count() or 8
    if os.environ.get('JOBLIB_WORKER_ID') is not None:
        return max(cpu_count // 4, 2)
    # ★ v4.2.3: 8 核 CPU 纯 CPU 最优 = 2 线程 (benchmark_v42 实测)
    if cpu_count >= 8:
        return 2
    return 1


def _make_lr_decay_callback(
    init_lr: float,
    decay_every: int = 50,
    decay_factor: float = 0.8,
):
    """LightGBM学习率阶梯衰减回调"""
    def callback(env):
        if (env.iteration > 0 and
            env.iteration % decay_every == 0):
            stage = env.iteration // decay_every
            new_lr = init_lr * (decay_factor ** stage)
            new_lr = max(new_lr, init_lr * 0.1)
            env.model.reset_parameter(
                {"learning_rate": new_lr})
    callback.order = 10
    return callback


class LGBMRanker:
    """LightGBM排序模型（GPU/CPU自适应）"""

    # PEP 683: MappingProxyType 替代可变 dict，避免 GIL refcount 开销
    HARD_CONSTRAINTS = types.MappingProxyType({
        "subsample": 0.8,
        "min_child_samples": 20,
        "boosting_type": "gbdt",
    })
    TUNABLE_DEFAULTS = types.MappingProxyType({
        "max_depth": 4,
        "colsample_bytree": 0.3,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.01,
        "verbose": -1,
    })

    def __init__(self, params: Optional[Dict] = None):
        # ★ 必须先浅拷贝，禁止原地修改调用方的dict
        params = dict(params) if params else {}

        # ★ pop提取自定义字段（这些key不能传给LightGBM API）
        self.lr_mode = params.pop("lr_mode", "fixed")
        self.decay_every = params.pop("decay_every", 50)
        self.decay_factor = params.pop("decay_factor", 0.8)
        self.depth_mode = params.pop("depth_mode", "fixed")
        self.early_stopping_rounds = params.pop(
            "early_stopping_rounds", 30)

        # 合并参数：默认值 < 用户传入 < 硬约束
        self.params = self.TUNABLE_DEFAULTS.copy()
        self.params.update(params)
        self.params.update(self.HARD_CONSTRAINTS)

        self.model_: Optional[lgb.Booster] = None
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
        """
        训练模型
        gpu_mode: True时使用OpenCL GPU加速
        """
        # GPU/CPU设备配置
        device_type = "gpu" if gpu_mode else "cpu"

        num_boost_round = self.params.get("n_estimators", 200)
        learning_rate   = self.params.get("learning_rate", 0.05)
        max_depth       = self.params.get("max_depth", 4)
        train_size      = len(X_train)

        # 自适应深度（num_leaves）
        if self.depth_mode == "adaptive":
            num_leaves = min(
                int(2 ** max_depth),
                max(15, train_size // 200),
                255,
            )
        else:
            num_leaves = min(int(2 ** max_depth), 63)

        # ★ 使用regression_l1（MAE）替代lambdarank
        # regression_l1对排名更鲁棒（对异常值不敏感），且避免
        # lambdarank在多线程/多Trial迭代中的native内存崩溃
        # y_train/y_val保持原始label_rank浮点数（0~1）

        lgb_params = {
            "objective":        "regression_l1",
            "metric":           "mae",
            "learning_rate":    learning_rate,
            "max_depth":        max_depth,
            "num_leaves":       num_leaves,
            "colsample_bytree": self.params.get(
                "colsample_bytree", 0.3),
            "subsample":        self.params["subsample"],
            "min_child_samples":self.params["min_child_samples"],
            "reg_alpha":        self.params["reg_alpha"],
            "reg_lambda":       self.params["reg_lambda"],
            "min_split_gain":   self.params["min_split_gain"],
            "verbosity":        -1,
            "device_type":      device_type,
            "histogram_pool_size": max(2048, (os.cpu_count() or 8) * 256),
            "max_bin":          int(self.params.get("max_bin", 128)),   # ★ v3.8: 读 params, 与 m2_engine_gpu 路径一致 (之前硬编码 128)
            "min_data_in_bin":  int(self.params.get("min_data_in_bin", 5)),  # ★ v3.8: 与 m2_engine_gpu 一致
        }

        # GPU模式不设nthread（GPU自管理线程）
        # CPU模式设nthread
        if not gpu_mode:
            # ★ v4.2: nthread 优先用用户传入
            lgb_params["nthread"] = lgb_params.get("nthread") or _get_optimal_nthread()

        # 自定义IC评估回调（用于监控，非训练目标）
        def ic_metric(y_pred, dataset):
            label = dataset.get_label().astype(np.float32)
            ic = np.corrcoef(y_pred, label)[0, 1]
            return "ic", (ic if np.isfinite(ic) else 0.0), True

        # 构建callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                verbose=False),
            lgb.log_evaluation(period=100),
        ]
        if self.lr_mode == "decay":
            callbacks.append(_make_lr_decay_callback(
                learning_rate,
                self.decay_every,
                self.decay_factor))

        lgb_train = lgb.Dataset(
            X_train.values, label=y_train.values,  # ★ 浮点数label_rank
            group=group_train)
        lgb_val = lgb.Dataset(
            X_val.values, label=y_val.values,  # ★ 浮点数label_rank
            group=group_val, reference=lgb_train)

        self.model_ = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "valid"],
            feval=ic_metric,
            callbacks=callbacks,
        )

        # ★ 早停回滚：使用best_iteration
        self.best_iteration_ = self.model_.best_iteration
        logger.info(
            f"LGBM [{device_type.upper()}] "
            f"best_iter={self.best_iteration_}/"
            f"{num_boost_round} "
            f"num_leaves={num_leaves}")

        # IC已在ensemble.py的_monthly_ic中计算，此处无需重复
        self.train_ic_ = 0.0
        self.val_ic_   = 0.0

        # ★ 修复: 显式删除 Dataset + gc.collect() 打破循环引用
        # Booster→callback→Dataset 循环导致 C 原生内存延迟释放
        try:
            del lgb_train, lgb_val
        except (NameError, UnboundLocalError):
            pass
        gc.collect()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """★ 使用best_iteration预测，不用默认的最后一轮"""
        if self.model_ is None:
            raise ValueError("模型未训练")
        # 兼容 numpy ndarray
        Xv = X.values if hasattr(X, "values") else X
        return self.model_.predict(
            Xv,
            num_iteration=self.best_iteration_)

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.Series:
        importance = self.model_.feature_importance(
            importance_type=importance_type)
        return pd.Series(
            importance,
            index=self.model_.feature_name())
