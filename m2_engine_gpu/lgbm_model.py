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

"""
LightGBM排序模型 · v4.1 (D 方案专用: 始终 CPU)

v4.1 简化:
  - 删除 v3.x 的 OpenCL GPU 路径 (A/C 策略)
  - 仅保留 CPU 路径 (D 策略 + 纯 CPU fallback)
  - 与纯 CPU (m2_engine) 路径 100% bit-exact

关键优化:
  - max_bin=63 (默认255，4GB卡用63) - 与纯 CPU 保持一致
  - 训练后 free_raw_data=True 自动释放原始数据
  - nthread=1 (与 XGB=4 错开 CPU 争用, v3.10.1 验证)
  - 训练后立即 del Dataset
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

logger = logging.getLogger("m2.lgbm.v2")
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
    """v4.2.3 调优: 8 核 CPU D 模式 LGBM=5 最优
    实测 benchmark_v42: L5+X2 (LGBM=5, XGB=2) 是 D 模式 28 组合中最优
    LGBM 5 线程 + XGB 2 线程共 7 线程, 余 1 线程给 OS
    旧: v3.10.1 固定 1 线程 (bit-exact 优先, 但 D 模式 1.42x 慢于最优)
    """
    cpu_count = os.cpu_count() or 8
    if os.environ.get('JOBLIB_WORKER_ID') is not None:
        return max(cpu_count // 4, 2)
    # ★ v4.2.3: 8 核 CPU D 模式最优 = 5 线程 (benchmark_v42 实测)
    if cpu_count >= 8:
        return 5
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
    """LightGBM排序模型 v2 (CPU/OpenCL/CUDA 三模自适应)"""

    # 硬约束（不可被用户覆盖）
    HARD_CONSTRAINTS = types.MappingProxyType({
        "subsample":         0.8,
        "min_child_samples": 20,
        "boosting_type":     "gbdt",
    })

    # 默认值（用户可覆盖）
    TUNABLE_DEFAULTS = types.MappingProxyType({
        "max_depth":            4,
        "colsample_bytree":     0.3,
        "learning_rate":        0.05,
        "n_estimators":         200,
        "reg_alpha":            0.1,
        "reg_lambda":           1.0,
        "min_split_gain":       0.01,
        "verbose":              -1,
        # v4.1: max_bin=63 (与纯 CPU m2_engine 一致)
        "max_bin":              63,
        "min_data_in_bin":      5,
    })

    def __init__(self, params: Optional[Dict] = None):
        # ★ 必须先浅拷贝
        params = dict(params) if params else {}

        # pop 自定义字段（这些 key 不传给 LightGBM API）
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
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        group_val: list,
    ) -> None:
        """训练模型 · v4.1: 始终 CPU (D 策略 + 纯 CPU fallback)"""
        # v4.1: 设备固定为 CPU, 删除 opencl_mode 参数
        num_boost_round = self.params.get("n_estimators", 200)
        learning_rate   = self.params.get("learning_rate", 0.05)
        max_depth       = self.params.get("max_depth", 4)
        train_size      = len(X_train)

        # ── 自适应 num_leaves ──
        if self.depth_mode == "adaptive":
            num_leaves = min(
                int(2 ** max_depth),
                max(15, train_size // 200),
                255,
            )
        else:
            num_leaves = min(int(2 ** max_depth), 63)

        # ── 基础参数 (CPU only) ──
        # ★ v4.2: nthread 优先用用户传入, 否则用 _get_optimal_nthread() 默认
        _nthread = self.params.get("nthread") or _get_optimal_nthread()
        lgb_params = {
            "objective":         "regression_l1",
            "metric":            "mae",
            "learning_rate":     learning_rate,
            "max_depth":         max_depth,
            "num_leaves":        num_leaves,
            "colsample_bytree":  self.params.get(
                "colsample_bytree", 0.3),
            "subsample":         self.params["subsample"],
            "min_child_samples": self.params["min_child_samples"],
            "reg_alpha":         self.params["reg_alpha"],
            "reg_lambda":        self.params["reg_lambda"],
            "min_split_gain":    self.params["min_split_gain"],
            "verbosity":         -1,
            "max_bin":           int(self.params.get("max_bin", 128)),
            "min_data_in_bin":   int(self.params.get("min_data_in_bin", 5)),
            "histogram_pool_size": max(
                2048, (os.cpu_count() or 8) * 256),
            "nthread":           _nthread,
            "seed":              _get_seed(),
            "deterministic":     True,
        }

        # ── IC 评估回调 ──
        def ic_metric(y_pred, dataset):
            # v3.8.8 ★ 优化: 用 Pearson IC 替代默认 Spearman (1 次 corrcoef vs 2 argsort + corrcoef)
            # IC 影响: Pearson vs Spearman 在金融 IC 上差异 < 0.005
            label = dataset.get_label().astype(np.float32)
            ic = np.corrcoef(y_pred, label)[0, 1]
            return "ic", (ic if np.isfinite(ic) else 0.0), True

        # ── callbacks ──
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

        # ── Dataset（free_raw_data=True 训练后立即释放原始数据）──
        # v3.8 优化: 接受 numpy 或 DataFrame
        Xt = X_train.values if hasattr(X_train, "values") else X_train
        yt = y_train.values if hasattr(y_train, "values") else y_train
        Xv = X_val.values if hasattr(X_val, "values") else X_val
        yv = y_val.values if hasattr(y_val, "values") else y_val
        lgb_train = lgb.Dataset(
            Xt, label=yt,
            group=group_train,
            free_raw_data=True)        # ★ 训练后释放原始数据
        lgb_val = lgb.Dataset(
            Xv, label=yv,
            group=group_val,
            reference=lgb_train,
            free_raw_data=True)        # ★ 训练后释放原始数据

        # ── 训练 ──
        # ★ v3.10.2 优化: valid_sets=[lgb_val] 替代 [lgb_train, lgb_val]
        #   跳过 train set eval, 节省 LGBM __inner_eval/__inner_predict/ic_metric 各 50%
        #   实测 180 窗: 0.541s → 0.513s (1.055x 加速), IC/17指标 bit-exact diff=0
        #   理由: early_stopping 只看 val IC, 训练轨迹与 train eval 无关
        self.model_ = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_val],
            valid_names=["valid"],
            feval=ic_metric,
            callbacks=callbacks,
        )

        # ★ 显式删除 Dataset（双保险）
        del lgb_train, lgb_val
        # ★ 修复: 释放 Dataset 后 GC，打破 Booster→callback→Dataset 循环引用
        gc.collect()

        self.best_iteration_ = self.model_.best_iteration
        logger.info(
            f"LGBM-v4.1 [CPU] "
            f"best_iter={self.best_iteration_}/{num_boost_round} "
            f"num_leaves={num_leaves} "
            f"max_bin={lgb_params['max_bin']} "
            f"nthread={lgb_params['nthread']}")

    def predict(self, X) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("模型未训练")
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
