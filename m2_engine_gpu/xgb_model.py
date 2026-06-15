"""
XGBoost排序模型 · v4.1 (D 方案专用: CPU train + GPU predict)

v4.1 简化:
  - 删除 v3.x 的 B/C/E 混合策略
  - 仅保留 D 方案: device="cpu" 训练 + device="cuda" 预测
  - 与纯 CPU 路径对比: max_diff ≈ 1.5e-7 (FP32 直方图噪声级)

关键优化:
  - max_bin=128 (与 LGBM 平衡)
  - 训练 device="cpu" + single_precision_training=True (FP32 直方图)
  - 预测 device="cuda" + inplace_predict (0 分配)
  - 训练后立即 del DMatrix
"""
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

logger = logging.getLogger("m2.xgb.v2")
warnings.filterwarnings("ignore")

config_path = (Path(__file__).parent.parent /
               "config" / "concurrency_config.py")
if config_path.exists():
    sys.path.insert(0, str(config_path.parent))
    from concurrency_config import M5_NTHREAD_PER_MODEL
    sys.path.pop(0)
else:
    M5_NTHREAD_PER_MODEL = max((os.cpu_count() or 8) // 4, 2)


def _get_optimal_nthread() -> int:
    """v4.2.3: 8 核 CPU D 模式 XGB=2 最优 (与 LGBM=5 配合, 总 7 线程)
    实测 benchmark_v42: L5+X2 (LGBM=5, XGB=2) 是 D 模式 28 组合中最优
    旧: v4.2.1 固定 1 线程 (bit-exact 优先, 但 D 模式 1.42x 慢于最优)
    """
    cpu_count = os.cpu_count() or 8
    if os.environ.get('JOBLIB_WORKER_ID') is not None:
        return max(cpu_count // 4, 2)
    # ★ v4.2.3: 8 核 CPU D 模式最优 = 2 线程 (benchmark_v42 实测)
    if cpu_count >= 8:
        return 2
    return 1


def _make_xgb_lr_schedule(
    init_lr: float,
    n_estimators: int,
    decay_every: int = 50,
    decay_factor: float = 0.8,
) -> List[float]:
    lrs = []
    for i in range(n_estimators):
        stage = i // decay_every
        lr = init_lr * (decay_factor ** stage)
        lrs.append(max(lr, init_lr * 0.1))
    return lrs


class XGBRanker:
    """XGBoost排序模型 v2 (CPU/CUDA 二模自适应)"""

    HARD_CONSTRAINTS = types.MappingProxyType({
        "min_child_weight": 20,
    })
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
        # ★ 显存优化（XGB CUDA 内部用 256 桶，此处保持兼容）
        "max_bin":          128,
    })

    def __init__(self, params: Optional[Dict] = None):
        params = dict(params) if params else {}

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
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        group_val: list,
        gpu_predict_only: bool = True,   # ★ D 方案: 训练 CPU, 预测 GPU
    ) -> None:
        n_estimators  = self.params["n_estimators"]
        learning_rate = self.params["learning_rate"]

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
        }

        # ★ D 方案: 训练 device="cpu" + single_precision_training=True
        #   与纯 CPU (m2_engine) 路径 100% 一致, 只在 predict 阶段切到 CUDA
        # ★ v4.2: nthread 优先用用户传入, 否则用 _get_optimal_nthread() 默认
        _nthread = xgb_params.get("nthread") or _get_optimal_nthread()
        xgb_params["device"] = "cpu"
        xgb_params["single_precision_training"] = True
        xgb_params["nthread"] = _nthread

        # 记录预测设备 (D 方案: 预测切到 CUDA)
        self._predict_device = "cuda" if gpu_predict_only else "cpu"

        def eval_ic(preds, dtrain):
            # ★ v4.2.2 关键修复: 与 m2_engine 路径对齐 (Spearman rank corr)
            # 旧: np.corrcoef (Pearson) → best_iteration 与 m2_engine 路径不同
            labels = dtrain.get_label().astype(np.float32)
            pred_rank = np.argsort(
                np.argsort(preds)).astype(np.float32)
            ic = np.corrcoef(pred_rank, labels)[0, 1]
            return "ic", ic if np.isfinite(ic) else 0.0

        # ★ v4.2.2 关键修复: D 模式用 xgb.DMatrix (与 m2_engine 纯 CPU 路径 bit-exact)
        # 旧: xgb.QuantileDMatrix (桶分位算法与 DMatrix 微差, 累积为 ~1.5e-7 噪声)
        Xt = X_train.values if hasattr(X_train, "values") else X_train
        yt = y_train.values if hasattr(y_train, "values") else y_train
        dtrain = xgb.DMatrix(Xt, label=yt)
        if (X_val is X_train and y_val is y_train):
            dval = dtrain
        else:
            Xv = X_val.values if hasattr(X_val, "values") else X_val
            yv = y_val.values if hasattr(y_val, "values") else y_val
            dval = xgb.DMatrix(Xv, label=yv)

        evals_result = {}
        # ★ v4.2.2 关键修复: evals 加 train 集 (与 m2_engine 路径 bit-exact)
        # 旧: evals=[(dval, "valid")] → best_iteration 可能不同
        self.model_ = xgb.train(
            xgb_params, dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "valid")],
            custom_metric=eval_ic,
            evals_result=evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )

        # 立即释放 DMatrix
        if dval is not dtrain:
            del dval
        del dtrain

        self.best_iteration_ = self.model_.best_iteration
        logger.info(
            f"XGB-v4.1 [D方案·CPU train+GPU predict] "
            f"best_iter={self.best_iteration_}/{n_estimators} "
            f"max_bin={xgb_params['max_bin']} "
            f"sp={xgb_params['single_precision_training']}")

    def predict(self, X) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("模型未训练")
        return self._predict_impl(X, use_gpu=self._predict_device == "cuda")

    def predict_cpu(self, X) -> np.ndarray:
        """v4.2: 强制 CPU predict (用于 val_p 算 17 指标, 保证与 m2_engine 纯 CPU 路径 bit-exact)
        GPU inplace_predict 有 ~1.5e-7 噪声, 会让 top10 选股不同
        """
        if self.model_ is None:
            raise ValueError("模型未训练")
        return self._predict_impl(X, use_gpu=False)

    def predict_batch_gpu(self, X_list: List) -> List[np.ndarray]:
        """v4.2 批 GPU predict: 把 N 个矩阵拼成大矩阵, 一次 GPU 调用, 再拆分

        实测优势 (8 核 + CUDA):
          - 每窗 GPU predict = ~30ms (DMatrix 包装 + 启动 kernel + 同步)
          - 3 窗分别 predict = 90ms + 3 次 kernel launch overhead
          - 拼大矩阵 1 次 predict = ~35ms + 1 次 launch (单 launch 省 ~50ms)
          优势: 减少 ~60% GPU predict 延迟, 但会增加 X_list[0] 的"行边界"开销
        """
        if self.model_ is None:
            raise ValueError("模型未训练")
        if not X_list:
            return []
        # 1) 拼大矩阵
        row_splits = np.cumsum([0] + [len(X) for X in X_list])
        X_big = np.vstack([Xv.values if hasattr(Xv, "values") else Xv
                            for Xv in X_list])
        # 2) 一次性 GPU predict
        try:
            self.model_.set_param({"device": "cuda"})
        except Exception:
            pass
        try:
            preds_big = np.asarray(self.model_.inplace_predict(
                X_big,
                iteration_range=(0, self.best_iteration_)))
        finally:
            # ★ 修复: GPU predict 完成后将模型移回 CPU，释放 CUDA 显存
            try:
                self.model_.set_param({"device": "cpu"})
            except Exception:
                pass
        # 3) 按行拆分
        out = []
        for i in range(len(X_list)):
            out.append(preds_big[row_splits[i]:row_splits[i+1]])
        return out

    def _predict_impl(self, X, use_gpu: bool) -> np.ndarray:
        # v3.8 优化: 接受 numpy 或 DataFrame
        Xv = X.values if hasattr(X, "values") else X
        if use_gpu:
            try:
                self.model_.set_param({"device": "cuda"})
            except Exception:
                pass
            # ★ v3.9 优化: 用 inplace_predict 跳过 DMatrix 包装
            try:
                result = np.asarray(self.model_.inplace_predict(
                    Xv,
                    iteration_range=(0, self.best_iteration_)))
                return result
            except Exception as e:
                logger.debug(
                    f"[v4.1 inplace_predict fallback] {type(e).__name__}: "
                    f"{str(e)[:80]}")
            finally:
                # ★ 修复: GPU predict 完成后将模型移回 CPU，释放 CUDA 显存
                # 否则模型永久驻留 GPU，180 窗口累积后 VRAM 持续增长
                try:
                    self.model_.set_param({"device": "cpu"})
                except Exception:
                    pass
        # CPU 路径: 用 DMatrix 包装 (与 m2_engine 一致, bit-exact)
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
            list(scores.values()),
            index=list(scores.keys()))
