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
GPU检测与配置模块 · v4.1 (仅 D 策略 · 与纯 CPU bit-exact)

历史策略枚举（v4.1 已删除 B 和 E，仅保留 D + 纯 CPU fallback）:
  A: LGBM OpenCL + XGB CPU             ← v4.1 已删除 (OpenCL 路径与 4.6 wheel 兼容性差)
  B: LGBM CPU    + XGB CUDA            ← v4.1 已删除 (CUDA hist 与 CPU hist 算法层 2.47 差异)
  C: LGBM OpenCL + XGB CUDA (serial)   ← v4.1 已删除 (双 GPU + OOM 风险)
  D: LGBM CPU + XGB CPU train + GPU predict  ★ 保留 (与纯 CPU 噪声级 1.5e-7 差异)
  E: LGBM CPU + XGB CUDA train + CPU predict  ★ v3.8 < v4.1 已删除

D 方案设计原理:
  - XGB 训练走 device="cpu" + single_precision_training=True
  - XGB 预测走 device="cuda" (inplace_predict, 0 分配)
  - LGBM 始终 CPU
  - 与纯 CPU (m2_engine) 路径对比: max_diff ≈ 1.5e-7 (FP32 直方图 vs FP64 噪声级)
  - 17 金融指标在 1e-4 精度内 bit-exact
"""
import logging

logger = logging.getLogger("m2.gpu.v2")


# ★ v4.1: 策略集仅 D (用户要求: 与纯 CPU bit-exact)
STRATEGIES = ("D",)

STRATEGY_DESC = {
    "D": "LGBM CPU + XGB CPU train + GPU predict  ★ 与纯CPU bit-exact",
}


class GPUConfig:
    """
    GPU配置管理器 v2
    单例模式，全局统一管理 3 种混合策略
    """
    _instance = None
    _mode = None
    _strategy = None
    _cuda_available = None
    _opencl_available = None
    _vram_total_mb = None
    _vram_free_mb = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._detect()
        return cls._instance

    def _detect(self):
        """检测 GPU 可用性（CUDA / OpenCL）+ 显存"""
        import numpy as np

        # ── 1. 检测 CUDA（XGBoost 预测路径用）────────
        self._cuda_available = False
        try:
            import xgboost as xgb
            X = np.random.randn(100, 10).astype(np.float32)
            y = np.random.randn(100).astype(np.float32)
            dm = xgb.DMatrix(X, label=y)
            params = {
                "tree_method": "hist",
                "device":      "cuda",
                "verbosity":    0,
            }
            xgb.train(params, dm, num_boost_round=1,
                      verbose_eval=False)
            self._cuda_available = True
            logger.info("CUDA 可用（XGBoost GPU 模式）")
        except Exception as e:
            logger.info(f"CUDA 不可用: {e}")

        # ── 2. 检测 LightGBM OpenCL（v4.1 已不使用，保留检测仅用于诊断）────────
        self._opencl_available = False
        try:
            import lightgbm as lgb
            import io
            import contextlib
            fbuf = io.StringIO()
            with contextlib.redirect_stderr(fbuf):
                X = np.random.randn(100, 10).astype(np.float32)
                y = np.random.randn(100).astype(np.float32)
                ds = lgb.Dataset(X, label=y, free_raw_data=False)
                params = {
                    "device_type": "gpu",
                    "verbosity":   -1,
                    "num_leaves":  4,
                }
                lgb.train(params, ds, num_boost_round=1)
            self._opencl_available = True
            logger.info("LightGBM OpenCL 可用（v4.1 已不用于训练，仅诊断）")
        except Exception as e:
            err_msg = str(e)[:120]
            if "Fatal" not in err_msg and "USE_CUDA" not in err_msg:
                logger.info(f"LightGBM OpenCL 不可用: "
                            f"{type(e).__name__}: {err_msg}")
            else:
                logger.info(f"LightGBM OpenCL 不可用: {err_msg.split(chr(10))[0]}")

        # ── 3. 显存检测（nvidia-smi 解析）────────
        self._vram_total_mb, self._vram_free_mb = (
            self._query_vram())

        # ── 4. 默认模式选择 (v4.1: 仅 D 策略) ──
        # D 方案需要 CUDA (用于 XGB predict)
        if self._cuda_available:
            self._mode = "gpu"
            self._strategy = "D"
        else:
            self._mode = "cpu"
            self._strategy = None

        logger.info(
            f"默认 mode={self._mode}, strategy={self._strategy}")
        if self._vram_total_mb:
            logger.info(
                f"GPU 显存: {self._vram_free_mb}/{self._vram_total_mb}MB 可用"
            )

    def _query_vram(self):
        """通过 nvidia-smi 查显存（MB），失败返回 (None, None)"""
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                timeout=5,
                stderr=subprocess.DEVNULL,
            ).decode().strip().splitlines()[0]
            total, free = out.split(",")
            return int(total.strip()), int(free.strip())
        except Exception:
            return None, None

    # ── getter / setter ──
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def cuda_available(self) -> bool:
        return self._cuda_available

    @property
    def opencl_available(self) -> bool:
        return self._opencl_available

    # ── getter / setter ──
    @property
    def vram_total_mb(self) -> int:
        return self._vram_total_mb or 0

    @property
    def vram_free_mb(self) -> int:
        return self._vram_free_mb or 0

    def set_mode(self, mode: str):
        if mode not in ("gpu", "cpu"):
            raise ValueError("mode必须是'gpu'或'cpu'")
        self._mode = mode
        if mode == "cpu":
            self._strategy = None

    def set_strategy(self, strategy: str):
        """
        设置混合策略 · v4.1: 仅 D 可用
        调用方传入 B/E 等历史策略时, 自动 fallback 到 D 或纯 CPU
        """
        if strategy not in STRATEGIES:
            # v4.1: 兼容历史 config (B/E) - 自动 fallback
            if strategy in ("A", "B", "C", "E"):
                logger.warning(
                    f"策略 {strategy} 已在 v4.1 中移除, fallback 到 D")
                strategy = "D"
            else:
                raise ValueError(f"strategy 必须是 {STRATEGIES}")
        if strategy == "D" and not self._cuda_available:
            logger.warning("策略 D 需要 CUDA (用于 XGB predict)，fallback 到纯 CPU")
            self._mode = "cpu"
            self._strategy = None
            return
        self._mode = "gpu"
        self._strategy = strategy
        logger.info(f"策略已切换: {strategy} ({STRATEGY_DESC[strategy]})")

    def _fallback_strategy(self, exclude: str):
        """v4.1: 仅 D 一个策略, 不可 fallback, 退回纯 CPU"""
        logger.warning("无可用 fallback 策略, 退回纯 CPU")
        self._mode = "cpu"
        return None

    def get_lgbm_device(self) -> str:
        """v4.1: LGBM 始终走 CPU"""
        return "cpu"

    def get_xgb_train_device(self) -> str:
        """v4.1: XGB 训练始终走 CPU (D 方案)"""
        if self._mode != "gpu":
            return "cpu"
        return "cpu"   # D 方案: train 在 CPU

    def get_xgb_predict_device(self) -> str:
        """XGB 预测 device
        v4.1 D 方案: 训练 CPU, 预测 GPU
        """
        if self._mode != "gpu":
            return "cpu"
        if self._strategy == "D":
            return "cuda"
        return "cpu"

    def get_xgb_device(self) -> str:
        """向后兼容：返回 XGB 训练 device"""
        return self.get_xgb_train_device()

    def is_gpu_mode(self) -> bool:
        return self._mode == "gpu"

    def summary(self) -> str:
        return (
            f"GPUConfig: mode={self._mode}, "
            f"strategy={self._strategy}, "
            f"CUDA={self._cuda_available}, "
            f"OpenCL={self._opencl_available}, "
            f"VRAM={self._vram_free_mb}/{self._vram_total_mb}MB"
        )
