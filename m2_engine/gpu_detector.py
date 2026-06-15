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
GPU检测与配置模块
自适应检测CUDA/OpenCL可用性
支持运行时切换GPU/CPU模式

GTX 1650 4GB VRAM限制：
* 单模型训练约1.5GB VRAM
* LGBM和XGB必须串行使用GPU（并行会OOM）
* CPU模式下LGBM+XGB并行（ThreadPoolExecutor）
"""
import logging

logger = logging.getLogger("m2.gpu")


class GPUConfig:
    """
    GPU配置管理器
    单例模式，全局统一管理GPU/CPU模式
    """
    _instance = None
    _mode = None          # "gpu" or "cpu"
    _cuda_available = None
    _opencl_available = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._detect()
        return cls._instance

    def _detect(self):
        """检测GPU可用性"""
        # 检测CUDA（XGBoost使用）
        self._cuda_available = False
        try:
            import xgboost as xgb
            # 用小数据测试CUDA是否真正可用
            import numpy as np
            X = np.random.randn(100, 10).astype(np.float32)
            y = np.random.randn(100).astype(np.float32)
            dm = xgb.DMatrix(X, label=y)
            params = {
                "tree_method": "hist",
                "device": "cuda",
                "verbosity": 0,
            }
            xgb.train(params, dm, num_boost_round=1,
                     verbose_eval=False)
            self._cuda_available = True
            logger.info("✅ CUDA可用（XGBoost GPU模式）")
        except Exception as e:
            logger.info(f"CUDA不可用: {e}")

        # 检测OpenCL（LightGBM使用）
        self._opencl_available = False
        try:
            import lightgbm as lgb
            import numpy as np
            X = np.random.randn(100, 10).astype(np.float32)
            y = np.random.randn(100).astype(np.float32)
            ds = lgb.Dataset(X, label=y, free_raw_data=False)
            params = {
                "device_type": "gpu",
                "verbosity": -1,
                "num_leaves": 4,
            }
            lgb.train(params, ds, num_boost_round=1)
            self._opencl_available = True
            logger.info("✅ OpenCL可用（LightGBM GPU模式）")
        except Exception as e:
            logger.info(f"OpenCL不可用: {e}")

        # 自动选择默认模式
        if self._cuda_available or self._opencl_available:
            self._mode = "gpu"
            logger.info("默认模式: GPU")
        else:
            self._mode = "cpu"
            logger.info("默认模式: CPU（无可用GPU）")

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def cuda_available(self) -> bool:
        return self._cuda_available

    @property
    def opencl_available(self) -> bool:
        return self._opencl_available

    def set_mode(self, mode: str):
        """手动设置模式（供M5 UI调用）"""
        if mode not in ("gpu", "cpu"):
            raise ValueError("mode必须是'gpu'或'cpu'")
        if mode == "gpu" and not (
                self._cuda_available or self._opencl_available):
            logger.warning("GPU不可用，强制使用CPU模式")
            self._mode = "cpu"
            return
        self._mode = mode
        logger.info(f"运行模式已切换: {mode}")

    def set_strategy(self, strategy: str):
        """v4.2.2 关键修复: 接受 strategy 字符串 (A/B/C/D/E)
        - "D" = CPU train + GPU predict (与 m2_engine_gpu 路径一致)
        - "A"/"B"/"C" = GPU train + GPU predict
        - None = auto (默认检测)
        """
        strategy_map = {
            "D":   "cpu",       # D 方案: 训练 CPU
            "E":   "cpu",       # E 方案: 训练 CPU
            "A":   "gpu",       # A 方案: 训练 GPU
            "B":   "gpu",       # B 方案: 训练 GPU
            "C":   "gpu",       # C 方案: 训练 GPU
            "cpu": "cpu",
            "gpu": "gpu",
        }
        if strategy not in strategy_map:
            raise ValueError(
                f"strategy 必须是 {list(strategy_map.keys())} 之一")
        self.set_mode(strategy_map[strategy])

    def get_lgbm_device(self) -> str:
        """获取LightGBM设备类型"""
        if self._mode == "gpu" and self._opencl_available:
            return "gpu"
        return "cpu"

    def get_xgb_device(self) -> str:
        """获取XGBoost设备类型"""
        if self._mode == "gpu" and self._cuda_available:
            return "cuda"
        return "cpu"

    def is_gpu_mode(self) -> bool:
        return self._mode == "gpu"

    def summary(self) -> str:
        return (
            f"GPU配置: mode={self._mode}, "
            f"CUDA={self._cuda_available}, "
            f"OpenCL={self._opencl_available}"
        )
