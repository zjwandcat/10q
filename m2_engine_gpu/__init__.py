"""
M2 GPU 选股算法引擎 · v2.0 (GTX 1650 4GB 显存优化版)

与 m2_engine/ 并列运行，完全不修改原 CPU 版代码。

支持 3 种 CUDA 混合策略（benchmark 用）：
  A: LGBM CUDA + XGB CPU        (显存~1.0GB)
  B: LGBM CPU  + XGB CUDA       (显存~1.5GB)
  C: LGBM CUDA + XGB CUDA 串行  (显存~2.5GB, OOM 风险)

显存优化（必开）：
  - LGBM: max_bin=63, gpu_use_dp=False, use_quantized_grad=True, gpu_max_memory=0.6
  - XGB:  max_bin=128, tree_method=hist

训练后自动释放原始数据 (free_raw_data=True + del + gc.collect)
"""
from .run_m2 import run_m2
from .gpu_detector import GPUConfig, STRATEGIES

__all__ = ["run_m2", "GPUConfig", "STRATEGIES"]
