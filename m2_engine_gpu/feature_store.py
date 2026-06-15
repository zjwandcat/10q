"""
M2 GPU v2 · 特征工程模块（直接复用 m2_engine 版本）

特征工程是纯 CPU 操作，无 GPU 依赖，直接 re-export。
"""
from m2_engine.feature_store import FeatureStore

__all__ = ["FeatureStore"]
