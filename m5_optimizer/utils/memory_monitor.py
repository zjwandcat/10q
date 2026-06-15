"""
M5内存监控工具
"""
import os
import psutil


def get_memory_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def check_memory(limit_gb: float = 9.5, logger=None) -> bool:
    mem = get_memory_gb()
    if mem > limit_gb:
        if logger:
            logger.warning(f"内存超限:{mem:.2f}GB>{limit_gb}GB")
        return False
    return True
