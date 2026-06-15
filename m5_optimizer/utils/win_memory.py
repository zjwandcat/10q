"""
Windows 内存强制归还工具
调用 SetProcessWorkingSetSize 强制 OS 回收进程未使用的内存页。
在非 Windows 平台自动退化为 no-op。
"""
import sys
import gc
import logging

logger = logging.getLogger("m5.win_memory")


def release_memory_to_os() -> float:
    """
    执行 gc.collect() 后，强制将未使用内存页归还给 Windows。
    返回释放前的进程 RSS（GB），用于日志记录。
    非 Windows 平台直接返回 -1.0。
    """
    import psutil
    rss_before = psutil.Process().memory_info().rss / 1e9

    gc.collect()

    if sys.platform != "win32":
        return rss_before

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # -1 表示让系统自动设置最小/最大工作集大小
        result = kernel32.SetProcessWorkingSetSize(
            kernel32.GetCurrentProcess(),
            ctypes.c_size_t(-1),
            ctypes.c_size_t(-1),
        )
        if result == 0:
            logger.debug("SetProcessWorkingSetSize 调用失败（不影响运行）")
    except Exception as e:
        logger.debug(f"Windows内存归还失败（不影响运行）: {e}")

    rss_after = psutil.Process().memory_info().rss / 1e9
    released = rss_before - rss_after
    if released > 0.1:
        logger.info(f"内存归还: {rss_before:.1f}GB → {rss_after:.1f}GB（释放{released:.1f}GB）")

    return rss_before
