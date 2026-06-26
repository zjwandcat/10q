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
Windows 内存强制归还工具
调用 SetProcessWorkingSetSize 强制 OS 回收进程未使用的内存页。
在非 Windows 平台自动退化为 no-op。
★ v4.2: 增加 msvcrt._heapmin() 强制把 C 堆归还给 Windows
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

    # ★ v4.2: full GC（含老年代），否则只清 0 代对累计大对象无效
    # 旧版 gc.collect() == gc.collect(0)，只清 youngest generation
    # 全代 GC 对周期性释放 LGBM/XGB Booster 等大对象更彻底
    try:
        gc.collect(2)
    except Exception:
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

    # ★ v4.2: msvcrt._heapmin() - 强制把 C 堆 free 后的内存归还 OS
    # 原因: Python 的 pymalloc 不直接调 free(), 所以 SetProcessWorkingSetSize
    #       看不到这部分内存。msvcrt._heapmin() 触发 CRT 堆整理, 真正释放。
    # 效果: 配合 gc.collect(2) 通常能再降 200-500MB RSS
    try:
        import msvcrt
        msvcrt._heapmin()
    except Exception as e:
        logger.debug(f"msvcrt._heapmin 失败（非Windows或CRT不可用）: {e}")

    rss_after = psutil.Process().memory_info().rss / 1e9
    released = rss_before - rss_after
    if released > 0.1:
        logger.info(f"内存归还: {rss_before:.1f}GB → {rss_after:.1f}GB（释放{released:.1f}GB）")

    return rss_before
