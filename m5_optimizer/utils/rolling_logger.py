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
M5滚动日志系统 - 崩溃定位专用

特性：
- 双通道写入：关键事件同步flush（崩溃安全），普通事件异步队列（不阻塞）
- 按小时滚动：TimedRotatingFileHandler，仅保留1个备份（最近1小时）
- 异常安全：所有公开方法内部try-except，日志失败不影响业务
- 结构化格式：每条日志为 时间戳 | 级别 | 事件类型 | JSON负载
- VRAM监控：记录GPU显存水位（如有pynvml）
"""
import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class RollingLogger:
    """M5滚动日志记录器（崩溃定位专用）

    ★ 双通道设计:
    - 同步通道 (_sync_file): trial_start/trial_end/critical 直接 write+flush,
      进程被强杀时日志不丢失 (每条 <0.5ms)
    - 异步通道 (QueueListener): window_progress/system_stats 等低优先级日志,
      不阻塞业务线程
    """

    def __init__(self):
        self._logger: Optional[logging.Logger] = None
        self._listener: Optional[logging.handlers.QueueListener] = None
        self._initialized: bool = False
        # ★ 同步写入文件句柄（崩溃安全）
        self._sync_file = None
        self._sync_lock = threading.Lock()

    def setup(self, log_dir: str = "logs/m5") -> None:
        if self._initialized:
            return
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            logger = logging.getLogger("m5.rolling")
            logger.setLevel(logging.DEBUG)
            logger.propagate = False

            if logger.handlers:
                self._logger = logger
                self._initialized = True
                return

            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=os.path.join(log_dir, "m5_rolling.log"),
                when="H",
                interval=1,
                backupCount=1,
                encoding="utf-8",
                delay=False,
            )
            file_handler.setLevel(logging.DEBUG)

            fmt = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(fmt)

            log_queue = queue.Queue(maxsize=1000)
            queue_handler = logging.handlers.QueueHandler(log_queue)
            logger.addHandler(queue_handler)

            listener = logging.handlers.QueueListener(
                log_queue, file_handler
            )
            listener.start()

            self._logger = logger
            self._listener = listener

            # ★ 同步写入文件（崩溃安全，独立于 QueueListener）
            sync_path = os.path.join(log_dir, "m5_crash_safe.log")
            self._sync_file = open(sync_path, "a", encoding="utf-8")

            self._initialized = True

        except Exception as e:
            print(f"[rolling_logger] 初始化失败，降级为no-op: {e}",
                  file=sys.stderr)
            self._initialized = False

    def _sync_write(self, level: str, message: str) -> None:
        """★ 同步写入+flush，进程被强杀时日志不丢失（每条 <0.5ms）"""
        if not self._initialized or self._sync_file is None:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{ts} | {level:<8s} | {message}\n"
            with self._sync_lock:
                self._sync_file.write(line)
                self._sync_file.flush()
                os.fsync(self._sync_file.fileno())
        except Exception:
            pass

    def log_trial_start(self, trial_number: int, params: Dict[str, Any]) -> None:
        if not self._initialized:
            return
        try:
            safe_params = {}
            for k, v in params.items():
                if isinstance(v, float):
                    safe_params[k] = round(v, 6)
                else:
                    safe_params[k] = v
            payload = json.dumps(
                {"trial_number": trial_number, "params": safe_params},
                ensure_ascii=False,
            )
            # ★ 同步写入（崩溃安全）
            self._sync_write("INFO", f"TRIAL_START | {payload}")
            # 异步通道也写一份（供 m5_rolling.log 滚动查看）
            self._logger.info(f"TRIAL_START | {payload}")
        except Exception:
            pass

    def log_trial_end(
        self,
        trial_number: int,
        score: float,
        elapsed_sec: float,
        memory_gb: float,
        val_ic: float,
        rolling6m_ir: float,
        pct_positive_excess: float,
        vram_mb: float = 0.0,
        sys_avail_gb: float = 0.0,
    ) -> None:
        """记录 trial 结束到崩溃定位日志.

        ★ B-9 职责说明:
        - 本方法: 崩溃定位日志, 同步+异步双写, 包含 trial_number+score+elapsed+内存+显存
        - logger.py.log_trial_result(): 性能统计, 同步写入, 包含完整 params+metrics
        - 两路不重复, 互补: 一个用于"出问题时快速定位", 一个用于"统计汇总"
        - 不要删除其中任何一路, 否则会丢失对应场景的可观测性
        """
        if not self._initialized:
            return
        try:
            data = {
                "trial_number": trial_number,
                "score": round(score, 6),
                "elapsed_sec": round(elapsed_sec, 1),
                "memory_gb": round(memory_gb, 2),
                "val_ic": round(val_ic, 6),
                "rolling6m_ir": round(rolling6m_ir, 6),
                "pct_positive_excess": round(pct_positive_excess, 4),
            }
            # ★ 新增: VRAM 和系统可用内存
            if vram_mb > 0:
                data["vram_mb"] = int(vram_mb)
            if sys_avail_gb > 0:
                data["sys_avail_gb"] = round(sys_avail_gb, 2)
            payload = json.dumps(data, ensure_ascii=False)
            # ★ 同步写入（崩溃安全）
            self._sync_write("INFO", f"TRIAL_END | {payload}")
            # 异步通道也写一份
            self._logger.info(f"TRIAL_END | {payload}")
        except Exception:
            pass

    def log_error(
        self,
        error_type: str,
        error_msg: str,
        traceback_str: str,
    ) -> None:
        if not self._initialized:
            return
        try:
            tb = _truncate_traceback(traceback_str)
            payload = json.dumps({
                "error_type": error_type,
                "error_msg": error_msg[:500],
                "traceback": tb,
            }, ensure_ascii=False)
            # ★ 同步写入（崩溃安全）
            self._sync_write("ERROR", f"ERROR | {payload}")
            self._logger.error(f"ERROR | {payload}")
        except Exception:
            pass

    def log_critical(self, message: str) -> None:
        """★ 同步写入 CRITICAL 级日志（OOM前兆等，必须落盘）"""
        if not self._initialized:
            return
        try:
            self._sync_write("CRITICAL", message)
            self._logger.critical(message)
        except Exception:
            pass

    def log_window_progress(
        self,
        processed: int,
        total: int,
        memory_gb: float,
    ) -> None:
        if not self._initialized:
            return
        try:
            payload = json.dumps({
                "processed": processed,
                "total": total,
                "memory_gb": round(memory_gb, 2),
            }, ensure_ascii=False)
            self._logger.debug(f"WINDOW_PROGRESS | {payload}")
        except Exception:
            pass

    def log_system_stats(
        self,
        proc_mem_gb: float,
        cpu_pct: float,
        sys_mem_avail_gb: float,
    ) -> None:
        if not self._initialized:
            return
        try:
            oom_risk = sys_mem_avail_gb < 0.5
            payload = json.dumps({
                "proc_mem_gb": round(proc_mem_gb, 2),
                "cpu_pct": round(cpu_pct, 1),
                "sys_mem_avail_gb": round(sys_mem_avail_gb, 2),
                "oom_risk": oom_risk,
            }, ensure_ascii=False)
            self._logger.info(f"SYSTEM_STATS | {payload}")
        except Exception:
            pass

    def shutdown(self) -> None:
        if self._sync_file is not None:
            try:
                self._sync_file.flush()
                self._sync_file.close()
            except Exception:
                pass
            self._sync_file = None
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
        self._initialized = False


_GLOBAL_ROLLING_LOGGER: Optional[RollingLogger] = None


def get_rolling_logger() -> RollingLogger:
    global _GLOBAL_ROLLING_LOGGER
    if _GLOBAL_ROLLING_LOGGER is None:
        _GLOBAL_ROLLING_LOGGER = RollingLogger()
        _GLOBAL_ROLLING_LOGGER.setup()
    return _GLOBAL_ROLLING_LOGGER


def _try_log(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def _collect_system_stats() -> Tuple[float, float, float]:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        proc_mem_gb = proc.memory_info().rss / (1024 ** 3)
        cpu_pct = proc.cpu_percent(interval=0)
        sys_mem_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        return proc_mem_gb, cpu_pct, sys_mem_avail_gb
    except Exception:
        return 0.0, 0.0, 0.0


def _get_vram_mb() -> float:
    """读取GPU显存占用（MB），无GPU返回0"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(mem.used / 1024**2)
    except Exception:
        return 0.0


def _truncate_traceback(tb_str: str, max_len: int = 2000) -> str:
    if len(tb_str) <= max_len:
        return tb_str
    return tb_str[:max_len] + "...[truncated]"
