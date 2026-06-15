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
M5日志工具
- 三个独立日志文件：主运行日志、错误日志、性能日志
- 自动按日期滚动，防止单文件过大
- 只记录关键节点，不影响性能
"""
import logging
import os
from logging.handlers import TimedRotatingFileHandler


def _safe_emit(fn, *args, **kwargs) -> None:
    """封装日志handler的emit，避免Windows平台下log文件被锁定时crash"""
    try:
        fn(*args, **kwargs)
    except (PermissionError, OSError):
        # 日志文件被占用时静默失败，不影响业务
        # （典型场景：上一次启动未正常退出，文件handle未释放）
        pass


class _SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    ★ Windows 安全的 TimedRotatingFileHandler：
    rollover 时若目标文件被占用（PermissionError [WinError 32]），
    不抛异常，直接放弃本次滚动，下次启动再处理。
    """

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except (PermissionError, OSError):
            # 文件被锁定（典型：上一次进程崩溃后文件handle残留）
            # 静默忽略，下次启动再尝试
            pass


def setup_logger(name: str, log_dir: str = "logs/m5") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # 已初始化，直接返回

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ★ 文件1：主运行日志（INFO级，按天滚动，保留7天）
    # 记录：Trial完成、参数、得分、时间预估
    main_handler = _SafeTimedRotatingFileHandler(
        os.path.join(log_dir, "m5_run.log"),
        when="midnight", interval=1, backupCount=7,
        encoding="utf-8"
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(formatter)

    # ★ 文件2：错误日志（WARNING级以上，永久保留）
    # 记录：OOM、Trial异常、崩溃、堆栈
    error_handler = logging.FileHandler(
        os.path.join(log_dir, "m5_error.log"),
        encoding="utf-8"
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)

    # ★ 文件3：性能日志（INFO级，按天滚动，保留30天）
    # 记录：每个Trial的参数+得分+耗时，供后续分析
    perf_handler = _SafeTimedRotatingFileHandler(
        os.path.join(log_dir, "m5_perf.log"),
        when="midnight", interval=1, backupCount=30,
        encoding="utf-8"
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # 控制台（WARNING以上，不刷屏）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(main_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    # 性能logger单独实例
    if name == "m5":
        perf_logger = logging.getLogger("m5.perf")
        perf_logger.setLevel(logging.INFO)
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False  # 不往上传，只写自己的文件

    return logger


def get_logger(name: str = "m5") -> logging.Logger:
    return setup_logger(name)


def log_trial_result(
    trial_number: int,
    params: dict,
    metrics: dict,
    score: float,
    elapsed_sec: float,
):
    """★ 记录单个Trial结果到性能日志，供后续分析"""
    perf_logger = logging.getLogger("m5.perf")

    # 只记录关键指标，不记录全部29个参数（太长）
    key_params = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in params.items()
        if k in {
            "lgbm_learning_rate", "lgbm_n_estimators", "lgbm_max_depth",
            "xgb_learning_rate", "xgb_n_estimators", "xgb_max_depth",
            "lgbm_weight", "min_keep_factors"
        }
    }
    key_metrics = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in metrics.items()
        if k in {
            "val_ic", "val_icir", "ic_gap_penalty",
            "penalized_rate", "val_rolling6m_ir"
        }
    }

    perf_logger.info(
        f"Trial#{trial_number} | score={score:.4f} | "
        f"time={elapsed_sec:.1f}s | "
        f"params={key_params} | "
        f"metrics={key_metrics}"
    )
