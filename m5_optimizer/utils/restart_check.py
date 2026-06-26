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
M5 RSS 增长率检测器（方案 D）

用途：
    在 trial_callback 中调用，跟踪 RSS 随 Trial 数的增长趋势。
    若 RSS 增长率超过阈值（典型内存泄漏特征），主动触发 stop_now_event
    并写 CRITICAL 日志提示用户重启程序。

判定逻辑：
    - 记录最近 N 个 Trial 完成时的 RSS
    - 计算 (latest - baseline) / baseline
    - 若 > LEAK_RATIO_THRESHOLD（默认 20%），判定为泄漏

调用方式（在 trial_callback 末尾）：
    from m5_optimizer.utils.restart_check import check_rss_leak
    check_rss_leak(trial.number, stop_now_event)
"""
import os
import logging
from collections import deque
from typing import Optional

import psutil

logger = logging.getLogger("m5.restart_check")

# ── 配置 ──────────────────────────────────────────
# ★ v5.2 调整: 放宽阈值，避免 ML 工作负载的自然 RSS 波动被误判为泄漏
#   原因: 不同 Trial 的 train_months/colsample_bytree/n_estimators 差异
#         会导致 RSS 在 0.6-1.3GB 间自然波动，旧阈值 (20%/0.5GB) 误报率高
#   新策略: 同时满足以下 3 条才判定泄漏
#     1) 相对增长 > 60% (原 20%)
#     2) 绝对增长 > 2.0GB (原 0.5GB)
#     3) 系统可用内存 < 1.5GB (内存真的紧张才停)
LEAK_RATIO_THRESHOLD = 0.60   # RSS 增长 60% 才判定为泄漏（原 0.20）
LEAK_MIN_TRIALS = 10          # 至少跑 10 个 Trial 后才开始检测
LEAK_WINDOW = 20              # 用最近 20 个 Trial 的 RSS 做基线
LEAK_MIN_ABS_GB = 2.0         # 增长绝对值 < 2.0GB 不算泄漏（原 0.5）
LEAK_SYS_AVAIL_GB = 1.5       # 系统可用内存高于此值时不触发停止（留余地给 OS）

# ★ v5.2: 软清理重试次数。检测到 RSS 增长时先尝试强制 GC+归还，
#   只有清理后仍超阈值且系统内存紧张才真正触发停止。
LEAK_SOFT_CLEAN_MAX_RETRIES = 2

# ── 全局状态（每个进程一份）
# ★ v5.0: P1/P2 分离的 RSS 历史，避免同时运行时基线互相污染
_rss_history_p1: "deque" = deque(maxlen=LEAK_WINDOW)
_rss_history_p2: "deque" = deque(maxlen=LEAK_WINDOW)
_last_alert_trial: int = -1   # 上次告警的 Trial 号，避免重复刷屏
# ★ v5.2: 软清理重试计数器（按 phase 分离）
_soft_clean_retries_p1: int = 0
_soft_clean_retries_p2: int = 0


def _get_soft_clean_retries(phase: str) -> int:
    return _soft_clean_retries_p1 if phase == "p1" else _soft_clean_retries_p2


def _inc_soft_clean_retries(phase: str) -> int:
    """增加软清理计数器，返回新值"""
    global _soft_clean_retries_p1, _soft_clean_retries_p2
    if phase == "p1":
        _soft_clean_retries_p1 += 1
        return _soft_clean_retries_p1
    else:
        _soft_clean_retries_p2 += 1
        return _soft_clean_retries_p2


def _reset_soft_clean_retries(phase: str) -> None:
    """RSS 回落或判定为正常时重置计数器"""
    global _soft_clean_retries_p1, _soft_clean_retries_p2
    if phase == "p1":
        _soft_clean_retries_p1 = 0
    else:
        _soft_clean_retries_p2 = 0


def check_rss_leak(
    trial_number: int,
    stop_now_event=None,
    graceful_stop_event=None,
    phase: str = "p1",  # ★ v5.0: 区分 P1/P2 的 RSS 历史
) -> bool:
    """
    在每个 Trial 完成后调用，检测 RSS 是否持续增长（内存泄漏特征）。

    ★ v5.2 新策略: 三重判定 + 软清理
      1) RSS 相对增长 > 60% 且绝对增长 > 2.0GB
      2) 系统可用内存 < 1.5GB（内存真的紧张）
      3) 软清理（GC + SetProcessWorkingSetSize + _heapmin）后仍超阈值
      只有 3 条同时满足才触发停止，避免 ML 工作负载自然波动被误判。

    参数：
        trial_number: 当前 Trial 号
        stop_now_event: 触发后立即停止 Optuna
        graceful_stop_event: 可选，触发优雅停止
        phase: "p1" 或 "p2"，区分不同 Phase 的 RSS 历史

    返回：
        True 表示检测到泄漏并已触发停止
        False 表示正常
    """
    global _last_alert_trial

    try:
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return False

    # ★ v5.0: 按 phase 选择对应的 RSS 历史
    rss_history = _rss_history_p1 if phase == "p1" else _rss_history_p2
    rss_history.append(rss_gb)

    # Trial 数不够，不做判定
    if len(rss_history) < LEAK_MIN_TRIALS:
        return False

    # 取最早的 RSS 作为基线（deque 最早元素）
    baseline = rss_history[0]
    latest = rss_history[-1]

    # 基线太小没意义（< 0.5GB 时不算）
    if baseline < 0.5:
        return False

    growth = latest - baseline
    growth_ratio = growth / baseline

    # ★ v5.2: RSS 回落到正常范围时重置软清理计数器
    #   判定"回落"的标准：相对增长 < 30% 或绝对增长 < 1.0GB
    if growth_ratio < 0.30 or growth < 1.0:
        _reset_soft_clean_retries(phase)

    # 条件 1: 相对增长 + 绝对增长 同时超阈值
    if not (growth_ratio > LEAK_RATIO_THRESHOLD
            and growth > LEAK_MIN_ABS_GB):
        return False

    # ★ v5.2 条件 2: 检查系统可用内存
    #   系统内存充足时（> 1.5GB 可用），即使 RSS 增长也不触发停止
    #   原因: ML 工作负载的 RSS 波动是正常的，只要系统不缺内存就让它继续跑
    try:
        sys_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        sys_avail_gb = 999.0  # 读取失败时不阻塞

    if sys_avail_gb > LEAK_SYS_AVAIL_GB:
        # 系统内存充足，只记录 WARNING 不停止
        if _last_alert_trial != trial_number:
            logger.warning(
                f"RSS_GROWTH | Trial#{trial_number} "
                f"baseline={baseline:.2f}GB → latest={latest:.2f}GB "
                f"(+{growth:.2f}GB, +{growth_ratio*100:.1f}%) "
                f"超过阈值，但系统可用内存={sys_avail_gb:.2f}GB "
                f"> {LEAK_SYS_AVAIL_GB}GB，继续运行"
            )
            try:
                from m5_optimizer.utils.rolling_logger import get_rolling_logger
                get_rolling_logger().log_critical(
                    f"RSS_GROWTH | Trial#{trial_number} "
                    f"baseline={baseline:.2f}GB → latest={latest:.2f}GB "
                    f"(+{growth:.2f}GB, +{growth_ratio*100:.1f}%) "
                    f"系统可用={sys_avail_gb:.2f}GB 充足，继续运行"
                )
            except Exception:
                pass
            _last_alert_trial = trial_number
        # 仍然尝试主动释放内存（无害操作）
        try:
            from m5_optimizer.utils.win_memory import release_memory_to_os
            release_memory_to_os()
        except Exception:
            pass
        return False

    # ★ v5.2 条件 3: 软清理阶段
    #   系统内存紧张 + RSS 增长超阈值时，先尝试强制 GC + 归还内存
    #   清理后 RSS 回落则继续运行，否则累计重试次数，超过上限才停止
    retries = _get_soft_clean_retries(phase)
    if retries < LEAK_SOFT_CLEAN_MAX_RETRIES:
        _inc_soft_clean_retries(phase)
        logger.warning(
            f"RSS_SOFT_CLEAN | Trial#{trial_number} "
            f"baseline={baseline:.2f}GB → latest={latest:.2f}GB "
            f"(+{growth:.2f}GB, +{growth_ratio*100:.1f}%) "
            f"系统可用={sys_avail_gb:.2f}GB 紧张，"
            f"尝试软清理 ({retries+1}/{LEAK_SOFT_CLEAN_MAX_RETRIES})"
        )
        try:
            from m5_optimizer.utils.rolling_logger import get_rolling_logger
            get_rolling_logger().log_critical(
                f"RSS_SOFT_CLEAN | Trial#{trial_number} "
                f"baseline={baseline:.2f}GB → latest={latest:.2f}GB "
                f"系统可用={sys_avail_gb:.2f}GB，"
                f"软清理 {retries+1}/{LEAK_SOFT_CLEAN_MAX_RETRIES}"
            )
        except Exception:
            pass

        # 强制 GC + 归还内存给 OS
        try:
            from m5_optimizer.utils.win_memory import release_memory_to_os
            release_memory_to_os()
        except Exception:
            pass

        # 清理后重新测量 RSS，更新 history 末尾
        try:
            rss_after = psutil.Process(
                os.getpid()).memory_info().rss / 1e9
            rss_history[-1] = rss_after
            # 如果清理后 RSS 回落到安全范围，重置计数器并继续
            growth_after = rss_after - baseline
            growth_ratio_after = growth_after / baseline if baseline > 0 else 0
            if (growth_ratio_after < LEAK_RATIO_THRESHOLD
                    or growth_after < LEAK_MIN_ABS_GB):
                logger.info(
                    f"RSS_RECOVERED | Trial#{trial_number} "
                    f"软清理后 RSS {latest:.2f}GB → {rss_after:.2f}GB，"
                    f"继续运行"
                )
                _reset_soft_clean_retries(phase)
                return False
        except Exception:
            pass
        # 清理后仍超阈值但还有重试机会，继续运行下一 Trial
        return False

    # ★ 三重条件全满足：触发停止
    # 避免同一 Trial 反复告警
    if _last_alert_trial == trial_number:
        return True
    _last_alert_trial = trial_number

    msg = (
        f"RSS_LEAK_DETECTED | Trial#{trial_number} "
        f"baseline={baseline:.2f}GB → latest={latest:.2f}GB "
        f"(+{growth:.2f}GB, +{growth_ratio*100:.1f}%) "
        f"系统可用={sys_avail_gb:.2f}GB < {LEAK_SYS_AVAIL_GB}GB，"
        f"软清理 {retries}/{LEAK_SOFT_CLEAN_MAX_RETRIES} 次无效，"
        f"判定为内存泄漏，触发停止"
    )
    logger.critical(msg)

    # 同步写崩溃安全日志
    try:
        from m5_optimizer.utils.rolling_logger import get_rolling_logger
        get_rolling_logger().log_critical(msg)
    except Exception:
        pass

    # 主动释放一次内存（万一能救回来）
    try:
        from m5_optimizer.utils.win_memory import release_memory_to_os
        release_memory_to_os()
    except Exception:
        pass

    # 触发停止事件
    if stop_now_event is not None:
        stop_now_event.set()
    if graceful_stop_event is not None:
        graceful_stop_event.set()

    return True


def reset_rss_baseline(phase: str = "p1") -> None:
    """
    重置 RSS 基线（重启 Optuna study 或用户手动清理后调用）。
    ★ v5.0: 按 phase 分别重置
    ★ v5.2: 同时重置软清理计数器
    """
    global _last_alert_trial
    if phase == "p1":
        _rss_history_p1.clear()
    else:
        _rss_history_p2.clear()
    _reset_soft_clean_retries(phase)
    _last_alert_trial = -1


def get_rss_trend(phase: str = "p1") -> dict:
    """返回当前 RSS 趋势诊断信息（供 UI/日志展示）"""
    rss_history = _rss_history_p1 if phase == "p1" else _rss_history_p2
    if len(rss_history) < 2:
        return {"samples": len(rss_history), "status": "insufficient_data"}

    baseline = rss_history[0]
    latest = rss_history[-1]
    growth = latest - baseline
    growth_ratio = growth / baseline if baseline > 0 else 0.0

    # ★ v5.2: 状态判定也加入系统可用内存条件
    try:
        sys_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        sys_avail_gb = 999.0

    is_leak = (
        growth_ratio > LEAK_RATIO_THRESHOLD
        and growth > LEAK_MIN_ABS_GB
        and len(rss_history) >= LEAK_MIN_TRIALS
        and sys_avail_gb < LEAK_SYS_AVAIL_GB
    )
    status = "leak_detected" if is_leak else (
        "rss_growth_warning" if (
            growth_ratio > LEAK_RATIO_THRESHOLD
            and growth > LEAK_MIN_ABS_GB
        ) else "normal"
    )

    return {
        "samples": len(rss_history),
        "baseline_gb": round(baseline, 2),
        "latest_gb": round(latest, 2),
        "growth_gb": round(growth, 2),
        "growth_ratio": round(growth_ratio, 3),
        "leak_threshold": LEAK_RATIO_THRESHOLD,
        "sys_avail_gb": round(sys_avail_gb, 2),
        "soft_clean_retries": _get_soft_clean_retries(phase),
        "status": status,
    }
