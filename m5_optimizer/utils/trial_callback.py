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
M5 trial_callback 共享工具
phase1 / phase2 的 progress_callback 逻辑完全一致，提取为公共函数。
"""
import time
from collections import deque
from typing import Optional, Callable, List

import optuna


def make_trial_callback(
    n_trials: int,
    stop_graceful_event,
    stop_now_event=None,
    progress_callback: Optional[Callable] = None,
) -> Callable:
    """
    生成 trial_callback：每完成一个Trial触发一次progress_callback，
    附带 best_value、累计统计、单Trial耗时、ETA。

    参数：
        n_trials: 用户设定总Trial数（用于计算ETA）
        stop_graceful_event: 优雅停止事件
        stop_now_event: 立即停止事件（触发后立即 study.stop()，与优雅停止等价地阻止下一 trial）
        progress_callback: 形如
            fn(trial_num, total, best_val, attrs, time_info)
    """
    # ★ B-5 修复: deque(maxlen=20) 替代无界 List, 防 5000+ trial 内存浪费
    _trial_times: "deque" = deque(maxlen=20)   # 单Trial objective 实际耗时（user_attrs["elapsed_sec"]）
    # ★ B-13 修复: 新增 _wall_times 跟踪 callback 间 wall-clock
    # 原因: 旧版 _trial_times 用 objective 内部耗时（串行口径）算 ETA,
    #       当 2 个 Trial 并行执行时（同一 study DB 被多进程/多 worker 写入）,
    #       真实吞吐率 = callback 间 wall-clock（远小于 objective 耗时）,
    #       ETA 高估约 2~3 倍。
    # 新版: ETA 用 _wall_times 计算（吞吐口径），_trial_times 保留为诊断信息。
    _wall_times: "deque" = deque(maxlen=20)
    # ★ B-3 修复: 初始化时用 None 标记"未开始", 第一次 trial_callback 再设 wall_time
    # 旧: time.time() 在 make_trial_callback() 调用时立即记录, 含数据加载等冷启动
    # 新: 用 -1.0 占位, callback 第一次进来时再 set
    _last_trial_end: List[float] = [-1.0]

    def trial_callback(
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ):
        # ★ 任一停止事件触发都立即调用 study.stop()，避免继续发起新 Trial
        if (stop_graceful_event and stop_graceful_event.is_set()) or \
           (stop_now_event and stop_now_event.is_set()):
            study.stop()
            return
        if progress_callback is None:
            return

        from m5_optimizer.project_manager import get_study_stats

        now = time.time()
        # ★ B-3 修复: 第一次回调初始化 _last_trial_end (避免冷启动污染首个 elapsed)
        if _last_trial_end[0] < 0:
            _last_trial_end[0] = now
        elapsed = now - _last_trial_end[0]
        _last_trial_end[0] = now

        # ★ B-4 修复: 接受 PRUNED / FAIL 状态, 不只 COMPLETE
        # 旧: 仅 state==COMPLETE 记录 elapsed, 启用 Pruner 时 avg_sec 严重低估
        # 新: 任何"已结束"状态都记录, 但是 set_user_attr 持久化的 elapsed_sec 优先
        # ★ B-6 修复: 跳过 P1 注入的 trial (避免被误计入 P2 的 _trial_times)
        _is_p1_injected = False
        try:
            _is_p1_injected = bool(
                (trial.user_attrs or {}).get("_injected_p1", False))
        except Exception:
            pass

        _trial_elapsed = None
        if trial.state.is_finished() and not _is_p1_injected:
            # 优先用 user_attrs 持久化的 (B-2 修复提供)
            try:
                _ua = trial.user_attrs or {}
                _persisted = _ua.get("elapsed_sec")
                if _persisted is not None:
                    _trial_elapsed = float(_persisted)
            except Exception:
                pass
            if _trial_elapsed is None:
                # fallback: 用 trial 自身 datetime 计算
                try:
                    if trial.datetime_complete and trial.datetime_start:
                        _trial_elapsed = (
                            trial.datetime_complete - trial.datetime_start
                        ).total_seconds()
                except Exception:
                    pass
            if _trial_elapsed is None or _trial_elapsed <= 0:
                _trial_elapsed = elapsed  # 最后 fallback

            _trial_times.append(_trial_elapsed)
            # ★ B-13 修复: 同步记录 wall-clock (callback 间真实耗时)
            # 第一次回调的 elapsed 含数据加载开销, 跳过避免污染
            if len(_wall_times) >= 0 and elapsed > 0 and elapsed < 7200:
                _wall_times.append(elapsed)

        # ★ B-13 修复: 分别计算 objective 耗时均值 和 wall-clock 吞吐均值
        # obj_avg_sec: 单 Trial 在 objective 内的计算耗时 (诊断用, 串行口径)
        # wall_avg_sec: callback 间真实耗时 (ETA 用, 吞吐口径, 自动适配并行)
        if len(_trial_times) >= 2:
            _recent_obj = list(_trial_times)[-5:]
            obj_avg_sec = sum(_recent_obj) / len(_recent_obj)
        elif len(_trial_times) == 1:
            obj_avg_sec = _trial_times[0]
        else:
            obj_avg_sec = None

        if len(_wall_times) >= 2:
            _recent_wall = list(_wall_times)[-5:]
            wall_avg_sec = sum(_recent_wall) / len(_recent_wall)
        elif len(_wall_times) == 1:
            wall_avg_sec = _wall_times[0]
        else:
            wall_avg_sec = None

        # ★ B-13 修复: ETA 用 wall-clock (吞吐口径), 兼容并行场景
        # 旧: avg_sec (objective 耗时) × remaining → 并行时高估 2~3 倍
        # 新: wall_avg_sec (实际 wall-clock/完成 1 个 trial) × remaining → 准确
        _eta_avg_sec = wall_avg_sec if wall_avg_sec else obj_avg_sec

        stats = get_study_stats(study)
        attrs = trial.user_attrs if trial.state.is_finished() else {}

        remaining_trials = max(0, n_trials - len(study.trials))
        # ★ B-7 修复: 命名更清晰, 同时返回数值 avg_sec (供二次计算)
        if _eta_avg_sec and remaining_trials > 0:
            remaining_min = _eta_avg_sec * remaining_trials / 60
            eta_str = f"约{remaining_min:.0f}分钟"
            # ★ B-13: 区分"单Trial计算耗时"(objective) vs "实际吞吐率"(wall)
            # 用 wall 时, 标注"实际"提示用户已计入并行
            if wall_avg_sec and obj_avg_sec and wall_avg_sec < obj_avg_sec * 0.85:
                per_trial_str = (
                    f"{wall_avg_sec/60:.1f}分钟/Trial "
                    f"(实际,含并行)"
                )
            else:
                per_trial_str = f"{_eta_avg_sec/60:.1f}分钟/Trial"
        else:
            eta_str = "计算中..."
            per_trial_str = "计算中..."

        time_info = {
            "per_trial_str": per_trial_str,        # ★ B-7: 重命名 (旧: per_trial_min, 实际是字符串)
            "per_trial_sec": round(_eta_avg_sec, 1) if _eta_avg_sec else None,  # ETA 用的吞吐口径
            # ★ B-13 修复: 暴露诊断字段
            "obj_avg_sec": round(obj_avg_sec, 1) if obj_avg_sec else None,    # objective 内部耗时(串行口径)
            "wall_avg_sec": round(wall_avg_sec, 1) if wall_avg_sec else None,  # wall-clock 吞吐(并行口径)
            "eta": eta_str,
            "avg_sec": _eta_avg_sec,
        }

        progress_callback(
            stats["complete"],
            stats["total"],
            # ★ 修复：使用 stats["best_value"]（已排除 value<=-999 的异常 Trial），
            # 不能再用 study.best_value，否则当存在 -999 兜底值 Trial 时，
            # Optuna 会把 -999 当作"最优"显示在日志中，误导用户。
            stats.get("best_value") if stats["complete"] > 0 else None,
            attrs,
            time_info,
        )

    return trial_callback
