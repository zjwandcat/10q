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
M5 Phase2 局部精调
从phase1最优参数出发，独立study
"""
import os
import logging
import gc
import datetime
from typing import Optional, Dict, Callable, Any

import optuna
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)

from m5_optimizer.objective import ObjectiveFunction
from m5_optimizer.search_space import calc_window_count
from m5_optimizer.utils.trial_callback import make_trial_callback
from m5_optimizer.config_manager import load_config as _load_m5_config  # ★ v3.8

# ★ v3.8: 动态读 config (支持 UI 切换)
def _get_m5_config():
    return _load_m5_config()

logger = logging.getLogger("m5.phase2")


def _build_p2_distributions(
    param_name: str,
    custom_ranges: dict,
) -> "optuna.distributions.BaseDistribution | None":
    """
    根据P2的custom_ranges为单个参数构建Optuna分布对象。
    返回None表示该参数为固定参数（不在搜索空间）。
    """
    from m5_optimizer.search_space import ALL_PARAMS

    rng  = custom_ranges.get(param_name, {})
    pdef = ALL_PARAMS.get(param_name, {})

    # 固定参数（有adjusted_default但无low/high/choices）
    if "adjusted_default" in rng and "low" not in rng and "choices" not in rng:
        return None

    ptype = pdef.get("type", "")
    if ptype == "float_log":
        lo = float(rng.get("low", pdef["low"]))
        hi = float(rng.get("high", pdef["high"]))
        return FloatDistribution(lo, hi, log=True)
    elif ptype == "float":
        lo = float(rng.get("low", pdef["low"]))
        hi = float(rng.get("high", pdef["high"]))
        return FloatDistribution(lo, hi)
    elif ptype == "int":
        lo = int(rng.get("low", pdef["low"]))
        hi = int(rng.get("high", pdef["high"]))
        return IntDistribution(lo, hi)
    elif ptype == "categorical":
        choices = rng.get("choices", pdef.get("choices", []))
        return CategoricalDistribution(choices)
    return None


def _inject_reused_trials(
    study: "optuna.Study",
    reusable: list,
    custom_ranges: dict,
    active_params: "list | None",
) -> int:
    """
    将P1可复用Trial以COMPLETE状态注入P2 study。
    返回实际成功注入的数量。
    """

    injected = 0
    now = datetime.datetime.now()

    for rt in reusable:
        params     = rt["params"]
        value      = rt["value"]
        user_attrs = rt["user_attrs"]

        # 只包含P2参与搜索的参数（active_params中的）
        filtered_params = {}
        distributions   = {}

        for name, val in params.items():
            if active_params is not None and name not in active_params:
                continue   # 固定参数不进distributions
            dist = _build_p2_distributions(name, custom_ranges)
            if dist is None:
                continue   # 无法构建分布 → 跳过
            filtered_params[name] = val
            distributions[name]   = dist

        if not filtered_params:
            continue   # 没有可用参数 → 跳过

        try:
            # ★ B-6 修复: P1 注入 trial 的 datetime_start 设为 now - estimated_sec
            # 旧: datetime_start=datetime_complete=now → Optuna 计算 duration=0
            #     → len(study.trials) 立即 = n_trials, 进度条/ETA 永远 "100%/计算中"
            # 新: 用 5s 估算 (P1 阶段 trial 平均 ~2-10s), 让 progress 直观反映"已用时"
            # 同时 user_attrs 加 _reused_from_p1 标志, trial_callback 跳过累计 elapsed
            # (避免被误计入 P2 的 ETA 基线)
            from datetime import timedelta
            _estimated_sec = 5.0
            frozen = optuna.trial.FrozenTrial(
                number=-1,
                trial_id=-1,
                state=optuna.trial.TrialState.COMPLETE,
                value=value,
                values=None,
                datetime_start=now - timedelta(seconds=_estimated_sec),
                datetime_complete=now,
                params=filtered_params,
                distributions=distributions,
                user_attrs={
                    **user_attrs,
                    "_reused_from_p1": rt.get("p1_trial_id", -1),
                    # ★ 标记为复用, trial_callback 不会把这个 trial 的 _estimated_sec
                    # 误加入 _trial_times (避免污染 P2 ETA)
                    "_injected_p1": True,
                    "elapsed_sec": _estimated_sec,
                },
                system_attrs={},
                intermediate_values={},
            )
            study.add_trial(frozen)
            injected += 1
        except Exception as e:
            logger.warning(f"注入P1 Trial失败（跳过）: {e}")

    logger.info(f"P1 Trial复用：成功注入 {injected}/{len(reusable)} 个")
    return injected


def run_phase2(
    factor_df,
    project: dict = None,
    n_trials: int = 30,
    fast_mode: bool = False,
    window_count: int = None,
    objective_weights: Optional[Dict[str, float]] = None,
    active_params: Optional[list] = None,
    init_params: Optional[Dict[str, Any]] = None,
    param_ranges: Optional[Dict[str, Dict]] = None,
    scheme: str = "scheme_d",
    stop_now_event=None,
    stop_graceful_event=None,
    progress_callback: Optional[Callable[[int, int, float, Dict], None]] = None,
    storage: str = None,
    reuse_p1_trials: bool = True,
    max_p1_inject: int = 30,
    enable_normalization: bool = False,
    norm_config: Optional[Dict] = None,
) -> optuna.Study:
    # ★ 主动清理内存：避免P2启动时缓存未释放导致OOM
    try:
        from m5_optimizer.utils.win_memory import release_memory_to_os
        release_memory_to_os()
        gc.collect()
    except Exception:
        pass

    from m5_optimizer.project_manager import _ensure_wal_safe

    # storage优先用project里的db_path
    if project is not None:
        db_path = project["p2_db_path"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        actual_storage = f"sqlite:///{db_path}"
        study_name = project["p2_study_name"]
    else:
        actual_storage = storage or "sqlite:///output/m5/phase2.db"
        study_name = "phase2_local"

    # 创建或加载study
    # 修复：扩大异常捕获范围，覆盖 sqlite3.OperationalError（DB表已存在但study不存在）
    # 修复：移除 _ensure_wal 前置调用，避免裸 sqlite3 连接与 SQLAlchemy 连接冲突
    import sqlite3 as _sqlite3
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=actual_storage,
        )
        logger.info(f"加载已有study: {study_name}")
    except (KeyError, ValueError, _sqlite3.OperationalError):
        study = optuna.create_study(
            study_name=study_name,
            storage=actual_storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        )
        logger.info(f"创建新study: {study_name}")

    # 开WAL（study创建后DB才存在，通过SQLAlchemy engine设置避免裸连接冲突）
    if project is not None:
        _ensure_wal_safe(project["p2_db_path"])

    # 从P2配置读取init_from_p1_best，自动从P1 DB读best_params
    if project is not None and init_params is None:
        from m5_optimizer.project_manager import load_p2_config, get_p1_study
        p2_cfg = load_p2_config(project)
        if p2_cfg.get("init_from_p1_best"):
            p1_study = get_p1_study(project)
            if p1_study and p1_study.best_trial:
                init_params = dict(p1_study.best_trial.params)

    # ── P1 Trial 复用注入 ─────────────────────────────────────
    if reuse_p1_trials and project is not None:
        try:
            from m5_optimizer.project_manager import get_reusable_p1_trials

            # 仅在P2 study全新时复用（避免重复注入）
            existing_complete = sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            )
            if existing_complete == 0:
                reuse_result = get_reusable_p1_trials(
                    project=project,
                    p2_param_ranges=param_ranges or {},
                    p2_objective_weights=objective_weights or {},
                    p2_active_params=active_params,
                    p2_scheme=scheme,
                    p2_window_count=window_count,
                    max_inject=max_p1_inject,
                    enable_normalization=enable_normalization,
                    norm_config=norm_config,
                )
                if reuse_result["blocked"] and not reuse_result["blocked"].startswith("⚠️"):
                    logger.warning(f"P1复用被阻断: {reuse_result['blocked']}")
                elif reuse_result["injected"] > 0:
                    _inject_reused_trials(
                        study=study,
                        reusable=reuse_result["trials"],
                        custom_ranges=param_ranges or {},
                        active_params=active_params,
                    )
                    logger.info(
                        f"P1复用摘要: P1有效={reuse_result['total_p1']}, "
                        f"范围内={reuse_result['qualified']}, "
                        f"已注入={reuse_result['injected']}"
                    )
                else:
                    logger.info(
                        f"P1复用: P1有效={reuse_result['total_p1']}, "
                        f"范围内={reuse_result['qualified']}，无可注入Trial"
                    )
            else:
                logger.info(f"P2 study已有{existing_complete}个Trial，跳过P1复用")
        except Exception as e:
            logger.warning(f"P1复用失败（不影响P2正常运行）: {e}")
    # ─────────────────────────────────────────────────────────

    if init_params:
        study.enqueue_trial(init_params)

    objective = ObjectiveFunction(
        preloaded_factor_df=factor_df,
        preloaded_windows=None,
        window_count=window_count if window_count is not None else calc_window_count(),
        compute_val_metrics=True,
        objective_weights=objective_weights,
        active_params=active_params,
        scheme=scheme,
        fast_mode=fast_mode,
        stop_event=stop_now_event,
        custom_ranges=param_ranges,
        gpu_mode=bool(_get_m5_config().get("optimization", {}).get("gpu_mode", False)),  # ★ v3.8
        enable_normalization=enable_normalization,
        norm_config=norm_config,
        project_id=project.get("project_name") if project else None,
        neutralization_type=_get_m5_config().get("data", {}).get("neutralization", {}).get("active_scheme"),
    )

    # 时间统计 / trial_callback（共享工具）
    trial_callback = make_trial_callback(
        n_trials=n_trials,
        stop_graceful_event=stop_graceful_event,
        stop_now_event=stop_now_event,
        progress_callback=progress_callback,
        phase="p2",  # ★ v5.0: P2 独立 RSS 历史
    )

    # ★ Trial结束后清理内存，避免累积OOM
    def _post_trial_cleanup(study, trial):
        import gc
        gc.collect()
        try:
            from m5_optimizer.utils.win_memory import release_memory_to_os
            release_memory_to_os()
        except Exception:
            pass

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[trial_callback, _post_trial_cleanup],
        catch=(Exception,),
    )

    # 运行完更新配置文件的trials_history
    if project is not None:
        from m5_optimizer.project_manager import update_trials_history
        update_trials_history(project, n_trials, phase="p2")

    return study
