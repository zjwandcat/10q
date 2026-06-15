"""
M5 Phase1 全局探索
TPE全局探索，断点续跑，支持双停止机制
"""
import os
import logging
from typing import Optional, Dict, Callable

import optuna

from m5_optimizer.objective import ObjectiveFunction
from m5_optimizer.search_space import calc_window_count
from m5_optimizer.utils.trial_callback import make_trial_callback
from m5_optimizer.config_manager import load_config as _load_m5_config  # ★ v3.5

# ★ v3.8: 改用函数, 每次创建 ObjectiveFunction 时重新读 config (支持 UI 动态切换)
def _get_m5_config():
    return _load_m5_config()

logger = logging.getLogger("m5.phase1")


def _trial_constraints(trial):
    """
    返回约束违反量列表（<=0表示满足，>0表示违反）
    Optuna会倾向于避开违反约束的区域
    """
    lgbm_lr = trial.params.get("lgbm_learning_rate", 0.05)
    lgbm_n  = trial.params.get("lgbm_n_estimators", 100)
    xgb_lr  = trial.params.get("xgb_learning_rate", 0.05)
    xgb_n   = trial.params.get("xgb_n_estimators", 100)

    constraints = []
    # 约束1：LR × n_estimators 乘积上限（防止过拟合且耗时）
    constraints.append(lgbm_lr * lgbm_n - 15.0)  # <=0 为满足
    constraints.append(xgb_lr  * xgb_n  - 15.0)

    return constraints


def run_phase1(
    factor_df,
    project: dict = None,
    n_trials: int = 50,
    fast_mode: bool = False,      # ★ 固定False
    window_count: int = None,      # None时由calc_window_count()动态计算
    objective_weights: Optional[Dict[str, float]] = None,
    active_params: Optional[list] = None,
    scheme: str = "scheme_d",
    param_ranges: dict = None,   # ★ 新增
    stop_now_event=None,
    stop_graceful_event=None,
    progress_callback: Optional[Callable[[int, int, float, Dict], None]] = None,
    storage: str = None,
    warm_start: Optional[Dict] = None,
    enable_normalization: bool = False,
    norm_config: Optional[Dict] = None,
) -> optuna.Study:
    from m5_optimizer.project_manager import _ensure_wal_safe

    # storage优先用project里的db_path
    if project is not None:
        db_path = project["p1_db_path"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        actual_storage = f"sqlite:///{db_path}"
        study_name = project["p1_study_name"]
    else:
        actual_storage = storage or "sqlite:///output/m5/phase1.db"
        study_name = "phase1_global"

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
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=15,
                n_ei_candidates=24,
                constraints_func=_trial_constraints,
            ),
        )
        logger.info(f"创建新study: {study_name}")

    # 开WAL（study创建后DB才存在，通过SQLAlchemy engine设置避免裸连接冲突）
    if project is not None:
        _ensure_wal_safe(project["p1_db_path"])

    # 热启动先验注入：仅在study全新（0个已完成Trial）时注入一次
    if warm_start and len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE
                           or t.state == optuna.trial.TrialState.RUNNING]) == 0:
        try:
            study.enqueue_trial(warm_start)
            logger.info(f"热启动先验已注入（{len(warm_start)}个参数）")
        except Exception as e:
            logger.warning(f"热启动先验注入失败（不影响运行）: {e}")

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
        custom_ranges=param_ranges or {},  # ★ 传入
        gpu_mode=bool(_get_m5_config().get("optimization", {}).get("gpu_mode", False)),  # ★ v3.8
        enable_normalization=enable_normalization,
        norm_config=norm_config,
    )

    # 时间统计 / trial_callback（共享工具）
    trial_callback = make_trial_callback(
        n_trials=n_trials,
        stop_graceful_event=stop_graceful_event,
        stop_now_event=stop_now_event,
        progress_callback=progress_callback,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[trial_callback],
        catch=(Exception,),
    )

    # 运行完更新配置文件的trials_history
    if project is not None:
        from m5_optimizer.project_manager import update_trials_history
        update_trials_history(project, n_trials, phase="p1")

        try:
            from m5_optimizer.project_manager import load_p1_config
            p1_config = load_p1_config(project)
            p1_config["config"]["active_params"] = active_params
            p1_config["config"]["param_ranges"] = param_ranges or {}
            p1_config["config"]["enable_normalization"] = enable_normalization
            import json
            with open(project["p1_config_path"], "w", encoding="utf-8") as f:
                json.dump(p1_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存P1配置失败（不影响运行）: {e}")

    return study
