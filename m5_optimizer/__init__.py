"""
M5贝叶斯超参数优化器
"""
from m5_optimizer.search_space import (
    ALL_PARAMS, OBJECTIVE_VARS, DEFAULT_PARAMS, PARAM_PASS_THROUGH
)
from m5_optimizer.objective import ObjectiveFunction, IC_GAP_PENALTY_MULTIPLIER
from m5_optimizer.phase1_global import run_phase1
from m5_optimizer.phase2_local import run_phase2
from m5_optimizer.range_analyzer import analyze_ranges, count_matched
from m5_optimizer.result_analyzer import get_best_study, get_best_params, run_full_backtest
from m5_optimizer.config_manager import (
    load_config, save_config, write_best_params_to_config,
    save_ranges_json, load_ranges_json
)

__all__ = [
    "ALL_PARAMS",
    "OBJECTIVE_VARS",
    "DEFAULT_PARAMS",
    "PARAM_PASS_THROUGH",
    "ObjectiveFunction",
    "IC_GAP_PENALTY_MULTIPLIER",
    "run_phase1",
    "run_phase2",
    "analyze_ranges",
    "count_matched",
    "get_best_study",
    "get_best_params",
    "run_full_backtest",
    "load_config",
    "save_config",
    "write_best_params_to_config",
    "save_ranges_json",
    "load_ranges_json",
]
