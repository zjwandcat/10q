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
