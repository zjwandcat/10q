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
M5配置管理器
配置文件读写、最优参数写回
"""
import os
import json
import shutil
import logging
from typing import Dict, Any

import yaml

from m5_optimizer.search_space import assemble_params

logger = logging.getLogger("m5.config_manager")


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str = "config/config.yaml"):
    if os.path.exists(path):
        shutil.copy(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def write_best_params_to_config(
    best_params: Dict[str, Any],
    path: str = "config/config.yaml",
):
    config = load_config(path)

    lgbm_params, xgbm_params, feature_params, lgbm_weight, _ = (
        assemble_params(best_params)
    )

    if "m2" not in config:
        config["m2"] = {}

    config["m2"]["lgbm"] = lgbm_params
    config["m2"]["xgb"] = xgbm_params

    if "ensemble" not in config["m2"]:
        config["m2"]["ensemble"] = {}
    config["m2"]["ensemble"]["lgbm_weight"] = lgbm_weight

    if "feature_store" not in config["m2"]:
        config["m2"]["feature_store"] = {}
    config["m2"]["feature_store"].update(feature_params)

    save_config(config, path)
    logger.info(f"最优参数已写回 {path}")


def save_ranges_json(
    ranges: Dict[str, Any],
    path: str = "output/m5/p2_ranges.json",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ranges, f, ensure_ascii=False, indent=2)


def load_ranges_json(
    path: str = "output/m5/p2_ranges.json",
) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
