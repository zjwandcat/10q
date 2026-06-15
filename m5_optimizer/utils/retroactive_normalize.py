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
追溯式归一化工具
对已有 Optuna Study 的历史 Trial 计算归一化评分，
写入 user_attrs["retro_norm_score"]，不修改原始 value。

由于 Optuna 不允许修改已完成 Trial 的 user_attrs，
本工具直接操作 SQLite 数据库写入。

用法:
    python -m m5_optimizer.utils.retroactive_normalize --db output/15B/p1/15B_p1_study.db --study phase1_global
    python -m m5_optimizer.utils.retroactive_normalize --db output/15B/p2/15B_p2_study.db --study phase2_local --weights val_icir:0.2 ic_gap_penalty:0.25
"""
import argparse
import json
import logging
import sqlite3
from typing import Dict, Optional

import optuna

from m5_optimizer.objective import _apply_normalization, _safe_float
from m5_optimizer.search_space import NORM_CONFIG, OBJECTIVE_VARS

logger = logging.getLogger("m5.retroactive_normalize")

# 方向映射
_DIRECTION_MAP = {v["name"]: v["direction"] for v in OBJECTIVE_VARS}

# ic_gap_penalty 惩罚乘子（与 objective.py 保持一致）
_IC_GAP_PENALTY_MULTIPLIER = 1.5


def _compute_norm_score(
    user_attrs: dict,
    objective_weights: Dict[str, float],
    norm_config: dict,
) -> float:
    """从 user_attrs 计算归一化评分。"""
    norm_score = 0.0
    for metric, weight in objective_weights.items():
        raw_value = _safe_float(user_attrs.get(metric, 0))
        norm_value = _apply_normalization(metric, raw_value, norm_config)
        multiplier = (
            _IC_GAP_PENALTY_MULTIPLIER
            if metric == "ic_gap_penalty"
            else 1.0
        )
        direction = _DIRECTION_MAP.get(metric, "max")
        if direction == "max":
            norm_score += weight * multiplier * norm_value
        else:
            norm_score -= weight * multiplier * norm_value
    return norm_score


def retroactively_normalize_study(
    db_path: str,
    study_name: str,
    objective_weights: Optional[Dict[str, float]] = None,
    norm_config: Optional[Dict] = None,
    v2: bool = False,
) -> dict:
    """对已有 Study 的历史 Trial 计算归一化评分并写入 user_attrs。

    通过直接操作 SQLite 数据库写入，绕过 Optuna 对已完成 Trial 的写保护。
    不修改原始 value 字段。

    参数:
        db_path: SQLite 数据库文件路径
        study_name: Study 名称
        objective_weights: 目标权重字典
        norm_config: 归一化配置，None 则使用默认 NORM_CONFIG
        v2: True 时写入 retro_norm_score_v2 字段（包含 P1 2.0 新增 3 个指标）
            False 时写入 retro_norm_score 字段（向后兼容）

    返回:
        统计字典 {"total": int, "processed": int, "skipped": int, "errors": int}
    """
    _norm_cfg = norm_config or NORM_CONFIG

    if objective_weights is None:
        logger.error("必须提供 objective_weights")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 1}

    # ★ v2 模式下使用全量 19 项 P1 指标 + 默认 0.0 兜底新指标
    # 目标写入 key：v2=True → retro_norm_score_v2，否则 retro_norm_score
    target_key = "retro_norm_score_v2" if v2 else "retro_norm_score"

    # 先通过 Optuna 读取 Trial 数据
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        logger.error(f"加载 study 失败: {e}")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 1}

    total = len(study.trials)
    processed = 0
    skipped = 0
    errors = 0

    # 预计算每个 Trial 的归一化评分
    trial_scores = {}
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            skipped += 1
            continue
        if trial.value is not None and trial.value <= -999.0:
            skipped += 1
            continue

        try:
            user_attrs = trial.user_attrs if hasattr(trial, "user_attrs") else {}
            norm_score = _compute_norm_score(user_attrs, objective_weights, _norm_cfg)
            trial_scores[trial.number] = norm_score
        except Exception as e:
            logger.warning(f"Trial {trial.number} 计算失败: {e}")
            errors += 1

    # 直接操作 SQLite 写入 target_key
    if trial_scores:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 获取 study_id
            cursor.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (study_name,),
            )
            row = cursor.fetchone()
            if row is None:
                logger.error(f"Study '{study_name}' 不存在于数据库中")
                conn.close()
                return {"total": total, "processed": 0, "skipped": skipped, "errors": errors + 1}

            study_id = row[0]

            # 获取所有 trial_id 和 number 的映射
            cursor.execute(
                "SELECT trial_id, number FROM trials WHERE study_id = ?",
                (study_id,),
            )
            trial_id_map = {row[1]: row[0] for row in cursor.fetchall()}

            # 写入 target_key 到 trial_user_attributes 表
            for trial_number, norm_score in trial_scores.items():
                trial_id = trial_id_map.get(trial_number)
                if trial_id is None:
                    continue

                # 检查是否已存在
                cursor.execute(
                    "SELECT COUNT(*) FROM trial_user_attributes "
                    "WHERE trial_id = ? AND key = ?",
                    (trial_id, target_key),
                )
                exists = cursor.fetchone()[0] > 0

                if exists:
                    cursor.execute(
                        "UPDATE trial_user_attributes SET value_json = ? "
                        "WHERE trial_id = ? AND key = ?",
                        (json.dumps(norm_score), trial_id, target_key),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO trial_user_attributes (trial_id, key, value_json) "
                        "VALUES (?, ?, ?)",
                        (trial_id, target_key, json.dumps(norm_score)),
                    )
                processed += 1

            conn.commit()
            conn.close()
            logger.info(f"SQLite 写入完成: {processed} 条 {target_key} 记录")

        except Exception as e:
            logger.error(f"SQLite 写入失败: {e}")
            errors += len(trial_scores) - processed + skipped
            processed = 0

    result = {
        "total": total,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info(
        f"追溯归一化({target_key})完成: 总数={total}, 处理={processed}, "
        f"跳过={skipped}, 错误={errors}"
    )
    return result


def _parse_weights(weight_args: list) -> Dict[str, float]:
    """解析命令行权重参数，格式: metric:weight"""
    weights = {}
    for arg in weight_args:
        try:
            name, val = arg.split(":")
            weights[name.strip()] = float(val.strip())
        except (ValueError, AttributeError):
            logger.warning(f"忽略无效权重参数: {arg}")
    return weights


def main():
    parser = argparse.ArgumentParser(description="追溯式归一化工具")
    parser.add_argument("--db", required=True, help="SQLite 数据库文件路径")
    parser.add_argument("--study", required=True, help="Study 名称")
    parser.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="权重列表，格式: metric:weight (如 val_icir:0.2 ic_gap_penalty:0.25)",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        default=False,
        help="使用 P1 自适应型 2.0 模式：写入 retro_norm_score_v2 字段，"
             "包含新增 3 个 P1 因变量（val_ic_stability/val_ir_stability/turnover_penalty）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    weights = _parse_weights(args.weights) if args.weights else None
    if weights is None:
        logger.error("必须提供 --weights 参数")
        return

    result = retroactively_normalize_study(
        db_path=args.db,
        study_name=args.study,
        objective_weights=weights,
        v2=args.v2,
    )
    print(f"结果: {result}")


if __name__ == "__main__":
    main()
