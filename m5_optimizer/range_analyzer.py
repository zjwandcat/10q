"""
M5范围分析器
从study的Trial中，按因变量过滤条件反推参数范围
"""
import logging
from typing import Dict, Tuple, List, Any
from math import floor, ceil
import numpy as np

import optuna

from m5_optimizer.search_space import ALL_PARAMS

logger = logging.getLogger("m5.range_analyzer")


def count_matched(
    study: optuna.Study,
    filter_conditions: Dict[str, Tuple[float, float]],
) -> Tuple[int, int]:
    """
    快速统计满足条件的trial数，供Tab2实时更新

    filter_conditions约定用(-999, 999)表示该指标不限制
    """
    n_matched = 0
    n_total = 0

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        # ★ 排除异常Trial（value<=-999）
        if trial.value is not None and trial.value <= -999.0:
            continue
        n_total += 1

        matched = True
        for metric, (min_val, max_val) in filter_conditions.items():
            value = trial.user_attrs.get(metric)
            if value is None:
                continue
            if min_val is not None and min_val > -999 and value < min_val:
                matched = False
                break
            if max_val is not None and max_val < 999 and value > max_val:
                matched = False
                break

        if matched:
            n_matched += 1

    return n_matched, n_total


def analyze_ranges(
    study: optuna.Study,
    filter_conditions: Dict[str, Tuple[float, float]],
    top_pct: float = 100.0,
) -> Dict[str, Any]:
    """
    从study的Trial中，按因变量过滤条件反推参数范围

    filter_conditions约定用(-999, 999)表示该指标不限制
    top_pct: 在满足因变量条件的trial中，按 score(trial.value) 降序保留前 top_pct%。
             100.0 = 全部保留（不过滤），10.0 = 仅保留前 10%。
    """
    matched_trials = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        # ★ 排除异常Trial（value<=-999）
        if trial.value is not None and trial.value <= -999.0:
            continue

        matched = True
        for metric, (min_val, max_val) in filter_conditions.items():
            value = trial.user_attrs.get(metric)
            if value is None:
                continue
            if min_val is not None and min_val > -999 and value < min_val:
                matched = False
                break
            if max_val is not None and max_val < 999 and value > max_val:
                matched = False
                break

        if matched:
            matched_trials.append(trial)

    n_matched = len(matched_trials)
    n_total = len([t for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE
                   and not (t.value is not None and t.value <= -999.0)])

    if n_matched == 0:
        return {
            "n_matched": 0,
            "n_total": n_total,
            "ranges": {},
        }

    # ★ 按 score (trial.value) 降序保留前 top_pct%
    # top_pct=100 时不过滤（保持全部）
    if top_pct < 100.0 and n_matched > 1:
        scored = [(t.value, t) for t in matched_trials if t.value is not None]
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            keep_n = max(1, int(len(scored) * top_pct / 100.0 + 1e-9))
            matched_trials = [t for _, t in scored[:keep_n]]
            n_matched = len(matched_trials)

    param_values: Dict[str, List[Any]] = {name: [] for name in ALL_PARAMS}

    for trial in matched_trials:
        for name in ALL_PARAMS:
            if name in trial.params:
                param_values[name].append(trial.params[name])

    ranges: Dict[str, Dict] = {}

    for name, values in param_values.items():
        if not values:
            continue

        pdef = ALL_PARAMS[name]
        ptype = pdef["type"]

        if ptype == "categorical":
            from collections import Counter
            all_vals = [
                t.params[name]
                for t in matched_trials
                if name in t.params
            ]
            if not all_vals:
                continue
            counter = Counter(all_vals)
            mode_val, mode_count = counter.most_common(1)[0]
            mode_freq = mode_count / len(all_vals)
            ranges[name] = {
                "type":              "categorical",
                "choices":           pdef["choices"],
                "mode":              mode_val,
                "mode_freq":         round(mode_freq, 4),
                "recommended_fixed": mode_freq >= 0.80,
                "default":           pdef["default"],
                "low":  None,
                "high": None,
            }
            continue

        try:
            vals = np.array(values, dtype=float)
            # ★ top_pct 已在前面按 score 筛过 trial，这里用 min/max 即可
            p_low = float(np.min(vals))
            p_high = float(np.max(vals))

            if ptype == "int":
                p_low = int(floor(p_low))
                p_high = int(ceil(p_high))
            else:
                p_low = round(float(p_low), 4)
                p_high = round(float(p_high), 4)

            ranges[name] = {
                "low": p_low,
                "high": p_high,
                "type": ptype,
                "default": pdef["default"],
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"参数{name}范围计算失败: {e}")
            continue

    return {
        "n_matched": n_matched,
        "n_total": n_total,
        "ranges": ranges,
    }
