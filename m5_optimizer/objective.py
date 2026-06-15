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
M5目标函数
Optuna Trial目标函数封装
"""
import gc
import copy
import traceback
import time
from typing import Dict, Optional, Any, List

import psutil
import numpy as np

from m5_optimizer.search_space import (
    ALL_PARAMS, OBJECTIVE_VARS, DEFAULT_PARAMS,
    assemble_params, NORM_CONFIG,
)

from m5_optimizer.utils.logger import get_logger, log_trial_result
from m5_optimizer.utils.win_memory import release_memory_to_os
from m5_optimizer.utils.rolling_logger import (
    get_rolling_logger, _try_log, _collect_system_stats,
)
logger = get_logger("m5.objective")

IC_GAP_PENALTY_MULTIPLIER = 1.5


def _safe_float(value) -> float:
    """安全转换为float，支持list/np.ndarray/None等类型"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return 0.0
        try:
            return float(np.mean(value))
        except Exception:
            return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


# ── 评分归一化数学函数 ──────────────────────────────────────────────
_EPS = 1e-7


def _norm_tanh(value: float, scale: float) -> float:
    """Tanh 压缩：f(x) = tanh(x / scale)

    将值压缩到 (-1, 1)，scale 控制高灵敏度区间宽度。
    严格单调递增。
    """
    s = scale if scale > 0 else 1.0
    return float(np.tanh(value / s))


def _norm_signed_log(value: float, eps: float = _EPS) -> float:
    """Signed Log 压缩：f(x) = sign(x) * log(1 + |x| + eps)

    压缩长尾分布，保留符号。基底 e。
    严格单调递增。
    """
    sign = 1.0 if value >= 0 else -1.0
    return sign * float(np.log(1.0 + abs(value) + eps))


def _norm_linear(value: float, bounds: list, eps: float = _EPS) -> float:
    """Linear 截断映射：f(x) = (clip(x, lo, hi) - lo) / (hi - lo + eps) - 0.5

    将值映射到 [-0.5, 0.5]，超出 bounds 的值被截断到边界。
    在 [lo, hi] 内严格单调递增。
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi - lo < eps:
        return 0.0
    clipped = float(np.clip(value, lo, hi))
    return (clipped - lo) / (hi - lo + eps) - 0.5


def _apply_normalization(
    metric_name: str,
    value: float,
    norm_config: Optional[Dict] = None,
) -> float:
    """对单个指标应用归一化映射。

    根据 NORM_CONFIG 中声明的 method 路由到对应实现：
      - tanh        → _norm_tanh
      - signed_log  → _norm_signed_log
      - linear      → _norm_linear
    未命中配置的指标直接返回原值（identity 通道）。

    参数:
        metric_name: 指标名称
        value: 原始值（已通过 _safe_float 处理）
        norm_config: 归一化配置字典，None 则使用默认 NORM_CONFIG

    返回:
        归一化后的值
    """
    cfg = (norm_config or NORM_CONFIG).get(metric_name)
    if cfg is None:
        return value

    # NaN / Inf 防御
    if not np.isfinite(value):
        return 0.0

    method = cfg.get("method", "identity")
    if method == "tanh":
        return _norm_tanh(value, cfg.get("scale", 1.0))
    elif method == "signed_log":
        return _norm_signed_log(value)
    elif method == "linear":
        return _norm_linear(value, cfg.get("bounds", [0.0, 1.0]))
    else:
        return value


class ObjectiveFunction:
    def __init__(
        self,
        preloaded_factor_df,
        preloaded_windows: Optional[List] = None,
        window_count: int = None,
        compute_val_metrics: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
        active_params: Optional[List[str]] = None,
        scheme: str = "scheme_d",
        fast_mode: bool = True,
        stop_event=None,
        custom_ranges: Optional[Dict[str, Dict]] = None,
        enable_normalization: bool = False,
        norm_config: Optional[Dict] = None,
        gpu_mode: bool = False,  # ★ v3.5 新增：M5 评估时是否走 GPU 路径
    ):
        self.factor_df = preloaded_factor_df
        self.preloaded_windows = preloaded_windows
        self.window_count = window_count
        self.compute_val_metrics = compute_val_metrics
        self.objective_weights = objective_weights or {}
        self.active_params = active_params
        self.scheme = scheme
        self.fast_mode = fast_mode
        self.stop_event = stop_event
        self.custom_ranges = custom_ranges or {}
        self.gpu_mode = gpu_mode  # ★ v3.5
        self._direction_map = {v["name"]: v["direction"] for v in OBJECTIVE_VARS}
        # ★ 归一化开关与配置（实例级深拷贝，线程安全）
        self.enable_normalization = enable_normalization
        self.norm_config = copy.deepcopy(norm_config) if norm_config else copy.deepcopy(NORM_CONFIG)

    def _sample_param(self, trial, name: str, pdef: Dict) -> Any:
        custom = self.custom_ranges.get(name, {})

        if self.active_params and name not in self.active_params:
            # 非搜索参数：用默认值，但要确保在范围内
            custom = self.custom_ranges.get(name, {})
            # ★ 优先用adjusted_default，其次原始default
            default = custom.get("adjusted_default", DEFAULT_PARAMS[name])
            return default

        ptype = pdef["type"]
        match ptype:
            case "float_log":
                low = custom.get("low", pdef["low"])
                high = custom.get("high", pdef["high"])
                if low >= high:
                    logger.warning(f"{name} low({low})>=high({high})，使用原始范围")
                    low, high = pdef["low"], pdef["high"]
                return trial.suggest_float(name, low, high, log=True)
            case "float":
                low = custom.get("low", pdef["low"])
                high = custom.get("high", pdef["high"])
                if low >= high:
                    low, high = pdef["low"], pdef["high"]
                return trial.suggest_float(name, low, high)
            case "int":
                low = int(custom.get("low", pdef["low"]))
                high = int(custom.get("high", pdef["high"]))
                if low >= high:
                    low, high = int(pdef["low"]), int(pdef["high"])
                return trial.suggest_int(name, low, high)
            case "categorical":
                choices = custom.get("choices", pdef["choices"])
                return trial.suggest_categorical(name, choices)
            case _:
                return DEFAULT_PARAMS[name]

    def _assemble_params(self, sampled: Dict[str, Any]) -> tuple:
        return assemble_params(sampled)

    def _apply_lr_estimator_constraint(self, lgbm_params, xgbm_params):
        MAX_PRODUCT = 15.0
        lr = lgbm_params.get("learning_rate", 0.05)
        n_est = lgbm_params.get("n_estimators", 200)
        if lr * n_est > MAX_PRODUCT:
            new_n = int(MAX_PRODUCT / lr)
            lgbm_params = dict(lgbm_params)
            lgbm_params["n_estimators"] = max(new_n, 100)
            logger.debug(f"LGBM约束：lr={lr}×n_est={n_est}→{lgbm_params['n_estimators']}")

        lr_x = xgbm_params.get("learning_rate", 0.05)
        n_est_x = xgbm_params.get("n_estimators", 200)
        if lr_x * n_est_x > MAX_PRODUCT:
            new_n_x = int(MAX_PRODUCT / lr_x)
            xgbm_params = dict(xgbm_params)
            xgbm_params["n_estimators"] = max(new_n_x, 100)
            logger.debug(f"XGB约束：lr={lr_x}×n_est={n_est_x}→{xgbm_params['n_estimators']}")

        return lgbm_params, xgbm_params

    def __call__(self, trial) -> float:
        _trial_start_time = time.time()
        # ★ 立即停止：在Trial入口检查stop_event
        # 若已触发，抛出异常让 study.optimize 的 catch=(Exception,) 把当前 Trial 标记为 FAIL
        # 配合 trial_callback 中的 study.stop() 阻止下一 Trial
        if self.stop_event is not None and self.stop_event.is_set():
            raise RuntimeError(
                f"Trial#{trial.number} 因用户点击【立即停止】而中止"
            )
        try:
            sampled = {}
            for name, pdef in ALL_PARAMS.items():
                sampled[name] = self._sample_param(trial, name, pdef)

            rl = get_rolling_logger()
            _try_log(rl.log_trial_start, trial.number, sampled)
            if trial.number > 0 and trial.number % 10 == 0:
                _try_log(rl.log_system_stats, *_collect_system_stats())

            lgbm_params, xgbm_params, feature_params, lgbm_weight, window_params = (
                self._assemble_params(sampled)
            )

            lgbm_params, xgbm_params = self._apply_lr_estimator_constraint(
                lgbm_params, xgbm_params
            )

            from m5_optimizer.config_manager import load_config as _load_m5_cfg_fn
            _m5_cfg = _load_m5_cfg_fn()
            _gpu_mode = bool(self.gpu_mode) and _m5_cfg.get(
                "optimization", {}).get("gpu_mode", False)
            _gpu_strategy = _m5_cfg.get(
                "optimization", {}).get("gpu_strategy", "D")

            if _gpu_mode:
                # ★ v3.8: 走 m2_engine_gpu 路径, 支持 strategy
                from m2_engine_gpu.run_m2 import run_m2
                _run_m2_kwargs = dict(
                    lgbm_params=lgbm_params,
                    xgbm_params=xgbm_params,
                    feature_params=feature_params,
                    lgbm_weight=lgbm_weight,
                    fast_mode=self.fast_mode,
                    fast_window_count=self.window_count,
                    compute_val_metrics=self.compute_val_metrics,
                    compute_shap=False,
                    verbose=False,
                    preloaded_windows=None,
                    preloaded_factor_df=self.factor_df,
                    gpu_mode=True,
                    strategy=_gpu_strategy,
                    stop_event=self.stop_event,
                    train_months=window_params.get("train_months", 36),
                )
            else:
                from m2_engine.run_m2 import run_m2
                _run_m2_kwargs = dict(
                    lgbm_params=lgbm_params,
                    xgbm_params=xgbm_params,
                    feature_params=feature_params,
                    lgbm_weight=lgbm_weight,
                    fast_mode=self.fast_mode,
                    fast_window_count=self.window_count,
                    compute_val_metrics=self.compute_val_metrics,
                    compute_shap=False,
                    verbose=False,
                    preloaded_windows=None,
                    preloaded_factor_df=self.factor_df,
                    gpu_mode=False,
                    stop_event=self.stop_event,
                    train_months=window_params.get("train_months", 36),
                )

            portfolios, stats = run_m2(**_run_m2_kwargs)

            # ★ 修复: 立即释放 run_m2_kwargs 中对 factor_df 的引用
            # 避免在后续指标计算期间同时持有 portfolios 和 factor_df
            del _run_m2_kwargs

            all_metrics = {}
            all_metrics["val_ic"] = _safe_float(stats.get("avg_val_ic", 0))
            all_metrics["val_icir"] = _safe_float(stats.get("avg_val_icir", 0))
            all_metrics["ic_gap_penalty"] = _safe_float(stats.get("avg_ic_gap", 0))
            all_metrics["penalized_rate"] = float(
                int(stats.get("low_confidence_months", 0)) /
                max(int(stats.get("success", 1)), 1)
            )

            avg_val_metrics = stats.get("avg_val_portfolio_metrics", {})
            all_metrics["val_rolling6m_ir"] = _safe_float(
                avg_val_metrics.get("val_rolling6m_ir", 0)
            )
            all_metrics["val_rolling6m_dir"] = _safe_float(
                avg_val_metrics.get("val_rolling6m_dir", 0)
            )
            all_metrics["val_rolling6m_sortino"] = _safe_float(
                avg_val_metrics.get("val_rolling6m_sortino", 0)
            )
            all_metrics["val_rolling6m_return"] = _safe_float(
                avg_val_metrics.get("val_rolling6m_return", 0)
            )
            all_metrics["val_global_ir"] = _safe_float(
                avg_val_metrics.get("val_global_ir") or stats.get("val_global_ir", 0)
            )
            all_metrics["val_annual_return"] = _safe_float(
                avg_val_metrics.get("val_annual_return") or stats.get("val_annual_return", 0)
            )
            all_metrics["pct_positive_excess"] = _safe_float(
                avg_val_metrics.get("pct_positive_excess", 0)
            )
            all_metrics["ir_worst_quartile"] = _safe_float(
                avg_val_metrics.get("ir_worst_quartile", 0)
            )
            all_metrics["up_capture_ratio"] = _safe_float(
                avg_val_metrics.get("up_capture_ratio", 0)
            )
            all_metrics["down_capture_ratio"] = _safe_float(
                avg_val_metrics.get("down_capture_ratio", 0)
            )
            all_metrics["capture_ratio"] = _safe_float(
                avg_val_metrics.get("capture_ratio", 0)
            )
            all_metrics["val_jensen_alpha"] = _safe_float(
                avg_val_metrics.get("val_jensen_alpha", 0)
            )
            all_metrics["val_appraisal_ratio"] = _safe_float(
                avg_val_metrics.get("val_appraisal_ratio", 0)
            )
            all_metrics["val_beta"] = _safe_float(
                avg_val_metrics.get("val_beta", 0)
            )

            # ★ 释放 Trial 间内存
            del portfolios
            del stats
            gc.collect()

            for metric, value in all_metrics.items():
                trial.set_user_attr(metric, value)

            # ★ 评分计算（支持归一化开关）
            score = 0.0
            raw_score = 0.0  # 始终记录未归一化的原始评分
            for metric, weight in self.objective_weights.items():
                value = _safe_float(all_metrics.get(metric, 0))
                multiplier = (
                    IC_GAP_PENALTY_MULTIPLIER
                    if metric == "ic_gap_penalty"
                    else 1.0
                )
                direction = self._direction_map.get(metric, "max")

                # 原始评分（始终计算，用于 raw_score 记录）
                if direction == "max":
                    raw_score += weight * multiplier * value
                else:
                    raw_score -= weight * multiplier * value

                # 归一化评分（仅在开关开启时与 raw_score 不同）
                if self.enable_normalization:
                    norm_value = _apply_normalization(metric, value, self.norm_config)
                else:
                    norm_value = value

                if direction == "max":
                    score += weight * multiplier * norm_value
                else:
                    score -= weight * multiplier * norm_value

            # ★ 写入历史痕迹
            trial.set_user_attr("raw_score", raw_score)
            trial.set_user_attr("normalized_score", score)

            mem_gb = psutil.Process().memory_info().rss / 1e9
            if mem_gb > 9.5:
                logger.warning(f"内存超限:{mem_gb:.1f}GB，Trial继续")

            release_memory_to_os()

            # ★ 记录Trial结果到性能日志
            try:
                _elapsed = time.time() - _trial_start_time
                # ★ B-2 修复: 持久化 elapsed 到 user_attrs
                # 原因: trial_callback 进程重启/续跑时 _trial_times 为空, 无法预热 ETA
                # 修复后: 重启时从 study.trials[*].user_attrs["elapsed_sec"] 恢复
                try:
                    trial.set_user_attr("elapsed_sec", round(_elapsed, 1))
                except Exception:
                    pass
                log_trial_result(
                    trial_number=trial.number,
                    params=sampled,
                    metrics=all_metrics,
                    score=score,
                    elapsed_sec=_elapsed,
                )
            except Exception:
                pass  # 日志失败不影响主流程

            _try_log(
                rl.log_trial_end, trial.number, score,
                time.time() - _trial_start_time, mem_gb,
                all_metrics.get("val_ic", 0.0),
                all_metrics.get("val_rolling6m_ir", 0.0),
                all_metrics.get("pct_positive_excess", 0.0),
            )

            # ★ 每个Trial完成后主动释放内存，避免连续Trial累积OOM
            release_memory_to_os()
            return -score

        except Exception as e:
            # ★ B-10 修复: except 分支也记录 elapsed_sec (用于诊断 OOM/超时)
            _err_elapsed = time.time() - _trial_start_time
            logger.error(f"Trial异常 (耗时 {_err_elapsed:.1f}s): {e}\n{traceback.format_exc()}")
            try:
                trial.set_user_attr("elapsed_sec", round(_err_elapsed, 1))
                trial.set_user_attr("trial_error", f"{type(e).__name__}: {str(e)[:200]}")
            except Exception:
                pass
            rl = get_rolling_logger()
            _try_log(rl.log_error, type(e).__name__, str(e),
                     traceback.format_exc())
            release_memory_to_os()
            return -999.0
