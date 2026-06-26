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
import multiprocessing as mp
import sys
import traceback
import time
from typing import Dict, Optional, Any, List

import psutil
import numpy as np

from m5_optimizer.search_space import (
    ALL_PARAMS, OBJECTIVE_VARS, DEFAULT_PARAMS,
    assemble_params, NORM_CONFIG, FIXED_WINDOW_COUNT,
    calc_window_count,
)
from m5_optimizer.adaptive_normalizer import AdaptiveNormalizer

from m5_optimizer.utils.logger import get_logger, log_trial_result
from m5_optimizer.utils.win_memory import release_memory_to_os
from m5_optimizer.utils.rolling_logger import (
    get_rolling_logger, _try_log, _collect_system_stats, _get_vram_mb,
)
from m5_optimizer.config_manager import load_config as _load_m5_config_fn
logger = get_logger("m5.objective")

IC_GAP_PENALTY_MULTIPLIER = 1.5

# ★ 内存硬限制配置
# ★ v4.2 调整: 阈值分层 + 与 watchdog 对齐
#   - RSS 12GB: 进程主动防御（自身能撑住时主动归还）
#   - AVAIL 0.5GB: 系统真没空间了才跳过Trial
#     （半夜从 1.0 降到 0.5 反而合理，因为 watchdog 已经接管"立即停"，这里负责"下一轮跳过"）
#   - VRAM 7GB: GPU 显存警戒（避免 CUDA OOM）
_OOM_RSS_LIMIT_GB = 12.0
_OOM_AVAIL_LIMIT_GB = 0.5
_OOM_VRAM_LIMIT_MB = 7000

# Trial 超时 30 分钟，防止单 Trial 挂死拖垮进程
_TRIAL_TIMEOUT_SEC = 1800


def _run_m2_worker(run_kwargs: dict, result_queue: mp.Queue) -> None:
    """0xc0000005 子进程隔离: 在独立 spawn 子进程中执行 LightGBM 训练.

    必须定义在模块顶层, 以便 Windows spawn 模式通过 pickle 序列化目标函数。
    子进程内发生的任何 Python 异常都会被捕获并写入 result_queue, 确保子进程以
    exitcode=0 正常退出; C 层崩溃则会导致 exitcode != 0, 由父进程识别。

    v5.2 内存优化: 子进程内自加载 factor_df (不通过 pickle 传递 ~1-2GB),
    避免 spawn 序列化导致的内存峰值。
    """
    try:
        if run_kwargs.get("gpu_mode"):
            from m2_engine_gpu.run_m2 import run_m2
        else:
            from m2_engine.run_m2 import run_m2

        portfolios, stats = run_m2(**run_kwargs)

        # 0xc0000005 子进程隔离: 在子进程内提取指标, 只把可序列化的轻量结果传回父进程
        all_metrics = {}
        all_metrics["val_ic"] = _safe_float(stats.get("avg_val_ic", 0))
        all_metrics["val_icir"] = _safe_float(stats.get("avg_val_icir", 0))
        all_metrics["ic_gap_penalty"] = _safe_float(stats.get("avg_ic_gap", 0))
        all_metrics["penalized_rate"] = float(
            int(stats.get("low_confidence_months", 0)) /
            max(int(stats.get("success", 1)), 1)
        )

        avg_val_metrics = stats.get("avg_val_portfolio_metrics", {})
        _metric_keys = [
            "val_rolling6m_ir", "val_rolling6m_dir",
            "val_rolling6m_sortino", "val_rolling6m_return",
            "pct_positive_excess", "ir_worst_quartile",
            "val_jensen_alpha",
            "val_appraisal_ratio", "val_beta",
            "val_net_annual_return", "val_sqn",
        ]
        for _mk in _metric_keys:
            all_metrics[_mk] = _safe_float(avg_val_metrics.get(_mk, 0))

        # ════════════════════════════════════════════════════════════════════
        # ★ Task 1: 核心重构 — M4 引擎注入（统一金融指标口径，消灭双轨制）
        # 在 M2 跑完后立即拦截 portfolios DataFrame, 用向量化 groupby 快速构建
        # monthly DataFrame, 再调 m4_report 的 PerformanceMetrics.calculate()
        # 完成 28+ 项扣费后净收益指标计算。
        #
        # 性能优化: 跳过 M4.compute_monthly_returns() 中的慢 groupby.apply 路径,
        #   自建向量化 monthly DataFrame, 实测 76ms / Trial (vs 原 1270ms, 16x↑).
        # 用 M4 精准口径覆盖/补全 M2 简化指标, 0.2s 预算内完成。
        # ════════════════════════════════════════════════════════════════════
        m4_metrics = {}
        if portfolios is not None and len(portfolios) > 0:
            try:
                from m4_report.metrics import (
                    PerformanceMetrics, _load_benchmark,
                    _calc_benchmark_returns_point_to_point,
                )
                from m2_engine.portfolio_builder import calculate_turnover_cost

                # ─── Step A: 向量化构造 monthly DataFrame (替代慢 groupby.apply) ───
                holdings = portfolios[portfolios['is_holding'] == 1].copy()
                if len(holdings) > 0:
                    monthly = (
                        holdings.groupby('pred_month', sort=True)
                        .agg(
                            portfolio_return=(
                                'Target_Return_1M',
                                lambda x: (holdings.loc[x.index, 'weight'] * x).sum(),
                            ),
                            is_penalized=('is_penalized', 'first'),
                            val_ic=('val_ic', 'first'),
                            ic_gap=('ic_gap', 'first'),
                        )
                        .reset_index()
                    )
                    # 换手成本: 用 M4 同样的 calculate_turnover_cost
                    mh = (
                        holdings.groupby('pred_month')
                        .apply(lambda g: dict(zip(g['stock_code'], g['weight'])))
                        .to_dict()
                    )
                    prev_h = {}
                    tc = []
                    for m in sorted(monthly['pred_month'].tolist()):
                        curr = mh[m]
                        cost = calculate_turnover_cost(
                            prev_h, curr,
                            stamp_duty=0.001, commission=0.0003, slippage=0.001,
                        ) if prev_h else 0.0
                        tc.append(cost)
                        prev_h = curr
                    monthly['turnover_cost'] = tc
                    monthly['net_return'] = monthly['portfolio_return'] - monthly['turnover_cost']
                    # 基准: 复用 M4 内部点对点算法
                    bench_df = _load_benchmark()
                    if not bench_df.empty:
                        br = _calc_benchmark_returns_point_to_point(bench_df)
                        monthly['benchmark_return'] = monthly['pred_month'].map(br).fillna(0.0)
                    else:
                        monthly['benchmark_return'] = 0.0
                    monthly['excess_return'] = monthly['portfolio_return'] - monthly['benchmark_return']

                    # ─── Step B: 调 M4 的 calculate() 取 37 项指标 ───
                    pm = PerformanceMetrics()
                    full_m4 = pm.calculate(monthly) or {}

                    # M2 漏算 + 需覆盖的 M4 精准指标 (与 OBJECTIVE_VARS 对齐)
                    _M4_TARGETS = [
                        "cagr", "net_cagr_after_cost", "monthly_win_rate",
                        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "ir",
                        "max_drawdown", "up_capture_ratio", "down_capture_ratio",
                        "capture_ratio", "var_95", "cvar_95", "pain_index",
                        "ulcer_index", "skewness", "kurtosis", "omega_ratio",
                        "tail_ratio", "burke_ratio", "martin_ratio", "sterling_ratio",
                        "downside_volatility", "upside_volatility", "volatility_ratio",
                        "n_months", "avg_monthly_turnover_cost",
                        "avg_annual_turnover_cost", "rolling6m_win_rate",
                        "rolling6m_ir", "annual_excess",
                    ]
                    for _k in _M4_TARGETS:
                        if _k in full_m4:
                            m4_metrics[_k] = _safe_float(full_m4[_k])
            except Exception as m4_err:
                # M4 失败仅降级 M4 独有字段, 不影响 M2 主流程
                import logging as _lg
                _lg.getLogger("m5_optimizer.worker").warning(
                    f"[M4 Engine Alignment Failed] 降级至原 M2 口径. 原因: {m4_err}"
                )

        # ─── [A] 任务 1 重叠指标覆盖: 用 M4 精准口径替换 M2 简化值 ───
        # 这些指标 M2 旧版有简版, 数值与 M4 不一致, 统一以 M4 为准
        _OVERLAP_KEYS = [
            "up_capture_ratio", "down_capture_ratio", "capture_ratio",
            "var_95", "cvar_95", "pain_index", "ir", "monthly_win_rate",
            "max_drawdown", "calmar_ratio", "sortino_ratio", "sharpe_ratio",
        ]
        for _k in _OVERLAP_KEYS:
            if _k in m4_metrics:
                all_metrics[_k] = m4_metrics[_k]
            elif _k in ("var_95", "cvar_95", "pain_index"):
                # M4 偶发失败时, 保留 M2 旧 fallback 路径
                _m2_fb = {
                    "var_95": "val_var_95", "cvar_95": "val_cvar_95",
                    "pain_index": "val_pain_index",
                }
                all_metrics[_k] = _safe_float(avg_val_metrics.get(_m2_fb[_k], 0))

        # ─── [B] 任务 1 盲区激活: M4 独有指标 (M2 完全没有) 全量注入 ───
        # 这些指标 M2 不计算, Trial 阶段原为 0.0, 现已能被 Objective 评分感知
        _M4_EXCLUSIVE = [
            "cagr", "net_cagr_after_cost", "annual_excess", "n_months",
            "skewness", "kurtosis", "omega_ratio", "tail_ratio",
            "burke_ratio", "martin_ratio", "sterling_ratio", "ulcer_index",
            "downside_volatility", "upside_volatility", "volatility_ratio",
            "avg_monthly_turnover_cost", "avg_annual_turnover_cost",
            "rolling6m_win_rate", "rolling6m_ir",
        ]
        for _k in _M4_EXCLUSIVE:
            all_metrics[_k] = m4_metrics.get(_k, 0.0)

        # ════════════════════════════════════════════════════════════════════
        # 旧 M2 路径保留 (向后兼容, 防止 M2-only 字段丢失)
        # ════════════════════════════════════════════════════════════════════
        # 注: 这两个指标若 M4 未提供 (极少见), 用 M2 简化版兜底
        all_metrics["val_global_ir"] = _safe_float(
            avg_val_metrics.get("val_global_ir") or stats.get("val_global_ir", 0))
        all_metrics["val_annual_return"] = _safe_float(
            avg_val_metrics.get("val_annual_return") or stats.get("val_annual_return", 0))

        # 0xc0000005 子进程隔离: 收集子进程系统指标回传父进程
        rss_gb = psutil.Process().memory_info().rss / 1e9
        sys_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        vram_mb = _get_vram_mb()

        result_queue.put({
            "success": True,
            "all_metrics": all_metrics,
            "rss_gb": rss_gb,
            "sys_avail_gb": sys_avail_gb,
            "vram_mb": vram_mb,
        })
        return
    except Exception as e:
        # 0xc0000005 子进程隔离: Python 异常不 raise, 让子进程以 exitcode=0 退出
        result_queue.put({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)[:500]}",
            "traceback": traceback.format_exc()[:2000],
        })
        return


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
        project_id=None,
        neutralization_type=None,
        use_global_anchor: bool = True,
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
        self.project_id = project_id
        # 当前 Trial 真实运行的中性化策略（仅用于元数据 trial attr 与 Level 2 fallback）
        self.neutralization_type = neutralization_type
        # ★★★ 方案A 全局单一锚定开关（默认 True）★★★
        # True  → 归一化计算统一读 strategy_b_quantiles.json
        # False → 归一化计算读 self.neutralization_type 自身 json
        self.use_global_anchor = use_global_anchor
        self._direction_map = {v["name"]: v["direction"] for v in OBJECTIVE_VARS}
        # ★ 归一化开关与配置（实例级深拷贝，线程安全）
        self.enable_normalization = enable_normalization
        self.norm_config = copy.deepcopy(norm_config) if norm_config else copy.deepcopy(NORM_CONFIG)
        # ★ 自适应分位数归一化器（基于经验分位数 [P5, P50, P95] 替代硬编码边界）
        # 在 __init__ 中实例化，根据 neutralization_type 动态加载对应策略的分位数配置
        # 启用方案A：归一化计算使用 _GLOBAL_ANCHOR_KEY（默认 strategy_b）
        # 三级 Fallback 链确保永不崩溃：L1 全局锚定 → L2 自身 json → L3 静态 NORM_CONFIG
        self._adaptive_normalizer = AdaptiveNormalizer(
            neutralization_type=self.neutralization_type,
            use_global_anchor=self.use_global_anchor,
        )
        logger.info(
            "[ObjectiveFunction] 真实策略=%s | 归一化锚定=%s | 加载层级=L%d | "
            "污染指标=%d",
            self.neutralization_type,
            self._adaptive_normalizer.get_anchor_strategy()
                or "<static_fallback>",
            self._adaptive_normalizer.get_load_level(),
            len(self._adaptive_normalizer.get_corrupted_metrics()),
        )
        # ★ date_return_map 跨 Trial 缓存（避免每次重建 ~300MB dict）
        self._cached_date_return_map = None
        self._cached_date_return_scheme = None
        self._scheme_str = scheme

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
        # ★ 记录元信息：即使 Trial 因 OOM/停止被跳过也要写入
        # 【极其重要】元数据隔离：meta_neutralization_type 必须记录
        # 真实运行的策略，即使归一化计算使用了全局锚定。
        try:
            trial.set_user_attr("meta_project_id", str(self.project_id) if self.project_id else "20BB")
            trial.set_user_attr(
                "meta_neutralization_type",
                str(self.neutralization_type) if self.neutralization_type else "strategy_b",
            )
            # 诊断性元数据（不参与评分逻辑，仅用于报告层透明化）
            trial.set_user_attr(
                "meta_score_anchor",
                self._adaptive_normalizer.get_anchor_strategy()
                or "static_fallback",
            )
            trial.set_user_attr(
                "meta_score_load_level",
                int(self._adaptive_normalizer.get_load_level()),
            )
        except Exception:
            pass

        # ★ 立即停止：在Trial入口检查stop_event
        # 若已触发，抛出异常让 study.optimize 的 catch=(Exception,) 把当前 Trial 标记为 FAIL
        # 配合 trial_callback 中的 study.stop() 阻止下一 Trial
        if self.stop_event is not None and self.stop_event.is_set():
            raise RuntimeError(
                f"Trial#{trial.number} 因用户点击【立即停止】而中止"
            )

        # ★ OOM 前置检查：系统可用内存不足时跳过 Trial（返回 -999）
        # 避免在内存已满时继续跑 Trial 导致 Windows 强杀进程
        try:
            _avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            if _avail_gb < _OOM_AVAIL_LIMIT_GB:
                rl = get_rolling_logger()
                rl.log_critical(
                    f"OOM_PREVENT | Trial#{trial.number} 跳过: "
                    f"系统可用内存={_avail_gb:.1f}GB < {_OOM_AVAIL_LIMIT_GB}GB"
                )
                logger.critical(
                    f"Trial#{trial.number} 因内存不足跳过: "
                    f"可用={_avail_gb:.1f}GB < {_OOM_AVAIL_LIMIT_GB}GB"
                )
                try:
                    trial.set_user_attr("trial_error",
                        f"OOM_SKIP: avail={_avail_gb:.1f}GB")
                    trial.set_user_attr("elapsed_sec", 0.0)
                except Exception:
                    pass
                return -999.0
        except Exception:
            pass  # psutil 失败不阻止 Trial

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

            # ★ 优化: 局部导入移到模块级, 避免每次 Trial 重复导入查找
            _m5_cfg = _load_m5_config_fn()
            _gpu_mode = bool(self.gpu_mode) and _m5_cfg.get(
                "optimization", {}).get("gpu_mode", False)
            _gpu_strategy = _m5_cfg.get(
                "optimization", {}).get("gpu_strategy", "D")

            # ★ Bug fix: 之前这里误写为 _get_m5_config()（仅 phase1/phase2 定义）,
            #   实际本文件内只有 _load_m5_config_fn() 与局部变量 _m5_cfg / _gpu_mode.
            #   现统一改用 _gpu_mode (已合并 self.gpu_mode 与配置 gpu_mode).
            if _gpu_mode:
                _run_m2_kwargs = dict(
                    lgbm_params=lgbm_params,
                    xgbm_params=xgbm_params,
                    feature_params=feature_params,
                    lgbm_weight=lgbm_weight,
                    # ★ 关键: 必须用与 m5tab2 一致的窗口数, 否则 M4 指标无法对齐
                    # m5tab2 用 fast_mode=False 跑全量 (train_months=36 → 180窗)
                    # fast_window_count >= total_months 时 M2 内部不会预切片
                    # (=等同 fast_mode=False 的窗口范围)
                    fast_mode=True,
                    fast_window_count=calc_window_count(
                        window_params.get("train_months", 36)),
                    compute_val_metrics=self.compute_val_metrics,
                    compute_shap=False,
                    verbose=False,
                    preloaded_windows=None,
                    preloaded_factor_df=None,
                    date_return_map=None,
                    gpu_mode=True,
                    strategy=_gpu_strategy,
                    stop_event=self.stop_event,
                    train_months=window_params.get("train_months", 36),
                    scheme=self._scheme_str,
                    m5_optimize=True,
                )
            else:
                _run_m2_kwargs = dict(
                    lgbm_params=lgbm_params,
                    xgbm_params=xgbm_params,
                    feature_params=feature_params,
                    lgbm_weight=lgbm_weight,
                    # ★ CPU 路径: 同样用动态窗口数, 与 m5tab2 + GPU 路径一致
                    fast_mode=True,
                    fast_window_count=calc_window_count(
                        window_params.get("train_months", 36)),
                    compute_val_metrics=self.compute_val_metrics,
                    compute_shap=False,
                    verbose=False,
                    preloaded_windows=None,
                    preloaded_factor_df=None,
                    date_return_map=None,
                    gpu_mode=False,
                    stop_event=self.stop_event,
                    train_months=window_params.get("train_months", 36),
                    scheme=self._scheme_str,
                    m5_optimize=True,
                )

            # 0xc0000005 子进程隔离: 显式 flush 标准输出/错误,
            # 避免子进程继承未 flush 的缓冲导致日志错乱
            sys.stdout.flush()
            sys.stderr.flush()

            # 0xc0000005 子进程隔离: threading.Event 无法 pickle, 不传给子进程
            _run_m2_kwargs.pop("stop_event", None)

            # 0xc0000005 子进程隔离: 使用 spawn 子进程运行 LightGBM,
            # C 层崩溃只影响子进程, 主进程存活
            _trial_ctx = mp.get_context("spawn")
            _result_queue = _trial_ctx.Queue(maxsize=1)
            _child = _trial_ctx.Process(
                target=_run_m2_worker,
                args=(_run_m2_kwargs, _result_queue),
            )
            _child.start()
            _child.join(timeout=_TRIAL_TIMEOUT_SEC)

            # 0xc0000005 子进程隔离: 超时处理
            if _child.is_alive():
                _child.terminate()
                _child.join(timeout=5)
                if _child.is_alive():
                    _child.kill()
                    _child.join(timeout=5)
                # 0xc0000005 子进程隔离: 向队列放入哨兵, 避免后续 get 阻塞
                try:
                    _result_queue.put(None, timeout=5)
                except Exception:
                    pass
                sys.stdout.flush()
                sys.stderr.flush()
                _timeout_elapsed = time.time() - _trial_start_time
                logger.critical(
                    f"Trial#{trial.number} 超时 ({_timeout_elapsed:.0f}s > "
                    f"{_TRIAL_TIMEOUT_SEC}s)，强制终止"
                )
                rl = get_rolling_logger()
                _try_log(rl.log_critical,
                    f"TRIAL_TIMEOUT | Trial#{trial.number} "
                    f"elapsed={_timeout_elapsed:.0f}s > limit={_TRIAL_TIMEOUT_SEC}s")
                raise RuntimeError(
                    f"Trial#{trial.number} 超时 "
                    f"({_timeout_elapsed:.0f}s > {_TRIAL_TIMEOUT_SEC}s)"
                )

            sys.stdout.flush()
            sys.stderr.flush()

            # 0xc0000005 子进程隔离: 检查子进程退出码, 非 0 即 C 层崩溃
            if _child.exitcode != 0:
                try:
                    trial.set_user_attr(
                        "crash_reason",
                        f"C-level crash with exitcode {_child.exitcode}"
                    )
                except KeyError:
                    pass
                raise RuntimeError("Child process crashed unexpectedly.")

            # 0xc0000005 子进程隔离: 读取子进程返回结果
            try:
                _trial_result = _result_queue.get(timeout=30)
            except Exception:
                try:
                    trial.set_user_attr(
                        "crash_reason", "C-level crash: result queue empty")
                except KeyError:
                    pass
                raise RuntimeError("Child process crashed unexpectedly.")

            # 0xc0000005 子进程隔离: 显式关闭 Queue 释放缓冲区
            try:
                _result_queue.close()
                _result_queue.join_thread()
            except Exception:
                pass

            if _trial_result is None or not isinstance(_trial_result, dict):
                try:
                    trial.set_user_attr(
                        "crash_reason", "C-level crash: invalid result")
                except KeyError:
                    pass
                raise RuntimeError("Child process crashed unexpectedly.")

            # 0xc0000005 子进程隔离: 子进程内发生 Python 异常, 按 Trial 失败处理
            if not _trial_result.get("success", False):
                _err_msg = _trial_result.get("error", "Unknown error")
                _err_tb = _trial_result.get("traceback", "")
                logger.error(
                    f"Trial#{trial.number} 子进程异常: {_err_msg}\n{_err_tb}"
                )
                try:
                    trial.set_user_attr("trial_error", _err_msg)
                except Exception:
                    pass
                release_memory_to_os()
                return -999.0

            # 0xc0000005 子进程隔离: 子进程成功返回指标与系统监控数据
            all_metrics = _trial_result["all_metrics"]

            # 0xc0000005 子进程隔离: 立即释放 run_m2_kwargs 中对 factor_df 的引用
            # 避免在后续指标计算期间继续持有大 DataFrame
            del _run_m2_kwargs

            # 0xc0000005 子进程隔离: 立即释放子进程返回结果中的大对象引用
            # all_metrics 已提取完毕, _trial_result 中的 rss_gb/sys_avail_gb/vram_mb
            # 在后续几行读取后也无需保留整个 dict
            mem_gb = _trial_result.get(
                "rss_gb", psutil.Process().memory_info().rss / 1e9)
            vram_mb = _trial_result.get("vram_mb", _get_vram_mb())
            avail_gb = _trial_result.get(
                "sys_avail_gb",
                psutil.virtual_memory().available / (1024 ** 3))
            del _trial_result

            # 0xc0000005 子进程隔离: 指标已在子进程内提取, 父进程直接使用
            # 无需再持有 portfolios/stats 大对象
            gc.collect()

            # set_user_attr 可能因 Trial 记录被 cleanup 删除而抛 KeyError
            # 捕获后跳过写入，不影响 Trial 评分
            _trial_record_alive = True
            for metric, value in all_metrics.items():
                if not _trial_record_alive:
                    break
                try:
                    trial.set_user_attr(metric, value)
                except KeyError:
                    logger.warning(
                        f"Trial#{trial.number} 记录已不存在，跳过写入 metrics"
                    )
                    _trial_record_alive = False

            # ★ 评分计算（自适应分位数归一化 + 方案A 全局锚定）
            # 数学映射（统一在全局锚定 strategy_b 标尺上度量）：
            #   - Linear: score = (clip(x, P5, P95) - P5) / max(P95-P5, 1e-6) - 0.5
            #   - Tanh:   score = tanh(clip((x - P50) / max((P95-P5)/2, 1e-6), -50, 50))
            #   - identity/未覆盖: 全面纳入自适应 Tanh 桶
            #   - 污染指标(missing_rate/zero_rate > 0.9): 直接赋予 0.0 且不计入有效指标数
            #   - 元数据隔离: meta_neutralization_type 写真实策略，
            #     归一化锚定(meta_score_anchor)与真实策略解耦
            score = 0.0
            raw_score = 0.0  # 始终记录未归一化的原始评分
            effective_metric_count = 0  # ★ 有效指标数（排除污染指标）
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

                # 自适应归一化评分
                if self.enable_normalization:
                    # 读取原始归一化方法（作为路由桶选择依据）
                    cfg = self.norm_config.get(metric)
                    method = cfg.get("method", "identity") if cfg else "identity"

                    # 第一步：污染剔除 - 污染指标直接赋予 0.0，不参与加减分
                    # 且不计入 effective_metric_count，避免拉低综合平均分
                    if self._adaptive_normalizer.is_corrupted(metric):
                        norm_value = 0.0
                    else:
                        # 第二步：动态替换参数（在全局锚定 strategy_b 标尺上度量）
                        # - linear: 使用锚定 P5/P95 作为动态 bounds
                        # - tanh/signed_log/identity: 全面纳入自适应 Tanh 桶
                        norm_value = self._adaptive_normalizer.normalize(
                            metric, value, method, self.norm_config
                        )
                        # 仅非污染且归一化成功的指标计入有效数量
                        if np.isfinite(norm_value):
                            effective_metric_count += 1
                else:
                    norm_value = value
                    effective_metric_count += 1

                if direction == "max":
                    score += weight * multiplier * norm_value
                else:
                    score -= weight * multiplier * norm_value

            # 记录有效指标数（用于报告层透明化）
            try:
                trial.set_user_attr(
                    "meta_effective_metric_count",
                    int(effective_metric_count),
                )
            except Exception:
                pass

            # 第三步：总分平滑（防止 NaN）
            # 若由于极特殊情况产生 NaN，强制 fallback 回填为 -1000.0 恶性惩罚分
            if not np.isfinite(score):
                logger.warning(
                    f"Trial#{trial.number} 评分产生 NaN/Inf，"
                    f"回填为 -1000.0 惩罚分"
                )
                score = -1000.0

            # ★ 写入历史痕迹（Bug#2 防护）
            try:
                trial.set_user_attr("raw_score", raw_score)
                trial.set_user_attr("normalized_score", score)
            except KeyError:
                _trial_record_alive = False

            # 0xc0000005 子进程隔离: 内存+显存+系统可用内存监控
            # (mem_gb/vram_mb/avail_gb 已在 del _trial_result 前提取)

            # 写入 trial user_attrs 供诊断
            try:
                trial.set_user_attr("rss_gb", round(mem_gb, 2))
                trial.set_user_attr("sys_avail_gb", round(avail_gb, 2))
                if vram_mb > 0:
                    trial.set_user_attr("vram_mb", int(vram_mb))
            except Exception:
                pass

            # ★ RSS 超限：强制 GC + 归还内存
            if mem_gb > _OOM_RSS_LIMIT_GB:
                logger.warning(
                    f"RSS超限:{mem_gb:.1f}GB>{_OOM_RSS_LIMIT_GB}GB，"
                    f"强制GC+归还"
                )
                gc.collect()
                release_memory_to_os()
                # 归还后重新检查
                mem_gb_after = psutil.Process().memory_info().rss / 1e9
                if mem_gb_after > _OOM_RSS_LIMIT_GB:
                    rl.log_critical(
                        f"OOM_WARNING | Trial#{trial.number} RSS={mem_gb_after:.1f}GB "
                        f"归还后仍超限，后续Trial可能被Windows强杀"
                    )

            # ★ 系统可用内存不足：CRITICAL 日志
            if avail_gb < _OOM_AVAIL_LIMIT_GB:
                rl.log_critical(
                    f"OOM_IMMINENT | Trial#{trial.number} 系统可用内存="
                    f"{avail_gb:.1f}GB < {_OOM_AVAIL_LIMIT_GB}GB，"
                    f"Windows可能即将强杀进程"
                )

            # ★ VRAM 超限：警告
            if vram_mb > _OOM_VRAM_LIMIT_MB:
                logger.warning(
                    f"VRAM超限:{vram_mb}MB>{_OOM_VRAM_LIMIT_MB}MB，"
                    f"可能触发CUDA OOM"
                )
                # 尝试释放 CUDA 缓存
                try:
                    gc.collect()
                    # XGB Booster 析构会释放显存
                except Exception:
                    pass

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
                vram_mb, avail_gb,
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
