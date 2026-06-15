"""
M5优化器 Gradio Web界面
Tab1: Phase1全局探索
Tab2: 结果分析与反推
Tab3: Phase2独立精调
Tab4: 结果与部署
"""
import os
import sys
import gc   # v4.1: 显式 GC 防止 16GB OOM
import time
import json
import queue
import shutil
import threading
import traceback
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from typing import Dict, Optional

import yaml
import psutil
import numpy as np
import gradio as gr
import optuna
import pandas as pd

from m5_optimizer.search_space import ALL_PARAMS, OBJECTIVE_VARS, METRIC_BOUNDS, calc_window_count, assemble_params, NORM_CONFIG
from m5_optimizer.result_analyzer import run_full_backtest
from m5_optimizer.phase1_global import run_phase1
from m5_optimizer.phase2_local import run_phase2
from m5_optimizer.range_analyzer import analyze_ranges, count_matched
from m5_optimizer.project_manager import (
    init_project, force_init_project,
    load_p1_config, load_p2_config, save_p2_config,
    export_p2_config_from_p1,
    get_p1_study, get_p2_study, get_study_stats,
    reset_p1, reset_p2, reset_all,
    cleanup_bad_trials,
)
from m5_optimizer.utils.logger import get_logger
from m5_optimizer.utils.win_memory import release_memory_to_os

from m1_engine.data_loader import DataLoader
from m1_engine.label_maker import LabelMaker

logger = get_logger("m5.app")

_stop_now_event = threading.Event()
_stop_graceful_event = threading.Event()
_phase1_thread: Optional[threading.Thread] = None
_phase2_thread: Optional[threading.Thread] = None
_factor_df_cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()  # v4.1: LRU 缓存 (有界, 最多 2 个)
_timer_thread: Optional[threading.Thread] = None
_timer_stop_event = threading.Event()  # 用于取消定时器

# ★ 日志队列（线程安全）—— Tab4 改用 gr.State 会话字典，详见 t4_session_state
_log_queue_p1 = queue.Queue()
_log_queue_p2 = queue.Queue()

PRESET_TEMPLATES = {
    "稳健型": {
        "weights": {
            "val_icir": 0.20,
            "ic_gap_penalty": 0.25,
            "penalized_rate": 0.15,
            "val_ic": 0.15,
            "val_rolling6m_sortino": 0.10,
        },
        "active": [
            "lgbm_max_depth", "xgb_max_depth",
            "lgbm_colsample_bytree", "xgb_colsample_bytree",
            "lgbm_reg_lambda", "xgb_reg_lambda",
            "min_keep_factors"
        ]
    },
    "激进型": {
        "weights": {
            "val_ic": 0.30,
            "val_rolling6m_ir": 0.25,
            "val_rolling6m_return": 0.25,
            "ic_gap_penalty": 0.10,
            "penalized_rate": 0.10,
        },
        "active": [
            "lgbm_learning_rate", "lgbm_n_estimators",
            "xgb_learning_rate", "xgb_n_estimators",
            "lgbm_weight", "min_ic_abs"
        ]
    },
    "防御型": {
        "weights": {
            "ic_gap_penalty": 0.15,
            "val_icir": 0.15,
        },
        "active": [
            "lgbm_max_depth", "xgb_max_depth",
            "lgbm_colsample_bytree", "xgb_colsample_bytree",
            "lgbm_reg_alpha", "lgbm_reg_lambda",
            "drop_short_term_noise"
        ]
    },
    "全自适应型": {
        "weights": {
            "val_icir":                 0.20,
            "ic_gap_penalty":           0.20,
            "val_rolling6m_excess_ann": 0.25,
            "pct_positive_excess":      0.20,
            "val_rolling6m_ir":         0.05,
            "ir_worst_quartile":        0.05,
            "val_ic":                   0.03,
            "penalized_rate":           0.02,
        },
        "active": "ALL_29"
    },

    "自适应型-b": {
        "weights": {
            "val_icir":                 0.20,
            "ic_gap_penalty":           0.20,
            "val_rolling6m_excess_ann": 0.25,
            "pct_positive_excess":      0.20,
            "val_rolling6m_ir":         0.05,
            "ir_worst_quartile":        0.05,
            "val_ic":                   0.03,
            "penalized_rate":           0.02,
        },
        "active": "ALL_29",
        "fixed_categorical": {
            "lgbm_lr_mode":          ["decay"],
            "lgbm_depth_mode":       ["adaptive"],
            "xgb_lr_mode":           ["decay"],
            "drop_short_term_noise": [True],
        },
        "fixed_int": ["lgbm_max_depth", "xgb_max_depth"],
        "fixed_int_values": {"lgbm_max_depth": 2, "xgb_max_depth": 2},
        "param_ranges": {
            "lgbm_learning_rate":    {"low": 0.015, "high": 0.060},
            "lgbm_n_estimators":     {"low": 80,    "high": 200},
            "lgbm_reg_lambda":       {"low": 5.0,   "high": 10.0},
            "lgbm_colsample_bytree": {"low": 0.08,  "high": 0.18},
            "lgbm_reg_alpha":        {"low": 0.05,  "high": 0.50},
            "lgbm_weight":           {"low": 0.45,  "high": 0.75},
            "xgb_learning_rate":     {"low": 0.030, "high": 0.090},
            "xgb_n_estimators":      {"low": 150,   "high": 250},
            "xgb_reg_lambda":        {"low": 0.8,   "high": 2.5},
            "xgb_colsample_bytree":  {"low": 0.08,  "high": 0.20},
            "xgb_reg_alpha":         {"low": 0.5,   "high": 2.0},
            "xgb_gamma":             {"low": 0.001, "high": 0.015},
            "xgb_decay_every":       {"low": 50,    "high": 70},
            "min_ic_abs":            {"low": 0.001, "high": 0.006},
            "min_keep_factors":      {"low": 96,    "high": 124},
            "max_corr":              {"low": 0.87,  "high": 0.93},
            "train_months":          {"low": 38,    "high": 48},
        },
        "warm_start": {
            "lgbm_learning_rate": 0.025035, "lgbm_n_estimators": 103,
            "lgbm_max_depth": 2, "lgbm_colsample_bytree": 0.121998,
            "lgbm_reg_alpha": 0.054697, "lgbm_reg_lambda": 8.875172,
            "lgbm_min_split_gain": 0.001511, "lgbm_lr_mode": "decay",
            "lgbm_decay_every": 62, "lgbm_decay_factor": 0.851966,
            "lgbm_depth_mode": "adaptive", "lgbm_early_stopping_rounds": 48,
            "xgb_learning_rate": 0.053149, "xgb_n_estimators": 200,
            "xgb_max_depth": 2, "xgb_colsample_bytree": 0.123482,
            "xgb_reg_alpha": 1.250456, "xgb_reg_lambda": 1.348387,
            "xgb_gamma": 0.003584, "xgb_lr_mode": "decay",
            "xgb_decay_every": 61, "xgb_decay_factor": 0.837351,
            "xgb_early_stopping_rounds": 40, "lgbm_weight": 0.699092,
            "min_valid_rate": 0.269893, "max_corr": 0.90811,
            "min_ic_abs": 0.002238, "min_keep_factors": 112,
            "drop_short_term_noise": True, "train_months": 42,
        },
    },

    "自适应型-a": {
        "weights": {
            "val_icir":                 0.20,
            "ic_gap_penalty":           0.20,
            "val_rolling6m_excess_ann": 0.25,
            "pct_positive_excess":      0.20,
            "val_rolling6m_ir":         0.05,
            "ir_worst_quartile":        0.05,
            "val_ic":                   0.03,
            "penalized_rate":           0.02,
        },
        "active": "ALL_29",
        "fixed_categorical": {
            "lgbm_lr_mode":          ["decay"],
            "lgbm_depth_mode":       ["fixed"],
            "xgb_lr_mode":           ["fixed"],
            "drop_short_term_noise": [True],
        },
        "fixed_int": ["lgbm_max_depth", "xgb_max_depth"],
        "fixed_int_values": {"lgbm_max_depth": 2, "xgb_max_depth": 2},
        "param_ranges": {
            "lgbm_learning_rate":    {"low": 0.030, "high": 0.100},
            "lgbm_n_estimators":     {"low": 90,    "high": 210},
            "lgbm_reg_lambda":       {"low": 2.5,   "high": 9.0},
            "lgbm_colsample_bytree": {"low": 0.12,  "high": 0.22},
            "lgbm_reg_alpha":        {"low": 0.05,  "high": 1.5},
            "lgbm_weight":           {"low": 0.30,  "high": 0.60},
            "xgb_learning_rate":     {"low": 0.010, "high": 0.045},
            "xgb_n_estimators":      {"low": 150,   "high": 300},
            "xgb_reg_lambda":        {"low": 0.8,   "high": 3.0},
            "xgb_reg_alpha":         {"low": 0.5,   "high": 2.0},
            "xgb_gamma":             {"low": 0.001, "high": 0.020},
            "xgb_decay_every":       {"low": 28,    "high": 56},
            "xgb_early_stopping_rounds": {"low": 18,"high": 32},
            "min_ic_abs":            {"low": 0.001, "high": 0.006},
            "min_keep_factors":      {"low": 82,    "high": 111},
            "max_corr":              {"low": 0.84,  "high": 0.90},
            "train_months":          {"low": 39,    "high": 51},
        },
        "warm_start": {
            "lgbm_learning_rate": 0.093656, "lgbm_n_estimators": 145,
            "lgbm_max_depth": 2, "lgbm_colsample_bytree": 0.174777,
            "lgbm_reg_alpha": 0.615223, "lgbm_reg_lambda": 9.815442,
            "lgbm_min_split_gain": 0.003723, "lgbm_lr_mode": "decay",
            "lgbm_decay_every": 76, "lgbm_decay_factor": 0.906511,
            "lgbm_depth_mode": "fixed", "lgbm_early_stopping_rounds": 31,
            "xgb_learning_rate": 0.013853, "xgb_n_estimators": 270,
            "xgb_max_depth": 2, "xgb_colsample_bytree": 0.180466,
            "xgb_reg_alpha": 1.692311, "xgb_reg_lambda": 1.389292,
            "xgb_gamma": 0.003629, "xgb_lr_mode": "fixed",
            "xgb_decay_every": 42, "xgb_decay_factor": 0.867913,
            "xgb_early_stopping_rounds": 22, "lgbm_weight": 0.348698,
            "min_valid_rate": 0.261382, "max_corr": 0.889073,
            "min_ic_abs": 0.00344, "min_keep_factors": 106,
            "drop_short_term_noise": True, "train_months": 44,
        },
    },

    "自适应型-e": {
        "weights": {
            "val_icir":                 0.20,
            "ic_gap_penalty":           0.20,
            "val_rolling6m_excess_ann": 0.25,
            "pct_positive_excess":      0.20,
            "val_rolling6m_ir":         0.05,
            "ir_worst_quartile":        0.05,
            "val_ic":                   0.03,
            "penalized_rate":           0.02,
        },
        "active": "ALL_29",
        "fixed_categorical": {
            "lgbm_lr_mode":          ["fixed"],
            "lgbm_depth_mode":       ["fixed"],
            "xgb_lr_mode":           ["decay"],
            "drop_short_term_noise": [True],
        },
        "fixed_int": ["lgbm_max_depth", "xgb_max_depth"],
        "fixed_int_values": {"lgbm_max_depth": 2, "xgb_max_depth": 3},
        "param_ranges": {
            "lgbm_learning_rate":    {"low": 0.040, "high": 0.120},
            "lgbm_n_estimators":     {"low": 220,   "high": 300},
            "lgbm_reg_lambda":       {"low": 2.0,   "high": 6.0},
            "lgbm_colsample_bytree": {"low": 0.17,  "high": 0.32},
            "lgbm_reg_alpha":        {"low": 0.05,  "high": 1.2},
            "lgbm_min_split_gain":   {"low": 0.001, "high": 0.025},
            "lgbm_early_stopping_rounds": {"low": 34, "high": 46},
            "lgbm_weight":           {"low": 0.35,  "high": 0.70},
            "xgb_learning_rate":     {"low": 0.008, "high": 0.045},
            "xgb_n_estimators":      {"low": 150,   "high": 250},
            "xgb_reg_lambda":        {"low": 0.5,   "high": 2.5},
            "xgb_reg_alpha":         {"low": 0.05,  "high": 1.5},
            "xgb_gamma":             {"low": 0.001, "high": 0.050},
            "xgb_decay_every":       {"low": 27,    "high": 51},
            "min_ic_abs":            {"low": 0.005, "high": 0.014},
            "min_keep_factors":      {"low": 98,    "high": 123},
            "max_corr":              {"low": 0.85,  "high": 0.91},
            "train_months":          {"low": 57,    "high": 60},
        },
        "warm_start": {
            "lgbm_learning_rate": 0.048464, "lgbm_n_estimators": 252,
            "lgbm_max_depth": 2, "lgbm_colsample_bytree": 0.244372,
            "lgbm_reg_alpha": 0.624999, "lgbm_reg_lambda": 3.831356,
            "lgbm_min_split_gain": 0.008577, "lgbm_lr_mode": "fixed",
            "lgbm_decay_every": 46, "lgbm_decay_factor": 0.616922,
            "lgbm_depth_mode": "fixed", "lgbm_early_stopping_rounds": 43,
            "xgb_learning_rate": 0.015183, "xgb_n_estimators": 162,
            "xgb_max_depth": 3, "xgb_colsample_bytree": 0.129024,
            "xgb_reg_alpha": 0.425986, "xgb_reg_lambda": 0.693289,
            "xgb_gamma": 0.038517, "xgb_lr_mode": "decay",
            "xgb_decay_every": 34, "xgb_decay_factor": 0.635908,
            "xgb_early_stopping_rounds": 22, "lgbm_weight": 0.442898,
            "min_valid_rate": 0.425391, "max_corr": 0.866507,
            "min_ic_abs": 0.009569, "min_keep_factors": 117,
            "drop_short_term_noise": True, "train_months": 59,
        },
    },

    # ★ P1 自适应型 2.0：四象限分层加权（19 项 P1 因变量，权重和=1.0）
    "自适应型 2.0": {
        "weights": {
            # --- 象限一：核心预测力与稳定性 (0.30) ---
            "val_icir":          0.15,
            "val_ic_stability":  0.10,
            "val_ic":            0.05,
            # --- 象限二：过拟合与成本控制 (0.23) ---
            "ic_gap_penalty":    0.14,
            "turnover_penalty":  0.05,
            "penalized_rate":    0.04,
            # --- 象限三：收益绝对幅度与胜率 (0.27) ---
            "val_rolling6m_excess_ann": 0.10,
            "pct_positive_excess":       0.07,
            "val_jensen_alpha":          0.05,
            "val_rolling6m_return":      0.02,
            "val_annual_return":         0.01,
            "val_rolling6m_excess":      0.01,
            # --- 象限四：风险调整后收益与抗跌性 (0.20) ---
            "val_rolling6m_ir":    0.05,
            "val_rolling6m_sortino": 0.05,
            "ir_worst_quartile":    0.05,
            "val_ir_stability":     0.03,
            "val_rolling6m_dir":    0.01,
            "val_global_ir":        0.01,
            "val_appraisal_ratio":  0.01,
        },
        "active": "ALL_29",
    },
}

# ★ 启动时断言：自适应型 2.0 权重和必须 = 1.0
assert abs(sum(PRESET_TEMPLATES["自适应型 2.0"]["weights"].values()) - 1.0) < 1e-6, \
    f"自适应型 2.0 权重总和异常: {sum(PRESET_TEMPLATES['自适应型 2.0']['weights'].values())}"


# ★ v3.8 新增: 把 UI 上的 GPU 设置写到 config/config.yaml
# phase1_global / phase2_local 启动时通过 _get_m5_config() 动态读
def _update_m5_gpu_config(enable_gpu: bool, strategy: str = "D"):
    """
    更新 M5 GPU 配置 · v4.1
    仅允许 D 策略 (用户要求: 与纯 CPU bit-exact)
    B/E 等历史策略自动 fallback 到 D
    """
    # v4.1: 强制 strategy 只能为 D
    valid_strategies = ("D",)
    if strategy not in valid_strategies:
        logger.warning(
            f"GPU strategy '{strategy}' 已在 v4.1 中移除, fallback 到 D")
        strategy = "D"
    try:
        cfg_path = "config/config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "m5" not in cfg:
            cfg["m5"] = {}
        if "optimization" not in cfg["m5"]:
            cfg["m5"]["optimization"] = {}
        cfg["m5"]["optimization"]["gpu_mode"] = bool(enable_gpu)
        cfg["m5"]["optimization"]["gpu_strategy"] = str(strategy)
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        logger.info(
            f"M5 GPU config 已更新: gpu_mode={enable_gpu}, strategy={strategy}")
    except Exception as e:
        logger.warning(f"更新 M5 GPU config 失败 (不影响运行): {e}")


def _load_factor_df(scheme: str = None):
    """
    v4.1: LRU 缓存 (最多 2 个 scheme), 防止 16GB 系统 OOM
    旧版 _factor_df_cache 是无界 dict, 用户切换 scheme 时会累积所有
    加载过的 factor_df (每个 ~1-2GB), 最终触发 OS OOM 弹窗
    """
    global _factor_df_cache
    cache_key = scheme or "default"
    MAX_CACHE_SIZE = 2   # v4.1: 最多缓存 2 个 scheme (避免 16GB OOM)

    if cache_key in _factor_df_cache:
        # v4.1: LRU - 移到末尾
        _factor_df_cache.move_to_end(cache_key)
        return _factor_df_cache[cache_key]

    try:
        # ★ 检查可用内存
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9
        logger.info(f"可用内存: {available_gb:.1f} GB")

        if available_gb < 2.0:
            raise MemoryError(
                f"可用内存不足 ({available_gb:.1f} GB < 2 GB)，"
                f"请关闭其他程序后重试"
            )

        # v4.1: 缓存满时清理最早加载的 scheme (OrderedDict 第一项)
        if len(_factor_df_cache) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(_factor_df_cache))
            logger.info(
                f"[v4.1 LRU] 缓存满 ({len(_factor_df_cache)}/{MAX_CACHE_SIZE}), "
                f"清理最早 scheme: {oldest_key}")
            del _factor_df_cache[oldest_key]
            gc.collect()

        loader = DataLoader(scheme=scheme)
        factor_df = loader.load()
        factor_df = LabelMaker().make_labels(factor_df)

        # ★ 加载后再次检查内存
        mem_after = psutil.virtual_memory()
        used_gb = (mem.total - mem_after.available) / 1e9
        logger.info(f"数据加载完成(scheme={cache_key})，已用内存: {used_gb:.1f} GB")

        _factor_df_cache[cache_key] = factor_df
        return factor_df

    except MemoryError as e:
        logger.error(f"内存不足: {e}")
        raise
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise


def start_graceful_timer(hours: float):
    """在后台线程里等待指定时间后触发优雅停止"""
    global _timer_thread, _timer_stop_event

    # 取消已有定时器
    _timer_stop_event.set()
    if _timer_thread and _timer_thread.is_alive():
        _timer_thread.join(timeout=1)

    _timer_stop_event = threading.Event()

    def _timer_worker():
        total_seconds = hours * 3600
        end_time = time.time() + total_seconds
        while not _timer_stop_event.wait(timeout=60):  # 每分钟检查一次
            remaining = end_time - time.time()
            if remaining <= 0:
                # 触发优雅停止（同用户点击🏁优雅停止）
                _stop_graceful_event.set()
                logger.info(f"定时器触发：{hours}小时已到，执行优雅停止")
                break

    _timer_thread = threading.Thread(target=_timer_worker, daemon=True)
    _timer_thread.start()


def cancel_graceful_timer():
    """取消定时器"""
    global _timer_stop_event
    _timer_stop_event.set()


def _card(status, project=None, stats=None, message="") -> str:
    """渲染项目状态卡片HTML（border-left色块，兼容深色/浅色主题）"""
    name = project["project_name"] if project else ""
    root = project["project_root"] if project else ""

    base = ("border-radius:6px;padding:12px 16px;margin:4px 0;"
            "border:1px solid var(--border-color-primary,#e0e0e0);"
            "border-left-width:4px;font-size:14px;")

    if status == "unconfigured":
        return (f"<div style='{base}border-left-color:#9e9e9e;opacity:0.7'>"
                f"⬜ 未加载项目 · 请输入文件夹路径后点击\"打开/新建\""
                f"</div>")

    if status == "conflict":
        return (f"<div style='{base}border-left-color:#ff9800'>"
                f"⚠️ {message}</div>")

    if status == "fresh":
        return (f"<div style='{base}border-left-color:#ffc107'>"
                f"🟡 <b>{name}</b> · 待开始<br>"
                f"<small>📂 {root}</small></div>")

    if status == "in_progress" and stats:
        c = stats["complete"]
        ab = stats.get("abnormal", 0)
        f_ = stats["fail"]
        t = stats["total"]
        best = f"{stats['best_value']:.4f}" if stats.get("best_value") is not None else "暂无有效得分"
        return (f"<div style='{base}border-left-color:#2196f3'>"
                f"🔵 <b>{name}</b> · 进行中<br>"
                f"<small>📂 {root}</small><br>"
                f"<small>累计{t}个Trial：有效{c}✅ 异常{ab}⚠️ 失败{f_}❌ · 最优{best}</small>"
                f"</div>")

    if status == "completed" and stats:
        c = stats["complete"]
        best = f"{stats['best_value']:.4f}" if stats.get("best_value") is not None else "—"
        return (f"<div style='{base}border-left-color:#4caf50'>"
                f"🟢 <b>{name}</b> · 已完成<br>"
                f"<small>📂 {root}</small><br>"
                f"<small>共{c}个有效Trial · 最优{best}</small>"
                f"</div>")

    return f"<div style='{base}border-left-color:#9e9e9e'>⬜ 未知状态</div>"


def _make_progress_callback(log_queue, log_interval_sec: float = 120.0):
    """
    生成 phase 进度回调：每 Trial 写一次完整日志（带时间预估）。
    log_queue: 任意 queue.Queue 实例（p1用_log_queue_p1，p2用_log_queue_p2）
    log_interval_sec: 时间预估行的刷新间隔
    """
    _last_log_update = [0.0]

    def progress_callback(trial_num, total, best_val, attrs, time_info=None):
        now = time.time()
        should_update_time = (now - _last_log_update[0]) >= log_interval_sec

        lines = [
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Trial {total} | 有效 {trial_num} | "
            f"最优 {f'{best_val:.4f}' if best_val is not None else '—'}"
        ]
        if time_info and (should_update_time or total <= 3):
            # ★ 修复: trial_callback 改名为 per_trial_str (旧: per_trial_min, 已废弃)
            lines.append(
                f"  ⏱ 单Trial耗时：{time_info.get('per_trial_str', time_info.get('per_trial_min', '计算中...'))} | "
                f"剩余预估：{time_info['eta']}"
            )
            _last_log_update[0] = now
        if attrs:
            ic   = attrs.get("val_ic", 0) or 0
            icir = attrs.get("val_icir", 0) or 0
            gap  = attrs.get("ic_gap_penalty", 0) or 0
            rate = attrs.get("penalized_rate", 0) or 0
            lines.append(
                f"  val_ic={ic:.4f} | icir={icir:.4f} | "
                f"ic_gap={gap:.4f} | 降权率={rate:.1%}"
            )
        log_text = "\n".join(lines) + "\n---\n"
        logger.info(log_text.strip())
        log_queue.put(log_text)

    return progress_callback


def _sig4(v, digits: int = 4) -> str:
    """v4.1: 4位有效数字格式 (用户要求)
    例: 0.083336 → 0.08334, 1.234567 → 1.235, 12.34567 → 12.35

    支持指定有效数字位数（digits），调用方按数值量级选择 2/3/4 位
    """
    if not isinstance(v, (int, float)):
        return str(v)
    return f"{float(v):.{max(1, int(digits))}g}"


def _get_metric_actual_range(study, metric_name):
    """从实际Trial中获取指标的真实分布范围（覆盖全部有效 Trial）

    策略：用 [min, max] + 极小 epsilon margin，确保初始化滑块后
          全部有效 Trial 都能通过过滤（== 214/214）。
    注：p2~p98+20% 会被长尾/离群值截断，导致少量 Trial 漏匹配，
        此处改用全量 min/max 兜底。

    v4.1 修复: 4位有效数字 (用户要求)
      0.083336 → 0.08334 (4位有效数字)
      对金融指标 1e-4 精度足够, 滑块 UI 更清爽
    """
    vals = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.value is not None and t.value <= -999.0:
            continue
        v = t.user_attrs.get(metric_name)
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        # ★ 过滤 NaN / Inf：避免后续 np.min/max 返回 NaN 导致 log10/整型转换异常
        if not np.isfinite(vf):
            continue
        vals.append(vf)

    if len(vals) < 2:
        return None, None

    arr = np.array(vals, dtype=np.float64)
    data_min = float(arr.min())
    data_max = float(arr.max())

    # 兜底：NaN/Inf 残留（理论上已被过滤）→ 当作无数据
    if not (np.isfinite(data_min) and np.isfinite(data_max)):
        return None, None

    # 全零/无方差：用 [0, 1] 兜底，避免后续 gr.update(minimum==maximum) 报错
    if data_max - data_min < 1e-9:
        return 0.0, 1.0

    # 极小 epsilon margin（相对量程的 5%），保证不顶到边界
    span = data_max - data_min
    margin = max(span * 0.05, 1e-6)
    # v4.1: 4位有效数字 (0.083336 → 0.08334)
    actual_min = float(np.float64(data_min - margin))
    actual_max = float(np.float64(data_max + margin))

    return actual_min, actual_max


def _compute_metric_slider_updates(study):
    """
    按OBJECTIVE_VARS顺序为每个m2来源指标生成 gr.update(...) 列表。

    行为（"初始化滑块"按钮 = 重置滑块的 minimum/maximum/value，
    全部基于有效 Trial 实际数据范围，确保能捕捉全部 Trial）：

        1) 有足够有效数据（≥2 个）→ 用 p2~p98+20% margin
           - 下限滑块：minimum=actual_min, maximum=actual_max, value=actual_min
           - 上限滑块：minimum=actual_min, maximum=actual_max, value=actual_max
        2) 实际值若超出 METRIC_BOUNDS → 仍以 actual_min/actual_max 为准
           （不再回退到 METRIC_BOUNDS，避免"实际数据超出滑块轨道"导致漏匹配）
        3) 数据不足（<2 个）→ 滑块保持当前状态（返回 gr.update()）

    返回 (min_updates, max_updates)，每个元素是 gr.update(minimum=, maximum=, value=, step=)
    """
    min_updates, max_updates = [], []
    for var in OBJECTIVE_VARS:
        if (var.get("source") != "m2"
                or not var.get("enabled")
                or var["name"].startswith("stress_")):
            continue
        name = var["name"]
        s_min, s_max = METRIC_BOUNDS.get(name, (-2, 2))
        actual_min, actual_max = _get_metric_actual_range(study, name)
        if actual_min is None or actual_max is None or actual_min >= actual_max:
            # ★ 无足够数据：不修改滑块（避免覆盖用户已有设定）
            min_updates.append(gr.update())
            max_updates.append(gr.update())
            continue
        v_min, v_max = actual_min, actual_max
        # ★ v4.1: 4位有效数字用于滑块显示
        #   rounding 精确到 4 sig figs: 0.08333609139234538 → 0.08334
        #   功能值仍是原始精度, 不影响过滤逻辑
        label_min = _sig4(v_min)
        label_max = _sig4(v_max)
        # round 到 4 sig figs: 修复 Gradio 滑块数值显示为 0.08334
        r_min = round(v_min, max(0, 4 - int(np.floor(np.log10(abs(v_min) + 1e-15))) - 1))
        r_max = round(v_max, max(0, 4 - int(np.floor(np.log10(abs(v_max) + 1e-15))) - 1))
        # ★ 下限滑块
        min_updates.append(gr.update(minimum=r_min, maximum=r_max, value=r_min,
                                      step=0.001, label=f"下限 ({label_min})"))
        # ★ 上限滑块
        max_updates.append(gr.update(minimum=r_min, maximum=r_max, value=r_max,
                                      step=0.001, label=f"上限 ({label_max})"))
    return min_updates, max_updates


def build_app():
    with gr.Blocks(title="TTHH M5优化器") as demo:
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# TTHH量化系统 · M5贝叶斯超参数优化器")
            btn_reconnect = gr.Button("🔄 重连当前会话", scale=1, min_width=120)
            btn_reset_p1_only = gr.Button("🗑️ 初始化P1", variant="stop", scale=1, min_width=100)
            btn_reset_p2_only = gr.Button("🗑️ 初始化P2", variant="stop", scale=1, min_width=100)
            btn_reset_all = gr.Button("🗑️ 全局初始化", variant="stop", scale=1, min_width=120)

        reconnect_tip = gr.Markdown("", visible=True)
        reset_confirm_state = gr.State({"target": None, "count": 0})
        reset_status = gr.Markdown("", visible=True)

        # ━━━ Tab1: Phase1 全局探索 ━━━
        with gr.Tab("Phase1 全局探索"):
            # 项目管理区
            with gr.Group():
                gr.Markdown("### 📁 项目管理")
                with gr.Row():
                    p1_project_path = gr.Textbox(
                        label="项目文件夹路径",
                        placeholder="E:/runs/my_run_01  （文件夹名即项目名）",
                        scale=5
                    )
                    btn_p1_open = gr.Button("📂 打开/新建", variant="primary", scale=1)

                p1_project_card = gr.HTML(
                    "<div style='border:2px dashed #ccc;border-radius:8px;"
                    "padding:16px;text-align:center;color:#999'>"
                    "⬜ 未加载项目 · 请输入文件夹路径后点击\"打开/新建\""
                    "</div>"
                )

                # 冲突确认行（默认隐藏）
                with gr.Row(visible=False) as conflict_row:
                    conflict_tip = gr.Markdown(
                        "⚠️ 该文件夹已有内容但不含项目配置，继续将在此创建项目文件"
                    )
                    btn_confirm_overwrite = gr.Button("✅ 确认创建", variant="primary")
                    btn_cancel_overwrite = gr.Button("取消")

                # 操作按钮行（有项目时显示）
                with gr.Row(visible=False):
                    btn_p1_reset = gr.Button("🗑️ 初始化P1", variant="stop")
                    p1_reset_tip = gr.Markdown("")

            project_state = gr.State(None)
            warm_start_state = gr.State(None)

            with gr.Row():
                btn_stable = gr.Button("稳健型")
                btn_aggressive = gr.Button("激进型")
                btn_defensive = gr.Button("防御型")
                btn_adaptive = gr.Button("全自适应型")

            with gr.Row():
                btn_preset_b = gr.Button("自适应型-b (scheme_b专用)")
                btn_preset_a = gr.Button("自适应型-a (scheme_a专用)")
                btn_preset_e = gr.Button("自适应型-e (scheme_e专用)")

            with gr.Row():
                btn_preset_v2 = gr.Button(
                    "⭐ 自适应型 2.0（P1 19项·四象限分层）",
                    variant="primary",
                )

            scheme_radio = gr.Radio(
                choices=[
                    ("方案D：仅行业OLS", "scheme_d"),
                    ("方案B：Rank-Z+双重OLS", "scheme_b"),
                    ("方案B1：非线性市值OLS（B+ln(M)²+ln(M)³）", "scheme_b1"),
                    ("方案B2：WLS加权（sqrt(mktcap)权重）", "scheme_b2"),
                    ("方案F：风格因子剥离（Barra风格）", "scheme_f"),
                    ("方案G：PCA隐式风险（60日SVD）", "scheme_g"),
                    ("方案A：双重OLS正交化", "scheme_a"),
                    ("方案E：分层中性化", "scheme_e"),
                ],
                value="scheme_d",
                label="中性化方案"
            )

            # ★ 归一化开关：开启后 P1 实时评分走 NORM_CONFIG 归一化路径
            # ★ v2.0 默认开启：避免不同指标量级差异导致 TPE 偏置
            enable_normalization_checkbox = gr.Checkbox(
                label="✅ 启用评分归一化（v2.0 默认开启）",
                value=True,
                info="开启后每个 Trial 的 score 走 NORM_CONFIG 归一化 + 取负，"
                     "Tab4 可切换双模态展示。",
            )

            history_warn = gr.Markdown("", elem_id="warn_box")  # noqa: F841

            with gr.Accordion("参数搜索范围（勾选=激活）", open=True):
                param_widgets = {}
                numeric_param_names = []
                categorical_param_names = []
                for name, pdef in ALL_PARAMS.items():
                    with gr.Row():
                        cb = gr.Checkbox(label=name, value=True, interactive=True)  # ★ 显式开启交互
                        if pdef["type"] == "categorical":
                            sel = gr.CheckboxGroup(
                                choices=pdef["choices"],
                                value=pdef["choices"],
                                label="可选值",
                                visible=True,
                                interactive=True,   # ★ 显式开启交互（Gradio 会按"是否作 input/output"推断，未连线的会变 False）
                            )
                            param_widgets[name] = {"checkbox": cb, "choices": sel}
                            categorical_param_names.append(name)
                        else:
                            low_box = gr.Number(
                                label="下限",
                                value=pdef["low"],
                                visible=True,
                                interactive=True
                            )
                            high_box = gr.Number(
                                label="上限",
                                value=pdef["high"],
                                visible=True,
                                interactive=True
                            )
                            param_widgets[name] = {
                                "checkbox": cb,
                                "low": low_box,
                                "high": high_box
                            }
                            numeric_param_names.append(name)

            with gr.Accordion("目标函数权重（合计应=1）", open=True):
                weight_sliders = {}
                for var in OBJECTIVE_VARS:
                    if var["enabled"] and not var["name"].startswith("stress_"):
                        label_text = var.get("label", var["name"])
                        weight_sliders[var["name"]] = gr.Slider(
                            0, 1, step=0.01,
                            label=label_text,
                            value=0
                        )
                weight_sum_md = gr.Markdown("权重合计：0.00")

            with gr.Row():
                n_trials_slider = gr.Slider(
                    10, 200, step=10, value=50, label="Trial数"
                )
            gr.Markdown(
                "<small>⚠️ P1固定使用全量窗口，"
                "单Trial约需4~6分钟，请合理设置Trial数和定时停止</small>"
            )

            # ★ 收集所有low/high数字框，用于读取用户设定的范围
            all_low_boxes = []
            all_high_boxes = []
            for name in numeric_param_names:  # 已有这个列表
                all_low_boxes.append(param_widgets[name]["low"])
                all_high_boxes.append(param_widgets[name]["high"])

            # ★ 收集所有可锁定/解锁的组件（必须在 weight_sliders 和 fast_mode_cb 定义之后）
            all_num_boxes = []
            all_checkboxes = []
            for name, widgets in param_widgets.items():
                all_checkboxes.append(widgets["checkbox"])
                if "low" in widgets:
                    all_num_boxes.append(widgets["low"])
                    all_num_boxes.append(widgets["high"])
                if "choices" in widgets:
                    all_checkboxes.append(widgets["choices"])

            all_lockable = (all_num_boxes +
                            [scheme_radio] +
                            list(weight_sliders.values()))

            def _unlock_updates():
                """返回所有可锁定组件的解锁update列表"""
                return [gr.update(interactive=True)] * len(all_lockable)

            def _lock_updates():
                """返回所有可锁定组件的锁定update列表"""
                return [gr.update(interactive=False)] * len(all_lockable)

            def _load_config_to_ui(project):
                """
                读取项目的p1_config.json，
                返回所有UI组件的update列表。
                如果读取失败返回全部gr.update()（不改变现有值）
                """
                n_weights = len(weight_sliders)
                n_numeric = len(numeric_param_names)
                n_categorical = len(categorical_param_names)
                n_lockable = len(all_lockable)

                empty = (
                    gr.update(value=True),                 # enable_normalization_checkbox（v2 默认开启）
                    gr.update(),                          # scheme_radio
                    *([gr.update()] * n_weights),         # weight_sliders
                    *([gr.update()] * n_numeric),         # low_boxes
                    *([gr.update()] * n_numeric),         # high_boxes
                    *([gr.update()] * n_categorical),     # ★ 新增: categorical choices
                    *([gr.update()] * n_lockable),        # lockable（锁定状态）
                )

                if project is None:
                    return empty

                try:
                    cfg = load_p1_config(project)["config"]
                except Exception:
                    return empty

                saved_weights = cfg.get("objective_weights", {})
                saved_ranges  = cfg.get("param_ranges", {})
                saved_scheme  = cfg.get("scheme", "scheme_b")
                # ★ v2.0 默认开启归一化
                saved_norm    = bool(cfg.get("enable_normalization", True))

                # 权重滑块
                weight_updates = [
                    gr.update(value=saved_weights.get(n, 0))
                    for n in weight_sliders.keys()
                ]

                # 参数范围数字框
                low_updates  = []
                high_updates = []
                for name in numeric_param_names:
                    pdef = ALL_PARAMS[name]
                    if name in saved_ranges:
                        r = saved_ranges[name]
                        low_updates.append(gr.update(value=r["low"]))
                        high_updates.append(gr.update(value=r["high"]))
                    else:
                        # 没有保存过的参数，用search_space默认范围
                        low_updates.append(gr.update(value=pdef["low"]))
                        high_updates.append(gr.update(value=pdef["high"]))

                # ★ 分类参数的可选值（按 categorical_param_names 顺序）
                cat_choices_updates = []
                for name in categorical_param_names:
                    pdef = ALL_PARAMS[name]
                    if name in saved_ranges and "choices" in saved_ranges[name]:
                        saved_choices = saved_ranges[name]["choices"]
                        # 兜底：若保存的空，回退全集
                        if saved_choices:
                            cat_choices_updates.append(gr.update(value=list(saved_choices)))
                        else:
                            cat_choices_updates.append(gr.update(value=list(pdef["choices"])))
                    else:
                        # 没有保存过，用 search_space 全集
                        cat_choices_updates.append(gr.update(value=list(pdef["choices"])))

                # 锁定状态：有有效Trial则锁定
                p1_study = get_p1_study(project)
                if p1_study:
                    stats = get_study_stats(p1_study)
                    if stats["complete"] > 0:
                        lock_updates = _lock_updates()
                    else:
                        lock_updates = _unlock_updates()
                else:
                    lock_updates = _unlock_updates()

                return (
                    gr.update(value=saved_norm),   # ★ enable_normalization_checkbox
                    gr.update(value=saved_scheme),
                    *weight_updates,
                    *low_updates,
                    *high_updates,
                    *cat_choices_updates,           # ★ 新增: 4 个 categorical choices
                    *lock_updates,
                )

            with gr.Row():
                btn_save_p1_config = gr.Button(
                    "💾 保存当前配置",
                    variant="secondary",
                    scale=1
                )
                save_p1_config_status = gr.Markdown("")

            # ★ v3.8 新增: GPU 加速开关 + 策略选择
            with gr.Row():
                gpu_p1_checkbox = gr.Checkbox(
                    label="🟢 启用 GPU 加速 (M2 评估)",
                    value=True,    # 默认开启, 与 config.yaml 一致
                    scale=1
                )
                # v4.1: 仅允许 D 策略 (A/B/C/E 已移除, 与纯 CPU bit-exact)
                gpu_strategy_p1 = gr.Dropdown(
                    choices=["D"],
                    value="D",
                    label="GPU 策略 (D=CPU训+GPU测 ★固定, v4.1)",
                    interactive=False,   # v4.1: 锁定 D
                    scale=2
                )
                gpu_strategy_desc_p1 = gr.Markdown(
                    "ℹ️  D 方案: LGBM CPU + XGB CPU 训 + GPU 测, 180窗 1.33min, "
                    "IC 与纯 CPU 差 1.5e-7 噪声级. v4.1 锁死.",
                    scale=5
                )

            with gr.Row():
                btn_start_p1 = gr.Button("▶ 开始Phase1", variant="primary", scale=3)
                btn_stop_now = gr.Button("⚡ 立即停止", variant="stop", scale=1)
                btn_stop_graceful = gr.Button("🏁 优雅停止", scale=1)
                btn_cleanup_p1 = gr.Button(      # ★ 新增
                    "🧹 清理异常Trial",
                    scale=1, min_width=120
                )
                dry_run_p1 = gr.Checkbox(       # ★ 干运行开关
                    label="仅审计（不删除）",
                    value=True,
                    scale=0, min_width=140,
                )

            with gr.Row():
                enable_timer_p1 = gr.Checkbox(
                    label="⏰ 定时自动停止",
                    value=False,
                    scale=1
                )
                timer_hours_p1 = gr.Number(
                    label="小时后自动停止",
                    value=2.0,
                    minimum=0.1,
                    maximum=24.0,
                    step=0.5,
                    visible=False,
                    scale=2
                )
                with gr.Column(scale=3):
                    timer_status_p1 = gr.Markdown("")
                btn_check_timer = gr.Button("🔍 检查定时器", scale=0, min_width=80)

            log_box_p1 = gr.Textbox(label="运行日志", lines=8, autoscroll=True)
            with gr.Row():
                btn_refresh_log_p1 = gr.Button("🔄 刷新运行日志", scale=0, min_width=120)
            with gr.Row():
                cur_trial_num = gr.Number(label="当前Trial", value=0)  # noqa: F841
                best_score_num = gr.Number(label="最优得分", value=0)  # noqa: F841

        # ━━━ Tab2: 结果分析与反推 ━━━
        with gr.Tab("结果分析与反推"):
            with gr.Group():
                gr.Markdown("### 📂 加载项目")
                with gr.Row():
                    t2_project_path = gr.Textbox(
                        label="项目文件夹路径",
                        placeholder="E:/runs/my_run_01",
                        scale=5
                    )
                    btn_t2_load = gr.Button("📂 加载", variant="primary", scale=1)
                t2_project_status = gr.Markdown("未加载任何项目")

            t2_project_state = gr.State(None)

            gr.Markdown("基于Phase1结果，按因变量过滤Trial，反推参数搜索范围。")

            with gr.Row():
                trial_count_display = gr.Markdown("📊 满足条件：- / - 个有效Trial")
                btn_refresh_count = gr.Button("🔄 刷新统计", scale=0, min_width=100)
                btn_reset_sliders = gr.Button(
                    "↕️ 初始化滑块",
                    scale=0, min_width=110,
                    variant="secondary"
                )

            filter_sliders = {}
            filter_slider_names = []
            for var in OBJECTIVE_VARS:
                if var["source"] == "m2" and var["enabled"] and not var["name"].startswith("stress_"):
                    filter_slider_names.append(var["name"])
                    label_text = var.get("label", var["name"])
                    sl_min_val = var.get("slider_min", -2)
                    sl_max_val = var.get("slider_max", 2)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=150):
                            gr.Markdown(f"**{label_text}**")
                        sl_min = gr.Slider(
                            minimum=sl_min_val, maximum=sl_max_val, step=0.001,
                            value=sl_min_val,
                            label="下限",
                            scale=2
                        )
                        sl_max = gr.Slider(
                            minimum=sl_min_val, maximum=sl_max_val, step=0.001,
                            value=sl_max_val,
                            label="上限",
                            scale=2
                        )
                    filter_sliders[var["name"]] = {"min": sl_min, "max": sl_max}

            matched_md = gr.Markdown("")  # noqa: F841

            with gr.Row():
                top_pct = gr.Slider(1, 100, value=100, step=1, label="筛选后仅保留分数前百分之多少（100=全部保留）")

            with gr.Row():
                btn_analyze = gr.Button("📊 分析参数范围")
                btn_export_p2 = gr.Button("📤 导出为P2配置", variant="primary")

            # ★ Top-5 排名（点击"分析参数范围"后渲染，复用 P1 全部有效 Trial）
            with gr.Group():
                gr.Markdown("### 🏆 Top-5 Trial 排名（归一化评分）")
                t2_top5_html = gr.HTML(
                    "<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                    "border:1px solid var(--border-color-primary,#e0e0e0);"
                    "border-left-width:4px;border-left-color:#9e9e9e;opacity:0.7;"
                    "font-size:14px'>⬜ 点击「分析参数范围」后显示排名</div>"
                )

            range_df = gr.DataFrame(
                label="反推参数范围（整数已取整，可直接导出为P2配置）"
            )
            range_result_state = gr.State(None)
            status_box = gr.Textbox(label="状态")

        # ━━━ Tab3: Phase2 独立精调 ━━━
        with gr.Tab("Phase2 独立精调"):
            with gr.Group():
                gr.Markdown("### 📁 项目管理")
                with gr.Row():
                    t3_project_path = gr.Textbox(
                        label="项目文件夹路径",
                        placeholder="E:/runs/my_run_01",
                        scale=5
                    )
                    btn_t3_load = gr.Button("📂 加载", variant="primary", scale=1)

                t3_project_card = gr.HTML(
                    "<div style='border:2px dashed #ccc;border-radius:8px;"
                    "padding:16px;text-align:center;color:#999'>"
                    "⬜ 未加载项目"
                    "</div>"
                )

                with gr.Row(visible=False) as p2_action_row:
                    btn_p2_reset = gr.Button("🗑️ 初始化P2", variant="stop")
                    p2_reset_tip = gr.Markdown("")

            t3_project_state = gr.State(None)

            p2_scheme_radio = gr.Radio(
                choices=[
                    ("方案D：仅行业OLS（IR最优）", "scheme_d"),
                    ("方案B：Rank-Z+双重OLS", "scheme_b"),
                    ("方案B1：非线性市值OLS（B+ln(M)²+ln(M)³）", "scheme_b1"),
                    ("方案B2：WLS加权（sqrt(mktcap)权重）", "scheme_b2"),
                    ("方案F：风格因子剥离（Barra风格）", "scheme_f"),
                    ("方案G：PCA隐式风险（60日SVD）", "scheme_g"),
                    ("方案A：双重OLS正交化", "scheme_a"),
                    ("方案E：分层中性化", "scheme_e"),
                ],
                value="scheme_d",
                label="中性化方案（默认与P1一致）"
            )

            gr.Markdown("建议先完成Tab2并导出P2配置，再执行Phase2精调。")

            with gr.Accordion("Phase2搜索范围", open=True):
                gr.Markdown("**分类参数设置**（单选=固定，全选=搜索）")
                p2_cat_widgets = {}
                CATEGORICAL_PARAMS = [
                    "lgbm_lr_mode",
                    "lgbm_depth_mode",
                    "xgb_lr_mode",
                    "drop_short_term_noise",
                ]
                with gr.Row():
                    for name in CATEGORICAL_PARAMS:
                        pdef = ALL_PARAMS[name]
                        cg = gr.CheckboxGroup(
                            choices=pdef["choices"],
                            value=pdef["choices"],
                            label=name,
                            interactive=True,
                        )
                        p2_cat_widgets[name] = cg

                gr.Markdown("---")
                gr.Markdown("**数值参数范围**（勾选=搜索范围，取消=固定默认值）")
                p2_param_widgets = {}
                for name, pdef in ALL_PARAMS.items():
                    if pdef["type"] != "categorical":
                        with gr.Row():
                            cb2 = gr.Checkbox(
                                value=True,
                                label="",
                                scale=1,
                                min_width=40,
                                interactive=True,
                            )
                            with gr.Column(scale=2, min_width=80):
                                gr.Markdown(f"**{name}**")
                            lo2 = gr.Number(
                                label="下限",
                                value=pdef["low"],
                                interactive=True,
                                scale=3,
                                visible=True,
                            )
                            hi2 = gr.Number(
                                label="上限",
                                value=pdef["high"],
                                interactive=True,
                                scale=3,
                                visible=True,
                            )
                            dv2 = gr.Number(
                                label="固定值",
                                value=pdef["default"],
                                interactive=True,
                                scale=3,
                                visible=False,
                            )
                        p2_param_widgets[name] = {
                            "checkbox":    cb2,
                            "low":         lo2,
                            "high":        hi2,
                            "default_val": dv2,
                        }

                def _make_toggle_fn():
                    def _toggle(checked):
                        return (
                            gr.update(visible=checked),
                            gr.update(visible=checked),
                            gr.update(visible=not checked),
                        )
                    return _toggle

                for _name, _widgets in p2_param_widgets.items():
                    _widgets["checkbox"].change(
                        fn=_make_toggle_fn(),
                        inputs=[_widgets["checkbox"]],
                        outputs=[_widgets["low"], _widgets["high"], _widgets["default_val"]],
                    )

            with gr.Accordion("Phase2目标权重", open=False):
                p2_weight_sliders = {}
                for var in OBJECTIVE_VARS:
                    if var["enabled"] and not var["name"].startswith("stress_"):
                        label_text = var.get("label", var["name"])
                        p2_weight_sliders[var["name"]] = gr.Slider(
                            0, 1, step=0.01,
                            label=label_text,
                            value=0
                        )

            with gr.Row():
                p2_trials = gr.Slider(5, 300, step=5, value=60, label="Trial数")

            p2_from_best = gr.Checkbox(label="以Phase1最优参数为起点", value=True)

            # ★ v3.8 新增: P2 GPU 加速开关 + 策略选择
            with gr.Row():
                gpu_p2_checkbox = gr.Checkbox(
                    label="🟢 启用 GPU 加速 (M2 评估)",
                    value=True,    # 默认开启
                    scale=1
                )
                # v4.1: 仅允许 D 策略
                gpu_strategy_p2 = gr.Dropdown(
                    choices=["D"],
                    value="D",
                    label="GPU 策略 (D=CPU训+GPU测 ★固定, v4.1)",
                    interactive=False,
                    scale=2
                )
                gpu_strategy_desc_p2 = gr.Markdown(
                    "ℹ️  D 方案 180窗 1.33min, IC 与纯 CPU 差 1.5e-7 噪声级. v4.1 锁死.",
                    scale=5
                )

            with gr.Row():
                btn_save_p2_config = gr.Button("💾 保存P2配置", variant="secondary", scale=1)
                save_p2_config_status = gr.Markdown("")

            with gr.Row():
                btn_start_p2 = gr.Button("▶ 开始Phase2", variant="primary", scale=3)
                btn_stop_p2_now = gr.Button("⚡ 立即停止", variant="stop", scale=1)
                btn_stop_p2_graceful = gr.Button("🏁 优雅停止", scale=1)
                btn_cleanup_p2 = gr.Button(      # ★ 新增
                    "🧹 清理异常Trial",
                    scale=1, min_width=120
                )
                dry_run_p2 = gr.Checkbox(       # ★ 干运行开关
                    label="仅审计（不删除）",
                    value=True,
                    scale=0, min_width=140,
                )

            with gr.Row():
                enable_timer_p2 = gr.Checkbox(
                    label="⏰ 定时自动停止",
                    value=False,
                    scale=1
                )
                timer_hours_p2 = gr.Number(
                    label="小时后自动停止",
                    value=2.0,
                    minimum=0.1,
                    maximum=24.0,
                    step=0.5,
                    visible=False,
                    scale=2
                )
                with gr.Column(scale=3):
                    timer_status_p2 = gr.Markdown("")

            log_box_p2 = gr.Textbox(label="运行日志", lines=8, autoscroll=True)
            with gr.Row():
                btn_refresh_log_p2 = gr.Button("🔄 刷新运行日志", scale=0, min_width=120)
            with gr.Row():
                p2_trial_num = gr.Number(label="当前Trial", value=0)  # noqa: F841
                p2_best_score = gr.Number(label="最优得分", value=0)  # noqa: F841

            # ★ Tab3 可锁定组件（P2运行中/DB存在时禁止修改）
            p2_all_lockable = (
                [btn_save_p2_config, p2_scheme_radio]
                + list(p2_cat_widgets.values())
                + [w["checkbox"]    for w in p2_param_widgets.values()]
                + [w["low"]         for w in p2_param_widgets.values()]
                + [w["high"]        for w in p2_param_widgets.values()]
                + [w["default_val"] for w in p2_param_widgets.values()]
                + list(p2_weight_sliders.values())
            )

            def _p2_lock_updates():
                """P2 DB存在时锁定所有参数控件"""
                return [gr.update(interactive=False)] * len(p2_all_lockable)

            def _p2_unlock_updates():
                """P2初始化后解锁所有参数控件"""
                return [gr.update(interactive=True)] * len(p2_all_lockable)

        # ━━━ Tab4: 结果与部署 ━━━
        t4_tab = gr.Tab("结果与部署")
        with t4_tab:
            # 项目管理区
            with gr.Group():
                gr.Markdown("### 📁 P2结果分析")
                with gr.Row():
                    t4_project_path = gr.Textbox(
                        label="项目文件夹路径",
                        placeholder="E:/runs/my_run_01",
                        scale=5,
                    )
                    btn_t4_load = gr.Button("📂 加载", variant="primary", scale=1)
                t4_project_card = gr.HTML(
                    "<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                    "border:1px solid var(--border-color-primary,#e0e0e0);"
                    "border-left-width:4px;border-left-color:#9e9e9e;opacity:0.7;"
                    "font-size:14px'>"
                    "⬜ 未加载项目 · 请输入文件夹路径后点击\"加载\"</div>"
                )

            t4_project_state = gr.State(None)

            # ★ 会话级安全：每个浏览器/标签页独立的日志队列
            t4_session_state = gr.State({
                "queue": None,         # 懒加载：首次回测时创建 queue.Queue()
                "running": False,      # 当前会话是否在跑
                "log_buffer": [],      # 已派发的日志（防止 timer 漏读）
            })

            db_status = gr.Markdown("数据源：未加载")

            # ── Top-N Trial 排名 ──
            with gr.Group():
                gr.Markdown("### 🏆 Top-N Trial 排名")
                with gr.Row():
                    t4_top_n = gr.Slider(
                        minimum=2, maximum=20, value=5, step=1,
                        label="展示最优的N个Trial",
                        scale=3,
                    )
                    t4_score_mode = gr.Radio(
                        choices=["原始评分 (Raw)", "归一化评分 (Normalized)"],
                        value="原始评分 (Raw)",
                        label="🏆 排序与分析依据",
                        scale=2,
                    )
                    btn_t4_refresh = gr.Button("🔄 刷新排名", variant="secondary", scale=1)

                t4_score_mode_tip = gr.Markdown(
                    "💡 当前为**原始评分**模式。归一化数据可使用 `retroactive_normalize.py` 追溯生成。",
                    visible=True,
                )

                t4_ranking_html = gr.HTML(
                    "<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                    "border:1px solid var(--border-color-primary,#e0e0e0);"
                    "border-left-width:4px;border-left-color:#9e9e9e;opacity:0.7;"
                    "font-size:14px'>"
                    "⬜ 加载项目后显示排名</div>"
                )

            # ── 参数收敛分析 ──
            with gr.Group():
                gr.Markdown("### 📊 参数收敛分析")
                t4_convergence_html = gr.HTML(
                    "<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                    "border:1px solid var(--border-color-primary,#e0e0e0);"
                    "border-left-width:4px;border-left-color:#9e9e9e;opacity:0.7;"
                    "font-size:14px'>"
                    "⬜ 加载项目后显示收敛分析</div>"
                )

            # ── 选择Trial部署 ──
            with gr.Group():
                gr.Markdown("### 🚀 部署所选Trial")
                with gr.Row():
                    t4_trial_selector = gr.Dropdown(
                        label="选择Trial编号",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    btn_t4_load_trial = gr.Button("📋 读取参数", variant="secondary", scale=1)
                t4_selected_params_df = gr.DataFrame(label="所选Trial参数")
                t4_selected_metrics = gr.Markdown("")

                with gr.Row():
                    btn_write_config = gr.Button("💾 写回config.yaml", variant="primary")  # noqa: F841
                    btn_run_full = gr.Button("🔄 触发M2+M4完整重跑", variant="primary")  # noqa: F841

            log_box_deploy = gr.Textbox(label="执行日志", lines=10, autoscroll=True)  # noqa: F841
            report_path_box = gr.Textbox(label="报告路径", interactive=False)  # noqa: F841
            btn_open_report = gr.Button("🌐 打开回测报告")  # noqa: F841

        # ★ Tab4 选中时自动加载（若已有项目路径且未加载）
        def on_tab4_select(project, path):
            """切到Tab4时自动触发加载（仅在 project 为空且 path 有效时）"""
            if project is None and path and path.strip() and os.path.exists(path.strip()):
                # 返回 None 表示无需更新，由 click 事件继续处理
                return gr.update(value=path.strip())
            return gr.update()

        try:
            t4_tab.select(
                fn=on_tab4_select,
                inputs=[t4_project_state, t4_project_path],
                outputs=[t4_project_path],
            )
        except (AttributeError, TypeError):
            pass  # Gradio版本不支持select事件

        # ━━━ 事件绑定 ━━━

        # 重连当前会话
        # 收集所有low_box和high_box（供on_reconnect回填用）
        all_low_boxes = [param_widgets[n]["low"] for n in numeric_param_names]
        all_high_boxes = [param_widgets[n]["high"] for n in numeric_param_names]
        # ★ 收集所有 categorical choices CheckboxGroup（供 load_config_to_ui 回填用）
        all_cat_choices = [param_widgets[n]["choices"] for n in categorical_param_names]

        def on_reconnect(p1_path, t2_path, t3_path):
            tip_parts = []
            project_update = None
            card_update = _card("unconfigured")
            ui_updates = _load_config_to_ui(None)  # 默认空

            if p1_path and p1_path.strip() and os.path.exists(p1_path.strip()):
                try:
                    result = init_project(p1_path.strip())
                    if result["status"] in ("loaded", "created"):
                        project_update = result["project"]

                        p1_study = get_p1_study(project_update)
                        if p1_study:
                            stats = get_study_stats(p1_study)
                            card_update = _card(
                                "in_progress", project=project_update, stats=stats
                            )
                        else:
                            card_update = _card("fresh", project=project_update)

                        # ★ 复用_load_config_to_ui
                        ui_updates = _load_config_to_ui(project_update)
                        tip_parts.append("Tab1✅")
                    else:
                        tip_parts.append("Tab1⚠️")
                except Exception:
                    tip_parts.append("Tab1❌")
            else:
                tip_parts.append("Tab1—")

            for path, label in [(t2_path,"Tab2"),(t3_path,"Tab3")]:
                tip_parts.append(
                    f"{label}✅" if (path and os.path.exists(path.strip()))
                    else f"{label}—"
                )

            return (
                f"重连完成：{' | '.join(tip_parts)}",
                project_update,
                card_update,
                *ui_updates,
            )

        btn_reconnect.click(
            fn=on_reconnect,
            inputs=[p1_project_path, t2_project_path, t3_project_path],
            outputs=[
                reconnect_tip,
                project_state,
                p1_project_card,
                enable_normalization_checkbox,  # ★ v2.0
                scheme_radio,
                *list(weight_sliders.values()),
                *all_low_boxes,
                *all_high_boxes,
                *all_cat_choices,            # ★ 新增: 4 个 categorical choices
                *all_lockable,
            ]
        )

        # 全局初始化（三按钮：P1/P2/全部）
        def on_reset_click(target: str, confirm_state: dict, project):
            if project is None:
                return (confirm_state, "⚠️ 请先加载项目",
                        *([gr.update()] * len(all_lockable)))
            prev = confirm_state
            if prev.get("target") != target or prev.get("count", 0) == 0:
                # 第一次点击：进入待确认状态
                label = 'P1' if target == 'p1' else 'P2' if target == 'p2' else 'P1+P2全部'
                return ({"target": target, "count": 1},
                        f"⚠️ 再次点击确认{label}数据清除",
                        *([gr.update()] * len(all_lockable)))
            else:
                # 第二次点击：执行
                if target == "p1":
                    reset_p1(project)
                    msg = "✅ P1数据库已清除，参数配置已解锁"
                elif target == "p2":
                    reset_p2(project)
                    msg = "✅ P2数据库已清除，参数配置已解锁"
                else:
                    reset_all(project)
                    msg = "✅ P1+P2数据库全部清除，参数配置已解锁"
                return ({"target": None, "count": 0},
                        msg,
                        *_unlock_updates())

        btn_reset_p1_only.click(
            fn=lambda s, p: on_reset_click("p1", s, p),
            inputs=[reset_confirm_state, project_state],
            outputs=[reset_confirm_state, reset_status, *all_lockable]
        )
        btn_reset_p2_only.click(
            fn=lambda s, p: on_reset_click("p2", s, p),
            inputs=[reset_confirm_state, project_state],
            outputs=[reset_confirm_state, reset_status, *all_lockable]
        )
        btn_reset_all.click(
            fn=lambda s, p: on_reset_click("all", s, p),
            inputs=[reset_confirm_state, project_state],
            outputs=[reset_confirm_state, reset_status, *all_lockable]
        )

        # 权重合计
        def update_weight_sum(*weights):
            total = sum(w for w in weights if w is not None)
            color = "red" if abs(total - 1.0) > 0.01 else "green"
            return gr.Markdown(
                f'<span style="color:{color}">权重合计：{total:.2f}</span>'
            )

        weight_inputs = list(weight_sliders.values())
        for slider in weight_inputs:
            slider.change(
                fn=update_weight_sum,
                inputs=weight_inputs,
                outputs=weight_sum_md
            )

        # 预设模板
        def apply_preset(preset_name):
            preset = PRESET_TEMPLATES.get(preset_name, {})
            weights = preset.get("weights", {})
            active = preset.get("active", [])

            weight_updates = {}
            for name in weight_sliders:
                weight_updates[name] = gr.update(value=weights.get(name, 0))

            param_updates = {}
            for name in param_widgets:
                if active == "ALL_29":
                    is_active = True
                else:
                    is_active = name in active
                param_updates[name] = gr.update(value=is_active)

            return list(weight_updates.values()) + list(param_updates.values())

        btn_stable.click(
            fn=lambda: apply_preset("稳健型"),
            outputs=list(weight_sliders.values()) + [
                w["checkbox"] for w in param_widgets.values()
            ]
        )
        btn_aggressive.click(
            fn=lambda: apply_preset("激进型"),
            outputs=list(weight_sliders.values()) + [
                w["checkbox"] for w in param_widgets.values()
            ]
        )
        btn_defensive.click(
            fn=lambda: apply_preset("防御型"),
            outputs=list(weight_sliders.values()) + [
                w["checkbox"] for w in param_widgets.values()
            ]
        )
        btn_adaptive.click(
            fn=lambda: apply_preset("全自适应型"),
            outputs=list(weight_sliders.values()) + [
                w["checkbox"] for w in param_widgets.values()
            ]
        )

        # ── 专用预设按钮（含范围覆盖 + 热启动） ──
        def _apply_named_preset(preset_name: str):
            p = PRESET_TEMPLATES[preset_name]
            weights    = p.get("weights", {})
            p_ranges   = p.get("param_ranges", {})
            fixed_cat  = p.get("fixed_categorical", {})
            fixed_ints = set(p.get("fixed_int", []))
            warm       = p.get("warm_start", None)

            updates = []

            # 1. 权重滑块
            for var_name in weight_sliders:
                updates.append(gr.update(value=weights.get(var_name, 0)))

            # 2. numeric low boxes
            for name in numeric_param_names:
                pdef = ALL_PARAMS[name]
                if name in p_ranges:
                    updates.append(gr.update(value=p_ranges[name]["low"]))
                else:
                    updates.append(gr.update(value=pdef["low"]))

            # 3. numeric high boxes
            for name in numeric_param_names:
                pdef = ALL_PARAMS[name]
                if name in p_ranges:
                    updates.append(gr.update(value=p_ranges[name]["high"]))
                else:
                    updates.append(gr.update(value=pdef["high"]))

            # 4. categorical choices
            for name in categorical_param_names:
                if name in fixed_cat:
                    updates.append(gr.update(value=fixed_cat[name]))
                else:
                    updates.append(gr.update(value=ALL_PARAMS[name]["choices"]))

            # 5. checkboxes（param_widgets 迭代顺序）
            for name in param_widgets:
                if name in fixed_ints:
                    updates.append(gr.update(value=False))
                else:
                    updates.append(gr.update(value=True))

            # 6. warm_start_state
            updates.append(warm)

            return updates

        _preset_outputs = (
            list(weight_sliders.values()) +
            [param_widgets[n]["low"]  for n in numeric_param_names] +
            [param_widgets[n]["high"] for n in numeric_param_names] +
            [param_widgets[n]["choices"] for n in categorical_param_names] +
            [param_widgets[n]["checkbox"] for n in param_widgets] +
            [warm_start_state]
        )

        btn_preset_b.click(
            fn=lambda: _apply_named_preset("自适应型-b"),
            outputs=_preset_outputs
        )
        btn_preset_a.click(
            fn=lambda: _apply_named_preset("自适应型-a"),
            outputs=_preset_outputs
        )
        btn_preset_e.click(
            fn=lambda: _apply_named_preset("自适应型-e"),
            outputs=_preset_outputs
        )

        # ★ P1 自适应型 2.0：19 项因变量 + 四象限分层
        btn_preset_v2.click(
            fn=lambda: _apply_named_preset("自适应型 2.0"),
            outputs=_preset_outputs
        )

        # ━━━ Tab1: 项目管理事件 ━━━

        def on_open_project(path):
            if not path or not path.strip():
                return (
                    None,
                    _card("unconfigured"),
                    gr.update(visible=False),
                    "",
                    gr.update(value=""),
                    gr.update(value=""),
                    # UI组件全部不变
                    *_load_config_to_ui(None),
                )

            result = init_project(path.strip())
            project = result["project"]

            if result["status"] == "conflict":
                return (
                    None,
                    _card("conflict", message=result["message"]),
                    gr.update(visible=True),
                    result["message"],
                    gr.update(value=""),
                    gr.update(value=""),
                    *_load_config_to_ui(None),
                )

            # ★ 渲染卡片
            p1_study = get_p1_study(project)
            if p1_study is None:
                card_html = _card("fresh", project=project)
            else:
                stats = get_study_stats(p1_study)
                if stats["complete"] == 0 and stats["fail"] == 0:
                    card_html = _card("fresh", project=project)
                elif stats["complete"] > 0:
                    card_html = _card("in_progress", project=project, stats=stats)
                else:
                    card_html = _card("fresh", project=project)

            # ★ 读config回填UI
            ui_updates = _load_config_to_ui(project)

            return (
                project,
                card_html,
                gr.update(visible=False),
                "",
                gr.update(value=path.strip()),  # 同步到Tab2路径框
                gr.update(value=path.strip()),  # 同步到Tab4路径框
                *ui_updates,
            )

        btn_p1_open.click(
            fn=on_open_project,
            inputs=[p1_project_path],
            outputs=[
                project_state,
                p1_project_card,
                conflict_row,
                conflict_tip,
                t2_project_path,
                t4_project_path,
                # ★ 以下全部新增
                enable_normalization_checkbox,  # ★ v2.0 归一化开关
                scheme_radio,
                *list(weight_sliders.values()),
                *all_low_boxes,
                *all_high_boxes,
                *all_cat_choices,            # ★ 新增: 4 个 categorical choices
                *all_lockable,
            ]
        )

        # 冲突确认
        def on_confirm_overwrite(path):
            if not path or not path.strip():
                return (
                    None,
                    _card("unconfigured"),
                    gr.update(visible=False),
                    gr.update(value=""),
                )
            result = force_init_project(path.strip())
            project = result["project"]
            card_html = _card("fresh", project=project)
            return (
                project,
                card_html,
                gr.update(visible=False),
                gr.update(value=path.strip()),
            )

        btn_confirm_overwrite.click(
            fn=on_confirm_overwrite,
            inputs=[p1_project_path],
            outputs=[project_state, p1_project_card, conflict_row, t2_project_path]
        )

        def on_cancel_overwrite():
            return (
                None,
                _card("unconfigured"),
                gr.update(visible=False),
                "",
            )

        btn_cancel_overwrite.click(
            fn=on_cancel_overwrite,
            outputs=[project_state, p1_project_card, conflict_row, conflict_tip]
        )

        # P1重置
        def on_reset_p1(project):
            if project is None:
                return project, _card("unconfigured"), "❌ 请先加载项目"
            reset_p1(project)
            card_html = _card("fresh", project=project)
            return project, card_html, "✅ P1数据库已删除，配置保留"

        btn_p1_reset.click(
            fn=on_reset_p1,
            inputs=[project_state],
            outputs=[project_state, p1_project_card, p1_reset_tip]
        )

        # 保存P1配置
        def on_save_p1_config(project, scheme, enable_norm, *all_values):
            if project is None:
                return "❌ 请先加载项目"
            if os.path.exists(project.get("p1_db_path", "")):
                return "🔒 DB已存在，不允许修改配置（请先初始化P1）"

            num_weights = len(weight_sliders)
            num_checkboxes = len(param_widgets)
            num_numeric = len(numeric_param_names)
            num_categorical = len(categorical_param_names)

            weight_values = all_values[:num_weights]
            checkbox_values = all_values[num_weights:num_weights+num_checkboxes]
            # ★ 新增：分类参数的可选值（按 categorical_param_names 顺序）
            cat_choices_values = all_values[num_weights+num_checkboxes:
                                            num_weights+num_checkboxes+num_categorical]
            low_values = all_values[num_weights+num_checkboxes+num_categorical:
                                    num_weights+num_checkboxes+num_categorical+num_numeric]
            high_values = all_values[num_weights+num_checkboxes+num_categorical+num_numeric:]

            # 收集权重
            objective_weights = {}
            weight_names = list(weight_sliders.keys())
            for i, name in enumerate(weight_names):
                if weight_values[i] and weight_values[i] > 0:
                    objective_weights[name] = float(weight_values[i])

            # 收集激活参数
            active_params = []
            param_names = list(param_widgets.keys())
            for i, name in enumerate(param_names):
                if checkbox_values[i]:
                    active_params.append(name)

            # ★ 收集参数范围（low/high数字框的值）
            param_ranges = {}
            for i, name in enumerate(numeric_param_names):
                pdef = ALL_PARAMS[name]
                low_val = float(low_values[i])
                high_val = float(high_values[i])

                if low_val >= high_val:
                    return f"❌ 参数 {name} 下限({low_val})≥上限({high_val})"

                original_default = pdef["default"]

                # ★ 自动修正默认值
                if isinstance(original_default, (int, float)):
                    if original_default < low_val or original_default > high_val:
                        if pdef["type"] == "int":
                            adjusted = int(round((low_val + high_val) / 2))
                        else:
                            adjusted = round((low_val + high_val) / 2, 6)
                    else:
                        adjusted = original_default
                else:
                    adjusted = original_default

                param_ranges[name] = {
                    "low": low_val,
                    "high": high_val,
                    "type": pdef["type"],
                    "original_default": original_default,
                    "adjusted_default": adjusted,  # ★ 新增字段
                }

            # ★ 把 4 个分类参数也写入 param_ranges（避免被误判为"锁死"）
            # 注：UI 现在会把 sub-choices 传给 on_save_p1_config，按用户勾选保存。
            for i, name in enumerate(categorical_param_names):
                pdef = ALL_PARAMS[name]
                # ★ 修复: 用用户在 UI 上勾选的可选值 (而非 pdef 全集) 持久化
                user_choices = cat_choices_values[i] if cat_choices_values[i] else list(pdef["choices"])
                # ★ 兜底: 若用户全取消，自动回退到全集 (避免 sampler 抛错)
                if not user_choices:
                    user_choices = list(pdef["choices"])
                param_ranges[name] = {
                    "type": pdef["type"],
                    "choices": list(user_choices),
                    "original_default": pdef["default"],
                    "adjusted_default": pdef["default"],
                }

            try:
                config_data = load_p1_config(project)
                config_data["config"]["scheme"] = scheme
                config_data["config"]["fast_mode"] = False        # ★ 固定False
                config_data["config"]["window_count"] = calc_window_count()
                config_data["config"]["objective_weights"] = objective_weights
                config_data["config"]["active_params"] = active_params if active_params else None
                config_data["config"]["param_ranges"] = param_ranges
                # ★ v2.0：归一化开关写入 config（v2 默认开启）
                config_data["config"]["enable_normalization"] = bool(enable_norm)

                with open(project["p1_config_path"], "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)

                # ★ 提示哪些默认值被修正
                adjusted_tips = []
                for name, r in param_ranges.items():
                    if r["adjusted_default"] != r["original_default"]:
                        adjusted_tips.append(
                            f"{name}：默认值 {r['original_default']} → {r['adjusted_default']}"
                        )
                tip_str = ""
                if adjusted_tips:
                    tip_str = "\n⚠️ 以下参数默认值已自动修正：\n" + "\n".join(adjusted_tips)

                return f"✅ 配置已保存（{len(param_ranges)}个参数范围，{len(objective_weights)}个权重）{tip_str}"
            except Exception as e:
                return f"❌ 保存失败：{e}"

        btn_save_p1_config.click(
            fn=on_save_p1_config,
            inputs=(
                [project_state, scheme_radio, enable_normalization_checkbox]
                + list(weight_sliders.values())
                + [w["checkbox"] for w in param_widgets.values()]
                + [w["choices"] for w in param_widgets.values() if "choices" in w]   # ★ 新增
                + all_low_boxes    # ★ 新增
                + all_high_boxes   # ★ 新增
            ),
            outputs=[save_p1_config_status]
        )

        # ★ 自动保存配置（权重或参数变化时自动保存到config，DB不存在时才允许）
        def auto_save_config(project, scheme, enable_norm, *all_values):
            """权重或参数变化时自动保存到config（DB不存在时才允许）"""
            if project is None:
                return  # 未加载项目，不保存
            if os.path.exists(project.get("p1_db_path", "")):
                return  # DB已存在，禁止修改

            try:
                num_weights = len(weight_sliders)
                num_checkboxes = len(param_widgets)
                num_categorical = len(categorical_param_names)

                weight_values = all_values[:num_weights]
                checkbox_values = all_values[num_weights:num_weights+num_checkboxes]
                cat_choices_values = all_values[num_weights+num_checkboxes:
                                                num_weights+num_checkboxes+num_categorical]

                objective_weights = {}
                for i, name in enumerate(list(weight_sliders.keys())):
                    if weight_values[i] and weight_values[i] > 0:
                        objective_weights[name] = float(weight_values[i])

                active_params = []
                for i, name in enumerate(list(param_widgets.keys())):
                    if checkbox_values[i]:
                        active_params.append(name)

                config_data = load_p1_config(project)
                config_data["config"]["scheme"] = scheme
                config_data["config"]["fast_mode"] = False        # ★ 固定False
                config_data["config"]["window_count"] = calc_window_count()
                config_data["config"]["objective_weights"] = objective_weights
                config_data["config"]["active_params"] = active_params if active_params else None
                # ★ v2.0：归一化开关同步到 config
                config_data["config"]["enable_normalization"] = bool(enable_norm)

                # ★ 同步分类参数的可选值（按 categorical_param_names 顺序）
                saved_ranges = config_data["config"].get("param_ranges", {})
                for i, name in enumerate(categorical_param_names):
                    pdef = ALL_PARAMS[name]
                    user_choices = cat_choices_values[i] if cat_choices_values[i] else list(pdef["choices"])
                    if not user_choices:
                        user_choices = list(pdef["choices"])
                    # ★ 修复: 若分类参数尚未在 param_ranges 中, 主动添加(避免"新项目只自动保存时不写入"导致的丢失)
                    if name in saved_ranges and isinstance(saved_ranges[name], dict):
                        saved_ranges[name]["choices"] = list(user_choices)
                    else:
                        saved_ranges[name] = {
                            "type": pdef["type"],
                            "choices": list(user_choices),
                            "original_default": pdef["default"],
                            "adjusted_default": pdef["default"],
                        }
                config_data["config"]["param_ranges"] = saved_ranges

                # 直接写文件，绕过save_p1_config的DB检查
                with open(project["p1_config_path"], "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass  # 自动保存失败不打扰用户

        # 绑定自动保存到所有权重滑块和勾选框的change事件
        auto_save_inputs = (
            [project_state, scheme_radio, enable_normalization_checkbox]
            + list(weight_sliders.values())
            + [w["checkbox"] for w in param_widgets.values()]
            + [w["choices"] for w in param_widgets.values() if "choices" in w]   # ★ 新增
        )

        for slider in list(weight_sliders.values()):
            slider.change(
                fn=auto_save_config,
                inputs=auto_save_inputs,
                outputs=[]
            )

        for name, widgets in param_widgets.items():
            widgets["checkbox"].change(
                fn=auto_save_config,
                inputs=auto_save_inputs,
                outputs=[]
            )
            # ★ 新增: 分类参数的 choices 变化时也自动保存
            if "choices" in widgets:
                widgets["choices"].change(
                    fn=auto_save_config,
                    inputs=auto_save_inputs,
                    outputs=[]
                )

        scheme_radio.change(
            fn=auto_save_config,
            inputs=auto_save_inputs,
            outputs=[]
        )

        # ★ 归一化开关变化时也自动保存
        enable_normalization_checkbox.change(
            fn=auto_save_config,
            inputs=auto_save_inputs,
            outputs=[]
        )

        # 开始Phase1
        def start_phase1(project, n_trials, scheme, enable_timer_val, timer_hours_val, warm_start, enable_norm, gpu_p1_val, gpu_strategy_p1_val, *all_values):
            if project is None:
                return "❌ 请先加载项目", ""

            # ★ v3.8: 把 UI 上的 GPU 设置写到 config.yaml
            # phase1_global 通过 _get_m5_config() 动态读 config
            _update_m5_gpu_config(gpu_p1_val, gpu_strategy_p1_val)

            # ★ 开始前强制保存当前配置（DB不存在时）
            if not os.path.exists(project.get("p1_db_path", "")):
                save_msg = on_save_p1_config(
                    project, scheme, enable_norm, *all_values
                )
                if save_msg.startswith("❌"):
                    return save_msg, ""
                logger.info(f"开始前自动保存配置：{save_msg}")

            # 从config读取（此时已是最新保存的值）
            cfg = load_p1_config(project)["config"]
            param_ranges = cfg.get("param_ranges", {})
            # ★ 读取归一化开关（用于 ObjectiveFunction 实时归一化评分）
            # ★ v2.0：UI 开关为准，config 中不存在或为 False 时以 UI 为准
            enable_norm_p1 = bool(enable_norm)

            if warm_start:
                logger.info(f"热启动先验已加载（{len(warm_start)}个参数）")

            log_msg = f"开始Phase1: {n_trials} trials, fast_mode=False(全量{calc_window_count()}窗口)\n"

            # ★ 在启动线程/定时器之前先清除停止事件
            _stop_now_event.clear()
            _stop_graceful_event.clear()

            # 启动定时器
            if enable_timer_val and timer_hours_val:
                start_graceful_timer(timer_hours_val)
                end_dt = (datetime.now() + timedelta(hours=timer_hours_val)).strftime("%H:%M")
                timer_msg = f"⏰ 定时停止已设置：将于 {end_dt} 后优雅停止"
            else:
                timer_msg = ""

            def run_thread():
                try:
                    factor_df = _load_factor_df(scheme=scheme)
                except MemoryError as e:
                    _log_queue_p1.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"❌ 内存不足，无法加载数据\n"
                        f"  {e}\n"
                        f"  建议：关闭其他程序或重启优化器\n---\n"
                    )
                    return
                except Exception as e:
                    _log_queue_p1.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"❌ 加载数据失败: {e}\n---\n"
                    )
                    return

                # 带时间预估的progress_callback
                progress_callback = _make_progress_callback(_log_queue_p1)

                try:
                    run_phase1(
                        factor_df=factor_df,
                        project=project,
                        n_trials=n_trials,
                        # fast_mode和window_count已在phase1_global中固定
                        objective_weights=cfg.get("objective_weights", {}),
                        active_params=cfg.get("active_params"),
                        scheme=cfg.get("scheme", scheme),
                        param_ranges=param_ranges,
                        stop_now_event=_stop_now_event,
                        stop_graceful_event=_stop_graceful_event,
                        progress_callback=progress_callback,
                        warm_start=warm_start,
                        # ★ 归一化开关：P1 实时评分也走归一化逻辑
                        enable_normalization=enable_norm_p1,
                        norm_config=NORM_CONFIG,
                    )
                except Exception as e:
                    logger.error(f"Phase1运行异常: {e}")

            global _phase1_thread
            _phase1_thread = threading.Thread(target=run_thread, daemon=True)
            _phase1_thread.start()

            return log_msg, timer_msg

        btn_start_p1.click(
            fn=start_phase1,
            inputs=(
                [project_state, n_trials_slider, scheme_radio, enable_timer_p1, timer_hours_p1, warm_start_state, enable_normalization_checkbox, gpu_p1_checkbox, gpu_strategy_p1]
                + list(weight_sliders.values())
                + [w["checkbox"] for w in param_widgets.values()]
                + [w["choices"] for w in param_widgets.values() if "choices" in w]   # ★ 新增
                + all_low_boxes    # ★ 新增
                + all_high_boxes   # ★ 新增
            ),
            outputs=[log_box_p1, timer_status_p1]
        )

        # 停止按钮
        def stop_now(project):
            _stop_now_event.set()
            cancel_graceful_timer()
            release_memory_to_os()

            if project is None:
                return "⚡ 立即停止已触发", ""

            # ★ 立即停止：先做一次"快速清理"删除 DB 中已有的 RUNNING/FAIL 残留
            # 这样用户立刻就能看到 DB 处于一致状态，可以安全关闭程序
            # （即使主线程仍在跑，DB 已经没有"卡住"的 Trial）
            try:
                quick_result = cleanup_bad_trials(project, phase="p1")
                _log_queue_p1.put(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"⚡ 立即停止：已快速清理 {quick_result.get('deleted', 0)} 个异常Trial（保留 {quick_result.get('kept', 0)} 个）\n"
                )
            except Exception as e:
                logger.warning(f"快速清理失败（不影响主流程）: {e}")

            # ★ 等待主线程结束（最多等 10 秒，缩短等待时间）
            # 由于 objective.__call__ 入口已检查 stop_event 并抛异常，
            # 当前 Trial 会立即被标记为 FAIL；trial_callback 中的 study.stop()
            # 会阻止 Optuna 继续发起新 Trial，主线程会快速返回。
            def _delayed_cleanup():
                if _phase1_thread and _phase1_thread.is_alive():
                    _phase1_thread.join(timeout=10)

                # 线程退出后再次清理（兜底，可能有 FAIL 状态的 Trial 写入）
                try:
                    result = cleanup_bad_trials(project, phase="p1")
                    _log_queue_p1.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"⚡ 立即停止完成 | {result['message']}\n"
                        f"  有效Trial保留：{result['kept']}个\n---\n"
                    )
                except Exception as e:
                    _log_queue_p1.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"⚡ 立即停止：兜底清理异常: {e}\n---\n"
                    )

            cleanup_thread = threading.Thread(
                target=_delayed_cleanup, daemon=True
            )
            cleanup_thread.start()

            return "⚡ 立即停止已触发，正在终止当前Trial并清理异常Trial...", ""

        def stop_graceful():
            _stop_graceful_event.set()
            cancel_graceful_timer()
            return "🏁 优雅停止已触发，当前Trial完成后停止", ""

        btn_stop_now.click(
            fn=stop_now,
            inputs=[project_state],   # ★ 追加project_state
            outputs=[log_box_p1, timer_status_p1]
        )
        btn_stop_graceful.click(fn=stop_graceful, outputs=[log_box_p1, timer_status_p1])

        # ★ 手动清理按钮
        def manual_cleanup_p1(project, dry_run):
            if project is None:
                return "❌ 请先加载项目"
            result = cleanup_bad_trials(project, phase="p1", dry_run=dry_run)
            lines = [result["message"]]
            # 分类摘要
            cat = result.get("category", {})
            if any(v > 0 for v in cat.values()):
                cat_lines = [f"  {k}: {v}" for k, v in cat.items() if v > 0]
                lines.append("📊 异常分类:")
                lines.extend(cat_lines)
            if result.get("detail"):
                lines.append(f"📋 详情(前{len(result['detail'])}条):")
                lines.extend(f"  {d}" for d in result["detail"])
            return "\n".join(lines)

        btn_cleanup_p1.click(
            fn=manual_cleanup_p1,
            inputs=[project_state, dry_run_p1],
            outputs=[log_box_p1]
        )

        # ★ 刷新运行日志按钮
        def refresh_log_p1(current_log):
            lines = []
            while not _log_queue_p1.empty():
                try:
                    lines.append(_log_queue_p1.get_nowait())
                except queue.Empty:
                    break
            if lines:
                return current_log + "".join(lines)
            return current_log + ""  # 无新内容不变

        btn_refresh_log_p1.click(
            fn=refresh_log_p1,
            inputs=[log_box_p1],
            outputs=[log_box_p1]
        )

        # ★ 尝试添加自动刷新Timer（Gradio 4.x支持）
        try:
            log_timer_p1 = gr.Timer(value=30, active=False)
            log_timer_p1.tick(fn=refresh_log_p1, inputs=[log_box_p1], outputs=[log_box_p1])
            # ★ 修复: 给 start_phase1 追加 GPU args (与上面主 handler 一致)
            # 否则 GPU=False 时 args 错位, _update_m5_gpu_config(weight_val, weight_val) 导致 P1 崩溃
            btn_start_p1.click(
                fn=start_phase1,
                inputs=(
                    [project_state, n_trials_slider, scheme_radio, enable_timer_p1, timer_hours_p1, warm_start_state, enable_normalization_checkbox, gpu_p1_checkbox, gpu_strategy_p1]
                    + list(weight_sliders.values())
                    + [w["checkbox"] for w in param_widgets.values()]
                    + [w["choices"] for w in param_widgets.values() if "choices" in w]   # ★ 新增
                    + all_low_boxes    # ★ 新增
                    + all_high_boxes   # ★ 新增
                ),
                outputs=[log_box_p1, timer_status_p1]
            ).then(fn=lambda: gr.update(active=True), outputs=[log_timer_p1])
            btn_stop_now.click(fn=stop_now, inputs=[project_state], outputs=[log_box_p1, timer_status_p1]).then(fn=lambda: gr.update(active=False), outputs=[log_timer_p1])
            btn_stop_graceful.click(fn=stop_graceful, outputs=[log_box_p1, timer_status_p1]).then(fn=lambda: gr.update(active=False), outputs=[log_timer_p1])
        except (AttributeError, TypeError):
            pass  # Gradio版本不支持Timer，仅手动刷新

        # 定时器勾选事件
        enable_timer_p1.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[enable_timer_p1],
            outputs=[timer_hours_p1]
        )

        # ★ 定时器状态检查按钮
        def check_timer_status():
            if _stop_graceful_event.is_set():
                return "🏁 定时器已触发，等待停止..."
            if _timer_thread and _timer_thread.is_alive():
                return "⏰ 定时器运行中"
            return "未启动定时器"

        btn_check_timer.click(
            fn=check_timer_status,
            outputs=[timer_status_p1]
        )

        # ━━━ Tab2: 结果分析事件 ━━━

        all_min_sliders = [filter_sliders[name]["min"] for name in filter_slider_names]
        all_max_sliders = [filter_sliders[name]["max"] for name in filter_slider_names]
        all_filter_inputs = all_min_sliders + all_max_sliders

        def load_project_for_t2(path):
            if not path or not path.strip():
                no_update = [gr.update()] * len(all_min_sliders) * 2
                return (None, "⚠️ 请输入项目文件夹路径", *no_update)
            result = init_project(path.strip())
            if result["status"] == "conflict":
                no_update = [gr.update()] * len(all_min_sliders) * 2
                return (None, "❌ 该路径不是有效项目文件夹", *no_update)
            project = result["project"]
            p1_study = get_p1_study(project)
            if p1_study is None:
                no_update = [gr.update()] * len(all_min_sliders) * 2
                return (project, "⚠️ P1数据库尚未创建，请先运行Phase1", *no_update)
            stats = get_study_stats(p1_study)

            # ★ 计算滑块范围时跳过无数据/全 NaN 的指标；
            #   即便个别指标异常也不会让整个 Tab2 加载崩溃。
            try:
                min_updates, max_updates = _compute_metric_slider_updates(p1_study)
            except Exception as e:
                no_update = [gr.update()] * len(all_min_sliders) * 2
                return (project,
                        f"⚠️ 项目已加载，但滑块初始化失败（{e}）。请尝试「初始化滑块」按钮。",
                        *no_update)

            slider_updates = min_updates + max_updates

            status_text = (
                f"✅ 已加载 **{project['project_name']}**｜"
                f"累计 {stats['total']} 个Trial，"
                f"有效 {stats['complete']} 个，异常 {stats.get('abnormal', 0)} 个，失败 {stats['fail']} 个\n"
                f"因变量滑块已更新为实际数据范围"
            )
            return (project, status_text, *slider_updates)

        btn_t2_load.click(
            fn=load_project_for_t2,
            inputs=[t2_project_path],
            outputs=[t2_project_state, t2_project_status,
                     *all_min_sliders, *all_max_sliders]
        )

        def update_matched_count(*args):
            *filter_values, project = args
            try:
                if project is None:
                    return "📊 请先在上方加载项目"

                p1_study = get_p1_study(project)
                if p1_study is None:
                    return "📊 P1数据库尚未创建，请先运行Phase1"

                stats = get_study_stats(p1_study)

                n = len(filter_slider_names)
                min_values = filter_values[:n]
                max_values = filter_values[n:]

                existing_metrics = set()
                for t in p1_study.trials:
                    if (t.state == optuna.trial.TrialState.COMPLETE
                            and t.value is not None
                            and t.value > -999.0):
                        existing_metrics.update(t.user_attrs.keys())

                filter_conditions = {}
                for i, name in enumerate(filter_slider_names):
                    if name not in existing_metrics:
                        continue
                    min_v = min_values[i]
                    max_v = max_values[i]
                    sl_min_val, sl_max_val = METRIC_BOUNDS.get(name, (-2, 2))
                    filter_conditions[name] = (
                        min_v if min_v > (sl_min_val + 0.001) else -999,
                        max_v if max_v < (sl_max_val - 0.001) else 999
                    )

                n_matched, n_total = count_matched(p1_study, filter_conditions)
                pct = f"{n_matched/n_total*100:.1f}%" if n_total > 0 else "0%"
                ab = stats.get("abnormal", 0)
                return (
                    f"📊 数据库共 {stats['total']} 个Trial "
                    f"（有效 {stats['complete']} / 异常 {ab} / 失败 {stats['fail']}）｜"
                    f"当前过滤条件满足：**{n_matched} / {stats['complete']}** 个有效Trial（{pct}）"
                )
            except Exception as e:
                return f"📊 读取失败：{str(e)}"

        btn_refresh_count.click(
            fn=update_matched_count,
            inputs=all_filter_inputs + [t2_project_state],
            outputs=[trial_count_display]
        ).then(
            fn=lambda project: cleanup_bad_trials(project, phase="p1", dry_run=True) if project else {"message": "无项目", "kept": 0, "cleaned": 0},
            inputs=[t2_project_state],
            outputs=[]
        )

        # ★ 初始化滑块按钮：从实际Trial数据计算真实范围
        def reset_sliders_to_data_range(project):
            """
            把所有因变量滑块重置为实际数据范围：
            - minimum/maximum 基于P2/P98+20%扩展
            - value: 下限=实际最小，上限=实际最大
            """
            def _empty():
                return [gr.update()] * len(filter_slider_names)

            if project is None:
                return (*_empty(), *_empty())

            p1_study = get_p1_study(project)
            if p1_study is None:
                return (*_empty(), *_empty())

            complete_trials = [
                t for t in p1_study.trials
                if (t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None
                    and t.value > -999.0)
            ]

            if not complete_trials:
                return (*_empty(), *_empty())

            min_updates, max_updates = _compute_metric_slider_updates(p1_study)
            return (*min_updates, *max_updates)

        # ★ 链式调用：先更新滑块，再重新计算统计
        btn_reset_sliders.click(
            fn=reset_sliders_to_data_range,
            inputs=[t2_project_state],
            outputs=[
                *all_min_sliders,
                *all_max_sliders,
            ]
        ).then(
            fn=update_matched_count,
            inputs=all_filter_inputs + [t2_project_state],
            outputs=[trial_count_display]
        )

        for sl in all_min_sliders + all_max_sliders:
            sl.change(
                fn=update_matched_count,
                inputs=all_filter_inputs + [t2_project_state],
                outputs=[trial_count_display]
            )

        # ★ Tab2 Top-5 排名展示的关键指标（与 Tab4 保持一致）
        TOP5_METRIC_KEYS = [
            ("val_ic", "IC"),
            ("val_icir", "ICIR"),
            ("val_rolling6m_ir", "6M_IR"),
            ("pct_positive_excess", "月度超额胜率"),
            ("val_jensen_alpha", "Jα"),
            ("val_appraisal_ratio", "AR"),
            ("up_capture_ratio", "上行捕获比"),
            ("capture_ratio", "综合捕获比"),
            ("penalized_rate", "降权率"),
            ("val_beta", "β"),
        ]

        def _render_top5_html(study, filter_conditions=None, top_n: int = 5, source_tag: str = "P1"):
            """渲染 Top-N Trial 排名（统一指标集，支持归一化/原始双模态）

            filter_conditions: dict[str, tuple[lo, hi]]
                - 与 do_analyze 中 build 的 filter_conditions 完全一致
                - 仅在 [lo, hi] 内的 Trial 参与排名（不通过则视为被过滤）
            """
            base = ("border-radius:6px;padding:12px 16px;margin:4px 0;"
                    "border:1px solid var(--border-color-primary,#e0e0e0);"
                    "border-left-width:4px;font-size:14px;")
            empty_style = (f"{base}border-left-color:#ff9800;")

            def _score_of(trial):
                """归一化优先：retro_norm_score → normalized_score → -trial.value"""
                m = trial.user_attrs if hasattr(trial, "user_attrs") else {}
                norm = m.get("retro_norm_score")
                if norm is None:
                    norm = m.get("normalized_score")
                if norm is not None:
                    return float(norm)
                return -float(trial.value) if trial.value is not None else 0.0

            # 1) 取全部有效 Trial
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE
                         and t.value is not None and t.value > -999]
            if not completed:
                return (f"<div style='{empty_style}'>"
                        f"⚠️ {source_tag} 尚无有效Trial</div>")

            n_total = len(completed)

            # 2) 应用过滤条件（关键修复：以前未应用，导致排名与过滤不一致）
            if filter_conditions:
                filtered = []
                for t in completed:
                    m = t.user_attrs if hasattr(t, "user_attrs") else {}
                    pass_all = True
                    for name, (lo, hi) in filter_conditions.items():
                        v = m.get(name, None)
                        if v is None:
                            pass_all = False
                            break
                        try:
                            vf = float(v)
                        except (TypeError, ValueError):
                            pass_all = False
                            break
                        # 浮点容差：上下边界值在 1e-6 内视为通过
                        if vf < lo - 1e-6 or vf > hi + 1e-6:
                            pass_all = False
                            break
                    if pass_all:
                        filtered.append(t)
                completed = filtered

            n_filtered = len(completed)
            if n_filtered == 0:
                # 区分"全部 Trial 被过滤"和"本就无 Trial"
                return (f"<div style='{empty_style}'>"
                        f"⚠️ 当前过滤条件下 {source_tag} 无匹配 Trial"
                        f"（{n_total} 个有效 Trial 全部被过滤）"
                        f"</div>")

            # 3) 排序取 Top-N
            completed.sort(key=lambda t: -_score_of(t))
            top_trials = completed[:min(int(top_n), len(completed))]

            # 过滤信息徽章
            filter_info = ""
            if filter_conditions and n_filtered != n_total:
                filter_info = (f" <span style='color:#9e9e9e;font-size:12px'>"
                               f"[过滤后 {n_filtered}/{n_total}]</span>")
            elif filter_conditions:
                filter_info = f" <span style='color:#9e9e9e;font-size:12px'>[全部 {n_total} 命中]</span>"

            rows_html = ""
            for rank, trial in enumerate(top_trials, 1):
                m = trial.user_attrs if hasattr(trial, 'user_attrs') else {}
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
                border_color = {1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32"}.get(rank, "#2196f3")

                norm_score = m.get("retro_norm_score")
                if norm_score is None:
                    norm_score = m.get("normalized_score")
                raw_score = -float(trial.value) if trial.value is not None else 0.0
                if norm_score is not None:
                    score_str = f"{float(norm_score):.4f}"
                    score_label = "归一化分"
                    is_norm = True
                else:
                    score_str = f"{raw_score:.4f}"
                    score_label = "评分"
                    is_norm = False

                # 关键指标 10 项
                # v4.1: 用 fp16 cast 减少有效数字 (0.083336 → 0.0834)
                metrics_parts = []
                for key, label in TOP5_METRIC_KEYS:
                    v = m.get(key, None)
                    if v is None:
                        metrics_parts.append(f"{label}=—")
                    elif isinstance(v, (int, float)):
                        if abs(v) < 0.01:
                            metrics_parts.append(f"{label}={_sig4(v, 4)}")
                        elif abs(v) < 1:
                            metrics_parts.append(f"{label}={_sig4(v, 3)}")
                        else:
                            metrics_parts.append(f"{label}={_sig4(v, 2)}")
                    else:
                        metrics_parts.append(f"{label}={v}")
                metrics_str = " · ".join(metrics_parts) if metrics_parts else "无指标"

                mode_badge = ""
                if is_norm:
                    mode_badge = " <span style='color:#2196f3;font-size:12px'>[归一化]</span>"
                else:
                    mode_badge = (" <span style='color:#ff9800;font-size:12px'>"
                                  "[无归一化数据,降级为原始分]</span>")

                rows_html += (
                    f"<div style='{base}border-left-color:{border_color}'>"
                    f"{medal} <b>Trial {trial.number}</b> · "
                    f"<span style='color:#4caf50'><b>★ {score_label}: {score_str}</b></span>"
                    f"<span style='color:#888;font-size:13px'> | 原始分: {raw_score:.4f}</span>"
                    f"{mode_badge}<br>"
                    f"<small>{metrics_str}</small>"
                    f"</div>"
                )
            # 标题栏也带过滤信息
            return f"<div style='{base}border-left-color:#9e9e9e;opacity:0.9'>🏆 Top-{len(top_trials)} 排名{filter_info}</div>" + rows_html

        def do_analyze(*args):
            *filter_and_pct, project = args
            top_pct_val = filter_and_pct[0]
            filter_values = filter_and_pct[1:]
            empty_top5 = ("<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                          "border:1px solid var(--border-color-primary,#e0e0e0);"
                          "border-left-width:4px;border-left-color:#ff9800;font-size:14px'>"
                          "⚠️ 请先加载项目并确保P1有有效Trial</div>")
            try:
                if project is None:
                    return (
                        None,
                        gr.update(value=pd.DataFrame()),
                        gr.update(value="请先在上方加载项目"),
                        gr.update(value=empty_top5),
                    )

                p1_study = get_p1_study(project)
                if p1_study is None:
                    return (
                        None,
                        gr.update(value=pd.DataFrame()),
                        gr.update(value="P1数据库尚未创建"),
                        gr.update(value=empty_top5),
                    )

                n = len(filter_slider_names)
                min_values = filter_values[:n]
                max_values = filter_values[n:]

                existing_metrics = set()
                for t in p1_study.trials:
                    if (t.state == optuna.trial.TrialState.COMPLETE
                            and t.value is not None
                            and t.value > -999.0):
                        existing_metrics.update(t.user_attrs.keys())

                filter_conditions = {}
                for i, name in enumerate(filter_slider_names):
                    if name not in existing_metrics:
                        continue
                    min_v = min_values[i]
                    max_v = max_values[i]
                    sl_min_val, sl_max_val = METRIC_BOUNDS.get(name, (-2, 2))
                    filter_conditions[name] = (
                        min_v if min_v > (sl_min_val + 0.001) else -999,
                        max_v if max_v < (sl_max_val - 0.001) else 999
                    )

                result = analyze_ranges(
                    p1_study, filter_conditions,
                    top_pct=top_pct_val
                )

                ranges = result["ranges"]
                rows = []
                cat_rows = []
                for name, r in ranges.items():
                    if r.get("type") == "categorical":
                        cat_rows.append([
                            name,
                            str(r["mode"]),
                            f"{r['mode_freq']:.1%}",
                            "✅ 推荐固定" if r["recommended_fixed"] else "🔍 继续搜索",
                        ])
                    else:
                        rows.append([
                            name,
                            r["low"],
                            r["high"],
                            r["type"],
                            r["default"],
                        ])

                df_numeric = pd.DataFrame(
                    rows,
                    columns=["参数名", "新下限", "新上限", "类型", "默认值"]
                )
                df_cat = pd.DataFrame(
                    cat_rows,
                    columns=["参数名", "众数值", "频率", "建议"]
                )

                cat_text = ""
                if cat_rows:
                    cat_text = "【Categorical参数分析】\n" + df_cat.to_string(index=False)

                status_msg = (
                    f"分析完成：{result['n_matched']}/{result['n_total']}个Trial"
                    f"（按分数前{top_pct_val:.0f}%筛选后）"
                )
                if cat_text:
                    status_msg = cat_text + "\n\n" + status_msg

                # ★ Top-5 排名（基于过滤后的 Trial，标题会显示 X/Y 命中数）
                top5_html = _render_top5_html(
                    p1_study,
                    filter_conditions=filter_conditions,
                    top_n=5,
                    source_tag="P1",
                )
                return (
                    ranges,                                    # gr.State: 直接存 dict
                    gr.update(value=df_numeric),                # DataFrame: 显式 update
                    gr.update(value=status_msg),                # Textbox: 显式 update
                    gr.update(value=top5_html),                 # HTML: 显式 update
                )
            except Exception as e:
                return (
                    None,
                    gr.update(value=pd.DataFrame()),
                    gr.update(value=f"分析失败：{str(e)}"),
                    gr.update(value=empty_top5),
                )

        btn_analyze.click(
            fn=do_analyze,
            inputs=[top_pct] + all_filter_inputs + [t2_project_state],
            outputs=[range_result_state, range_df, status_box, t2_top5_html]
        )

        # 导出P2配置
        def export_to_p2(project, ranges_data, *p2_weight_and_active):
            if project is None:
                return "请先加载项目", gr.update(value="")

            num_p2_weights = len(p2_weight_sliders)
            p2_weight_values = p2_weight_and_active[:num_p2_weights]

            p2_weights = {}
            p2_weight_names = list(p2_weight_sliders.keys())
            for i, name in enumerate(p2_weight_names):
                if p2_weight_values[i] and p2_weight_values[i] > 0:
                    p2_weights[name] = p2_weight_values[i]

            try:
                p1_cfg = load_p1_config(project)["config"]
                p1_active_params = p1_cfg.get("active_params")
                p1_param_ranges  = p1_cfg.get("param_ranges", {})
            except Exception:
                p1_active_params = None
                p1_param_ranges  = {}

            p2_param_ranges = dict(p1_param_ranges)

            if ranges_data:
                for name, r in ranges_data.items():
                    if r.get("type") == "categorical":
                        if r.get("recommended_fixed"):
                            p2_param_ranges[name] = {"choices": [r["mode"]]}
                        else:
                            p2_param_ranges.setdefault(name, {})
                            p2_param_ranges[name]["choices"] = r["choices"]
                    else:
                        p2_param_ranges[name] = {
                            "low": r["low"], "high": r["high"]
                        }

            p2_active_params = p1_active_params

            # ── 预估可复用Trial数（信息展示，不阻断）──────────────────────
            reuse_preview = ""
            try:
                from m5_optimizer.project_manager import get_reusable_p1_trials
                try:
                    _p2_scheme = load_p2_config(project)["config"].get("scheme", "scheme_b")
                except Exception:
                    _p2_scheme = "scheme_b"
                _preview = get_reusable_p1_trials(
                    project=project,
                    p2_param_ranges=p2_param_ranges,
                    p2_objective_weights=p2_weights,
                    p2_active_params=p2_active_params,
                    p2_scheme=_p2_scheme,
                    p2_window_count=calc_window_count(),
                    max_inject=30,
                )
                if _preview["blocked"] and not _preview["blocked"].startswith("⚠️"):
                    reuse_preview = f"\n⚠️ P1复用不可用: {_preview['blocked']}"
                else:
                    reuse_preview = (
                        f"\n🔄 P1复用预估: "
                        f"P1有效={_preview['total_p1']}个，"
                        f"符合P2范围={_preview['qualified']}个，"
                        f"将注入top-{_preview['injected']}个"
                    )
            except Exception:
                reuse_preview = ""
            # ──────────────────────────────────────────────────────────────

            try:
                _ok_cfg = export_p2_config_from_p1(  # noqa: F841
                    project=project,
                    param_ranges=p2_param_ranges,
                    objective_weights=p2_weights,
                    active_params=p2_active_params,
                )
                cat_fixed = [
                    f"{n}={v['choices'][0]}"
                    for n, v in p2_param_ranges.items()
                    if "choices" in v and len(v["choices"]) == 1
                ]
                summary = f"固定categorical: {', '.join(cat_fixed) if cat_fixed else '无'}"
                return (
                    f"✅ P2配置已导出\n{summary}{reuse_preview}\nactive_params: {'继承P1' if p2_active_params else '全部搜索'}",
                    gr.update(value=project["project_root"]),
                )
            except Exception as e:
                return f"❌ 导出失败：{e}", gr.update(value="")

        btn_export_p2.click(
            fn=export_to_p2,
            inputs=[t2_project_state, range_result_state] + list(p2_weight_sliders.values()),
            outputs=[status_box, t3_project_path]
        )

        # ━━━ Tab3: Phase2项目管理事件 ━━━

        def on_load_t3_project(path):
            _default_cat_updates = [gr.update(value=ALL_PARAMS[n]["choices"]) for n in CATEGORICAL_PARAMS]

            _default_cb_updates  = []
            _default_lo_updates  = []
            _default_hi_updates  = []
            _default_dv_updates  = []
            for name, pdef in ALL_PARAMS.items():
                if pdef["type"] != "categorical":
                    _default_cb_updates.append(gr.update(value=True))
                    _default_lo_updates.append(gr.update(value=pdef["low"], visible=True))
                    _default_hi_updates.append(gr.update(value=pdef["high"], visible=True))
                    _default_dv_updates.append(gr.update(value=pdef["default"], visible=False))

            _default_num_updates = _default_cb_updates + _default_lo_updates + _default_hi_updates + _default_dv_updates
            _default_weight_updates = [gr.update(value=0) for _ in p2_weight_sliders]

            if not path or not path.strip():
                return (
                    None,
                    _card("unconfigured"),
                    gr.update(visible=False),
                    "",
                    gr.update(value=""),
                ) + (gr.update(value="scheme_d"),) + tuple(_default_cat_updates) + tuple(_default_num_updates) + tuple(_default_weight_updates) + tuple(_p2_unlock_updates())

            result = init_project(path.strip())
            if result["status"] == "conflict":
                return (
                    None,
                    _card("conflict", message=result["message"]),
                    gr.update(visible=False),
                    "❌ 该路径不是有效项目文件夹",
                    gr.update(value=""),
                ) + (gr.update(value="scheme_d"),) + tuple(_default_cat_updates) + tuple(_default_num_updates) + tuple(_default_weight_updates) + tuple(_p2_unlock_updates())

            project = result["project"]

            cat_updates = []
            cb_updates  = []
            lo_updates  = []
            hi_updates  = []
            dv_updates  = []
            weight_updates = []
            try:
                p2_cfg = load_p2_config(project)
                p1_cfg = load_p1_config(project)
                p2_ranges = p2_cfg.get("config", {}).get("param_ranges", {})
                p2_active = p2_cfg.get("config", {}).get("active_params")
                p2_weights = p2_cfg.get("config", {}).get("objective_weights", {})
                p2_scheme = p2_cfg.get("config", {}).get("scheme") or p1_cfg.get("config", {}).get("scheme") or "scheme_d"
                for name in CATEGORICAL_PARAMS:
                    if name in p2_ranges and "choices" in p2_ranges[name]:
                        cat_updates.append(gr.update(value=p2_ranges[name]["choices"]))
                    else:
                        cat_updates.append(gr.update(value=ALL_PARAMS[name]["choices"]))
                for name, widgets in p2_param_widgets.items():
                    pdef = ALL_PARAMS[name]
                    rng  = p2_ranges.get(name, {})
                    is_active = (p2_active is None) or (name in p2_active)
                    if is_active:
                        low_val  = rng.get("low",  pdef["low"])
                        high_val = rng.get("high", pdef["high"])
                        cb_updates.append(gr.update(value=True))
                        lo_updates.append(gr.update(value=low_val,  visible=True))
                        hi_updates.append(gr.update(value=high_val, visible=True))
                        dv_updates.append(gr.update(visible=False))
                    else:
                        adj_default = rng.get("adjusted_default", pdef["default"])
                        cb_updates.append(gr.update(value=False))
                        lo_updates.append(gr.update(visible=False))
                        hi_updates.append(gr.update(visible=False))
                        dv_updates.append(gr.update(value=adj_default, visible=True))
                # ★ 加载权重滑块
                for wname in p2_weight_sliders:
                    weight_updates.append(gr.update(value=p2_weights.get(wname, 0)))
            except Exception:
                cat_updates = list(_default_cat_updates)
                cb_updates  = list(_default_cb_updates)
                lo_updates  = list(_default_lo_updates)
                hi_updates  = list(_default_hi_updates)
                dv_updates  = list(_default_dv_updates)
                weight_updates = list(_default_weight_updates)
                p2_scheme = "scheme_d"

            numeric_updates = cb_updates + lo_updates + hi_updates + dv_updates

            p2_study = get_p2_study(project)
            if p2_study is None:
                card_html = _card("fresh", project=project)
                show_action = False
            else:
                stats = get_study_stats(p2_study)
                if stats["complete"] == 0 and stats["fail"] == 0:
                    card_html = _card("fresh", project=project)
                    show_action = True
                else:
                    card_html = _card("in_progress", project=project, stats=stats)
                    show_action = True

            # ★ P2 DB存在时锁定参数，否则解锁
            if os.path.exists(project.get("p2_db_path", "")):
                lock_updates = _p2_lock_updates()
            else:
                lock_updates = _p2_unlock_updates()

            return (
                project,
                card_html,
                gr.update(visible=show_action),
                "",
                gr.update(value=path.strip()),
            ) + (gr.update(value=p2_scheme),) + tuple(cat_updates) + tuple(numeric_updates) + tuple(weight_updates) + tuple(lock_updates)

        btn_t3_load.click(
            fn=on_load_t3_project,
            inputs=[t3_project_path],
            outputs=(
                [t3_project_state, t3_project_card, p2_action_row, p2_reset_tip, t4_project_path, p2_scheme_radio]
                + list(p2_cat_widgets.values())
                + [w["checkbox"]    for w in p2_param_widgets.values()]
                + [w["low"]         for w in p2_param_widgets.values()]
                + [w["high"]        for w in p2_param_widgets.values()]
                + [w["default_val"] for w in p2_param_widgets.values()]
                + list(p2_weight_sliders.values())
                + p2_all_lockable
            )
        )

        # P2重置
        def on_reset_p2(project):
            if project is None:
                return (project, _card("unconfigured"), "❌ 请先加载项目"
                        ) + tuple(_p2_unlock_updates())
            reset_p2(project)
            card_html = _card("fresh", project=project)
            return (project, card_html, "✅ P2数据库已删除，参数已解锁，可重新修改"
                    ) + tuple(_p2_unlock_updates())

        btn_p2_reset.click(
            fn=on_reset_p2,
            inputs=[t3_project_state],
            outputs=[t3_project_state, t3_project_card, p2_reset_tip] + p2_all_lockable
        )

        # 保存P2配置
        def on_save_p2_config(project, lgbm_lr_mode_val, lgbm_depth_mode_val, xgb_lr_mode_val, drop_stn_val, *all_values):
            if project is None:
                return "❌ 请先加载项目", gr.update(visible=True)

            num_weights  = len(p2_weight_sliders)
            num_params   = len(p2_param_widgets)

            weight_values   = all_values[:num_weights]
            rest            = all_values[num_weights:]

            checkbox_values = rest[: num_params]
            low_values      = rest[num_params   : num_params * 2]
            high_values     = rest[num_params*2 : num_params * 3]
            default_values  = rest[num_params*3 : num_params * 4]

            objective_weights = {}
            weight_names = list(p2_weight_sliders.keys())
            for i, name in enumerate(weight_names):
                if weight_values[i] and weight_values[i] > 0:
                    objective_weights[name] = weight_values[i]

            param_ranges   = {}
            excluded_names = []

            param_names = list(p2_param_widgets.keys())

            for i, name in enumerate(param_names):
                checked     = checkbox_values[i]
                low_val     = low_values[i]
                high_val    = high_values[i]
                default_val = default_values[i]

                if checked:
                    if low_val is not None and high_val is not None:
                        param_ranges[name] = {"low": low_val, "high": high_val}
                else:
                    param_ranges[name] = {
                        "adjusted_default": default_val
                        if default_val is not None
                        else ALL_PARAMS[name]["default"]
                    }
                    excluded_names.append(name)

            cat_vals = {
                "lgbm_lr_mode":          lgbm_lr_mode_val   or ALL_PARAMS["lgbm_lr_mode"]["choices"],
                "lgbm_depth_mode":       lgbm_depth_mode_val or ALL_PARAMS["lgbm_depth_mode"]["choices"],
                "xgb_lr_mode":           xgb_lr_mode_val     or ALL_PARAMS["xgb_lr_mode"]["choices"],
                "drop_short_term_noise": drop_stn_val        or ALL_PARAMS["drop_short_term_noise"]["choices"],
            }
            for cat_name, selected in cat_vals.items():
                if selected and len(selected) >= 1:
                    param_ranges[cat_name] = {"choices": list(selected)}

            try:
                config_data = load_p2_config(project)
                existing_ranges = config_data["config"].get("param_ranges", {})
                merged_ranges = dict(existing_ranges)
                merged_ranges.update(param_ranges)
                config_data["config"]["objective_weights"] = objective_weights
                config_data["config"]["param_ranges"] = merged_ranges

                p2_cfg_active = config_data["config"].get("active_params")
                if p2_cfg_active is None:
                    active_params_final = [
                        n for n in ALL_PARAMS
                        if n not in excluded_names
                    ]
                else:
                    active_params_final = [
                        n for n in p2_cfg_active
                        if n not in excluded_names
                    ]
                config_data["config"]["active_params"] = active_params_final

                ok, msg = save_p2_config(project, config_data)
                return msg, gr.update(visible=True)
            except Exception as e:
                return f"❌ 保存失败：{e}", gr.update(visible=True)

        btn_save_p2_config.click(
            fn=on_save_p2_config,
            inputs=(
                [t3_project_state]
                + list(p2_cat_widgets.values())
                + list(p2_weight_sliders.values())
                + [w["checkbox"]    for w in p2_param_widgets.values()]
                + [w["low"]         for w in p2_param_widgets.values()]
                + [w["high"]        for w in p2_param_widgets.values()]
                + [w["default_val"] for w in p2_param_widgets.values()]
            ),
            outputs=[save_p2_config_status]
        )

        # 开始Phase2
        def start_phase2(project, scheme, n_trials, from_best, enable_timer_val, timer_hours_val, lgbm_lr_mode_val, lgbm_depth_mode_val, xgb_lr_mode_val, drop_stn_val, *all_values):
            if project is None:
                return ("❌ 请先加载项目", "") + tuple(_p2_unlock_updates())

            # ★ v3.8: GPU 设置直接从 UI 组件值读取并写入 config.yaml
            # （gpu_p2_checkbox/gpu_strategy_p2 不再通过函数参数传入，避免 inputs 错位）
            try:
                _gpu_p2_val = gpu_p2_checkbox.value if hasattr(gpu_p2_checkbox, 'value') else False
                _gpu_strat_p2_val = gpu_strategy_p2.value if hasattr(gpu_strategy_p2, 'value') else "D"
                _update_m5_gpu_config(_gpu_p2_val, _gpu_strat_p2_val)
            except Exception:
                pass

            # ★ 内存预警：全量P2需要约5-8GB可用内存
            try:
                import psutil
                avail_gb = psutil.virtual_memory().available / 1024**3
                if avail_gb < 3.0:
                    msg = (
                        f"❌ 当前可用内存仅 {avail_gb:.1f}GB，P2全量运行至少需要3GB以上\n"
                        f"建议：\n"
                        f"  • 关闭其他程序（浏览器/IDE/其他Python进程）\n"
                        f"  • 重启电脑释放内存\n"
                    )
                    return (msg, "") + tuple(_p2_unlock_updates())
                elif avail_gb < 6.0:
                    logger.warning(f"可用内存偏低 ({avail_gb:.1f}GB)，P2可能因OOM失败")
            except Exception:
                pass

            num_weights  = len(p2_weight_sliders)
            num_params   = len(p2_param_widgets)

            weight_values   = all_values[:num_weights]
            rest            = all_values[num_weights:]

            checkbox_values = rest[: num_params]
            low_values      = rest[num_params   : num_params * 2]
            high_values     = rest[num_params*2 : num_params * 3]
            default_values  = rest[num_params*3 : num_params * 4]

            objective_weights = {}
            weight_names = list(p2_weight_sliders.keys())
            for i, name in enumerate(weight_names):
                if weight_values[i] and weight_values[i] > 0:
                    objective_weights[name] = weight_values[i]

            param_ranges   = {}
            excluded_names = []

            param_names = list(p2_param_widgets.keys())

            for i, name in enumerate(param_names):
                checked     = checkbox_values[i]
                low_val     = low_values[i]
                high_val    = high_values[i]
                default_val = default_values[i]

                if checked:
                    if low_val is not None and high_val is not None:
                        param_ranges[name] = {"low": low_val, "high": high_val}
                else:
                    param_ranges[name] = {
                        "adjusted_default": default_val
                        if default_val is not None
                        else ALL_PARAMS[name]["default"]
                    }
                    excluded_names.append(name)

            cat_vals = {
                "lgbm_lr_mode":          lgbm_lr_mode_val   or ALL_PARAMS["lgbm_lr_mode"]["choices"],
                "lgbm_depth_mode":       lgbm_depth_mode_val or ALL_PARAMS["lgbm_depth_mode"]["choices"],
                "xgb_lr_mode":           xgb_lr_mode_val     or ALL_PARAMS["xgb_lr_mode"]["choices"],
                "drop_short_term_noise": drop_stn_val        or ALL_PARAMS["drop_short_term_noise"]["choices"],
            }
            for cat_name, selected in cat_vals.items():
                if selected and len(selected) >= 1:
                    param_ranges[cat_name] = {"choices": list(selected)}

            try:
                cfg = load_p2_config(project)["config"]
                if not os.path.exists(project["p2_db_path"]):
                    config_data = load_p2_config(project)
                    existing_ranges = config_data["config"].get("param_ranges", {})
                    merged_ranges = dict(existing_ranges)
                    merged_ranges.update(param_ranges)
                    config_data["config"]["objective_weights"] = objective_weights
                    config_data["config"]["param_ranges"] = merged_ranges
                    config_data["config"]["scheme"] = scheme
                    save_p2_config(project, config_data)
                    cfg = config_data["config"]
            except Exception:
                cfg = {
                    "objective_weights": objective_weights,
                    "param_ranges": param_ranges,
                    "scheme": scheme,
                }

            p2_cfg_active = cfg.get("active_params")

            if p2_cfg_active is None:
                active_params_final = [
                    n for n in ALL_PARAMS
                    if n not in excluded_names
                ]
            else:
                active_params_final = [
                    n for n in p2_cfg_active
                    if n not in excluded_names
                ]

            log_msg = f"开始Phase2: {n_trials} trials, scheme={scheme}\n"
            log_msg += "🔄 将自动检测并复用符合条件的P1 Trial（仅全新study有效）\n"

            _stop_now_event.clear()
            _stop_graceful_event.clear()

            if enable_timer_val and timer_hours_val:
                start_graceful_timer(timer_hours_val)
                end_dt = (datetime.now() + timedelta(hours=timer_hours_val)).strftime("%H:%M")
                timer_msg = f"⏰ 定时停止已设置：将于 {end_dt} 后优雅停止"
            else:
                timer_msg = ""

            def run_thread():
                try:
                    factor_df = _load_factor_df(scheme=scheme)
                except MemoryError as e:
                    _log_queue_p2.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"❌ 内存不足，无法加载数据\n"
                        f"  {e}\n"
                        f"  建议：关闭其他程序或重启优化器\n---\n"
                    )
                    return
                except Exception as e:
                    _log_queue_p2.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"❌ 加载数据失败: {e}\n---\n"
                    )
                    return

                progress_callback = _make_progress_callback(_log_queue_p2)

                try:
                    run_phase2(
                        factor_df=factor_df,
                        project=project,
                        n_trials=n_trials,
                        fast_mode=False,
                        window_count=calc_window_count(),
                        objective_weights=cfg.get("objective_weights", objective_weights),
                        active_params=active_params_final,
                        init_params=None,
                        param_ranges={**cfg.get("param_ranges", {}), **param_ranges},
                        scheme=scheme,
                        stop_now_event=_stop_now_event,
                        stop_graceful_event=_stop_graceful_event,
                        progress_callback=progress_callback,
                        reuse_p1_trials=True,
                        max_p1_inject=min(30, max(10, n_trials // 3)),
                    )
                except Exception as e:
                    logger.error(f"Phase2运行异常: {e}")

            global _phase2_thread
            _phase2_thread = threading.Thread(target=run_thread, daemon=True)
            _phase2_thread.start()

            return (log_msg, timer_msg) + tuple(_p2_lock_updates())

        btn_start_p2.click(
            fn=start_phase2,
            inputs=(
                [t3_project_state, p2_scheme_radio, p2_trials, p2_from_best,
                 enable_timer_p2, timer_hours_p2]
                + list(p2_cat_widgets.values())
                + list(p2_weight_sliders.values())
                + [w["checkbox"]    for w in p2_param_widgets.values()]
                + [w["low"]         for w in p2_param_widgets.values()]
                + [w["high"]        for w in p2_param_widgets.values()]
                + [w["default_val"] for w in p2_param_widgets.values()]
            ),
            outputs=[log_box_p2, timer_status_p2] + p2_all_lockable
        )

        # Phase2停止按钮
        def stop_p2_now(project):
            _stop_now_event.set()
            cancel_graceful_timer()
            release_memory_to_os()
            if project is None:
                return "⚡ 立即停止已触发", ""

            # ★ 立即停止：先做一次"快速清理"删除 DB 中已有的 RUNNING/FAIL 残留
            try:
                quick_result = cleanup_bad_trials(project, phase="p2", dry_run=False)
                _log_queue_p2.put(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"⚡ 立即停止：已快速清理 {quick_result.get('deleted', 0)} 个异常Trial（保留 {quick_result.get('kept', 0)} 个）\n"
                )
            except Exception as e:
                logger.warning(f"快速清理失败（不影响主流程）: {e}")

            def _delayed_cleanup():
                if _phase2_thread and _phase2_thread.is_alive():
                    _phase2_thread.join(timeout=10)
                try:
                    result = cleanup_bad_trials(project, phase="p2")
                    _log_queue_p2.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"⚡ 立即停止完成 | {result['message']}\n---\n"
                    )
                except Exception as e:
                    _log_queue_p2.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"⚡ 立即停止：兜底清理异常: {e}\n---\n"
                    )

            threading.Thread(target=_delayed_cleanup, daemon=True).start()
            return "⚡ 立即停止已触发，正在终止当前Trial并清理异常Trial...", ""

        def stop_p2_graceful():
            _stop_graceful_event.set()
            cancel_graceful_timer()
            return "🏁 优雅停止已触发，当前Trial完成后停止", ""

        btn_stop_p2_now.click(
            fn=stop_p2_now,
            inputs=[t3_project_state],
            outputs=[log_box_p2, timer_status_p2]
        )
        btn_stop_p2_graceful.click(fn=stop_p2_graceful, outputs=[log_box_p2, timer_status_p2])

        # ★ P2手动清理按钮
        def manual_cleanup_p2(project, dry_run):
            if project is None:
                return "❌ 请先加载项目"
            result = cleanup_bad_trials(project, phase="p2", dry_run=dry_run)
            lines = [result["message"]]
            cat = result.get("category", {})
            if any(v > 0 for v in cat.values()):
                cat_lines = [f"  {k}: {v}" for k, v in cat.items() if v > 0]
                lines.append("📊 异常分类:")
                lines.extend(cat_lines)
            if result.get("detail"):
                lines.append(f"📋 详情(前{len(result['detail'])}条):")
                lines.extend(f"  {d}" for d in result["detail"])
            return "\n".join(lines)

        btn_cleanup_p2.click(
            fn=manual_cleanup_p2,
            inputs=[t3_project_state, dry_run_p2],
            outputs=[log_box_p2]
        )

        # ★ 刷新Phase2运行日志按钮
        def refresh_log_p2(current_log):
            lines = []
            while not _log_queue_p2.empty():
                try:
                    lines.append(_log_queue_p2.get_nowait())
                except queue.Empty:
                    break
            if lines:
                return current_log + "".join(lines)
            return current_log + ""

        btn_refresh_log_p2.click(
            fn=refresh_log_p2,
            inputs=[log_box_p2],
            outputs=[log_box_p2]
        )

        # ★ 尝试添加Phase2自动刷新Timer
        try:
            log_timer_p2 = gr.Timer(value=30, active=False)
            log_timer_p2.tick(fn=refresh_log_p2, inputs=[log_box_p2], outputs=[log_box_p2])
            btn_start_p2.click(
                fn=start_phase2,
                inputs=(
                    [t3_project_state, p2_scheme_radio, p2_trials, p2_from_best,
                     enable_timer_p2, timer_hours_p2]
                    + list(p2_cat_widgets.values())
                    + list(p2_weight_sliders.values())
                    + [w["checkbox"]    for w in p2_param_widgets.values()]
                    + [w["low"]         for w in p2_param_widgets.values()]
                    + [w["high"]        for w in p2_param_widgets.values()]
                    + [w["default_val"] for w in p2_param_widgets.values()]
                ),
                outputs=[log_box_p2, timer_status_p2] + p2_all_lockable
            ).then(fn=lambda: gr.update(active=True), outputs=[log_timer_p2])
            btn_stop_p2_now.click(fn=stop_p2_now, inputs=[t3_project_state], outputs=[log_box_p2, timer_status_p2]).then(fn=lambda: gr.update(active=False), outputs=[log_timer_p2])
            btn_stop_p2_graceful.click(fn=stop_p2_graceful, outputs=[log_box_p2, timer_status_p2]).then(fn=lambda: gr.update(active=False), outputs=[log_timer_p2])
        except (AttributeError, TypeError):
            pass  # Gradio版本不支持Timer，仅手动刷新

        # Phase2定时器勾选事件
        enable_timer_p2.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[enable_timer_p2],
            outputs=[timer_hours_p2]
        )

        # ━━━ Tab4: 结果与部署事件 ━━━

        # ── 辅助：从P2 study获取Top-N Trials ──
        def _get_p2_study(project):
            """安全加载P2 study"""
            if project is None:
                return None
            p2_db = project.get("p2_db_path", "")
            if not p2_db or not os.path.exists(p2_db):
                return None
            return optuna.load_study(
                study_name=project.get("p2_study_name", "phase2_local"),
                storage=f"sqlite:///{p2_db}",
            )

        # ── 辅助：安全 float 转换（None / 异常 → 0.0） ──
        def _safe_float(x, default: float = 0.0) -> float:
            """把输入安全转换为 float；None、NaN、字符串、异常统一回退到 default。"""
            if x is None:
                return default
            try:
                v = float(x)
                # NaN / Inf 也视作无效
                if v != v or v in (float("inf"), float("-inf")):
                    return default
                return v
            except (TypeError, ValueError):
                return default

        # ── 辅助：统一从 trial 中提取评分（支持双模态）──
        def _resolve_trial_score(trial, score_mode):
            """按评分模式提取 Trial 的有效评分（用于排序与卡片展示）。

            参数:
                trial: optuna.Trial
                score_mode: "原始评分 (Raw)" | "归一化评分 (Normalized)"
            返回:
                (display_score, raw_score, mode_tag) 元组
                  - display_score: 排序/卡片主显示分
                  - raw_score: 始终为 -trial.value（用于对比展示）
                  - mode_tag: "raw" | "norm" 标记
            """
            raw_score = (-trial.value) if (trial.value is not None) else 0.0
            raw_score = _safe_float(raw_score)

            if score_mode == "归一化评分 (Normalized)":
                # 优先取 retro_norm_score（追溯工具生成），其次 normalized_score（新内核），最后回退到 raw
                user_attrs = trial.user_attrs if hasattr(trial, "user_attrs") else {}
                norm = user_attrs.get("retro_norm_score", None)
                if norm is None:
                    norm = user_attrs.get("normalized_score", None)
                # 区分"归一化分=0"和"无归一化数据"：仅当原始字段为None时回退
                if norm is None:
                    display_score = raw_score
                    mode_tag = "raw_fallback"
                else:
                    display_score = _safe_float(norm)
                    mode_tag = "norm"
            else:
                display_score = raw_score
                mode_tag = "raw"

            return display_score, raw_score, mode_tag

        # ── 辅助：检测 Study 是否含有归一化数据 ──
        def _has_normalized_data(study):
            """检测 Study 中是否至少有一个 Trial 包含归一化评分字段。"""
            try:
                for t in study.trials:
                    if t.state != optuna.trial.TrialState.COMPLETE:
                        continue
                    user_attrs = t.user_attrs if hasattr(t, "user_attrs") else {}
                    if user_attrs.get("retro_norm_score") is not None:
                        return True
                    if user_attrs.get("normalized_score") is not None:
                        return True
            except Exception:
                pass
            return False

        # ── 辅助：渲染Top-N排名HTML ──
        def _render_ranking_html(study, top_n, score_mode="原始评分 (Raw)"):
            """渲染Top-N Trial排名卡片（支持双模态排序与对比展示）"""
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE
                         and t.value is not None and t.value > -999]
            if not completed:
                return ("<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                        "border:1px solid var(--border-color-primary,#e0e0e0);"
                        "border-left-width:4px;border-left-color:#ff9800;font-size:14px'>"
                        "⚠️ P2尚无有效Trial</div>"), gr.update(choices=[])

            # ★ 双模态排序键
            is_norm_mode = (score_mode == "归一化评分 (Normalized)")

            def _sort_key(trial):
                display_score, _, _ = _resolve_trial_score(trial, score_mode)
                # Optuna minimize ⇒ value 越小越好 ⇒ display_score 越大越好
                return -display_score

            completed.sort(key=_sort_key)
            top_trials = completed[:min(int(top_n), len(completed))]

            # ★ 关键指标列表（与 Tab2 保持统一：10 项）
            metric_keys = [
                ("val_ic", "IC"), ("val_icir", "ICIR"),
                ("val_rolling6m_ir", "6M_IR"),
                ("pct_positive_excess", "月度超额胜率"),
                ("val_jensen_alpha", "Jα"), ("val_appraisal_ratio", "AR"),
                ("up_capture_ratio", "上行捕获比"),
                ("capture_ratio", "综合捕获比"),
                ("penalized_rate", "降权率"),
                ("val_beta", "β"),
            ]

            # 构建HTML表格
            base = ("border-radius:6px;padding:12px 16px;margin:4px 0;"
                    "border:1px solid var(--border-color-primary,#e0e0e0);"
                    "border-left-width:4px;font-size:14px;")

            rows_html = ""
            for rank, trial in enumerate(top_trials, 1):
                m = trial.user_attrs if hasattr(trial, 'user_attrs') else {}
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
                border_color = {1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32"}.get(rank, "#2196f3")

                # ★ 双模态分数解析
                display_score, raw_score, mode_tag = _resolve_trial_score(trial, score_mode)

                # 主显示分（带模式标签）
                if mode_tag == "raw" or mode_tag == "raw_fallback":
                    score_str = f"{display_score:.4f}"
                    score_label = "评分"
                else:
                    score_str = f"{display_score:.4f}"
                    score_label = "归一化分"

                # 原始分（始终展示，用于对比）
                raw_str = f"{raw_score:.4f}"

                # 因变量指标
                # v4.1: fp16 cast 减少有效数字
                metrics_parts = []
                for key, label in metric_keys:
                    v = m.get(key, None)
                    if v is None:
                        metrics_parts.append(f"{label}=—")
                    elif isinstance(v, (int, float)):
                        if abs(v) < 0.01:
                            metrics_parts.append(f"{label}={_sig4(v, 4)}")
                        elif abs(v) < 1:
                            metrics_parts.append(f"{label}={_sig4(v, 3)}")
                        else:
                            metrics_parts.append(f"{label}={_sig4(v, 2)}")
                metrics_str = " · ".join(metrics_parts) if metrics_parts else "无指标"

                # ★ 双分数对比渲染
                mode_badge = ""
                if is_norm_mode and mode_tag == "raw_fallback":
                    mode_badge = (" <span style='color:#ff9800;font-size:12px'>"
                                  "[无归一化数据,降级为原始分]</span>")
                elif is_norm_mode:
                    mode_badge = " <span style='color:#2196f3;font-size:12px'>[归一化]</span>"

                rows_html += (
                    f"<div style='{base}border-left-color:{border_color}'>"
                    f"{medal} <b>Trial {trial.number}</b> · "
                    f"<span style='color:#4caf50'><b>★ {score_label}: {score_str}</b></span>"
                    f"<span style='color:#888;font-size:13px'> | 原始分: {raw_str}</span>"
                    f"{mode_badge}<br>"
                    f"<small>{metrics_str}</small>"
                    f"</div>"
                )

            # Dropdown choices（按当前模式显示对应分）
            choices = []
            for t in top_trials:
                display_score, raw_score, mode_tag = _resolve_trial_score(t, score_mode)
                if mode_tag in ("norm",):
                    label = f"Trial {t.number} (归一化 {display_score:.4f} | 原始 {raw_score:.4f})"
                else:
                    label = f"Trial {t.number} (原始 {display_score:.4f})"
                choices.append(label)

            return rows_html, gr.update(choices=choices, value=choices[0] if choices else None)

        # ── 辅助：参数收敛分析 ──
        def _render_convergence_html(study, score_mode="原始评分 (Raw)"):
            """渲染参数收敛分析HTML（联动评分模式切片）"""
            import numpy as np
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE
                         and t.value is not None and t.value > -999]
            if len(completed) < 5:
                mode_hint = "（归一化模式）" if score_mode == "归一化评分 (Normalized)" else "（原始模式）"
                return ("<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                        "border:1px solid var(--border-color-primary,#e0e0e0);"
                        "border-left-width:4px;border-left-color:#ff9800;font-size:14px'>"
                        f"⚠️ {mode_hint} 有效Trial不足5个，无法进行收敛分析</div>")

            # ★ 按选定评分模式排序
            def _sort_key(trial):
                display_score, _, _ = _resolve_trial_score(trial, score_mode)
                return -display_score

            completed.sort(key=_sort_key)
            half = len(completed) // 2
            top_half = completed[:half]
            bot_half = completed[half:]

            base = ("border-radius:6px;padding:12px 16px;margin:4px 0;"
                    "border:1px solid var(--border-color-primary,#e0e0e0);"
                    "border-left-width:4px;font-size:14px;")

            # 收集所有参数名
            all_param_names = set()
            for t in completed:
                all_param_names.update(t.params.keys())

            converged_count = 0
            total_count = 0
            rows_converged = ""
            rows_unconverged = ""

            for pname in sorted(all_param_names):
                p_info = ALL_PARAMS.get(pname, {})
                p_type = p_info.get("type", "float")
                total_count += 1

                # 收集参数值
                top_vals = [t.params.get(pname) for t in top_half if pname in t.params]
                bot_vals = [t.params.get(pname) for t in bot_half if pname in t.params]

                if not top_vals:
                    continue

                # 类别型/布尔型
                is_cat = p_type == "categorical" or isinstance(top_vals[0], bool) or isinstance(top_vals[0], str)
                if is_cat:
                    mode_val = Counter(top_vals).most_common(1)[0][0]
                    mode_freq = Counter(top_vals).most_common(1)[0][1] / len(top_vals)
                    if mode_freq > 0.80:
                        converged_count += 1
                        rows_converged += (
                            f"<tr><td>{pname}</td><td>mode={mode_val} ({mode_freq:.0%})</td>"
                            f"<td style='color:#4caf50'>✓ 收敛</td></tr>"
                        )
                    else:
                        rows_unconverged += (
                            f"<tr><td>{pname}</td><td>mode={mode_val} ({mode_freq:.0%})</td>"
                            f"<td style='color:#ff9800'>→ 继续搜索</td></tr>"
                        )
                    continue

                # 数值型
                try:
                    top_arr = np.array([float(v) for v in top_vals])
                    bot_arr = np.array([float(v) for v in bot_vals]) if bot_vals else np.array([])
                    top_mean = float(top_arr.mean())
                    top_std = float(top_arr.std())
                    cv = top_std / (abs(top_mean) + 1e-8)

                    if len(bot_arr) > 1 and bot_arr.std() > 0:
                        std_ratio = top_std / bot_arr.std()
                    else:
                        std_ratio = float("inf")

                    # 收敛条件：std_ratio < 0.7 且 CV < 0.30
                    is_converged = std_ratio < 0.7 and cv < 0.30

                    # 搜索范围占原始范围比
                    p_low = p_info.get("low", 0)
                    p_high = p_info.get("high", 1)
                    search_range = top_arr.max() - top_arr.min()
                    original_range = abs(p_high - p_low) if p_high != p_low else 1
                    range_ratio = search_range / original_range

                    if is_converged:
                        converged_count += 1
                        rows_converged += (
                            f"<tr><td>{pname}</td>"
                            f"<td>mean={top_mean:.4f} std={top_std:.4f} CV={cv:.2%}</td>"
                            f"<td>范围占比={range_ratio:.0%}</td>"
                            f"<td style='color:#4caf50'>✓ 收敛</td></tr>"
                        )
                    else:
                        reason = ""
                        if std_ratio >= 0.7:
                            reason += "好/差Trial分散度接近 "
                        if cv >= 0.30:
                            reason += f"CV={cv:.0%}偏高 "
                        rows_unconverged += (
                            f"<tr><td>{pname}</td>"
                            f"<td>mean={top_mean:.4f} std={top_std:.4f} CV={cv:.2%}</td>"
                            f"<td>范围占比={range_ratio:.0%}</td>"
                            f"<td style='color:#ff9800'>→ {reason.strip()}</td></tr>"
                        )
                except (ValueError, TypeError):
                    continue

            conv_rate = converged_count / total_count if total_count > 0 else 0
            conv_color = "#4caf50" if conv_rate >= 0.70 else "#ff9800" if conv_rate >= 0.40 else "#f44336"

            html = (
                f"<div style='{base}border-left-color:{conv_color}'>"
                f"<b>收敛率：{converged_count}/{total_count} = {conv_rate:.0%}</b><br>"
                f"<small>判定条件：std_ratio&lt;0.7 且 CV&lt;30%（类别型mode&gt;80%）</small><br><br>"
                f"<table style='width:100%;border-collapse:collapse;font-size:13px'>"
                f"<tr style='border-bottom:1px solid #666'><th align='left'>参数</th>"
                f"<th align='left'>统计</th><th align='left'>范围</th><th align='left'>状态</th></tr>"
                f"{rows_converged}{rows_unconverged}"
                f"</table></div>"
            )
            return html

        # Task 13: Tab4 加载项目
        def on_load_t4_project(path, current_mode=None):
            """Tab4 加载项目（仅分析P2）"""
            try:
                result = init_project(path.strip())
                if result["status"] not in ("loaded", "created"):
                    card_html = _card("conflict", message=result.get("message", "加载失败"))
                    empty_card = ("<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                                  "border:1px solid var(--border-color-primary,#e0e0e0);"
                                  "border-left-width:4px;border-left-color:#9e9e9e;opacity:0.7;"
                                  "font-size:14px'>⬜ 加载项目后显示</div>")
                    tip_no_norm = "⚠️ 加载失败，无法检测归一化数据。"
                    return (None, "数据源：加载失败", card_html, path, path,
                            empty_card, empty_card, gr.update(choices=[]),
                            gr.update(interactive=False, value="原始评分 (Raw)"),
                            tip_no_norm)

                project = result["project"]
                p2_exists = os.path.exists(project.get("p2_db_path", ""))

                if p2_exists:
                    study = _get_p2_study(project)
                    stats = get_study_stats(study)
                    c = stats["complete"]
                    ab = stats.get("abnormal", 0)
                    f_ = stats["fail"]
                    t = stats["total"]
                    best = f"{stats['best_value']:.4f}" if stats.get("best_value") is not None else "暂无"

                    p2_cfg = load_p2_config(project).get("config", {})
                    scheme = p2_cfg.get("scheme", "未知")

                    card_html = _card("in_progress" if c < t else "completed",
                                      project=project, stats=stats)
                    db_info = f"P2 · scheme={scheme} · 有效{c}✅ 异常{ab}⚠️ 失败{f_}❌ · 最优{best}"

                    # ★ 检测归一化数据，决定 Radio 是否可切换
                    has_norm = _has_normalized_data(study)
                    enable_norm = p2_cfg.get("enable_normalization", False) or has_norm
                    current_mode_eff = current_mode if (current_mode and enable_norm) else "原始评分 (Raw)"

                    # Top-5排名 + 收敛分析（按当前评分模式）
                    ranking_html, dropdown_update = _render_ranking_html(
                        study, 5, score_mode=current_mode_eff
                    )
                    convergence_html = _render_convergence_html(
                        study, score_mode=current_mode_eff
                    )

                    # ★ 模式切换提示
                    if has_norm:
                        tip_text = (
                            "💡 检测到归一化数据（`retro_norm_score` 或 `normalized_score`）。"
                            "切换至**归一化评分**模式可查看综合多维度的真实排名。"
                        )
                    elif enable_norm:
                        tip_text = "💡 项目配置启用了 `enable_normalization`，但尚未发现归一化数据。归一化模式可降级为原始分。"
                    else:
                        tip_text = (
                            "💡 当前为**原始评分**模式。归一化数据可使用 "
                            "`python -m m5_optimizer.utils.retroactive_normalize` 追溯生成。"
                        )

                    radio_update = gr.update(
                        interactive=enable_norm,
                        value=current_mode_eff,
                    )
                else:
                    card_html = _card("fresh", project=project)
                    db_info = "P2数据库未找到，请先在Tab3运行Phase2"
                    empty_card = ("<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                                  "border:1px solid var(--border-color-primary,#e0e0e0);"
                                  "border-left-width:4px;border-left-color:#9e9e9e;opacity:0.7;"
                                  "font-size:14px'>⬜ P2数据库不存在</div>")
                    ranking_html = empty_card
                    convergence_html = empty_card
                    dropdown_update = gr.update(choices=[])
                    tip_text = "⚠️ P2数据库不存在，加载后无法检测归一化数据。"
                    radio_update = gr.update(interactive=False, value="原始评分 (Raw)")

                return (project, db_info, card_html, path, path,
                        ranking_html, convergence_html, dropdown_update,
                        radio_update, tip_text)
            except Exception as e:
                card_html = _card("conflict", message=f"加载失败：{e}")
                empty_card = ("<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                              "border:1px solid var(--border-color-primary,#e0e0e0);"
                              "border-left-width:4px;border-left-color:#9e9e9e;opacity:0.7;"
                              "font-size:14px'>⬜ 加载失败</div>")
                return (None, "数据源：加载失败", card_html, "", "",
                        empty_card, empty_card, gr.update(choices=[]),
                        gr.update(interactive=False, value="原始评分 (Raw)"),
                        f"⚠️ 加载失败：{e}")

        btn_t4_load.click(
            fn=on_load_t4_project,
            inputs=[t4_project_path, t4_score_mode],
            outputs=[t4_project_state, db_status, t4_project_card, t2_project_path, t3_project_path,
                     t4_ranking_html, t4_convergence_html, t4_trial_selector,
                     t4_score_mode, t4_score_mode_tip],
        )

        # Task 13b: 刷新Top-N排名（联动评分模式）
        def on_t4_refresh(project, top_n, score_mode):
            """刷新Top-N排名和收敛分析（按当前评分模式）"""
            study = _get_p2_study(project)
            if study is None:
                empty_card = ("<div style='border-radius:6px;padding:12px 16px;margin:4px 0;"
                              "border:1px solid var(--border-color-primary,#e0e0e0);"
                              "border-left-width:4px;border-left-color:#9e9e9e;opacity:0.7;"
                              "font-size:14px'>⬜ P2数据库不存在</div>")
                return empty_card, empty_card, gr.update(choices=[])
            ranking_html, dropdown_update = _render_ranking_html(
                study, int(top_n), score_mode=score_mode
            )
            convergence_html = _render_convergence_html(study, score_mode=score_mode)
            return ranking_html, convergence_html, dropdown_update

        btn_t4_refresh.click(
            fn=on_t4_refresh,
            inputs=[t4_project_state, t4_top_n, t4_score_mode],
            outputs=[t4_ranking_html, t4_convergence_html, t4_trial_selector],
        )

        # ★ 评分模式切换：实时刷新排名与收敛
        def on_score_mode_change(project, top_n, new_mode):
            """切换评分模式时立刻重渲排名+收敛"""
            if project is None:
                return gr.update(), gr.update(), gr.update()
            return on_t4_refresh(project, top_n, new_mode)

        t4_score_mode.change(
            fn=on_score_mode_change,
            inputs=[t4_project_state, t4_top_n, t4_score_mode],
            outputs=[t4_ranking_html, t4_convergence_html, t4_trial_selector],
        )

        # ★ Top-N 数量变化时也按当前模式重渲
        def on_top_n_change(project, top_n, score_mode):
            if project is None:
                return gr.update(), gr.update(), gr.update()
            return on_t4_refresh(project, top_n, score_mode)

        t4_top_n.change(
            fn=on_top_n_change,
            inputs=[t4_project_state, t4_top_n, t4_score_mode],
            outputs=[t4_ranking_html, t4_convergence_html, t4_trial_selector],
        )

        # Task 14: 读取所选Trial参数
        def on_t4_load_trial(project, selection):
            """读取所选Trial的参数和指标"""
            if project is None or not selection:
                return None, "请先加载项目并选择Trial"
            try:
                study = _get_p2_study(project)
                if study is None:
                    return None, "P2数据库不存在"

                # 从selection解析Trial编号（用遍历查找，避免删除Trial后索引越界）
                trial_num = int(selection.split("(")[0].replace("Trial", "").strip())
                trial = next((t for t in study.trials if t.number == trial_num), None)
                if trial is None:
                    return None, f"Trial #{trial_num} 未找到（可能已被清理）"

                # 参数DataFrame
                rows = []
                for name, val in trial.params.items():
                    p = ALL_PARAMS.get(name, {})
                    rows.append({
                        "参数名": name,
                        "值": val,
                        "范围下限": p.get("low", p.get("choices", "-")),
                        "范围上限": p.get("high", "-"),
                        "组": p.get("group", "-"),
                    })
                df = pd.DataFrame(rows)

                # 指标摘要
                m = trial.user_attrs if hasattr(trial, 'user_attrs') else {}
                score = -trial.value if trial.value is not None else 0
                ic = m.get("val_ic", 0)
                icir = m.get("val_icir", 0)
                ir6m = m.get("val_rolling6m_ir", 0)
                pos = m.get("pct_positive_excess", 0)
                ja = m.get("val_jensen_alpha", 0)
                ar = m.get("val_appraisal_ratio", 0)
                upcap = m.get("up_capture_ratio", 0)
                cap = m.get("capture_ratio", 0)
                pr = m.get("penalized_rate", 0)
                beta = m.get("val_beta", 0)
                summary = (
                    f"**评分={score:.4f}** · IC={ic:.4f} · ICIR={icir:.4f} · "
                    f"6M_IR={ir6m:.4f} · 月度超额胜率={pos:.2%} · "
                    f"Jα={ja:.4f} · AR={ar:.4f} · "
                    f"上行捕获={upcap:.2f} · 综合捕获={cap:.2f} · "
                    f"降权率={pr:.2%} · β={beta:.3f}"
                )
                return df, summary
            except Exception as e:
                return None, f"读取失败：{e}"

        btn_t4_load_trial.click(
            fn=on_t4_load_trial,
            inputs=[t4_project_state, t4_trial_selector],
            outputs=[t4_selected_params_df, t4_selected_metrics],
        )

        # Task 15: 写回config.yaml（使用所选Trial）
        def on_write_config(project, selection):
            """写回config.yaml（使用所选Trial参数）"""
            if project is None or not selection:
                return "请先加载项目并选择Trial"
            try:
                study = _get_p2_study(project)
                if study is None:
                    return "P2数据库不存在"

                trial_num = int(selection.split("(")[0].replace("Trial", "").strip())
                trial = study.trials[trial_num]
                best_params = dict(trial.params)
                lgbm_params, xgbm_params, feature_params, lgbm_weight, window_params = (
                    assemble_params(best_params)
                )

                cfg_path = "config/config.yaml"
                # 备份
                shutil.copy2(cfg_path, cfg_path + ".bak")

                # 读取
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)

                # 更新 m2.lgbm
                if "m2" not in cfg:
                    cfg["m2"] = {}
                if "lgbm" not in cfg["m2"]:
                    cfg["m2"]["lgbm"] = {}
                cfg["m2"]["lgbm"].update(lgbm_params)

                # 更新 m2.xgb
                if "xgb" not in cfg["m2"]:
                    cfg["m2"]["xgb"] = {}
                cfg["m2"]["xgb"].update(xgbm_params)

                # 更新 m2.feature_store
                if "feature_store" not in cfg["m2"]:
                    cfg["m2"]["feature_store"] = {}
                cfg["m2"]["feature_store"].update(feature_params)

                # 更新 m2.ensemble.lgbm_weight
                if "ensemble" not in cfg["m2"]:
                    cfg["m2"]["ensemble"] = {}
                cfg["m2"]["ensemble"]["lgbm_weight"] = lgbm_weight

                # 更新 rolling.train_months
                if "rolling" not in cfg:
                    cfg["rolling"] = {}
                cfg["rolling"]["train_months"] = window_params.get("train_months", 36)

                # 同步 active_scheme（从P2配置读取）
                p2_cfg = load_p2_config(project).get("config", {})
                scheme = p2_cfg.get("scheme")
                if scheme and "data" in cfg and "neutralization" in cfg.get("data", {}):
                    cfg["data"]["neutralization"]["active_scheme"] = scheme

                # 写回
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

                return (f"✅ 已写回config.yaml（备份：config.yaml.bak）\n"
                        f"lgbm: {len(lgbm_params)}个参数\n"
                        f"xgb: {len(xgbm_params)}个参数\n"
                        f"feature: {len(feature_params)}个参数\n"
                        f"lgbm_weight: {lgbm_weight}\n"
                        f"train_months: {window_params.get('train_months', 36)}\n"
                        f"scheme: {scheme or '未同步'}")
            except Exception as e:
                return f"❌ 写回失败：{e}"

        btn_write_config.click(
            fn=on_write_config,
            inputs=[t4_project_state, t4_trial_selector],
            outputs=[log_box_deploy],
        )

        # Task 16: 触发M2+M4完整重跑（会话级安全）
        def on_run_full(project, selection, session_state):
            """触发M2+M4完整重跑（每个会话使用独立 queue）"""
            if project is None or not selection:
                return "请先加载项目并选择Trial", gr.update(interactive=True), session_state

            # 内存检查
            try:
                mem = psutil.virtual_memory()
                if mem.available < 3 * 1024**3:
                    return (f"❌ 可用内存不足（{mem.available/1024**3:.1f}GB < 3GB），"
                            f"请关闭其他程序"), gr.update(interactive=True), session_state
            except Exception:
                pass

            # ★ 懒加载创建本会话专属队列
            if session_state is None:
                session_state = {"queue": None, "running": False, "log_buffer": []}
            if session_state.get("queue") is None:
                session_state["queue"] = queue.Queue()
            session_queue = session_state["queue"]
            session_state["running"] = True
            session_state["log_buffer"] = []

            def _run(q, state):
                try:
                    study = _get_p2_study(project)
                    trial_num = int(selection.split("(")[0].replace("Trial", "").strip())
                    trial_obj = next((t for t in study.trials if t.number == trial_num), None)
                    if trial_obj is None:
                        q.put(f"❌ Trial #{trial_num} 未找到")
                        state["running"] = False
                        return
                    best_params = dict(trial_obj.params)
                    # 从P2配置读取scheme，而非project dict（project dict无scheme键）
                    try:
                        scheme = load_p2_config(project)["config"].get("scheme", "scheme_b")
                    except Exception:
                        scheme = "scheme_b"

                    result = run_full_backtest(
                        best_params=best_params,
                        scheme=scheme,
                        include_no_penalty=False,
                        progress_callback=lambda msg: q.put(msg),
                    )

                    q.put(f"✅ 回测完成！报告路径：{result.get('report_path', '未知')}")
                    state["running"] = False
                except Exception as e:
                    q.put(f"❌ 回测失败：{e}")
                    state["running"] = False

            thread = threading.Thread(
                target=_run, args=(session_queue, session_state), daemon=True
            )
            thread.start()
            return "⏳ 回测已启动，请等待...", gr.update(interactive=False), session_state

        btn_run_full.click(
            fn=on_run_full,
            inputs=[t4_project_state, t4_trial_selector, t4_session_state],
            outputs=[log_box_deploy, btn_run_full, t4_session_state],
        )

        # Tab4 定时器刷新日志（会话级：仅读取本会话队列）
        try:
            t4_timer = gr.Timer(value=2.0, active=False)

            def on_t4_timer_tick(session_state):
                """定时刷新Tab4日志（仅读取本会话队列，绝不串流）"""
                if session_state is None:
                    return gr.update()
                q = session_state.get("queue")
                if q is None:
                    return gr.update()

                msgs = []
                while True:
                    try:
                        msgs.append(q.get_nowait())
                    except queue.Empty:
                        break
                    except Exception:
                        break

                if not msgs:
                    return gr.update()

                # 追加到 buffer 防止被后续 tick 重复处理
                session_state["log_buffer"].extend(msgs)
                return "\n".join(session_state["log_buffer"])

            t4_timer.tick(
                fn=on_t4_timer_tick,
                inputs=[t4_session_state],
                outputs=[log_box_deploy],
            )

            # 启动回测时激活定时器
            btn_run_full.click(
                fn=lambda: gr.update(active=True),
                outputs=[t4_timer],
            )
        except (AttributeError, TypeError):
            pass  # Gradio版本不支持Timer

        # Task 17: 打开回测报告
        def on_open_report(report_path):
            """打开回测报告"""
            if not report_path or not os.path.exists(report_path):
                return f"报告文件不存在：{report_path}"
            try:
                import webbrowser
                webbrowser.open(f"file:///{os.path.abspath(report_path)}")
                return f"已打开报告：{report_path}"
            except Exception:
                return f"无法自动打开，请手动打开：{os.path.abspath(report_path)}"

        btn_open_report.click(
            fn=on_open_report,
            inputs=[report_path_box],
            outputs=[log_box_deploy],
        )

    return demo


def _memory_watchdog():
    """内存监控守护线程：每分钟检查一次，超13GB触发内存归还"""
    while True:
        time.sleep(60)
        try:
            mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
            if mem > 13.0:
                logger.critical(
                    f"内存严重超限：{mem:.1f}GB，"
                    f"程序可能被Windows强制终止"
                )
                release_memory_to_os()
        except Exception:
            pass

_wd = threading.Thread(target=_memory_watchdog, daemon=True)
_wd.start()


if __name__ == "__main__":
    def handle_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        try:
            logger.critical(
                "未捕获的异常导致程序退出：\n" +
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            )
        except Exception:
            # 避免日志失败导致 cascade 异常
            print(
                "未捕获的异常导致程序退出：\n" +
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
                file=sys.stderr,
            )
    sys.excepthook = handle_exception

    def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        """检测端口是否被占用（不抛异常，纯查询）"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                return s.connect_ex((host, port)) == 0
            except Exception:
                return False

    def _launch_with_port_fallback(app_obj, host: str, base_port: int, max_tries: int = 10) -> int:
        """
        ★ 端口自动回退：7860-7869 范围逐个尝试，被占用则递增。
        返回实际使用的端口号；全部失败抛出最后异常。
        """
        last_exc: Exception = OSError("no_port_attempted")
        for offset in range(max_tries):
            port = base_port + offset
            if _is_port_in_use(port):
                print(
                    f"[M5 启动] 端口 {port} 被占用，"
                    f"尝试 {port + 1 if offset + 1 < max_tries else '最终'}...",
                    file=sys.stderr,
                )
                continue
            try:
                app_obj.launch(
                    server_name=host,
                    server_port=port,
                    inbrowser=(offset == 0),
                )
                return port
            except OSError as e:
                last_exc = e
                # 端口冲突（OSError from Gradio http_server）继续尝试下一个
                if "Cannot find empty port" in str(e) or e.errno in (98, 10048):
                    continue
                raise
        raise last_exc

    try:
        with open("config/config.yaml", encoding="utf-8") as _f:
            cfg = yaml.safe_load(_f)
        app = build_app()
        host = cfg.get("m5", {}).get("gradio", {}).get("server_name", "0.0.0.0")
        base_port = int(
            cfg.get("m5", {}).get("gradio", {}).get("server_port", 7860)
        )
        actual_port = _launch_with_port_fallback(app, host, base_port)
        logger.info(f"M5 优化器已启动：http://{host}:{actual_port}")
    except Exception as e:
        try:
            logger.critical(f"程序启动或运行时崩溃：{e}\n{traceback.format_exc()}")
        except Exception:
            print(
                f"程序启动或运行时崩溃：{e}\n{traceback.format_exc()}",
                file=sys.stderr,
            )
        input("程序已崩溃，按回车键退出...")  # ★ 保留窗口
