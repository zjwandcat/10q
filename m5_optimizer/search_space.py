# PEP 683: 使用 frozenset/MappingProxyType 替代可变 set/dict，避免 GIL refcount 开销
"""
M5搜索空间定义
定义30个搜索参数和30个因变量
"""
import types
from typing import Dict, List, Any, Tuple

ALL_PARAMS: Dict[str, Dict[str, Any]] = {
    # ── LightGBM参数（12个）────────────────────────────────────
    "lgbm_learning_rate": {
        "type": "float_log",
        "low": 0.01,
        "high": 0.1,
        "default": 0.05,
        "group": "lgbm"
    },
    "lgbm_n_estimators": {
        "type": "int",
        "low": 100,
        "high": 300,
        "default": 200,
        "group": "lgbm"
    },
    "lgbm_max_depth": {
        "type": "int",
        "low": 2,
        "high": 4,
        "default": 3,
        "group": "lgbm"
    },
    "lgbm_colsample_bytree": {
        "type": "float",
        "low": 0.1,
        "high": 0.3,
        "default": 0.2,
        "group": "lgbm"
    },
    "lgbm_reg_alpha": {
        "type": "float_log",
        "low": 0.05,
        "high": 3.0,
        "default": 0.3,
        "group": "lgbm"
    },
    "lgbm_reg_lambda": {
        "type": "float_log",
        "low": 0.5,
        "high": 10.0,
        "default": 2.0,
        "group": "lgbm"
    },
    "lgbm_min_split_gain": {
        "type": "float_log",
        "low": 0.001,
        "high": 0.1,
        "default": 0.01,
        "group": "lgbm"
    },
    "lgbm_lr_mode": {
        "type": "categorical",
        "choices": ["fixed", "decay"],
        "default": "fixed",
        "group": "lgbm"
    },
    "lgbm_decay_every": {
        "type": "int",
        "low": 30,
        "high": 80,
        "default": 50,
        "group": "lgbm"
    },
    "lgbm_decay_factor": {
        "type": "float",
        "low": 0.6,
        "high": 0.92,
        "default": 0.8,
        "group": "lgbm"
    },
    "lgbm_depth_mode": {
        "type": "categorical",
        "choices": ["fixed", "adaptive"],
        "default": "fixed",
        "group": "lgbm"
    },
    "lgbm_early_stopping_rounds": {
        "type": "int",
        "low": 20,
        "high": 50,
        "default": 30,
        "group": "lgbm"
    },

    # ── XGBoost参数（11个）────────────────────────────────────
    "xgb_learning_rate": {
        "type": "float_log",
        "low": 0.01,
        "high": 0.1,
        "default": 0.05,
        "group": "xgb"
    },
    "xgb_n_estimators": {
        "type": "int",
        "low": 100,
        "high": 300,
        "default": 200,
        "group": "xgb"
    },
    "xgb_max_depth": {
        "type": "int",
        "low": 2,
        "high": 4,
        "default": 4,
        "group": "xgb"
    },
    "xgb_colsample_bytree": {
        "type": "float",
        "low": 0.1,
        "high": 0.3,
        "default": 0.2,
        "group": "xgb"
    },
    "xgb_reg_alpha": {
        "type": "float_log",
        "low": 0.05,
        "high": 3.0,
        "default": 0.3,
        "group": "xgb"
    },
    "xgb_reg_lambda": {
        "type": "float_log",
        "low": 0.5,
        "high": 10.0,
        "default": 2.0,
        "group": "xgb"
    },
    "xgb_gamma": {
        "type": "float_log",
        "low": 0.001,
        "high": 0.2,
        "default": 0.01,
        "group": "xgb"
    },
    "xgb_lr_mode": {
        "type": "categorical",
        "choices": ["fixed", "decay"],
        "default": "fixed",
        "group": "xgb"
    },
    "xgb_decay_every": {
        "type": "int",
        "low": 30,
        "high": 80,
        "default": 50,
        "group": "xgb"
    },
    "xgb_decay_factor": {
        "type": "float",
        "low": 0.6,
        "high": 0.92,
        "default": 0.8,
        "group": "xgb"
    },
    "xgb_early_stopping_rounds": {
        "type": "int",
        "low": 20,
        "high": 50,
        "default": 30,
        "group": "xgb"
    },

    # ── Ensemble参数（1个）────────────────────────────────────
    "lgbm_weight": {
        "type": "float",
        "low": 0.3,
        "high": 0.7,
        "default": 0.5,
        "group": "ensemble"
    },

    # ── FeatureStore参数（5个）────────────────────────────────
    "min_valid_rate": {
        "type": "float",
        "low": 0.20,
        "high": 0.50,
        "default": 0.30,
        "group": "feature"
    },
    "max_corr": {
        "type": "float",
        "low": 0.80,
        "high": 0.97,
        "default": 0.95,
        "group": "feature"
    },
    "min_ic_abs": {
        "type": "float_log",
        "low": 0.002,
        "high": 0.02,
        "default": 0.003,
        "group": "feature"
    },
    "min_keep_factors": {
        "type": "int",
        "low": 50,
        "high": 120,
        "default": 60,
        "group": "feature"
    },
    "drop_short_term_noise": {
        "type": "categorical",
        "choices": [True, False],
        "default": False,
        "group": "feature"
    },

    # ── Window参数（1个）────────────────────────────────────
    "train_months": {
        "type": "int",
        "low": 52,
        "high": 60,
        "default": 56,
        "group": "window",
        "description": "滚动训练窗口月数（DEBUG: 临时收紧到 [52,60] 以加速 OOM 复现）"
    },
}

OBJECTIVE_VARS: List[Dict[str, Any]] = [
    # ★ 滑块上下限已根据 14B/15B/15E 实际数据分布重新校准
    # 原则：能覆盖典型 P1 数据范围 + 留 ~30% 边距供反推过滤
    {"name": "val_ic", "source": "m2", "direction": "max", "enabled": True, "label": "验证集IC ↑", "slider_min": 0.00, "slider_max": 0.15},
    {"name": "val_icir", "source": "m2", "direction": "max", "enabled": True, "label": "IC信息比率 ↑", "slider_min": 0.0, "slider_max": 5.0},
    {"name": "val_rolling6m_ir", "source": "m2", "direction": "max", "enabled": True, "label": "6月滚动IR ↑", "slider_min": 0.0, "slider_max": 0.5},
    {"name": "val_rolling6m_sortino", "source": "m2", "direction": "max", "enabled": True, "label": "6月滚动索提诺比率 ↑", "slider_min": 0.0, "slider_max": 5.0},
    {"name": "val_rolling6m_return", "source": "m2", "direction": "max", "enabled": True, "label": "6月滚动累计收益 ↑", "slider_min": -0.5, "slider_max": 1.0},
    {"name": "ic_gap_penalty", "source": "m2", "direction": "min", "enabled": True, "label": "IC过拟合惩罚 →0", "slider_min": 0.0, "slider_max": 0.5},
    {"name": "penalized_rate", "source": "m2", "direction": "min", "enabled": True, "label": "降权月份占比 ↓", "slider_min": 0.0, "slider_max": 1.0},
    {"name": "val_global_ir", "source": "m2", "direction": "max", "enabled": True, "label": "全局IR ↑", "slider_min": 0.0, "slider_max": 0.5},
    {"name": "val_annual_return", "source": "m2", "direction": "max", "enabled": True, "label": "年化收益率 ↑", "slider_min": -0.5, "slider_max": 2.0},
    {"name": "pct_positive_excess", "source": "m2", "direction": "max", "enabled": True, "label": "月度超额胜率 ↑", "slider_min": 0.0, "slider_max": 1.0},
    {"name": "ir_worst_quartile", "source": "m2", "direction": "max", "enabled": True, "label": "最差四分位IR ↑", "slider_min": -2.0, "slider_max": -0.5},
    {"name": "val_rolling6m_excess", "source": "m2", "direction": "max", "enabled": True, "label": "6月滚动超额收益均值 ↑", "slider_min": -0.05, "slider_max": 0.05},
    {"name": "val_rolling6m_excess_ann", "source": "m2", "direction": "max", "enabled": True, "label": "年化滚动超额收益 ↑", "slider_min": -0.5, "slider_max": 0.5},
    {"name": "val_jensen_alpha", "source": "m2", "direction": "max", "enabled": True, "label": "Jensen's Alpha (年化) ↑", "slider_min": -0.20, "slider_max": 0.20},
    {"name": "val_appraisal_ratio", "source": "m2", "direction": "max", "enabled": True, "label": "Appraisal Ratio ↑", "slider_min": 0.0, "slider_max": 5.0},
    {"name": "val_beta", "source": "m2", "direction": "min", "enabled": True, "label": "Beta →1", "slider_min": 0.0, "slider_max": 2.0},
    # ★ P1 自适应型 2.0：仅保留 IC稳定性（std），与 IC信息比率（ICIR = mean/std）含义不同，不可互替
    {"name": "val_ic_stability", "source": "m2", "direction": "min", "enabled": True, "label": "IC稳定性 (std) →0", "slider_min": 0.0, "slider_max": 0.3},

    {"name": "stress_2008_excess", "source": "m2", "direction": "max", "enabled": False, "label": "2008超额 ↑", "slider_min": -2.0, "slider_max": 2.0},
    {"name": "stress_2015_mdd", "source": "m2", "direction": "min", "enabled": False, "label": "2015最大回撤 ↓", "slider_min": -2.0, "slider_max": 2.0},
    {"name": "stress_2022_excess", "source": "m2", "direction": "max", "enabled": False, "label": "2022超额 ↑", "slider_min": -2.0, "slider_max": 2.0},

    {"name": "cagr", "source": "m4", "direction": "max", "enabled": True, "label": "复合年化增长率 ↑", "slider_min": -0.5, "slider_max": 2.0},
    {"name": "monthly_win_rate", "source": "m4", "direction": "max", "enabled": True, "label": "月度胜率 ↑", "slider_min": 0.0, "slider_max": 1.0},
    {"name": "downside_volatility", "source": "m4", "direction": "min", "enabled": True, "label": "下行波动率 ↓", "slider_min": 0.0, "slider_max": 0.5},
    {"name": "upside_volatility", "source": "m4", "direction": "max", "enabled": True, "label": "上行波动率 ↑", "slider_min": 0.0, "slider_max": 0.5},
    {"name": "volatility_ratio", "source": "m4", "direction": "max", "enabled": True, "label": "波动率比率 ↑", "slider_min": 0.0, "slider_max": 3.0},
    {"name": "var_95", "source": "m4", "direction": "min", "enabled": True, "label": "95% VaR ↓", "slider_min": -0.3, "slider_max": 0.0},
    {"name": "cvar_95", "source": "m4", "direction": "min", "enabled": True, "label": "95% CVaR ↓", "slider_min": -0.5, "slider_max": 0.0},
    {"name": "skewness", "source": "m4", "direction": "max", "enabled": True, "label": "收益偏度 ↑", "slider_min": -2.0, "slider_max": 2.0},
    {"name": "kurtosis", "source": "m4", "direction": "min", "enabled": True, "label": "峰度 →0", "slider_min": 0.0, "slider_max": 10.0},
    {"name": "pain_index", "source": "m4", "direction": "min", "enabled": True, "label": "痛苦指数 ↓", "slider_min": 0.0, "slider_max": 0.5},
    {"name": "omega_ratio", "source": "m4", "direction": "max", "enabled": True, "label": "欧米伽比率 ↑", "slider_min": 0.0, "slider_max": 3.0},
    {"name": "burke_ratio", "source": "m4", "direction": "max", "enabled": True, "label": "伯克比率 ↑", "slider_min": -2.0, "slider_max": 2.0},
    {"name": "martin_ratio", "source": "m4", "direction": "max", "enabled": True, "label": "马丁比率 ↑", "slider_min": -2.0, "slider_max": 2.0},
    {"name": "tail_ratio", "source": "m4", "direction": "max", "enabled": True, "label": "尾部比率 ↑", "slider_min": 0.0, "slider_max": 3.0},
    # ★ 捕获比三件套：用户决策 — 只看 up_capture_ratio（越大越好），
    #   down_capture_ratio / capture_ratio 标记为 enabled=False，
    #   不再进入 P1/P2 权重滑块，不再贡献到贝叶斯优化 score。
    #   但仍写入 user_attrs，且 Top-5/Tab4 排名中作为信息继续显示。
    {"name": "up_capture_ratio", "source": "m4", "direction": "max", "enabled": True,  "label": "上行捕获比 ↑",   "slider_min": 0.0, "slider_max": 2.0},
    {"name": "down_capture_ratio", "source": "m4", "direction": "min", "enabled": False, "label": "下行捕获比 ↓（已禁用）", "slider_min": 0.0, "slider_max": 2.0},
    {"name": "capture_ratio", "source": "m4", "direction": "max", "enabled": False, "label": "综合捕获比 ↑（已禁用）", "slider_min": -2.0, "slider_max": 2.0},
]

DEFAULT_PARAMS: Dict[str, Any] = {
    name: pdef["default"] for name, pdef in ALL_PARAMS.items()
}

PARAM_PASS_THROUGH = {
    "lgbm_lr_mode": 'lgbm_params["lr_mode"]',
    "lgbm_decay_every": 'lgbm_params["decay_every"]',
    "lgbm_decay_factor": 'lgbm_params["decay_factor"]',
    "lgbm_depth_mode": 'lgbm_params["depth_mode"]',
    "lgbm_early_stopping_rounds": 'lgbm_params["early_stopping_rounds"]',
    "xgb_lr_mode": 'xgbm_params["lr_mode"]',
    "xgb_decay_every": 'xgbm_params["decay_every"]',
    "xgb_decay_factor": 'xgbm_params["decay_factor"]',
    "xgb_early_stopping_rounds": 'xgbm_params["early_stopping_rounds"]',
}

# PEP 683: MappingProxyType 替代可变 dict，避免 GIL refcount 开销
SPECIAL_LGBM_KEYS = types.MappingProxyType({
    "lgbm_lr_mode":                "lr_mode",
    "lgbm_decay_every":            "decay_every",
    "lgbm_decay_factor":           "decay_factor",
    "lgbm_depth_mode":             "depth_mode",
    "lgbm_early_stopping_rounds":  "early_stopping_rounds",
})
SPECIAL_XGB_KEYS = types.MappingProxyType({
    "xgb_lr_mode":                "lr_mode",
    "xgb_decay_every":            "decay_every",
    "xgb_decay_factor":           "decay_factor",
    "xgb_early_stopping_rounds":  "early_stopping_rounds",
})


def assemble_params(
    sampled: Dict[str, Any],
) -> Tuple[Dict, Dict, Dict, float, Dict]:
    """
    将 30 个扁平化搜索参数按 group 拆装回 lgbm / xgbm / feature / ensemble / window 5 组。

    返回：
        (lgbm_params, xgbm_params, feature_params, lgbm_weight, window_params)
    """
    lgbm_params: Dict[str, Any] = {}
    xgbm_params: Dict[str, Any] = {}
    feature_params: Dict[str, Any] = {}
    window_params: Dict[str, Any] = {}
    lgbm_weight: float = 0.5

    for name, value in sampled.items():
        if name not in ALL_PARAMS:
            continue
        pdef = ALL_PARAMS[name]
        group = pdef["group"]

        match group:
            case "lgbm":
                if name in SPECIAL_LGBM_KEYS:
                    lgbm_params[SPECIAL_LGBM_KEYS[name]] = value
                else:
                    lgbm_params[name.replace("lgbm_", "", 1)] = value
            case "xgb":
                if name in SPECIAL_XGB_KEYS:
                    xgbm_params[SPECIAL_XGB_KEYS[name]] = value
                else:
                    xgbm_params[name.replace("xgb_", "", 1)] = value
            case "ensemble":
                lgbm_weight = value
            case "feature":
                feature_params[name] = value
            case "window":
                window_params[name] = value

    return lgbm_params, xgbm_params, feature_params, lgbm_weight, window_params

assert len(ALL_PARAMS) == 30, f"参数数量:{len(ALL_PARAMS)}"
assert len(OBJECTIVE_VARS) == 37, f"因变量数量错误:{len(OBJECTIVE_VARS)}"

# ★ 因变量滑块范围查询表（供app.py的Tab2 filter_sliders使用）
METRIC_BOUNDS = {
    v["name"]: (v.get("slider_min", -2), v.get("slider_max", 2))
    for v in OBJECTIVE_VARS
}

# ★ 全量窗口数动态计算
# 数据: 200701~202512 = 228个月
# WINDOW_SIZE = train_months + 12(valid) + 1(test)
# window_count = total_months - WINDOW_SIZE + 1
TOTAL_DATA_MONTHS = 228


def calc_window_count(train_months: int = None) -> int:
    """根据train_months动态计算全量窗口数"""
    tm = train_months if train_months is not None else ALL_PARAMS["train_months"]["default"]
    window_size = tm + 12 + 1
    return max(TOTAL_DATA_MONTHS - window_size + 1, 0)


# ★ 评分归一化配置表（NORM_CONFIG）
# ──────────────────────────────────────────────────────────────────────
# 用途：为 M5 评分系统提供因变量归一化映射，解决量级差异导致的评分失真问题。
# 线程安全：本字典在模块导入时一次性定义，后续只读，不修改。
# 跨项目可比性前提：所有项目共享同一 NORM_CONFIG + 同一套 objective_weights。
# ──────────────────────────────────────────────────────────────────────
# 三种归一化方法：
#   linear      → f(x) = (clip(x, lo, hi) - lo) / (hi - lo + eps) - 0.5
#                 居中到 [-0.5, 0.5]，适用于值域有明确边界的率指标
#   tanh        → f(x) = tanh(x / scale)
#                 压缩到 (-1, 1)，适用于集中0附近但偶尔大幅偏离的指标
#   signed_log  → f(x) = sign(x) * log(1 + |x| + eps)
#                 压缩长尾分布，适用于捕获比、收益率等剧烈波动指标
# ──────────────────────────────────────────────────────────────────────
NORM_CONFIG = {
    # === A 类：值域有明确边界的率指标 → linear（居中到 [-0.5, 0.5]）===
    "val_ic":                  {"method": "linear",  "bounds": [-0.05, 0.05]},
    "pct_positive_excess":     {"method": "linear",  "bounds": [0.30, 0.70]},
    "penalized_rate":          {"method": "linear",  "bounds": [0.0, 0.50]},
    "ic_gap_penalty":          {"method": "linear",  "bounds": [0.0, 0.30]},
    "monthly_win_rate":        {"method": "linear",  "bounds": [0.30, 0.70]},
    # === B 类：IR/Sortino 类（集中 0 附近、有时大幅偏离）→ tanh ===
    "val_icir":                {"method": "tanh",    "scale": 2.0},
    "val_rolling6m_ir":        {"method": "tanh",    "scale": 1.0},
    "val_global_ir":           {"method": "tanh",    "scale": 1.5},
    "val_rolling6m_sortino":   {"method": "tanh",    "scale": 2.0},
    "ir_worst_quartile":       {"method": "tanh",    "scale": 2.0},
    "val_jensen_alpha":        {"method": "tanh",    "scale": 0.05},
    "val_appraisal_ratio":     {"method": "tanh",    "scale": 0.5},
    # ★ Beta 集中 1.0 附近 → tanh scale=0.5（让 [0.5, 1.5] 区间灵敏度合理）
    "val_beta":                {"method": "tanh",    "scale": 0.5},
    # ★ P1 自适应型 2.0 仅保留 IC稳定性（std）
    "val_ic_stability":        {"method": "tanh",    "scale": 0.05},
    # === C 类：长尾/剧烈波动（捕获比、收益率）→ signed_log ===
    "capture_ratio":           {"method": "signed_log"},
    "up_capture_ratio":        {"method": "signed_log"},
    "down_capture_ratio":      {"method": "signed_log"},
    "val_annual_return":       {"method": "signed_log"},
    "val_rolling6m_return":    {"method": "signed_log"},
    "val_rolling6m_excess":    {"method": "signed_log"},
    "val_rolling6m_excess_ann": {"method": "signed_log"},
}


import logging

_logger = logging.getLogger("m5.search_space")
_logger.debug("search_space验证通过")
