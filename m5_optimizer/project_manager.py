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
M5 项目管理器
基于文件夹的P1/P2项目管理
"""
import os
import json
import logging
from datetime import datetime
from typing import Optional

import optuna

from m5_optimizer.search_space import calc_window_count

logger = logging.getLogger("m5.project_manager")


# ━━━ 异常 Trial 检测规则配置 ━━━
# 设计原则：宁可漏杀不可误杀
# - Layer 1（DB 字段级）：全部启用，成本低、风险低
# - Layer 2（指标一致性）：全部启用，使用宽阈值
# - Layer 3（硬阈值）：关闭。"不可能"策略可能有合理计算、不同市场时段边界不同、
#   幸存者偏差，这三类不该被自动清掉
# - Layer 4（窗口级）：关闭。老数据里的窗口 NaN 不应被误判
DETECTION_RULES = {
    # ── Layer 1：DB 字段级体检（低风险，全部启用）──
    "state_abnormal":           True,   # FAIL / RUNNING / WAITING
    "value_below_-999":         True,   # COMPLETE 但 value <= -999
    "value_nan_inf":            True,   # value 自身是 NaN / inf
    "trial_value_missing":      True,   # COMPLETE 但 trial_value 缺失
    "datetime_complete_null":   True,   # COMPLETE 但 datetime_complete 为 NULL
    "heartbeat_stale":          True,   # RUNNING 的 heartbeat 超过阈值
    "intermediate_values_nan":  True,   # trial_intermediate_values 含 NaN/inf
    "empty_user_attrs":         True,   # 完全无 user_attrs
    "all_required_metrics_missing": True,  # 必需指标全 None
    "critical_metrics_nan":     True,   # 关键指标 NaN / inf / 不可解析

    # ── Layer 2：指标内部一致性（启用，宽阈值）──
    "ic_icir_sign":             True,   # IC / ICIR 符号一致（仅 |IC| 大时）
    "rolling_vs_global_ir":     True,   # rolling6m_ir vs global_ir 比例

    # ── Layer 3：硬阈值（全部关闭）──
    "metric_out_of_range":      False,  # val_ic / val_icir 越界
    "pct_positive_excess_low":  False,  # 偏低超额胜率
    "low_excess_return":        False,  # 无超额收益（贝叶斯优化要多样性）

    # ── Layer 4：窗口级（全部关闭）──
    "window_failure_rate":      False,  # 窗口失败率
    "window_nan_misjudge":      False,  # 老数据窗口 NaN
}

# 阈值常量（安全默认值，可后续放 config）
IC_SIGN_CHECK_THRESHOLD = 0.1        # |IC| > 此值才检查符号
IR_RATIO_TOLERANCE = 10.0            # rolling6m_ir / global_ir 上限（10× 极宽）
HEARTBEAT_STALE_SEC = 3600           # heartbeat 超过 1 小时视为僵死
INTERMEDIATE_NAN_QUERY = True        # 是否检查 trial_intermediate_values


def _get_param_ranges_from_config(project: dict, phase: str) -> dict:
    """
    从配置文件读取 param_ranges（用于 Layer 1 参数越界检查，可选）
    失败时返回空 dict，调用方应跳过参数越界检查
    """
    try:
        cfg_path = project["p1_config_path"] if phase == "p1" else project["p2_config_path"]
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("config", {}).get("param_ranges", {}) or {}
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError):
        return {}


def _default_p1_config(project_name: str) -> dict:
    """生成默认P1配置（默认全自适应型权重）"""
    return {
        "version": "1.0",
        "phase": "phase1",
        "project_name": project_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "study_name": "phase1_global",
        "config": {
            "scheme": "scheme_d",
            "fast_mode": True,
            "window_count": 155,
            "objective_weights": {
                "val_icir": 0.18,
                "ic_gap_penalty": 0.22,
                "pct_positive_excess": 0.18,
                "val_rolling6m_ir": 0.18,
                "ir_worst_quartile": 0.05,
                "val_ic": 0.05,
                "penalized_rate": 0.05,
                "cvar_95": 0.05,
                "pain_index": 0.04,
            },
            "active_params": None,
            "param_ranges": {},
            "enable_normalization": True,   # ★ v2.0 默认开启
        },
        "trials_history": [],
        "trials_total": 0,
    }


def _default_p2_config(project_name: str) -> dict:
    """生成默认P2配置（默认全自适应型权重）"""
    return {
        "version": "1.0",
        "phase": "phase2",
        "project_name": project_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "study_name": "phase2_local",
        "config": {
            "scheme": "scheme_d",
            "fast_mode": True,
            "objective_weights": {
                "val_icir": 0.18,
                "ic_gap_penalty": 0.22,
                "pct_positive_excess": 0.18,
                "val_rolling6m_ir": 0.18,
                "ir_worst_quartile": 0.05,
                "val_ic": 0.05,
                "penalized_rate": 0.05,
                "cvar_95": 0.05,
                "pain_index": 0.04,
            },
            "active_params": None,
            "param_ranges": {},
            "enable_normalization": True,   # ★ v2.0 默认开启
        },
        "init_from_p1_best": True,
        "trials_history": [],
        "trials_total": 0,
    }


def _build_project_dict(project_root: str) -> dict:
    """根据项目根路径构建project dict（路径+名称+study名）"""
    project_name = os.path.basename(project_root.rstrip("/\\"))
    p1_dir = os.path.join(project_root, "p1")
    p2_dir = os.path.join(project_root, "p2")
    return {
        "project_name": project_name,
        "project_root": project_root,
        "p1_config_path": os.path.join(p1_dir, f"{project_name}_p1_config.json"),
        "p1_db_path": os.path.join(p1_dir, f"{project_name}_p1_study.db"),
        "p2_config_path": os.path.join(p2_dir, f"{project_name}_p2_config.json"),
        "p2_db_path": os.path.join(p2_dir, f"{project_name}_p2_study.db"),
        "p1_study_name": "phase1_global",
        "p2_study_name": "phase2_local",
    }


def _has_config_files(project: dict) -> bool:
    """检查项目目录中是否已有配置文件"""
    return (
        os.path.exists(project["p1_config_path"])
        or os.path.exists(project["p2_config_path"])
    )


def _create_project_files(project: dict):
    """创建项目目录和配置文件"""
    p1_dir = os.path.dirname(project["p1_config_path"])
    p2_dir = os.path.dirname(project["p2_config_path"])
    os.makedirs(p1_dir, exist_ok=True)
    os.makedirs(p2_dir, exist_ok=True)

    if not os.path.exists(project["p1_config_path"]):
        with open(project["p1_config_path"], "w", encoding="utf-8") as f:
            json.dump(_default_p1_config(project["project_name"]), f, ensure_ascii=False, indent=2)

    if not os.path.exists(project["p2_config_path"]):
        with open(project["p2_config_path"], "w", encoding="utf-8") as f:
            json.dump(_default_p2_config(project["project_name"]), f, ensure_ascii=False, indent=2)


def init_project(project_root: str) -> dict:
    """
    初始化或加载项目。
    返回 {"status": "created"|"loaded"|"conflict",
           "project": project_dict,
           "message": str}

    逻辑：
    1. project_root不存在 → 创建文件夹及p1/p2子目录，生成空配置文件，status="created"
    2. project_root存在且为空文件夹 → 同上，status="created"
    3. project_root存在且有内容，但没有配置文件 → status="conflict"，
       message="文件夹非空且不含项目配置，请确认是否覆盖"，不创建任何文件
    4. project_root存在且有配置文件 → 加载，status="loaded"
    """
    project = _build_project_dict(project_root)

    if not os.path.exists(project_root):
        os.makedirs(project_root, exist_ok=True)
        _create_project_files(project)
        return {
            "status": "created",
            "project": project,
            "message": f"项目 {project['project_name']} 已创建",
        }

    # project_root存在
    if _has_config_files(project):
        return {
            "status": "loaded",
            "project": project,
            "message": f"项目 {project['project_name']} 已加载",
        }

    # 没有配置文件，检查是否为空文件夹
    contents = os.listdir(project_root)
    if len(contents) == 0:
        _create_project_files(project)
        return {
            "status": "created",
            "project": project,
            "message": f"项目 {project['project_name']} 已创建",
        }

    # 非空且无配置文件 → 冲突
    return {
        "status": "conflict",
        "project": project,
        "message": "文件夹非空且不含项目配置，请确认是否覆盖",
    }


def force_init_project(project_root: str) -> dict:
    """用户确认覆盖后调用，强制创建（不删除DB，只创建配置文件）"""
    project = _build_project_dict(project_root)
    os.makedirs(project_root, exist_ok=True)
    _create_project_files(project)
    return {
        "status": "created",
        "project": project,
        "message": f"项目 {project['project_name']} 已强制创建",
    }


def load_p1_config(project: dict) -> dict:
    """加载P1配置文件"""
    path = project["p1_config_path"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_p2_config(project: dict) -> dict:
    """加载P2配置文件"""
    path = project["p2_config_path"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_p1_config(project: dict, config_data: dict):
    """保存P1配置，DB存在时拒绝修改"""
    if os.path.exists(project["p1_db_path"]):
        return False, "P1数据库已存在，不允许修改配置。请先初始化P1后再修改。"
    path = project["p1_config_path"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    return True, "配置已保存"


def save_p2_config(project: dict, config_data: dict):
    """保存P2配置，P2 DB存在时拒绝修改"""
    if os.path.exists(project["p2_db_path"]):
        return False, "P2数据库已存在，不允许修改配置。请先初始化P2后再修改。"
    path = project["p2_config_path"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    return True, "配置已保存"


def export_p2_config_from_p1(
    project: dict,
    param_ranges: dict,
    objective_weights: dict,
    active_params: list,
) -> dict:
    """
    P1结果分析导出P2配置：
    1. 从P1 DB读取best_params作为init_params
    2. 将param_ranges写入P2配置
    3. 写入P2配置文件（如果P2 DB已存在则拒绝）
    返回生成的P2配置dict
    """
    if os.path.exists(project["p2_db_path"]):
        raise RuntimeError("P2数据库已存在，不允许覆盖P2配置。请先重置P2。")

    p2_config = _default_p2_config(project["project_name"])

    # 从P1 DB读取best_params
    p1_study = get_p1_study(project)
    if p1_study is not None and p1_study.best_trial is not None:
        p2_config["init_from_p1_best"] = True
    else:
        p2_config["init_from_p1_best"] = False

    p2_config["config"]["param_ranges"] = param_ranges
    p2_config["config"]["objective_weights"] = objective_weights
    p2_config["config"]["active_params"] = active_params

    # 写入P2配置文件
    path = project["p2_config_path"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(p2_config, f, ensure_ascii=False, indent=2)

    return p2_config


def get_p1_study(project: dict) -> Optional[optuna.Study]:
    """加载P1 study，DB不存在返回None"""
    db = project["p1_db_path"]
    if not os.path.exists(db):
        return None
    # 注意：不在 load_study 前调用 _ensure_wal，避免裸 sqlite3 连接与 SQLAlchemy 冲突
    return optuna.load_study(
        study_name=project["p1_study_name"],
        storage=f"sqlite:///{db}",
    )


def get_p2_study(project: dict) -> Optional[optuna.Study]:
    """加载P2 study，DB不存在返回None"""
    db = project["p2_db_path"]
    if not os.path.exists(db):
        return None
    # 注意：不在 load_study 前调用 _ensure_wal，避免裸 sqlite3 连接与 SQLAlchemy 冲突
    return optuna.load_study(
        study_name=project["p2_study_name"],
        storage=f"sqlite:///{db}",
    )


def get_study_stats(study: optuna.Study) -> dict:
    """
    返回Trial统计，不依赖配置文件里的trials_total字段：
    直接从DB实时读取，最准确。

    异常判定（与 cleanup_bad_trials 保持一致，只做 Layer 1 + Layer 2）：
    - FAIL / RUNNING / WAITING 状态
    - COMPLETE 但 value<=-999.0 / NaN / inf
    - COMPLETE 但 trial 完全没有 user_attrs
    - COMPLETE 但关键指标全部缺失/全为 None
    - COMPLETE 但关键指标为 NaN / inf
    - COMPLETE 但 IC/ICIR 符号不一致（仅 |IC| > 0.1）
    - COMPLETE 但 rolling6m_ir / global_ir 比例 > 10×

    不做硬阈值（Layer 3），不做窗口级（Layer 4）—— 见 cleanup_bad_trials 文档。
    """
    import math

    complete, fail, running, abnormal = 0, 0, 0, 0

    REQUIRED_METRICS = [
        "val_ic", "val_icir", "val_rolling6m_ir", "val_global_ir",
    ]
    CRITICAL_METRICS = [
        "val_ic", "val_icir", "val_rolling6m_ir",
        "pct_positive_excess", "val_global_ir",
    ]

    for t in study.trials:
        s = t.state
        if s == optuna.trial.TrialState.COMPLETE:
            v = t.value
            # L1.2 / L1.3: value 异常
            if v is None or (isinstance(v, float)
                             and (math.isnan(v) or math.isinf(v))):
                abnormal += 1
            elif isinstance(v, float) and v <= -999.0:
                abnormal += 1
            else:
                attrs = t.user_attrs or {}
                is_bad = False

                # L1.8: 完全无 user_attrs
                if not attrs:
                    is_bad = True
                else:
                    # L1.9: 关键指标全缺失
                    missing = [m for m in REQUIRED_METRICS
                               if attrs.get(m) is None]
                    if len(missing) == len(REQUIRED_METRICS):
                        is_bad = True
                    else:
                        # L1.10: 关键指标 NaN/inf/不可解析
                        for metric in CRITICAL_METRICS:
                            raw_val = attrs.get(metric)
                            if raw_val is None:
                                continue
                            try:
                                mv = float(raw_val)
                                if math.isnan(mv) or math.isinf(mv):
                                    is_bad = True
                                    break
                            except (ValueError, TypeError):
                                is_bad = True
                                break
                        # L2.1: IC/ICIR 符号一致
                        if not is_bad:
                            ic = attrs.get("val_ic")
                            icir = attrs.get("val_icir")
                            if (ic is not None and icir is not None):
                                try:
                                    ic_f, icir_f = float(ic), float(icir)
                                    if (not math.isnan(ic_f)
                                            and not math.isinf(ic_f)
                                            and not math.isnan(icir_f)
                                            and not math.isinf(icir_f)
                                            and abs(ic_f) > IC_SIGN_CHECK_THRESHOLD):
                                        if (ic_f > 0) != (icir_f > 0):
                                            is_bad = True
                                except (ValueError, TypeError):
                                    pass
                        # L2.2: rolling/global IR 比例
                        if not is_bad:
                            roll = attrs.get("val_rolling6m_ir")
                            glob = attrs.get("val_global_ir")
                            if (roll is not None and glob is not None):
                                try:
                                    roll_f, glob_f = float(roll), float(glob)
                                    if (not math.isnan(roll_f)
                                            and not math.isinf(roll_f)
                                            and not math.isnan(glob_f)
                                            and not math.isinf(glob_f)
                                            and abs(glob_f) > 1e-6):
                                        ratio = abs(roll_f) / abs(glob_f)
                                        if ratio > IR_RATIO_TOLERANCE:
                                            is_bad = True
                                except (ValueError, TypeError):
                                    pass
                if is_bad:
                    abnormal += 1
                else:
                    complete += 1
        elif s == optuna.trial.TrialState.FAIL:
            fail += 1
        elif s == optuna.trial.TrialState.RUNNING:
            running += 1

    # 计算 best_value 时排除 -999 / NaN / inf
    best_value = None
    for t in study.trials:
        v = t.value
        if (t.state == optuna.trial.TrialState.COMPLETE
                and v is not None
                and isinstance(v, float)
                and not math.isnan(v)
                and not math.isinf(v)
                and v > -999.0):
            if best_value is None or v < best_value:
                best_value = v

    return {
        "complete": complete,        # 真正有效的Trial
        "fail": fail,                # FAIL状态
        "abnormal": abnormal,        # 异常Trial（Layer 1 + Layer 2）
        "running": running,
        "total": complete + fail + abnormal,
        "best_value": best_value,
        # ★ B-12 修复: 暴露 elapsed_sec 聚合, 支持历史 study 统计 + 续跑预热
        # 旧: 仅 trial_callback 内部用 _trial_times 局部变量, 重启即丢失
        # 新: 每次 get_study_stats 从 study.trials[*].user_attrs["elapsed_sec"] 聚合
        "elapsed_stats": _compute_elapsed_stats(study),
    }


def _compute_elapsed_stats(study: optuna.Study) -> dict:
    """从 study.trials[*].user_attrs["elapsed_sec"] 聚合 elapsed 统计.

    用途:
    1. 历史 study 报表: 累计总耗时 / 平均 / 最大最小
    2. 续跑预热: trial_callback 启动时遍历已有 trial 的 elapsed_sec
    3. 诊断: 找出耗时最长的 trial (可能是 OOM/timeout 前兆)
    """
    import math
    elapseds = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        ua = t.user_attrs or {}
        e = ua.get("elapsed_sec")
        if e is None:
            # 兼容: 旧数据没有 set_user_attr, 用 datetime 算
            try:
                if t.datetime_complete and t.datetime_start:
                    e = (t.datetime_complete - t.datetime_start).total_seconds()
            except Exception:
                continue
        try:
            ef = float(e)
            if not (math.isnan(ef) or math.isinf(ef)) and ef > 0:
                elapseds.append(ef)
        except (ValueError, TypeError):
            continue
    if not elapseds:
        return {
            "count": 0, "total_sec": 0.0, "avg_sec": None,
            "min_sec": None, "max_sec": None,
        }
    return {
        "count": len(elapseds),
        "total_sec": sum(elapseds),
        "avg_sec": sum(elapseds) / len(elapseds),
        "min_sec": min(elapseds),
        "max_sec": max(elapseds),
    }


def reset_p1(project: dict):
    """删除P1 DB，不动P1配置文件"""
    db = project["p1_db_path"]
    if os.path.exists(db):
        os.remove(db)


def reset_p2(project: dict):
    """删除P2 DB，不动P2配置文件"""
    db = project["p2_db_path"]
    if os.path.exists(db):
        os.remove(db)


def reset_all(project: dict):
    """删除P1+P2的DB，不动配置文件"""
    reset_p1(project)
    reset_p2(project)


def _ensure_wal(db_path: str):
    """开启SQLite WAL模式，提升读写性能和续跑稳定性"""
    import sqlite3
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA cache_size=-64000;")  # 64MB缓存
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.commit()
    except Exception:
        pass  # 不影响主流程


def _ensure_wal_safe(db_path: str):
    """安全开启SQLite WAL模式，避免与SQLAlchemy连接冲突。

    与 _ensure_wal 的区别：
    1. 使用 WAL 检查点确保 WAL 文件合并后再关闭
    2. 使用 timeout 避免连接被锁死
    3. 适用于 Optuna study 创建/加载后调用（此时 SQLAlchemy 可能仍持有连接池）
    """
    import sqlite3
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA cache_size=-64000;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.commit()
    except Exception:
        pass  # 不影响主流程


def update_trials_history(project: dict, n_trials_this_run: int, phase: str = "p1"):
    """每次运行完更新配置文件里的trials_history（追加本次trial数）"""
    if phase == "p1":
        config = load_p1_config(project)
        path = project["p1_config_path"]
    else:
        config = load_p2_config(project)
        path = project["p2_config_path"]

    if "trials_history" not in config:
        config["trials_history"] = []
    config["trials_history"].append(n_trials_this_run)
    config["trials_total"] = sum(config["trials_history"])

    # 绕过DB检查直接写（这是运行后的更新不是用户修改）
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def cleanup_bad_trials(
    project: dict,
    phase: str = "p1",
    dry_run: bool = False,
    skip_running: bool = False,
) -> dict:
    """
    清理 study 里的异常 Trial（直接操作 SQLite，绕过 Optuna API 限制）。

    检测规则（按用户要求：只做 Layer 1 + Layer 2，宁可漏杀不可误杀）：
    ─ Layer 1：DB 字段级体检（极低风险） ─
    1. 状态异常：FAIL / RUNNING / WAITING
    2. 评分异常：value <= -999（objective 抛异常时的兜底值）
    3. 评分异常：value 是 NaN / inf
    4. trial_value 缺失（state=COMPLETE 但 trial_values 表里没记录）
    5. datetime_complete 缺失（state=COMPLETE 但该字段为 NULL，Optuna 异常退出）
    6. heartbeat 过期（RUNNING 状态但 heartbeat 超过 1 小时 → 僵死）
    7. intermediate_values 含 NaN/inf（仅当表存在时检查）
    8. user_attrs 完全为空（objective 异常未记录）
    9. 关键指标（val_ic / val_icir / val_rolling6m_ir / val_global_ir）全缺失
    10. 关键指标为 NaN / inf / 不可解析
    ─ Layer 2：核心指标内部一致性（启用宽阈值） ─
    11. IC / ICIR 符号不一致（仅当 |IC| > 0.1 时才检查，避免小 IC 噪声触发）
    12. val_rolling6m_ir / val_global_ir 比例 > 10×（极宽阈值，只抓极端背离）
    ─ 不做的检查（按用户原则） ─
    Layer 3 硬阈值（metric_out_of_range / pct_positive_excess_low /
                    low_excess_return）："不可能"策略可能有效、幸存者偏差、
                    不同市场时段边界不同，不该自动清掉；贝叶斯优化需要多样性。
    Layer 4 窗口级（window_failure_rate / window_nan_misjudge）：老数据里的
                  窗口 NaN 可能是数据瑕疵而非策略失败。

    Args:
        project: 项目字典
        phase: "p1" 或 "p2"
        dry_run: True = 只扫描不删除，返回将被删除的 trial 清单

    Returns:
        {
            "deleted": int,
            "kept": int,
            "message": str,
            "detail": list,         # 全部异常详情（最多保留 30 条）
            "dry_run": bool,
            "category": dict,       # 各类异常的计数
        }
    """
    import sqlite3
    import math

    db_path = project["p1_db_path"] if phase == "p1" else project["p2_db_path"]
    study_name = project["p1_study_name"] if phase == "p1" else project["p2_study_name"]

    if not os.path.exists(db_path):
        return _empty_cleanup_result(
            "数据库不存在", dry_run=dry_run
        )

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()

        # 1. 找 study_id
        cursor.execute(
            "SELECT study_id FROM studies WHERE study_name=?",
            (study_name,)
        )
        row = cursor.fetchone()
        if not row:
            return _empty_cleanup_result(
                "study不存在", dry_run=dry_run
            )
        study_id = row[0]

        # 2. 检查中间值表是否存在
        has_intermediate_table = False
        if INTERMEDIATE_NAN_QUERY and DETECTION_RULES["intermediate_values_nan"]:
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='trial_intermediate_values'"
            )
            has_intermediate_table = cursor.fetchone() is not None

        # 3. 收集异常 trial
        bad_trial_ids = []
        bad_id_set = set()
        detail = []
        cat = {
            "state":           0,   # 状态异常
            "value":           0,   # value 异常 (<= -999 / NaN / inf)
            "value_missing":   0,   # trial_value 缺失
            "datetime_null":   0,   # datetime_complete 缺失
            "heartbeat_stale": 0,   # heartbeat 过期
            "intermediate_nan":0,   # 中间值含 NaN/inf
            "empty_attrs":     0,   # 无 user_attrs
            "missing_metrics": 0,   # 关键指标全缺失
            "nan_metrics":     0,   # 关键指标 NaN/inf
            "ic_icir_sign":    0,   # IC/ICIR 符号不一致
            "ir_ratio":        0,   # rolling/global IR 比例失衡
        }

        # ── 2a. 状态异常 ──
        # skip_running=True 用于 study 运行期间停止按钮，避免删除 RUNNING Trial
        # 导致正在执行的 trial.set_user_attr() 抛 KeyError 引起 Phase1 崩溃
        if DETECTION_RULES["state_abnormal"]:
            if skip_running:
                cursor.execute(
                    """SELECT trial_id, number, state FROM trials
                       WHERE study_id=? AND state IN ('FAIL','WAITING')""",
                    (study_id,)
                )
            else:
                cursor.execute(
                    """SELECT trial_id, number, state FROM trials
                       WHERE study_id=? AND state IN ('FAIL','RUNNING','WAITING')""",
                    (study_id,)
                )
            for tid, tnum, state in cursor.fetchall():
                if tid not in bad_id_set:
                    bad_trial_ids.append(tid)
                    bad_id_set.add(tid)
                    cat["state"] += 1
                detail.append(f"Trial#{tnum}: 状态={state} [L1-状态]")

        # ── 2b. value 异常（<= -999 / NaN / inf）──
        # SQLite 里的 NaN 用 != NaN 判定，inf 用 abs>1e308 判定
        if (DETECTION_RULES["value_below_-999"]
                or DETECTION_RULES["value_nan_inf"]):
            cursor.execute(
                """SELECT t.trial_id, t.number, tv.value FROM trials t
                   JOIN trial_values tv ON t.trial_id=tv.trial_id
                   WHERE t.study_id=? AND t.state='COMPLETE'
                     AND (tv.value <= -999.0
                          OR tv.value IS NULL
                          OR tv.value != tv.value
                          OR ABS(tv.value) > 1.0e308)""",
                (study_id,)
            )
            for tid, tnum, val in cursor.fetchall():
                if tid not in bad_id_set:
                    bad_trial_ids.append(tid)
                    bad_id_set.add(tid)
                    cat["value"] += 1
                detail.append(
                    f"Trial#{tnum}: value={val} [L1-value异常]"
                )

        # ── 2c. COMPLETE 但 trial_value 缺失 ──
        if DETECTION_RULES["trial_value_missing"]:
            cursor.execute(
                """SELECT t.trial_id, t.number FROM trials t
                   LEFT JOIN trial_values tv ON t.trial_id=tv.trial_id
                   WHERE t.study_id=? AND t.state='COMPLETE'
                     AND tv.trial_value_id IS NULL""",
                (study_id,)
            )
            for tid, tnum in cursor.fetchall():
                if tid not in bad_id_set:
                    bad_trial_ids.append(tid)
                    bad_id_set.add(tid)
                    cat["value_missing"] += 1
                detail.append(f"Trial#{tnum}: trial_value缺失 [L1-DB异常]")

        # ── 2d. COMPLETE 但 datetime_complete 为 NULL ──
        if DETECTION_RULES["datetime_complete_null"]:
            cursor.execute(
                """SELECT trial_id, number FROM trials
                   WHERE study_id=? AND state='COMPLETE'
                     AND datetime_complete IS NULL""",
                (study_id,)
            )
            for tid, tnum in cursor.fetchall():
                if tid not in bad_id_set:
                    bad_trial_ids.append(tid)
                    bad_id_set.add(tid)
                    cat["datetime_null"] += 1
                detail.append(
                    f"Trial#{tnum}: datetime_complete缺失 [L1-DB异常]"
                )

        # ── 2e. heartbeat 过期（仅 RUNNING）──
        if DETECTION_RULES["heartbeat_stale"]:
            cursor.execute(
                """SELECT t.trial_id, t.number,
                          (julianday('now') - julianday(h.heartbeat)) * 86400.0
                   FROM trials t
                   JOIN trial_heartbeats h ON t.trial_id=h.trial_id
                   WHERE t.study_id=? AND t.state='RUNNING'
                     AND (julianday('now') - julianday(h.heartbeat)) * 86400.0 > ?""",
                (study_id, HEARTBEAT_STALE_SEC)
            )
            for tid, tnum, age_sec in cursor.fetchall():
                if tid not in bad_id_set:
                    bad_trial_ids.append(tid)
                    bad_id_set.add(tid)
                    cat["heartbeat_stale"] += 1
                detail.append(
                    f"Trial#{tnum}: heartbeat过期({age_sec/3600:.1f}h) [L1-僵死]"
                )

        # ── 2f. intermediate_values 含 NaN/inf ──
        if has_intermediate_table and DETECTION_RULES["intermediate_values_nan"]:
            cursor.execute(
                """SELECT t.trial_id, t.number, COUNT(*)
                   FROM trials t
                   JOIN trial_intermediate_values iv ON t.trial_id=iv.trial_id
                   WHERE t.study_id=? AND t.state='COMPLETE'
                     AND (iv.intermediate_value IS NULL
                          OR iv.intermediate_value != iv.intermediate_value
                          OR ABS(iv.intermediate_value) > 1.0e308)
                   GROUP BY t.trial_id""",
                (study_id,)
            )
            for tid, tnum, n_bad in cursor.fetchall():
                if tid not in bad_id_set:
                    bad_trial_ids.append(tid)
                    bad_id_set.add(tid)
                    cat["intermediate_nan"] += 1
                detail.append(
                    f"Trial#{tnum}: intermediate含{n_bad}个NaN/inf [L1-中间值]"
                )

        # ── 2g. 指标级体检（针对 COMPLETE 状态的 trial）──
        cursor.execute(
            """SELECT trial_id, number FROM trials
               WHERE study_id=? AND state='COMPLETE'""",
            (study_id,)
        )
        complete_trials = cursor.fetchall()

        CRITICAL_METRICS = [
            "val_ic", "val_icir", "val_rolling6m_ir",
            "pct_positive_excess", "val_global_ir",
        ]
        REQUIRED_METRICS = [
            "val_ic", "val_icir", "val_rolling6m_ir", "val_global_ir",
        ]

        for tid, tnum in complete_trials:
            if tid in bad_id_set:
                continue

            # 读 user_attrs
            cursor.execute(
                '''SELECT "key", value_json FROM trial_user_attributes
                   WHERE trial_id=?''',
                (tid,)
            )
            raw_rows = cursor.fetchall()
            user_attrs = {}
            for k, vjson in raw_rows:
                if vjson is None:
                    user_attrs[k] = None
                    continue
                try:
                    user_attrs[k] = json.loads(vjson)
                except (ValueError, TypeError):
                    user_attrs[k] = vjson

            reasons = []
            is_bad = False

            # ── L1.8 无 user_attrs ──
            if DETECTION_RULES["empty_user_attrs"] and not user_attrs:
                is_bad = True
                reasons.append("无任何评估指标")
                cat["empty_attrs"] += 1
            elif user_attrs:
                # ── L1.9 关键指标全缺失 ──
                missing_req = [
                    m for m in REQUIRED_METRICS
                    if user_attrs.get(m) is None
                ]
                if (DETECTION_RULES["all_required_metrics_missing"]
                        and len(missing_req) == len(REQUIRED_METRICS)):
                    is_bad = True
                    reasons.append(
                        f"关键指标全缺失({','.join(REQUIRED_METRICS)})"
                    )
                    cat["missing_metrics"] += 1
                else:
                    # ── L1.10 关键指标 NaN / inf / 不可解析 ──
                    if DETECTION_RULES["critical_metrics_nan"]:
                        for metric in CRITICAL_METRICS:
                            raw_val = user_attrs.get(metric)
                            if raw_val is None:
                                continue
                            try:
                                mv = float(raw_val)
                                if math.isnan(mv) or math.isinf(mv):
                                    is_bad = True
                                    reasons.append(
                                        f"{metric}={'NaN' if math.isnan(mv) else 'inf'}"
                                    )
                                    break
                            except (ValueError, TypeError):
                                is_bad = True
                                reasons.append(f"{metric}=不可解析")
                                break
                        if is_bad:
                            cat["nan_metrics"] += 1

                    # ── L2.1 IC/ICIR 符号一致（仅 |IC| > 0.1）──
                    if not is_bad and DETECTION_RULES["ic_icir_sign"]:
                        ic = user_attrs.get("val_ic")
                        icir = user_attrs.get("val_icir")
                        if (ic is not None and icir is not None):
                            try:
                                ic_f, icir_f = float(ic), float(icir)
                                if (not math.isnan(ic_f)
                                        and not math.isinf(ic_f)
                                        and not math.isnan(icir_f)
                                        and not math.isinf(icir_f)
                                        and abs(ic_f) > IC_SIGN_CHECK_THRESHOLD):
                                    # 符号不一致视为异常
                                    if (ic_f > 0) != (icir_f > 0):
                                        is_bad = True
                                        reasons.append(
                                            f"IC/ICIR符号不一致"
                                            f"(ic={ic_f:+.4f},icir={icir_f:+.4f})"
                                        )
                                        cat["ic_icir_sign"] += 1
                            except (ValueError, TypeError):
                                pass  # 解析失败已被 NaN 检查覆盖

                    # ── L2.2 rolling6m_ir / global_ir 比例（10× 极宽）──
                    if not is_bad and DETECTION_RULES["rolling_vs_global_ir"]:
                        roll = user_attrs.get("val_rolling6m_ir")
                        glob = user_attrs.get("val_global_ir")
                        if (roll is not None and glob is not None):
                            try:
                                roll_f, glob_f = float(roll), float(glob)
                                if (not math.isnan(roll_f)
                                        and not math.isinf(roll_f)
                                        and not math.isnan(glob_f)
                                        and not math.isinf(glob_f)
                                        and abs(glob_f) > 1e-6):
                                    ratio = abs(roll_f) / abs(glob_f)
                                    if ratio > IR_RATIO_TOLERANCE:
                                        is_bad = True
                                        reasons.append(
                                            f"rolling/global_ir比例={ratio:.2f}"
                                            f"超出{IR_RATIO_TOLERANCE:.0f}x"
                                        )
                                        cat["ir_ratio"] += 1
                            except (ValueError, TypeError):
                                pass

            if is_bad:
                bad_trial_ids.append(tid)
                bad_id_set.add(tid)
                detail.append(f"Trial#{tnum}: {', '.join(reasons)} [L1-2]")

        if not bad_trial_ids:
            return {
                "deleted": 0,
                "kept": _count_complete(conn, study_id),
                "message": "✅ 无需清理，未发现异常 Trial",
                "detail": [],
                "dry_run": dry_run,
                "category": cat,
            }

        # 4. 干运行：返回清单不删除
        if dry_run:
            kept = _count_complete(conn, study_id)
            return {
                "deleted": 0,  # 干运行不实际删除
                "would_delete": len(bad_trial_ids),
                "kept": kept,
                "message": (
                    f"🔍 干运行：将清理{len(bad_trial_ids)}个异常 Trial，"
                    f"保留{kept}个有效 Trial（未实际删除）"
                ),
                "detail": detail[:50],
                "dry_run": True,
                "category": cat,
            }

        # 5. 实际删除：级联删除子表 → 主表
        placeholders = ",".join("?" * len(bad_trial_ids))

        for table in [
            "trial_user_attributes",
            "trial_system_attributes",
            "trial_params",
            "trial_values",
            "trial_intermediate_values",
            "trial_heartbeats",
        ]:
            try:
                cursor.execute(
                    f'DELETE FROM "{table}" WHERE trial_id IN ({placeholders})',
                    bad_trial_ids
                )
            except sqlite3.OperationalError:
                pass  # 旧版本可能没有某张表
        cursor.execute(
            f"DELETE FROM trials WHERE trial_id IN ({placeholders})",
            bad_trial_ids
        )
        conn.commit()

        kept = _count_complete(conn, study_id)
        return {
            "deleted": len(bad_trial_ids),
            "kept": kept,
            "message": (
                f"🧹 已清理{len(bad_trial_ids)}个异常 Trial，"
                f"保留{kept}个有效 Trial"
            ),
            "detail": detail[:30],
            "dry_run": False,
            "category": cat,
        }
    except Exception as e:
        logger.error(f"清理异常Trial失败: {e}", exc_info=True)
        return _empty_cleanup_result(
            f"清理失败：{e}", dry_run=dry_run
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _empty_cleanup_result(message: str, dry_run: bool) -> dict:
    """统一的空结果格式"""
    return {
        "deleted": 0,
        "kept": 0,
        "message": message,
        "detail": [],
        "dry_run": dry_run,
        "category": {},
    }


def _count_complete(conn, study_id: int) -> int:
    """统计 study 当前的 COMPLETE trial 数量（内部使用）"""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM trials WHERE study_id=? AND state='COMPLETE'",
                (study_id,)
            )
            return cur.fetchone()[0]
    except Exception:
        return 0


def get_reusable_p1_trials(
    project: dict,
    p2_param_ranges: dict,
    p2_objective_weights: dict,
    p2_active_params: Optional[list],
    p2_scheme: str,
    p2_window_count: int,
    max_inject: int = 30,
    enable_normalization: bool = False,
    norm_config: Optional[dict] = None,
) -> dict:
    """
    从P1 study中筛选可复用的Trial，用P2权重重算score。

    Returns:
        {
            "trials": [{"params": ..., "value": ..., "user_attrs": ...}, ...],
            "total_p1":    int,   # P1有效Trial总数
            "qualified":   int,   # 通过范围过滤的数量
            "injected":    int,   # 实际注入的数量（top-N）
            "blocked":     str,   # 非空表示被阻断，含原因
        }
    """
    from m5_optimizer.search_space import ALL_PARAMS, OBJECTIVE_VARS

    _direction_map = {v["name"]: v["direction"] for v in OBJECTIVE_VARS}

    # ── 前置校验：scheme 必须一致 ──────────────────────────────
    try:
        p1_cfg = load_p1_config(project)["config"]
        p1_scheme       = p1_cfg.get("scheme", "")
        p1_window_count = p1_cfg.get("window_count", -1)
    except Exception as e:
        return {"trials": [], "total_p1": 0, "qualified": 0,
                "injected": 0, "blocked": f"读取P1配置失败: {e}"}

    if p1_scheme != p2_scheme:
        return {"trials": [], "total_p1": 0, "qualified": 0, "injected": 0,
                "blocked": (f"scheme不一致：P1={p1_scheme}, P2={p2_scheme}。"
                            f"复用被阻断，metrics不可比。")}

    # window_count 不一致：警告但不阻断（允许近似复用）
    _wc_warn = ""
    if p1_window_count > 0 and p1_window_count != p2_window_count:
        _wc_warn = (f"⚠️ window_count不一致(P1={p1_window_count}, "
                    f"P2={p2_window_count})，已标记为近似复用")

    # ── 加载P1 study ─────────────────────────────────────────
    p1_study = get_p1_study(project)
    if p1_study is None:
        return {"trials": [], "total_p1": 0, "qualified": 0,
                "injected": 0, "blocked": "P1 study不存在"}

    complete_trials = [
        t for t in p1_study.trials
        if (t.state.name == "COMPLETE"
            and t.value is not None
            and t.value > -999.0)
    ]
    total_p1 = len(complete_trials)

    if total_p1 == 0:
        return {"trials": [], "total_p1": 0, "qualified": 0,
                "injected": 0, "blocked": "P1无有效Trial"}

    # ── 参数范围校验辅助函数 ──────────────────────────────────
    def _param_in_p2_range(name: str, value) -> bool:
        """检查P1 Trial的某个参数值是否在P2范围内"""
        rng  = p2_param_ranges.get(name, {})
        pdef = ALL_PARAMS.get(name, {})
        ptype = pdef.get("type", "")

        if ptype == "categorical":
            choices = rng.get("choices", pdef.get("choices", []))
            return value in choices

        # int / float / float_log
        lo = rng.get("low",  pdef.get("low",  float("-inf")))
        hi = rng.get("high", pdef.get("high", float("inf")))
        try:
            v = float(value)
            return lo <= v <= hi
        except (TypeError, ValueError):
            return False

    def _recompute_score(user_attrs: dict) -> float:
        """用P2权重从user_attrs重算score（Optuna minimize方向）
        支持归一化开关：当 enable_normalization=True 时应用归一化
        """
        from m5_optimizer.objective import _apply_normalization
        score = 0.0
        for metric, weight in p2_objective_weights.items():
            val  = float(user_attrs.get(metric, 0.0) or 0.0)
            mult = 1.5 if metric == "ic_gap_penalty" else 1.0
            direction = _direction_map.get(metric, "max")

            # ★ 归一化处理
            if enable_normalization:
                norm_val = _apply_normalization(metric, val, norm_config)
            else:
                norm_val = val

            if direction == "max":
                score += weight * mult * norm_val
            else:
                score -= weight * mult * norm_val
        return -score   # Optuna minimize

    # ── 过滤：参数在P2范围内 ──────────────────────────────────
    qualified = []
    for t in complete_trials:
        ok = True
        for name, value in t.params.items():
            # 仅检查P2参与搜索的参数
            if p2_active_params is not None and name not in p2_active_params:
                continue
            if name not in ALL_PARAMS:
                continue
            if not _param_in_p2_range(name, value):
                ok = False
                break
        if ok:
            re_score = _recompute_score(t.user_attrs)
            qualified.append({
                "params":     dict(t.params),
                "value":      re_score,      # P2权重下的score
                "user_attrs": dict(t.user_attrs),
                "p1_trial_id": t.number,
            })

    qualified_count = len(qualified)
    if qualified_count == 0:
        return {"trials": [], "total_p1": total_p1,
                "qualified": 0, "injected": 0,
                "blocked": _wc_warn or ""}

    # ── 取 top-N（按P2 re-scored value升序，越小越优）──────────
    qualified.sort(key=lambda x: x["value"])
    inject_n = min(qualified_count, max_inject)
    selected = qualified[:inject_n]

    return {
        "trials":    selected,
        "total_p1":  total_p1,
        "qualified": qualified_count,
        "injected":  inject_n,
        "blocked":   _wc_warn,
    }
