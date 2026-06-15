请实现 P1 Trial 复用功能：将符合P2参数范围的P1 Trial作为已完成结果注入P2 study，
节省Trial计算配额，同时为 TPE 提供高质量先验。

## 铁律

- run\_m2 第二参数必须是 xgbm\_params
- load\_if\_exists=True 禁止修改
- 外层 joblib 必须用 backend='threading'

***

## 文件1：project\_manager.py — 新增 get\_reusable\_p1\_trials()

在文件末尾追加以下完整函数：

```python
def get_reusable_p1_trials(
    project: dict,
    p2_param_ranges: dict,
    p2_objective_weights: dict,
    p2_active_params: Optional[list],
    p2_scheme: str,
    p2_window_count: int,
    max_inject: int = 30,
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
        """用P2权重从user_attrs重算score（Optuna minimize方向）"""
        score = 0.0
        for metric, weight in p2_objective_weights.items():
            val  = float(user_attrs.get(metric, 0.0) or 0.0)
            mult = 1.5 if metric == "ic_gap_penalty" else 1.0
            direction = _direction_map.get(metric, "max")
            if direction == "max":
                score += weight * mult * val
            else:
                score -= weight * mult * val
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


文件2：phase2_local.py — 新增注入逻辑

2-A：在文件顶部 import 区添加

import datetime
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)


2-B：新增辅助函数（放在 run_phase2 之前）

def _build_p2_distributions(
    param_name: str,
    custom_ranges: dict,
) -> "optuna.distributions.BaseDistribution | None":
    """
    根据P2的custom_ranges为单个参数构建Optuna分布对象。
    返回None表示该参数为固定参数（不在搜索空间）。
    """
    from m5_optimizer.search_space import ALL_PARAMS

    rng  = custom_ranges.get(param_name, {})
    pdef = ALL_PARAMS.get(param_name, {})

    # 固定参数（有adjusted_default但无low/high/choices）
    if "adjusted_default" in rng and "low" not in rng and "choices" not in rng:
        return None

    ptype = pdef.get("type", "")
    if ptype == "float_log":
        lo = float(rng.get("low", pdef["low"]))
        hi = float(rng.get("high", pdef["high"]))
        return FloatDistribution(lo, hi, log=True)
    elif ptype == "float":
        lo = float(rng.get("low", pdef["low"]))
        hi = float(rng.get("high", pdef["high"]))
        return FloatDistribution(lo, hi)
    elif ptype == "int":
        lo = int(rng.get("low", pdef["low"]))
        hi = int(rng.get("high", pdef["high"]))
        return IntDistribution(lo, hi)
    elif ptype == "categorical":
        choices = rng.get("choices", pdef.get("choices", []))
        return CategoricalDistribution(choices)
    return None


def _inject_reused_trials(
    study: "optuna.Study",
    reusable: list,
    custom_ranges: dict,
    active_params: "list | None",
) -> int:
    """
    将P1可复用Trial以COMPLETE状态注入P2 study。
    返回实际成功注入的数量。
    """
    from m5_optimizer.search_space import ALL_PARAMS

    injected = 0
    now = datetime.datetime.now()

    for rt in reusable:
        params     = rt["params"]
        value      = rt["value"]
        user_attrs = rt["user_attrs"]

        # 只包含P2参与搜索的参数（active_params中的）
        filtered_params = {}
        distributions   = {}

        for name, val in params.items():
            if active_params is not None and name not in active_params:
                continue   # 固定参数不进distributions
            dist = _build_p2_distributions(name, custom_ranges)
            if dist is None:
                continue   # 无法构建分布 → 跳过
            filtered_params[name] = val
            distributions[name]   = dist

        if not filtered_params:
            continue   # 没有可用参数 → 跳过

        try:
            frozen = optuna.trial.FrozenTrial(
                number=-1,
                trial_id=-1,
                state=optuna.trial.TrialState.COMPLETE,
                value=value,
                values=None,
                datetime_start=now,
                datetime_complete=now,
                params=filtered_params,
                distributions=distributions,
                user_attrs={
                    **user_attrs,
                    "_reused_from_p1": rt.get("p1_trial_id", -1),
                },
                system_attrs={},
                intermediate_values={},
            )
            study.add_trial(frozen)
            injected += 1
        except Exception as e:
            logger.warning(f"注入P1 Trial失败（跳过）: {e}")

    logger.info(f"P1 Trial复用：成功注入 {injected}/{len(reusable)} 个")
    return injected


2-C：修改 run_phase2() 函数签名，新增参数

在 storage: str = None, 之后添加：

    reuse_p1_trials: bool = True,   # 是否尝试复用P1 Trial
    max_p1_inject: int = 30,        # 最多注入数量


2-D：在 study 创建/加载完成后、init_params 注入之前，添加P1复用逻辑

找到 if init_params: 这一行，在其之前插入：

    # ── P1 Trial 复用注入 ─────────────────────────────────────
    if reuse_p1_trials and project is not None:
        try:
            from m5_optimizer.project_manager import get_reusable_p1_trials

            # 仅在P2 study全新时复用（避免重复注入）
            existing_complete = sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            )
            if existing_complete == 0:
                reuse_result = get_reusable_p1_trials(
                    project=project,
                    p2_param_ranges=param_ranges or {},
                    p2_objective_weights=objective_weights or {},
                    p2_active_params=active_params,
                    p2_scheme=scheme,
                    p2_window_count=window_count,
                    max_inject=max_p1_inject,
                )
                if reuse_result["blocked"] and not reuse_result["blocked"].startswith("⚠️"):
                    logger.warning(f"P1复用被阻断: {reuse_result['blocked']}")
                elif reuse_result["injected"] > 0:
                    _inject_reused_trials(
                        study=study,
                        reusable=reuse_result["trials"],
                        custom_ranges=param_ranges or {},
                        active_params=active_params,
                    )
                    logger.info(
                        f"P1复用摘要: P1有效={reuse_result['total_p1']}, "
                        f"范围内={reuse_result['qualified']}, "
                        f"已注入={reuse_result['injected']}"
                    )
                else:
                    logger.info(
                        f"P1复用: P1有效={reuse_result['total_p1']}, "
                        f"范围内={reuse_result['qualified']}，无可注入Trial"
                    )
            else:
                logger.info(f"P2 study已有{existing_complete}个Trial，跳过P1复用")
        except Exception as e:
            logger.warning(f"P1复用失败（不影响P2正常运行）: {e}")
    # ─────────────────────────────────────────────────────────


文件3：app.py — start_phase2() 传入复用参数

3-A：在 run_phase2() 调用处添加两个新参数

run_phase2(
    ...（现有参数不变）...
    reuse_p1_trials=True,
    max_p1_inject=min(30, max(10, n_trials // 3)),
    # 注入上限：最多30个，最少10个，或P2 Trial总数的1/3
)


3-B：在 P2 启动日志中显示复用信息

在 log_msg = f"开始Phase2: {n_trials} trials\n" 之后追加：

log_msg += "🔄 将自动检测并复用符合条件的P1 Trial（仅全新study有效）\n"


文件4：Tab2 export_to_p2() — 显示预估可复用数量

在 export_to_p2() 函数内、调用 export_p2_config_from_p1() 之前，
添加预估计算（用于向用户展示，不阻断流程）：

# ── 预估可复用Trial数（信息展示，不阻断）──────────────────────
reuse_preview = ""
try:
    from m5_optimizer.project_manager import get_reusable_p1_trials
    _preview = get_reusable_p1_trials(
        project=project,
        p2_param_ranges=p2_param_ranges,
        p2_objective_weights=p2_weights,
        p2_active_params=p2_active_params,
        p2_scheme=_p2_scheme,     # 从project p2 config读取
        p2_window_count=173,
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


在最终返回的 status 字符串中追加 reuse_preview：

return (
    f"✅ P2配置已导出\n{summary}{reuse_preview}\n"
    f"active_params: {'继承P1' if p2_active_params else '全部搜索'}",
    gr.update(value=project["project_root"]),
)


其中 _p2_scheme 从 p2 config 中读取：

try:
    _p2_scheme = load_p2_config(project)["config"].get("scheme", "scheme_b")
except Exception:
    _p2_scheme = "scheme_b"


测试

# 保存为 test_p1_reuse.py，在项目根目录执行
from m5_optimizer.project_manager import (
    _build_project_dict,
    get_reusable_p1_trials,
    load_p2_config,
)

project = _build_project_dict("E:/10q/10q-202604gpu/output/15B")
p2_cfg  = load_p2_config(project)["config"]

result = get_reusable_p1_trials(
    project=project,
    p2_param_ranges=p2_cfg["param_ranges"],
    p2_objective_weights=p2_cfg["objective_weights"],
    p2_active_params=p2_cfg["active_params"],
    p2_scheme=p2_cfg["scheme"],
    p2_window_count=p2_cfg["window_count"],
    max_inject=30,
)

print(f"阻断原因: '{result['blocked']}'")
print(f"P1有效Trial: {result['total_p1']}")
print(f"符合P2范围: {result['qualified']}")
print(f"将注入: {result['injected']}")
if result["trials"]:
    print("\nTop-3 注入Trial预览:")
    for t in result["trials"][:3]:
        print(f"  P1 Trial#{t['p1_trial_id']}: "
              f"P2重算score={t['value']:.4f}, "
              f"lgbm_lr={t['params'].get('lgbm_learning_rate'):.4f}")


期望结果：

	•	blocked 为空或以 ⚠️ 开头（scheme_b一致则不阻断）
	•	qualified > 0（15B P1有219个合格Trial，P2范围已缩窄，预计10-50个符合）
	•	injected = min(qualified, 30)

完成后返回

	1.	测试脚本的实际输出（blocked/total/qualified/injected 四项数值）
	2.	_build_p2_distributions 中各类型参数的处理是否覆盖全部30个参数
	3.	study.add_trial(frozen) 调用是否报 distribution inconsistency 警告
	4.	如发现 _reused_from_p1 字段在 user_attrs 中影响 Tab2 分析，列出具体影响
```

