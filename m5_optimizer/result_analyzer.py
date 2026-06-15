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
M5结果分析器
读最优Trial，触发M2+M4完整重跑
"""
import os
import logging
from typing import Dict, Any

import optuna

from m5_optimizer.search_space import assemble_params

logger = logging.getLogger("m5.result_analyzer")


def get_best_params(study: optuna.Study) -> Dict[str, Any]:
    if study.best_trial is None:
        raise ValueError("Study中没有完成的Trial，无法获取最优参数")
    return dict(study.best_trial.params)


def get_best_study(
    phase2_path: str = "output/m5/phase2.db",
    phase1_path: str = "output/m5/phase1.db",
) -> optuna.Study:
    for path, name in [
        (phase2_path, "phase2_local"),
        (phase1_path, "phase1_global"),
    ]:
        if os.path.exists(path):
            return optuna.load_study(
                study_name=name,
                storage=f"sqlite:///{path}"
            )
    raise FileNotFoundError("未找到任何study数据库")


def get_best_study_from_project(
    project: dict,
    prefer: str = "p2",
) -> optuna.Study:
    """从项目目录加载study，优先P2，其次P1。"""
    if prefer == "p2":
        studies = [
            (project.get("p2_db_path", ""), project.get("p2_study_name", "phase2_local")),
            (project.get("p1_db_path", ""), project.get("p1_study_name", "phase1_global")),
        ]
    else:
        studies = [
            (project.get("p1_db_path", ""), project.get("p1_study_name", "phase1_global")),
            (project.get("p2_db_path", ""), project.get("p2_study_name", "phase2_local")),
        ]

    for db_path, study_name in studies:
        if db_path and os.path.exists(db_path):
            return optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///{db_path}",
            )

    raise FileNotFoundError(
        f"项目 {project.get('project_name', '未知')} 未找到任何study数据库"
    )


def _assemble_params_from_best(best_params: Dict[str, Any]) -> tuple:
    lgbm_params, xgbm_params, feature_params, lgbm_weight, window_params = (
        assemble_params(best_params)
    )
    return lgbm_params, xgbm_params, feature_params, lgbm_weight, window_params


def run_full_backtest(
    best_params: Dict[str, Any],
    scheme: str = "scheme_b",
    include_no_penalty: bool = True,
    progress_callback=None,
    use_gpu: bool = True,        # ★ v3.8: 默认 GPU
    gpu_strategy: str = "D",     # ★ v3.8: 默认 D 模式
) -> Dict[str, Any]:
    if use_gpu:
        from m2_engine_gpu.run_m2 import run_m2
        run_kwargs_extra = dict(strategy=gpu_strategy)
    else:
        from m2_engine.run_m2 import run_m2
        run_kwargs_extra = {}

    lgbm_params, xgbm_params, feature_params, lgbm_weight, window_params = (
        _assemble_params_from_best(best_params)
    )

    result = {
        "report_path": "",
        "metrics": {},
        "compare_report_path": "",
    }

    if progress_callback:
        progress_callback("开始M2全量回测...")

    # 通过 preloaded_factor_df 传入指定 scheme 的数据
    preloaded_factor_df = None
    try:
        from m1_engine.data_loader import DataLoader
        from m1_engine.label_maker import LabelMaker
        if progress_callback:
            progress_callback(f"加载数据方案: {scheme}...")
        loader = DataLoader(scheme=scheme)
        preloaded_factor_df = loader.load()
        preloaded_factor_df = LabelMaker().make_labels(preloaded_factor_df)
    except Exception as e:
        logger.warning(f"预加载数据失败，将使用默认方案: {e}")
        preloaded_factor_df = None

    portfolios, stats = run_m2(
        lgbm_params=lgbm_params,
        xgbm_params=xgbm_params,
        feature_params=feature_params,
        lgbm_weight=lgbm_weight,
        train_months=window_params.get("train_months"),
        fast_mode=False,
        compute_val_metrics=True,
        compute_shap=False,
        verbose=True,
        preloaded_factor_df=preloaded_factor_df,
        gpu_mode=use_gpu,  # ★ v3.8: 默认 GPU
        disable_penalty=False,
        **run_kwargs_extra,
    )

    result["metrics"] = {
        "avg_val_ic": stats.get("avg_val_ic", 0),
        "avg_ic_gap": stats.get("avg_ic_gap", 0),
        "penalized_rate": stats.get("low_confidence_months", 0) /
        max(stats.get("success", 1), 1),
        "success_windows": stats.get("success", 0),
        "total_windows": stats.get("total_windows", 0),
    }

    if progress_callback:
        progress_callback("M2回测完成，生成报告...")

    try:
        from m4_report.report_generator import generate_report
        report_path = f"output/backtest_report_{scheme}.html"
        m4_result = generate_report(
            portfolios,
            stats=stats,
            output_path=report_path,
        )
        result["report_path"] = m4_result.get("report_path", report_path)
        if "metrics" in m4_result:
            result["metrics"].update(m4_result["metrics"])
    except Exception as e:
        logger.warning(f"报告生成失败: {e}")
        result["report_path"] = ""

    # 当前 is_penalized 不影响仓位，两次运行结果相同，跳过无降权对比
    if include_no_penalty:
        logger.info("当前版本 is_penalized 不影响仓位，跳过无降权对比")

    if progress_callback:
        progress_callback("完整回测完成！")

    return result
