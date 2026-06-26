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
import sys
import threading
import contextlib
import logging
from pathlib import Path
from typing import Dict, Any

import optuna

from m5_optimizer.search_space import assemble_params

logger = logging.getLogger("m5.result_analyzer")


# ━━━ stdout/stderr 重定向（核心修复） ━━━
class _StreamRedirector:
    """
    将 sys.stdout / sys.stderr 的写入按行切分后转发到 callback。
    ★ 修复: M2+M4 重跑期间，run_m2 会大量 print() 进度信息，
    这些输出原本会刷爆 PowerShell 终端，让用户无法继续操作。
    重定向后，print() 输出走 progress_callback → Gradio log box，
    终端保持干净，用户在 UI 里就能看到实时进度。
    """
    def __init__(self, callback):
        self.callback = callback
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        # 锁内只做"切行"，锁外调 callback 防止 callback 内部死锁
        pending = []
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    pending.append(line.rstrip())
        for line in pending:
            try:
                self.callback(line)
            except Exception:
                # 防止 callback 异常污染主流程
                pass
        return len(text) if isinstance(text, (str, bytes)) else 0

    def flush(self):
        # 不需要：重定向器只负责转发，原 stream 仍由其自身刷新
        pass

    def isatty(self):
        # 报告非 tty，避免某些库（如 tqdm）进入交互模式
        return False

    @property
    def closed(self):
        return False


@contextlib.contextmanager
def _redirect_output(callback):
    """
    临时接管 sys.stdout / sys.stderr，把写入转发到 callback。
    - callback=None: 不接管（保持原行为，直接打到终端）
    - callback!=None: 接管后 print() 不再落到原 stream，终端保持干净
    退出 with 块时自动恢复原 stream，并把残留 buffer flush 给 callback。

    ★ v5.0: 线程安全 —— 使用 threading.RLock 保护 sys.stdout/stderr 的替换和恢复，
    防止 Tab2/Tab4 回测与 Tab1/Tab3 Trial 同时运行时互相覆盖 sys.stdout 导致输出串流。
    """
    if callback is None:
        yield
        return

    # ★ v5.0: 模块级重定向锁，保证同一时刻只有一个线程在替换/恢复 sys.stdout
    if not hasattr(_redirect_output, '_lock'):
        _redirect_output._lock = threading.RLock()

    new_out = _StreamRedirector(callback)
    new_err = _StreamRedirector(callback)

    with _redirect_output._lock:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = new_out
            sys.stderr = new_err
        except Exception:
            sys.stdout = old_out
            sys.stderr = old_err
            raise

    try:
        yield
    finally:
        with _redirect_output._lock:
            # 残留 buffer（最后一行没换行）兜底送出去
            for stream in (new_out, new_err):
                tail = stream._buf
                if tail and tail.strip() and stream.callback:
                    try:
                        stream.callback(tail.rstrip())
                    except Exception:
                        pass
            # ★ 只恢复自己设置的 redirector（防止覆盖其他线程的重定向）
            if sys.stdout is new_out:
                sys.stdout = old_out
            if sys.stderr is new_err:
                sys.stderr = old_err


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
    report_tag: str = "",        # ★ v5.0: 报告路径标识（"t2"/"t4"），防止 Tab2/Tab4 同时回测互相覆盖
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

    # ★ v5.1 修复: train_months 感知的内存前置检查 (低内存友好版)
    # 原因: train_months=50 全量回测内存峰值可达 10-17GB，
    #   但用户希望"6GB 也能跑，慢点没事"。改为：
    #   - 可用 < 2.5GB: 硬性阻止（系统+Python 都不能正常运作）
    #   - 2.5GB <= 可用 < 4GB: 警告 + 建议关闭GPU
    #   - 可用 >= 4GB: 正常放行（不再硬性阻止 4-6GB 场景）
    # ★ 修复: 估算值偏保守，6GB 实测可跑 (昨晚成功)
    report_train_months = window_params.get("train_months", 36)
    try:
        import psutil
        _avail_gb = psutil.virtual_memory().available / (1024**3)
        # ★ 修复: 把估算值拉低，避免误报"内存不足"
        # 旧: 8.8 + (train_months - 36) * 0.04 → 估计过满
        # 新: 4.5 + (train_months - 36) * 0.02 → 更贴近实际 RSS 增量
        #   实际: factor_df ~0.5-1GB (不在 run_m2 持有), 窗口数据 ~0.3GB,
        #     SHAP/模型 ~0.3GB, Prefetch ~0.2GB, 系统保留 ~0.8GB
        _estimated_gb = 4.5 + (report_train_months - 36) * 0.02
        if use_gpu:
            _estimated_gb += 0.5  # GPU Prefetcher + CUDA 上下文额外开销
        if _avail_gb < _estimated_gb:
            _msg = (
                f"⚠️ 内存可能不足：可用={_avail_gb:.1f}GB，"
                f"预估需要={_estimated_gb:.1f}GB "
                f"(train_months={report_train_months}, "
                f"gpu={use_gpu})。建议：1)关闭其他程序 2)降低train_months 3)关闭GPU模式"
            )
            logger.warning(_msg)
            if progress_callback:
                progress_callback(_msg)
            # ★ 关键: 不再硬性阻止 4-6GB 场景（用户明确说"慢点没事"）
            # 只在 < 2.5GB（系统已无法正常运作）时硬性阻止
            if _avail_gb < 2.5:
                raise MemoryError(
                    f"可用内存严重不足({_avail_gb:.1f}GB < 2.5GB)，"
                    f"系统+Python 都无法正常运作，请关闭其他程序后重试"
                )
    except MemoryError:
        raise
    except Exception:
        pass

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
        # ★ 修复: DataLoader/LabelMaker 内部也可能 print()，一并重定向
        with _redirect_output(progress_callback):
            loader = DataLoader(scheme=scheme)
            preloaded_factor_df = loader.load()
            preloaded_factor_df = LabelMaker().make_labels(preloaded_factor_df)
    except Exception as e:
        logger.warning(f"预加载数据失败，将使用默认方案: {e}")
        preloaded_factor_df = None

    # ★ 修复: run_m2 会大量 print()（每10窗口一行 + 启动/收尾总结 ~50行），
    # 之前会全部刷到 PowerShell 终端导致 "终端无法正常运行"。
    # 通过 _redirect_output 接管 stdout/ststderr，转发到 progress_callback
    # （即 Gradio 的 log_box_deploy），终端保持干净。
    # ★ 改造: compute_shap=True → 走完整 SHAP，
    #   让 M4 报告能产出 SHAP 因子归因。P1/P2 走 objective.py 不受此影响。
    # train_months 使用 Trial 优化出的值（从 window_params 取），
    # 不做强制覆盖。
    if progress_callback:
        progress_callback(
            f"M4 报告使用 train_months={report_train_months}（来自 Trial）")

    # ★ v5.1 修复: M2 运行前记录 factor_df 大小，用于 M4 阶段按需重载
    _factor_df_size_info = ""
    if preloaded_factor_df is not None:
        _factor_df_size_info = (
            f"rows={len(preloaded_factor_df):,}, "
            f"cols={len(preloaded_factor_df.columns)}, "
            f"mem={preloaded_factor_df.memory_usage(deep=True).sum() / 1024**3:.1f}GB"
        )
        if progress_callback:
            progress_callback(f"因子表大小: {_factor_df_size_info}")

    try:
        with _redirect_output(progress_callback):
            portfolios, stats = run_m2(
                lgbm_params=lgbm_params,
                xgbm_params=xgbm_params,
                feature_params=feature_params,
                lgbm_weight=lgbm_weight,
                train_months=report_train_months,
                fast_mode=False,
                compute_val_metrics=True,
                compute_shap=True,        # ★ M4 路径开 SHAP
                verbose=True,
                preloaded_factor_df=preloaded_factor_df,
                gpu_mode=use_gpu,
                disable_penalty=False,
                **run_kwargs_extra,
            )
    except Exception as e:
        # ★ 修复: 完整 traceback 写日志 + 推给 UI，
        #   不然用户只看到 "回测失败：{e}" 不知道 M2 哪一行炸的
        import traceback
        tb = traceback.format_exc()
        logger.error(f"M2 回测失败: {e}\n{tb}")
        if progress_callback:
            progress_callback(f"❌ M2 失败: {e}")
            progress_callback(tb[-2000:])  # 最近 2KB
        raise  # 让 app.py 兜底 catch 显示

    # ★ v5.1 关键修复: M2 完成后立即释放 preloaded_factor_df
    # 原因: M2 返回后 factor_df 已不再需要（portfolios 已生成），
    #   但 preloaded_factor_df 仍被本函数持有，占用 4-8GB。
    #   M4 归因需要 industry + Barra 因子列，简化到精简列集会丢列。
    #   修复: M2 完成后保留全量表给 M4 用 (短时间占用 ~5GB),
    #   M4 完成后立即释放。这样 M4 阶段只多占 ~5GB 持续 < 5s。
    #   关键: 必须 del 一次再重命名, 让原引用彻底释放 (Python 引用计数).
    preloaded_factor_df_full = preloaded_factor_df
    preloaded_factor_df = None
    import gc as _gc_mid
    _gc_mid.collect()

    result["metrics"] = {
        "avg_val_ic": stats.get("avg_val_ic", 0),
        "avg_ic_gap": stats.get("avg_ic_gap", 0),
        "penalized_rate": stats.get("low_confidence_months", 0) /
        max(stats.get("success", 1), 1),
        "success_windows": stats.get("success", 0),
        "total_windows": stats.get("total_windows", 0),
    }

    if progress_callback:
        progress_callback("M2回测完成，M3风控处理中...")

    m3_portfolios = None
    try:
        from m3_engine.m3_runner import run_m3
        m3_portfolios = run_m3(
            m2_path="output/all_portfolios_gpu.parquet"
                if Path("output/all_portfolios_gpu.parquet").exists()
                else "output/all_portfolios.parquet",
            scheme=scheme,
            progress_callback=progress_callback,
        )
        if m3_portfolios is not None:
            if progress_callback:
                progress_callback("M3风控处理完成，生成报告...")
        else:
            if progress_callback:
                progress_callback("⚠️ M3风控失败，已回退至纯M2模式，生成报告...")
    except Exception as e:
        logger.warning(f"M3 集成异常: {e}")
        m3_portfolios = None
        if progress_callback:
            progress_callback(f"⚠️ M3风控失败: {e}，已回退至纯M2模式，生成报告...")

    try:
        from m4_report.report_generator import generate_report
        # ★ v5.0: 报告路径加入 tag 标识，防止 Tab2/Tab4 同时回测互相覆盖
        if report_tag:
            report_path = f"output/backtest_report_{scheme}_{report_tag}.html"
        else:
            report_path = f"output/backtest_report_{scheme}.html"
        # ★ 改造: 把 M0 全量因子表传给 M4，归因需要 industry/Barra
        #   SHAP 列（shap_top1/2/3_factor/value）已嵌在 portfolios 里
        # ★ v5.1 修复: 必须传 preloaded_factor_df 全量表（不是精简版）
        # 原因: M4 归因需要 industry + 所有 Barra 因子列（约 14 列），
        #   精简版只取了部分列，可能缺 critical 列。同时精简版占用 ~0.5GB，
        #   M4 报告阶段完成后立即释放，OOM 风险低。
        with _redirect_output(progress_callback):
            m4_result = generate_report(
                portfolios,
                stats=stats,
                output_path=report_path,
                factor_df=preloaded_factor_df_full,
                m3_portfolios=m3_portfolios,
            )
        result["report_path"] = m4_result.get("report_path", report_path)
        if "metrics" in m4_result:
            result["metrics"].update(m4_result["metrics"])
        if "attribution" in m4_result:
            result["attribution"] = m4_result["attribution"]
        if "summary_cards_m2" in m4_result:
            result["summary_cards_m2"] = m4_result["summary_cards_m2"]
        if "summary_cards_m3" in m4_result:
            result["summary_cards_m3"] = m4_result["summary_cards_m3"]
        if "m3_metrics" in m4_result and m4_result["m3_metrics"] is not None:
            result["m3_metrics"] = m4_result["m3_metrics"]
    except Exception as e:
        # ★ 修复: 失败原因 + 完整 traceback 写日志 + 推给 UI，
        # 不然用户点"打开回测报告" 报"文件不存在"却不知道 M4 阶段到底炸在哪
        import traceback
        tb = traceback.format_exc()
        msg = f"M4 报告生成失败: {e}"
        logger.error(f"{msg}\n{tb}")
        if progress_callback:
            progress_callback(f"❌ M4 {msg}")
            progress_callback(tb[-2000:])
        result["report_path"] = ""
    finally:
        # ★ v5.1: M4 完成后释放全量因子表（M4 报告不再需要）
        # 这步必须在 include_no_penalty 之前，确保 finally 总能跑
        preloaded_factor_df_full = None
        import gc as _gc_end
        _gc_end.collect()

    # 当前 is_penalized 不影响仓位，两次运行结果相同，跳过无降权对比
    if include_no_penalty:
        logger.info("当前版本 is_penalized 不影响仓位，跳过无降权对比")

    if progress_callback:
        progress_callback("完整回测完成！")

    return result
