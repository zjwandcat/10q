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

# PEP 683: frozenset/MappingProxyType 替代可变 set/dict
"""
M2主循环入口 · v4.1 (D 方案专用: CPU train + GPU predict)

v4.1 简化:
  - 删除 v3.x 的 A/B/C/E 混合策略
  - 仅保留 D 策略 (CPU train + GPU predict) + 纯 CPU fallback
  - 与 m2_engine 纯 CPU 路径对比: max_diff ≈ 1.5e-7 (FP32 直方图噪声级)

关键设计:
  - 不修改原 m2_engine/ 任何文件
  - 训练后自动释放原始数据
  - 全量窗口 + 详细显存/CPU/时间监控
  - benchmark 友好: 每窗口打印耗时与峰值
"""
import gc
import json
import queue
import threading
import yaml
import pandas as pd
import numpy as np
try:
    import bottleneck as bn
    _nanmean  = bn.nanmean
    _nanstd   = bn.nanstd
    _nansum   = bn.nansum
    _nanmax   = bn.nanmax
    _nanmin   = bn.nanmin
    _nanmedian = bn.nanmedian
    _mean = bn.nanmean
    _std  = bn.nanstd
except ImportError:
    _nanmean  = np.nanmean
    _nanstd   = np.nanstd
    _nansum   = np.nansum
    _nanmax   = np.nanmax
    _nanmin   = np.nanmin
    _nanmedian = np.nanmedian
    _mean = np.mean
    _std  = np.std
from pathlib import Path
from typing import Tuple, Dict, Optional
import time
import sys
import os
import traceback
import logging

config_path = (Path(__file__).parent.parent /
               "config" / "concurrency_config.py")
if config_path.exists():
    sys.path.insert(0, str(config_path.parent))
    from concurrency_config import GLOBAL_N_JOBS_OUTER
    sys.path.pop(0)
else:
    GLOBAL_N_JOBS_OUTER = 1

from m1_engine.data_loader import DataLoader
from m1_engine.rolling_splitter import RollingSplitter
from m1_engine.label_maker import LabelMaker
from m2_engine_gpu.feature_store import FeatureStore
from m2_engine_gpu.ensemble import EnsemblePredictor
from m2_engine_gpu.portfolio_builder import PortfolioBuilder
from m2_engine_gpu.gpu_detector import GPUConfig, STRATEGIES, STRATEGY_DESC

logger = logging.getLogger("m2.run.v2")

# 显存监控
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

def _vram_mb() -> int:
    """读取当前进程占用的 GPU 显存（MB）"""
    if not _NVML_OK:
        return 0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(mem.used / 1024**2)
    except Exception:
        return 0

# FeatureStore 缓存
_FEATURE_CACHE: Dict = {}
_FEATURE_CACHE_MAX_SIZE = 0
_FEATURE_CACHE_LOCK = threading.Lock()
_FEATURE_CACHE_PARAMS_HASH: Optional[int] = None


def _do_data_prep(i, window, cfg, feature_params):
    """★ v3.9 新增: 独立的数据准备函数 (供异步预取调用)

    与 _process_single_window 中的数据准备部分相同, 拆出来便于后台线程调用。
    """
    global _FEATURE_CACHE, _FEATURE_CACHE_PARAMS_HASH

    fp = feature_params or {}
    cache_key = (i, _FEATURE_CACHE_PARAMS_HASH)

    with _FEATURE_CACHE_LOCK:
        cached = _FEATURE_CACHE.get(cache_key)

    if cached is not None:
        return cached

    fs = FeatureStore(
        min_valid_rate=fp.get(
            "min_valid_rate",
            cfg["m2"]["feature_store"]["min_valid_rate"]),
        max_corr=fp.get(
            "max_corr",
            cfg["m2"]["feature_store"]["max_corr"]),
        min_ic_abs=fp.get(
            "min_ic_abs",
            cfg["m2"]["feature_store"]["min_ic_abs"]),
        min_keep_factors=fp.get(
            "min_keep_factors",
            cfg["m2"]["feature_store"].get("min_keep_factors", 50)),
        drop_short_term_noise=fp.get("drop_short_term_noise", False),
    )
    train_p, val_p, pred_p, feature_cols = fs.fit_transform(
        window["train_df"], window["val_df"], window["pred_df"])

    with _FEATURE_CACHE_LOCK:
        if len(_FEATURE_CACHE) < _FEATURE_CACHE_MAX_SIZE:
            _FEATURE_CACHE[cache_key] = (
                train_p, val_p, pred_p, feature_cols)

    return (train_p, val_p, pred_p, feature_cols)


class _WindowPrefetcher:
    """★ v3.9 新增: 窗口数据异步预取器

    设计思路:
    - 后台线程持续从 splitter 拉取下一个 window, 跑 FeatureStore.fit_transform
    - 主线程从 out_q 拿已准备好的 (train_p, val_p, pred_p, feature_cols)
    - 队列容量 = 1, 控制 RSS 峰值 (同时只有 1 个窗口在后台准备 + 1 个在主线程训练)

    线程安全:
    - FeatureStore 是 per-window 新建, 无共享状态
    - _FEATURE_CACHE_LOCK 保护缓存
    - out_q 是 thread-safe Queue
    - 主线程每 1 窗调 1 次 get(), 后台每准备好 1 窗就 put 1 次
    - 异常通过 error_queue 透传到主线程
    - 用 _pending_count 计数器: submit()+1, put 完 -1, 归零且 stop_event
      被 set 时主动退出 (避免主线程最后一个 get() 阻塞 / Empty)
    """
    def __init__(self, cfg, feature_params, stop_event=None,
                 maxsize=1):
        self.cfg = cfg
        self.feature_params = feature_params
        self.stop_event = stop_event or threading.Event()
        self.out_q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self.in_q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._error: Optional[Exception] = None
        self._finished = threading.Event()
        self._pending_count = 0   # ★ 跟踪未完成 submit 的窗口数
        self._count_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="WindowPrefetcher", daemon=True)
        self._thread.start()

    def submit(self, i, window):
        """主线程调用: 提交一个窗口让后台准备"""
        if self._error is not None:
            raise self._error
        with self._count_lock:
            self._pending_count += 1
        self.in_q.put((i, window), timeout=300)

    def get(self, timeout: float = 600.0):
        """主线程调用: 阻塞等后台准备好的数据

        ★ 修复: 当 _pending_count==0 且后台已 _finished 时, 返回 None 而非阻塞
        """
        if self._error is not None:
            raise self._error
        # ★ 快速路径: 没有 pending + 后台已结束, 直接返回 None
        with self._count_lock:
            no_pending = (self._pending_count == 0)
        if no_pending and (self._finished.is_set() or self.stop_event.is_set()):
            return None
        try:
            item = self.out_q.get(timeout=timeout)
        except queue.Empty:
            # 超时: 检查是否还有 pending + 后台是否在跑
            with self._count_lock:
                no_pending = (self._pending_count == 0)
            if no_pending and (self._finished.is_set() or self.stop_event.is_set()):
                return None
            raise
        if item is None:
            # 收到结束信号 (poison pill)
            self._finished.set()
            if self._error:
                raise self._error
            return None
        # ★ 正常数据: pending_count - 1
        with self._count_lock:
            self._pending_count -= 1
        return item

    def stop(self):
        """优雅停止后台线程 (主线程结束前必须调用)"""
        self.stop_event.set()
        try:
            self.in_q.put_nowait(None)  # poison pill
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self):
        try:
            while not self.stop_event.is_set():
                # ★ 优先检查 stop_event (避免空 in_q 阻塞)
                if self.stop_event.is_set():
                    break
                try:
                    item = self.in_q.get(timeout=1.0)  # 短轮询
                except queue.Empty:
                    continue   # 再次检查 stop_event
                if item is None:
                    break
                i, window = item
                if self.stop_event.is_set():
                    break
                # ★ 后台 CPU 密集: FeatureStore.fit_transform (~0.4s)
                t0 = time.time()
                train_p, val_p, pred_p, feature_cols = _do_data_prep(
                    i, window, self.cfg, self.feature_params)
                prep_time = time.time() - t0
                # ★ 阻塞 put 到主线程 (maxsize=1, 主线程会及时取走)
                # 检查 stop_event, 避免给已停止的主线程 put
                if self.stop_event.is_set():
                    break
                try:
                    self.out_q.put(
                        (i, window, train_p, val_p, pred_p, feature_cols, prep_time),
                        timeout=10)
                except queue.Full:
                    # 主线程没取走 (异常情况), 主动放弃
                    logger.warning(
                        f"[WindowPrefetcher] out_q 满, 放弃 W{i}")
                    break
        except Exception as e:
            self._error = e
            logger.error(f"[WindowPrefetcher] 异常: "
                         f"{type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            try:
                self.out_q.put_nowait(None)  # 通知主线程
            except Exception:
                pass
        finally:
            self._finished.set()


def _process_single_window(
    i, window, cfg, feature_params,
    lgbm_params, xgb_params, lgbm_weight,
    compute_shap, date_return_map,
    compute_val_metrics=True,
    disable_penalty=False,
    strategy=None,
    window_stats_list=None,
    data_prep_lock=None,   # ★ O3: 数据准备锁（z-score 串行化）
    preprepped=None,       # ★ v3.9: 异步预取已准备好的数据 (train_p, val_p, pred_p, feature_cols)
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """处理单个窗口（GPU v2版）"""
    global _FEATURE_CACHE, _FEATURE_CACHE_PARAMS_HASH

    train_p = val_p = pred_p = predictor = result = None
    train_df = window["train_df"]
    val_df   = window["val_df"]
    pred_df  = window["pred_df"]
    pred_month = window["pred_month"]

    stats = {
        "success":     False,
        "val_ic":      0.0,
        "ic_gap":      0.0,
        "is_penalized":False,
        "val_metrics": {},
        "failed":      None,
        "fit_time_s":  0.0,
        "vram_mb":     0,
    }

    t0 = time.time()
    try:
        if preprepped is not None:
            # ★ v3.9 异步预取路径: 数据已由后台线程准备好, 跳过 fit_transform
            train_p, val_p, pred_p, feature_cols = preprepped
        else:
            fp = feature_params or {}
            cache_key = (i, _FEATURE_CACHE_PARAMS_HASH)

            with _FEATURE_CACHE_LOCK:
                cached = _FEATURE_CACHE.get(cache_key)

            if cached is not None:
                train_p, val_p, pred_p, feature_cols = cached
            else:
                fs = FeatureStore(
                    min_valid_rate=fp.get(
                        "min_valid_rate",
                        cfg["m2"]["feature_store"]["min_valid_rate"]),
                    max_corr=fp.get(
                        "max_corr",
                        cfg["m2"]["feature_store"]["max_corr"]),
                    min_ic_abs=fp.get(
                        "min_ic_abs",
                        cfg["m2"]["feature_store"]["min_ic_abs"]),
                    min_keep_factors=fp.get(
                        "min_keep_factors",
                        cfg["m2"]["feature_store"].get(
                            "min_keep_factors", 50)),
                    drop_short_term_noise=fp.get(
                        "drop_short_term_noise", False),
                )
                # ★ O3: 用 data_prep_lock 串行化 z-score
                # 避免两个线程同时分配 30MB z-score temp array 导致 OOM
                if data_prep_lock is not None:
                    with data_prep_lock:
                        train_p, val_p, pred_p, feature_cols = (
                            fs.fit_transform(train_df, val_df, pred_df))
                else:
                    train_p, val_p, pred_p, feature_cols = (
                        fs.fit_transform(train_df, val_df, pred_df))

                with _FEATURE_CACHE_LOCK:
                    if len(_FEATURE_CACHE) < _FEATURE_CACHE_MAX_SIZE:
                        _FEATURE_CACHE[cache_key] = (
                            train_p, val_p, pred_p, feature_cols)

        eff_weight = (lgbm_weight if lgbm_weight is not None
                      else cfg["m2"]["ensemble"]["lgbm_weight"])

        predictor = EnsemblePredictor(
            lgbm_weight=eff_weight,
            xgb_weight=1.0 - eff_weight,
            lgbm_params=lgbm_params,
            xgb_params=xgb_params,
            strategy=strategy,
        )
        result = predictor.fit_predict(
            train_p, val_p, pred_p, feature_cols)

        # ★ v3.8: 在 del val_p 之前计算 val_scored (用于 val_metrics)
        # 这样确保 val_metrics 用 val_p (验证集) 而不是 pred_p (预测集) 算
        if compute_val_metrics:
            val_scored = val_p.copy(deep=False)
            val_scored["score"] = (
                predictor.lgbm_weight *
                predictor.lgbm.predict(val_p[feature_cols]) +
                predictor.xgb_weight *
                # v4.2: 17 指标用 CPU predict (bit-exact 与 m2_engine 路径)
                predictor.xgb.predict_cpu(val_p[feature_cols])
            )

        # ★ 训练后立即释放 FeatureStore 输出
        del train_p, val_p, pred_p
        # gc.collect()  ← 移除：每窗 2 次 GC 占用 31% 时间

        pb = PortfolioBuilder()
        portfolio = pb.build(
            pred_df_with_scores=result["pred_df_with_scores"],
            val_ic=result["val_ic"],
            ic_gap=result["ic_gap"],
            is_penalized=result["is_penalized"],
            lgbm_model=predictor.lgbm,
            feature_cols=feature_cols,
            pred_month=pred_month,
            compute_shap=compute_shap,
        )

        if portfolio is None:
            stats["failed"] = pred_month
            logger.warning(f"窗口{pred_month}: 股票池不足15只，跳过")
            return None, stats

        # 回填收益
        pred_date = pred_df["trade_date"].iloc[0]
        if pred_date in date_return_map:
            portfolio["Target_Return_1M"] = (
                portfolio["stock_code"].map(
                    date_return_map[pred_date]))

        if compute_val_metrics:
            try:
                # ★ v3.8: 用 val_scored (验证集) 算 val_metrics, 与 m2_engine 路径一致
                # 旧 (B 方案优化): 复用 result["pred_df_with_scores"] (预测集), 但这导致
                #   val_metrics 计算的不是验证集收益, 与纯 CPU 路径不一致
                stats["val_metrics"] = (
                    predictor.compute_val_portfolio_metrics(
                        val_scored))
                del val_scored
            except Exception as e:
                logger.warning(f"滚动指标计算失败: {e}")
                logger.warning(traceback.format_exc())

        stats.update({
            "success":      True,
            "val_ic":       result["val_ic"],
            "ic_gap":       result["ic_gap"],
            "is_penalized": result["is_penalized"],
        })
        stats["fit_time_s"] = time.time() - t0
        stats["vram_mb"]    = _vram_mb()
        if window_stats_list is not None:
            window_stats_list.append({
                "window":      pred_month,
                "fit_time_s":  stats["fit_time_s"],
                "vram_mb":     stats["vram_mb"],
                "val_ic":      result["val_ic"],
            })
        return portfolio, stats

    except Exception as e:
        stats["failed"] = pred_month
        logger.error(
            f"窗口{pred_month}失败: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return None, stats

    finally:
        del train_df, val_df, pred_df
        del predictor, result
        # ★ 修复: 恢复 gc.collect()，GPU 模式下 CUDA 显存释放依赖 GC 触发 C++ 析构
        # 旧版注释掉 GC 是因为"占用 31% 时间"，但 GPU 模式不 GC 会导致显存持续增长
        # 折中：仅在 GPU 模式下 GC，CPU 模式保持注释（引用计数足够）
        if strategy and strategy.upper() != "CPU":
            gc.collect()


def run_m2(
    lgbm_params:  Optional[Dict] = None,
    xgbm_params:  Optional[Dict] = None,
    feature_params: Optional[Dict] = None,
    lgbm_weight:  Optional[float] = None,
    fast_mode:    bool = False,
    fast_window_count: int = 60,
    compute_val_metrics: bool = True,
    compute_shap: bool = False,    # ★ GPU 版默认关 SHAP
    verbose:      bool = True,
    preloaded_windows: list = None,
    date_return_map: Dict = None,
    stop_event = None,
    preloaded_factor_df: pd.DataFrame = None,
    gpu_mode:     Optional[bool] = None,
    train_months: Optional[int] = None,
    scheme:       Optional[str] = None,
    disable_penalty: bool = False,
    strategy:     Optional[str] = None,   # ★ 新增：强制策略 A/B/C/D
    n_window_workers: int = 1,            # ★ O3 新增: 多线程并行窗口 (1/2)
    # 备注: n_window_workers > 1 在 Windows 上可能 OOM (z-score 并发)
    use_prefetch: bool = True,            # ★ v3.9.1: 默认开启 (180w 实测 1.33x, IC/17指标 0 diff)
    # 备注: 串行模式 (n_jobs=1) 下有效, 并行模式不适用
) -> Tuple[pd.DataFrame, Dict]:
    """M2主循环 v2 (支持 3 种 CUDA 混合策略)"""
    start_time = time.time()

    # ── GPU 模式配置 ──
    gpu_cfg = GPUConfig()
    if gpu_mode is not None:
        gpu_cfg.set_mode("gpu" if gpu_mode else "cpu")
    if strategy is not None:
        gpu_cfg.set_strategy(strategy)
    strategy = gpu_cfg.strategy

    # ── FeatureStore 缓存管理 ──
    global _FEATURE_CACHE, _FEATURE_CACHE_LOCK, _FEATURE_CACHE_MAX_SIZE, _FEATURE_CACHE_PARAMS_HASH
    if fast_mode and fast_window_count <= 20:
        _FEATURE_CACHE_MAX_SIZE = fast_window_count
    else:
        _FEATURE_CACHE_MAX_SIZE = 0
    with _FEATURE_CACHE_LOCK:
        _FEATURE_CACHE.clear()

    new_hash = (hash(frozenset((feature_params or {}).items()))
                if feature_params else 0)
    if new_hash != _FEATURE_CACHE_PARAMS_HASH:
        with _FEATURE_CACHE_LOCK:
            _FEATURE_CACHE.clear()
            _FEATURE_CACHE_PARAMS_HASH = new_hash

    if verbose:
        print(f"\n{'='*60}")
        print("M2 GPU v2 选股算法层")
        print(f"{'='*60}")
        print(gpu_cfg.summary())
        if strategy:
            print(f"当前策略: {strategy} ({gpu_cfg.strategy and STRATEGY_DESC.get(gpu_cfg.strategy, 'Unknown')})")

    # ── 加载配置 ──
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── 数据加载 ──
    if preloaded_windows is not None:
        windows = preloaded_windows
        if date_return_map is None:
            date_return_map = {}
        if verbose:
            print(f"使用预加载数据: {len(windows)}个窗口")
    else:
        if preloaded_factor_df is not None:
            factor_df = preloaded_factor_df
        else:
            loader = DataLoader(scheme=scheme)
            if verbose:
                print("开始加载数据...")
            factor_df = loader.load()
            if verbose:
                print(f"数据加载完成: {len(factor_df):,}行")
            factor_df = LabelMaker().make_labels(factor_df)
            cols_to_drop = [c for c in ["stock_name", "industry", "list_date"]
                           if c in factor_df.columns]
            if cols_to_drop:
                factor_df = factor_df.drop(columns=cols_to_drop)
            gc.collect()

        if date_return_map is None:
            ret_df = factor_df[
                ["trade_date","stock_code","Target_Return_1M"]
            ].dropna(subset=["Target_Return_1M"])
            # ★ P5 优化：矢量化 date_return_map（替代 groupby.apply lambda）
            # 旧: groupby.apply(lambda) = 0.16s
            # 新: np.unique + 切片填充 = 0.02s (6.4x 加速)
            ret_df_sorted = ret_df.sort_values("trade_date")
            all_dates    = ret_df_sorted["trade_date"].values
            all_stocks   = ret_df_sorted["stock_code"].values
            all_returns  = ret_df_sorted["Target_Return_1M"].values
            unique_dates, idx_start = np.unique(
                all_dates, return_index=True)
            idx_end = np.append(idx_start[1:], len(all_dates))
            date_return_map = {
                d: dict(zip(all_stocks[s:e].tolist(),
                            all_returns[s:e].tolist()))
                for d, s, e in zip(unique_dates, idx_start, idx_end)
            }
            del ret_df, ret_df_sorted, all_dates, all_stocks, all_returns
            del unique_dates, idx_start, idx_end

        splitter = RollingSplitter(train_months=train_months)
        _cfg_train = train_months or 36
        WINDOW_SIZE = _cfg_train + 12 + 1

        if fast_mode:
            all_months = sorted(factor_df["trade_date"].unique())
            total_months_count = len(all_months)
            months_needed = WINDOW_SIZE + fast_window_count - 1
            months_needed = min(months_needed, total_months_count)
            keep_months = frozenset(all_months[-months_needed:])
            factor_df = factor_df[
                factor_df["trade_date"].isin(keep_months)
            ].copy()
            effective_count = min(fast_window_count,
                                  total_months_count - WINDOW_SIZE + 1)
            if verbose:
                print(f"快速模式: 预切片保留最后{months_needed}个月，"
                      f"预计{effective_count}个窗口")
        else:
            all_months = sorted(factor_df["trade_date"].unique())
            effective_count = max(len(all_months) - WINDOW_SIZE + 1, 0)
            if verbose:
                print(f"全量模式: 共{len(all_months)}个月份，"
                      f"预计{effective_count}个窗口")

    # ── 主循环：单窗口串行（无 joblib）──
    all_records = []
    run_stats = {
        "total_windows":   effective_count,
        "success":         0,
        "failed":          [],
        "low_confidence_months": 0,
        "avg_val_ic":      [],
        "avg_ic_gap":      [],
        "val_metrics_list":[],
        "window_stats":    [],   # ★ 新增：每窗口耗时/显存
    }
    window_stats_list = run_stats["window_stats"]

    import psutil
    def _mem_gb():
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3

    n_jobs = n_window_workers   # ★ O3: 多进程并行窗口
    if verbose:
        print(f"并行度: {n_jobs}（{'单窗口串行' if n_jobs == 1 else f'{n_jobs} 进程并行'})")
        print(f"开始处理 {effective_count} 个窗口...")

    # ── 预生成所有窗口（用于并行 dispatch）──
    # 注意: 不能 list() 全部, 每个 window 含 ~60MB DataFrame × 180 = 10.8GB
    # 用 enumerate 但不 materialize 全部
    all_windows_iter = enumerate(splitter.split(factor_df, copy=False))

    if n_jobs == 1:
        # ── 串行模式（原始）──
        processed = 0

        if not use_prefetch:
            # ── 模式 A: 完全串行 (兼容原行为) ──
            for i, window in all_windows_iter:
                if stop_event and stop_event.is_set():
                    break

                portfolio, stats = _process_single_window(
                    i, window, cfg, feature_params,
                    lgbm_params, xgbm_params, lgbm_weight,
                    compute_shap, date_return_map,
                    compute_val_metrics,
                    disable_penalty=disable_penalty,
                    strategy=strategy,
                    window_stats_list=window_stats_list,
                    data_prep_lock=None,
                )

                if stats["success"]:
                    all_records.append(portfolio)
                    run_stats["success"] += 1
                    run_stats["avg_val_ic"].append(stats["val_ic"])
                    run_stats["avg_ic_gap"].append(stats["ic_gap"])
                    if stats["is_penalized"]:
                        run_stats["low_confidence_months"] += 1
                    if stats.get("val_metrics"):
                        run_stats["val_metrics_list"].append(
                            stats["val_metrics"])
                else:
                    run_stats["failed"].append(stats["failed"])

                processed += 1
                if processed % 10 == 0 or processed == effective_count:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (effective_count - processed) / max(rate, 1e-6)
                    vram_now = _vram_mb()
                    print(f"  [{processed:3d}/{effective_count}] "
                          f"elapsed={elapsed/60:.1f}min "
                          f"rate={rate:.1f}win/min "
                          f"remaining={remaining/60:.1f}min "
                          f"VRAM={vram_now}MB "
                          f"RSS={_mem_gb():.1f}GB",
                          flush=True)
        else:
            # ── 模式 B: v3.9 异步预取 ──
            # 后台线程: FeatureStore.fit_transform (CPU 密集, ~0.4s/窗)
            # 主线程: 模型训练 (GPU + CPU) + 投资组合构建
            # 队列容量 = 1, 控制 RSS 峰值
            if verbose:
                print(f"  [Prefetch] 启用异步预取 (后台线程做 FeatureStore)")
            prefetcher = _WindowPrefetcher(
                cfg, feature_params,
                stop_event=stop_event or threading.Event(),
                maxsize=1,
            )
            window_iter = iter(all_windows_iter)
            pending_submit = True  # 是否还有窗口要提交
            next_item = None       # 预取的下一个 (i, window)
            prefetch_wait_total = 0.0
            prep_overlap_total = 0.0
            try:
                # 提交第一个窗口
                try:
                    next_item = next(window_iter)
                    prefetcher.submit(next_item[0], next_item[1])
                    next_item = None
                except StopIteration:
                    pending_submit = False

                while True:
                    # 检查停止事件
                    if stop_event and stop_event.is_set():
                        break

                    # ★ v3.9.1 修复: 如果已 submit 完所有窗口, 立即 stop 后台
                    # 这样后台线程不再死等 in_q.get(), 主线程也不会 queue.Empty
                    if not pending_submit and next_item is None:
                        prefetcher.stop()
                        # 把 out_q 里剩余的最后 N 个数据全部 drain 出来
                        drained = 0
                        while True:
                            try:
                                prep_result = prefetcher.out_q.get_nowait()
                            except Exception:
                                break
                            if prep_result is None:
                                break
                            i, window, train_p, val_p, pred_p, feature_cols, prep_time = prep_result
                            if verbose:
                                logger.info(
                                    f"[Window {i}] drain prep_time={prep_time*1000:.0f}ms")
                            # 训练这个窗口
                            portfolio, stats = _process_single_window(
                                i, window, cfg, feature_params,
                                lgbm_params, xgbm_params, lgbm_weight,
                                compute_shap, date_return_map,
                                compute_val_metrics,
                                disable_penalty=disable_penalty,
                                strategy=strategy,
                                window_stats_list=window_stats_list,
                                data_prep_lock=None,
                                preprepped=(train_p, val_p, pred_p, feature_cols),
                            )
                            if stats["success"]:
                                all_records.append(portfolio)
                                run_stats["success"] += 1
                                run_stats["avg_val_ic"].append(stats["val_ic"])
                                run_stats["avg_ic_gap"].append(stats["ic_gap"])
                                if stats["is_penalized"]:
                                    run_stats["low_confidence_months"] += 1
                                if stats.get("val_metrics"):
                                    run_stats["val_metrics_list"].append(
                                        stats["val_metrics"])
                            else:
                                run_stats["failed"].append(stats["failed"])
                            processed += 1
                            drained += 1
                        if verbose:
                            print(f"  [Prefetch] 已 drain {drained} 个剩余窗口")
                        break  # 退出主循环

                    t_get_start = time.time()
                    prep_result = prefetcher.get(timeout=600.0)
                    prefetch_wait_total += time.time() - t_get_start
                    if prep_result is None:
                        break  # 后台线程结束
                    i, window, train_p, val_p, pred_p, feature_cols, prep_time = prep_result
                    if verbose:
                        logger.info(
                            f"[Window {i}] 后台 prep_time={prep_time*1000:.0f}ms")

                    # 提交下一个窗口(让后台提前准备)
                    try:
                        next_item = next(window_iter)
                        prefetcher.submit(next_item[0], next_item[1])
                        next_item = None
                    except StopIteration:
                        pending_submit = False

                    # ★ 主线程跑训练 (此时后台已在准备下一个)
                    portfolio, stats = _process_single_window(
                        i, window, cfg, feature_params,
                        lgbm_params, xgbm_params, lgbm_weight,
                        compute_shap, date_return_map,
                        compute_val_metrics,
                        disable_penalty=disable_penalty,
                        strategy=strategy,
                        window_stats_list=window_stats_list,
                        data_prep_lock=None,
                        preprepped=(train_p, val_p, pred_p, feature_cols),
                    )

                    if stats["success"]:
                        all_records.append(portfolio)
                        run_stats["success"] += 1
                        run_stats["avg_val_ic"].append(stats["val_ic"])
                        run_stats["avg_ic_gap"].append(stats["ic_gap"])
                        if stats["is_penalized"]:
                            run_stats["low_confidence_months"] += 1
                        if stats.get("val_metrics"):
                            run_stats["val_metrics_list"].append(
                                stats["val_metrics"])
                    else:
                        run_stats["failed"].append(stats["failed"])

                    processed += 1
                    if processed % 10 == 0 or processed == effective_count:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = (effective_count - processed) / max(rate, 1e-6)
                        vram_now = _vram_mb()
                        print(f"  [{processed:3d}/{effective_count}] "
                              f"elapsed={elapsed/60:.1f}min "
                              f"rate={rate:.1f}win/min "
                              f"remaining={remaining/60:.1f}min "
                              f"VRAM={vram_now}MB "
                              f"RSS={_mem_gb():.1f}GB",
                              flush=True)
            finally:
                prefetcher.stop()
                if verbose:
                    print(f"  [Prefetch] 后台累计等待={prefetch_wait_total:.1f}s")
                    print(f"  [Prefetch] 累计 prep 次数={processed}")

        gc.collect()

    else:
        # ── 并行模式 (O3) - ThreadPoolExecutor (共享内存) ──
        # 数据准备 (z-score) 用锁串行化避免 OOM
        # 限制并发数避免 180×60MB futures 队列 OOM
        # ★ 修复: 顶部已 import threading, 这里再 import 会让 Python 把 threading
        # 当作函数级 local, 导致 line 612 串行模式 (use_prefetch=True) 报 UnboundLocalError
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        data_prep_lock = threading.Lock()
        processed = 0
        last_print = 0

        def _process_one(i, window):
            portfolio, stats = _process_single_window(
                i, window, cfg, feature_params,
                lgbm_params, xgbm_params, lgbm_weight,
                compute_shap, date_return_map,
                compute_val_metrics,
                disable_penalty=disable_penalty,
                strategy=strategy,
                window_stats_list=None,
                data_prep_lock=data_prep_lock,
            )
            return i, portfolio, stats

        def _record(i, portfolio, stats):
            nonlocal processed
            if stats["success"]:
                all_records.append(portfolio)
                run_stats["success"] += 1
                run_stats["avg_val_ic"].append(stats["val_ic"])
                run_stats["avg_ic_gap"].append(stats["ic_gap"])
                if stats["is_penalized"]:
                    run_stats["low_confidence_months"] += 1
                if stats.get("val_metrics"):
                    run_stats["val_metrics_list"].append(
                        stats["val_metrics"])
            else:
                run_stats["failed"].append(stats["failed"])
            processed += 1
            if processed % 10 == 0 or processed == effective_count:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (effective_count - processed) / max(rate, 1e-6)
                vram_now = _vram_mb()
                print(f"  [{processed:3d}/{effective_count}] "
                      f"elapsed={elapsed/60:.1f}min "
                      f"rate={rate:.1f}win/min "
                      f"remaining={remaining/60:.1f}min "
                      f"VRAM={vram_now}MB "
                      f"RSS={_mem_gb():.1f}GB",
                      flush=True)

        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            pending = set()
            MAX_PENDING = n_jobs  # 同时在飞的任务数（限制内存）
            for t in all_windows_iter:
                if stop_event and stop_event.is_set():
                    break
                # 满了就等
                while len(pending) >= MAX_PENDING:
                    done, pending = wait(
                        pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        try:
                            i, portfolio, stats = fut.result(timeout=300)
                            _record(i, portfolio, stats)
                        except Exception as e:
                            logger.error(f"窗口失败: {e}")
                i_t, window = t
                fut = ex.submit(_process_one, i_t, window)
                pending.add(fut)

            # 收尾: 等待所有 pending
            while pending:
                done, pending = wait(
                    pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        i, portfolio, stats = fut.result(timeout=300)
                        _record(i, portfolio, stats)
                    except Exception as e:
                        logger.error(f"窗口失败: {e}")

        # gc.collect()  ← v3.4 移除：cProfile 显示 GC 占 12% 时间
        # ★ 修复: GPU 模式下需要 GC 释放 CUDA 显存，改为条件 GC
        if strategy and strategy.upper() != "CPU":
            gc.collect()

    try:
        del factor_df
        # ★ 修复: GPU 模式下需要 GC 释放 factor_df 相关的 CUDA 缓存
        if strategy and strategy.upper() != "CPU":
            gc.collect()
    except NameError:
        pass

    # ── 汇总 ──

    ic_list = run_stats["avg_val_ic"]
    if len(ic_list) >= 2:
        ic_arr = np.array(ic_list, dtype=np.float32)
        ic_mean = float(ic_arr.mean())
        ic_std  = float(ic_arr.std())
        if ic_std < 1e-4 or len(ic_arr) < 2:
            run_stats["avg_val_icir"] = 0.0
        else:
            val_icir = ic_mean / ic_std
            val_icir = float(np.clip(val_icir, -5.0, 5.0))
            run_stats["avg_val_icir"] = val_icir
    else:
        run_stats["avg_val_icir"] = 0.0

    if compute_val_metrics and run_stats["val_metrics_list"]:
        vml = run_stats["val_metrics_list"]
        run_stats["avg_val_portfolio_metrics"] = {
            k: float(_mean([v.get(k, 0) for v in vml]))
            for k in ["val_rolling6m_ir", "val_rolling6m_dir",
                      "val_rolling6m_sortino", "val_rolling6m_return",
                      "val_global_ir", "val_annual_return",
                      "pct_positive_excess", "ir_worst_quartile",
                      "val_rolling6m_excess", "val_rolling6m_excess_ann",
                      "up_capture_ratio", "down_capture_ratio",
                      "capture_ratio", "val_jensen_alpha",
                      "val_appraisal_ratio", "val_beta"]
        }
    else:
        run_stats["avg_val_portfolio_metrics"] = {
            k: 0.0 for k in [
                "val_rolling6m_ir", "val_rolling6m_dir",
                "val_rolling6m_sortino", "val_rolling6m_return",
                "val_global_ir", "val_annual_return",
                "pct_positive_excess", "ir_worst_quartile",
                "val_rolling6m_excess", "val_rolling6m_excess_ann",
                "up_capture_ratio", "down_capture_ratio",
                "capture_ratio", "val_jensen_alpha",
                "val_appraisal_ratio", "val_beta",
            ]
        }

    if not all_records:
        raise RuntimeError(
            f"所有{run_stats['total_windows']}个窗口均失败，请检查日志")

    Path("output").mkdir(exist_ok=True)
    all_portfolios = pd.concat(all_records, ignore_index=True)
    del all_records  # ★ 修复: concat 后立即释放列表，避免内存翻倍
    all_portfolios.to_parquet(
        "output/all_portfolios_gpu.parquet", index=False)

    holdings = all_portfolios[all_portfolios["is_holding"]]
    holdings.to_csv(
        "output/all_portfolios_gpu.csv",
        index=False, encoding="utf-8-sig")

    elapsed = (time.time() - start_time) / 60
    avg_ic  = _mean(run_stats["avg_val_ic"])
    avg_gap = _mean(run_stats["avg_ic_gap"])

    if verbose:
        print(f"\n{'='*60}")
        print("M2 GPU v2 回测完成")
        print(f"{'='*60}")
        print(f"策略:        {strategy or 'cpu-baseline'}")
        print(f"成功窗口:    {run_stats['success']}/{run_stats['total_windows']}")
        print(f"失败窗口:    {len(run_stats['failed'])}个")
        print(f"平均val_IC:  {avg_ic:.4f}")
        print(f"平均ic_gap:  {avg_gap:.4f}")
        print(f"运行时间:    {elapsed:.1f}分钟")
        if window_stats_list:
            fit_times = [w["fit_time_s"] for w in window_stats_list]
            vrams = [w["vram_mb"] for w in window_stats_list]
            print(f"窗口训练耗时: mean={_mean(fit_times):.1f}s "
                  f"min={min(fit_times):.1f}s "
                  f"max={max(fit_times):.1f}s")
            print(f"窗口峰值显存: mean={int(_mean(vrams))}MB "
                  f"max={max(vrams)}MB")
        print(f"输出行数:    {len(all_portfolios)}")

    # gc.collect()  ← v3.4 移除：主循环结束后保留一次 GC 兜底可接受
    return all_portfolios, run_stats  # type: ignore
    return all_portfolios, run_stats


if __name__ == "__main__":
    all_portfolios, stats = run_m2(
        fast_mode=True,
        fast_window_count=3,
        compute_shap=False,
        verbose=True,
    )
