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

# ★ 关键: 不再用 _shared_fs 复用 FitState
# 原因: 之前 P0-1 优化（跨窗口复用第1窗 _fit_state，只做 z-score）会
#   导致 fit_state 中的 keep_cols / factor_df_columns 与新窗口
#   train_df 不匹配，触发 KeyError，进而 "所有 161 窗口均失败"。
# 修复: 每窗新建 FeatureStore，安全稳定。
#   性能损失: 161 窗 × ~0.4s/fit_transform ≈ 60s，慢一点可接受。

# ★ v5.1: 首个窗口的失败异常 (供 run_m2 汇总时输出)
# 原因: 旧版 stats["failed"] 只存 pred_month (如 "202006")，
#   161 窗口全败时用户看不到原因。改为存详细异常。
_FIRST_ERROR: Optional[str] = None


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

    # ★ 修复: 每窗新建 FeatureStore (不再 _shared_fs 复用)
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
    stop_event=None,       # 窗口内可响应停止信号
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
        # 窗口内 stop_event 检查：数据准备之前
        if stop_event is not None and stop_event.is_set():
            stats["failed"] = pred_month
            return None, stats

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
                # ★ 修复: 每窗新建 FeatureStore (不再 _shared_fs 复用)
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

        # 窗口内 stop_event 检查：训练之前
        if stop_event is not None and stop_event.is_set():
            stats["failed"] = pred_month
            return None, stats

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
        # ★ v5.5 修复: 股票收益率获取失败的几个
        #   - 原因 1: pred_date 不在 date_return_map 里 → Target_Return_1M 列不存在
        #   - 原因 2: stock_code 不在 map 里 → 列为 NaN
        #   - 修复: 显式建列 (NaN), 并对未匹配的 stock_code 用 pred_df 内置 fallback.
        pred_date = pred_df["trade_date"].iloc[0]
        if pred_date in date_return_map:
            portfolio["Target_Return_1M"] = (
                portfolio["stock_code"].map(
                    date_return_map[pred_date]))
        else:
            if "Target_Return_1M" in pred_df.columns:
                _pred_return_map = pred_df.set_index("stock_code")[
                    "Target_Return_1M"].to_dict()
                portfolio["Target_Return_1M"] = (
                    portfolio["stock_code"].map(_pred_return_map))
                logger.warning(
                    f"窗口{pred_month}: pred_date={pred_date} 不在 "
                    f"date_return_map, 已用 pred_df 内置收益回填"
                    f"（{portfolio['Target_Return_1M'].notna().sum()}"
                    f"/{len(portfolio)} 匹配）")
            else:
                portfolio["Target_Return_1M"] = np.nan
                logger.warning(
                    f"窗口{pred_month}: pred_date={pred_date} 不在 "
                    f"date_return_map 且 pred_df 无 Target_Return_1M, "
                    f"该月收益全为 NaN")
        if (portfolio["Target_Return_1M"].isna().any()
                and "Target_Return_1M" in pred_df.columns):
            if "_pred_return_map" not in dir():
                _pred_return_map = pred_df.set_index("stock_code")[
                    "Target_Return_1M"].to_dict()
            mask = portfolio["Target_Return_1M"].isna()
            if mask.any():
                portfolio.loc[mask, "Target_Return_1M"] = (
                    portfolio.loc[mask, "stock_code"].map(_pred_return_map))
                still_na = portfolio["Target_Return_1M"].isna().sum()
                if still_na > 0:
                    logger.warning(
                        f"窗口{pred_month}: {still_na} 只股票二次 "
                        f"回填后仍无收益")

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
        # ★ v5.1 修复: failed 字段存详细异常 (原版只存 pred_month，无法定位 161 全败原因)
        import traceback as _tb
        _tb_str = _tb.format_exc()
        stats["failed"] = f"{pred_month}: {type(e).__name__}: {e}"
        # 同时存到 module-level _FIRST_ERROR，供 run_m2 汇总时输出
        global _FIRST_ERROR
        if _FIRST_ERROR is None:
            _FIRST_ERROR = (
                f"窗口{pred_month} 异常: {type(e).__name__}: {e}\n"
                f"{_tb_str}"
            )
        logger.error(
            f"窗口{pred_month}失败: {type(e).__name__}: {e}")
        logger.error(_tb_str)
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
    compute_shap: bool = False,
    verbose:      bool = True,
    preloaded_windows: list = None,
    date_return_map: Dict = None,
    stop_event = None,
    preloaded_factor_df: pd.DataFrame = None,
    gpu_mode:     Optional[bool] = None,
    train_months: Optional[int] = None,
    scheme:       Optional[str] = None,
    disable_penalty: bool = False,
    strategy:     Optional[str] = None,
    n_window_workers: int = 1,
    use_prefetch: bool = True,
    m5_optimize:  bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """M2主循环 v2 (支持 3 种 CUDA 混合策略)"""
    start_time = time.time()

    # ★ v5.1 修复: 重置首错缓存 + 低内存场景自动降级
    # 原因: 旧版 use_prefetch 默认 True 在低内存 (6GB) 场景下会因
    #   后台线程持有 factor_df 主引用 + out_q 数据副本而触发 OOM。
    # 方案: 可用 < 8GB 时强制 use_prefetch=False，使用纯串行。
    global _FIRST_ERROR
    _FIRST_ERROR = None
    try:
        import psutil as _ps
        _avail_gb = _ps.virtual_memory().available / (1024**3)
        if _avail_gb < 8.0 and use_prefetch:
            print(f"  [低内存模式] 可用内存={_avail_gb:.1f}GB < 8GB，"
                  f"自动关闭 Prefetch (use_prefetch=False)，"
                  f"避免后台线程持有 factor_df 主引用导致 OOM")
            use_prefetch = False
    except Exception:
        pass

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
        # ★ v5.1 修复: 保留原始 industry 列供 M4 归因使用
        # 原因: 旧版直接 drop(["stock_name","industry","list_date"])，
        #   但 M4 报告需要 industry 做行业归因。M4 阶段只能从 preloaded_factor_df
        #   重取，但 preloaded_factor_df 在 M2 完成后已被 del 释放 (见 result_analyzer)。
        # 修复: 在 run_m2 内部保存一份精简版 _m4_meta（trade_date/stock_code/industry），
        #   挂到 all_portfolios.attrs['m4_meta']，M4 报告按需取。
        if preloaded_factor_df is not None:
            factor_df = preloaded_factor_df
            _m4_meta = None  # M2 内部不需要，M4 直接用 preloaded_factor_df 的精简版即可
        else:
            _m4_meta = None
            loader = DataLoader(scheme=scheme)
            if verbose:
                print("开始加载数据...")
            factor_df = loader.load()
            if verbose:
                print(f"数据加载完成: {len(factor_df):,}行")
            factor_df = LabelMaker().make_labels(factor_df)
            # ★ v5.1 修复: 保留 industry 列 (M4 归因必需)，
            #   只 drop 与训练无关的 stock_name 和 list_date
            # ★ 关键: industry 在 FeatureStore.fit_transform 中是 META_COLS，
            #   不会被当成因子入模型 (见 feature_store.py L74)
            cols_to_drop = [c for c in ["stock_name", "list_date"]
                           if c in factor_df.columns]
            if cols_to_drop:
                factor_df = factor_df.drop(columns=cols_to_drop)
            gc.collect()

        if date_return_map is None:
            ret_df = factor_df[
                ["trade_date","stock_code","Target_Return_1M"]
            ].dropna(subset=["Target_Return_1M"])
            # ★ P0-4 优化: 用 np.argsort 替代 pandas sort_values，避免 ~1.5GB 副本
            # 旧: ret_df.sort_values("trade_date") 创建整个 DataFrame 排序副本
            # 新: np.argsort 只创建索引数组 (~数MB)，然后通过索引取值
            _dates = ret_df["trade_date"].values
            _order = np.argsort(_dates, kind="stable")
            all_dates   = _dates[_order]
            all_stocks  = ret_df["stock_code"].values[_order]
            all_returns = ret_df["Target_Return_1M"].values[_order]
            del _dates, _order  # 立即释放排序索引
            unique_dates, idx_start = np.unique(
                all_dates, return_index=True)
            idx_end = np.append(idx_start[1:], len(all_dates))
            date_return_map = {
                d: dict(zip(all_stocks[s:e].tolist(),
                            all_returns[s:e].tolist()))
                for d, s, e in zip(unique_dates, idx_start, idx_end)
            }
            del ret_df, all_dates, all_stocks, all_returns
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
            total_months_count = len(all_months)
            full_count = max(total_months_count - WINDOW_SIZE + 1, 0)
            FIXED_WINDOW_COUNT = 155
            if full_count > FIXED_WINDOW_COUNT:
                months_needed = WINDOW_SIZE + FIXED_WINDOW_COUNT - 1
                keep_months = frozenset(all_months[-months_needed:])
                factor_df = factor_df[
                    factor_df["trade_date"].isin(keep_months)
                ].copy()
                effective_count = FIXED_WINDOW_COUNT
            else:
                effective_count = full_count
            if verbose:
                print(f"全量模式: 强制{FIXED_WINDOW_COUNT}窗口，"
                      f"预计{effective_count}个窗口")

    # ── 主循环：单窗口串行（无 joblib）──
    all_records = []
    # ★ 修复: SHAP 收集 dict。portfolio_builder 会把
    #   shap_values numpy 数组写到 top20.attrs，pd.concat 多个
    #   top20 时 pandas 会比较 attrs，numpy 数组比较会炸
    #   "truth value of an array is ambiguous"。所以在每个
    #   append 前先提取到 shap_data 字典，再清空 attrs。
    shap_data: Dict[str, tuple] = {}

    def _collect_shap(portfolio):
        """提取 portfolio.attrs['shap_values'] 等到 shap_data 字典，
        然后清空 attrs 避免 pd.concat 失败。"""
        if (compute_shap
            and hasattr(portfolio, "attrs")
            and "shap_values" in portfolio.attrs):
            m = str(portfolio["pred_month"].iloc[0]) \
                if "pred_month" in portfolio.columns else "?"
            shap_data[m] = (
                portfolio.attrs["shap_values"],
                portfolio.attrs.get("shap_features", []),
                portfolio.attrs.get(
                    "shap_source", "shap.TreeExplainer"),
            )
            portfolio.attrs.clear()
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

    # ★ v5.1 修复: 全量模式渐进式 OOM 防御
    # 原因: train_months=50 全量回测时，内存可能持续增长直到 Windows 强杀进程
    # 方案: 每 10 个窗口检查一次可用内存，低于阈值时主动中止
    _OOM_AVAIL_LIMIT_GB = 0.8

    def _check_oom():
        """检查系统可用内存，低于阈值返回警告信息，否则返回 None"""
        try:
            _avail = psutil.virtual_memory().available / (1024**3)
            if _avail < _OOM_AVAIL_LIMIT_GB:
                return (
                    f"OOM_ABORT | 系统可用内存={_avail:.1f}GB < "
                    f"{_OOM_AVAIL_LIMIT_GB}GB，中止回测防止 Windows 强杀"
                )
        except Exception:
            pass
        return None

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
                    stop_event=stop_event,
                )

                if stats["success"]:
                    _collect_shap(portfolio)   # ★ 修复 attrs 炸 pd.concat
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
                    # ★ v5.1 修复: 每 10 窗检查系统可用内存
                    _oom_msg = _check_oom()
                    if _oom_msg is not None:
                        logger.critical(_oom_msg)
                        print(f"  ⚠️ {_oom_msg}")
                        break
            # ── 模式 B: v3.9 异步预取 ──
            # 后台线程: FeatureStore.fit_transform (CPU 密集, ~0.4s/窗)
            # 主线程: 模型训练 (GPU + CPU) + 投资组合构建
            # 队列容量 = 1, 控制 RSS 峰值
            if verbose:
                print(f"  [Prefetch] 启用异步预取 (后台线程做 FeatureStore)")
            prefetcher = _WindowPrefetcher(
                cfg, feature_params,
                stop_event=stop_event or threading.Event(),
                maxsize=2,  # ★ P1-7 优化: 提前准备2个窗口，减少流水线气泡
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
                                stop_event=stop_event,
                            )
                            if stats["success"]:
                                _collect_shap(portfolio)   # ★ 修复 attrs 炸
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
                    # prefetcher 后台卡住时 get() 抛 queue.Empty，需捕获避免崩溃
                    try:
                        prep_result = prefetcher.get(timeout=600.0)
                    except queue.Empty:
                        logger.warning("Prefetcher 超时 (600s)，跳过该窗口")
                        continue
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
                        stop_event=stop_event,
                    )

                    if stats["success"]:
                        _collect_shap(portfolio)   # ★ 修复 attrs 炸
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
                        # ★ v5.1 修复: 每 10 窗检查系统可用内存
                        _oom_msg = _check_oom()
                        if _oom_msg is not None:
                            logger.critical(_oom_msg)
                            print(f"  ⚠️ {_oom_msg}")
                            break
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
                stop_event=stop_event,
            )
            return i, portfolio, stats

        def _record(i, portfolio, stats):
            nonlocal processed
            if stats["success"]:
                _collect_shap(portfolio)   # ★ 修复 attrs 炸
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
                      "val_net_annual_return",
                      "pct_positive_excess", "ir_worst_quartile",
                      "val_rolling6m_excess", "val_rolling6m_excess_ann",
                      "up_capture_ratio", "down_capture_ratio",
                      "capture_ratio", "val_jensen_alpha",
                      "val_appraisal_ratio", "val_beta",
                      "val_var_95", "val_cvar_95", "val_pain_index",
                      "val_sqn",
                      "val_ic_stability", "val_ir_stability",
                      "turnover_penalty",
                      ]
        }
        # ★ 全期拼接重算6m_ir（净收益口径，按日期去重，与M4 metrics.py一致）
        _all_rets = []
        _all_bms = []
        _all_costs = []
        _all_dates = []
        for v in vml:
            _all_rets.extend(v.get("_monthly_returns", []))
            _all_bms.extend(v.get("_monthly_benchmarks", []))
            _all_costs.extend(v.get("_monthly_costs", []))
            _all_dates.extend(v.get("_monthly_dates", []))
        if len(_all_rets) >= 6 and len(_all_dates) == len(_all_rets):
            _dedup = {}
            for i, d in enumerate(_all_dates):
                _dedup[d] = i
            _idx = sorted(_dedup.values())
            _rets = np.array([_all_rets[i] for i in _idx], dtype=np.float64)
            _bms = np.array([_all_bms[i] for i in _idx], dtype=np.float64)
            _costs = np.array([_all_costs[i] for i in _idx], dtype=np.float64)
            _ex = _rets - _bms
            _n = len(_rets)
            _avm = run_stats["avg_val_portfolio_metrics"]

            from numpy.lib.stride_tricks import sliding_window_view
            _win_ex = sliding_window_view(_ex, 6)
            _win_ex_mean = _win_ex.mean(axis=1)
            _win_ex_std = win_ex_std = _win_ex.std(axis=1, ddof=1)
            _irs = np.where(
                _win_ex_std > 1e-8,
                _win_ex_mean / np.where(_win_ex_std > 1e-8, _win_ex_std, 1.0) * np.sqrt(12),
                0.0)
            _avm["val_rolling6m_ir"] = float(np.mean(_irs))

            _dirs = np.empty(_n - 5, dtype=np.float64)
            for j in range(_n - 5):
                _w_ex = _win_ex[j]
                _down_ex = _w_ex[_w_ex < 0]
                _down_std = (float(np.std(_down_ex, ddof=1))
                             if len(_down_ex) > 1
                             else max(float(win_ex_std[j]), 1e-8))
                _dirs[j] = float(_win_ex_mean[j]) / _down_std * np.sqrt(12)
            _avm["val_rolling6m_dir"] = float(np.mean(_dirs))

            _rf = 0.03 / 12
            _win_ret = sliding_window_view(_rets, 6)
            _sortinos = np.empty(_n - 5, dtype=np.float64)
            for j in range(_n - 5):
                _w_ret = _win_ret[j]
                _down_ret = _w_ret[_w_ret < _rf]
                _d_std = (float(np.std(_down_ret, ddof=1))
                          if len(_down_ret) > 1
                          else max(float(np.std(_w_ret, ddof=1)), 1e-6))
                _sortinos[j] = float(np.mean(_w_ret - _rf)) / _d_std * np.sqrt(12)
            _avm["val_rolling6m_sortino"] = float(np.mean(_sortinos))

            _cumrets = np.prod(1 + _win_ret, axis=1) - 1
            _avm["val_rolling6m_return"] = float(np.mean(_cumrets))

            _ex_std = float(np.std(_ex, ddof=1))
            _avm["val_global_ir"] = (float(np.mean(_ex)) / _ex_std * np.sqrt(12)
                                      if _ex_std > 1e-8 else 0.0)

            _cum_ret = np.prod(1 + _rets) - 1
            _avm["val_annual_return"] = ((1 + _cum_ret) ** (12 / _n) - 1
                                          if _n > 0 else 0.0)

            # _rets 已是净收益，不再重复扣费
            _net_rets = _rets
            _cum_net = np.prod(1 + _net_rets) - 1
            _avm["val_net_annual_return"] = ((1 + _cum_net) ** (12 / _n) - 1
                                              if _n > 0 else 0.0)

            _avm["pct_positive_excess"] = float(np.mean(_ex > 0))

            _sorted_ex = np.sort(_ex)
            _cutoff = max(1, _n // 4)
            _worst_q = _sorted_ex[:_cutoff]
            _worst_std = float(np.std(_worst_q, ddof=1))
            _avm["ir_worst_quartile"] = (float(np.mean(_worst_q) / _worst_std)
                                          if _worst_std > 1e-8 else 0.0)

            if _n >= 6:
                _excess_windows = sliding_window_view(_ex, 6)
                _rolling6_means = _excess_windows.mean(axis=1)
                _avm["val_rolling6m_excess"] = float(np.mean(_rolling6_means))
                _avm["val_rolling6m_excess_ann"] = _avm["val_rolling6m_excess"] * 12

            _up_mask = _bms > 0
            _dn_mask = _bms < 0
            if _up_mask.sum() >= 3 and _dn_mask.sum() >= 3:
                _up_cap = float(_rets[_up_mask].mean() / _bms[_up_mask].mean())
                _dn_cap = float(_rets[_dn_mask].mean() / _bms[_dn_mask].mean())
                _avm["up_capture_ratio"] = round(_up_cap, 4)
                _avm["down_capture_ratio"] = round(_dn_cap, 4)
                _avm["capture_ratio"] = round(_up_cap / _dn_cap if abs(_dn_cap) > 1e-6 else 0.0, 4)

            if _n >= 12:
                _rf_m = 0.03 / 12
                _y = _rets - _rf_m
                _x = _bms - _rf_m
                _x_mean = float(_x.mean())
                _y_mean = float(_y.mean())
                _ss_xx = float(((_x - _x_mean) ** 2).sum())
                if _ss_xx > 1e-12 and np.unique(_x).size >= 2:
                    _beta = float(((_x - _x_mean) * (_y - _y_mean)).sum() / _ss_xx)
                    _alpha = _y_mean - _beta * _x_mean
                    _resid = _y - (_alpha + _beta * _x)
                    _sigma = float(_resid.std(ddof=1))
                    _avm["val_jensen_alpha"] = round(_alpha * 12.0, 6)
                    _avm["val_appraisal_ratio"] = round(_alpha / _sigma if _sigma > 1e-8 else 0.0, 6)
                    _avm["val_beta"] = round(_beta, 6)

            _avm["val_var_95"] = float(np.percentile(_rets, 5))
            _var_th = _avm["val_var_95"]
            _avm["val_cvar_95"] = (float(_rets[_rets <= _var_th].mean())
                                    if (_rets <= _var_th).any() else _var_th)
            _cum_arr = np.cumprod(1 + _rets)
            _peak_arr = np.maximum.accumulate(_cum_arr)
            _dd_arr = (_cum_arr - _peak_arr) / _peak_arr
            _avm["val_pain_index"] = float(np.abs(_dd_arr).mean())
            _r_std = float(_rets.std(ddof=1))
            _avm["val_sqn"] = (float(_rets.mean()) / _r_std * np.sqrt(_n)
                                if _r_std > 1e-8 and _n > 1 else 0.0)
    else:
        run_stats["avg_val_portfolio_metrics"] = {
            k: 0.0 for k in [
                "val_rolling6m_ir", "val_rolling6m_dir",
                "val_rolling6m_sortino", "val_rolling6m_return",
                "val_global_ir", "val_annual_return",
                "val_net_annual_return",
                "pct_positive_excess", "ir_worst_quartile",
                "val_rolling6m_excess", "val_rolling6m_excess_ann",
                "up_capture_ratio", "down_capture_ratio",
                "capture_ratio", "val_jensen_alpha",
                "val_appraisal_ratio", "val_beta",
                "val_var_95", "val_cvar_95", "val_pain_index",
                "val_sqn",
                "val_ic_stability", "val_ir_stability",
                "turnover_penalty",
            ]
        }

    if not all_records:
        # ★ v5.1 修复: 输出首个详细异常，旧版只说"所有XX窗口均失败"无任何线索
        _err = _FIRST_ERROR or "（无具体异常被捕获）"
        raise RuntimeError(
            f"所有{run_stats['total_windows']}个窗口均失败。"
            f"首个失败原因: {_err}"
        )

    Path("output").mkdir(exist_ok=True)
    # ★ 修复: 兜底清理每个 record 的 attrs。即使某些 append 站点
    #   忘了调用 _collect_shap，pd.concat 也不会因为 numpy 数组比较
    #   而炸。SHAP 数据已由 _collect_shap 收集到 shap_data 字典。
    for r in all_records:
        if hasattr(r, "attrs"):
            r.attrs.clear()
    all_portfolios = pd.concat(all_records, ignore_index=True)
    del all_records

    all_portfolios.attrs["shap_data"] = shap_data

    if not m5_optimize:
        all_portfolios_for_save = all_portfolios.copy()
        all_portfolios_for_save.attrs = {}
        all_portfolios_for_save.to_parquet(
            "output/all_portfolios_gpu.parquet", index=False)
        del all_portfolios_for_save

        holdings = all_portfolios[all_portfolios["is_holding"]]
        holdings.to_csv(
            "output/all_portfolios_gpu.csv",
            index=False, encoding="utf-8-sig")
        del holdings

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



if __name__ == "__main__":
    all_portfolios, stats = run_m2(
        fast_mode=True,
        fast_window_count=3,
        compute_shap=False,
        verbose=True,
    )
