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
regenerator.py
=====================================
M0 多进程并发续跑生成器（4 套新方案 B1/B2/F/G）

工作模型（spec 3.1）：
  - 任务粒度: (scheme, month) 单元
  - 基准: scheme_b 已生成的月份列表
  - 并发: ProcessPoolExecutor(n_jobs)
  - 失败隔离: 单任务失败 → 记录 bad_months.json → 继续
  - 写入互斥: 每任务写独立 parquet 文件

入口:
    python m0_database/regenerator.py
    python m0_database/regenerator.py --schemes scheme_b1 scheme_b2 --n_jobs 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[regenerator][%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/m0/regenerator.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("m0.regenerator")

BAD_MONTHS_PATH = Path("output/m0/regenerator_bad_months.json")
BAD_MONTHS_PATH.parent.mkdir(parents=True, exist_ok=True)

# 4 套新方案
DEFAULT_NEW_SCHEMES = ["scheme_b1", "scheme_b2", "scheme_f", "scheme_g"]


# ════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════

def load_config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_months(pq_dir: Path) -> list[str]:
    return sorted(f.stem for f in Path(pq_dir).glob("*.parquet"))


def pool_dir_for(cfg: dict, scheme: str) -> Path:
    return Path(cfg["data"]["pool_dirs"][scheme])


# ════════════════════════════════════════════════
# 进程级 worker
# ════════════════════════════════════════════════

def process_one(
    scheme: str,
    month: str,
    project_root: str,
    phase: str = "neutralize",
) -> dict:
    """
    单任务：(scheme, month) → 跑 pipeline 写 parquet
    必须在模块顶层可被 pickle 序列化（multiprocessing 要求）

    phase:
      "cache"      → 跑 Tushare+因子计算，写 data/cache_m0_factors/{month}.parquet
      "neutralize" → 读 cache + apply_neutralization + 写方案目录
      "full"       → 原行为（cache+neutralize 一次跑完）
    """
    import os
    os.chdir(project_root)
    sys.path.insert(0, project_root)
    try:
        from m0_database.pipeline import run_pipeline
        run_pipeline(
            schemes=[scheme],
            months=[month],
            force_rebuild=False,
            phase=phase,
        )
        return {"scheme": scheme, "month": month, "phase": phase, "ok": True}
    except Exception as e:
        return {
            "scheme": scheme,
            "month":  month,
            "phase":  phase,
            "ok":     False,
            "error":  str(e),
            "trace":  traceback.format_exc(limit=4),
        }
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


# ════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════

def run_regenerator(
    schemes: Iterable[str] = DEFAULT_NEW_SCHEMES,
    n_jobs: int = 4,
    project_root: str | None = None,
    phase: str = "two_phase",
) -> int:
    """
    多进程并发续跑主入口

    参数:
        schemes: 待生成方案
        n_jobs: 并发进程数
        project_root: 项目根目录
        phase:
          "two_phase"（默认）:
            - Phase 1 cache: 1 worker 跑 228 月 cache（内存重）
            - Phase 2 neutralize: n_jobs worker 跑 912 任务（内存轻）
            - 内存峰值：~1.5GB
            - 预计耗时：~50-80 分钟
          "full":  每任务 = cache+neutralize 一次跑完（与旧版一致，内存大）
          "cache": 只跑 Phase 1
          "neutralize": 只跑 Phase 2

    返回:
        0 = 全部成功
        1 = 有失败任务
        2 = 配置/路径异常
    """
    cfg = load_config()
    project_root = project_root or str(Path.cwd())
    schemes = list(schemes)

    # 基准目录
    ref_dir = pool_dir_for(cfg, "scheme_b")
    if not ref_dir.exists() or not list(ref_dir.glob("*.parquet")):
        logger.error(
            f"基准目录 {ref_dir} 不存在或为空，无法确定月份列表。"
            f"请先生成 scheme_b。"
        )
        return 2

    ref_months = list_months(ref_dir)
    logger.info(
        f"基准 scheme_b 月份数={len(ref_months)}，"
        f"待生成方案={schemes}，并发={n_jobs}，phase={phase}"
    )

    if phase == "two_phase":
        rc1 = _run_phase1_cache(
            ref_months, project_root, n_jobs=1)  # Phase 1 单进程（重）
        if rc1 != 0:
            logger.error(f"Phase 1 cache 失败 rc={rc1}，中止")
            return rc1
        rc2 = _run_phase2_neutralize(
            schemes, ref_months, project_root, n_jobs=n_jobs)
        return rc2

    # 单 phase 模式
    if phase == "cache":
        return _run_phase1_cache(ref_months, project_root, n_jobs=1)
    if phase == "neutralize":
        return _run_phase2_neutralize(
            schemes, ref_months, project_root, n_jobs=n_jobs)
    # phase == "full"
    return _run_phase_full(
        schemes, ref_months, project_root, n_jobs=n_jobs)


def _run_phase1_cache(
    ref_months: list[str],
    project_root: str,
    n_jobs: int = 1,
) -> int:
    """
    Phase 1：跑 cache
    - 单进程（内存重，Tushare+因子计算 ~1.5GB/worker）
    - 任务粒度：(dummy_scheme, month)
    - 跳过已存在 cache 的月份
    """
    logger.info(f"=== Phase 1 cache: {len(ref_months)} 月 ===")
    total = 0
    for m in ref_months:
        cp = Path("data/cache_m0_factors") / f"{m}.parquet"
        if not cp.exists():
            total += 1
    logger.info(f"  待生成 cache: {total}")
    if total == 0:
        return 0

    bad: list[dict] = []
    t_start = time.time()
    done = 0

    # Phase 1 单进程串行（避免多进程内存翻倍）
    from m0_database.pipeline import run_pipeline
    for m in ref_months:
        cp = Path("data/cache_m0_factors") / f"{m}.parquet"
        if cp.exists():
            continue
        try:
            run_pipeline(
                schemes=["scheme_b1"],
                months=[m],
                force_rebuild=False,
                phase="cache",
            )
            done += 1
        except Exception as e:
            bad.append({"month": m, "reason": str(e)})
            done += 1
            logger.warning(f"  [cache fail {done}] {m}: {str(e)[:120]}")
        if done % 5 == 0 or done == total:
            elapsed = time.time() - t_start
            eta = elapsed / max(done, 1) * (total - done)
            logger.info(
                f"  [cache {done}/{total}] "
                f"elapsed={elapsed:.0f}s "
                f"ETA={eta:.0f}s bad={len(bad)}")

    logger.info(f"Phase 1 完成: 成功={total-len(bad)} 失败={len(bad)} "
                f"耗时={time.time()-t_start:.0f}s")
    if bad:
        BAD_MONTHS_PATH.write_text(
            json.dumps(bad, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if not bad else 1


def _run_phase2_neutralize(
    schemes: list[str],
    ref_months: list[str],
    project_root: str,
    n_jobs: int = 4,
) -> int:
    """
    Phase 2：跑中性化
    - 多进程（内存轻，只读 cache+OLS ~300MB/worker）
    - 任务粒度：(scheme, month)
    """
    cfg = load_config()
    logger.info(
        f"=== Phase 2 neutralize: {len(schemes)}方案 × "
        f"{len(ref_months)}月 = {len(schemes)*len(ref_months)} 任务 "
        f"× {n_jobs} worker ==="
    )

    tasks: list[tuple[str, str]] = []
    for s in schemes:
        for m in ref_months:
            out_path = pool_dir_for(cfg, s) / f"{m}.parquet"
            if not out_path.exists():
                tasks.append((s, m))
    total = len(tasks)
    logger.info(f"  待生成 neutralize: {total}")
    if total == 0:
        return 0

    bad: list[dict] = []
    t_start = time.time()
    done = 0
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {
                ex.submit(process_one, s, m, project_root, "neutralize"): (s, m)
                for s, m in tasks
            }
            for f in as_completed(futs):
                sm = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    res = {"ok": False, "error": str(e)}
                done += 1
                if not res.get("ok"):
                    bad.append({"scheme": sm[0], "month": sm[1],
                                "reason": res.get("error", "unknown")})
                    logger.warning(
                        f"  [失败 {done}/{total}] "
                        f"({sm[0]}, {sm[1]}): "
                        f"{str(res.get('error',''))[:120]}")
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - t_start
                    eta = elapsed / done * (total - done) if done else 0
                    logger.info(
                        f"  [进度 {done}/{total}] "
                        f"elapsed={elapsed:.0f}s "
                        f"ETA={eta:.0f}s bad={len(bad)}")
    except KeyboardInterrupt:
        logger.warning("Phase 2 用户中断")

    BAD_MONTHS_PATH.write_text(
        json.dumps(bad, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Phase 2 完成: 成功={total-len(bad)} 失败={len(bad)} "
                f"耗时={time.time()-t_start:.0f}s")
    if bad:
        logger.info(f"失败列表: {BAD_MONTHS_PATH}")
        return 1
    return 0


def _run_phase_full(
    schemes: list[str],
    ref_months: list[str],
    project_root: str,
    n_jobs: int = 4,
) -> int:
    """原 full 模式（每任务 = cache+neutralize 一次跑完，内存大）"""
    cfg = load_config()
    tasks: list[tuple[str, str]] = []
    for s in schemes:
        for m in ref_months:
            out_path = pool_dir_for(cfg, s) / f"{m}.parquet"
            if not out_path.exists():
                tasks.append((s, m))
    total = len(tasks)
    logger.info(f"=== full mode: {total} 任务 × {n_jobs} worker ===")
    if total == 0:
        return 0

    bad: list[dict] = []
    t_start = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {
            ex.submit(process_one, s, m, project_root, "full"): (s, m)
            for s, m in tasks
        }
        for f in as_completed(futs):
            sm = futs[f]
            try:
                res = f.result()
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            done += 1
            if not res.get("ok"):
                bad.append({"scheme": sm[0], "month": sm[1],
                            "reason": res.get("error", "unknown")})
            if done % 10 == 0 or done == total:
                logger.info(
                    f"  [{done}/{total}] "
                    f"elapsed={time.time()-t_start:.0f}s bad={len(bad)}")
    BAD_MONTHS_PATH.write_text(
        json.dumps(bad, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if not bad else 1


# ════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M0 多进程并发续跑生成器（4 套新方案）"
    )
    p.add_argument(
        "--schemes", nargs="+", default=DEFAULT_NEW_SCHEMES,
        help=f"待生成方案（默认: {' '.join(DEFAULT_NEW_SCHEMES)}）",
    )
    p.add_argument(
        "--n_jobs", type=int, default=2,
        help="Phase 2 并发进程数（Phase 1 cache 始终单进程）。默认 2。",
    )
    p.add_argument(
        "--phase", choices=["two_phase", "full", "cache", "neutralize"],
        default="two_phase",
        help=("执行阶段：two_phase（默认，推荐，内存安全）"
              " | full（旧版，内存大）"
              " | cache（仅 Phase 1）"
              " | neutralize（仅 Phase 2，需先有 cache）"),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rc = run_regenerator(
        schemes=args.schemes,
        n_jobs=args.n_jobs,
        phase=args.phase,
    )
    sys.exit(rc)
