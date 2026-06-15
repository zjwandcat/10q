"""
M1时序引擎 - 独立运行入口
执行完整M1流水线：DataLoader → LabelMaker → RollingSplitter
将所有滚动窗口保存为parquet文件

用法：
    python -m m1_engine.run_m1
    python m1_engine/run_m1.py
"""
import sys
import json
import time
import logging
import pandas as pd
from pathlib import Path

# 确保项目根目录在sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from m1_engine.data_loader import DataLoader  # noqa: E402
from m1_engine.label_maker import LabelMaker  # noqa: E402
from m1_engine.rolling_splitter import RollingSplitter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("m1.run")

# ── 模块级常量 ──────────────────────────────────────
DEFAULT_OUTPUT_DIR = "output/m1_windows"
SEPARATOR = "=" * 60


def run_m1(
    force_rebuild: bool = True,
    output_dir: str = None,
    scheme: str = None,
):
    """
    M1主流程

    参数：
        force_rebuild: 是否覆盖已有输出
        output_dir: 窗口parquet输出目录，None时自动按scheme命名
        scheme: 中性化方案名（scheme_a/b/d/e），None时从config读取
    """
    start_time = time.time()

    # 自动推断output_dir
    if output_dir is None:
        if scheme is not None:
            output_dir = f"output/m1_windows_{scheme}"
        else:
            output_dir = DEFAULT_OUTPUT_DIR

    print(f"\n{SEPARATOR}")
    print("M1 时序引擎 - 滚动窗口生成")
    print(f"{SEPARATOR}")
    print(f"scheme:        {scheme or '(from config)'}")
    print(f"force_rebuild: {force_rebuild}")
    print(f"output_dir:    {output_dir}")

    # ── Step1: 数据加载 ──────────────────────────────
    print("\n[Step1] DataLoader - 加载M0数据")
    loader = DataLoader(scheme=scheme)
    print(f"  数据源: {loader.pool_dir}")

    factor_df = loader.load()
    print(f"  加载完成: {len(factor_df):,}行 × {len(factor_df.columns)}列")
    print(f"  日期范围: "
          f"{factor_df['trade_date'].min().strftime('%Y-%m')} ~ "
          f"{factor_df['trade_date'].max().strftime('%Y-%m')}")
    print(f"  月份数: {factor_df['trade_date'].nunique()}")

    # ── Step2: 标签生成 ──────────────────────────────
    print("\n[Step2] LabelMaker - 生成label_rank")
    factor_df = LabelMaker().make_labels(factor_df)

    coverage = factor_df["label_rank"].notna().mean()
    print(f"  label_rank覆盖率: {coverage:.1%}")

    # ── Step3: 滚动窗口切分 ──────────────────────────
    print("\n[Step3] RollingSplitter - 生成滚动窗口")
    splitter = RollingSplitter()
    n_windows = splitter.get_n_windows(factor_df)
    print(f"  预估窗口数: {n_windows}")

    # ── Step4: 保存窗口 ──────────────────────────────
    out_path = Path(output_dir)

    if force_rebuild and out_path.exists():
        # 清除旧输出
        import shutil
        shutil.rmtree(out_path)
        print(f"  已清除旧输出: {out_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step4] 保存窗口到 {out_path}")

    window_stats = []
    for window in splitter.split(factor_df):
        idx = window["window_idx"]
        pred_month = window["pred_month"]
        train_df = window["train_df"]
        val_df = window["val_df"]
        pred_df = window["pred_df"]

        # 每个窗口保存一个parquet（包含train/val/pred用标记列区分）
        train_df["_split"] = "train"
        val_df["_split"] = "val"
        pred_df["_split"] = "pred"

        combined = pd.concat(
            [train_df, val_df, pred_df], ignore_index=True)
        combined.to_parquet(
            out_path / f"window_{idx:03d}_{pred_month}.parquet",
            index=False)

        # 统计
        stat = {
            "window_idx": idx,
            "pred_month": pred_month,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "pred_rows": len(pred_df),
            "train_months": train_df["trade_date"].nunique(),
            "val_months": val_df["trade_date"].nunique(),
            "train_start": train_df["trade_date"].min().strftime("%Y%m"),
            "train_end": train_df["trade_date"].max().strftime("%Y%m"),
            "val_start": val_df["trade_date"].min().strftime("%Y%m"),
            "val_end": val_df["trade_date"].max().strftime("%Y%m"),
        }
        window_stats.append(stat)

        if idx % 20 == 0 or idx == n_windows - 1:
            print(f"  窗口{idx:03d}/{n_windows-1}: "
                  f"pred={pred_month} "
                  f"train={stat['train_rows']:,}行 "
                  f"val={stat['val_rows']:,}行 "
                  f"pred={stat['pred_rows']:,}行")

    # ── 汇总 ────────────────────────────────────────
    elapsed = time.time() - start_time

    stats_df = pd.DataFrame(window_stats)

    print(f"\n{SEPARATOR}")
    print("M1 运行完成")
    print(f"{SEPARATOR}")
    print(f"总窗口数:     {len(window_stats)}")
    print(f"输出目录:     {out_path}")
    print(f"时间范围:     {stats_df['train_start'].iloc[0]}~{stats_df['val_end'].iloc[-1]}")
    print(f"train平均行数: {stats_df['train_rows'].mean():,.0f}")
    print(f"val平均行数:   {stats_df['val_rows'].mean():,.0f}")
    print(f"pred平均行数:  {stats_df['pred_rows'].mean():,.0f}")
    print(f"运行时间:     {elapsed:.1f}秒")

    # 保存统计摘要
    summary = {
        "total_windows": len(window_stats),
        "output_dir": str(out_path),
        "time_range": f"{stats_df['train_start'].iloc[0]}~{stats_df['val_end'].iloc[-1]}",
        "avg_train_rows": int(stats_df["train_rows"].mean()),
        "avg_val_rows": int(stats_df["val_rows"].mean()),
        "avg_pred_rows": int(stats_df["pred_rows"].mean()),
        "elapsed_seconds": round(elapsed, 1),
        "data_scheme": str(loader.pool_dir),
        "scheme": scheme or "from_config",
    }

    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    stats_df.to_csv(out_path / "window_stats.csv", index=False)

    print("\n=== M1指纹 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary, window_stats


if __name__ == "__main__":
    run_m1(force_rebuild=True)
