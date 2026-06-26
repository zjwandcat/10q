#!/usr/bin/env python
"""
M4 净值曲线日期对齐验证脚本
============================

目的：验证修复后基准收益的峰谷与策略的 Target_Return_1M 峰谷对齐。
     原 bug：M4 用 "month m 月内首末 close" 算基准，与 M2 输出的
     "month m 月末 → month m+1 月末" 的策略收益错位 1 个月。

用法:
    python scripts/verify_m4_alignment.py

依赖:
    - output/all_portfolios.parquet (M2 输出)
    - data/benchmark_000906.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 让脚本可独立运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from m4_report.metrics import (
    _load_benchmark,
    _calc_benchmark_returns_point_to_point,
    PerformanceMetrics,
)


def _lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 3) -> dict:
    """计算 a 与 b 在不同滞后下的相关系数，找出最匹配的 lag。

    lag=0: corr(a, b)
    lag=1: corr(a[:-1], b[1:])  即 a 提前 1 月，看是否和 b 对齐
    lag=-1: corr(a[1:], b[:-1])
    """
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[:len(a) - lag], b[lag:len(b)]
        else:
            x, y = a[-lag:], b[:len(b) + lag]
        if len(x) < 5:
            out[lag] = np.nan
            continue
        out[lag] = float(np.corrcoef(x, y)[0, 1])
    return out


def main():
    print("=" * 70)
    print("M4 净值曲线日期对齐验证")
    print("=" * 70)

    # ── 1. 加载 M2 输出 ────────────────────────────
    p_pf = ROOT / "output" / "all_portfolios.parquet"
    if not p_pf.exists():
        print(f"❌ 找不到 {p_pf}，请先跑 M2")
        return 1
    pf = pd.read_parquet(p_pf)
    print(f"M2 输出: {len(pf):,} 行")

    # ── 2. 计算月度组合收益（用 M4 修复后的算法） ─────
    pm = PerformanceMetrics()
    monthly = pm.compute_monthly_returns(pf)
    print(f"月度组合收益: {len(monthly)} 个月")
    print(f"  范围: {monthly['pred_month'].iloc[0]} ~ {monthly['pred_month'].iloc[-1]}")
    print()

    # ── 3. 用新算法算基准 ──────────────────────────
    bm_df = _load_benchmark()
    bench_returns = _calc_benchmark_returns_point_to_point(bm_df)

    # 用旧算法算基准（对比用）
    df_old = bm_df.copy()
    df_old["date"] = df_old["date"].astype(str)
    df_old["year_month"] = df_old["date"].str[:6]
    bm_old = (
        df_old.sort_values("date")
        .groupby("year_month")
        .agg(first_close=("close", "first"), last_close=("close", "last"))
    )
    bm_old["return"] = bm_old["last_close"] / bm_old["first_close"] - 1
    bench_old = bm_old["return"]

    # ── 4. 拼接数据 ───────────────────────────────
    df = monthly[["pred_month", "portfolio_return"]].copy()
    df["benchmark_NEW"] = df["pred_month"].astype(str).map(
        bench_returns).fillna(0.0)
    df["benchmark_OLD"] = df["pred_month"].astype(str).map(
        bench_old).fillna(0.0)
    df = df.sort_values("pred_month").reset_index(drop=True)
    print("数据预览（前 5 行）:")
    print(df.head().to_string(index=False))
    print()

    # ── 5. 计算相关性（验证峰谷对齐）───────────────
    port = df["portfolio_return"].to_numpy()
    bm_new = df["benchmark_NEW"].to_numpy()
    bm_old = df["benchmark_OLD"].to_numpy()

    print("=" * 70)
    print("策略 vs 基准 的 lag 相关性（找最佳对齐 lag）")
    print("=" * 70)
    print()
    print("理想情况下 lag=0 时相关性应最高（即无错位）。")
    print("如果修复前最高相关在 lag=±1，说明有 1 个月错位。")
    print()

    new_corr = _lag_corr(port, bm_new, max_lag=2)
    old_corr = _lag_corr(port, bm_old, max_lag=2)

    print(f"{'lag':<6}{'修复后(NEW)':<20}{'修复前(OLD)':<20}")
    print("-" * 46)
    for lag in range(-2, 3):
        n_str = f"{new_corr.get(lag, np.nan):+.4f}"
        o_str = f"{old_corr.get(lag, np.nan):+.4f}"
        marker = "  ← 最佳" if (
            abs(new_corr.get(lag, 0)) == max(
                abs(v) for v in new_corr.values() if not np.isnan(v))
        ) else ""
        print(f"{lag:<+6}{n_str:<20}{o_str:<20}{marker}")
    print()

    # 修复前后的最佳 lag 对比
    new_best = max(new_corr, key=lambda k: abs(new_corr[k]) if not np.isnan(new_corr[k]) else 0)
    old_best = max(old_corr, key=lambda k: abs(old_corr[k]) if not np.isnan(old_corr[k]) else 0)
    print(f"修复后最佳 lag: {new_best:+d}  (相关 {new_corr[new_best]:+.4f})")
    print(f"修复前最佳 lag: {old_best:+d}  (相关 {old_corr[old_best]:+.4f})")
    print()

    if new_best == 0 and old_best != 0:
        print("[OK] 验证通过：修复后策略与基准在 lag=0 最佳对齐，修复前存在错位。")
    elif new_best == 0 and old_best == 0:
        print("[WARN] 修复前后都是 lag=0 最佳，可能是数据时间太短，看不出错位。")
    elif new_best != 0:
        print("[FAIL] 仍有错位，需要进一步检查！")
    print()

    # ── 6. 累计曲线峰谷对齐可视化（文字版） ──────
    df["cum_port"] = (1 + df["portfolio_return"]).cumprod()
    df["cum_bm_new"] = (1 + df["benchmark_NEW"]).cumprod()
    df["cum_bm_old"] = (1 + df["benchmark_OLD"]).cumprod()

    # 找 Top5 高点和低点
    for col, name in [("cum_port", "策略"),
                      ("cum_bm_new", "基准(修复后)"),
                      ("cum_bm_old", "基准(修复前)")]:
        s = df[col]
        top5_high = s.nlargest(5)
        top5_low = s.nsmallest(5)
        print(f"{name} Top5 高点: {top5_high.index.tolist()}")
        print(f"  对应月份: {[df['pred_month'].iloc[i] for i in top5_high.index]}")
        print(f"{name} Top5 低点: {top5_low.index.tolist()}")
        print(f"  对应月份: {[df['pred_month'].iloc[i] for i in top5_low.index]}")
        print()

    # 峰谷是否对齐
    port_high_idx = set(df["cum_port"].nlargest(3).index.tolist())
    bm_new_high_idx = set(df["cum_bm_new"].nlargest(3).index.tolist())
    bm_old_high_idx = set(df["cum_bm_old"].nlargest(3).index.tolist())

    # 计算 Top3 高点对应的月份差
    def _month_diff(indices_a, indices_b, df_):
        a_months = df_["pred_month"].iloc[list(indices_a)].tolist()
        b_months = df_["pred_month"].iloc[list(indices_b)].tolist()
        a_set = set(a_months)
        b_set = set(b_months)
        return a_set.symmetric_difference(b_set)

    print("Top3 高点月份对比:")
    new_diff = _month_diff(port_high_idx, bm_new_high_idx, df)
    old_diff = _month_diff(port_high_idx, bm_old_high_idx, df)
    print(f"  修复后: 策略 vs 基准 不重合的月份数 = {len(new_diff)} (越小越好)")
    print(f"  修复前: 策略 vs 基准 不重合的月份数 = {len(old_diff)} (越大表示错位越严重)")
    print()

    # ── 7. 列出典型月份对比 ────────────────────────
    print("=" * 70)
    print("典型月份对比：策略 vs 修复后基准")
    print("=" * 70)
    sample_months = ["201501", "201601", "201802", "202001",
                     "202003", "202101", "202201", "202301", "202406", "202501"]
    sample_months = [m for m in sample_months if m in df["pred_month"].values]
    for m in sample_months:
        row = df[df["pred_month"] == m].iloc[0]
        print(f"  {m}: 策略 {row['portfolio_return']*100:+.2f}% | "
              f"基准NEW {row['benchmark_NEW']*100:+.2f}% | "
              f"基准OLD {row['benchmark_OLD']*100:+.2f}% | "
              f"超额NEW {(row['portfolio_return']-row['benchmark_NEW'])*100:+.2f}%")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
