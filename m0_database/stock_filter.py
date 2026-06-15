"""
股票池筛选模块
筛选标准：
  1. 剔除ST/*ST/PT
  2. 上市满252个交易日
  3. 近20交易日日均换手率：1%~15%
  4. 总市值：100亿~3000亿
  5. 当月停牌超过5个交易日的剔除
  6. 股价低于2元的剔除（仙股）
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_filter_config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["data"]["stock_filter"]


def filter_stock_pool(
    df_month: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    对单月截面数据执行股票池筛选

    参数：
        df_month: 单月截面DataFrame，必须包含以下列：
            stock_name, days_listed, avg_turnover_rate,
            market_cap（单位：亿元）, suspend_days,
            close_price

    返回：
        筛选后的DataFrame
    """
    cfg = load_filter_config()
    n_before = len(df_month)

    # ── 条件1：剔除ST/*ST/PT ──────────────────────
    mask_st = ~df_month["stock_name"].str.contains(
        r"ST|PT", na=False, regex=True)

    # ── 条件2：上市满252交易日 ─────────────────────
    mask_listed = (
        df_month["days_listed"] >= cfg["min_listed_days"])

    # ── 条件3：换手率范围 1%~15% ───────────────────
    mask_turnover = (
        (df_month["avg_turnover_rate"] >= cfg["turnover_rate_min"]) &
        (df_month["avg_turnover_rate"] <= cfg["turnover_rate_max"])
    )

    # ── 条件4：市值范围 100亿~3000亿 ──────────────
    mask_mktcap = (
        (df_month["market_cap"] >= cfg["market_cap_min"]) &
        (df_month["market_cap"] <= cfg["market_cap_max"])
    )

    # ── 条件5：停牌天数≤5 ────────────────────────
    mask_suspend = (
        df_month["suspend_days"] <= cfg["max_suspend_days"])

    # ── 条件6：股价≥2元 ──────────────────────────
    mask_price = (
        df_month["close_price"] >= cfg["min_price"])

    # ── 合并所有条件 ──────────────────────────────
    final_mask = (mask_st & mask_listed & mask_turnover &
                  mask_mktcap & mask_suspend & mask_price)
    result = df_month[final_mask].copy()

    if verbose:
        n_after = len(result)
        print(f"  [股票池筛选] {n_before} → {n_after} "
              f"（剔除{n_before-n_after}只）")
        print(f"    ST剔除: {(~mask_st).sum()}")
        print(f"    上市不足: {(~mask_listed).sum()}")
        print(f"    换手率越界: {(~mask_turnover).sum()}")
        print(f"    市值越界: {(~mask_mktcap).sum()}")
        print(f"    停牌过多: {(~mask_suspend).sum()}")
        print(f"    仙股: {(~mask_price).sum()}")

    return result
