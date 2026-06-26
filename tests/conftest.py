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
M3 TET 单元测试 conftest

提供 mock parquet fixture，覆盖正常/缺失/异常三种数据场景。
"""
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# 完整因子列表（与 config_m3.yaml 对齐）
ALL_MOMENTUM = [
    "momentum_return_5d", "momentum_return_10d", "momentum_return_20d",
    "momentum_return_40d", "momentum_return_60d", "momentum_return_90d",
    "momentum_return_120d", "momentum_return_180d", "momentum_return_240d",
    "sup_mom_jt_12_1", "technical_roc_10d", "technical_roc_20d",
    "technical_roc_60d", "technical_roc_120d",
]
ALL_MA_RATIO_GT1 = [
    "technical_price_ma_ratio_5d", "technical_price_ma_ratio_10d",
    "technical_price_ma_ratio_20d", "technical_price_ma_ratio_60d",
]
ALL_MA_RATIO_THR = {
    "technical_boll_pos_20d": 0.5,
    "jq_price_position_20d": 0.5,
    "jq_price_position_60d": 0.5,
    "jq_ma_deviation_20d": 0.0,
}
ALL_CROSSOVER = [
    ["technical_ma_10d", "technical_ma_60d"],
    ["technical_ma_20d", "technical_ma_120d"],
    ["technical_macd_dif", "technical_macd_dea"],
    ["sup_tech_ma_cross_5_20", 0],
    ["sup_tech_ma_cross_20_60", 0],
]
ALL_STRENGTH = [
    "sup_mom_strength_20d", "sup_mom_strength_60d", "sup_mom_strength_120d",
    "sup_mom_accel", "momentum_sharpe_20d", "momentum_sharpe_60d",
    "momentum_sharpe_120d", "momentum_sharpe_240d",
    "technical_momentum_10d", "technical_momentum_20d",
    "technical_macd_hist", "momentum_score", "momentum_weighted_20d",
]
ALL_EMOTION = [
    {"name": "momentum_rsi_6d", "direction": "positive"},
    {"name": "momentum_rsi_14d", "direction": "positive"},
    {"name": "momentum_rsi_20d", "direction": "positive"},
    {"name": "momentum_rsi_30d", "direction": "positive"},
    {"name": "sup_tech_rsi_6", "direction": "positive"},
    {"name": "sup_tech_rsi_14", "direction": "positive"},
    {"name": "sup_tech_rsi_24", "direction": "positive"},
    {"name": "technical_kdj_rsv_9d", "direction": "positive"},
    {"name": "technical_kdj_rsv_14d", "direction": "positive"},
    {"name": "technical_kdj_rsv_20d", "direction": "positive"},
    {"name": "technical_kdj_rsv_30d", "direction": "positive"},
    {"name": "technical_willr_10d", "direction": "negative"},
]


def _make_stock_df(stock_codes: list[str], trade_date: str, seed: int = 0) -> pl.DataFrame:
    """生成一组完整因子的 mock DataFrame。"""
    rng = np.random.default_rng(seed)
    n = len(stock_codes)
    data: dict[str, list] = {
        "stock_code": stock_codes,
        "trade_date": [trade_date] * n,
        "macro_shibor_1m": [3.0] * n,
    }
    # 所有列添加随机数据
    for col in ALL_MOMENTUM + ALL_STRENGTH:
        data[col] = rng.normal(0, 1, n).tolist()
    for col in ALL_MA_RATIO_GT1:
        data[col] = rng.normal(1.0, 0.1, n).tolist()
    for col in ALL_MA_RATIO_THR:
        data[col] = rng.normal(0.5, 0.2, n).tolist()
    for a, b in ALL_CROSSOVER:
        data[a] = rng.normal(0, 1, n).tolist()
        if b != 0:
            data[b] = rng.normal(0, 1, n).tolist()
    # 情绪振荡器 [0, 100]
    for item in ALL_EMOTION:
        col = item["name"]
        if item["direction"] == "positive":
            data[col] = rng.uniform(20, 80, n).tolist()
        else:  # WR 等反向
            data[col] = rng.uniform(20, 80, n).tolist()
    return pl.DataFrame(data)


def _make_m2_df(
    pred_month: str,
    trade_date: str,
    stock_codes: list[str],
    weights: list[float] | None = None,
) -> pl.DataFrame:
    """生成 mock M2 持仓。"""
    if weights is None:
        weights = [round(1.0 / len(stock_codes), 4)] * len(stock_codes)
    return pl.DataFrame(
        {
            "trade_date": [trade_date] * len(stock_codes),
            "pred_month": [pred_month] * len(stock_codes),
            "stock_code": stock_codes,
            "weight": weights,
            "is_holding": [True] * len(stock_codes),
            "stock_name": [f"Stock_{c}" for c in stock_codes],
            "Target_Return_1M": [0.05] * len(stock_codes),
        }
    )


@pytest.fixture
def tmp_workdir(tmp_path: Path):
    """临时工作目录：包含 M2 + M0 + config。"""
    m0_dir = tmp_path / "data" / "pool_v2_scheme_d"
    m0_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield {"m0_dir": m0_dir, "output_dir": output_dir, "tmp": tmp_path}
    # 清理（tmp_path 会自动清理）


@pytest.fixture
def stock_codes_10() -> list[str]:
    return [f"00000{i}.SZ" for i in range(1, 11)]


@pytest.fixture
def make_m0_month():
    """工厂函数：make_m0_month(pred_month, trade_date, stock_codes) -> m0_path"""
    def _factory(m0_dir: Path, pred_month: str, trade_date: str,
                 stock_codes: list[str], drop_cols: list[str] | None = None,
                 trade_date_override: str | None = None,
                 shibor: float | None = 3.0,
                 seed: int = 0) -> Path:
        df = _make_stock_df(stock_codes, trade_date_override or trade_date, seed=seed)
        if drop_cols:
            df = df.drop(drop_cols)
        if shibor is None and "macro_shibor_1m" in df.columns:
            df = df.drop("macro_shibor_1m")
        elif shibor is not None and "macro_shibor_1m" in df.columns:
            df = df.with_columns(pl.lit(shibor).alias("macro_shibor_1m"))
        path = m0_dir / f"{pred_month}.parquet"
        df.write_parquet(path)
        return path
    return _factory


@pytest.fixture
def make_m2_month():
    """工厂函数：make_m2_month(pred_month, trade_date, stock_codes, weights)"""
    return _make_m2_df


@pytest.fixture
def valid_config_dict() -> dict:
    """构造一个有效的 M3Config 字典。"""
    return {
        "sell_threshold": -1.0,
        "hys_band": 0.05,
        "risk_free_rate_annual": 0.025,
        "rf_source": "shibor_dynamic",
        "scheme": "scheme_d",
        "debug_csv": False,
        "momentum": ALL_MOMENTUM,
        "ma_ratio_gt1": ALL_MA_RATIO_GT1,
        "ma_ratio_gt_threshold": ALL_MA_RATIO_THR,
        "crossover": ALL_CROSSOVER,
        "strength": ALL_STRENGTH,
        "emotion_indicators": ALL_EMOTION,
    }
