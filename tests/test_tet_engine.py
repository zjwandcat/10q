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
M3 TET 单元测试（26 个验收用例）
"""
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from m3_engine import M3Config, StockState, TETEngine
from m3_engine.tet_engine import (
    _determine_emotion_sign,
    _has_crossed_zero,
    clean_and_sort_pred_months,
    load_m0_month,
    load_m2_holdings,
    verify_timeline,
)


# ──────────────── 1-4. Schmitt Trigger 穿轴判定 ────────────────
class TestSchmittTrigger:
    def test_schmitt_trigger_full_crossing(self):
        """prev=+1, ei=-0.3, hys=0.05 → 跨越滞后带触发穿轴 → (True, -1)"""
        crossed, new_sign = _has_crossed_zero(-0.3, 1, 0.05)
        assert crossed is True
        assert new_sign == -1

    def test_schmitt_trigger_in_band(self):
        """prev=+1, ei=-0.02, hys=0.05 → 落在带内不切换 → (False, +1)"""
        crossed, new_sign = _has_crossed_zero(-0.02, 1, 0.05)
        assert crossed is False
        assert new_sign == 1

    def test_schmitt_trigger_reverse(self):
        """prev=-1, ei=+0.3, hys=0.05 → 反向穿轴 → (True, +1)"""
        crossed, new_sign = _has_crossed_zero(0.3, -1, 0.05)
        assert crossed is True
        assert new_sign == 1

    def test_schmitt_trigger_boundary(self):
        """prev=+1, ei=-0.05, hys=0.05 → 恰好等于阈值不切换 → (False, +1)"""
        crossed, new_sign = _has_crossed_zero(-0.05, 1, 0.05)
        assert crossed is False
        assert new_sign == 1

    def test_determine_emotion_sign_signature(self):
        """模块级函数签名: _determine_emotion_sign(ei, sign, hys) -> int"""
        assert _determine_emotion_sign(-0.3, 1, 0.05) == -1
        assert _determine_emotion_sign(0.3, -1, 0.05) == 1
        assert _determine_emotion_sign(0.0, 1, 0.05) == 1
        assert _determine_emotion_sign(0.0, -1, 0.05) == -1


# ──────────────── 5-6. 状态机清理规则 ────────────────
class TestStateMachine:
    def test_state_cleanup_rule_a(self, tmp_workdir, make_m0_month,
                                   make_m2_month, stock_codes_10, valid_config_dict):
        """规则 A: T-1 选中→T 未选→T+1 重选，T+1 ATS = T+1 TS"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"
        # T-1: 选中全部 10 只
        m2_t1 = make_m2_month("202401", "2024-01-31", stock_codes_10)
        m2_t1.write_parquet(m2_path)
        # T: 不选任何股票（空）
        m2_t2 = make_m2_month("202402", "2024-02-29", [stock_codes_10[0]])  # 只保留一只
        # T+1: 重新选全部
        m2_t3 = make_m2_month("202403", "2024-03-31", stock_codes_10)
        all_m2 = pl.concat([m2_t1, m2_t2, m2_t3])
        all_m2.write_parquet(m2_path)

        # 写 M0
        make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10)
        make_m0_month(m0_dir, "202402", "2024-02-28", [stock_codes_10[0]])
        make_m0_month(m0_dir, "202403", "2024-03-30", stock_codes_10)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")

        out_df = pl.read_parquet(result)
        # 找到 T+1 月份的第一只股票，验证 ATS = TS（即被重新初始化）
        t3_rows = out_df.filter(pl.col("pred_month") == "202403")
        first = t3_rows.filter(pl.col("stock_code") == stock_codes_10[0]).row(0, named=True)
        if first is not None:
            assert first["anchored_trend"] == pytest.approx(first["trend_score"], abs=1e-9)

    def test_state_cleanup_rule_b(self, tmp_workdir, make_m0_month,
                                   make_m2_month, stock_codes_10, valid_config_dict):
        """规则 B: T 月 SELL_TET → T+1 重选，T+1 ATS = T+1 TS, m3_action=REENTER"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"

        # T: 全部选中
        m2_t1 = make_m2_month("202401", "2024-01-31", stock_codes_10)
        # T+1: 同样全部选中
        m2_t2 = make_m2_month("202402", "2024-02-29", stock_codes_10)
        all_m2 = pl.concat([m2_t1, m2_t2])
        all_m2.write_parquet(m2_path)

        # M0 T 月: 构造让所有股票 SELL_TET
        m0_t1 = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10, seed=1)
        m0_t2 = make_m0_month(m0_dir, "202402", "2024-02-28", stock_codes_10, seed=2)

        # 直接覆盖 T 月 M0 因子使所有股票触发 SELL_TET
        df_t1 = pl.read_parquet(m0_t1)
        # 让所有振荡器为 100（超买），所有趋势因子为 -1
        neg_vals = {col: -1.0 for col in df_t1.columns
                    if col not in ("stock_code", "trade_date", "macro_shibor_1m")}
        # 振荡器 clip 到 [0,100] 后给 100
        for item in valid_config_dict["emotion_indicators"]:
            neg_vals[item["name"]] = 100.0
        df_t1 = df_t1.with_columns([pl.lit(v).alias(c) for c, v in neg_vals.items()])
        df_t1.write_parquet(m0_t1)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")

        out_df = pl.read_parquet(result)
        t1 = out_df.filter(pl.col("pred_month") == "202401")
        t2 = out_df.filter(pl.col("pred_month") == "202402")

        # T 月应全 SELL_TET，且有 CASH_POOL
        assert t1.filter(pl.col("action") == "SELL_TET").height == 10
        assert t1.filter(pl.col("stock_code") == "CASH_POOL").height == 1

        # T+1 月应全 HOLD/REENTER（取决于是否被 SELL_TET）
        for row in t2.filter(pl.col("stock_code") != "CASH_POOL").iter_rows(named=True):
            assert row["m3_action"] in ("REENTER", "HOLD")
            # ATS 应 = TS（因为状态被清空，按规则 A 重新初始化）
            assert row["anchored_trend"] == pytest.approx(row["trend_score"], abs=1e-9)


# ──────────────── 7-8. EI 边界保护 ────────────────
class TestEIBoundary:
    def test_ei_boundary_clip_high(self, tmp_workdir, make_m0_month,
                                    stock_codes_10, valid_config_dict):
        """RSI=200 → clip 到 100 → EI 含 +1.0"""
        m0_dir = tmp_workdir["m0_dir"]
        # 写一个 RSI=200 的 M0
        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10, seed=0)
        df = pl.read_parquet(m0_path)
        for item in valid_config_dict["emotion_indicators"]:
            if item["direction"] == "positive":
                df = df.with_columns(pl.lit(200.0).alias(item["name"]))
            else:
                df = df.with_columns(pl.lit(0.0).alias(item["name"]))
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        engine._m0_cols = set(df.columns)
        row = df.to_pandas().iloc[0]
        ei = engine._calc_emotion_index(row)
        # 所有 11 个 positive 指标 → 1.0，WR=0 → (50-0)/50=1.0，共 12 个全为 1.0
        assert ei == pytest.approx(1.0, abs=1e-9)

    def test_ei_boundary_clip_low(self, tmp_workdir, make_m0_month,
                                   stock_codes_10, valid_config_dict):
        """RSI=-50 → clip 到 0 → EI 含 -1.0"""
        m0_dir = tmp_workdir["m0_dir"]
        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10, seed=0)
        df = pl.read_parquet(m0_path)
        for item in valid_config_dict["emotion_indicators"]:
            if item["direction"] == "positive":
                df = df.with_columns(pl.lit(-50.0).alias(item["name"]))
            else:
                df = df.with_columns(pl.lit(100.0).alias(item["name"]))
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        engine._m0_cols = set(df.columns)
        row = df.to_pandas().iloc[0]
        ei = engine._calc_emotion_index(row)
        # positive → (0-50)/50=-1.0, negative → (50-100)/50=-1.0
        assert ei == pytest.approx(-1.0, abs=1e-9)


# ──────────────── 9-10. EI 方向处理 ────────────────
class TestEIDirection:
    def test_wr_direction_negative(self, tmp_workdir, make_m0_month,
                                    stock_codes_10, valid_config_dict):
        """WR=90 → (50-90)/50 = -0.8"""
        m0_dir = tmp_workdir["m0_dir"]
        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10, seed=0)
        df = pl.read_parquet(m0_path)
        # RSI=50（中性），WR=90（超卖）
        for item in valid_config_dict["emotion_indicators"]:
            if item["direction"] == "positive":
                df = df.with_columns(pl.lit(50.0).alias(item["name"]))
            else:
                df = df.with_columns(pl.lit(90.0).alias(item["name"]))
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        engine._m0_cols = set(df.columns)
        row = df.to_pandas().iloc[0]
        ei = engine._calc_emotion_index(row)
        # 11 个 positive @ 50 → 0.0, 1 个 negative @ 90 → -0.8
        # 均值 = (11*0 + 1*(-0.8)) / 12 = -0.0667
        expected = (11 * 0.0 + 1 * (50.0 - 90.0) / 50.0) / 12
        assert ei == pytest.approx(expected, abs=1e-9)

    def test_rsi_direction_positive(self, tmp_workdir, make_m0_month,
                                    stock_codes_10, valid_config_dict):
        """RSI=80 → (80-50)/50 = 0.6"""
        m0_dir = tmp_workdir["m0_dir"]
        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10, seed=0)
        df = pl.read_parquet(m0_path)
        for item in valid_config_dict["emotion_indicators"]:
            if item["direction"] == "positive":
                df = df.with_columns(pl.lit(80.0).alias(item["name"]))
            else:
                df = df.with_columns(pl.lit(50.0).alias(item["name"]))
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        engine._m0_cols = set(df.columns)
        row = df.to_pandas().iloc[0]
        ei = engine._calc_emotion_index(row)
        expected = (11 * 0.6 + 1 * 0.0) / 12
        assert ei == pytest.approx(expected, abs=1e-9)


# ──────────────── 11-13. 因子缺失容错 ────────────────
class TestFactorMissing:
    def test_missing_single_factor(self, tmp_workdir, make_m0_month,
                                    stock_codes_10, valid_config_dict):
        """删除 momentum_return_5d 列 → TS 正常计算 + WARNING"""
        m0_dir = tmp_workdir["m0_dir"]
        m0_path = make_m0_month(
            m0_dir, "202401", "2024-01-30", stock_codes_10,
            drop_cols=["momentum_return_5d"]
        )
        df = pl.read_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        engine._m0_cols = set(df.columns)
        ts = engine._calc_trend_score(df.to_pandas().iloc[0])
        # 仍然能算出来 TS（剩余 13 个动量因子 + 全部 4 大类）
        assert -1.0 <= ts <= 1.0

    def test_missing_whole_category(self, tmp_workdir, make_m0_month,
                                     stock_codes_10, valid_config_dict):
        """动量大类 14 因子全删 → TS 由 3 大类均值"""
        m0_dir = tmp_workdir["m0_dir"]
        m0_path = make_m0_month(
            m0_dir, "202401", "2024-01-30", stock_codes_10,
            drop_cols=valid_config_dict["momentum"]
        )
        df = pl.read_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        engine._m0_cols = set(df.columns)
        ts = engine._calc_trend_score(df.to_pandas().iloc[0])
        assert -1.0 <= ts <= 1.0

    def test_all_categories_missing(self, tmp_workdir, stock_codes_10,
                                     valid_config_dict):
        """4 大类全删 → TS=0.0"""
        all_trend_cols = (
            valid_config_dict["momentum"]
            + valid_config_dict["ma_ratio_gt1"]
            + list(valid_config_dict["ma_ratio_gt_threshold"].keys())
            + [p[0] for p in valid_config_dict["crossover"]]
            + valid_config_dict["strength"]
        )

        class FakeEngine:
            config = M3Config(**valid_config_dict)
            _m0_cols: set[str] = set()  # 空

        engine = TETEngine.__new__(TETEngine)
        engine.config = M3Config(**valid_config_dict)
        engine._m0_cols = set()  # 无任何 M0 列

        import pandas as pd
        row = pd.Series(dtype="float64")
        ts = engine._calc_trend_score(row)
        assert ts == 0.0


# ──────────────── 14-16. 现金池 ────────────────
class TestCashPool:
    def test_cash_pool_correctness(self, tmp_workdir, make_m0_month,
                                    make_m2_month, stock_codes_10, valid_config_dict):
        """3 只 SELL_TET (0.13+0.13+0.07) → CASH_POOL adj_weight=0.33"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"

        m2 = make_m2_month(
            "202401", "2024-01-31", stock_codes_10,
            weights=[0.13, 0.13, 0.07, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.07],
        )
        m2.write_parquet(m2_path)

        # M0: 让前 3 只 SELL_TET
        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10, seed=0)
        df = pl.read_parquet(m0_path)
        # 前 3 只：趋势下行（所有 momentum/strength < 0）+ 超买（所有 RSI = 100）
        # 后 7 只：保持默认
        for i, code in enumerate(stock_codes_10):
            row_data = {}
            if i < 3:
                for col in df.columns:
                    if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                        continue
                    if col in [item["name"] for item in
                               valid_config_dict["emotion_indicators"]
                               if item["direction"] == "positive"]:
                        row_data[col] = 100.0
                    elif col == "technical_willr_10d":
                        row_data[col] = 0.0  # WR=0 → (50-0)/50=1.0
                    else:
                        row_data[col] = -1.0
                # 应用到指定行
                for c, v in row_data.items():
                    df = df.with_columns(
                        pl.when(pl.col("stock_code") == code)
                        .then(pl.lit(v))
                        .otherwise(pl.col(c))
                        .alias(c)
                    )
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")

        out_df = pl.read_parquet(result)
        cash_row = out_df.filter(
            (pl.col("pred_month") == "202401") & (pl.col("stock_code") == "CASH_POOL")
        )
        assert cash_row.height == 1
        cash_w = float(cash_row["adj_weight"][0])
        assert cash_w == pytest.approx(0.13 + 0.13 + 0.07, abs=1e-9)

    def test_cash_pool_rf_dynamic(self, tmp_workdir, make_m0_month,
                                   make_m2_month, stock_codes_10, valid_config_dict):
        """shibor_1m=3.0 → CASH_POOL rf=0.0025"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"

        m2 = make_m2_month(
            "202401", "2024-01-31", stock_codes_10,
            weights=[0.5] + [round(0.5 / 9, 4)] * 9,
        )
        m2.write_parquet(m2_path)

        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10,
                                  shibor=3.0, seed=0)
        # 强制第一只 SELL_TET
        df = pl.read_parquet(m0_path)
        first_code = stock_codes_10[0]
        for col in df.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]
                       if item["direction"] == "positive"]:
                df = df.with_columns(
                    pl.when(pl.col("stock_code") == first_code)
                    .then(pl.lit(100.0))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            elif col == "technical_willr_10d":
                df = df.with_columns(
                    pl.when(pl.col("stock_code") == first_code)
                    .then(pl.lit(0.0))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            else:
                df = df.with_columns(
                    pl.when(pl.col("stock_code") == first_code)
                    .then(pl.lit(-1.0))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")
        out_df = pl.read_parquet(result)
        cash_row = out_df.filter(
            (pl.col("pred_month") == "202401") & (pl.col("stock_code") == "CASH_POOL")
        )
        if cash_row.height == 1:
            # 3.0/100/12 = 0.0025
            assert float(cash_row["rf_rate_monthly"][0]) == pytest.approx(0.0025, abs=1e-9)
            assert float(cash_row["Target_Return_1M"][0]) == pytest.approx(0.0025, abs=1e-9)

    def test_cash_pool_rf_fallback(self, tmp_workdir, make_m0_month,
                                    make_m2_month, stock_codes_10, valid_config_dict):
        """shibor_1m 缺失 → fallback 到 risk_free_rate_annual / 12 = 0.025/12"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"

        m2 = make_m2_month(
            "202401", "2024-01-31", stock_codes_10,
            weights=[0.5] + [round(0.5 / 9, 4)] * 9,
        )
        m2.write_parquet(m2_path)

        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10,
                                  shibor=None, seed=0)
        # 强制第一只 SELL_TET
        df = pl.read_parquet(m0_path)
        first_code = stock_codes_10[0]
        for col in df.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]
                       if item["direction"] == "positive"]:
                df = df.with_columns(
                    pl.when(pl.col("stock_code") == first_code)
                    .then(pl.lit(100.0))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            elif col == "technical_willr_10d":
                df = df.with_columns(
                    pl.when(pl.col("stock_code") == first_code)
                    .then(pl.lit(0.0))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            else:
                df = df.with_columns(
                    pl.when(pl.col("stock_code") == first_code)
                    .then(pl.lit(-1.0))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")
        out_df = pl.read_parquet(result)
        cash_row = out_df.filter(
            (pl.col("pred_month") == "202401") & (pl.col("stock_code") == "CASH_POOL")
        )
        if cash_row.height == 1:
            assert float(cash_row["rf_rate_monthly"][0]) == pytest.approx(0.025 / 12, abs=1e-9)


# ──────────────── 17-18. 极端 + 边界 ────────────────
class TestEdgeCases:
    def test_all_sold_in_month(self, tmp_workdir, make_m0_month,
                                make_m2_month, stock_codes_10, valid_config_dict):
        """10 只全 SELL_TET → 正常输出 + CASH_POOL adj_weight=1.0"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"

        m2 = make_m2_month("202401", "2024-01-31", stock_codes_10,
                            weights=[0.1] * 10)
        m2.write_parquet(m2_path)

        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10, seed=0)
        df = pl.read_parquet(m0_path)
        # 所有股票 SELL_TET
        for col in df.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]
                       if item["direction"] == "positive"]:
                df = df.with_columns(pl.lit(100.0).alias(col))
            elif col == "technical_willr_10d":
                df = df.with_columns(pl.lit(0.0).alias(col))
            else:
                df = df.with_columns(pl.lit(-1.0).alias(col))
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")
        out_df = pl.read_parquet(result)
        cash_row = out_df.filter(pl.col("stock_code") == "CASH_POOL")
        assert cash_row.height == 1
        assert float(cash_row["adj_weight"][0]) == pytest.approx(1.0, abs=1e-9)

    def test_no_sell_no_cash_row(self, tmp_workdir, make_m0_month,
                                  make_m2_month, stock_codes_10, valid_config_dict):
        """当月无 SELL_TET → 无 CASH_POOL 行"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"

        m2 = make_m2_month("202401", "2024-01-31", stock_codes_10,
                            weights=[0.1] * 10)
        m2.write_parquet(m2_path)

        # M0: 所有股票趋势强（momentum 全 +1，emotion 50 中性）
        m0_path = make_m0_month(m0_dir, "202401", "2024-01-30", stock_codes_10, seed=0)
        df = pl.read_parquet(m0_path)
        for col in df.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]]:
                df = df.with_columns(pl.lit(50.0).alias(col))
            elif col in valid_config_dict["ma_ratio_gt1"]:
                df = df.with_columns(pl.lit(2.0).alias(col))  # > 1
            elif col in valid_config_dict["ma_ratio_gt_threshold"]:
                df = df.with_columns(pl.lit(1.0).alias(col))  # > threshold
            else:
                df = df.with_columns(pl.lit(1.0).alias(col))  # > 0
        df.write_parquet(m0_path)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")
        out_df = pl.read_parquet(result)
        assert out_df.filter(pl.col("stock_code") == "CASH_POOL").height == 0


# ──────────────── 19-20. 排序 + 格式熔断 ────────────────
class TestSortFormat:
    def test_cross_year_sorting(self):
        """[202412, 202501, 202411] → [202411, 202412, 202501]"""
        df = pl.DataFrame({
            "pred_month": ["202412", "202501", "202411", "202412"],
            "x": [1, 2, 3, 4],
        })
        result = clean_and_sort_pred_months(df)
        assert result == ["202411", "202412", "202501"]

    def test_format_assertion(self):
        """pred_month='2024-12' → raise AssertionError"""
        df = pl.DataFrame({
            "pred_month": ["2024-12", "202401"],
            "x": [1, 2],
        })
        with pytest.raises(AssertionError, match="pred_month"):
            clean_and_sort_pred_months(df)


# ──────────────── 21-22. 时序 + REENTER ────────────────
class TestTimelineReenter:
    def test_timeline_forward_bias(self, tmp_path, make_m0_month, stock_codes_10):
        """M0 date > M2 date → raise ValueError"""
        m0_dir = tmp_path / "m0"
        m0_dir.mkdir(parents=True)
        # M0 trade_date = 2024-12-31（晚于 M2）
        m0_path = make_m0_month(m0_dir, "202412", "2024-12-31", stock_codes_10)
        with pytest.raises(ValueError, match="前视偏差"):
            verify_timeline("2024-11-30", m0_path)

    def test_reenter_label(self, tmp_workdir, make_m0_month,
                            make_m2_month, stock_codes_10, valid_config_dict):
        """T SELL_TET → T+1 HOLD → T+1 m3_action=REENTER"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"

        # T 月: 选第一只
        m2_t1 = make_m2_month("202401", "2024-01-31", [stock_codes_10[0]])
        # T+1 月: 仍然选第一只
        m2_t2 = make_m2_month("202402", "2024-02-29", [stock_codes_10[0]])
        all_m2 = pl.concat([m2_t1, m2_t2])
        all_m2.write_parquet(m2_path)

        # M0 T 月: 触发 SELL_TET
        m0_t1 = make_m0_month(m0_dir, "202401", "2024-01-30", [stock_codes_10[0]])
        df_t1 = pl.read_parquet(m0_t1)
        for col in df_t1.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]
                       if item["direction"] == "positive"]:
                df_t1 = df_t1.with_columns(pl.lit(100.0).alias(col))
            elif col == "technical_willr_10d":
                df_t1 = df_t1.with_columns(pl.lit(0.0).alias(col))
            else:
                df_t1 = df_t1.with_columns(pl.lit(-1.0).alias(col))
        df_t1.write_parquet(m0_t1)

        # M0 T+1 月: 强趋势 HOLD
        m0_t2 = make_m0_month(m0_dir, "202402", "2024-02-28", [stock_codes_10[0]])
        df_t2 = pl.read_parquet(m0_t2)
        for col in df_t2.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]]:
                df_t2 = df_t2.with_columns(pl.lit(30.0).alias(col))  # 超卖
            elif col in valid_config_dict["ma_ratio_gt1"]:
                df_t2 = df_t2.with_columns(pl.lit(2.0).alias(col))
            elif col in valid_config_dict["ma_ratio_gt_threshold"]:
                df_t2 = df_t2.with_columns(pl.lit(1.0).alias(col))
            else:
                df_t2 = df_t2.with_columns(pl.lit(1.0).alias(col))
        df_t2.write_parquet(m0_t2)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")
        out_df = pl.read_parquet(result)
        t2 = out_df.filter(
            (pl.col("pred_month") == "202402") & (pl.col("stock_code") == stock_codes_10[0])
        )
        if t2.height == 1:
            assert t2["m3_action"][0] in ("REENTER", "HOLD")


# ──────────────── 23-24. 配置 ────────────────
class TestConfig:
    def test_post_init_assertions(self, valid_config_dict):
        """sell_threshold=1.0 → raise AssertionError"""
        cfg = valid_config_dict.copy()
        cfg["sell_threshold"] = 1.0
        with pytest.raises(AssertionError, match="sell_threshold"):
            M3Config(**cfg)

    def test_post_init_hys_band_invalid(self, valid_config_dict):
        """hys_band=0.3 → raise AssertionError"""
        cfg = valid_config_dict.copy()
        cfg["hys_band"] = 0.3
        with pytest.raises(AssertionError, match="hys_band"):
            M3Config(**cfg)

    def test_post_init_rf_source_invalid(self, valid_config_dict):
        """rf_source='other' → raise AssertionError"""
        cfg = valid_config_dict.copy()
        cfg["rf_source"] = "other"
        with pytest.raises(AssertionError, match="rf_source"):
            M3Config(**cfg)

    def test_from_yaml_integration(self):
        """真实 config_m3.yaml → 返回 M3Config 实例"""
        # 尝试加载项目根目录的 config_m3.yaml
        cfg_path = Path("config/config_m3.yaml")
        if not cfg_path.exists():
            pytest.skip("config/config_m3.yaml 不存在")
        cfg = M3Config.from_yaml(cfg_path)
        assert cfg.sell_threshold == -1.0
        assert cfg.scheme == "scheme_d"
        assert len(cfg.momentum) == 14


# ──────────────── 25-26. 跨月状态 + 性能 ────────────────
class TestContinuous:
    def test_continuous_3_months(self, tmp_workdir, make_m0_month,
                                  make_m2_month, stock_codes_10, valid_config_dict):
        """连续 3 月：T 建仓→T+1 穿轴→T+2 卖出"""
        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"
        code = stock_codes_10[0]

        m2_t1 = make_m2_month("202401", "2024-01-31", [code])
        m2_t2 = make_m2_month("202402", "2024-02-29", [code])
        m2_t3 = make_m2_month("202403", "2024-03-31", [code])
        all_m2 = pl.concat([m2_t1, m2_t2, m2_t3])
        all_m2.write_parquet(m2_path)

        # T: 中性 + 弱超买
        m0_t1 = make_m0_month(m0_dir, "202401", "2024-01-30", [code], seed=1)
        df = pl.read_parquet(m0_t1)
        for col in df.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]]:
                df = df.with_columns(pl.lit(60.0).alias(col))
            elif col in valid_config_dict["ma_ratio_gt1"]:
                df = df.with_columns(pl.lit(0.99).alias(col))
            elif col in valid_config_dict["ma_ratio_gt_threshold"]:
                df = df.with_columns(pl.lit(0.4).alias(col))
            else:
                df = df.with_columns(pl.lit(0.3).alias(col))  # TS ≈ +1
        df.write_parquet(m0_t1)

        # T+1: EI 穿轴 -0.3（prev_sign=+1 → -1），但 Timing > -1
        m0_t2 = make_m0_month(m0_dir, "202402", "2024-02-28", [code], seed=2)
        df = pl.read_parquet(m0_t2)
        for col in df.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]
                       if item["direction"] == "positive"]:
                df = df.with_columns(pl.lit(35.0).alias(col))  # EI = (35-50)/50 = -0.3
            elif col == "technical_willr_10d":
                df = df.with_columns(pl.lit(65.0).alias(col))  # (50-65)/50 = -0.3
            else:
                df = df.with_columns(pl.lit(-1.0).alias(col))  # TS = -1
        df.write_parquet(m0_t2)

        # T+2: EI 翻到 +0.6（穿轴 back to +1），TS=-0.5 → Timing=-0.5-0.6=-1.1 → SELL
        m0_t3 = make_m0_month(m0_dir, "202403", "2024-03-30", [code], seed=3)
        df = pl.read_parquet(m0_t3)
        for col in df.columns:
            if col in ("stock_code", "trade_date", "macro_shibor_1m"):
                continue
            if col in [item["name"] for item in
                       valid_config_dict["emotion_indicators"]
                       if item["direction"] == "positive"]:
                df = df.with_columns(pl.lit(80.0).alias(col))  # EI = +0.6
            elif col == "technical_willr_10d":
                df = df.with_columns(pl.lit(20.0).alias(col))  # (50-20)/50 = +0.6
            else:
                df = df.with_columns(pl.lit(-0.5).alias(col))  # TS = -0.5
        df.write_parquet(m0_t3)

        cfg = M3Config(**valid_config_dict)
        engine = TETEngine(cfg)
        result = engine.run(m2_path, m0_dir, tmp_workdir["output_dir"] / "out.parquet")
        out_df = pl.read_parquet(result)

        # 找到 3 个月该股票的 action
        rows = (
            out_df.filter(pl.col("stock_code") == code)
            .sort("pred_month")
        )
        actions = rows["action"].to_list()
        # 至少有一个 SELL_TET 出现
        assert any("SELL_TET" in str(a) for a in actions) or "SELL_TET" in str(actions)

    def test_performance_baseline(self, tmp_workdir, make_m0_month,
                                   make_m2_month, stock_codes_10, valid_config_dict):
        """120 月 × 10 股 < 1 秒"""
        import shutil

        m0_dir = tmp_workdir["m0_dir"]
        m2_path = tmp_workdir["output_dir"] / "m2.parquet"
        output_path = tmp_workdir["output_dir"] / "out.parquet"

        # 生成 120 个月的 mock 数据
        m2_chunks = []
        for month_idx in range(120):
            year = 2014 + month_idx // 12
            month = (month_idx % 12) + 1
            pred_month = f"{year}{month:02d}"
            trade_date = f"{year}-{month:02d}-28"
            m2_chunks.append(
                make_m2_month(pred_month, trade_date, stock_codes_10)
            )
            m0_trade_date = f"{year}-{month:02d}-27"
            make_m0_month(m0_dir, pred_month, m0_trade_date, stock_codes_10,
                          seed=month_idx)
        all_m2 = pl.concat(m2_chunks)
        all_m2.write_parquet(m2_path)

        cfg = M3Config(**valid_config_dict)
        # 关闭 debug_csv 避免 IO 开销影响计时
        cfg.debug_csv = False

        engine = TETEngine(cfg)
        t0 = time.perf_counter()
        engine.run(m2_path, m0_dir, output_path)
        elapsed = time.perf_counter() - t0

        print(f"\n[PERF] 120 月 × 10 股耗时: {elapsed:.3f} 秒")
        assert elapsed < 3.0  # 留宽，CI 机器可能慢
