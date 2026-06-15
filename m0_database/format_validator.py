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
format_validator.py
=====================================
M0 续跑生成器配套的格式校验模块

铁律（spec 2.2 特征索引保护契约）：
  1) 结构一致性: df.columns.tolist() == ref_columns
  2) 精度一致性: 所有数值列 dtype == np.float32
  3) 索引完全对齐: df.index.equals(ref_df.index)  ← 严格轴对齐
  4) 元数据保护: 关键 Meta 列存在且 dtype 与基准一致
  5) 可选字节级索引校验: hashlib.sha256(df.index.values.tobytes())
                         只校验 index 字节，不校验因子值

严禁对整个 Parquet 文件或因子数据列做 SHA256 比对——新方案的因子值
必然与 scheme_b 不同，强行比对必然失败（逻辑死锁）。
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("m0.format_validator")

# 关键 Meta 列（必须存在）
KEY_META_COLS = (
    "stock_code",
    "trade_date",
    "industry",
    "market_cap",
    "Target_Return_1M",
    "close_price",
    "date",            # 与 trade_date 重复
    "ticker",          # 与 stock_code 重复
    "stock_name",
    "trade_date_dt",   # Timestamp
    "list_date",       # Timestamp
    "days_listed",
    "suspend_days",
    "avg_turnover_rate",   # 原始数据是 float64，豁免 dtype 校验
    "benchmark_return",
    "excess_return_1m",
)


def learn_expected_schema(
    scheme_b_dir: str | Path,
    expected_cols_default: int = 478,
) -> dict:
    """
    扫描 scheme_b 全部 parquet，学 expected_cols（取众数）+ ref_columns（首文件顺序）

    参数:
        scheme_b_dir: scheme_b 输出目录（基准）
        expected_cols_default: 无 parquet 时的默认列数

    返回:
        dict {
            "expected_cols": int,
            "ref_columns":   list[str],
            "ref_index_hash": str | None,  # 首个 parquet 的 index sha256
            "ref_index":      pd.Index,    # 首个 parquet 的 df.index
            "n_files":        int,
        }
    """
    scheme_b_dir = Path(scheme_b_dir)
    files = sorted(scheme_b_dir.glob("*.parquet"))
    if not files:
        logger.warning(
            f"[learn_expected_schema] {scheme_b_dir} 无 parquet，"
            f"用默认 expected_cols={expected_cols_default}"
        )
        return {
            "expected_cols":  expected_cols_default,
            "ref_columns":    [],
            "ref_index_hash": None,
            "ref_index":      pd.Index([]),
            "n_files":        0,
        }

    # 列数众数（用 pyarrow 只读 schema，避免 columns=[] 返回 0 列的 bug）
    import pyarrow.parquet as _pq
    col_counts: list[int] = []
    for f in files:
        try:
            n = len(_pq.ParquetFile(f).schema_arrow.names)
            col_counts.append(n)
        except Exception as e:
            logger.warning(f"  跳过 {f.name}: {e}")
    if not col_counts:
        return {
            "expected_cols":  expected_cols_default,
            "ref_columns":    [],
            "ref_index_hash": None,
            "ref_index":      pd.Index([]),
            "n_files":        0,
        }

    expected_cols = max(set(col_counts), key=col_counts.count)
    logger.info(
        f"[learn_expected_schema] scheme_b={len(files)} 个 parquet，"
        f"列数众数={expected_cols}"
    )

    # 取第一个列数==众数的合法 parquet 作 ref（保证 ref_columns 与 expected_cols 匹配）
    ref_file = None
    for f in files:
        if len(_pq.ParquetFile(f).schema_arrow.names) == expected_cols:
            ref_file = f
            break
    if ref_file is None:
        ref_file = files[0]
    ref_df = pd.read_parquet(ref_file)
    ref_columns = ref_df.columns.tolist()
    ref_index = ref_df.index
    ref_index_hash = hashlib.sha256(ref_index.values.tobytes()).hexdigest()

    return {
        "expected_cols":  int(expected_cols),
        "ref_columns":    ref_columns,
        "ref_index_hash": ref_index_hash,
        "ref_index":      ref_index,
        "n_files":        len(files),
    }


def validate_parquet(
    path: str | Path,
    ref_columns: list[str],
    expected_cols: int,
    ref_df: Optional[pd.DataFrame] = None,
    factor_cols: Optional[list[str]] = None,
    check_sha256_index: bool = True,
) -> dict:
    """
    对单个 parquet 执行 4 项硬核断言 + 可选字节级 index 校验

    参数:
        path: 待校验 parquet
        ref_columns: scheme_b 列名+顺序
        expected_cols: scheme_b 列数
        ref_df: scheme_b 当月 DataFrame（用于 index.equals 严格比对）
        factor_cols: 数值列名列表（默认 = ref_columns - KEY_META_COLS）
        check_sha256_index: 是否对 index 做 sha256 比对

    返回:
        dict {
            "valid":              bool,
            "reason":             str | None,
            "n_cols":             int,
            "n_stocks":           int,
            "index_match":        bool,
            "sha256_index_match": bool,
            "sha256_index_self":  str | None,
            "sha256_index_ref":   str | None,
        }
    """
    path = Path(path)
    result = {
        "valid":              False,
        "reason":             None,
        "n_cols":             0,
        "n_stocks":           0,
        "index_match":        False,
        "sha256_index_match": False,
        "sha256_index_self":  None,
        "sha256_index_ref":   None,
    }

    if not path.exists():
        result["reason"] = "file_not_found"
        return result

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        result["reason"] = f"read_error:{e}"
        return result

    result["n_cols"]   = int(df.shape[1])
    result["n_stocks"] = int(df.shape[0])

    # ── 断言 0：ref 中期望的关键 Meta 列必须在 df 中存在 ──
    # 放在 col_count_mismatch 之前，让 missing_meta 路径在"列数不对齐"
    # 时也能命中（避免被 col_count_mismatch 提前吞掉）。
    # 只检查 ref 中"确实需要"的 meta 子集：旧 v1 输出可能不含
    # date/ticker（已是 trade_date/stock_code 重复列），KEY_META_COLS
    # 中保留它们作为"可选豁免"，不强制所有产出都带。
    ref_required_meta = [
        c for c in KEY_META_COLS if c in ref_columns
    ]
    missing_meta = [c for c in ref_required_meta if c not in df.columns]
    if missing_meta:
        result["reason"] = f"missing_meta:{missing_meta}"
        return result

    # ── 断言 1：结构一致性 ──
    if df.columns.tolist() != ref_columns:
        result["reason"] = "col_count_mismatch" \
            if df.shape[1] != len(ref_columns) else "col_order_mismatch"
        return result

    # ── 断言 1.5：列数 = expected_cols ──
    if df.shape[1] != expected_cols:
        result["reason"] = "col_count_mismatch"
        return result

    # ── 断言 2：精度一致性（数值列 dtype=float32） ──
    fc = factor_cols or [c for c in df.columns if c not in KEY_META_COLS]
    for col in fc:
        if col not in df.columns:
            result["reason"] = f"missing_factor_col:{col}"
            return result
        if df[col].dtype != np.float32:
            result["reason"] = f"dtype_mismatch:{col}={df[col].dtype}"
            return result

    # ── 断言 3：索引完全对齐（严格轴对齐） ──
    if ref_df is not None:
        if not df.index.equals(ref_df.index):
            result["reason"] = "index_mismatch"
            return result
        result["index_match"] = True

    # ── 断言 4：元数据保护（已由断言 0 统一处理） ──
    # 关键 meta 缺失检查在读盘后立即执行（断言 0），
    # 此处不再重复，仅作为代码结构占位。

    # ── 断言 5（可选）：字节级 index 校验 ──
    if check_sha256_index:
        sha_self = hashlib.sha256(df.index.values.tobytes()).hexdigest()
        result["sha256_index_self"] = sha_self
        if ref_df is not None:
            sha_ref = hashlib.sha256(
                ref_df.index.values.tobytes()).hexdigest()
            result["sha256_index_ref"] = sha_ref
            result["sha256_index_match"] = (sha_self == sha_ref)
            if not result["sha256_index_match"]:
                result["reason"] = "sha256_index_mismatch"
                return result

    result["valid"] = True
    return result


def validate_directory(
    target_dir: str | Path,
    ref_dir:    str | Path,
    factor_cols: Optional[list[str]] = None,
    check_sha256_index: bool = True,
    log_every: int = 20,
) -> dict:
    """
    批量校验 target_dir 全 parquet

    参数:
        target_dir: 待校验方案目录
        ref_dir: scheme_b 基准目录
        factor_cols: 数值列名列表（None = 自动推断）
        check_sha256_index: 是否对 index 做 sha256 比对
        log_every: 每 N 个打印一次

    返回:
        dict {
            "schema": ...,           # learn_expected_schema 输出
            "results": [validate_parquet 输出 * N],
            "n_valid":  int,
            "n_invalid": int,
            "n_missing_ref": int,    # target 有但 ref 缺
            "n_extra":   int,        # target 缺但 ref 有
        }
    """
    target_dir = Path(target_dir)
    ref_dir    = Path(ref_dir)

    schema = learn_expected_schema(ref_dir)
    ref_columns   = schema["ref_columns"]
    expected_cols = schema["expected_cols"]
    if not ref_columns:
        return {
            "schema": schema, "results": [],
            "n_valid": 0, "n_invalid": 0,
            "n_missing_ref": 0, "n_extra": 0,
        }

    target_files = sorted(target_dir.glob("*.parquet"))
    ref_files    = {f.stem for f in ref_dir.glob("*.parquet")}
    target_stems = {f.stem for f in target_files}

    n_missing_ref = len(ref_files - target_stems)
    n_extra       = len(target_stems - ref_files)

    # 预加载 ref 索引（按 stem）
    ref_index_map: dict[str, pd.DataFrame] = {}
    if check_sha256_index:
        for f in ref_dir.glob("*.parquet"):
            try:
                ref_index_map[f.stem] = pd.read_parquet(f)
            except Exception:
                pass

    results = []
    n_valid = 0
    n_invalid = 0

    for i, f in enumerate(target_files):
        ref_df = ref_index_map.get(f.stem)
        r = validate_parquet(
            path=f,
            ref_columns=ref_columns,
            expected_cols=expected_cols,
            ref_df=ref_df,
            factor_cols=factor_cols,
            check_sha256_index=check_sha256_index,
        )
        results.append({"file": f.name, **r})
        if r["valid"]:
            n_valid += 1
        else:
            n_invalid += 1
        if (i + 1) % log_every == 0:
            logger.info(
                f"  [validate] {i+1}/{len(target_files)} "
                f"valid={n_valid} invalid={n_invalid}"
            )

    return {
        "schema":         schema,
        "results":        results,
        "n_valid":        n_valid,
        "n_invalid":      n_invalid,
        "n_missing_ref":  n_missing_ref,
        "n_extra":        n_extra,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 3:
        print("Usage: python format_validator.py <target_dir> <ref_dir>")
        sys.exit(2)
    out = validate_directory(sys.argv[1], sys.argv[2])
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"valid:   {out['n_valid']}")
    print(f"invalid: {out['n_invalid']}")
    print(f"missing: {out['n_missing_ref']}")
    print(f"extra:   {out['n_extra']}")
    if out["n_invalid"] > 0:
        print("\nFirst 10 invalid files:")
        for r in out["results"]:
            if not r["valid"]:
                print(f"  {r['file']}: {r['reason']}")
                if len([x for x in out['results'] if not x['valid']]) >= 10:
                    break
    sys.exit(0 if out["n_invalid"] == 0 else 1)
