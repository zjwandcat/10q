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
M2 超轻量级数据加载器
完全绕过 DataLoader，解决 OOM 问题

设计原则：
1. 逐文件读取，增量追加（避免pd.concat内存峰值）
2. 即时优化（删除冗余列、转换数据类型）
3. 最小化内存占用
4. 支持日期范围过滤（减少数据量）
"""
import sys
import gc
import os
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("m2.lightweight_loader")


def _get_data_dir() -> Path:
    """获取数据目录"""
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    active = cfg["data"]["neutralization"]["active_scheme"]
    return Path(cfg["data"]["pool_dirs"][active])


def lightweight_load_factor_df(
    verbose: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    required_months: int = None,
) -> pd.DataFrame:
    """
    超轻量级加载factor_df

    参数：
        verbose: 是否显示日志
        start_date: 起始日期（YYYYMM格式），None表示不限
        end_date: 结束日期（YYYYMM格式），None表示不限
        required_months: 只加载最后N个月的数据

    返回：
        优化后的factor_df（含label_rank）
    """
    from m1_engine.label_maker import LabelMaker

    if verbose:
        print("=" * 60)
        print("M2 超轻量级数据加载器")
        print("=" * 60)

    data_dir = _get_data_dir()

    if verbose:
        print(f"数据目录: {data_dir}")

    # 扫描文件
    files = sorted(data_dir.glob("*.parquet"))
    total_files = len(files)

    if not files:
        raise FileNotFoundError(f"目录为空: {data_dir}")

    # 日期过滤
    if start_date:
        files = [f for f in files if f.stem >= start_date]
    if end_date:
        files = [f for f in files if f.stem <= end_date]
    if required_months and required_months > 0:
        files = files[-required_months:]

    filtered_files = len(files)

    if verbose:
        print(f"总文件数: {total_files}")
        if start_date or end_date or required_months:
            print(f"过滤后:   {filtered_files}个文件")

    # 增量加载（避免pd.concat内存峰值）
    if verbose:
        print("\n开始增量加载...")

    factor_df = None
    loaded_count = 0

    for i, file_path in enumerate(files):
        try:
            # 读取单个文件
            df = pd.read_parquet(file_path)

            # 即时优化
            # 1. 删除冗余列
            for col in ["stock_name", "industry", "list_date"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # 2. 优化数据类型（向量化）
            meta_cols = {"trade_date", "stock_code"}
            # ★ 向量化：select_dtypes一次性处理float64
            float64_cols = [c for c in df.select_dtypes(include="float64").columns if c not in meta_cols]
            if float64_cols:
                df[float64_cols] = df[float64_cols].astype(np.float32)
            # int64列同样向量化处理
            int64_cols = [c for c in df.select_dtypes(include="int64").columns if c not in meta_cols]
            if int64_cols:
                df[int64_cols] = df[int64_cols].astype(np.int32)

            # 3. 增量追加（使用append而非concat）
            if factor_df is None:
                factor_df = df
            else:
                # 使用pd.concat但只处理两个DataFrame
                factor_df = pd.concat([factor_df, df], ignore_index=True)

            loaded_count += 1

            # 每20个文件输出进度并释放内存
            if (i + 1) % 20 == 0 or i == len(files) - 1:
                if verbose:
                    print(f"  [{i+1}/{filtered_files}] "
                          f"已加载{loaded_count}个文件, "
                          f"当前{len(factor_df):,}行")
                gc.collect()

        except Exception as e:
            logger.warning(f"读取失败: {file_path.name} - {e}")
            continue

    if factor_df is None or len(factor_df) == 0:
        raise RuntimeError("没有成功加载任何数据")

    if verbose:
        print(f"\n✓ 数据加载完成: {len(factor_df):,}行 × {len(factor_df.columns)}列")

    # 制作标签
    if verbose:
        print("\n正在制作标签...")

    factor_df = LabelMaker().make_labels(factor_df)

    # 最终整理
    if not pd.api.types.is_datetime64_any_dtype(factor_df["trade_date"]):
        factor_df["trade_date"] = pd.to_datetime(factor_df["trade_date"])

    # 按日期排序（使用更节省内存的方式）
    factor_df = factor_df.sort_values(
        "trade_date",
        kind="mergesort",  # mergesort更省内存
    ).reset_index(drop=True)

    gc.collect()

    if verbose:
        mem_mb = factor_df.memory_usage(deep=True).sum() / 1024**2
        print("\n✅ 轻量级加载完成！")
        print(f"   总行数:     {len(factor_df):,}")
        print(f"   总列数:     {len(factor_df.columns)}")
        print(f"   内存占用:   {mem_mb:.1f} MB")
        print(f"   日期范围:   {factor_df['trade_date'].min()} ~ "
              f"{factor_df['trade_date'].max()}")
        print("=" * 60)

    return factor_df


if __name__ == "__main__":
    import psutil

    def mem_gb():
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3

    print(f"[开始] 内存={mem_gb():.2f}GB\n")

    try:
        # 测试1：只加载最后49个月（支持1个窗口）
        print("测试1: 加载最后49个月...")
        df1 = lightweight_load_factor_df(
            verbose=True,
            required_months=49,
        )
        print(f"\n成功！内存={mem_gb():.2f}GB\n")

        del df1
        gc.collect()

        # 测试2：加载最后80个月（支持约31个窗口）
        print("测试2: 加载最后80个月...")
        df2 = lightweight_load_factor_df(
            verbose=True,
            required_months=80,
        )
        print(f"\n成功！内存={mem_gb():.2f}GB")

    except Exception as e:
        print(f"\n失败: {e}")
        import traceback
        traceback.print_exc()
