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
M2 高效数据预处理器
解决 DataLoader.load() 的 OOM 问题

核心策略：
1. 分块读取parquet文件（每次几个文件）
2. 即时优化每块数据（删除冗余列、转换类型）
3. 增量合并到最终DataFrame
4. 避免一次性pd.concat导致的内存峰值

使用方法：
    from m2_engine.smart_preprocessor import smart_load_factor_df
    factor_df = smart_load_factor_df()
"""
import sys
import gc
import os
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger("m2.smart_preloader")


def _load_config() -> dict:
    """加载配置文件"""
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_parquet_safe(path: Path) -> pd.DataFrame:
    """安全读取单个parquet文件"""
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        logger.warning(f"读取失败: {path.name} - {e}")
        return None


def _optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化单个DataFrame的内存占用

    优化内容：
    1. 删除冗余元信息列
    2. float64转float32
    3. int64转int32
    """
    if df is None or len(df) == 0:
        return df

    # 删除冗余列
    cols_to_drop = [col for col in ["stock_name", "industry", "list_date"]
                   if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 元信息列保留原类型
    meta_cols_set = {"trade_date", "stock_code"}
    # ★ 向量化：一次性筛选float64
    float64_cols = [c for c in df.select_dtypes(include="float64").columns
                    if c not in meta_cols_set]
    int64_cols   = [c for c in df.select_dtypes(include="int64").columns
                    if c not in meta_cols_set]

    if float64_cols:
        df[float64_cols] = df[float64_cols].astype(np.float32)
    if int64_cols:
        df[int64_cols] = df[int64_cols].astype(np.int32)

    return df


def smart_load_factor_df(
    verbose: bool = True,
    chunk_size: int = 12,  # 每次处理的月份数
) -> pd.DataFrame:
    """
    智能加载factor_df，避免OOM

    参数：
        verbose: 是否显示详细日志
        chunk_size: 每次处理的文件数量（默认12个月）

    返回：
        优化后的factor_df
    """
    from m1_engine.label_maker import LabelMaker

    if verbose:
        print("=" * 60)
        print("M2 智能数据预处理器")
        print("=" * 60)
        print(f"\n策略: 分块加载（每批{chunk_size}个文件）+ 即时优化")

    # 获取配置和数据目录
    cfg = _load_config()
    active_scheme = cfg["data"]["neutralization"]["active_scheme"]
    pool_dir = Path(cfg["data"]["pool_dirs"][active_scheme])

    if verbose:
        print(f"数据目录: {pool_dir}")

    # 扫描所有parquet文件
    files = sorted(pool_dir.glob("*.parquet"))
    total_files = len(files)

    if not files:
        raise FileNotFoundError(f"目录为空: {pool_dir}")

    if verbose:
        print(f"发现{total_files}个parquet文件")

    # 分块处理
    all_dfs = []
    processed = 0

    for chunk_start in range(0, total_files, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_files)
        chunk_files = files[chunk_start:chunk_end]

        if verbose:
            print(f"\n[批次 {processed//chunk_size + 1}] "
                  f"处理文件 {chunk_start+1}-{chunk_end}/{total_files}")

        # 并行读取当前块的文件
        chunk_dfs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_read_parquet_safe, f)
                      for f in chunk_files]
            for future in futures:
                result = future.result()
                if result is not None:
                    # 立即优化这个DataFrame
                    result = _optimize_dataframe(result)
                    chunk_dfs.append(result)

        if chunk_dfs:
            # 合并当前块
            chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
            all_dfs.append(chunk_combined)
            processed += len(chunk_dfs)

            if verbose:
                print(f"  ✓ 本批次加载{len(chunk_dfs)}个文件, "
                      f"{len(chunk_combined):,}行")

            # 释放当前块的内存
            del chunk_dfs, chunk_combined
            gc.collect()

    if verbose:
        print(f"\n全部文件加载完成，共{processed}个文件")

    # 最终合并所有块
    if verbose:
        print("\n正在合并所有数据块...")

    if len(all_dfs) == 0:
        raise RuntimeError("没有成功加载任何数据")

    factor_df = pd.concat(all_dfs, ignore_index=True)

    # 释放中间变量
    del all_dfs
    gc.collect()

    # 确保日期排序
    if not pd.api.types.is_datetime64_any_dtype(factor_df["trade_date"]):
        factor_df["trade_date"] = pd.to_datetime(factor_df["trade_date"])

    factor_df = factor_df.sort_values("trade_date").reset_index(drop=True)

    # 制作标签
    if verbose:
        print("\n正在制作标签...")

    factor_df = LabelMaker().make_labels(factor_df)

    # 最终优化
    factor_df = _optimize_dataframe(factor_df)

    # 强制垃圾回收
    gc.collect()

    if verbose:
        mem_mb = factor_df.memory_usage(deep=True).sum() / 1024**2
        print("\n✅ 数据预处理完成！")
        print(f"   总行数:     {len(factor_df):,}")
        print(f"   总列数:     {len(factor_df.columns)}")
        print(f"   内存占用:   {mem_mb:.1f} MB")
        print(f"   日期范围:   {factor_df['trade_date'].min()} ~ "
              f"{factor_df['trade_date'].max()}")
        print(f"   股票数量:   {factor_df['stock_code'].nunique()}")
        print("=" * 60)

    return factor_df


if __name__ == "__main__":
    # 测试智能数据加载
    import psutil

    def mem_gb():
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3

    print(f"[开始] 内存={mem_gb():.2f}GB\n")

    try:
        factor_df = smart_load_factor_df(verbose=True)
        print(f"\n[成功] 最终内存={mem_gb():.2f}GB")
        print(f"数据形状: {factor_df.shape}")
    except Exception as e:
        print(f"\n[失败] 错误: {e}")
        import traceback
        traceback.print_exc()
