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
M2 数据预加载与优化工具
解决 DataLoader.load() 的 OOM 问题

使用方法：
    from m2_preprocessor import load_and_optimize_factor_df
    factor_df = load_and_optimize_factor_df()

    portfolios, stats = run_m2(
        preloaded_factor_df=factor_df,
        ...
    )
"""
import sys
import gc
import os
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("m2.preprocessor")


def load_and_optimize_factor_df(
    verbose: bool = True,
) -> pd.DataFrame:
    """
    安全加载并优化 factor_df，避免 OOM

    优化策略：
    1. 分步加载，及时释放内存
    2. 删除冗余列（stock_name, industry, list_date等）
    3. 确保数值列为float32
    4. 调用gc.collect()回收内存

    返回：
        优化后的factor_df
    """
    from m1_engine.data_loader import DataLoader
    from m1_engine.label_maker import LabelMaker

    if verbose:
        print("=" * 60)
        print("M2 数据预处理器")
        print("=" * 60)

    # Step 1: 加载数据
    if verbose:
        print("\n[Step 1/4] 加载原始数据...")

    loader = DataLoader()
    factor_df = loader.load()

    if verbose:
        print(f"  ✓ 原始数据: {len(factor_df):,}行 × {len(factor_df.columns)}列")
        mem_mb = factor_df.memory_usage(deep=True).sum() / 1024**2
        print(f"  ✓ 内存占用: {mem_mb:.1f} MB")

    # Step 2: 制作标签
    if verbose:
        print("\n[Step 2/4] 制作标签...")

    factor_df = LabelMaker().make_labels(factor_df)

    if verbose:
        print("  ✓ 标签制作完成")
        print(f"  ✓ 当前列数: {len(factor_df.columns)}")

    # Step 3: 删除冗余列
    if verbose:
        print("\n[Step 3/4] 删除冗余列以节省内存...")

    cols_to_drop = []
    for col in ["stock_name", "industry", "list_date"]:
        if col in factor_df.columns:
            cols_to_drop.append(col)

    if cols_to_drop:
        factor_df = factor_df.drop(columns=cols_to_drop)
        if verbose:
            print(f"  ✓ 已删除: {cols_to_drop}")
    else:
        if verbose:
            print("  - 无需删除的列")

    # Step 4: 优化数据类型
    if verbose:
        print("\n[Step 4/4] 优化数据类型...")

    meta_cols = {"trade_date", "stock_code"}
    # ★ 向量化：用select_dtypes一次性筛选，避免Python循环
    float64_cols = factor_df.select_dtypes(include="float64").columns
    float64_cols = [c for c in float64_cols if c not in meta_cols]
    if float64_cols:
        factor_df[float64_cols] = factor_df[float64_cols].astype(np.float32)
        if verbose:
            print(f"  ✓ 已优化{len(float64_cols)}列为float32")

    # 强制垃圾回收
    gc.collect()

    if verbose:
        final_mem_mb = factor_df.memory_usage(deep=True).sum() / 1024**2
        print("\n✅ 数据预处理完成！")
        print(f"   最终数据: {len(factor_df):,}行 × {len(factor_df.columns)}列")
        print(f"   最终内存: {final_mem_mb:.1f} MB")
        print(f"   日期范围: {factor_df['trade_date'].min()} ~ {factor_df['trade_date'].max()}")
        print("=" * 60)

    return factor_df


if __name__ == "__main__":
    # 测试数据预加载
    import psutil

    def mem_gb():
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3

    print(f"[开始] 内存={mem_gb():.2f}GB\n")

    try:
        factor_df = load_and_optimize_factor_df(verbose=True)
        print(f"\n[成功] 最终内存={mem_gb():.2f}GB")
        print(f"数据形状: {factor_df.shape}")
    except Exception as e:
        print(f"\n[失败] 错误: {e}")
        import traceback
        traceback.print_exc()
