"""
M0全量数据生成脚本
支持断点续跑：已存在的文件自动跳过
运行方式：python run_m0_full.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')

from m0_database.pipeline import run_pipeline, get_month_list

print("=" * 60)
print("M0 全量数据生成")
print("=" * 60)

# 检查现有文件数
import yaml
with open("config/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

pool_dirs = cfg["data"]["pool_dirs"]
print("\n当前各方案已有文件数：")
for scheme, d in pool_dirs.items():
    count = len(list(Path(d).glob("*.parquet")))
    print(f"  {scheme}: {count}/228 个")

# 全部月份
all_months = get_month_list("200701", "202512")
print(f"\n目标：{len(all_months)}个月 × 4套方案 = "
      f"{len(all_months)*4}个parquet文件")

# ── 分阶段运行（每阶段完成后打印进度）────────────
# 建议先跑推荐方案（scheme_b），再跑其他方案
# 这样M1/M2可以更早开始测试

print("\n" + "=" * 60)
print("阶段1：生成推荐方案 scheme_b（Rank-Z+双重OLS）")
print("=" * 60)
t0 = time.time()

run_pipeline(
    schemes=["scheme_b"],
    months=all_months,
    force_rebuild=False,  # 断点续跑
)

elapsed = (time.time() - t0) / 60
count_b = len(list(Path(pool_dirs["scheme_b"]).glob("*.parquet")))
print(f"\nscheme_b完成：{count_b}/228个文件，耗时{elapsed:.1f}分钟")

print("\n" + "=" * 60)
print("阶段2：生成方案A（双重OLS正交化）")
print("=" * 60)
t0 = time.time()

run_pipeline(
    schemes=["scheme_a"],
    months=all_months,
    force_rebuild=False,
)

elapsed = (time.time() - t0) / 60
count_a = len(list(Path(pool_dirs["scheme_a"]).glob("*.parquet")))
print(f"\nscheme_a完成：{count_a}/228个文件，耗时{elapsed:.1f}分钟")

print("\n" + "=" * 60)
print("阶段3：生成方案D（对照组，仅行业OLS）")
print("=" * 60)
t0 = time.time()

run_pipeline(
    schemes=["scheme_d"],
    months=all_months,
    force_rebuild=False,
)

elapsed = (time.time() - t0) / 60
count_d = len(list(Path(pool_dirs["scheme_d"]).glob("*.parquet")))
print(f"\nscheme_d完成：{count_d}/228个文件，耗时{elapsed:.1f}分钟")

print("\n" + "=" * 60)
print("阶段4：生成方案E（分层中性化）")
print("=" * 60)
t0 = time.time()

run_pipeline(
    schemes=["scheme_e"],
    months=all_months,
    force_rebuild=False,
)

elapsed = (time.time() - t0) / 60
count_e = len(list(Path(pool_dirs["scheme_e"]).glob("*.parquet")))
print(f"\nscheme_e完成：{count_e}/228个文件，耗时{elapsed:.1f}分钟")

# ── 最终汇总 ──────────────────────────────────
print("\n" + "=" * 60)
print("全量生成完成汇总")
print("=" * 60)
for scheme, d in pool_dirs.items():
    count = len(list(Path(d).glob("*.parquet")))
    status = "[OK]" if count == 228 else f"[WARN] 仅{count}/228"
    print(f"  {status} {scheme}: {count}个文件")

# 抽样验证最后一个文件
print("\n抽样验证（scheme_b最新月）：")
import pandas as pd
files = sorted(Path(pool_dirs["scheme_b"]).glob("*.parquet"))
if files:
    df = pd.read_parquet(files[-1])
    print(f"  文件: {files[-1].name}")
    print(f"  行数: {len(df)}")
    print(f"  列数: {len(df.columns)}")
