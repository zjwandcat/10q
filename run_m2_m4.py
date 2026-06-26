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

"""M2+M4完整流程"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import gc
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

print("="*60)
print("M2+M4 完整流程（iOS 26 风格报告）")
print("="*60)

# 1. 运行M2
print("\n【阶段1】运行M2...")
start = time.time()
from m2_engine.run_m2 import run_m2

all_portfolios, stats = run_m2(
    fast_mode=False,
    compute_val_metrics=False,
    compute_shap=True,    # ★ 改造: M4 报告路径启用 SHAP 归因
    # train_months 不指定，使用 config.yaml 默认值
    gpu_mode=False,
    verbose=True,
)
m2_time = time.time() - start
print(f"\n✅ M2完成，耗时: {m2_time/60:.1f}分钟")
print(f"   生成持仓: {len(all_portfolios)}行")
print(f"   月份数: {all_portfolios['pred_month'].nunique()}")

# 保存
all_portfolios.to_parquet("output/all_portfolios.parquet", index=False)
print("   已保存到 output/all_portfolios.parquet")

gc.collect()

# 2. 运行M4
print("\n【阶段2】运行M4（iOS 26 风格 + 归因）...")
from m4_report.metrics import PerformanceMetrics
from m4_report.report_generator import ReportGenerator

# ★ 改造: 重新加载 M0 因子表（带 industry / Barra 因子），
#   归因模块需要。复用 M2 的 scheme（从 config.yaml 读取）。
import yaml
with open("config/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
scheme = cfg["data"]["neutralization"].get("active_scheme", "scheme_b")
pool_dir = cfg["data"]["pool_dirs"][scheme]
print(f"   加载 M0 因子: scheme={scheme} dir={pool_dir}")
import pandas as pd
from pathlib import Path

factor_df = None
try:
    files = sorted(Path(pool_dir).glob("*.parquet"))
    if files:
        dfs = [pd.read_parquet(f) for f in files[:50]]  # 限制 50 个文件省内存
        factor_df = pd.concat(dfs, ignore_index=True)
        # 只保留归因需要的列
        keep_cols = [
            "trade_date", "stock_code", "stock_name",
            "industry", "Target_Return_1M",
            # 五因子归因需要的列（兼容 M0 列名）
            "size_log_mcap", "ln_market_cap",
            "sup_val_pe_ttm", "pe_ttm",
            "quality_roe", "roe",
            "growth_netprofit_yoy", "netprofit_yoy",
        ] + [f"barra_{n}" for n in [
            "beta", "momentum", "size", "earnings_yield",
            "value", "volatility", "liquidity", "leverage",
            "growth", "quality"]]
        keep_cols = [c for c in keep_cols if c in factor_df.columns]
        factor_df = factor_df[keep_cols].copy()
        # 统一类型
        if "trade_date" in factor_df.columns:
            factor_df["trade_date"] = pd.to_datetime(
                factor_df["trade_date"])
        print(f"   已加载 {len(factor_df)}行 因子用于归因")
        gc.collect()
except Exception as e:
    print(f"   ⚠️ 加载 M0 因子失败，归因模块降级: {e}")
    factor_df = None

pm = PerformanceMetrics()
monthly = pm.compute_monthly_returns(all_portfolios)
metrics = pm.calculate(monthly)

rg = ReportGenerator()
result = rg.generate(
    all_portfolios,
    stats=stats,
    factor_df=factor_df,
)

print(f"\n✅ M4完成")
print(f"   月份数:    {len(monthly)}")
print(f"   CAGR:      {metrics['cagr']:.2%}")
print(f"   夏普:      {metrics['sharpe_ratio']:.2f}")
print(f"   IR:        {metrics['ir']:.2f}")
print(f"   报告:      {result['report_path']}")

# 归因数据落盘（供 M5 / 后续分析用）
if "attribution" in result:
    import json
    attr_path = "output/attribution.json"
    with open(attr_path, "w", encoding="utf-8") as f:
        json.dump(result["attribution"], f,
                  ensure_ascii=False, indent=2)
    print(f"   归因数据:  {attr_path}")

print("\n" + "="*60)
print("全部完成！")
print("="*60)
