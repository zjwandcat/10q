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

print("="*60)
print("M2+M4 完整流程")
print("="*60)

# 1. 运行M2
print("\n【阶段1】运行M2...")
start = time.time()
from m2_engine.run_m2 import run_m2

all_portfolios, stats = run_m2(
    fast_mode=False,
    compute_val_metrics=False,
    compute_shap=False,
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
print("\n【阶段2】运行M4...")
from m4_report.metrics import PerformanceMetrics
from m4_report.report_generator import ReportGenerator

pm = PerformanceMetrics()
monthly = pm.compute_monthly_returns(all_portfolios)
metrics = pm.calculate(monthly)
checks = pm.check_thresholds(metrics)

rg = ReportGenerator()
result = rg.generate(all_portfolios)

passed = sum(1 for k, v in checks.items() if k != "all_pass" and v)
print(f"\n✅ M4完成")
print(f"   月份数: {len(monthly)}")
print(f"   CAGR: {metrics['cagr']:.2%}")
print(f"   夏普: {metrics['sharpe_ratio']:.2f}")
print(f"   IR: {metrics['ir']:.2f}")
print(f"   门槛检验: {passed}/9通过")
print(f"   报告: output/backtest_report.html")

print("\n" + "="*60)
print("全部完成！")
print("="*60)
