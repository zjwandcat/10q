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
全局并发配置 - 低内存高CPU模式
所有模块统一从此文件读取，禁止硬编码线程数
"""
import os

# ── 硬件自动检测 ──────────────────────────────────
_LOGICAL_CORES  = os.cpu_count() or 12
_PHYSICAL_CORES = max(_LOGICAL_CORES // 2, 1)

# ── 外层窗口并行（run_m2.py，threading后端）────────
# 原配置：物理核//3，最少1，最多3
# 内存优化：强制=1（串行窗口），大幅降低内存
GLOBAL_N_JOBS_OUTER = 1

# ── 内层模型线程（LGBM/XGB单模型）────────────────
# 原配置：物理核//外层//2
# 内存优化：=4（固定，不随外层变化）
GLOBAL_NTHREAD_INNER = 4

# ── M5双模型并行专用线程数 ──────────────────────
# 内存优化：=4（固定）
M5_NTHREAD_PER_MODEL = 4

# ── 数据加载IO并发 ───────────────────────────────
DATA_LOADER_MAX_WORKERS = min(_LOGICAL_CORES, 4)

# ── M0数据拉取并发 ───────────────────────────────
M0_FETCH_MAX_WORKERS = min(_PHYSICAL_CORES, 2)

# ── 内存红线（降低避免OOM）──────────────────────
# ★ v4.2: 物理 RAM 16GB 时建议 ≤ 6GB 留出系统+其他程序
#   旧值 8GB 太激进，半夜会撑爆分页文件（你当前分页文件只有 2GB）
MEMORY_LIMIT_GB  = 6.0
MEMORY_BUFFER_GB = 1.5

# ── 启动时打印确认 ───────────────────────────────
print(f"[并发配置] 物理核={_PHYSICAL_CORES} "
      f"逻辑核={_LOGICAL_CORES}")
print(f"[并发配置] 外层并行={GLOBAL_N_JOBS_OUTER} "
      f"每模型线程={M5_NTHREAD_PER_MODEL}")
print(f"[并发配置] 理论总线程="
      f"{GLOBAL_N_JOBS_OUTER}×2×{M5_NTHREAD_PER_MODEL}="
      f"{GLOBAL_N_JOBS_OUTER*2*M5_NTHREAD_PER_MODEL}"
      f"/{_LOGICAL_CORES}逻辑核")
print(f"[并发配置] 模式=低内存高CPU（单窗口串行）")
