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
M5内存监控工具
"""
import os
import psutil


def get_memory_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def check_memory(limit_gb: float = 9.5, logger=None) -> bool:
    mem = get_memory_gb()
    if mem > limit_gb:
        if logger:
            logger.warning(f"内存超限:{mem:.2f}GB>{limit_gb}GB")
        return False
    return True
