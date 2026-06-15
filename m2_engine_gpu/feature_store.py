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
M2 GPU v2 · 特征工程模块（直接复用 m2_engine 版本）

特征工程是纯 CPU 操作，无 GPU 依赖，直接 re-export。
"""
from m2_engine.feature_store import FeatureStore

__all__ = ["FeatureStore"]
