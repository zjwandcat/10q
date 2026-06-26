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

"""M3 TET 风控外挂模块

基于 Dr. Oliver Reiss《Trend-Emotion-Timing》(NAAIM Founders Award 2025)
的 TET 理论，对 M2 选股结果进行"持仓体检"，识别"趋势下行且情绪超买"
的标的并建议卖出，输出供 M4 双轨回测使用。
"""
from .tet_engine import TETEngine, M3Config, StockState

__version__ = "1.0.0"
__all__ = ["TETEngine", "M3Config", "StockState"]
