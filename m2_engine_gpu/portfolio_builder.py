"""
M2 GPU v2 · 持仓构建模块（直接复用 m2_engine 版本）

持仓构建是纯 CPU 操作，无 GPU 依赖，直接 re-export。
"""
from m2_engine.portfolio_builder import (
    PortfolioBuilder,
    calculate_turnover_cost,
)

__all__ = ["PortfolioBuilder", "calculate_turnover_cost"]
