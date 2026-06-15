"""M4 回测绩效报告模块"""
from .metrics import PerformanceMetrics
from .report_generator import ReportGenerator, generate_report
__all__ = ["PerformanceMetrics", "ReportGenerator", "generate_report"]
