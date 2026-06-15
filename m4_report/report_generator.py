"""HTML回测报告生成"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
from .metrics import PerformanceMetrics

logger = logging.getLogger("m4.report")


class ReportGenerator:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.pm = PerformanceMetrics()

    def generate(
        self,
        all_portfolios: pd.DataFrame,
        output_path: str = "output/backtest_report.html",
        stats: Dict = None,
        title: str = "TTHH量化回测报告",
    ) -> Dict:
        p = Path(output_path)
        if not p.is_absolute():
            output_path = str(self.output_dir / p)

        monthly = self.pm.compute_monthly_returns(all_portfolios)
        metrics = self.pm.calculate(monthly)
        checks  = self.pm.check_thresholds(metrics)
        cum_r   = (1 + monthly["portfolio_return"]).cumprod()
        cum_bm  = (1 + monthly["benchmark_return"]).cumprod()

        months  = monthly["pred_month"].tolist()
        port_v  = cum_r.round(4).tolist()
        bm_v    = cum_bm.round(4).tolist()

        pass_color = "#27ae60"
        fail_color = "#e74c3c"

        def row(label, val, fmt="{:.4f}", ok=None):
            color = ""
            if ok is True:  color = f'style="color:{pass_color}"'
            if ok is False: color = f'style="color:{fail_color}"'
            return f"<tr><td>{label}</td><td {color}>{fmt.format(val)}</td></tr>"

        check_rows = "".join([
            f"<tr><td>{k}</td>"
            f"<td style='color:{'#27ae60' if v else '#e74c3c'}'>"
            f"{'✅' if v else '❌'}</td></tr>"
            for k, v in checks.items() if k != "all_pass"
        ])

        all_ok = checks["all_pass"]
        summary_color = pass_color if all_ok else fail_color
        summary_text  = "✅ 全部通过" if all_ok else "❌ 存在未达标项"

        if stats is not None:
            ic_list = stats.get("avg_val_ic", [])
            avg_ic = float(np.mean(ic_list)) if ic_list else 0.0
            icir = stats.get("avg_val_icir", "-")
            low_conf = stats.get("low_confidence_months", "-")
            success = stats.get("success", 1)
            if isinstance(low_conf, (int, float)) and isinstance(success, (int, float)) and success > 0:
                low_conf_pct = f"{low_conf / success:.2%}"
            else:
                low_conf_pct = "-"
            _summary_table = (
                "<table><tr><th>项目</th><th>数值</th></tr>"
                f"<tr><td>总窗口数</td><td>{stats.get('total_windows', '-')}</td></tr>"
                f"<tr><td>成功窗口</td><td>{success}</td></tr>"
                f"<tr><td>低置信度月份</td><td>{low_conf}</td></tr>"
                f"<tr><td>平均IC</td><td>{avg_ic:.4f}</td></tr>"
                f"<tr><td>ICIR</td><td>{icir}</td></tr>"
                f"<tr><td>降权月份占比</td><td>{low_conf_pct}</td></tr>"
                "</table>"
            )
        else:
            _summary_table = "<p>（未提供窗口统计信息）</p>"

        html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>TTHH回测报告 {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
  body{{font-family:Arial,sans-serif;margin:20px;background:#f5f6fa}}
  h1{{color:#2c3e50}} h2{{color:#34495e;border-bottom:2px solid #3498db}}
  table{{border-collapse:collapse;width:100%;margin:10px 0}}
  th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
  th{{background:#3498db;color:white}}
  tr:nth-child(even){{background:#f2f2f2}}
  .summary{{font-size:1.3em;font-weight:bold;color:{summary_color}}}
  .chart{{width:100%;height:400px}}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head><body>
<h1>{title}</h1>
<p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p class='summary'>综合评级：{summary_text}</p>

<h2>参数摘要</h2>
{_summary_table}

<h2>净值曲线</h2>
<canvas id='chart' class='chart'></canvas>
<script>
new Chart(document.getElementById('chart'),{{
  type:'line',
  data:{{
    labels:{json.dumps(months)},
    datasets:[
      {{label:'策略',data:{json.dumps(port_v)},
       borderColor:'#3498db',fill:false,pointRadius:0}},
      {{label:'基准',data:{json.dumps(bm_v)},
       borderColor:'#e74c3c',fill:false,pointRadius:0}}
    ]
  }},
  options:{{responsive:true,scales:{{y:{{beginAtZero:false}}}}}}
}});
</script>

<h2>核心绩效指标</h2>
<table><tr><th>指标</th><th>数值</th></tr>
{row('CAGR', metrics['cagr'], '{:.2%}')}
{row('超额CAGR', metrics['annual_excess'], '{:.2%}')}
{row('最大回撤', metrics['max_drawdown'], '{:.2%}')}
{row('夏普比率', metrics['sharpe_ratio'])}
{row('Sortino', metrics['sortino_ratio'])}
{row('Calmar', metrics['calmar_ratio'])}
{row('Jensen α (年化)', metrics['jensen_alpha'], '{:.2%}')}
{row('Appraisal Ratio', metrics['appraisal_ratio'])}
{row('IR', metrics['ir'])}
{row('月度胜率', metrics['monthly_win_rate'], '{:.2%}')}
{row('VaR(95%)', metrics['var_95'], '{:.2%}')}
{row('CVaR(95%)', metrics['cvar_95'], '{:.2%}')}
{row('偏度', metrics['skewness'])}
{row('峰度', metrics['kurtosis'])}
{row('痛苦指数', metrics['pain_index'])}
{row('溃疡指数', metrics['ulcer_index'])}
{row('Omega', metrics['omega_ratio'])}
{row('尾部比率', metrics['tail_ratio'])}
{row('上行捕获', metrics['up_capture_ratio'])}
{row('下行捕获', metrics['down_capture_ratio'])}
{row('综合捕获', metrics['capture_ratio'])}
{row('Sterling', metrics['sterling_ratio'])}
{row('Burke', metrics['burke_ratio'])}
{row('Martin', metrics['martin_ratio'])}
{row('滚动6月超额胜率', metrics['rolling6m_win_rate'], '{:.2%}')}
</table>

<h2>门槛检验（9项）</h2>
<table><tr><th>检验项</th><th>结果</th></tr>
{check_rows}
</table>

</body></html>"""

        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"报告已生成: {output_path}")
        return {
            "report_path": str(output_path),
            "metrics": metrics,
            "checks": checks,
        }


def generate_report(
    all_portfolios: pd.DataFrame,
    stats: Dict = None,
    output_path: str = "output/backtest_report.html",
    suffix: str = "",
    title: str = "TTHH量化回测报告",
) -> Dict:
    """
    便捷函数：生成回测报告。
    """
    if suffix:
        p = Path(output_path)
        output_path = str(p.with_stem(p.stem + suffix))

    rg = ReportGenerator()
    return rg.generate(
        all_portfolios,
        output_path=output_path,
        stats=stats,
        title=title,
    )
