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

import optuna
import pandas as pd
import numpy as np
import json
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 找到最新的P1 study ──
db_files = list(Path("output").rglob("*p1*.db"))
if not db_files:
    db_files = list(Path("output").rglob("*.db"))
print(f"找到db文件: {[str(f) for f in db_files]}")

# 加载最新study（取最后修改时间最新的）
db_path = max(db_files, key=lambda f: f.stat().st_mtime)
storage = f"sqlite:///{db_path}"
summaries = optuna.get_all_study_summaries(storage=storage)
study = optuna.load_study(
    study_name=summaries[-1].study_name,
    storage=storage
)
print(f"Study: {study.study_name}, 路径: {db_path}")

# ── 过滤有效Trial ──
valid = [t for t in study.trials
         if t.state == optuna.trial.TrialState.COMPLETE
         and t.value is not None
         and t.value > -900.0]
failed = [t for t in study.trials
          if t.value is None or t.value <= -900.0]

print(f"总Trial: {len(study.trials)} | 有效: {len(valid)} | 失败/异常: {len(failed)}")

# ── 关键指标列表 ──
KEY_METRICS = [
    "val_rolling6m_ir", "val_ic", "val_icir",
    "ic_gap", "penalized_rate",
    "pct_positive_excess", "ir_worst_quartile",
    "val_global_ir",
]

# ── TOP10 Trial ──
top10 = sorted(valid, key=lambda t: t.value)[:10]

rows = []
for t in top10:
    row = {
        "trial_id": t.number,
        "score": round(-t.value, 4),
    }
    for m in KEY_METRICS:
        v = t.user_attrs.get(m)
        row[m] = round(float(v), 4) if v is not None else None
    rows.append(row)

df_top = pd.DataFrame(rows)
print("\n=== TOP10 Trial ===")
print(df_top.to_string(index=False))

# ── 全量指标分布 ──
print("\n=== 全量有效Trial指标分布 ===")
dist_rows = []
for m in KEY_METRICS:
    vals = [t.user_attrs.get(m) for t in valid
            if t.user_attrs.get(m) is not None]
    if vals:
        arr = np.array(vals)
        dist_rows.append({
            "指标": m,
            "均值": round(arr.mean(), 4),
            "标准差": round(arr.std(), 4),
            "最小": round(arr.min(), 4),
            "P25": round(np.percentile(arr, 25), 4),
            "中位数": round(np.median(arr), 4),
            "P75": round(np.percentile(arr, 75), 4),
            "最大": round(arr.max(), 4),
            ">0占比": f"{(arr>0).mean():.1%}",
        })
print(pd.DataFrame(dist_rows).to_string(index=False))

# ── 参数收敛分析（上半区 vs 下半区）──
print("\n=== 参数收敛分析（好的Trial vs 差的Trial）===")
median_score = np.median([-t.value for t in valid])
top_half = [t for t in valid if -t.value >= median_score]
bot_half = [t for t in valid if -t.value < median_score]

KEY_PARAMS = [
    "lgbm_learning_rate", "lgbm_n_estimators",
    "lgbm_max_depth", "lgbm_reg_lambda",
    "xgb_learning_rate", "xgb_n_estimators",
    "xgb_max_depth", "xgb_reg_lambda",
    "lgbm_weight", "min_ic_abs",
    "min_keep_factors", "max_corr",
]

param_rows = []
best_t = top10[0]
for p in KEY_PARAMS:
    top_vals = [t.params[p] for t in top_half if p in t.params]
    bot_vals = [t.params[p] for t in bot_half if p in t.params]
    if not top_vals:
        continue
    param_rows.append({
        "参数": p,
        "TOP半区均值": round(np.mean(top_vals), 4),
        "TOP半区std": round(np.std(top_vals), 4),
        "BOT半区均值": round(np.mean(bot_vals), 4),
        "最优Trial值": round(best_t.params.get(p, float("nan")), 4),
    })
print(pd.DataFrame(param_rows).to_string(index=False))

# ── rolling6m_ir趋势（按trial顺序）──
print("\n=== rolling6m_ir随Trial进展趋势 ===")
ir_vals = [(t.number,
            round(-t.value, 4),
            round(t.user_attrs.get("val_rolling6m_ir", 0), 4))
           for t in sorted(valid, key=lambda x: x.number)]
print(f"{'Trial':>6} {'score':>8} {'rolling6m_ir':>14}")
for tid, sc, ir in ir_vals:
    print(f"{tid:>6} {sc:>8} {ir:>14}")

# ── 热启动先验JSON ──
print("\n=== 热启动先验（TOP3，供enqueue_trial使用）===")
warm = []
for t in top10[:3]:
    warm.append({
        "trial_id": t.number,
        "score": round(-t.value, 4),
        "params": {k: round(v, 6) if isinstance(v, float) else v
                   for k, v in t.params.items()}
    })
print(json.dumps(warm, indent=2, ensure_ascii=False))

# ── 保存报告 ──
report_path = Path("output/m5_p1_50trials_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"# M5 Phase1 Trial报告\n\n")
    f.write(f"- Study: {study.study_name}\n")
    f.write(f"- 有效Trial: {len(valid)} / 总计: {len(study.trials)}\n\n")
    f.write("## TOP10\n\n")
    f.write(df_top.to_markdown(index=False))
    f.write("\n\n## 指标分布\n\n")
    f.write(pd.DataFrame(dist_rows).to_markdown(index=False))
    f.write("\n\n## 参数收敛\n\n")
    f.write(pd.DataFrame(param_rows).to_markdown(index=False))
    f.write("\n\n## 热启动先验\n\n```json\n")
    f.write(json.dumps(warm, indent=2, ensure_ascii=False))
    f.write("\n```\n")

print(f"\n报告已保存: {report_path}")
