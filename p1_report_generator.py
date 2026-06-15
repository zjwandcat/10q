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

import optuna, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)


def generate_report(study_dir):
    """Generate p1_analysis_report.md for a given study directory."""
    study_name = study_dir.name
    db_path = study_dir / "p1" / f"{study_name}_p1_study.db"
    if not db_path.exists():
        print(f"  SKIP {study_name}: db not found at {db_path}")
        return

    study = optuna.load_study(
        study_name=optuna.get_all_study_summaries(f"sqlite:///{db_path}")[0].study_name,
        storage=f"sqlite:///{db_path}",
    )

    # ==================== Stage 1 Filter ====================
    stage1 = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value > -900.0
    ]
    if not stage1:
        print(f"  SKIP {study_name}: no Stage1 trials")
        return

    # ==================== Stage 2 Quality Gate ====================
    penalized_key = None
    for candidate in ["penalized_rate", "low_confidence_months_ratio"]:
        if any(candidate in t.user_attrs for t in stage1):
            penalized_key = candidate
            break

    stage2 = []
    for t in stage1:
        attrs = t.user_attrs
        ok = True
        ok = ok and attrs.get("val_rolling6m_ir", -999) > 0.05
        ok = ok and attrs.get("val_global_ir", -999) > 0.05
        ok = ok and attrs.get("pct_positive_excess", 0) > 0.5
        if penalized_key:
            ok = ok and attrs.get(penalized_key, 999) < 0.5
        if ok:
            stage2.append(t)

    n_stage1 = len(stage1)
    n_stage2 = len(stage2)

    # ==================== Edge Cases ====================
    if n_stage2 == 0:
        out_path = study_dir / "p1" / f"{study_name}_p1_analysis_report.md"
        content = f"# M5 Phase1 分析报告 - {study_name}\n\n"
        content += f"> 生成: {datetime.now():%Y-%m-%d %H:%M} | 路径: `{db_path}`\n\n"
        content += "## 运行概览\n\n"
        content += f"全部Trial未通过质量筛选，Stage1通过{n_stage1}条。Stage2合格数=0，分析终止。\n"
        content += f"\n---\n*tthh自动生成 | 供Claude(AA)分析*\n"
        out_path.write_text(content, encoding="utf-8")
        print(f"  DONE {study_name}: Stage2=0, report saved ({out_path.stat().st_size/1024:.1f}KB)")
        return

    low_count = n_stage2 < 10

    # ==================== Build DataFrame ====================
    all_keys = set()
    for t in stage2:
        all_keys.update(t.user_attrs.keys())

    rows = []
    for t in stage2:
        r = {"trial_id": t.number, "score": t.value}
        for k in all_keys:
            r[k] = t.user_attrs.get(k)
        for k, v in t.params.items():
            r[f"p_{k}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)

    KM = [
        k
        for k in [
            "val_ic",
            "val_icir",
            "val_rolling6m_ir",
            "val_rolling6m_excess_ann",
            "ic_gap",
            "penalized_rate",
            "pct_positive_excess",
            "ir_worst_quartile",
            "val_global_ir",
        ]
        if k in df.columns
    ]

    KP = [
        k
        for k in [
            "p_lgbm_learning_rate",
            "p_lgbm_n_estimators",
            "p_lgbm_max_depth",
            "p_lgbm_reg_lambda",
            "p_xgb_learning_rate",
            "p_xgb_n_estimators",
            "p_xgb_max_depth",
            "p_xgb_reg_lambda",
            "p_lgbm_weight",
            "p_min_ic_abs",
            "p_min_keep_factors",
            "p_max_corr",
            "p_drop_short_term_noise",
            "p_train_months",
        ]
        if k in df.columns
    ]

    best_score = df["score"].min()
    med_score = df["score"].median()
    best_trial = df.loc[df["score"].idxmin()]

    top10 = df.nsmallest(10, "score")
    topH = df.nsmallest(len(df) // 2, "score")
    botH = df.nlargest(len(df) // 2, "score")
    top25 = df.nsmallest(max(1, len(df) // 4), "score")

    L = []
    def A(s=""):
        L.append(s)

    # ==================== S1: 运行概览 ====================
    A(f"# M5 Phase1 分析报告 - {study_name}")
    A("<!-- AA_QUICK_READ: 滑块→S6 | P2范围→S7 | 热启动→S5 | 参数收敛→S4 -->")
    A(
        f"\n> 生成: {datetime.now():%Y-%m-%d %H:%M} | "
        f"路径: `{db_path}` | "
        f"Stage1: {n_stage1} | Stage2: {n_stage2}\n"
    )
    if low_count:
        A("> ⚠️ **警告**: Stage2合格Trial数 < 10，Section 4/6/7 将跳过。\n")

    A("## 1. 运行概览\n")
    A("| 项目 | 值 |")
    A("|------|-----|")
    pass_rate = n_stage2 / n_stage1 if n_stage1 > 0 else 0
    A(f"| Study总Trial数 | {len(study.trials)} |")
    A(f"| Stage1通过数（COMPLETE + value > -900） | {n_stage1} |")
    A(f"| Stage2合格数（质量筛选后） | {n_stage2} |")
    A(f"| 合格率（Stage2/Stage1） | {pass_rate:.2%} |")
    A(f"| 最优得分 | {best_score:.6f} |")
    A(f"| 中位得分 | {med_score:.6f} |")
    A(f"| 最优Trial编号 | #{int(best_trial['trial_id'])} |")

    # ==================== S2: 指标实际分布 ====================
    A("\n## 2. 指标实际分布（Stage2合格Trial）\n")
    A("| Metric | P5 | P25 | Median | P75 | P95 | Max | >0% |")
    A("|--------|-----|------|--------|-----|-----|-----|-----|")
    for m in KM:
        c = df[m].dropna()
        if c.empty:
            continue
        p = lambda x: np.percentile(c, x)
        A(
            f"| {m} | {p(5):.4f} | {p(25):.4f} | {c.median():.4f} | "
            f"{p(75):.4f} | {p(95):.4f} | {c.max():.4f} | {(c > 0).mean():.1%} |"
        )

    # ==================== S3: TOP10 Trial ====================
    A("\n## 3. TOP10 Trial\n")
    dc = [
        "trial_id",
        "score",
        "val_rolling6m_ir",
        "val_global_ir",
        "pct_positive_excess",
    ]
    if penalized_key:
        dc.append(penalized_key)
    dc.append("val_icir")

    existing_dc = [c for c in dc if c in top10.columns]
    rename_map = {
        "trial_id": "Trial#",
        "score": "Score",
        "val_rolling6m_ir": "val_rolling6m_ir",
        "val_global_ir": "val_global_ir",
        "pct_positive_excess": "pct_positive_excess",
        "val_icir": "val_icir",
    }
    if penalized_key:
        rename_map[penalized_key] = "penalized_rate"

    top10_display = top10[existing_dc].copy()
    top10_display.insert(0, "Rank", range(1, len(top10_display) + 1))
    top10_display = top10_display.rename(columns=rename_map)
    fmt_fields = [c for c in top10_display.columns if c not in ("Rank", "Trial#")]
    for c in fmt_fields:
        if c in top10_display.columns:
            top10_display[c] = top10_display[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    A(top10_display.to_markdown(index=False))

    # ==================== S4: 参数收敛分析 ====================
    A("\n## 4. 参数收敛分析\n")
    if low_count:
        A("> 数据不足，跳过。\n")
    else:
        A("| Param | TOP_mean | TOP_std | BOT_mean | BOT_std | std_ratio | Converged? |")
        A("|-------|----------|---------|----------|---------|-----------|-------------|")
        for p in KP:
            tv = topH[p].dropna()
            bv = botH[p].dropna()
            if tv.empty:
                continue

            col = df[p].dropna()
            is_str = pd.api.types.is_string_dtype(col.dtype)
            is_bool = col.nunique() <= 2 and set(col.unique()).issubset({True, False, 0, 1, 0.0, 1.0})
            is_cat = is_str or col.dtype == object or col.dtype == bool or is_bool

            if is_cat:
                mode_val = tv.mode()
                mode_v = mode_val.iloc[0] if not mode_val.empty else None
                if is_bool:
                    mode_str = f"mode={bool(mode_v)}"
                    freq = (tv == mode_v).mean() if mode_v is not None else 0
                    A(f"| {p[2:]} | {mode_str} ({freq:.0%}) | — | — | — | — | categorical |")
                else:
                    mode_str = str(mode_v)
                    freq = (tv == mode_v).mean() if mode_v is not None else 0
                    A(f"| {p[2:]} | mode={mode_str} ({freq:.0%}) | — | — | — | — | categorical |")
                continue

            top_mean = tv.mean()
            top_std = tv.std()
            bot_mean = bv.mean() if not bv.empty else np.nan
            bot_std = bv.std() if not bv.empty else np.nan

            if bv.empty or bot_std == 0:
                std_ratio = "N/A"
                converged = "N/A"
            else:
                sr = top_std / bot_std if bot_std > 0 else float("inf")
                std_ratio = f"{sr:.3f}"
                converged = "✓" if (sr < 0.7 and top_std / (abs(top_mean) + 1e-8) < 0.30) else "—"

            A(
                f"| {p[2:]} | {top_mean:.4f} | {top_std:.4f} | "
                f"{bot_mean:.4f} | {bot_std:.4f} | {std_ratio} | {converged} |"
            )

    # ==================== S5: 热启动先验JSON ====================
    A("\n## 5. 热启动先验JSON\n")
    A("```json")
    warm = []
    for _, row in top10.head(3).iterrows():
        ps = {}
        for k, v in row.items():
            if k.startswith("p_") and pd.notna(v):
                key = k[2:]
                if isinstance(v, (np.integer,)):
                    ps[key] = int(v)
                elif isinstance(v, (np.floating,)):
                    ps[key] = round(float(v), 6)
                elif isinstance(v, float) and v == int(v):
                    ps[key] = int(v)
                elif isinstance(v, bool):
                    ps[key] = bool(v)
                elif isinstance(v, np.bool_):
                    ps[key] = bool(v)
                else:
                    ps[key] = v
        warm.append({"trial_id": int(row["trial_id"]), "score": float(row["score"]), "params": ps})
    A(json.dumps(warm, indent=2, ensure_ascii=False))
    A("```")

    # ==================== S6: P1重跑建议 ====================
    A("\n## 6. P1重跑建议（缩窄搜索范围）\n")
    if low_count:
        A("> 数据不足，跳过。\n")
    else:
        A("| 参数 | 推荐模式 | 推荐值/范围 | 依据 |")
        A("|------|---------|------------|------|")

        key_params = [
            ("p_lgbm_learning_rate", "lgbm_learning_rate", "float"),
            ("p_lgbm_max_depth", "lgbm_max_depth", "int"),
            ("p_xgb_learning_rate", "xgb_learning_rate", "float"),
            ("p_drop_short_term_noise", "drop_short_term_noise", "bool"),
        ]

        converged_list = []

        for p_key, display_name, ptype in key_params:
            if p_key not in topH.columns:
                continue
            tv = topH[p_key].dropna()
            if tv.empty:
                continue
            top_mean = tv.mean()
            top_std = tv.std()
            cv = top_std / (abs(top_mean) + 1e-8)

            if ptype == "bool":
                mode_val = tv.mode().iloc[0] if not tv.mode().empty else None
                freq = (tv == mode_val).mean() if mode_val is not None else 0
                if freq > 0.85:
                    mode = f"Fixed-{'True' if mode_val else 'False'}"
                    val = str(bool(mode_val))
                    A(f"| {display_name} | {mode} | {val} | TOP-half mode 频率={freq:.0%} |")
                else:
                    A(f"| {display_name} | Search | True/False | mode 频率={freq:.0%}<85% |")

            elif ptype == "float":
                if cv < 0.15:
                    if "learning_rate" in p_key:
                        val = round(float(top_mean), 6)
                        A(f"| {display_name} | Fixed | {val} | TOP_std/mean={cv:.3f}<0.15 |")
                        converged_list.append((display_name, val))
                    else:
                        val = round(float(top_mean), 4)
                        A(f"| {display_name} | Fixed | {val} | TOP_std/mean={cv:.3f}<0.15 |")
                        converged_list.append((display_name, val))
                else:
                    if "learning_rate" in p_key:
                        lo = max(0.005, top_mean * 0.5)
                        hi = min(0.20, top_mean * 2.0)
                        A(
                            f"| {display_name} | Log-Range | [{lo:.6f}, {hi:.6f}] | "
                            f"TOP_std/mean={cv:.3f}≥0.15 |"
                        )
                    else:
                        lo = max(0.0, top_mean - 1.5 * top_std)
                        hi = top_mean + 1.5 * top_std
                        A(
                            f"| {display_name} | Range | [{lo:.4f}, {hi:.4f}] | "
                            f"TOP_std/mean={cv:.3f}≥0.15 |"
                        )

            elif ptype == "int":
                if cv < 0.15:
                    val = int(round(top_mean))
                    A(f"| {display_name} | Fixed | {val} | TOP_std/mean={cv:.3f}<0.15 |")
                    converged_list.append((display_name, val))
                else:
                    lo = max(0, int(round(top_mean - 1.5 * top_std)))
                    hi = int(round(top_mean + 1.5 * top_std))
                    A(
                        f"| {display_name} | Range | [{lo}, {hi}] | "
                        f"TOP_std/mean={cv:.3f}≥0.15 |"
                    )

        if converged_list:
            A("\n其余已收敛参数（Converged ✓）：\n")
            A("| Param | Fixed值 |")
            A("|-------|---------|")
            for name, val in converged_list:
                A(f"| {name} | {val} |")

    # ==================== S7: P2精调建议 ====================
    A("\n## 7. P2精调建议\n")
    if low_count:
        A("> 数据不足，跳过。\n")
    else:
        A("| 参数 | 推荐模式 | 推荐值/范围 | TOP25%均值参考 |")
        A("|------|---------|------------|---------------|")

        all_param_cols = [c for c in df.columns if c.startswith("p_")]
        converged_count = 0
        total_params = 0

        for p_key in sorted(all_param_cols):
            display_name = p_key[2:]
            total_params += 1
            if p_key not in top25.columns:
                continue
            tv = top25[p_key].dropna()
            if tv.empty:
                continue

            col = df[p_key].dropna()
            is_str = pd.api.types.is_string_dtype(col.dtype)
            is_bool = col.nunique() <= 2 and set(col.unique()).issubset({True, False, 0, 1, 0.0, 1.0})
            is_cat_type = is_str or col.dtype == object or col.dtype == bool or is_bool

            if is_cat_type:
                mode_val = tv.mode().iloc[0] if not tv.mode().empty else None
                freq = (tv == mode_val).mean() if mode_val is not None else 0
                if is_bool:
                    if freq > 0.80:
                        mode = f"Fixed-{'True' if mode_val else 'False'}"
                        val = str(bool(mode_val))
                        A(f"| {display_name} | {mode} | {val} | {val} |")
                        converged_count += 1
                    else:
                        A(f"| {display_name} | Search | True/False | mode freq={freq:.0%} |")
                else:
                    val = str(mode_val)
                    if freq > 0.80:
                        A(f"| {display_name} | Fixed | {val} | {val} |")
                        converged_count += 1
                    else:
                        A(f"| {display_name} | Search | {val} (freq={freq:.0%}) | mode={val} freq={freq:.0%} |")
                continue

            # Numeric type: compute mean/std now
            top25_mean = tv.mean()
            top25_std = tv.std()

            is_int = col.dropna().apply(lambda x: x == int(x)).all() if not col.empty else False
            cv25 = top25_std / (abs(top25_mean) + 1e-8)

            if cv25 < 0.15:
                if is_int:
                    val = int(round(top25_mean))
                    A(f"| {display_name} | Fixed | {val} | {val} |")
                else:
                    val = round(float(top25_mean), 6)
                    A(f"| {display_name} | Fixed | {val} | {val:.6f} |")
                converged_count += 1
            else:
                lo = top25_mean - 1.0 * top25_std
                hi = top25_mean + 1.0 * top25_std
                if is_int:
                    lo = max(0, int(round(lo)))
                    hi = int(round(hi))
                    A(f"| {display_name} | Range | [{lo}, {hi}] | {top25_mean:.2f} |")
                elif "learning_rate" in p_key:
                    A(
                        f"| {display_name} | Log-Range | [{lo:.6f}, {hi:.6f}] | "
                        f"{top25_mean:.6f} |"
                    )
                else:
                    A(f"| {display_name} | Range | [{lo:.4f}, {hi:.4f}] | {top25_mean:.4f} |")

        convergence_rate = converged_count / total_params if total_params > 0 else 0
        median_ir = df["val_rolling6m_ir"].median() if "val_rolling6m_ir" in df.columns else 0.0
        pass_rate_val = n_stage2 / n_stage1 if n_stage1 > 0 else 0

        xgb_depth_vals = top25["p_xgb_max_depth"].dropna() if "p_xgb_max_depth" in top25.columns else pd.Series(dtype=float)
        xgb_depth_mode = int(xgb_depth_vals.mode().iloc[0]) if not xgb_depth_vals.empty else None
        if xgb_depth_mode is not None and xgb_depth_mode != 3:
            A("")
            A(f"> ⚠️ **P2修正提示**: xgb_max_depth 在TOP试验中实际为 `{xgb_depth_mode}`（adaptive depth_mode修正），P2搜索预设若为 `3` 应更正为 `2`。")

        A("")
        if n_stage2 >= 30 and pass_rate_val > 0.50:
            A(f"**✅ GO P2** — Stage2合格Trial ≥ 30，合格率 {pass_rate_val:.1%}，高维空间强信号；TOP中位IR={median_ir:.4f}，数据充分。")
        elif n_stage2 >= 30 and convergence_rate >= 0.70:
            A(f"**✅ GO P2** — Stage2合格Trial ≥ 30 且 核心参数收敛率 {convergence_rate:.0%} ≥ 70%")
        else:
            reasons = []
            if n_stage2 < 30:
                reasons.append(f"Stage2合格Trial={n_stage2} < 30")
            if n_stage2 >= 30 and pass_rate_val <= 0.50 and convergence_rate < 0.70:
                reasons.append(f"合格率={pass_rate_val:.1%}，收敛率={convergence_rate:.0%}")
            A(f"**🔄 RERUN P1** — {'; '.join(reasons[:2])}")

    # ==================== Footer ====================
    A(f"\n---\n*tthh自动生成 | 供Claude(AA)分析*")

    out_path = study_dir / "p1" / f"{study_name}_p1_analysis_report.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  DONE {study_name}: {out_path.stat().st_size/1024:.1f}KB, {len(L)} lines → {out_path}")


def main():
    root = Path(r"E:\10q\10q-202604gpu\output")
    target_dirs = ["15B", "15E", "15A"]

    for d in target_dirs:
        study_dir = root / d
        if not study_dir.exists():
            print(f"SKIP {d}: directory not found")
            continue
        print(f"Processing {d}...")
        generate_report(study_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()