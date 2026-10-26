"""Write the Fairness Metrics Report, Model Card and CSV tables from the run results (numbers always match the run)."""
import json

import pandas as pd

from . import config


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def write_all(out, tier_tbl, dec_tbl, imp, cf, sens):
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    R = config.REPORTS
    (R / "results.json").write_text(json.dumps(out, indent=2, default=float))
    rows = []
    for name, m in out["models"].items():
        t = m["test"]
        rows.append({"model": name, "val_roc_auc": m["validation"]["roc_auc"], "test_roc_auc": t["roc_auc"], "test_ks": t["ks"],
                     "test_capture_top20": t["capture_top20"], "test_precision": t["precision"], "test_recall": t["recall"],
                     "test_f1": t["f1"], "test_accuracy": t["accuracy"], **{f"param_{k}": v for k, v in m["best_params"].items()}})
    pd.DataFrame(rows).round(4).to_csv(R / "table1_model_comparison.csv", index=False)
    tier_tbl.round(4).to_csv(R / "tier_calibration.csv", index=False)
    dec_tbl.round(4).to_csv(R / "decile_calibration.csv", index=False)
    imp.round(4).to_csv(R / "shap_global_importance.csv", index=False)
    cf.round(4).to_csv(R / "counterfactuals.csv", index=False)
    sens.round(4).to_csv(R / "sensitivity.csv", index=False)
    g = []
    for stage in ("fairness_before", "fairness_after"):
        for attr, s in out[stage].items():
            for r in s["groups"]:
                g.append({"stage": stage.split("_")[1], "attribute": attr, **r})
    pd.DataFrame(g).to_csv(R / "fairness_group_details.csv", index=False)
    (R / "fairness_metrics_report.md").write_text(_fairness_md(out), encoding="utf-8")
    (R / "REPORT_NUMBERS.md").write_text(_numbers_md(out, tier_tbl, dec_tbl, imp, cf, sens), encoding="utf-8")


def _fairness_md(out):
    b, a = out["fairness_before"], out["fairness_after"]
    demo = "> **DEMO RUN on synthetic data - do not quote these numbers.**\n\n" if out["demo_data"] else ""
    L = [f"# Fairness Metrics Report - BK credit-risk model (Module 4)\n", demo,
         f"Generated {out['generated']} from `{out['data_source']}`. Selected model: **{out['selected_model']}**, "
         f"scored on the 2015 hold-out set ({out['rows']['test']:,} loans).\n",
         "## Definitions (from the Module 2 plan)\n",
         "- **Flag** = the model sends the applicant to review / possible decline (score at or above the 2014-validation top-20% cutoff). **Approved** = not flagged.",
         f"- **Disparate impact ratio / approval ratio** = lowest group approval rate divided by highest. Limit: above {config.MIN_APPROVAL_RATIO} (four-fifths convention; Feldman et al., 2015).",
         "- **Demographic parity difference** = highest minus lowest group approval rate (points).",
         f"- **Equalized odds** (Hardt et al., 2016) = gaps in true-positive rate (TPR) and false-positive rate (FPR) between groups. Limit: {config.MAX_GAP_PP:.0f} points each.",
         f"- Groups with fewer than {config.MIN_GROUP_N} loans are excluded from the measures. LendingClub has no gender, age or race fields, so the groups are indirect: "
         "income quartile, home ownership, employment length and region.\n",
         "## Summary: before and after mitigation (ThresholdOptimizer, equalized odds)\n",
         "| Group | Stage | Approval ratio | Demographic parity diff (pts) | TPR gap (pts) | FPR gap (pts) | Within limits |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for attr in b:
        for stage, d in (("before", b), ("after", a)):
            s = d[attr]
            L.append(f"| {attr.replace('_', ' ')} | {stage} | {s['approval_ratio']:.3f} | {s['demographic_parity_diff_pp']:.1f} | "
                     f"{s['tpr_gap_pp']:.1f} | {s['fpr_gap_pp']:.1f} | {'yes' if s['passes'] else 'no'} |")
    L += ["", "![Fairness before and after](figures/fig3_fairness_before_after.png)\n", "## Group detail before mitigation\n",
          "| Attribute | Group | Loans | Approval rate | Default rate | TPR | FPR |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for attr, s in b.items():
        for r in s["groups"]:
            L.append(f"| {attr.replace('_', ' ')} | {r['group']} | {r['n']:,} | {pct(r['approval_rate'])} | {pct(r['default_rate'])} | {pct(r['tpr'])} | {pct(r['fpr'])} |")
    L += ["", "## Is risk over- or under-stated for a group? (calibration by group, 2015 test)\n",
          "Relative error = predicted default rate / actual default rate - 1. Positive = risk overstated.\n",
          "| Attribute | Group | Loans | Predicted | Actual | Relative error | Brier score |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for attr, rows in out.get("group_calibration", {}).items():
        for r in rows:
            L.append(f"| {attr.replace('_', ' ')} | {r['group']} | {r['n']:,} | {pct(r['predicted'])} | {pct(r['actual'])} | {r['relative_error'] * 100:+.1f}% | {r['brier']:.3f} |")
    L += ["", "## Cost of mitigation\n", "| Group | Flagged share | Precision | Accuracy |", "| --- | --- | --- | --- |"]
    base = out["models"][out["selected_model"]]["test"]
    L.append(f"| Before (single cutoff) | {pct(base['flagged_share'])} | {pct(base['precision'])} | {pct(base['accuracy'])} |")
    for attr, s in a.items():
        L.append(f"| After: {attr.replace('_', ' ')} | {pct(s['flagged_share'])} | {pct(s['precision'])} | {pct(s['accuracy'])} |")
    L += ["", "## Reading the results\n",
          "- Different approval rates do not by themselves prove unfairness, because default rates differ between groups; the TPR/FPR gaps show who is wrongly flagged or missed.",
          "- Group-specific cutoffs reduce the gaps but raise review workload and can raise policy and legal concerns. The governance team must weigh this trade-off before use.",
          "- Home ownership, income and employment length are also model inputs, so group results can be affected by the inputs themselves.",
          "- Bias checks use indirect groups; protected attributes (gender, age, ethnicity) are absent from the data and must be assessed on BK data.\n",
          "## References\n",
          "Feldman, M., Friedler, S., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and removing disparate impact. *KDD*.",
          "Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.",
          "Bird, S., et al. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft Technical Report."]
    return "\n".join(L) + "\n"


def write_model_card(out):
    m = out["models"][out["selected_model"]]
    t, v = m["test"], m["validation"]
    ml = out.get("mlflow")
    demo = "> **DEMO RUN on synthetic data - regenerate with the real curated file before submitting.**\n\n" if out["demo_data"] else ""
    top = out["importance"][:5]
    tc = {r["tier"]: r for r in out["tier_calibration"]}
    card = f"""# Model Card - BK Credit-Risk Default Model (Module 4)

{demo}## Model details
- **Task:** binary classification, probability that a funded loan defaults (charged off / default / late 31-120 days = 1; fully paid = 0), estimated at application time.
- **Model:** {out['selected_model']} with isotonic probability calibration and three risk tiers (high = top 20%, medium = next 30%, low = bottom 50%).
- **Hyperparameters:** {json.dumps(m['best_params'])}
- **Model choice:** {out['selection_rule']}
- **Version:** {('`' + ml['registered_model'] + '` v' + ml['version'] + ', alias `' + ml['alias'] + '`') if ml else 'not registered (MLflow disabled)'}; trained {out['generated']}.
- **Library versions and parameters** are recorded in the MLflow runs (`mlflow ui`).

## Intended use
- Decision support for credit-risk **review** at Bank of Kigali: prioritise applications for analyst review and explain why (top-3 SHAP reasons).
- **Not** for automated approval or rejection, pricing, or use on BK borrowers before retraining and recalibration on BK data.

## Training and evaluation data
- Source: Module 3 curated LendingClub table (`{out['data_source']}`), {out['rows']['total']:,} resolved loans issued 2007-2015, overall default rate {pct(out['default_rate']['overall'])}.
- Time-based split: train 2007-2013 ({out['rows']['train']:,}), validation 2014 ({out['rows']['validation']:,}), test 2015 ({out['rows']['test']:,}; used once).
- 22 inputs (18 numeric, 4 categorical). No post-loan fields; preprocessing fitted on training data only.

## Performance (2015 test set)
| Metric | Value | Module 2 target |
| --- | --- | --- |
| ROC-AUC | {t['roc_auc']:.3f} (2014 validation {v['roc_auc']:.3f}) | >= {config.OBJ_AUC} |
| KS | {t['ks']:.3f} | >= {config.OBJ_KS} |
| Defaulters in top 20% | {pct(t['capture_top20'])} | >= {pct(config.OBJ_CAPTURE, 0)} |
| Precision / recall / F1 at operating cutoff | {t['precision']:.3f} / {t['recall']:.3f} / {t['f1']:.3f} | - |
| Tier calibration (relative error) | low {tc['low']['relative_error'] * 100:+.1f}%, medium {tc['medium']['relative_error'] * 100:+.1f}%, high {tc['high']['relative_error'] * 100:+.1f}% | within +/-{config.OBJ_CALIB * 100:.0f}% |

## Fairness (see `reports/module4/fairness_metrics_report.md`)
- Groups: income quartile, home ownership, employment length, region (indirect; no protected attributes in the data).
- Metrics: approval (disparate impact) ratio, TPR and FPR gaps, demographic parity difference; before and after Fairlearn ThresholdOptimizer.
- Groups within all limits before mitigation: {sum(s['passes'] for s in out['fairness_before'].values())} of {len(out['fairness_before'])}; after: {sum(s['passes'] for s in out['fairness_after'].values())} of {len(out['fairness_after'])}. Mitigation raises review workload; governance sign-off is required.

## Explainability
- Global SHAP: top features are {', '.join(r['feature'] for r in top)}.
- Local: every API response returns the three largest SHAP contributions. Counterfactual what-ifs (term, amount) and sensitivity tests are in `reports/module4/`.

## Limitations and risks
- US LendingClub data stand in for BK data; retrain and recalibrate on BK loans before any use.
- Only funded loans are observed (selection bias), and performance can shift over time: compare the 2014 validation and 2015 test AUC above.
- Interest rate and grade are set by the lender, so the model partly repeats an earlier lending judgement (AUC without them: {out['ablation_auc']['without_grade_and_interest_rate']:.3f}).
- Moderate discrimination (AUC about {t['roc_auc']:.2f}): use to support, not replace, human review.

## Monitoring
Population stability index (alert above 0.2), AUC, KS, top-20% capture, calibration by tier and group, approval ratios, TPR/FPR gaps, out-of-range inputs and manual overrides; two consecutive breaches trigger review or retraining.

## References
Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\\**.
"""
    config.DOCS.mkdir(parents=True, exist_ok=True)
    (config.DOCS / "model_card.md").write_text(card, encoding="utf-8")


def _numbers_md(out, tier_tbl, dec_tbl, imp, cf, sens):
    """One page of every number the report quotes, so the Word/Google report can be synced in minutes."""
    best = out["selected_model"]
    m = out["models"][best]
    t, v = m["test"], m["validation"]
    tc = {r["tier"]: r for r in out["tier_calibration"]}
    d = out["decile_calibration"]
    bf = out["bias_facts"]
    L = ["# Numbers for the Module 4 report (generated from the last run)\n"]
    if out["demo_data"]:
        L.append("> **DEMO RUN - synthetic data. Do not quote.**\n")
    L += [f"Generated {out['generated']} | selected model: **{best}** | settings: `{json.dumps(m['best_params'])}`\n",
          f"- Model choice: {out['selection_rule']}\n",
          "## Section 2 - data and bias",
          f"- Rows {out['rows']['total']:,}; columns {bf['columns_in_curated_file']}; overall default rate {pct(out['default_rate']['overall'])}",
          f"- Income-quartile default rate, full file: {json.dumps(bf['income_quartile_default_full'])}",
          f"- Income-quartile default rate, 2015 test: {json.dumps(bf['income_quartile_default_test'])}",
          f"- Region default rates, full file: min {pct(min(bf['region_default_full'].values()))}, max {pct(max(bf['region_default_full'].values()))}",
          f"- Home-ownership group sizes: {json.dumps(bf['home_ownership_counts_full'])}",
          f"- AUC of interest rate alone (2015): {out['ablation_auc']['interest_rate_only']:.3f}; full model {t['roc_auc']:.3f}; without grade and interest rate {out['ablation_auc']['without_grade_and_interest_rate']:.3f}\n",
          "## Section 3 - split",
          f"- Train {out['rows']['train']:,} ({pct(out['default_rate']['train'])} default); validation {out['rows']['validation']:,} ({pct(out['default_rate']['validation'])}); test {out['rows']['test']:,} ({pct(out['default_rate']['test'])})",
          f"- Baseline (predict all repaid): accuracy {out['baseline_test']['accuracy']:.3f}, recall 0, AUC 0.50\n",
          "## Section 5 - Table 1 (2015 test)", "| Model | AUC | KS | Top-20% capture | Precision | Recall | F1 | Accuracy | 2014 val AUC |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, mm in out["models"].items():
        x = mm["test"]
        L.append(f"| {name} | {x['roc_auc']:.3f} | {x['ks']:.3f} | {pct(x['capture_top20'])} | {x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} | {x['accuracy']:.3f} | {mm['validation']['roc_auc']:.3f} |")
    L += ["", f"- Operating cutoff flagged {pct(t['flagged_share'])} of 2015 loans: caught {t['tp']:,} of {t['defaulters']:,} defaulters (recall {pct(t['recall'])}); wrongly flagged {t['fp']:,} repaid loans",
          f"- Tier calibration (relative error): low {tc['low']['relative_error'] * 100:+.1f}%, medium {tc['medium']['relative_error'] * 100:+.1f}%, high {tc['high']['relative_error'] * 100:+.1f}%; deciles {min(x['relative_error'] for x in d) * 100:+.1f}% to {max(x['relative_error'] for x in d) * 100:+.1f}%",
          f"- Default rate by tier (actual): low {pct(tc['low']['actual'])}, medium {pct(tc['medium']['actual'])}, high {pct(tc['high']['actual'])}",
          f"- Top decile predicted {pct(d[-1]['predicted'])} vs actual {pct(d[-1]['actual'])}", "",
          "## Table 2 - status against Module 2 objectives",
          f"- ROC-AUC >= {config.OBJ_AUC}: {t['roc_auc']:.3f} -> {'Met' if t['roc_auc'] >= config.OBJ_AUC else 'Not met'}",
          f"- KS >= {config.OBJ_KS}: {t['ks']:.3f} -> {'Met' if t['ks'] >= config.OBJ_KS else 'Not met'}",
          f"- Top-20% capture >= {pct(config.OBJ_CAPTURE, 0)}: {pct(t['capture_top20'])} -> {'Met' if t['capture_top20'] >= config.OBJ_CAPTURE else 'Not met'}",
          f"- Tier calibration within +/-10%: {'Met' if all(abs(tc[k]['relative_error']) <= config.OBJ_CALIB for k in tc) else 'Not met (see tiers above)'}", "",
          "## Section 6 - explainability",
          "- Global SHAP (mean |SHAP|), top 8: " + ", ".join(f"{r['feature']} {r['mean_abs_shap']:.3f}" for r in out["importance"][:8])]
    for lab, e in out["examples"].items():
        L.append(f"- {lab}: probability {e['default_probability']:.3f}; reasons: " + "; ".join(f"{r['feature']}={r['value']} ({r['effect']})" for r in e["reasons"]))
    L += ["", "### Counterfactuals (copy the sentence that fits)", "| Applicant | Scenario | Term | Amount | Instalment | Default prob. | Tier |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in out["counterfactuals"]:
        L.append(f"| {r['applicant']} | {r['scenario']} | {r['term']} | {r['loan_amnt']:,.0f} | {r['installment']:,.0f} | {r['default_probability']:.3f} | {r['tier']} |")
    L += ["", "### Sensitivity (percentage-point change in average predicted default)"]
    L += [f"- {r['scenario']}: {r['change_pp']:+.2f}" for r in out["sensitivity"]]
    L.append(f"- 5% random noise: AUC {out['sensitivity_auc']['clean']:.3f} -> {out['sensitivity_auc']['noise_5pct']:.3f}")
    L += ["", "### Table 3 - fairness (before / after)", "| Group | Approval ratio | TPR gap | FPR gap |", "| --- | --- | --- | --- |"]
    for attr in out["fairness_before"]:
        for stage, key in (("before", "fairness_before"), ("after", "fairness_after")):
            s = out[key][attr]
            L.append(f"| {attr.replace('_', ' ')}, {stage} | {s['approval_ratio']:.3f} | {s['tpr_gap_pp']:.1f} | {s['fpr_gap_pp']:.1f} |")
    base = t
    L += ["", f"- Mitigation cost: flagged share {pct(base['flagged_share'])} -> " + ", ".join(f"{pct(s['flagged_share'], 0)}" for s in out["fairness_after"].values()) +
          f"; precision {pct(base['precision'])} -> " + ", ".join(pct(s['precision'], 0) for s in out["fairness_after"].values()) +
          f"; accuracy {pct(base['accuracy'])} -> " + ", ".join(pct(s['accuracy'], 0) for s in out["fairness_after"].values())]
    L += ["", "### Calibration by group (relative error, 2015 test) - largest overstatements"]
    flat = [(a, r) for a, rows in out.get("group_calibration", {}).items() for r in rows]
    for a, r in sorted(flat, key=lambda x: -x[1]["relative_error"])[:5]:
        L.append(f"- {a.replace('_', ' ')} / {r['group']}: predicted {pct(r['predicted'])} vs actual {pct(r['actual'])} ({r['relative_error'] * 100:+.1f}%), Brier {r['brier']:.3f}")
    ml = out.get("mlflow")
    if ml:
        L += ["", f"## MLflow\n- Model `{ml['registered_model']}` version {ml['version']}, alias `{ml['alias']}`; experiment `{ml['experiment']}`"]
    return "\n".join(L) + "\n"
