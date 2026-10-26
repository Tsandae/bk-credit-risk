"""Train, tune, calibrate, evaluate, explain, audit fairness, log to MLflow and write all Module 4 outputs.

    python -m src.module4.run_all                      # real curated file from Module 3
    python -m src.module4.run_all --data path.parquet
    python -m src.module4.run_all --demo --quick       # synthetic dry run (NOT for reported results)
"""
import argparse
import json
from pathlib import Path
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from . import config, data, evaluate as ev, fairness, plots, scenarios
from .explain import Explainer, global_importance
from .model import calibrate, candidates, category_levels, feature_frame, tune
from . import reports


def _mlflow_setup():
    import mlflow
    config.ROOT.joinpath("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT) is None:
        mlflow.create_experiment(config.EXPERIMENT, artifact_location=config.ROOT.joinpath("mlruns").as_uri())
    mlflow.set_experiment(config.EXPERIMENT)
    return mlflow


def _log_model(mlflow, obj, name, registered=None):
    kw = dict(registered_model_name=registered, serialization_format="cloudpickle")
    try:
        return mlflow.sklearn.log_model(obj, name=name, **kw)
    except TypeError:                      # older MLflow versions use artifact_path
        return mlflow.sklearn.log_model(obj, artifact_path=name, **kw)


def run(path=None, quick=False, demo=False, use_mlflow=True):
    t0 = time.time()
    if demo:
        from .demo_data import make_demo_curated
        df = make_demo_curated(12000 if quick else 60000)
        source = "SYNTHETIC DEMO DATA (not for reporting)"
    else:
        df = data.load(path)
        source = "data/curated/loans_curated.parquet" if path is None else Path(path).name
    train, val, test = data.split(df)
    edges = data.income_edges(train)
    train, val, test = (data.add_groups(d, edges) for d in (train, val, test))
    Xtr, Xva, Xte = (feature_frame(d) for d in (train, val, test))
    ytr, yva, yte = (d[config.TARGET].astype(int) for d in (train, val, test))
    levels = category_levels(train)
    print(f"data: {source} | rows {len(df):,} | train {len(train):,} val {len(val):,} test {len(test):,}")

    mlflow = _mlflow_setup() if use_mlflow else None
    n_iter = 2 if quick else config.N_ITER
    sample = 4000 if quick else config.CV_SAMPLE

    # ---- 1. tune + validate each algorithm (2014 validation only; the 2015 test set is untouched) ----
    fitted, results = {}, {}
    for name, (pipe, grid) in candidates(levels).items():
        print(f"tuning {name} ...")
        run_ctx = mlflow.start_run(run_name=name) if mlflow else None
        pipe, best, cv_auc, cv_tbl = tune(name, pipe, grid, Xtr, ytr, n_iter=n_iter, sample=sample)
        model = calibrate(pipe, levels, Xva, yva, name)
        val_m = ev.evaluate(model, Xva, yva)
        fitted[name] = model
        results[name] = {"best_params": best, "cv_auc": cv_auc, "validation": val_m}
        if mlflow:
            mlflow.log_params({**best, "cv_folds": config.CV_FOLDS, "search_sample": sample, "n_iter": n_iter,
                               "class_weight": "balanced", "calibration": "isotonic on 2014",
                               "train_years": f"2007-{config.TRAIN_LAST_YEAR}", "data_source": source[:200]})
            mlflow.log_metric("cv_roc_auc", cv_auc)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items() if isinstance(v, float)})
            cv_tbl.to_csv(config.ROOT / f"cv_{name}.csv", index=False)
            mlflow.log_artifact(str(config.ROOT / f"cv_{name}.csv"))
            (config.ROOT / f"cv_{name}.csv").unlink()
            _log_model(mlflow, model.pipeline, "pipeline")
            mlflow.end_run()

    val_auc = {k: results[k]["validation"]["roc_auc"] for k in results}
    best_auc_name = max(val_auc, key=val_auc.get)
    best_name = config.MAIN_MODEL if val_auc[config.MAIN_MODEL] >= val_auc[best_auc_name] - config.SELECT_TOL else best_auc_name
    selection_rule = (f"{config.MAIN_MODEL} is the Module 2 main model; kept unless its 2014 validation AUC is more than "
                      f"{config.SELECT_TOL} below the best model. Validation AUCs: " + ", ".join(f"{k} {v:.4f}" for k, v in val_auc.items()))
    final = fitted[best_name]
    print(f"selected: {best_name} | {selection_rule}")

    # ---- 2. one-shot evaluation on 2015 test ----
    test_m = {k: ev.evaluate(m, Xte, yte) for k, m in fitted.items()}
    for k in results:
        results[k]["test"] = test_m[k]
    tier_tbl = ev.tier_calibration(final, Xte, yte)
    dec_tbl = ev.decile_calibration(final, Xte, yte)

    # removing grade + interest rate (leakage / lender-judgement check) and interest rate alone
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone
    ablation = {}
    for label, drop in {"interest_rate_only": None, "without_grade_and_interest_rate": ["grade", "int_rate"]}.items():
        if drop is None:
            ablation[label] = float(roc_auc_score(yte, Xte["int_rate"]))
        else:
            ablation[label] = float(_ablate(final, drop, Xtr, ytr, Xte, yte, levels))

    # ---- 3. explainability ----
    expl = Explainer(final)
    sample_idx = np.random.default_rng(config.SEED).choice(len(Xte), min(2000, len(Xte)), replace=False)
    Xs = Xte.iloc[sample_idx].reset_index(drop=True)
    agg, base = expl.values(Xs)
    imp = global_importance(agg)
    raw_s = final.raw(Xs)
    hi, lo, mid = int(np.argmax(raw_s)), int(np.argmin(raw_s)), int(np.argsort(raw_s)[len(raw_s) // 2])
    wf = plots.shap_waterfalls(agg, base, Xs, [hi, mid, lo], ["highest risk", "medium risk", "lowest risk"])
    f2 = plots.shap_summary(agg, Xs)
    examples = {lab: {"default_probability": float(final.predict_proba(Xs.iloc[[i]])[0]),
                      "reasons": expl.reasons(Xs.iloc[[i]])}
                for lab, i in [("highest_risk", hi), ("medium_risk", mid), ("lowest_risk", lo)]}
    cf = scenarios.counterfactuals(final, test.reset_index(drop=True))
    sens = scenarios.sensitivity(final, test.reset_index(drop=True))

    # ---- 4. fairness ----
    gcols = config.FAIR_GROUPS
    gva, gte = val[gcols].reset_index(drop=True), test[gcols].reset_index(drop=True)
    Xva_, Xte_, yva_, yte_ = Xva.reset_index(drop=True), Xte.reset_index(drop=True), yva.reset_index(drop=True), yte.reset_index(drop=True)
    before = fairness.audit(final, Xte_, yte_, gte)
    after = fairness.mitigate(final, Xva_, yva_, gva, Xte_, yte_, gte)
    group_calibration = {a: fairness.calibration_by_group(final, Xte_, yte_, gte[a]) for a in gcols}
    f3 = plots.fairness_chart(before, after)
    f1 = plots.figure1({k: fitted[k] for k in fitted}, Xte, yte, final, dec_tbl, test_m[best_name])

    full = data.add_groups(df, edges)
    bias_facts = {
        "income_quartile_default_full": full.groupby("income_quartile")[config.TARGET].mean().round(4).to_dict(),
        "income_quartile_default_test": test.groupby("income_quartile")[config.TARGET].mean().round(4).to_dict(),
        "region_default_full": full.groupby("region")[config.TARGET].mean().round(4).to_dict(),
        "home_ownership_counts_full": full["home_ownership"].value_counts().to_dict(),
        "default_rate_by_year": full.groupby("issue_year")[config.TARGET].mean().round(4).to_dict(),
        "columns_in_curated_file": int(df.shape[1]),
    }
    out = {
        "bias_facts": bias_facts, "generated": datetime.now().isoformat(timespec="seconds"), "demo_data": demo, "data_source": source,
        "rows": {"total": int(len(df)), "train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "default_rate": {"overall": float(df[config.TARGET].mean()), "train": float(ytr.mean()),
                         "validation": float(yva.mean()), "test": float(yte.mean())},
        "models": results, "selected_model": best_name, "selection_rule": selection_rule, "baseline_test": ev.baseline(yte),
        "tier_calibration": tier_tbl.to_dict(orient="records"), "decile_calibration": dec_tbl.to_dict(orient="records"),
        "ablation_auc": ablation, "importance": imp.round(4).to_dict(orient="records"),
        "examples": examples, "counterfactuals": cf.to_dict(orient="records"),
        "sensitivity": sens.to_dict(orient="records"),
        "sensitivity_auc": {"clean": sens.attrs["auc_clean"], "noise_5pct": sens.attrs["auc_noise5"]},
        "fairness_before": before, "fairness_after": after, "group_calibration": group_calibration, "runtime_seconds": round(time.time() - t0, 1),
    }

    # ---- 5. save model, register, write reports ----
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_FILE)
    reports.write_all(out, tier_tbl, dec_tbl, imp, cf, sens)
    if mlflow:
        with mlflow.start_run(run_name="final_evaluation_2015_test") as r:
            mlflow.set_tags({"stage": "test", "selected_model": best_name})
            mlflow.log_params(results[best_name]["best_params"])
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m[best_name].items() if isinstance(v, float)})
            mlflow.log_metric("auc_without_grade_and_rate", ablation["without_grade_and_interest_rate"])
            for p in [f1, f2, f3, *wf]:
                mlflow.log_artifact(str(p), "figures")
            for f in ["results.json", "fairness_metrics_report.md"]:
                mlflow.log_artifact(str(config.REPORTS / f), "reports")
            info = _log_model(mlflow, final, "credit_risk_model", registered=config.REGISTERED_MODEL)
        client = mlflow.tracking.MlflowClient()
        ver = getattr(info, "registered_model_version", None) or max(
            int(v.version) for v in client.search_model_versions(f"name='{config.REGISTERED_MODEL}'"))
        try:
            client.set_registered_model_alias(config.REGISTERED_MODEL, "staging", ver)
        except Exception as e:  # pragma: no cover
            print("alias not set:", e)
        out["mlflow"] = {"experiment": config.EXPERIMENT, "registered_model": config.REGISTERED_MODEL, "version": str(ver), "alias": "staging"}
    if mlflow:  # evidence table of the real, timestamped runs (for the report / GitHub)
        runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT])
        keep = [c for c in runs.columns if c.startswith(("run_id", "start_time", "tags.mlflow.runName", "params.", "metrics."))]
        runs[keep].to_csv(config.REPORTS / "mlflow_runs.csv", index=False)
    (config.REPORTS / "results.json").write_text(json.dumps(out, indent=2, default=float))
    reports.write_model_card(out)
    print(f"done in {out['runtime_seconds']}s -> {config.REPORTS}")
    return out


def _ablate(final, drop, Xtr, ytr, Xte, yte, levels):
    """Refit the chosen algorithm without some features and report test AUC (same hyperparameters)."""
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from .model import make_preprocessor, candidates
    keep_num = [c for c in config.NUMERIC if c not in drop]
    keep_cat = [c for c in config.CATEGORICAL if c not in drop]
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    lv = [l for c, l in zip(config.CATEGORICAL, levels) if c in keep_cat]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median", keep_empty_features=True), keep_num),
                             ("cat", OneHotEncoder(categories=lv, handle_unknown="ignore", sparse_output=False), keep_cat)])
    clf = clone(final.pipeline.named_steps["clf"])
    if "monotone_constraints" in clf.get_params():
        names = keep_num + [f"{c}={v}" for c, l in zip(keep_cat, lv) for v in l]
        clf.set_params(monotone_constraints=[1 if n in config.MONOTONE_UP else 0 for n in names])
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
    return roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--demo", action="store_true", help="synthetic dry run, not for reporting")
    ap.add_argument("--quick", action="store_true", help="tiny search, for tests")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    run(a.data, quick=a.quick, demo=a.demo, use_mlflow=not a.no_mlflow)
