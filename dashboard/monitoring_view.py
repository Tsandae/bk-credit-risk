"""Monitoring view (data drift, model drift, bias drift). Used by dashboard/app.py (page 7) and dashboard/monitoring_app.py."""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import helpers as H

NICE = {"score": "Model score", "int_rate": "Interest rate", "annual_inc": "Annual income", "dti": "Debt vs income",
        "loan_amnt": "Loan amount", "revol_util": "Credit limits in use", "loan_to_income": "Loan size vs income",
        "term": "Loan term", "grade": "Lender's grade", "home_ownership": "Housing status", "purpose": "Loan purpose"}


def _style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=70, b=60), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color=H.NAVY, size=13), legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
                      title=dict(x=0, xanchor="left", font=dict(size=15)))
    fig.update_xaxes(showgrid=False, linecolor="#c8d0da")
    fig.update_yaxes(gridcolor="#e6ebf1", zeroline=False, automargin=True)
    return fig


def load():
    p = Path(os.environ.get("MONITORING_DIR", H.DATA)) / "monitoring.json"
    if not p.exists():
        p = H.DATA / "monitoring.json"
    return json.loads(p.read_text()) if p.exists() else None


def render():
    rep = load()
    st.title("Monitoring: is the model still behaving as it did?")
    st.markdown('<div class="note">Each later year of loans is compared with the loans the model learned from (2007 to 2013). '
                'The checks cover <b>data drift</b> (are applicants changing?), <b>model drift</b> (is it ranking worse?), '
                '<b>calibration drift</b> (are risk estimates still honest?) and <b>bias drift</b> (are groups still treated comparably?). '
                'Two breaches in a row mean review or retrain.</div>', unsafe_allow_html=True)
    if rep is None:
        st.error("No monitoring report found. Run: python -m src.final.monitor")
        return
    wins = rep["windows"]
    names = list(wins)
    lim = rep["limits"]
    st.caption(f"Report generated {rep['generated']}. Reference: {rep['reference']} ({rep['reference_n']:,} loans). Data: US LendingClub proxy.")

    cols = st.columns(len(names))
    for c, n in zip(cols, names):
        m = wins[n]
        c.markdown(f"#### {n}")
        c.metric("Loans", f"{m['n']:,}")
        c.metric("Ranking quality (AUC)", f"{m['auc']:.3f}")
        status = ("⚠️ **Breach: " + m["action"] + "**") if m["breach"] else "✅ **Within limits**"
        c.markdown(status)
        c.caption(f"{len(m['alerts'])} alert(s) in this window.")

    st.subheader("Data drift: have applicants changed?")
    last = names[-1]
    psi = wins[last]["psi"]
    d = pd.DataFrame({"item": [NICE.get(k, k) for k in psi], "psi": list(psi.values())}).sort_values("psi")
    fig = go.Figure(go.Bar(x=d["psi"], y=d["item"], orientation="h", marker_color=[H.ORANGE if v > lim["psi"] else H.BLUE for v in d["psi"]],
                           text=[f"{v:.3f}" for v in d["psi"]], textposition="outside"))
    fig.add_vline(x=lim["psi"], line_dash="dash", line_color=H.NAVY, annotation_text="alert limit 0.2", annotation_position="top")
    fig.update_layout(title=f"Population stability index, {last} vs 2007 to 2013 (higher = more change)", xaxis_title="PSI")
    fig.update_xaxes(range=[0, max(0.3, float(d["psi"].max()) * 1.3)])
    st.plotly_chart(_style(fig, 420), use_container_width=True, theme=None)
    worst = d.iloc[-1]
    st.caption(f"Largest shift: {worst['item']} (PSI {worst['psi']:.3f}). "
               + ("All inputs are below the 0.2 alert limit." if (d["psi"] <= lim["psi"]).all() else "At least one input is above the 0.2 alert limit."))

    st.subheader("Model drift: is ranking quality holding?")
    perf = pd.DataFrame({"Window": names, "AUC": [wins[n]["auc"] for n in names], "KS": [wins[n]["ks"] for n in names],
                         "Defaulters in top 20%": [wins[n]["capture_top20"] for n in names]})
    fig = go.Figure()
    fig.add_bar(x=perf["Window"], y=perf["AUC"], name="AUC", marker_color=H.BLUE, text=[f"{v:.3f}" for v in perf["AUC"]], textposition="outside")
    fig.add_bar(x=perf["Window"], y=perf["KS"], name="KS", marker_color=H.AQUA, text=[f"{v:.3f}" for v in perf["KS"]], textposition="outside")
    fig.update_layout(barmode="group", title="AUC and KS by window (project goals: AUC 0.70, KS 0.25)", yaxis_range=[0, 1])
    st.plotly_chart(_style(fig), use_container_width=True, theme=None)

    st.subheader("Bias drift: are groups still treated comparably?")
    rows = []
    for n in names:
        for g, f in wins[n]["fairness"].items():
            if "error" in f:
                continue
            rows.append({"Window": n, "Group type": H.GROUP_LABELS[g], "Approval ratio": f["approval_ratio"],
                         "TPR gap (pts)": f["tpr_gap_pp"], "FPR gap (pts)": f["fpr_gap_pp"]})
    fd = pd.DataFrame(rows)
    fig = go.Figure()
    for i, n in enumerate(names):
        s = fd[fd["Window"] == n]
        fig.add_bar(x=s["Group type"], y=s["Approval ratio"], name=n, marker_color=[H.BLUE, H.ORANGE][i % 2],
                    text=[f"{v:.2f}" for v in s["Approval ratio"]], textposition="outside")
    fig.add_hline(y=lim["approval_ratio"], line_dash="dash", line_color=H.NAVY, annotation_text="floor 0.80", annotation_position="bottom right", layer="below")
    fig.update_layout(barmode="group", title="Approval ratio (lowest group ÷ highest group), single cut-off in use", yaxis_range=[0, 1.15])
    st.plotly_chart(_style(fig), use_container_width=True, theme=None)
    fd["Status"] = [("✅ Within limits" if (r["Approval ratio"] >= lim["approval_ratio"] and r["TPR gap (pts)"] <= lim["gap_pp"] and r["FPR gap (pts)"] <= lim["gap_pp"]) else "⚠️ Outside limit") for _, r in fd.iterrows()]
    st.dataframe(fd.assign(**{"Approval ratio": fd["Approval ratio"].round(2), "TPR gap (pts)": fd["TPR gap (pts)"].round(1), "FPR gap (pts)": fd["FPR gap (pts)"].round(1)}),
                 hide_index=True, use_container_width=True)
    st.markdown('<div class="warn"><b>Read with care.</b> These bias breaches are not new: they were already present in 2014 and match the Module 4 fairness audit. '
                'The group-specific cut-offs that fixed them are an analysis, not what the live model uses, so the breaches continue until governance decides whether to adopt them.</div>', unsafe_allow_html=True)

    st.subheader("All alerts")
    al = [{"Window": n, "Type": a["type"], "Item": a["item"], "Value": a["value"], "Limit": a["limit"]} for n in names for a in wins[n]["alerts"]]
    if al:
        st.dataframe(pd.DataFrame(al), hide_index=True, use_container_width=True)
    else:
        st.success("No alerts.")
    st.caption("Run on new data: python -m src.final.monitor --current new_batch.parquet --label 2026-Q1")
