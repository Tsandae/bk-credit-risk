"""BK Credit-Risk Stakeholder Dashboard (Module 5).
 
Run locally:   streamlit run dashboard/app.py
Deploy:        Streamlit Community Cloud, main file path = dashboard/app.py
This dashboard explains a decision-SUPPORT model built on US LendingClub data (a stand-in for Bank of Kigali data).
It does not make or recommend credit decisions.
"""
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
 
import helpers as H  # noqa: E402
 
st.set_page_config(page_title="BK Credit-Risk Insights", page_icon="📊", layout="wide")
 
st.markdown("""
<style>
/* Always show the light design, even if the viewer's browser is in dark mode */
:root {color-scheme: light;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#ffffff !important;}
[data-testid="stSidebar"] {background:#eef3f8 !important;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {color:#0b2545 !important;}
.note, .note *, .warn, .warn * {color:#0b2545 !important;}
.tier {color:#ffffff !important;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.note {background:#eef3f8; border-left:4px solid #2a78d6; padding:.7rem 1rem; border-radius:4px; margin:.5rem 0 1rem 0;}
.warn {background:#fff4ec; border-left:4px solid #eb6834; padding:.7rem 1rem; border-radius:4px; margin:.5rem 0 1rem 0;}
.tier {display:inline-block; padding:.25rem .8rem; border-radius:1rem; font-weight:700; color:white;}
</style>""", unsafe_allow_html=True)
 
TIER_COLOR = {"low": H.BLUE, "medium": "#8d99aa", "high": H.ORANGE}
TIER_LABEL = {"low": "Lower-risk group", "medium": "Medium-risk group", "high": "Higher-risk group"}
 
 
def style(fig, height=380, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=70, b=60), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color=H.NAVY, size=13), showlegend=legend,
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0), title=dict(x=0, xanchor="left", font=dict(size=15)))
    fig.update_xaxes(showgrid=False, linecolor="#c8d0da")
    fig.update_yaxes(gridcolor="#e6ebf1", zeroline=False, automargin=True)
    return fig
 
 
@st.cache_data
def results():
    return H.load_results()
 
 
@st.cache_data
def scored():
    return H.load_scored()
 
 
@st.cache_resource
def load_model():
    """Model + SHAP explainer. Returns (model, explainer, error_message)."""
    try:
        import joblib
        from src.module4 import config
        model = joblib.load(config.MODEL_FILE)
    except Exception as e:  # missing file or library-version mismatch
        return None, None, f"{type(e).__name__}: {e}"
    try:
        from src.module4.explain import Explainer
        return model, Explainer(model), None
    except Exception as e:
        return model, None, f"explanations unavailable ({type(e).__name__})"
 
 
R = results()
MODEL = R["selected_model"]
T = R["models"][MODEL]["test"]
 
# ------------------------------------------------------------------ sidebar
st.sidebar.title("BK Credit-Risk Insights")
st.sidebar.caption("Decision-support model for credit review. Built for a stakeholder audience (AIA6550, Module 5).")
page = st.sidebar.radio("Explore", ["1. Overview", "2. Score an applicant & what-if", "3. What drives predictions",
                                    "4. Ethical compliance (fairness)", "5. Limits & transparency"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.info("**Data:** US LendingClub loans (2007-2015), used as a stand-in for Bank of Kigali data. "
                "Results show how the method works; they are not BK results.")
 
# ------------------------------------------------------------------ 1. overview
if page.startswith("1"):
    st.title("How well does the credit-risk model separate safer from riskier loans?")
    st.markdown('<div class="note">The model gives each loan application a <b>risk estimate</b> and sorts applications into three groups. '
                'It is meant to <b>support</b> a credit officer, never to decide alone.</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Loans studied", f"{R['rows']['total']:,}")
    c[1].metric("Test loans (2015)", f"{R['rows']['test']:,}")
    c[2].metric("Defaults in the top-20% riskiest", f"{T['capture_top20']:.1%}", help="Share of all loans that later defaulted, found in the 20% of applications scored riskiest. Random scoring would find 20%.")
    c[3].metric("Ranking quality (AUC)", f"{T['roc_auc']:.3f}", help="0.5 = no better than chance, 1.0 = perfect.")
 
    tiers = pd.DataFrame(R["tier_calibration"])
    tiers["label"] = tiers["tier"].map({"low": "Lower risk", "medium": "Medium risk", "high": "Higher risk"})
    left, right = st.columns([5, 4])
    with left:
        fig = go.Figure()
        fig.add_bar(x=tiers["label"], y=tiers["actual"] * 100, name="Actually defaulted", marker_color=H.BLUE,
                    text=[f"{v:.1f}%" for v in tiers["actual"] * 100], textposition="outside")
        fig.add_bar(x=tiers["label"], y=tiers["predicted"] * 100, name="Model expected", marker_color=H.ORANGE,
                    text=[f"{v:.1f}%" for v in tiers["predicted"] * 100], textposition="outside")
        fig.update_layout(barmode="group", title="Share of loans that defaulted, by risk group (2015 test loans)",
                          yaxis_title="% of loans", yaxis_range=[0, 60])
        st.plotly_chart(style(fig), use_container_width=True, theme=None)
        st.caption(f"Loans in the higher-risk group defaulted about {tiers.loc[2, 'actual'] / tiers.loc[0, 'actual']:.1f} times as often as loans in the lower-risk group. "
                   "The model slightly over-states risk in the highest group.")
    with right:
        st.subheader("Progress against the project goals")
        goals = [
            ("Ranking quality (AUC) at least 0.70", f"{T['roc_auc']:.3f}", T["roc_auc"] >= 0.70),
            ("Separation (KS) at least 0.25", f"{T['ks']:.3f}", T["ks"] >= 0.25),
            ("40% of defaults found in the top 20%", f"{T['capture_top20']:.1%}", T["capture_top20"] >= 0.40),
            ("Risk estimates within ±10% in every group", "higher-risk group +12.0%", False),
        ]
        md = "| Goal | Result | Status |\n|---|---|---|\n" + "\n".join(f"| {g} | {r} | {'✅ Met' if ok else '⚠️ Not met'} |" for g, r, ok in goals)
        st.markdown(md)
        st.caption("A cost-saving goal (10% lower expected loss) cannot be assessed without Bank of Kigali cost data.")
 
# ------------------------------------------------------------------ 2. what-if
elif page.startswith("2"):
    st.title("Score an application and test a what-if")
    model, expl, err = load_model()
    if model is None:
        st.error("The model file could not be loaded in this environment, so this page is unavailable. "
                 f"Details: {err}. The other pages still work.")
        st.stop()
    st.markdown('<div class="note">Enter an application, or start from an example. The result is an <b>estimate for review support</b>, '
                'not a credit decision. Example values are in US dollars (LendingClub data).</div>', unsafe_allow_html=True)
    levels = dict(zip(["grade", "home_ownership", "verification_status", "purpose"], model.levels))
    PRESETS = {
        "Typical applicant": dict(loan_amnt=12000, term=36, int_rate=13.5, grade="C", annual_inc=55000, dti=18.0, revol_util=45.0,
                                  revol_bal=9000, credit_history_years=12, home_ownership="RENT", verification_status="Verified",
                                  purpose="debt_consolidation", delinq_2yrs=0, inq_last_6mths=0, open_acc=10, pub_rec=0, total_acc=20, emp_length=5),
        "Higher-risk example": dict(loan_amnt=21700, term=60, int_rate=26.0, grade="F", annual_inc=25000, dti=29.7, revol_util=80.0,
                                    revol_bal=15000, credit_history_years=6, home_ownership="RENT", verification_status="Verified",
                                    purpose="debt_consolidation", delinq_2yrs=1, inq_last_6mths=3, open_acc=9, pub_rec=0, total_acc=14, emp_length=2),
        "Lower-risk example": dict(loan_amnt=8000, term=36, int_rate=5.3, grade="A", annual_inc=250000, dti=8.0, revol_util=4.4,
                                   revol_bal=3000, credit_history_years=22, home_ownership="MORTGAGE", verification_status="Not Verified",
                                   purpose="credit_card", delinq_2yrs=0, inq_last_6mths=0, open_acc=12, pub_rec=0, total_acc=30, emp_length=10),
    }
    preset = st.selectbox("Start from", list(PRESETS))
    p, k = PRESETS[preset], preset.replace(" ", "_")
    a, b, c = st.columns(3)
    with a:
        loan = st.number_input("Loan amount ($)", 1000, 40000, int(p["loan_amnt"]), 500, key=f"loan{k}")
        term = st.selectbox("Loan term (months)", [36, 60], index=[36, 60].index(p["term"]), key=f"term{k}")
        rate = st.number_input("Interest rate (%)", 5.0, 31.0, float(p["int_rate"]), 0.5, key=f"rate{k}")
        grade = st.selectbox("Lender's credit grade", levels["grade"], index=levels["grade"].index(p["grade"]), key=f"grade{k}",
                             help="Grade and interest rate are set together by the lender. Change both for a realistic scenario.")
        purpose = st.selectbox("Loan purpose", levels["purpose"], format_func=lambda x: x.replace("_", " ").capitalize(), index=levels["purpose"].index(p["purpose"]), key=f"purp{k}")
    with b:
        income = st.number_input("Annual income ($)", 10000, 400000, int(p["annual_inc"]), 1000, key=f"inc{k}")
        dti = st.number_input("Debt compared with income (%)", 0.0, 45.0, float(p["dti"]), 0.5, key=f"dti{k}")
        util = st.number_input("Share of credit limits in use (%)", 0.0, 120.0, float(p["revol_util"]), 1.0, key=f"util{k}")
        hist = st.number_input("Credit history (years)", 1, 50, int(p["credit_history_years"]), 1, key=f"hist{k}")
        emp = st.number_input("Years in current job", 0, 10, int(p["emp_length"]), 1, key=f"emp{k}")
    with c:
        home = st.selectbox("Housing status", levels["home_ownership"], index=levels["home_ownership"].index(p["home_ownership"]), key=f"home{k}")
        ver = st.selectbox("Income verification", levels["verification_status"], index=levels["verification_status"].index(p["verification_status"]), key=f"ver{k}")
        inq = st.number_input("Credit enquiries (last 6 months)", 0, 10, int(p["inq_last_6mths"]), 1, key=f"inq{k}")
        delinq = st.number_input("Late payments (last 2 years)", 0, 10, int(p["delinq_2yrs"]), 1, key=f"del{k}")
        revbal = st.number_input("Revolving balance ($)", 0, 200000, int(p["revol_bal"]), 500, key=f"rb{k}")
    base = dict(loan_amnt=loan, term=term, int_rate=rate, grade=grade, annual_inc=income, dti=dti, revol_util=util,
                revol_bal=revbal, credit_history_years=hist, home_ownership=home, verification_status=ver, purpose=purpose,
                delinq_2yrs=delinq, inq_last_6mths=inq, open_acc=p["open_acc"], pub_rec=p["pub_rec"], total_acc=p["total_acc"], emp_length=emp)
    frame = H.build_frame(model, base)
    res = H.score(model, frame)
    bands = H.tier_bands(model)
 
    st.markdown("---")
    r1, r2 = st.columns([2, 3])
    with r1:
        st.markdown(f"### Risk estimate\n<span class='tier' style='background:{TIER_COLOR[res['tier']]}'>{TIER_LABEL[res['tier']]}</span>", unsafe_allow_html=True)
        pct = res["probability"] * 100
        st.metric("Estimated chance of default", "under 1%" if pct < 1 else f"{pct:.0f}%")
        st.caption(f"About {max(round(pct), 0):.0f} in 100 loans with a similar profile defaulted in the test data. "
                   f"Group boundaries: medium risk from about {bands['medium_from'] * 100:.0f}%, higher risk from about {bands['high_from'] * 100:.0f}%.")
        st.caption(f"Monthly instalment used: ${frame.iloc[0]['installment']:,.0f}")
    with r2:
        if expl is None:
            st.info("Explanations are unavailable in this environment.")
        else:
            d = H.local_drivers(expl, frame)
            d = d.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=d["push"], y=[f"{l}: {H.fmt_value(f, v)}" for l, f, v in zip(d["label"], d["feature"], d["value"])], orientation="h",
                marker_color=[H.ORANGE if x > 0 else H.BLUE for x in d["push"]],
                hovertext=d["direction"], hoverinfo="text+x"))
            fig.update_layout(title="What moves this estimate most<br><sup>orange = pushes risk up, blue = pushes risk down</sup>",
                              xaxis_title="Relative push (direction and size)")
            st.plotly_chart(style(fig, 330, False), use_container_width=True, theme=None)
 
    st.markdown("### What-if: change one thing and compare")
    w1, w2, w3, w4 = st.columns(4)
    new_term = w1.selectbox("Loan term", [36, 60], index=[36, 60].index(term), key=f"wterm{k}")
    amt_pct = w2.slider("Loan amount change", -30, 30, 0, 5, format="%d%%", key=f"wamt{k}")
    rate_chg = w3.slider("Interest rate change (points)", -5.0, 5.0, 0.0, 0.5, key=f"wrate{k}")
    inc_pct = w4.slider("Income change", -30, 30, 0, 5, format="%d%%", key=f"winc{k}")
    alt = dict(base, term=new_term, loan_amnt=max(1000, loan * (1 + amt_pct / 100)),
               int_rate=min(31.0, max(5.0, rate + rate_chg)), annual_inc=max(10000, income * (1 + inc_pct / 100)))
    alt_frame = H.build_frame(model, alt)
    alt_res = H.score(model, alt_frame)
    delta = (alt_res["probability"] - res["probability"]) * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("As entered", f"{res['probability'] * 100:.0f}%", TIER_LABEL[res["tier"]], delta_color="off")
    m2.metric("After the change", f"{alt_res['probability'] * 100:.0f}%", f"{delta:+.1f} points", delta_color="inverse")
    m3.metric("Monthly instalment", f"${alt_frame.iloc[0]['installment']:,.0f}",
              f"{alt_frame.iloc[0]['installment'] - frame.iloc[0]['installment']:+,.0f} vs entered", delta_color="off")
    if new_term != term and new_term < term:
        st.markdown('<div class="warn">A shorter term lowers the estimated risk but raises the monthly payment. Whether the applicant can afford it is a separate question this model does not answer.</div>', unsafe_allow_html=True)
    st.caption("What-if results show how the model responds to changes. They are not offers, advice, or predictions about what an applicant will actually be offered.")
 
# ------------------------------------------------------------------ 3. drivers
elif page.startswith("3"):
    st.title("What drives the model's predictions?")
    imp = pd.DataFrame(R["importance"]).head(10)
    imp["label"] = imp["feature"].map(H.PLAIN)
    imp = imp.iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["label"], orientation="h", marker_color=H.BLUE,
                           text=[f"{v:.2f}" for v in imp["mean_abs_shap"]], textposition="outside"))
    fig.update_layout(title="Ten biggest influences on predictions (average size of influence)", xaxis_title="Average influence on the risk estimate")
    fig.update_xaxes(range=[0, 0.55])
    st.plotly_chart(style(fig, 440, False), use_container_width=True, theme=None)
    a, b = st.columns(2)
    with a:
        st.subheader("Raises estimated risk")
        st.markdown("- Higher interest rate\n- Longer loan term (60 months)\n- Higher debt compared with income\n- Larger loan compared with income\n- More of the credit limit already used\n- Several recent credit enquiries\n- Renting rather than owning or having a mortgage")
    with b:
        st.subheader("Lowers estimated risk")
        st.markdown("- Higher income\n- Lower interest rate\n- Shorter loan term (36 months)\n- Less of the credit limit in use\n- Longer credit history")
    st.markdown('<div class="warn"><b>Read with care.</b> These are patterns the model found, not causes. The interest rate and credit grade are set by the lender after its own assessment, '
                f'so they carry earlier lending judgement. With both removed the model still ranks borrowers (AUC {R["ablation_auc"]["without_grade_and_interest_rate"]:.3f} versus {T["roc_auc"]:.3f}).</div>', unsafe_allow_html=True)
    img = H.DATA / "fig2_shap_summary.png"
    if img.exists():
        with st.expander("Technical view: SHAP summary plot"):
            st.image(str(img))
            st.caption("Each dot is a loan. Right of centre = pushes risk up; left = pushes risk down. Red = high value of the feature, blue = low.")
    st.subheader("What-if sensitivity (average change in estimated default risk)")
    sens = pd.DataFrame(R["sensitivity"])
    sens["Change in estimated risk (points)"] = sens["change_pp"].round(1)
    st.dataframe(sens.rename(columns={"scenario": "Scenario"})[["Scenario", "Change in estimated risk (points)"]], hide_index=True, use_container_width=True)
 
# ------------------------------------------------------------------ 4. ethical compliance
elif page.startswith("4"):
    st.title("Ethical compliance: is the model treating groups comparably?")
    st.markdown('<div class="note">The data has no gender, age or race fields, so fairness is checked on <b>indirect groups</b>: income band, housing status, time in job and region. '
                'A flagged applicant goes to review or possible decline; everyone else counts as approved. Limits come from the Module 2 plan.</div>', unsafe_allow_html=True)
    S, meta = scored(), H.load_meta()
    c1, c2 = st.columns([1, 2])
    attr = c1.selectbox("Compare groups by", list(H.GROUP_LABELS), format_func=H.GROUP_LABELS.get)
    mode = c2.radio("Setting", ["Project setting (flags the riskiest ~21% of applicants)", "Try a different share flagged"], horizontal=True)
    if mode.startswith("Project"):
        thr = meta["cut_high"]
    else:
        share = c2.slider("Share of applicants flagged for review", 10, 50, 21, 1, format="%d%%") / 100
        thr = float(np.quantile(S["raw"], 1 - share))
    tab, sm = H.fairness_view(S, attr, thr)
 
    k = st.columns(3)
    ok = lambda b: "✅ Within limit" if b else "⚠️ Outside limit"  # noqa: E731
    k[0].metric("Approval ratio (lowest ÷ highest group)", f"{sm['approval_ratio']:.2f}")
    k[0].markdown(f"**{ok(sm['approval_ok'])}** (needs above 0.80)")
    k[1].metric("Gap in defaulters correctly flagged", f"{sm['tpr_gap_pp']:.1f} pts")
    k[1].markdown(f"**{ok(sm['tpr_ok'])}** (maximum 5 points)")
    k[2].metric("Gap in repaying borrowers wrongly flagged", f"{sm['fpr_gap_pp']:.1f} pts")
    k[2].markdown(f"**{ok(sm['fpr_ok'])}** (maximum 5 points)")
    st.caption(f"{sm['flagged_share']:.1%} of 2015 test applicants flagged at this setting.")
 
    inc = tab[tab.included]
    fig = go.Figure()
    fig.add_bar(x=inc["group"], y=inc["approval_rate"] * 100, name="Approved (not flagged)", marker_color=H.BLUE,
                text=[f"{v:.0f}%" for v in inc["approval_rate"] * 100], textposition="outside")
    fig.add_bar(x=inc["group"], y=inc["default_rate"] * 100, name="Actually defaulted", marker_color=H.GREY,
                text=[f"{v:.0f}%" for v in inc["default_rate"] * 100], textposition="outside")
    fig.update_layout(barmode="group", title="Approval and actual default rate by group", yaxis_title="%", yaxis_range=[0, 105])
    f2 = go.Figure()
    f2.add_bar(x=inc["group"], y=inc["tpr"] * 100, name="Defaulters correctly flagged", marker_color=H.BLUE,
               text=[f"{v:.0f}%" for v in inc["tpr"] * 100], textposition="outside")
    f2.add_bar(x=inc["group"], y=inc["fpr"] * 100, name="Repaying borrowers wrongly flagged", marker_color=H.ORANGE,
               text=[f"{v:.0f}%" for v in inc["fpr"] * 100], textposition="outside")
    f2.update_layout(barmode="group", title="Who gets flagged correctly and wrongly", yaxis_title="%", yaxis_range=[0, 105])
    g1, g2 = st.columns(2)
    g1.plotly_chart(style(fig), use_container_width=True)
    g2.plotly_chart(style(f2), use_container_width=True)
    ex = tab[~tab.included]
    if len(ex):
        st.caption("Groups under 200 loans are left out because their results are too uncertain: " + ", ".join(ex["group"]))
    with st.expander("Table view of this chart"):
        show = inc.assign(approved=(inc.approval_rate * 100).round(1), default_rate=(inc.default_rate * 100).round(1),
                          flagged_defaulters=(inc.tpr * 100).round(1), flagged_repayers=(inc.fpr * 100).round(1))
        st.dataframe(show[["group", "n", "approved", "default_rate", "flagged_defaulters", "flagged_repayers"]]
                     .rename(columns={"group": "Group", "n": "Loans", "approved": "Approved %", "default_rate": "Defaulted %",
                                      "flagged_defaulters": "Defaulters flagged %", "flagged_repayers": "Repayers flagged %"}), hide_index=True, use_container_width=True)
 
    st.markdown("---")
    st.subheader("Bias mitigation: before and after group-specific cut-offs")
    rows = []
    for a_, lab in H.GROUP_LABELS.items():
        for stage, key in (("Before", "fairness_before"), ("After", "fairness_after")):
            s = R[key][a_]
            rows.append({"Group type": lab, "Stage": stage, "Approval ratio": round(s["approval_ratio"], 2),
                         "Defaulters-flagged gap (pts)": round(s["tpr_gap_pp"], 1), "Wrongly-flagged gap (pts)": round(s["fpr_gap_pp"], 1),
                         "Within all limits": "✅ Yes" if s["passes"] else "⚠️ No"})
    mt = pd.DataFrame(rows)
    fb = go.Figure()
    for stage, col in (("Before", H.GREY), ("After", H.BLUE)):
        sub = mt[mt.Stage == stage]
        fb.add_bar(x=sub["Group type"], y=sub["Defaulters-flagged gap (pts)"], name=f"{stage} mitigation", marker_color=col,
                   text=sub["Defaulters-flagged gap (pts)"], textposition="outside")
    fb.add_hline(y=5, line_dash="dash", line_color=H.ORANGE, annotation_text="Limit: 5 points", annotation_position="top left", layer="below")
    fb.update_layout(barmode="group", title="Gap in defaulters correctly flagged, between groups", yaxis_title="Percentage points", yaxis_range=[0, 34])
    st.plotly_chart(style(fb, 360), use_container_width=True, theme=None)
    st.dataframe(mt, hide_index=True, use_container_width=True)
    base = R["models"][MODEL]["test"]
    cost = pd.DataFrame([{"Setting": "Single cut-off (project setting)", "Applicants flagged": f"{base['flagged_share']:.0%}", "Flags that were real defaults": f"{base['precision']:.0%}", "Correct overall": f"{base['accuracy']:.0%}"}] +
                        [{"Setting": f"Group-specific cut-offs: {H.GROUP_LABELS[a_]}", "Applicants flagged": f"{s['flagged_share']:.0%}", "Flags that were real defaults": f"{s['precision']:.0%}", "Correct overall": f"{s['accuracy']:.0%}"} for a_, s in R["fairness_after"].items()])
    st.markdown("**What it costs:** narrowing the gaps flags far more applicants and makes each flag less accurate.")
    st.dataframe(cost, hide_index=True, use_container_width=True)
    st.markdown('<div class="warn">Group-specific cut-offs also raise policy and legal questions. This dashboard reports the trade-off; it does not decide it.</div>', unsafe_allow_html=True)
 
    st.subheader("Is risk over- or under-stated for any group?")
    cal = [{"Group type": H.GROUP_LABELS[a_], "Group": r["group"], "Loans": r["n"], "Model expected": f"{r['predicted']:.1%}", "Actually defaulted": f"{r['actual']:.1%}",
            "Difference": f"{r['relative_error'] * 100:+.1f}%"} for a_, rows_ in R["group_calibration"].items() for r in rows_]
    st.dataframe(pd.DataFrame(cal), hide_index=True, use_container_width=True)
    st.caption("A positive difference means the model expects more defaults than actually happened for that group.")
 
# ------------------------------------------------------------------ 5. limits
else:
    st.title("Limits and transparency statement")
    a, b = st.columns(2)
    with a:
        st.subheader("What the model can do")
        st.markdown("- Rank loan applications from lower to higher estimated risk\n- Give an estimated chance of default, grouped into three risk groups\n- Show the main reasons behind each estimate\n- Support a credit officer's review with a consistent, documented view")
    with b:
        st.subheader("What the model cannot do")
        st.markdown("- Decide on its own: a human reviewer remains responsible for every decision\n- Reflect Bank of Kigali borrowers: it learned from US LendingClub loans\n- Learn from applicants who were never funded\n- Judge affordability, character or special circumstances\n- Guarantee fairness: gaps remain between groups, and mitigation has costs")
    st.subheader("Known limitations")
    lim = pd.DataFrame([
        ["Data come from the US, 2007-2015", "Results may not transfer to Rwandan borrowers; retraining on BK data is required before any use."],
        ["Only funded loans are observed", "Applicants who were declined are missing, which can bias the results."],
        ["Performance dipped from 2014 to 2015", f"AUC {R['models'][MODEL]['validation']['roc_auc']:.3f} to {T['roc_auc']:.3f}: the model will drift over time and needs monitoring."],
        ["Moderate accuracy", f"At the project setting {T['precision']:.0%} of flagged applications defaulted, and {T['recall']:.0%} of all defaults were flagged."],
        ["Fairness checked on indirect groups", "Gender, age and ethnicity are not in the data, so they could not be tested."],
        ["Interest rate and grade are set by the lender", "The model partly repeats an earlier lending judgement."],
    ], columns=["Limitation", "What it means"])
    st.dataframe(lim, hide_index=True, use_container_width=True)
    st.markdown('<div class="note"><b>Transparency statement.</b> This is a decision-support tool built for a course project on public US data. It has not been validated on Bank of Kigali data, '
                'has not been independently reviewed, and is not used to make credit decisions. Every figure shown comes from the saved Module 4 model and its test set (2015 loans, never used in training).</div>', unsafe_allow_html=True)
    st.caption(f"Model: {MODEL} with probability calibration. Generated {R['generated']}. Repository: github.com/Tsandae/bk-credit-risk")
