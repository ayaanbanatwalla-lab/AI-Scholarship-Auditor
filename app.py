import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import core

st.set_page_config(page_title="AI Scholarship Auditor", page_icon="🎓", layout="wide")

NAVY, TEAL, AMBER, RED = "#1f2a44", "#14a39a", "#f0a202", "#d64545"
st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
.hero {{ background: linear-gradient(120deg, {NAVY}, #2f4b7c); color: white; padding: 22px 28px; border-radius: 14px; }}
.hero h1 {{ margin: 0; font-size: 1.9rem; }} .hero p {{ margin: 4px 0 0; opacity: .85; }}
.flow {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0 6px; align-items: center; }}
.step {{ background: #eef2f7; color: {NAVY}; padding: 5px 12px; border-radius: 20px; font-size: .78rem; font-weight: 600; }}
.arrow {{ color: #8a94a6; font-size: .8rem; }}
.tag {{ display: inline-block; background: #e6f6f4; color: #0b7a72; padding: 3px 10px; border-radius: 6px; font-size: .75rem; font-weight: 700; margin-bottom: 6px; }}
.banner {{ padding: 16px 20px; border-radius: 10px; font-weight: 700; font-size: 1.1rem; margin: 10px 0; }}
.warn {{ background: #fdecec; color: {RED}; border: 1px solid {RED}; }}
.ok {{ background: #e6f6f4; color: #0b7a72; border: 1px solid {TEAL}; }}
.card {{ background: #f7f9fc; border: 1px solid #e3e8ef; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; color: {NAVY} !important; }}
.card b {{ color: {NAVY} !important; }}
[data-testid="stMetric"] {{ background: #f7f9fc; border: 1px solid #e3e8ef; padding: 10px 14px; border-radius: 10px; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{ color: {NAVY} !important; opacity: .75; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{ color: {NAVY} !important; }}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {{ color: inherit !important; }}
</style>""", unsafe_allow_html=True)


def tag(t):
    st.markdown(f'<span class="tag">{t}</span>', unsafe_allow_html=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controls")
    seats = st.slider("Scholarship seats (per 1,500 applicants)", 50, 500, 200, 10, key="seats")
    mitigation = st.selectbox("Bias mitigation", ["None", "Class weighting", "Balanced sampling"], key="mit")
    if st.button("🔄 Reset Simulation", width='stretch'):
        st.cache_data.clear()
        st.cache_resource.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.caption("Seed fixed (42) for consistent demos.")
    st.info("Synthetic data only. Educational simulation.")

# ---------------- Cached computation ----------------
@st.cache_data(show_spinner=False)
def get_data():
    return core.make_data()


@st.cache_resource(show_spinner=False)
def get_bundle(mit):
    return core.train_model(get_data(), mit)


df = get_data()
bundle = get_bundle(mitigation)
baseline_bundle = get_bundle("None")
test = bundle["test"]
proba, pred = core.score_frame(bundle, test, seats)
y = test["Selected"].values
perf = core.perf_metrics(y, pred)
fair_tbl, dpd, eod = core.fairness(y, pred, test[core.SENSITIVE].values)
# unmitigated reference for before/after
b_proba, b_pred = core.score_frame(baseline_bundle, baseline_bundle["test"], seats)
b_perf = core.perf_metrics(baseline_bundle["test"]["Selected"].values, b_pred)
_, b_dpd, b_eod = core.fairness(baseline_bundle["test"]["Selected"].values, b_pred, baseline_bundle["test"][core.SENSITIVE].values)

# ---------------- Header ----------------
st.markdown('<div class="hero"><h1>AI Scholarship Selection</h1>'
            '<p>Fairness &amp; Explainability Auditor</p>'
            '<p style="font-size:.85rem;opacity:.75;margin-top:6px;">An Interactive Responsible AI Auditing Mini-Project</p></div>',
            unsafe_allow_html=True)
steps = ["DATASET", "BIAS", "MODEL", "FAIRNESS", "SHAP", "AIA", "DRIFT", "HUMAN REVIEW"]
st.markdown('<div class="flow">' + '<span class="arrow">→</span>'.join(f'<span class="step">{s}</span>' for s in steps) + '</div>', unsafe_allow_html=True)

exp_labels = ["Exp 1 — AI Ethics", "Exp 2 — Dataset Bias", "Exp 3 — Deepfake Risk", "Exp 4 — Cost-as-Proxy Bias",
              "Exp 5 — Fairness & Explainability", "Exp 6 — Algorithmic Impact Assessment", "Exp 7 — System Degradation"]
st.markdown('<p style="font-size:.78rem;color:#8a94a6;margin:8px 0 2px;font-weight:700;">7 EXPERIMENTS DEMONSTRATED</p>'
            + '<div class="flow">' + "".join(f'<span class="tag" style="margin-right:4px;">{e}</span>' for e in exp_labels) + '</div>',
            unsafe_allow_html=True)

st.caption("This application uses synthetic data and is intended only for demonstrating Responsible AI auditing concepts. "
           "It must not be used for actual scholarship or other high-impact decisions.")

tabs = st.tabs(["1 · Dataset & Bias", "2 · Model & Fairness", "3 · Explainability", "4 · Ethical Risk", "5 · Drift & Human Review"])

# ---------------- Tab 1 ----------------
with tabs[0]:
    tag("Experiment 2 — Dataset Bias")
    grp, miss = core.bias_stats(df)
    c = st.columns(4)
    c[0].metric("Applicants", f"{len(df):,}")
    c[1].metric("Largest group share", f"{grp['Share %'].max():.0f}%")
    c[2].metric("Smallest group share", f"{grp['Share %'].min():.0f}%")
    c[3].metric("Deserving (label=1)", f"{df['Selected'].mean()*100:.0f}%")
    l, r = st.columns(2)
    with l:
        fig = px.bar(grp, x="Region", y="Applicants", text="Share %", color="Region",
                     color_discrete_sequence=[NAVY, TEAL, AMBER], title="Representation by Region")
        fig.update_traces(texttemplate="%{text}%")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width='stretch')
    with r:
        fig = px.bar(grp, x="Region", y="Positive Rate %", color="Region", text="Positive Rate %",
                     color_discrete_sequence=[NAVY, TEAL, AMBER], title="Scholarship-worthy outcome rate (%) by Region")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width='stretch')
    st.warning("**Potential representation imbalance detected.** Rural applicants are only about 10% of the data and have a different "
               "outcome rate. This is a *potential* bias risk to investigate, not proof that the dataset is discriminatory.")
    with st.expander("Missing values"):
        st.dataframe(miss.rename("Missing count").to_frame(), width='stretch')
    with st.expander("Mitigation: BEFORE → AFTER (use the sidebar option)", expanded=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Mitigation selected", mitigation)
        m2.metric("Demographic Parity Diff.", f"{dpd:.3f}", f"{dpd - b_dpd:+.3f} vs no mitigation", delta_color="inverse")
        m3.metric("Accuracy", f"{perf['Accuracy']:.3f}", f"{perf['Accuracy'] - b_perf['Accuracy']:+.3f} vs no mitigation")
        st.caption("Mitigation may reduce a gap but does not remove bias, and can cost accuracy. Pick an option in the sidebar to compare.")

# ---------------- Tab 2 ----------------
with tabs[1]:
    tag("Experiment 5 — Fairness & Explainability")
    st.markdown(f"Logistic Regression ranks applicants by selection probability; the top seats are selected. "
                f"On the {len(test)}-applicant test cohort, **{int(pred.sum())} seats** are awarded. "
                f"Region is used for auditing only, not as a feature.")
    c = st.columns(4)
    for col, (k, v) in zip(c, perf.items()):
        col.metric(k, f"{v:.3f}")
    l, r = st.columns([1, 2])
    with l:
        cm = core.confusion_matrix(y, pred)
        fig = px.imshow(cm, text_auto=True, x=["Not selected", "Selected"], y=["Not deserving", "Deserving"],
                        color_continuous_scale="Blues", title="Confusion matrix")
        fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width='stretch')
    with r:
        long = fair_tbl.melt(id_vars=core.SENSITIVE, var_name="Metric", value_name="Value")
        fig = px.bar(long, x=core.SENSITIVE, y="Value", color="Metric", barmode="group",
                     color_discrete_sequence=[NAVY, TEAL, AMBER], title="Group-wise Selection Rate, TPR, FPR")
        fig.update_layout(height=320, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width='stretch')
    f = st.columns(2)
    f[0].metric("Demographic Parity Difference", f"{dpd:.3f}", "Higher = larger selection-rate gap", delta_color="off")
    f[1].metric("Equalized Odds Difference", f"{eod:.3f}", "Higher = larger error-rate gap", delta_color="off")
    st.info("Good performance does not imply fairness: accuracy can be high while groups are treated very differently. "
            "A fairness metric alone also cannot prove a system is ethical.")

# ---------------- Tab 3 ----------------
with tabs[2]:
    tag("Experiment 5 — SHAP Explainability")
    sv, base_val = core.explain(bundle, test)
    l, r = st.columns(2)
    with l:
        imp = sv.abs().mean().sort_values().reset_index()
        imp.columns = ["Feature", "Mean |SHAP|"]
        fig = px.bar(imp, x="Mean |SHAP|", y="Feature", orientation="h", color_discrete_sequence=[TEAL],
                     title="A. Global feature importance")
        fig.update_layout(height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width='stretch')
    with r:
        st.markdown("**B. Local explanation: pick one applicant**")
        ids = list(test.index)
        top_ids = list(test.index[np.argsort(-proba)[:5]])
        pick = st.selectbox("Applicant ID", ids, index=ids.index(top_ids[0]), key="applicant")
        pos = ids.index(pick)
        p = float(proba[pos])
        st.metric("Selection probability", f"{p:.1%}", "SELECTED" if pred[pos] else "NOT selected", delta_color="off")
        row = sv.loc[pick].sort_values(key=abs, ascending=False).head(6).sort_values()
        fig = go.Figure(go.Bar(x=row.values, y=row.index, orientation="h",
                               marker_color=[TEAL if v > 0 else RED for v in row.values]))
        fig.update_layout(height=280, margin=dict(t=30, b=10), xaxis_title="Effect on prediction (log-odds)",
                          title="Top features (green raises, red lowers)")
        st.plotly_chart(fig, width='stretch')
    st.caption("SHAP shows which features influenced the model's output. It explains the model, not whether the outcome is fair or correct.")

# ---------------- Tab 4 ----------------
with tabs[3]:
    tag("Experiment 1 & 6 — AI Ethics & Algorithmic Impact Assessment")
    st.markdown("**Use case:** AI-assisted scholarship applicant ranking.  \n"
                "**Stakeholders:** Students · University · Scholarship committee · Administrators")
    rows = []
    st.markdown("**Rate each risk (1–5).** Risk score = Likelihood × Impact")
    hdr = st.columns([1.3, 2.4, 1.5, 1.5])
    hdr[0].markdown("**Risk**"); hdr[1].markdown("**Description**"); hdr[2].markdown("**Likelihood**"); hdr[3].markdown("**Impact**")
    for i, (name, desc, L0, I0, map_, mit_) in enumerate(core.RISKS):
        c = st.columns([1.3, 2.4, 1.5, 1.5])
        c[0].write(name); c[1].caption(desc)
        L = c[2].slider("L", 1, 5, L0, key=f"L{i}", label_visibility="collapsed")
        I = c[3].slider("I", 1, 5, I0, key=f"I{i}", label_visibility="collapsed")
        rows.append({"Risk": name, "L": L, "I": I, "Score": L * I, "Level": core.classify(L * I),
                     "UNESCO / IEEE consideration": map_, "Mitigation": mit_})
    risk = pd.DataFrame(rows)
    l, r = st.columns([1, 1.4])
    with l:
        z = np.array([[i * j for i in range(1, 6)] for j in range(1, 6)])
        colors = {"Low": "#7bc47f", "Moderate": "#f7d154", "High": "#f28c38", "Critical": "#d64545"}
        levels = ["Low", "Moderate", "High", "Critical"]
        zi = np.array([[levels.index(core.classify(v)) for v in rowv] for rowv in z])
        txt = [["" for _ in range(5)] for _ in range(5)]
        for _, rr in risk.iterrows():
            t = txt[rr["I"] - 1][rr["L"] - 1]
            txt[rr["I"] - 1][rr["L"] - 1] = (t + "<br>" if t else "") + rr["Risk"][:4]
        fig = go.Figure(go.Heatmap(z=zi, x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5], text=txt, texttemplate="%{text}",
                                   colorscale=[[0, colors["Low"]], [.33, colors["Moderate"]], [.66, colors["High"]], [1, colors["Critical"]]],
                                   showscale=False, zmin=0, zmax=3))
        fig.update_layout(height=340, title="Risk matrix", xaxis_title="Likelihood", yaxis_title="Impact", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width='stretch')
    with r:
        st.dataframe(risk[["Risk", "L", "I", "Score", "Level"]], width='stretch', hide_index=True)
    st.markdown("**Risk → UNESCO/IEEE consideration → Mitigation**")
    st.dataframe(risk[["Risk", "UNESCO / IEEE consideration", "Mitigation"]], width='stretch', hide_index=True)
    worst = risk.loc[risk["Score"].idxmax()]
    overall = core.classify(int(risk["Score"].max()))
    st.metric("Overall risk level (highest risk)", overall, f"{worst['Risk']} ({worst['Score']})", delta_color="off")

# ---------------- Tab 5 ----------------
with tabs[4]:
    tag("Experiment 7 — Autonomous Selection & System Degradation")
    c = st.columns([2, 1, 1])
    drift = c[0].slider("Drift intensity", 0.0, 1.0, 0.0, 0.05, key="drift")
    ks_thr = c[1].number_input("KS threshold", 0.01, 0.5, 0.10, 0.01, key="ksthr")
    dpd_thr = c[2].number_input("DPD threshold", 0.01, 0.5, 0.15, 0.01, key="dpdthr")
    sim = core.drift_sim(baseline_bundle, drift, seats, ks_thr, dpd_thr)
    fig = go.Figure()
    for col, colr in [("Accuracy", NAVY), ("Selection Rate", TEAL), ("DPD", RED)]:
        fig.add_trace(go.Scatter(x=sim["Period"], y=sim[col], mode="lines+markers", name=col, line=dict(color=colr, width=3)))
    fig.add_hline(y=dpd_thr, line_dash="dot", line_color=RED, annotation_text="DPD threshold")
    fig.update_layout(height=340, title="Metrics over time (model fixed, population shifts)", margin=dict(t=50, b=10),
                      yaxis_range=[0, 1])
    st.plotly_chart(fig, width='stretch')
    show = sim[["Period", "Accuracy", "Selection Rate", "DPD", "KS (Academic)", "KS (Income)", "Trigger"]].copy()
    st.dataframe(show.round(3), width='stretch', hide_index=True)
    last = sim.iloc[-1]
    review = bool((sim["Trigger"] == "HUMAN REVIEW REQUIRED").any())
    if review:
        first = sim[sim["Trigger"] == "HUMAN REVIEW REQUIRED"].iloc[0]["Period"]
        st.markdown(f'<div class="banner warn">⚠ HUMAN REVIEW REQUIRED (first triggered at {first})</div>', unsafe_allow_html=True)
        st.write("Drift or fairness gap exceeded threshold. **No applicant is automatically approved or rejected.** "
                 "A human committee must review the model and pending decisions.")
    else:
        st.markdown('<div class="banner ok">✔ CONTINUE MONITORING</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><b>Recommended operating mode: Human-in-the-loop.</b> '
                'The AI ranks applicants; people make and own the final decisions.</div>', unsafe_allow_html=True)
    st.caption("Trigger rule: max KS statistic (Academic Score, Family Income) > KS threshold OR DPD > DPD threshold.")

# ---------------- Additional Experiment Demonstrations ----------------
st.divider()
st.markdown("### Additional Experiment Demonstrations")
st.caption("Small supporting modules — not full production systems. Each is a compact educational simulation.")

with st.expander("🎭 Experiment 3 — Deepfake Risk Demonstration", expanded=False):
    tag("Experiment 3 — Deepfake Risk Demonstration")
    st.caption("Conceptual educational demonstration, not a production deepfake detector.")
    dft = core.deepfake_table()
    for _, row in dft.iterrows():
        badge = "✅" if row["Type"] == "Authentic" else "⚠️"
        c1, c2 = st.columns([2, 1])
        c1.markdown(f"**{badge} {row['Media']}** — *{row['Type']}*  \n"
                    f"**Observed indicators:** {row['Indicators']}  \n**Potential risk:** {row['Risk']}")
        c2.metric("Confidence", f"{row['Confidence']}%")
        st.markdown("---")
    st.info("Deepfake-style manipulation of applicant media or documents creates risks of identity fraud, "
            "impersonation, privacy violation, and misinformation in the selection process. "
            "This is a conceptual demonstration, not a production deepfake detector.")

with st.expander("💰 Experiment 4 — Cost-as-a-Proxy Bias", expanded=False):
    tag("Experiment 4 — Cost-as-a-Proxy Bias")
    st.caption("Does the recommender treat higher cost as a proxy for higher quality?")
    mode = st.radio("Recommendation mode", ["General", "Cost-Neutral", "Budget-Constrained"],
                    horizontal=True, key="rec_mode")
    budget = None
    if mode == "Budget-Constrained":
        budget = st.slider("Budget cap (₹)", 0, 25000, 15000, 500, key="rec_budget")
    rec = core.recommend(mode, budget)
    st.dataframe(rec[["Rank", "Name", "Cost", "Quality", "Category", "Score"]].round(3),
                width='stretch', hide_index=True)

    gen_avg, neu_avg, diff_pct, flagged = core.cost_preference_indicator()
    c1, c2, c3 = st.columns(3)
    c1.metric("General top-3 avg cost", f"₹{gen_avg:,.0f}")
    c2.metric("Neutral top-3 avg cost", f"₹{neu_avg:,.0f}")
    c3.metric("Cost preference gap", f"{diff_pct:+.0f}%", delta_color="inverse")
    if flagged:
        st.warning("⚠ **Cost Preference Indicator: FLAGGED.** The General mode's top picks cost "
                   f"{diff_pct:.0f}% more on average than the Cost-Neutral picks, without a proportional "
                   "quality gain — a sign that cost is being used as an inappropriate proxy for quality.")
    else:
        st.success("✔ No significant cost preference detected in this mode.")

# ---------------- Final summary ----------------
st.divider()
st.subheader("FINAL AUDIT SUMMARY")
grp, _ = core.bias_stats(df)
sv, _ = core.explain(bundle, test)
top3 = ", ".join(sv.abs().mean().sort_values(ascending=False).head(3).index)
risk_scores = [st.session_state.get(f"L{i}", r[2]) * st.session_state.get(f"I{i}", r[3]) for i, r in enumerate(core.RISKS)]
overall = core.classify(max(risk_scores))
sim_now = core.drift_sim(baseline_bundle, st.session_state.get("drift", 0.0), seats,
                         st.session_state.get("ksthr", 0.10), st.session_state.get("dpdthr", 0.15))
need_review = bool((sim_now["Trigger"] == "HUMAN REVIEW REQUIRED").any())
rows = [
    ("Dataset", f"Potential representation imbalance detected (smallest group {grp['Share %'].min():.0f}% of applicants)"),
    ("Model", f"Acc {perf['Accuracy']:.2f} · Prec {perf['Precision']:.2f} · Rec {perf['Recall']:.2f} · F1 {perf['F1']:.2f}"),
    ("Fairness", f"DPD {dpd:.3f} · EOD {eod:.3f}"),
    ("Explainability", f"Top influencing features: {top3}"),
    ("Ethical Risk", f"Overall risk level: {overall}"),
    ("Drift", "Drift/fairness threshold exceeded" if need_review else "Within thresholds"),
    ("Human Oversight", "HUMAN REVIEW REQUIRED" if need_review else "Monitoring continues (human-in-the-loop)"),
    ("Exp 3 — Deepfake Risk", "Demonstration completed — 1 authentic + 2 simulated AI-generated samples assessed"),
    ("Exp 4 — Cost-as-a-Proxy Bias", f"Audit completed — cost preference gap {core.cost_preference_indicator()[2]:+.0f}% "
                                      f"({'flagged' if core.cost_preference_indicator()[3] else 'not flagged'})"),
]
st.table(pd.DataFrame(rows, columns=["Area", "Finding"]).set_index("Area"))
st.success("AI-assisted auditing — final decisions require human oversight.")
st.caption("Experiments integrated: Exp 1 (ethics, oversight) · Exp 2 (dataset bias) · Exp 3 (deepfake risk) · "
           "Exp 4 (cost-as-proxy bias) · Exp 5 (fairness, SHAP) · Exp 6 (AIA) · Exp 7 (drift, human review). "
           "Mitigation does not fully remove bias, and no single metric proves ethical behaviour.")

with st.expander("Experiment Mapping (academic traceability)"):
    mapping = pd.DataFrame([
        ("Exp 1", "AI Ethics", "Ethical risk assessment + human oversight"),
        ("Exp 2", "Dataset Bias", "Representation/class imbalance + mitigation"),
        ("Exp 3", "Deepfake", "Detection concepts + vulnerability + ethical/privacy risks"),
        ("Exp 4", "Cost-as-a-Proxy", "Recommendation cost-preference audit"),
        ("Exp 5", "Fairness & Explainability", "Fairness metrics (DPD/EOD) + SHAP"),
        ("Exp 6", "AIA", "Risk assessment + UNESCO/IEEE mapping + mitigation"),
        ("Exp 7", "System Degradation", "Drift (KS test) + fairness-over-time + human review"),
    ], columns=["Experiment", "Concept", "Where demonstrated"])
    st.dataframe(mapping, width='stretch', hide_index=True)
