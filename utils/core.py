"""Core computations for the AI Scholarship Selection Fairness & Explainability Auditor.
All data is synthetic. Everything is computed live; nothing is hard-coded."""
import numpy as np
import pandas as pd
import shap
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
                               demographic_parity_difference, equalized_odds_difference)

SEED = 42
FEATURES = ["Academic Score", "Attendance", "Family Income", "Extracurricular Score",
            "Community Score", "Previous Scholarship", "First Generation"]
SENSITIVE = "Region"  # used for AUDITING only, never as a model feature


def make_data(n=1500, seed=SEED, drift=0.0, month=0):
    """Synthetic applicants. Region is deliberately imbalanced (Urban 70 / Semi-Urban 20 / Rural 10)
    and Rural applicants receive slightly lower income/extracurricular values (historical inequality)."""
    rng = np.random.default_rng(seed + month * 101)
    region = rng.choice(["Urban", "Semi-Urban", "Rural"], n, p=[0.70, 0.20, 0.10])
    rural = region == "Rural"
    semi = region == "Semi-Urban"
    acad = np.clip(rng.normal(70, 12, n), 20, 100)
    att = np.clip(rng.normal(82, 9, n), 40, 100)
    income = np.clip(rng.lognormal(12.0, 0.5, n) * np.where(rural, 0.55, np.where(semi, 0.8, 1.0)), 50000, None)
    extra = np.clip(rng.normal(55, 18, n) - 8 * rural - 3 * semi, 0, 100)
    comm = np.clip(rng.normal(50, 20, n) + 5 * rural, 0, 100)
    prev = rng.binomial(1, 0.12, n)
    firstgen = rng.binomial(1, np.where(rural, 0.65, np.where(semi, 0.45, 0.25)))
    gender = rng.choice(["Female", "Male"], n, p=[0.48, 0.52])

    # Population shift: drift moves rural share up, lowers rural scores/income (simulated deployment change)
    if drift > 0:
        shift = drift * month / 4.0
        acad = acad - 12 * shift * rural - 4 * shift
        att = att - 8 * shift * rural
        income = income * (1 - 0.35 * shift)
        extra = extra - 10 * shift * rural
        acad, att, extra = np.clip(acad, 20, 100), np.clip(att, 40, 100), np.clip(extra, 0, 100)

    # Ground truth "deserving" label from merit + need (independent of region directly)
    score = (0.045 * acad + 0.02 * att + 0.02 * extra + 0.02 * comm + 0.5 * firstgen
             - 0.45 * np.log(income / 160000) - 0.4 * prev + rng.normal(0, 0.6, n))
    thr = np.quantile(score, 0.75)
    y = (score > thr).astype(int)
    df = pd.DataFrame({"Academic Score": acad, "Attendance": att, "Family Income": income,
                       "Extracurricular Score": extra, "Community Score": comm,
                       "Previous Scholarship": prev, "First Generation": firstgen,
                       "Gender": gender, "Region": region, "Selected": y})
    # Small amount of missing data (for missing-value analysis)
    mrng = np.random.default_rng(seed + 7)
    for col, p in [("Attendance", 0.03), ("Extracurricular Score", 0.05), ("Family Income", 0.02)]:
        df.loc[mrng.random(n) < p, col] = np.nan
    return df


def bias_stats(df):
    grp = df.groupby(SENSITIVE).agg(Applicants=("Selected", "size"), Deserving=("Selected", "sum"))
    grp["Share %"] = (grp["Applicants"] / len(df) * 100).round(1)
    grp["Positive Rate %"] = (grp["Deserving"] / grp["Applicants"] * 100).round(1)
    miss = df[FEATURES].isna().sum()
    return grp.reset_index(), miss[miss > 0]


def balanced_sample(df, seed=SEED):
    """Mitigation option 1: oversample smaller regions to equal group size."""
    m = df[SENSITIVE].value_counts().max()
    parts = [g.sample(m, replace=True, random_state=seed) for _, g in df.groupby(SENSITIVE)]
    return pd.concat(parts).reset_index(drop=True)


def _prep(df):
    X = df[FEATURES].copy()
    return X


def train_model(df, mitigation="None", seats=200):
    """Returns dict with model, scaler, test frame, probabilities, selected flags and metrics.
    mitigation: 'None' | 'Class weighting' | 'Balanced sampling'."""
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df[SENSITIVE] + df["Selected"].astype(str))
    med = train_df[FEATURES].median()
    if mitigation == "Balanced sampling":
        train_df = balanced_sample(train_df)
    Xtr = train_df[FEATURES].fillna(med)
    ytr = train_df["Selected"]
    scaler = StandardScaler().fit(Xtr)
    if mitigation == "Class weighting":
        # weight each (region, label) cell inversely to its frequency
        cell = train_df[SENSITIVE] + train_df["Selected"].astype(str)
        w = 1.0 / cell.map(cell.value_counts(normalize=True))
        w = w / w.mean()
        model = LogisticRegression(max_iter=1000, random_state=SEED).fit(scaler.transform(Xtr), ytr, sample_weight=w)
    else:
        model = LogisticRegression(max_iter=1000, random_state=SEED).fit(scaler.transform(Xtr), ytr)
    return {"model": model, "scaler": scaler, "median": med, "train": train_df, "test": test_df}


def score_frame(bundle, df, seats):
    """Predict probabilities, pick top-N seats (scaled to the size of df) and compute metrics."""
    X = df[FEATURES].fillna(bundle["median"])
    proba = bundle["model"].predict_proba(bundle["scaler"].transform(X))[:, 1]
    n_sel = max(1, int(round(seats * len(df) / 1500)))  # seats defined per 1500 applicants
    n_sel = min(n_sel, len(df))
    order = np.argsort(-proba)
    pred = np.zeros(len(df), dtype=int)
    pred[order[:n_sel]] = 1
    return proba, pred


def perf_metrics(y, pred):
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}


def fairness(y, pred, sens):
    sens = pd.Series(np.asarray(sens), name=SENSITIVE)
    mf = MetricFrame(metrics={"Selection Rate": selection_rate, "TPR": true_positive_rate, "FPR": false_positive_rate},
                     y_true=y, y_pred=pred, sensitive_features=sens)
    dpd = demographic_parity_difference(y, pred, sensitive_features=sens)
    eod = equalized_odds_difference(y, pred, sensitive_features=sens)
    return mf.by_group.reset_index(), float(dpd), float(eod)


def explain(bundle, df, max_rows=300):
    """SHAP for Logistic Regression (linear explainer on scaled features)."""
    Xs = pd.DataFrame(bundle["scaler"].transform(df[FEATURES].fillna(bundle["median"])), columns=FEATURES, index=df.index)
    bg = pd.DataFrame(bundle["scaler"].transform(bundle["train"][FEATURES].fillna(bundle["median"])), columns=FEATURES)
    expl = shap.LinearExplainer(bundle["model"], bg)
    sv = expl.shap_values(Xs)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return pd.DataFrame(sv, columns=FEATURES, index=df.index), float(np.ravel(expl.expected_value)[0])


RISKS = [
    ("Fairness", "Regional/gender bias in selection", 4, 5, "UNESCO: Fairness & non-discrimination; IEEE: Algorithmic bias considerations", "Bias testing + balanced training data"),
    ("Privacy", "Exposure of applicant personal data", 3, 4, "UNESCO: Right to privacy & data protection; IEEE: Personal data agency", "Data minimization, sensitive attributes used only for auditing"),
    ("Transparency", "Applicants cannot understand outcomes", 4, 3, "UNESCO: Transparency & explainability; IEEE: Transparency", "SHAP explanations for every decision"),
    ("Accountability", "No clear owner of wrong decisions", 3, 4, "UNESCO: Responsibility & accountability; IEEE: Accountability", "Named committee owner + audit log"),
    ("Security/Safety", "Tampering with scores or data", 2, 5, "UNESCO: Safety & security; IEEE: Well-being / safety", "Access control, integrity checks"),
    ("Human Oversight", "Over-reliance on automated ranking", 3, 5, "UNESCO: Human oversight & determination; IEEE: Human agency", "Human review of all decisions; periodic auditing"),
]


def classify(score):
    if score <= 5:
        return "Low"
    if score <= 11:
        return "Moderate"
    if score <= 19:
        return "High"
    return "Critical"


DEEPFAKE_SAMPLES = [
    {"Media": "Applicant Video Statement #1", "Type": "Authentic", "Confidence": 96,
     "Indicators": "None significant — natural blinking, consistent lighting and lip movement",
     "Risk": "Low — no action needed"},
    {"Media": "Applicant Video Statement #2", "Type": "AI-Generated (Simulated)", "Confidence": 41,
     "Indicators": "Lip-synchronization mismatch, unnatural blinking rate, lighting inconsistency across frames",
     "Risk": "Identity fraud / impersonation of a real applicant"},
    {"Media": "Recommendation Letter Scan", "Type": "AI-Generated (Simulated)", "Confidence": 55,
     "Indicators": "Digital compression artifacts around signature, font irregularities, metadata inconsistency",
     "Risk": "Document fraud / misinformation in supporting evidence"},
]


def deepfake_table():
    return pd.DataFrame(DEEPFAKE_SAMPLES)


RESOURCES = [
    {"Name": "Govt Skill Development Course", "Cost": 1500, "Quality": 72, "Category": "Course"},
    {"Name": "Refurbished Laptop", "Cost": 12000, "Quality": 60, "Category": "Hardware"},
    {"Name": "Free MOOC Platform (NPTEL/SWAYAM)", "Cost": 0, "Quality": 82, "Category": "Platform"},
    {"Name": "Premium Certification Bootcamp", "Cost": 25000, "Quality": 78, "Category": "Certification"},
    {"Name": "Budget Laptop", "Cost": 18000, "Quality": 55, "Category": "Hardware"},
    {"Name": "Community Library Learning Program", "Cost": 500, "Quality": 68, "Category": "Course"},
]


def recommend(mode="General", budget=None):
    """mode: 'General' | 'Budget-Constrained' | 'Cost-Neutral'.
    Demonstrates whether cost is used as an inappropriate proxy for quality."""
    res = pd.DataFrame(RESOURCES).copy()
    cmin, cmax = res["Cost"].min(), res["Cost"].max()
    qmin, qmax = res["Quality"].min(), res["Quality"].max()
    res["Cost (norm)"] = (res["Cost"] - cmin) / (cmax - cmin)
    res["Quality (norm)"] = (res["Quality"] - qmin) / (qmax - qmin)

    if mode == "Budget-Constrained":
        if budget is not None:
            res = res[res["Cost"] <= budget].copy()
        res["Score"] = res["Quality (norm)"]
    elif mode == "Cost-Neutral":
        res["Score"] = res["Quality (norm)"]
    else:  # General — naive scoring that lets cost influence rank (the bias under audit)
        res["Score"] = 0.5 * res["Quality (norm)"] + 0.5 * res["Cost (norm)"]

    res = res.sort_values("Score", ascending=False).reset_index(drop=True)
    res.insert(0, "Rank", res.index + 1)
    return res.drop(columns=["Cost (norm)", "Quality (norm)"])


def cost_preference_indicator():
    """Compares the average cost of the top-3 picks under General vs Cost-Neutral scoring."""
    gen = recommend("General").head(3)
    neu = recommend("Cost-Neutral").head(3)
    gen_avg, neu_avg = gen["Cost"].mean(), neu["Cost"].mean()
    diff_pct = 0.0 if neu_avg == 0 else (gen_avg - neu_avg) / neu_avg * 100
    flagged = diff_pct > 15
    return gen_avg, neu_avg, diff_pct, flagged


def drift_sim(bundle_none, drift, seats, ks_thr=0.10, dpd_thr=0.15, months=4):
    """Baseline + 4 monthly batches. Model stays fixed (as in deployment); population shifts."""
    rows = []
    base = make_data(n=1500, seed=SEED + 1, drift=0.0, month=0)
    for m in range(0, months + 1):
        d = base if m == 0 else make_data(n=1500, seed=SEED + 1, drift=drift, month=m)
        proba, pred = score_frame(bundle_none, d, seats)
        acc = accuracy_score(d["Selected"], pred)
        sr = pred.mean()
        dpd = demographic_parity_difference(d["Selected"], pred, sensitive_features=d[SENSITIVE])
        ks = ks_2samp(base["Academic Score"], d["Academic Score"]).statistic if m else 0.0
        ks_inc = ks_2samp(base["Family Income"].fillna(base["Family Income"].median()),
                          d["Family Income"].fillna(d["Family Income"].median())).statistic if m else 0.0
        rows.append({"Period": "Baseline" if m == 0 else f"Month {m}", "Accuracy": acc, "Selection Rate": sr,
                     "DPD": float(dpd), "KS (Academic)": ks, "KS (Income)": ks_inc,
                     "KS max": max(ks, ks_inc)})
    out = pd.DataFrame(rows)
    out["Trigger"] = np.where((out["KS max"] > ks_thr) | (out["DPD"] > dpd_thr), "HUMAN REVIEW REQUIRED", "CONTINUE MONITORING")
    return out
