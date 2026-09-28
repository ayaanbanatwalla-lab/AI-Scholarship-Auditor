# AI Scholarship Selection: Fairness & Explainability Auditor

> This application uses synthetic data and is intended only for demonstrating Responsible AI auditing concepts. It must not be used for actual scholarship or other high-impact decisions.

## Project overview
One AI scholarship-selection system audited end to end:
`DATASET → BIAS → MODEL → FAIRNESS → SHAP → AIA → DRIFT → HUMAN REVIEW`
Built as a single Streamlit app with five main tabs, two compact supporting-experiment cards, and a final audit summary. All numbers are computed live from synthetic data (seed 42). All 7 lab experiments are represented.

## Objectives
- Show how dataset imbalance can become a fairness risk.
- Show that model performance and fairness are separate.
- Explain individual and global model behaviour with SHAP.
- Assess ethical risk with Likelihood × Impact.
- Demonstrate deepfake-related risk concepts in applicant media/documents.
- Audit whether cost is used as an inappropriate proxy for quality in recommendations.
- Simulate drift and trigger human review.

## Experiments demonstrated
| Exp | Concept | Where |
|---|---|---|
| 1 | AI Ethics — ethical evaluation, human oversight | Ethical Risk tab, Final Summary |
| 2 | Dataset bias, imbalance, missing values, mitigation | Dataset & Bias tab |
| 3 | Deepfake risk — detection concepts, indicators, ethical/privacy risk | Additional Experiment Demonstrations |
| 4 | Cost-as-a-proxy bias — recommendation cost-preference audit | Additional Experiment Demonstrations |
| 5 | Fairlearn metrics (DPD, EOD), SHAP | Model & Fairness, Explainability |
| 6 | Algorithmic Impact Assessment (UNESCO/IEEE mapping) | Ethical Risk tab |
| 7 | Drift (KS test), fairness over time, human review | Drift & Human Review tab |

Experiments 1, 2, 5, 6, 7 form the main scholarship-auditing pipeline (5 tabs). Experiments 3 and 4 are intentionally small supporting demonstrations in expandable cards — not full production systems.

## Technology stack
Python, Streamlit, pandas, numpy, scikit-learn, Fairlearn, SHAP, SciPy, Plotly.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `README.md` and `utils/` to a public GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, click **New app**.
3. Select the repo, branch `main`, main file `app.py`, then **Deploy**.

## 3–5 minute demo flow
| Time | Action |
|---|---|
| 0:00–0:20 | Introduce the scenario, workflow bar, and 7-experiment badge row |
| 0:20–0:50 | Tab 1: show Region imbalance (Exp 2); try a mitigation in the sidebar |
| 0:50–1:35 | Tab 2: seats slider, performance vs fairness DPD/EOD (Exp 5) |
| 1:35–2:00 | Tab 3: global SHAP, then pick one applicant (Exp 5) |
| 2:00–2:30 | Tab 4: change Likelihood/Impact, show matrix and mapping (Exp 1 + 6) |
| 2:30–2:50 | Expand Exp 3 — Deepfake Risk card |
| 2:50–3:10 | Expand Exp 4 — Cost-as-a-Proxy Bias card; switch modes |
| 3:10–4:00 | Tab 5: raise Drift Intensity, watch metrics move (Exp 7) |
| 4:00–4:30 | Show HUMAN REVIEW REQUIRED banner |
| 4:30–5:00 | Final Audit Summary; explain how all 7 experiments were integrated |

Tip: drift intensity around 0.6–1.0 reliably triggers review. Use **Reset Simulation** between runs.

## Limitations
- Synthetic data with a deliberately controlled imbalance; not representative of real populations.
- Only one sensitive attribute (Region) is audited; Gender is present but not audited.
- Mitigation reduces but does not remove bias and may lower accuracy.
- Fairness metrics alone cannot prove a system is ethical.
- Drift is simulated, not from a real deployment; thresholds are illustrative.
- Risk scores are self-assessed estimates, not a legal compliance check.
- Exp 3 (deepfake) uses fixed illustrative sample data, not a real detector — a conceptual demonstration only.
- Exp 4 (cost-as-a-proxy) uses a small fixed resource list and a deterministic scoring rule, not a live recommendation engine.
