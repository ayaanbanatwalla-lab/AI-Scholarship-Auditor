# AI Scholarship Selection: Fairness & Explainability Auditor

**🔗 Live demo:** [ai-scholarship-auditor.streamlit.app](https://ai-scholarship-auditor.streamlit.app)
*(the app sleeps after inactivity — first load may take ~30–60 seconds to wake up)*

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

## Limitations
- Synthetic data with a deliberately controlled imbalance; not representative of real populations.
- Only one sensitive attribute (Region) is audited; Gender is present but not audited.
- Mitigation reduces but does not remove bias and may lower accuracy.
- Fairness metrics alone cannot prove a system is ethical.
- Drift is simulated, not from a real deployment; thresholds are illustrative.
- Risk scores are self-assessed estimates, not a legal compliance check.
- Exp 3 (deepfake) uses fixed illustrative sample data, not a real detector — a conceptual demonstration only.
- Exp 4 (cost-as-a-proxy) uses a small fixed resource list and a deterministic scoring rule, not a live recommendation engine.
