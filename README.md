# FedHealth Analytics — Medical Risk Predictor

A cloud-deployable, privacy-preserving healthcare risk prediction web app powered by your trained ensemble ML models.

---

## How It Works

```
User fills form  →  POST /api/predict  →  In-memory inference  →  Results returned
                           ↑
                    No data stored.
                    No logs written.
                    No third-party calls.
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Export your trained models

Add these lines to your notebook **after** `pd.get_dummies()` and **before** `scaler.fit_transform()`:

```python
feature_columns = list(X.columns)   # capture column order HERE
```

Then after training, run:

```python
import os, json, joblib
os.makedirs("models", exist_ok=True)

joblib.dump(reg_models["annual_cost"],   "models/annual_cost_model.joblib")
joblib.dump(reg_models["claims_paid"],   "models/claims_paid_model.joblib")
joblib.dump(reg_models["risk_score"],    "models/risk_score_model.joblib")
joblib.dump(clf_models["high_risk_sgd"], "models/high_risk_sgd_model.joblib")
joblib.dump(clf_models["high_risk_rf"],  "models/high_risk_rf_model.joblib")
joblib.dump(scaler,                      "models/scaler.joblib")
with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)
```

Or just run `save_models.py` in your notebook environment.

### 3. Run locally

```bash
python app.py
# → http://127.0.0.1:5000
```

### 4. Deploy to cloud

**Heroku / Render / Railway:**
```bash
git init && git add . && git commit -m "init"
# Push to your platform — Procfile handles gunicorn automatically
```

**Environment variables (optional):**
- `PORT` — server port (default: 5000)
- `FLASK_DEBUG` — set to `true` only in development

---

## Required Model Files

Place all of these in the `models/` directory before deploying:

| File | Source |
|------|--------|
| `annual_cost_model.joblib` | `reg_models["annual_cost"]` |
| `claims_paid_model.joblib` | `reg_models["claims_paid"]` |
| `risk_score_model.joblib`  | `reg_models["risk_score"]`  |
| `high_risk_sgd_model.joblib` | `clf_models["high_risk_sgd"]` |
| `high_risk_rf_model.joblib`  | `clf_models["high_risk_rf"]`  |
| `scaler.joblib` | your `StandardScaler` instance |
| `feature_columns.json` | `list(X.columns)` after `get_dummies` |

---

## Privacy Design

- Input data is processed in-memory only
- No request logging (`werkzeug` access logs suppressed)
- No database, no session storage, no third-party calls
- Predictions are returned to the user and immediately discarded server-side
- Suitable for deployment behind a secure HTTPS proxy (recommended for clinical use)
