"""
save_models.py
==============
Run this script AFTER training your models in the notebook.
It saves all required artifacts into the models/ directory so
the Flask app can load them.

Usage:
    Copy-paste the training section from your notebook, then
    append this script and run it.
"""

import os
import json
import joblib

# ── Re-run your full training notebook first, then execute below ──────────

os.makedirs("models", exist_ok=True)

# Regression models
joblib.dump(reg_models["annual_cost"], "models/annual_cost_model.joblib")
joblib.dump(reg_models["claims_paid"], "models/claims_paid_model.joblib")
joblib.dump(reg_models["risk_score"],  "models/risk_score_model.joblib")

# Classification models
joblib.dump(clf_models["high_risk_sgd"], "models/high_risk_sgd_model.joblib")
joblib.dump(clf_models["high_risk_rf"],  "models/high_risk_rf_model.joblib")

# Scaler (required — X must be scaled the same way at inference time)
joblib.dump(scaler, "models/scaler.joblib")

# Feature columns (the column list of X after get_dummies, before scaling)
# NOTE: X must still be a DataFrame at this point (before scaler.fit_transform
#       turned it into an ndarray). If you already converted it, reload like:
#
#   X_cols = pd.DataFrame(scaler.inverse_transform(X_train),
#                         columns=feature_columns).columns.tolist()
#
# The simplest fix: capture columns BEFORE transform:
#
#   feature_columns = list(X.columns)  # <- add this line right after get_dummies

with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

print("✅  All model artifacts saved to models/")
print("    Files created:")
for fname in os.listdir("models"):
    size = os.path.getsize(os.path.join("models", fname))
    print(f"      {fname:45s} {size/1024:8.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: capture feature_columns BEFORE scaling.
# Add this line to your notebook right after pd.get_dummies():
#
#   feature_columns = list(X.columns)
#
# Then call scaler.fit_transform(X) as usual.
# ─────────────────────────────────────────────────────────────────────────────
