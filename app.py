"""
FedHealth Analytics — Prediction Server
No data is stored, logged, or shared. Pure in-memory inference.
"""

import os, json, logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_models  = {}
_scaler  = None
_cols    = None
_ready   = False
_err_msg = ""

def _load_all():
    global _models, _scaler, _cols, _ready, _err_msg

    required = {
        "annual_cost":   "annual_cost_model.joblib",
        "risk_score":    "risk_score_model.joblib",
        "high_risk_sgd": "high_risk_sgd_model.joblib",
        "high_risk_rf":  "high_risk_rf_model.joblib",
    }

    missing = []
    for key, fname in required.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname); continue
        try:
            _models[key] = joblib.load(path)
        except Exception as e:
            _err_msg = f"Failed to load {fname}: {e}"
            print(f"  ERROR: {_err_msg}"); return

    for fname, attr in [("scaler.joblib", "_scaler"), ("feature_columns.json", "_cols")]:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)

    if missing:
        _err_msg = f"Missing files in models/: {', '.join(missing)}"
        print(f"  WARNING: {_err_msg}"); return

    try:
        globals()["_scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
            globals()["_cols"] = json.load(f)
        globals()["_ready"] = True
        print("  All model artifacts loaded successfully.")
    except Exception as e:
        globals()["_err_msg"] = str(e)
        print(f"  ERROR: {e}")

print("\nFedHealth Analytics"); print("=" * 40)
_load_all()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"ok": _ready, "error": _err_msg if not _ready else None, "models": list(_models.keys())})


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    if not _ready:
        return jsonify({"error": f"Server models not ready. {_err_msg}"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    try:
        row = {
            "age":                     float(data["age"]),
            "gender":                  str(data["gender"]),
            "area":                    str(data["area"]),
            "bmi":                     float(data["bmi"]),
            "blood_pressure":          float(data["blood_pressure"]),
            "diabetes":                int(data["diabetes"]),
            "physical_activity_level": str(data["physical_activity_level"]),
            "smoker":                  str(data["smoker"]),
            "annual_premium":          float(data["annual_premium"]),
            "monthly_premium":         float(data["monthly_premium"]),
            "network_tier":            str(data["network_tier"]),
            "deductible":              float(data["deductible"]),
            "ldl":                     float(data["ldl"]),
            "hbalc":                   float(data["hbalc"]),
            "income":                  float(data["income"]),
            "copd":                    int(data["copd"]),
            "hypertension":            int(data["hypertension"]),
            "mental_health":           int(data["mental_health"]),
            "arthritis":               int(data["arthritis"]),
            "asthma":                  int(data["asthma"]),
            "chronic_count":           int(data["chronic_count"]),
        }
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid field: {e}"}), 400

    df = pd.DataFrame([row])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=_cols, fill_value=0)
    X  = _scaler.transform(df)

    annual_cost = float(_models["annual_cost"].predict(X)[0])
    risk_score  = float(_models["risk_score"].predict(X)[0])
    p_sgd       = float(_models["high_risk_sgd"].predict_proba(X)[0][1])
    p_rf        = float(_models["high_risk_rf"].predict_proba(X)[0][1])
    avg_prob    = (p_sgd + p_rf) / 2.0

    return jsonify({
        "annual_medical_cost":   round(annual_cost, 2),
        "risk_score":            round(risk_score,  4),
        "is_high_risk":          int(avg_prob > 0.5),
        "high_risk_probability": round(avg_prob,    4),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n  Open in browser → http://127.0.0.1:{port}")
    print("=" * 40 + "\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
