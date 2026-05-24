"""
app.py
------
Flask application for Disease Prediction from Symptoms.
Loads pre-trained ML models (Random Forest, Decision Tree, Naive Bayes)
and serves a modern web UI for symptom-based disease prediction.

Start with:
    python app.py
Then visit: http://127.0.0.1:5000
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Model / artefact loading ──────────────────────────────────────────────────
REQUIRED_FILES = [
    "random_forest.pkl",
    "decision_tree.pkl",
    "naive_bayes.pkl",
    "symptoms.pkl",
    "label_encoder.pkl",
    "accuracies.pkl",
]

def _load(fname: str):
    """Load a pickle file from the models directory."""
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{fname}' not found in {MODEL_DIR}. "
            "Please run  python train_model.py  first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    rf_model  = _load("random_forest.pkl")
    dt_model  = _load("decision_tree.pkl")
    nb_model  = _load("naive_bayes.pkl")
    SYMPTOMS  = _load("symptoms.pkl")       # list of all symptom names
    le        = _load("label_encoder.pkl")  # LabelEncoder for disease names
    ACCURACIES = _load("accuracies.pkl")    # dict of model → accuracy %
    MODELS_LOADED = True
    print(f"✅  Loaded {len(SYMPTOMS)} symptoms and {len(le.classes_)} diseases.")
except FileNotFoundError as e:
    MODELS_LOADED = False
    MODEL_LOAD_ERROR = str(e)
    print(f"⚠️  {e}")


# ── Pretty-print symptom names for the UI ────────────────────────────────────
def _pretty(symptom: str) -> str:
    """Convert 'burning_micturition' → 'Burning Micturition'."""
    return symptom.replace("_", " ").replace("  ", " ").title()

SYMPTOM_DISPLAY = {s: _pretty(s) for s in (SYMPTOMS if MODELS_LOADED else [])}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main prediction page."""
    if not MODELS_LOADED:
        return render_template(
            "error.html",
            error=MODEL_LOAD_ERROR
        ), 500

    # Group symptoms alphabetically for the UI
    symptoms_sorted = sorted(SYMPTOM_DISPLAY.items(), key=lambda x: x[1])
    return render_template(
        "index.html",
        symptoms=symptoms_sorted,
        accuracies=ACCURACIES,
        disease_count=len(le.classes_),
        symptom_count=len(SYMPTOMS),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON  { "symptoms": ["fever", "cough", ...] }
    Returns  JSON with predictions from all three models.
    """
    if not MODELS_LOADED:
        return jsonify({"error": MODEL_LOAD_ERROR}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request – expected JSON body."}), 400

    selected = data.get("symptoms", [])
    if not selected:
        return jsonify({"error": "Please select at least one symptom."}), 400

    # Validate symptoms
    invalid = [s for s in selected if s not in SYMPTOMS]
    if invalid:
        return jsonify({"error": f"Unknown symptoms: {', '.join(invalid)}"}), 400

    if len(selected) < 1:
        return jsonify({"error": "Please select at least 1 symptom."}), 400

    # Build input vector (binary encoding)
    input_vec = pd.DataFrame(0, index=[0], columns=SYMPTOMS)
    for sym in selected:
        idx = SYMPTOMS.index(sym)
        input_vec.at[0, sym] = 1

    # ── Predict with each model ───────────────────────────────────────────────
    results = {}
    model_map = {
        "Random Forest": rf_model,
        "Decision Tree": dt_model,
        "Naive Bayes":   nb_model,
    }

    for model_name, clf in model_map.items():
        pred_idx  = clf.predict(input_vec)[0]
        disease   = le.inverse_transform([pred_idx])[0]

        # Confidence: use predict_proba if available
        if hasattr(clf, "predict_proba"):
            proba     = clf.predict_proba(input_vec)[0]
            confidence = round(float(proba[pred_idx]) * 100, 1)

            # Top-3 alternatives
            top3_idx  = np.argsort(proba)[::-1][:3]
            top3      = [
                {
                    "disease":    le.inverse_transform([i])[0],
                    "confidence": round(float(proba[i]) * 100, 1),
                }
                for i in top3_idx
            ]
        else:
            confidence = ACCURACIES.get(model_name, 0.0)
            top3       = [{"disease": disease, "confidence": confidence}]

        results[model_name] = {
            "disease":    disease,
            "confidence": confidence,
            "top3":       top3,
        }

    # Majority-vote consensus
    votes = [v["disease"] for v in results.values()]
    consensus = max(set(votes), key=votes.count)
    consensus_confidence = round(
        np.mean([v["confidence"] for v in results.values() if v["disease"] == consensus]),
        1
    )

    return jsonify({
        "success":              True,
        "selected_symptoms":    [_pretty(s) for s in selected],
        "results":              results,
        "consensus_disease":    consensus,
        "consensus_confidence": consensus_confidence,
        "accuracies":           ACCURACIES,
    })


@app.route("/symptoms")
def get_symptoms():
    """Return the full symptom list as JSON (for dynamic search)."""
    if not MODELS_LOADED:
        return jsonify({"error": MODEL_LOAD_ERROR}), 500
    return jsonify({
        "symptoms": [
            {"key": k, "label": v}
            for k, v in sorted(SYMPTOM_DISPLAY.items(), key=lambda x: x[1])
        ]
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Disease Predictor on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
