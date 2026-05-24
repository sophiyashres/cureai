"""
train_model.py
--------------
Trains three ML models (Random Forest, Decision Tree, Naive Bayes) on
disease-symptom data and saves them to the models/ directory using pickle.

Run once before starting the Flask app:
    python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_FILE   = os.path.join(DATASET_DIR, "disease_symptoms.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load & clean dataset ─────────────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_csv(DATA_FILE)

# Normalise symptom strings – strip whitespace, lower, replace spaces with _
symptom_cols = [c for c in df.columns if c.startswith("Symptom")]
for col in symptom_cols:
    df[col] = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_")
    df[col] = df[col].replace({"nan": None, "": None})

# Collect every unique symptom (drop None/NaN)
all_symptoms = sorted({
    s for col in symptom_cols
    for s in df[col].dropna().unique()
    if s and s != "nan"
})

print(f"  Total unique symptoms : {len(all_symptoms)}")
print(f"  Total diseases        : {df['Disease'].nunique()}")

# ── Build binary feature matrix ──────────────────────────────────────────────
print("Building feature matrix ...")
X = pd.DataFrame(0, index=df.index, columns=all_symptoms)
for idx, row in df.iterrows():
    for col in symptom_cols:
        symptom = row[col]
        if pd.notna(symptom) and symptom in all_symptoms:
            X.at[idx, symptom] = 1

# ── Encode target labels ──────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(df["Disease"])

# ── Train / test split (no stratify – dataset too small per class) ────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# ── Helper: train, evaluate, save ─────────────────────────────────────────────
def train_and_save(name, clf, fname):
    print(f"\nTraining {name} ...")
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test))

    # ShuffleSplit CV (works on tiny datasets unlike StratifiedKFold)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    cv_scores = []
    for tr, te in cv.split(X, y):
        clf_cv = type(clf)(**clf.get_params())
        clf_cv.fit(X.iloc[tr], y[tr])
        cv_scores.append(accuracy_score(y[te], clf_cv.predict(X.iloc[te])))

    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Test  accuracy : {test_acc:.4f}")
    print(f"  CV mean +/- std: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    path = os.path.join(MODEL_DIR, fname)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Saved -> {path}")
    return test_acc

# ── Train each model ──────────────────────────────────────────────────────────
rf_acc = train_and_save(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42),
    "random_forest.pkl"
)

dt_acc = train_and_save(
    "Decision Tree",
    DecisionTreeClassifier(random_state=42),
    "decision_tree.pkl"
)

nb_acc = train_and_save(
    "Naive Bayes (Multinomial)",
    MultinomialNB(),
    "naive_bayes.pkl"
)

# ── Save shared artefacts ─────────────────────────────────────────────────────
print("\nSaving symptom list and label encoder ...")

with open(os.path.join(MODEL_DIR, "symptoms.pkl"), "wb") as f:
    pickle.dump(all_symptoms, f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# Save model accuracy metadata (used by the Flask app)
accuracies = {
    "Random Forest": round(rf_acc * 100, 2),
    "Decision Tree": round(dt_acc * 100, 2),
    "Naive Bayes":   round(nb_acc * 100, 2),
}
with open(os.path.join(MODEL_DIR, "accuracies.pkl"), "wb") as f:
    pickle.dump(accuracies, f)

print("\n  All models trained and saved successfully!")
print("   You can now run:  python app.py")
