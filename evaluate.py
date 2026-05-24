"""
evaluate.py — Évaluation détaillée du modèle KNN
TIPE : Traduction de l'alphabet LSF en temps réel

Affiche : précision globale, rapport de classification, matrice de confusion.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Paramètres ──────────────────────────────────────────────────────────────
DATA_CSV    = os.path.join("data", "landmarks.csv")
MODEL_PATH  = os.path.join("models", "knn_lsf.pkl")
SCALER_PATH = os.path.join("models", "scaler_lsf.pkl")
TEST_SIZE   = 0.15
RANDOM_SEED = 42

# ── Chargement ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
if "image" in df.columns:
    df = df.drop(columns=["image"])

feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols].values.astype(np.float32)
y = df["label"].values
labels = sorted(df["label"].unique())

# Normalisation (même logique que train.py)
def normalize_landmarks(X):
    X_norm = X.copy()
    n_landmarks = X.shape[1] // 2
    for i in range(X.shape[0]):
        wrist_x, wrist_y = X[i, 0], X[i, 1]
        for j in range(n_landmarks):
            X_norm[i, 2*j]   -= wrist_x
            X_norm[i, 2*j+1] -= wrist_y
        max_dist = np.max(np.abs(X_norm[i])) + 1e-6
        X_norm[i] /= max_dist
    return X_norm

X = normalize_landmarks(X)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

knn    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X_test_scaled = scaler.transform(X_test)

# ── Évaluation ───────────────────────────────────────────────────────────────
y_pred = knn.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"Précision globale : {acc*100:.2f}%\n")
print("Rapport de classification :")
print(classification_report(y_test, y_pred, labels=labels))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=[l.upper() for l in labels],
            yticklabels=[l.upper() for l in labels], cmap="Blues")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title(f"Matrice de confusion — Précision : {acc*100:.1f}%")
plt.tight_layout()
plt.savefig(os.path.join("models", "confusion_matrix.jpg"), dpi=150)
plt.show()
