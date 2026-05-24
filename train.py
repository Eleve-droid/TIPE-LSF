"""
train.py — Entraînement du modèle KNN sur les landmarks MediaPipe
TIPE : Traduction de l'alphabet LSF en temps réel

Dataset attendu : data/landmarks.csv
  - Colonnes : landmark_1_x, landmark_1_y, ..., landmark_21_x, landmark_21_y, label
  - Labels : lettres de l'alphabet LSF (a, b, c, ...)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Paramètres ──────────────────────────────────────────────────────────────
DATA_CSV    = os.path.join("data", "landmarks.csv")
MODEL_DIR   = "models"
N_NEIGHBORS = 5
TEST_SIZE   = 0.15
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Chargement du dataset ────────────────────────────────────────────────────
print(f"[INFO] Chargement du dataset : {DATA_CSV}")
df = pd.read_csv(DATA_CSV)

# Supprimer la colonne 'image' si elle existe (non utilisée pour l'entraînement)
if "image" in df.columns:
    df = df.drop(columns=["image"])

feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols].values.astype(np.float32)
y = df["label"].values

labels = sorted(df["label"].unique())
print(f"[INFO] Classes détectées : {labels}")
print(f"[INFO] Taille du dataset  : {X.shape[0]} échantillons, {X.shape[1]} features")

# ── Normalisation des coordonnées (invariance à la taille de l'image) ────────
def normalize_landmarks(X):
    """
    Normalise chaque vecteur de landmarks par rapport au poignet (landmark 1).
    Rend la représentation invariante à la position et à l'échelle.
    """
    X_norm = X.copy()
    n_landmarks = X.shape[1] // 2
    for i in range(X.shape[0]):
        wrist_x = X[i, 0]
        wrist_y = X[i, 1]
        # Centrage par rapport au poignet
        for j in range(n_landmarks):
            X_norm[i, 2*j]   -= wrist_x
            X_norm[i, 2*j+1] -= wrist_y
        # Normalisation par la distance max (invariance à la taille de la main)
        max_dist = np.max(np.abs(X_norm[i])) + 1e-6
        X_norm[i] /= max_dist
    return X_norm

X = normalize_landmarks(X)

# ── Split train / test ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# ── StandardScaler ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Entraînement KNN ─────────────────────────────────────────────────────────
print(f"[INFO] Entraînement KNN (k={N_NEIGHBORS})...")
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=-1, algorithm="auto")
knn.fit(X_train, y_train)

# ── Évaluation ───────────────────────────────────────────────────────────────
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[RÉSULTAT] Précision sur le jeu de test : {acc*100:.2f}%")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title(f"Matrice de confusion — KNN (k={N_NEIGHBORS}) — Précision : {acc*100:.1f}%")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.jpg"), dpi=150)
plt.show()
print(f"[INFO] Matrice de confusion sauvegardée dans {MODEL_DIR}/confusion_matrix.jpg")

# ── Sauvegarde du modèle ─────────────────────────────────────────────────────
joblib.dump(knn,    os.path.join(MODEL_DIR, "knn_lsf.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_lsf.pkl"))
print(f"[INFO] Modèle sauvegardé dans {MODEL_DIR}/knn_lsf.pkl")
print(f"[INFO] Scaler sauvegardé dans {MODEL_DIR}/scaler_lsf.pkl")
