"""
prediction.py — Reconnaissance en temps réel de l'alphabet LSF
TIPE : Traduction de l'alphabet LSF via flux vidéo

Utilise MediaPipe pour extraire les landmarks de la main,
puis le modèle KNN entraîné pour prédire la lettre.

Prérequis : avoir exécuté train.py au préalable (génère models/knn_lsf.pkl)
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time

# ── Chargement du modèle ─────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "knn_lsf.pkl")
SCALER_PATH = os.path.join("models", "scaler_lsf.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("[ERREUR] Modèle introuvable. Lancez d'abord train.py.")
    exit(1)

knn    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("[INFO] Modèle chargé.")

# ── Initialisation MediaPipe ──────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── Fonction : extraction et normalisation des landmarks ─────────────────────
def extract_landmarks(hand_landmarks):
    """Extrait et normalise les 21 landmarks (x, y) d'une main détectée."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y])
    coords = np.array(coords, dtype=np.float32)

    # Centrage par rapport au poignet
    wrist_x, wrist_y = coords[0], coords[1]
    for i in range(21):
        coords[2*i]   -= wrist_x
        coords[2*i+1] -= wrist_y

    # Normalisation par la distance max
    max_dist = np.max(np.abs(coords)) + 1e-6
    coords /= max_dist
    return coords

# ── Boucle de capture ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERREUR] Impossible d'accéder à la webcam.")
    exit(1)

print("[INFO] Démarrage de la reconnaissance. Appuyez sur 'q' pour quitter.")

prediction_text  = ""
confidence_text  = ""
last_pred_time   = 0
PRED_INTERVAL    = 0.1   # secondes entre deux prédictions

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)          # miroir horizontal
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    now = time.time()

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            # Dessin des landmarks
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Prédiction à intervalle régulier
            if now - last_pred_time > PRED_INTERVAL:
                features = extract_landmarks(hand_lm).reshape(1, -1)
                # Conversion pixel → normalisé MediaPipe (déjà en [0,1])
                # Le scaler est appliqué directement
                features_scaled = scaler.transform(features)
                pred = knn.predict(features_scaled)[0]

                # Probabilité approchée via vote des voisins
                proba = knn.predict_proba(features_scaled).max() * 100
                prediction_text = pred.upper()
                confidence_text = f"{proba:.0f}%"
                last_pred_time  = now

    # ── Affichage ─────────────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    # Fond semi-transparent en bas
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if prediction_text:
        # Lettre principale (grande)
        cv2.putText(frame, prediction_text, (30, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 3.5, (255, 255, 255), 6)
        # Confiance
        cv2.putText(frame, f"Confiance : {confidence_text}", (180, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 100), 2)
    else:
        cv2.putText(frame, "Aucune main detectee", (30, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

    cv2.putText(frame, "LSF - Alphabet | q : quitter", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Reconnaissance LSF", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Programme terminé.")
