"""
capture_hand_data.py — Collecte de données landmarks pour l'entraînement
TIPE : Traduction de l'alphabet LSF en temps réel

Ce script capture des landmarks de main via la webcam et les enregistre
dans un fichier CSV prêt pour l'entraînement.

Utilisation :
  - Positionnez votre main devant la caméra
  - Appuyez sur la touche correspondant à la lettre souhaitée (a, b, c, ...)
  - Le script enregistre N captures par appui
  - Appuyez sur 'q' pour quitter et sauvegarder
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# ── Paramètres ──────────────────────────────────────────────────────────────
OUTPUT_CSV      = os.path.join("data", "landmarks.csv")
LABELS          = list("abcde")          # Lettres à capturer (étendre selon besoin)
N_PER_KEYPRESS  = 5                      # Captures sauvegardées par appui touche
CONFIDENCE      = 0.7

os.makedirs("data", exist_ok=True)

# ── Initialisation MediaPipe ──────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=CONFIDENCE,
    min_tracking_confidence=0.5,
)

# ── Chargement des données existantes ────────────────────────────────────────
if os.path.exists(OUTPUT_CSV):
    df_existing = pd.read_csv(OUTPUT_CSV)
    records = df_existing.to_dict("records")
    print(f"[INFO] {len(records)} entrées existantes chargées depuis {OUTPUT_CSV}")
else:
    records = []
    print(f"[INFO] Nouveau fichier CSV : {OUTPUT_CSV}")

# ── Construction des noms de colonnes ────────────────────────────────────────
columns = []
for i in range(1, 22):
    columns += [f"landmark_{i}_x", f"landmark_{i}_y"]
columns.append("label")

# ── Boucle de capture ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERREUR] Impossible d'accéder à la webcam.")
    exit(1)

print(f"[INFO] Appuyez sur une touche parmi {LABELS} pour capturer. 'q' pour sauvegarder et quitter.")

current_landmarks = None
label_counts = {l: sum(1 for r in records if r["label"] == l) for l in LABELS}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_landmarks = None

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            coords = []
            for lm in hand_lm.landmark:
                # Coordonnées normalisées MediaPipe [0,1]
                coords.extend([lm.x, lm.y])
            current_landmarks = coords

    # ── Affichage compteurs ───────────────────────────────────────────────────
    h, w = frame.shape[:2]
    y_offset = 30
    for label in LABELS:
        count = label_counts.get(label, 0)
        cv2.putText(frame, f"{label.upper()} : {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 100) if count >= 50 else (0, 150, 255), 2)
        y_offset += 28

    status = "Main detectee" if current_landmarks else "Aucune main"
    color  = (0, 255, 0) if current_landmarks else (0, 0, 255)
    cv2.putText(frame, status, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "Appui touche = capture | q = sauvegarder", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Capture LSF", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    key_char = chr(key).lower() if key < 128 else ""
    if key_char in LABELS and current_landmarks is not None:
        for _ in range(N_PER_KEYPRESS):
            row = dict(zip(columns[:-1], current_landmarks))
            row["label"] = key_char
            records.append(row)
        label_counts[key_char] = label_counts.get(key_char, 0) + N_PER_KEYPRESS
        print(f"[SAVE] '{key_char.upper()}' → {label_counts[key_char]} entrées au total")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

df_out = pd.DataFrame(records, columns=columns)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n[INFO] {len(df_out)} entrées sauvegardées dans {OUTPUT_CSV}")
for label in LABELS:
    n = (df_out["label"] == label).sum()
    print(f"  {label.upper()} : {n} échantillons")
