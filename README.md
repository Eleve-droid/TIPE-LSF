# 🤟 LSF Alphabet Recognition — TIPE

Reconnaissance en temps réel de l'alphabet de la **Langue des Signes Française (LSF)** à partir d'un flux vidéo, via un algorithme **KNN** et **MediaPipe**.

Projet réalisé dans le cadre du TIPE (Travaux d'Initiative Personnelle Encadrés) en CPGE TSI.

---

## Démonstration

<p align="center">
  <img src="images/realtime_prediction.jpg" width="700" alt="Détection des landmarks">
</p>

Le programme détecte la main en temps réel, extrait les **21 landmarks** articulaires via MediaPipe, et prédit la lettre signée.

**Précision obtenue : ~85 % sur l'alphabet A–E.**

---

## Architecture du projet

```text

TIPE-LSF/
│
├── data/
│   └── landmarks.csv
│
├── images/
│   ├── demo_detection.png
│   ├── realtime_prediction.png
│   └── confusion_matrix.png
│
├── models/
│   ├── knn_lsf.pkl
│   └── scaler_lsf.pkl
│
├── capture_hand_data.py
├── train.py
├── evaluate.py
├── prediction.py
├── requirements.txt
└── README.md
```

---

## Principe technique

### 1. Extraction des features — MediaPipe

MediaPipe Hands détecte **21 points clés** (landmarks) sur la main, chacun avec des coordonnées (x, y) normalisées.

```
                8   12  16  20
                |   |   |   |
            7   11  15  19
        6   10  14  18
    5   9   13  17
        4
        3
        2
    1 (poignet)
```

Ces 21 × 2 = **42 valeurs** constituent le vecteur de features.

<p align="center">
  <img src="images/demo_detection.jpg" width="700" alt="Prédiction temps réel">
</p>

### 2. Normalisation

Pour rendre le modèle invariant à la **position** et à la **taille de la main** :
- Centrage de tous les points par rapport au poignet (landmark 1)
- Division par la distance maximale au poignet

### 3. Classificateur KNN

- **k = 5 voisins**, distance euclidienne
- StandardScaler appliqué avant classification
- Entraîné sur le dataset `data/landmarks.csv`

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/TIPE-LSF.git
cd TIPE-LSF

# Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Étape 1 — (Optionnel) Collecter ses propres données

```bash
python capture_hand_data.py
```

- Positionnez votre main devant la caméra
- Appuyez sur `a`, `b`, `c`, `d` ou `e` pour capturer les landmarks
- `q` pour sauvegarder et quitter → génère `data/landmarks.csv`

### Étape 2 — Entraîner le modèle

```bash
python train.py
```

Génère `models/knn_lsf.pkl`, `models/scaler_lsf.pkl` et la matrice de confusion.

### Étape 3 — Reconnaissance en temps réel

```bash
python prediction.py
```

- La lettre prédite s'affiche en grand sur le flux vidéo
- `q` pour quitter

### Évaluation détaillée

```bash
python evaluate.py
```

---

## Dataset

Le fichier `data/landmarks.csv` contient les landmarks extraits d'images de l'alphabet LSF.

| Colonne | Description |
|---|---|
| `landmark_1_x` à `landmark_21_x` | Coordonnée x de chaque point clé |
| `landmark_1_y` à `landmark_21_y` | Coordonnée y de chaque point clé |
| `label` | Lettre signée (`a`, `b`, `c`, `d`, `e`) |

---

## Résultats

| Métrique | Valeur |
|---|---|
| Précision (test set) | ~85 % |
| Algorithme | KNN (k=5) |
| Features | 42 (21 landmarks × x,y) |
| Classes | A, B, C, D, E |

### Matrice de confusion

<p align="center">
  <img src="images/confusion_matrix.jpg" width="600" alt="Matrice de confusion">
</p>

---

## 🔬 Technologies utilisées

- **Python 3.10+**
- **MediaPipe** — détection des landmarks de la main
- **OpenCV** — traitement vidéo temps réel
- **scikit-learn** — KNN, StandardScaler
- **NumPy / pandas** — manipulation des données
- **Matplotlib / Seaborn** — visualisation et matrice de confusion

---

## 👨‍💻 Auteur

**TOURAB Ilyas**  
Étudiant ingénieur — Génie Informatique & Intelligence Artificielle  
Université de Technologie de Compiègne (UTC)
