
<div align="center">

# 🛡️ Détection de Phishing par Deep Learning
### CNN + Interface Streamlit

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

<br/>

> **Module :** Deep Learning et Visualisation des Données  
> **Établissement :** École Nationale des Sciences Appliquées — El Jadida (ENSA El Jadida)  
> **Année académique :** 2025 – 2026

</div>

---

## 👥 Membres du Groupe

| # | Nom & Prénom | Établissement |
|---|--------------|---------------|
| 1 | **ALOUANE Kawter** | ENSA El Jadida |
| 2 | **AYA Baida** | ENSA El Jadida |
| 3 | **LAKHTIRI Salma** | ENSA El Jadida |

---

## 📋 Table des Matières

1. [Présentation du Projet](#-présentation-du-projet)
2. [Vue d'ensemble du Fonctionnement](#-vue-densemble-du-fonctionnement)
3. [Structure des Fichiers](#-structure-des-fichiers)
4. [Technologies & Bibliothèques](#-technologies--bibliothèques)
5. [Données Utilisées](#-données-utilisées)
6. [Prétraitement des Données](#-prétraitement-des-données)
7. [Architecture du Modèle](#-architecture-du-modèle)
8. [Entraînement & Évaluation](#-entraînement--évaluation)
9. [Sauvegarde & Chargement](#-sauvegarde--chargement)
10. [Interface Utilisateur](#-interface-utilisateur)
11. [Lancement du Projet](#-lancement-du-projet)
12. [Résultats](#-résultats)

---

## 🎯 Présentation du Projet

Ce projet s'inscrit dans le cadre du module **Deep Learning et Visualisation des Données**. Il vise à développer un système intelligent de **détection de phishing** basé sur des réseaux de neurones convolutifs (**CNN 1D**), couplé à une interface web interactive construite avec **Streamlit**.

### Problème traité

| Aspect | Détail |
|--------|--------|
| **Type** | Classification binaire supervisée |
| **Classe 1** | 🔴 Phishing (malveillant) |
| **Classe 0** | 🟢 Légitime (sûr) |

### Entrées acceptées par le système

- 🔗 **URL manuelle** saisie par l'utilisateur
- 📄 **Fichier CSV** contenant un ensemble d'URLs
- 🌐 **Fichier HTML / TXT / PDF / DOCX** (analyse du contenu textuel)

### Sorties du système

- ✅ Prédiction : **Phishing** ou **Légitime**
- 📊 Score de probabilité (entre 0 et 1)
- 📈 Métriques d'évaluation : accuracy, précision, rappel, F1-score, matrice de confusion

---

## 🔄 Vue d'ensemble du Fonctionnement

```
Données ──► Prétraitement ──► Modèle CNN ──► Entraînement ──► Évaluation ──► Sauvegarde
                                                                                    │
                                                                                    ▼
                                                              Interface Streamlit ◄── Prédiction
```

### Pipeline complet

```
1. Chargement des données        (CSV / HTML / texte)
2. Nettoyage & normalisation     (labels, texte brut)
3. Tokenisation + Padding        (texte → séquences numériques)
4. Création du modèle CNN 1D
5. Compilation du modèle
6. Entraînement (fit)
7. Évaluation sur données test
8. Sauvegarde du modèle entraîné
9. Rechargement pour prédiction
10. Affichage du résultat (Streamlit)
```

---

## 📁 Structure des Fichiers

```
projet-phishing/
│
├── app.py                        # Point d'entrée Streamlit
│
├── pages/
│   ├── 1_Train_URL.py            # Entraînement sur URLs (CSV)
│   ├── 2_Train_HTML.py           # Entraînement sur pages HTML
│   ├── 3_Predict.py              # Prédiction manuelle
│   └── 4_Metrics.py              # Visualisation des métriques
│
├── data/
│   ├── urltest.csv               # Dataset URLs (par défaut)
│   ├── raw/
│   │   ├── urls_dataset.csv      # Dataset URLs alternatif
│   │   └── pghtml/
│   │       ├── training/         # Pages HTML d'entraînement
│   │       │   ├── Phish/
│   │       │   └── NotPhish/
│   │       └── validation/       # Pages HTML de validation
│   │           ├── Phish/
│   │           └── NotPhish/
│
├── models/                       # Modèles sauvegardés (.keras)
├── tokenizers/                   # Tokeniseurs sauvegardés (.pkl)
├── metrics/                      # Métriques JSON
│
└── requirements.txt
```

---

## 🛠️ Technologies & Bibliothèques

### Bibliothèques principales

| Bibliothèque | Rôle |
|---|---|
| `TensorFlow / Keras` | Construction et entraînement du modèle CNN |
| `Streamlit` | Interface utilisateur web interactive |
| `scikit-learn` | Métriques d'évaluation, split train/test |
| `Pandas / NumPy` | Manipulation des données |
| `BeautifulSoup` | Extraction de texte depuis les pages HTML |
| `joblib` | Sérialisation des tokeniseurs |
| `re / os / gc / json` | Utilitaires système et nettoyage |

### Installation

```bash
pip install -r requirements.txt
```

Contenu de `requirements.txt` :

```
tensorflow>=2.10
streamlit>=1.25
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
beautifulsoup4>=4.12
joblib>=1.3
```

---

## 📊 Données Utilisées

### Types de données

> Ce projet traite exclusivement du **texte** (pas d'images).

| Source | Format | Description |
|--------|--------|-------------|
| `data/urltest.csv` | CSV | URLs avec labels |
| `data/raw/urls_dataset.csv` | CSV | Dataset alternatif |
| `data/raw/pghtml/training/` | HTML | Pages web d'entraînement |
| `data/raw/pghtml/validation/` | HTML | Pages web de validation |

### Labels / Classes

- **CSV** : labels textuels normalisés → `phishing`, `legitimate`, `1`, `0`, `yes`, `no`…
- **HTML** : organisation par dossiers → `Phish/` = 1, `NotPhish/` = 0

---

## ⚙️ Prétraitement des Données

### 1. Normalisation des labels

Les labels arrivent sous des formes variées. Une fonction les convertit uniformément en `0` ou `1` :

```python
def normalize_labels(series: pd.Series):
    def _map(v):
        s = str(v).strip().lower()
        if s in POSITIVE_LABELS:
            return 1
        if s in NEGATIVE_LABELS:
            return 0
        return None
    return series.apply(_map)
```

### 2. Tokenisation + Padding

Le CNN ne lit pas du texte brut — il lit des **séquences de nombres de taille fixe** :

```python
def create_and_fit_tokenizer(texts, max_words, char_level=True):
    tokenizer = Tokenizer(
        num_words=max_words,
        char_level=char_level,
        lower=True,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts([clean_text(t) for t in texts])
    return tokenizer
```

> **Analogie débutant :** C'est comme transformer chaque caractère en un code numérique, puis aligner toutes les séquences à la même longueur afin que le modèle puisse les comparer.

### 3. Split Train / Test

```python
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    x_text, y, test_size=0.2, random_state=42, stratify=y
)
```

| Ensemble | Proportion | Rôle |
|----------|-----------|------|
| Train | 80 % | Apprentissage du modèle |
| Test | 20 % | Évaluation sur données non vues |

---

## 🧠 Architecture du Modèle

### Type : CNN 1D (Convolutional Neural Network)

> **Pourquoi CNN 1D ?** → Excellent pour détecter des **motifs locaux** dans des séquences textuelles (sous-séquences suspectes dans les URLs).

```python
def build_url_cnn_model(max_words, max_len):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')   # Sortie : probabilité entre 0 et 1
    ])
    return model
```

### Description couche par couche

| Couche | Rôle |
|--------|------|
| `Embedding` | Transforme les indices en vecteurs denses |
| `Conv1D` | Détecte des motifs locaux dans la séquence |
| `GlobalMaxPooling1D` | Conserve le signal le plus fort |
| `Dense (ReLU)` | Apprentissage de représentations non-linéaires |
| `Dropout` | Régularisation → prévient l'overfitting |
| `Dense (Sigmoid)` | Sortie : score de probabilité Phishing ∈ [0, 1] |

---

## 📈 Entraînement & Évaluation

### Compilation

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `optimizer` | `adam` | Méthode adaptative d'ajustement des poids |
| `loss` | `binary_crossentropy` | Fonction de perte pour classification binaire |
| `metrics` | `accuracy` | % de prédictions correctes |

### Entraînement

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    callbacks=[es],   # EarlyStopping
    verbose=0,
)
```

| Hyperparamètre | Valeur |
|---------------|--------|
| Epochs | 10 (max) |
| Batch size | 64 |
| Early Stopping | ✅ (arrêt si plus d'amélioration) |

### Interprétation des courbes

| Signal | Interprétation |
|--------|----------------|
| `loss` ↘ + `val_loss` ↘ | ✅ Bon apprentissage |
| `loss` ↘ + `val_loss` ↗ | ⚠️ Overfitting (surapprentissage) |

### Évaluation

```python
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)
metrics = calculate_metrics(y_test, y_pred, y_prob)
```

Métriques calculées : **Accuracy**, **Précision**, **Rappel**, **F1-Score**, **Matrice de confusion**

---

## 💾 Sauvegarde & Chargement

```python
# Sauvegarde
model.save(URL_MODEL_PATH)
save_tokenizer(tokenizer, URL_TOKENIZER_PATH)
save_metrics(metrics, URL_METRICS_PATH)

# Chargement
model     = load_cnn_model(URL_MODEL_PATH)
tokenizer = load_tokenizer(URL_TOKENIZER_PATH)
```

> **Analogie :** Comme sauvegarder une partie de jeu vidéo — on recharge l'état déjà appris sans recommencer l'entraînement.

---

## 🖥️ Interface Utilisateur

### Framework : Streamlit

L'application est lancée via `app.py`, qui :
- Configure la page Streamlit (`st.set_page_config`)
- Crée les dossiers nécessaires (`models/`, `tokenizers/`, `metrics/`)
- Affiche l'accueil et délègue aux pages dans `pages/`

### Flux de prédiction (URL manuelle)

```python
# 1. Prétraitement de l'entrée
X = texts_to_padded_sequences([url], tokenizer, MAX_LEN_URL)

# 2. Prédiction
prob = float(model.predict(X, verbose=0)[0][0])
pred = int(prob >= 0.5)

# 3. Affichage
st.success(f"Résultat : {'Phishing 🔴' if pred == 1 else 'Légitime 🟢'}")
```

| Score | Résultat |
|-------|---------|
| ≥ 0.5 | 🔴 **Phishing** |
| < 0.5 | 🟢 **Légitime** |

---

## 🚀 Lancement du Projet

### Prérequis

- Python 3.10+
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd projet-phishing

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

L'interface sera accessible sur : **http://localhost:8501**

---

## 📊 Résultats

> Les résultats détaillés (courbes d'apprentissage, matrices de confusion, métriques) sont visualisables directement dans l'onglet **Métriques** de l'interface Streamlit après entraînement.

| Modèle | Données | Accuracy | F1-Score |
|--------|---------|----------|----------|
| CNN 1D | URLs (CSV) | — | — |
| CNN 1D | Pages HTML | — | — |

*(À compléter après exécution)*

---

## 📚 Références

- LeCun, Y. et al. — *Gradient-based learning applied to document recognition* (1998)
- Kim, Y. — *Convolutional Neural Networks for Sentence Classification* (2014)
- [Documentation TensorFlow/Keras](https://www.tensorflow.org/api_docs)
- [Documentation Streamlit](https://docs.streamlit.io)
- [scikit-learn — Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

<div align="center">

**ENSA El Jadida — 2025/2026**  
*Kawter ALOUANE · Baida AYA · Salma LAKHTIRI*

</div>
