# deep-learning-detection-phising
# 🛡️ Phishing Detection System using CNN and Streamlit

## 📌 Description

Ce projet utilise des réseaux de neurones convolutifs (**CNN 1D**) pour détecter les tentatives de phishing à partir :

- d'URLs
- de pages HTML
- de fichiers texte (TXT, PDF, DOCX, EML, MSG)
- de fichiers CSV contenant des URLs

L'application est développée avec **Python**, **TensorFlow/Keras** et **Streamlit** afin de fournir une interface simple pour l'entraînement et la prédiction.

---

## 🎯 Objectifs

- Détecter automatiquement les URLs malveillantes.
- Analyser le contenu HTML des pages web.
- Fournir une interface intuitive pour les utilisateurs.
- Afficher les métriques d'évaluation du modèle.
- Expliquer les prédictions grâce à SHAP.

---

## 🏗️ Architecture du projet

```text
project/
│
├── app.py
│
├── pages/
│   ├── 1_Accueil.py
│   ├── 2_Detection_URL.py
│   └── 3_Detection__Fichiers.py
│
├── utils/
│   ├── cnn_model.py
│   ├── file_reader.py
│   ├── metrics_utils.py
│   ├── paths.py
│   ├── preprocessing.py
│   ├── shap_utils.py
│   ├── train.py
│   └── visualization.py
│
├── data/
│   ├── urltest.csv
│   └── raw/
│       ├── urls_dataset.csv
│       └── pghtml/
│
├── models/
│
├── outputs/
│
└── requirements.txt
