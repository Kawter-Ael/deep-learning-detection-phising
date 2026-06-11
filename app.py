import os
import streamlit as st

st.set_page_config(page_title="Détection Phishing ", page_icon="🛡️", layout="wide")

# Création automatique des dossiers demandés
REQUIRED_DIRS = [
    "data/raw",
    "data/raw/pghtml",
    "models",
    "outputs",
    "pages",
    "utils",
]
for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)

st.title("Application de Détection de Phishing (CNN 1D)")
st.write(
    
)
st.info("Le seul modèle utilisé dans ce projet est un CNN  (TensorFlow/Keras).")
