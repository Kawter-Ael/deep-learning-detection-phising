import os
import gc
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from utils.paths import (
    MODELS_DIR,
    URLS_DATASET_PATH,
    PGHTML_DIR,
    URL_MODEL_PATH,
    URL_TOKENIZER_PATH,
    URL_METRICS_PATH,
    HTML_MODEL_PATH,
    HTML_TOKENIZER_PATH,
    HTML_METRICS_PATH,
)
from utils.file_reader import read_csv_file, read_html_dataset_from_folders
from utils.preprocessing import (
    detect_url_column,
    detect_label_column,
    normalize_labels,
    create_and_fit_tokenizer,
    texts_to_padded_sequences,
)
from utils.cnn_model import build_url_cnn_model, build_html_cnn_model, save_tokenizer
from utils.metrics_utils import calculate_metrics, save_metrics, load_metrics
from utils.visualization import plot_accuracy_history, plot_confusion_matrix, plot_loss_history, plot_metrics_bar, plot_roc_curve
from utils.shap_utils import calculate_shap_values, get_top_shap_features, plot_shap_summary

MAX_LEN_URL = 200
MAX_WORDS_URL = 10000
MAX_LEN_HTML = 2000
MAX_WORDS_HTML = 20000

st.title("Accueil - Entrainement + Resultats")
st.write("Entrainer les modeles CNN (URL + HTML) et voir les metriques.")
os.makedirs(MODELS_DIR, exist_ok=True)

st.subheader("Donnees disponibles")
url_candidates = ["data/urltest.csv", URLS_DATASET_PATH]
for p in url_candidates:
    st.write(f"{'OK' if os.path.exists(p) else 'MANQUANT'} - `{p}`")

if os.path.isdir(PGHTML_DIR):
    for sub in ["training/NotPhish", "training/Phish", "validation/NotPhish", "validation/Phish"]:
        folder = os.path.join(PGHTML_DIR, sub)
        count = len([f for f in os.listdir(folder) if f.lower().endswith((".html", ".htm"))]) if os.path.isdir(folder) else 0
        st.write(f"{sub}: {count} fichiers")
else:
    st.write("MANQUANT - data/raw/pghtml")

st.markdown("---")
st.subheader("Entrainement URL CNN")
uploaded = st.file_uploader("CSV URL (optionnel)", type=["csv"], key="train_url_csv")
use_default_a = st.checkbox("Utiliser data/urltest.csv", value=True)
use_default_b = st.checkbox("Utiliser data/raw/urls_dataset.csv")
show_url_shap = st.checkbox("Afficher SHAP apres entrainement URL", value=False)

df_url = None
if uploaded is not None:
    df_url = read_csv_file(uploaded)
elif use_default_a and os.path.exists("data/urltest.csv"):
    df_url = read_csv_file("data/urltest.csv")
elif use_default_b and os.path.exists(URLS_DATASET_PATH):
    df_url = read_csv_file(URLS_DATASET_PATH)

if df_url is not None:
    st.dataframe(df_url.head())
    detected_url = detect_url_column(df_url)
    detected_label = detect_label_column(df_url)
    url_col = st.selectbox("Colonne URL", options=df_url.columns, index=list(df_url.columns).index(detected_url) if detected_url in df_url.columns else 0)
    label_col = st.selectbox("Colonne Label", options=df_url.columns, index=list(df_url.columns).index(detected_label) if detected_label in df_url.columns else 0)

    if st.button("Lancer entrainement URL"):
        y = normalize_labels(df_url[label_col])
        valid = y.notna()
        x_text = df_url.loc[valid, url_col].astype(str).tolist()
        y = y.loc[valid].astype(int).values

        if len(x_text) < 10:
            st.error("Pas assez de donnees valides.")
        else:
            X_train_txt, X_test_txt, y_train, y_test = train_test_split(
                x_text, y, test_size=0.2, random_state=42, stratify=y
            )
            tokenizer = create_and_fit_tokenizer(X_train_txt, MAX_WORDS_URL, char_level=True)
            X_train = texts_to_padded_sequences(X_train_txt, tokenizer, MAX_LEN_URL)
            X_test = texts_to_padded_sequences(X_test_txt, tokenizer, MAX_LEN_URL)

            model = build_url_cnn_model(MAX_WORDS_URL, MAX_LEN_URL)
            es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=10,
                batch_size=64,
                callbacks=[es],
                verbose=0,
            )

            y_prob = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = calculate_metrics(y_test, y_pred, y_prob)

            model.save(URL_MODEL_PATH)
            save_tokenizer(tokenizer, URL_TOKENIZER_PATH)
            save_metrics(metrics, URL_METRICS_PATH)

            st.success("Modele URL entraine et sauvegarde.")
            st.json({k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "y_prob_preview"]})
            st.text(metrics["classification_report"])
            st.pyplot(plot_confusion_matrix(np.array(metrics["confusion_matrix"])))
            st.pyplot(plot_accuracy_history(history))
            st.pyplot(plot_loss_history(history))
            st.pyplot(plot_roc_curve(y_test, y_prob))
            if show_url_shap:
                try:
                    shap_result = calculate_shap_values(
                        model,
                        tokenizer,
                        X_test_txt[:5],
                        MAX_LEN_URL,
                        max_features=80,
                        background_texts=X_train_txt[:5],
                        nsamples=80,
                    )
                    st.pyplot(
                        plot_shap_summary(
                            shap_result,
                            title="Influence globale des caracteristiques sur la detection phishing",
                            top_k=20,
                        )
                    )
                    st.write("Caracteristiques globales les plus influentes")
                    st.dataframe(get_top_shap_features(shap_result, top_k=20))
                except Exception as exc:
                    st.warning(f"SHAP indisponible pour URL: {exc}")

st.markdown("---")
st.subheader("Entrainement HTML CNN")
show_html_shap = st.checkbox("Afficher SHAP apres entrainement HTML", value=False)
if st.button("Lancer entrainement HTML"):
    if not os.path.isdir(PGHTML_DIR):
        st.error("Dossier data/raw/pghtml introuvable.")
    else:
        X_train_html, y_train_html, X_val_html, y_val_html, stats = read_html_dataset_from_folders(PGHTML_DIR)
        if len(X_train_html) == 0 or len(X_val_html) == 0:
            st.warning("Donnees HTML insuffisantes.")
        else:
            tokenizer = create_and_fit_tokenizer(X_train_html, MAX_WORDS_HTML, char_level=True)
            X_train = texts_to_padded_sequences(X_train_html, tokenizer, MAX_LEN_HTML, batch_size=200)
            X_val = texts_to_padded_sequences(X_val_html, tokenizer, MAX_LEN_HTML, batch_size=200)

            model = build_html_cnn_model(MAX_WORDS_HTML, MAX_LEN_HTML)
            es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
            history = model.fit(
                X_train,
                np.array(y_train_html),
                validation_data=(X_val, np.array(y_val_html)),
                epochs=8,
                batch_size=32,
                callbacks=[es],
                verbose=0,
            )

            y_prob = model.predict(X_val, verbose=0).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = calculate_metrics(np.array(y_val_html), y_pred, y_prob)
            metrics["stats_counts"] = stats

            model.save(HTML_MODEL_PATH)
            save_tokenizer(tokenizer, HTML_TOKENIZER_PATH)
            save_metrics(metrics, HTML_METRICS_PATH)

            st.success("Modele HTML entraine et sauvegarde.")
            st.json({k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report", "y_prob_preview"]})
            st.text(metrics["classification_report"])
            st.pyplot(plot_confusion_matrix(np.array(metrics["confusion_matrix"])))
            st.pyplot(plot_accuracy_history(history))
            st.pyplot(plot_loss_history(history))
            st.pyplot(plot_roc_curve(np.array(y_val_html), y_prob))
            if show_html_shap:
                try:
                    shap_result = calculate_shap_values(
                        model,
                        tokenizer,
                        X_val_html[:3],
                        MAX_LEN_HTML,
                        max_features=120,
                        background_texts=X_train_html[:3],
                        nsamples=80,
                    )
                    st.pyplot(
                        plot_shap_summary(
                            shap_result,
                            title="Influence globale des caracteristiques sur la detection phishing",
                            top_k=20,
                        )
                    )
                    st.write("Caracteristiques HTML les plus influentes")
                    st.dataframe(get_top_shap_features(shap_result, top_k=20))
                except Exception as exc:
                    st.warning(f"SHAP indisponible pour HTML: {exc}")

            del X_train, X_val, X_train_html, X_val_html
            gc.collect()

st.markdown("---")
st.subheader("Resultats sauvegardes")
if os.path.exists(URL_METRICS_PATH):
    st.write("Metriques URL")
    m = load_metrics(URL_METRICS_PATH)
    st.json({k: v for k, v in m.items() if k not in ["confusion_matrix", "classification_report", "y_prob_preview"]})
    st.pyplot(plot_metrics_bar(m, title="Metriques sauvegardees - URL"))
    if "confusion_matrix" in m:
        st.pyplot(plot_confusion_matrix(np.array(m["confusion_matrix"])))
else:
    st.write("Metriques URL non trouvees.")

if os.path.exists(HTML_METRICS_PATH):
    st.write("Metriques HTML")
    m = load_metrics(HTML_METRICS_PATH)
    st.json({k: v for k, v in m.items() if k not in ["confusion_matrix", "classification_report", "y_prob_preview"]})
    st.pyplot(plot_metrics_bar(m, title="Metriques sauvegardees - HTML"))
    if "confusion_matrix" in m:
        st.pyplot(plot_confusion_matrix(np.array(m["confusion_matrix"])))
else:
    st.write("Metriques HTML non trouvees.")
