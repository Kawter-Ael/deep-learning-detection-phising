import os
import pandas as pd
import streamlit as st

from utils.paths import URL_MODEL_PATH, URL_TOKENIZER_PATH, OUTPUTS_DIR, PREDICTIONS_URLS_CSV
from utils.file_reader import read_csv_file
from utils.preprocessing import detect_url_column, normalize_labels, texts_to_padded_sequences
from utils.cnn_model import load_cnn_model, load_tokenizer
from utils.metrics_utils import calculate_eval_summary
from utils.shap_utils import calculate_shap_values, get_top_shap_features, plot_shap_bar, plot_shap_summary

MAX_LEN_URL = 200

st.title("Detection URL")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

if not os.path.exists(URL_MODEL_PATH) or not os.path.exists(URL_TOKENIZER_PATH):
    st.warning("Modele URL non trouve. Lance d'abord l'entrainement depuis Accueil.")

mode = st.radio("Mode", ["URL manuelle", "Fichier CSV"], horizontal=True)

if mode == "URL manuelle":
    url = st.text_input("Saisir une URL")
    show_shap = st.checkbox("Afficher l'explication SHAP", value=True)
    if st.button("Predire URL"):
        if not url.strip():
            st.warning("URL vide.")
        elif not os.path.exists(URL_MODEL_PATH) or not os.path.exists(URL_TOKENIZER_PATH):
            st.warning("Modele URL manquant.")
        else:
            model = load_cnn_model(URL_MODEL_PATH)
            tokenizer = load_tokenizer(URL_TOKENIZER_PATH)
            X = texts_to_padded_sequences([url], tokenizer, MAX_LEN_URL)
            prob = float(model.predict(X, verbose=0)[0][0])
            pred = int(prob >= 0.5)
            st.success(f"Resultat: {'Phishing' if pred == 1 else 'Legitime'}")
            st.write(f"Probabilite phishing: {prob:.4f}")
            if show_shap:
                try:
                    shap_result = calculate_shap_values(
                        model,
                        tokenizer,
                        [url],
                        MAX_LEN_URL,
                        max_features=80,
                        nsamples=80,
                    )
                    st.pyplot(plot_shap_bar(shap_result, title="Explication SHAP de l'URL analysee", top_k=15))
                    st.write("Caracteristiques les plus influentes")
                    st.dataframe(get_top_shap_features(shap_result, top_k=15, row_index=0))
                except Exception as exc:
                    st.warning(f"SHAP indisponible pour cette prediction: {exc}")

else:
    uploaded = st.file_uploader("Uploader CSV contenant URLs", type=["csv"])
    if uploaded is not None:
        df = read_csv_file(uploaded)
        st.dataframe(df.head())
        detected = detect_url_column(df)
        url_col = st.selectbox("Colonne URL", options=df.columns, index=list(df.columns).index(detected) if detected in df.columns else 0)
        detected_label = next((c for c in df.columns if str(c).lower() in {"label", "status", "result", "class"}), None)
        label_options = ["Aucune"] + list(df.columns)
        label_index = label_options.index(detected_label) + 1 if detected_label in df.columns else 0
        label_col = st.selectbox("Colonne label (si disponible)", options=label_options, index=label_index)
        show_shap = st.checkbox("Afficher un graphe SHAP sur un petit echantillon", value=False)

        if st.button("Predire CSV"):
            if not os.path.exists(URL_MODEL_PATH) or not os.path.exists(URL_TOKENIZER_PATH):
                st.warning("Modele URL manquant.")
            else:
                model = load_cnn_model(URL_MODEL_PATH)
                tokenizer = load_tokenizer(URL_TOKENIZER_PATH)
                urls = df[url_col].astype(str).tolist()
                X = texts_to_padded_sequences(urls, tokenizer, MAX_LEN_URL)
                probs = model.predict(X, verbose=0).flatten()
                preds = (probs >= 0.5).astype(int)
                out = pd.DataFrame({
                    "url": urls,
                    "probabilite_phishing": probs,
                    "prediction": ["Phishing" if p == 1 else "Legitime" for p in preds],
                })
                st.dataframe(out)

                if label_col != "Aucune":
                    try:
                        label_series = normalize_labels(df[label_col])
                        valid_mask = label_series.notna().values
                        y_true = label_series[valid_mask].astype(int).values
                        y_prob = probs[valid_mask]
                        if len(y_true) == len(y_prob):
                            eval_summary = calculate_eval_summary(y_true, y_prob)
                            st.subheader("Evaluation sur les donnees importees")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Accuracy", f"{eval_summary['accuracy']:.4f}")
                            c2.metric("Loss", f"{eval_summary['loss']:.4f}")
                            c3.metric("ROC AUC", f"{eval_summary['roc_auc']:.4f}")
                    except Exception as exc:
                        st.warning(f"Impossible de calculer les metriques importees: {exc}")

                if show_shap:
                    try:
                        shap_texts = urls[:5]
                        shap_result = calculate_shap_values(
                            model,
                            tokenizer,
                            shap_texts,
                            MAX_LEN_URL,
                            max_features=80,
                            background_texts=urls[5:10],
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
                        st.warning(f"SHAP indisponible pour ce CSV: {exc}")
                out.to_csv(PREDICTIONS_URLS_CSV, index=False)
                st.download_button("Telecharger resultats", data=out.to_csv(index=False).encode("utf-8"), file_name="predictions_urls.csv", mime="text/csv")
