import os

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from utils.cnn_model import load_cnn_model, load_tokenizer
from utils.file_reader import (
    read_csv_file,
    read_docx_file,
    read_eml_file,
    read_excel_file,
    read_html_file,
    read_image_file,
    read_msg_file,
    read_pdf_file,
    read_txt_file,
)
from utils.paths import (
    HTML_MODEL_PATH,
    HTML_TOKENIZER_PATH,
    OUTPUTS_DIR,
    PREDICTIONS_HTML_CSV,
    URL_MODEL_PATH,
    URL_TOKENIZER_PATH,
)
from utils.preprocessing import detect_url_column, extract_urls_from_text, normalize_labels, texts_to_padded_sequences
from utils.metrics_utils import calculate_eval_summary
from utils.shap_utils import calculate_shap_values, get_top_shap_features, plot_shap_bar

MAX_LEN_URL = 200
MAX_LEN_HTML = 2000
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def _unique_urls(urls):
    seen = set()
    unique = []
    for url in urls:
        cleaned = str(url).strip().rstrip(".,);]}")
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    return unique


def _extract_urls_from_table(df):
    detected = detect_url_column(df)
    if detected in df.columns:
        return df[detected].dropna().astype(str).tolist()
    return extract_urls_from_text(df.astype(str).fillna("").to_string(index=False))


def _predict_urls(urls, model, tokenizer):
    X = texts_to_padded_sequences(urls, tokenizer, MAX_LEN_URL)
    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "url": urls,
            "probabilite_phishing": probs,
            "prediction": ["Phishing" if p == 1 else "Legitime" for p in preds],
        }
    )


st.title("Detection HTML / Fichiers")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

uploaded = st.file_uploader(
    "Importer un fichier",
    type=[
        "html",
        "htm",
        "txt",
        "pdf",
        "docx",
        "csv",
        "xlsx",
        "xls",
        "eml",
        "msg",
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff",
        "webp",
    ],
)

if uploaded is not None:
    ext = os.path.splitext(uploaded.name)[1].lower()

    has_html_model = os.path.exists(HTML_MODEL_PATH) and os.path.exists(HTML_TOKENIZER_PATH)
    has_url_model = os.path.exists(URL_MODEL_PATH) and os.path.exists(URL_TOKENIZER_PATH)

    content = ""
    urls = []
    df_table = None
    image_ocr_empty = False
    msg_reader_empty = False

    if ext == ".csv":
        df_table = read_csv_file(uploaded)
        st.dataframe(df_table.head())
        content = df_table.astype(str).fillna("").to_string(index=False)
    elif ext in [".xlsx", ".xls"]:
        uploaded.seek(0)
        df_table = pd.read_excel(uploaded, sheet_name=0)
        st.dataframe(df_table.head())
        content = read_excel_file(uploaded)
    elif ext in [".html", ".htm"]:
        content = read_html_file(uploaded)
        visible_text = BeautifulSoup(content, "html.parser").get_text(" ")
        urls = extract_urls_from_text(visible_text + "\n" + content)
    elif ext == ".txt":
        content = read_txt_file(uploaded)
        urls = extract_urls_from_text(content)
    elif ext == ".pdf":
        content = read_pdf_file(uploaded)
        urls = extract_urls_from_text(content)
    elif ext == ".docx":
        content = read_docx_file(uploaded)
        urls = extract_urls_from_text(content)
    elif ext == ".eml":
        content = read_eml_file(uploaded)
        urls = extract_urls_from_text(content)
    elif ext == ".msg":
        content = read_msg_file(uploaded)
        msg_reader_empty = not content.strip()
        urls = extract_urls_from_text(content)
    elif ext in IMAGE_EXTENSIONS:
        content = read_image_file(uploaded)
        image_ocr_empty = not content.strip()
        urls = extract_urls_from_text(content)

    urls = _unique_urls(urls)

    if image_ocr_empty:
        st.info("Image lue, mais aucun texte n'a ete extrait. EasyOCR doit etre installe et son modele OCR telecharge au premier lancement.")
    if msg_reader_empty:
        st.info("Fichier MSG detecte, mais la lecture necessite la dependance optionnelle extract-msg.")

    label_col = "Aucune"
    if df_table is not None:
        detected_url = detect_url_column(df_table)
        url_options = list(df_table.columns)
        url_index = url_options.index(detected_url) if detected_url in df_table.columns else 0
        url_col = st.selectbox("Colonne URL", options=url_options, index=url_index)
        urls = df_table[url_col].astype(str).tolist()
        detected_label = next((c for c in df_table.columns if str(c).lower() in {"label", "status", "result", "class"}), None)
        label_options = ["Aucune"] + list(df_table.columns)
        label_index = label_options.index(detected_label) + 1 if detected_label in df_table.columns else 0
        label_col = st.selectbox("Colonne label (si disponible)", options=label_options, index=label_index)

    show_url_shap = st.checkbox("Afficher SHAP pour l'URL la plus suspecte", value=True)
    show_html_shap = False
    if ext in [".html", ".htm"]:
        show_html_shap = st.checkbox("Afficher SHAP pour le contenu HTML", value=False)

    if st.button("Predire fichier"):
        final_label = None
        final_prob = None
        url_details = pd.DataFrame()

        if ext in [".html", ".htm"] and has_html_model:
            model_html = load_cnn_model(HTML_MODEL_PATH)
            tok_html = load_tokenizer(HTML_TOKENIZER_PATH)
            Xh = texts_to_padded_sequences([content], tok_html, MAX_LEN_HTML)
            ph = float(model_html.predict(Xh, verbose=0)[0][0])
            final_label = "Phishing" if ph >= 0.5 else "Legitime"
            final_prob = ph

        if has_url_model and urls:
            model_url = load_cnn_model(URL_MODEL_PATH)
            tok_url = load_tokenizer(URL_TOKENIZER_PATH)
            url_details = _predict_urls(urls, model_url, tok_url)
            pct_phish = float((url_details["prediction"] == "Phishing").mean())
            max_prob = float(url_details["probabilite_phishing"].max())

            st.write(f"URLs trouvees: {len(urls)}")
            st.write(f"Taux URLs phishing: {pct_phish:.2%}")
            st.dataframe(url_details)
            st.download_button(
                "Telecharger details URLs",
                data=url_details.to_csv(index=False).encode("utf-8"),
                file_name="predictions_urls_extraites.csv",
                mime="text/csv",
            )

            if final_label is None:
                final_label = "Phishing" if pct_phish >= 0.5 or max_prob >= 0.85 else "Legitime"
                final_prob = max_prob

            if label_col != "Aucune" and df_table is not None:
                try:
                    urls_for_eval = df_table[url_col].astype(str).tolist()
                    eval_mask = normalize_labels(df_table[label_col]).notna().values
                    y_series = normalize_labels(df_table[label_col])
                    y_true = y_series[eval_mask].astype(int).values
                    eval_urls = [u for i, u in enumerate(urls_for_eval) if eval_mask[i]]
                    eval_probs = _predict_urls(eval_urls, model_url, tok_url)["probabilite_phishing"].values
                    if len(y_true) == len(eval_probs) and len(y_true) > 0:
                        eval_summary = calculate_eval_summary(y_true, eval_probs)
                        st.subheader("Evaluation sur les donnees importees")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Accuracy", f"{eval_summary['accuracy']:.4f}")
                        c2.metric("Loss", f"{eval_summary['loss']:.4f}")
                        c3.metric("ROC AUC", f"{eval_summary['roc_auc']:.4f}")
                except Exception as exc:
                    st.warning(f"Impossible de calculer les metriques importees: {exc}")

            if show_url_shap:
                try:
                    suspicious_url = url_details.sort_values("probabilite_phishing", ascending=False).iloc[0]["url"]
                    shap_result = calculate_shap_values(
                        model_url,
                        tok_url,
                        [suspicious_url],
                        MAX_LEN_URL,
                        max_features=80,
                        background_texts=urls[:5],
                        nsamples=80,
                    )
                    st.pyplot(plot_shap_bar(shap_result, title="Explication SHAP de l'URL analysee", top_k=15))
                    st.write("Caracteristiques les plus influentes de l'URL")
                    st.dataframe(get_top_shap_features(shap_result, top_k=15, row_index=0))
                except Exception as exc:
                    st.warning(f"SHAP indisponible pour les URLs: {exc}")

        if show_html_shap and ext in [".html", ".htm"] and has_html_model and content.strip():
            try:
                if "model_html" not in locals():
                    model_html = load_cnn_model(HTML_MODEL_PATH)
                    tok_html = load_tokenizer(HTML_TOKENIZER_PATH)
                shap_result = calculate_shap_values(
                    model_html,
                    tok_html,
                    [content],
                    MAX_LEN_HTML,
                    max_features=120,
                    nsamples=80,
                )
                st.pyplot(plot_shap_bar(shap_result, title="Explication SHAP du contenu HTML", top_k=20))
                st.write("Caracteristiques les plus influentes du HTML")
                st.dataframe(get_top_shap_features(shap_result, top_k=20, row_index=0))
            except Exception as exc:
                st.warning(f"SHAP indisponible pour le HTML: {exc}")

        if final_label is None:
            if not has_url_model:
                st.warning("Modele URL manquant. Lancez d'abord l'entrainement URL depuis Accueil.")
            elif not urls:
                st.warning("Aucune URL exploitable trouvee dans ce fichier.")
            else:
                st.warning("Aucun modele disponible ou aucune information exploitable.")
        else:
            st.success(f"Resultat final: {final_label}")
            st.write(f"Score phishing: {final_prob:.4f}")
            out = pd.DataFrame(
                [
                    {
                        "fichier": uploaded.name,
                        "type_fichier": ext,
                        "prediction": final_label,
                        "score_phishing": final_prob,
                        "urls_detectees": len(urls),
                        "urls_phishing": int((url_details["prediction"] == "Phishing").sum()) if not url_details.empty else 0,
                    }
                ]
            )
            out.to_csv(PREDICTIONS_HTML_CSV, mode="a", header=not os.path.exists(PREDICTIONS_HTML_CSV), index=False)
            st.dataframe(out)
