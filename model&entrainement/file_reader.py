import os
import tempfile
from email import policy
from email.parser import BytesParser
from functools import lru_cache

import pandas as pd
from docx import Document
from pypdf import PdfReader


def read_csv_file(path_or_uploaded_file):
    return pd.read_csv(path_or_uploaded_file)


def read_excel_file(uploaded_file):
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    texts = []
    for sheet_name, df in sheets.items():
        texts.append(f"Feuille: {sheet_name}")
        texts.append(df.astype(str).fillna("").to_string(index=False))
    return "\n".join(texts)


def read_txt_file(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


def read_html_file(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


def read_pdf_file(uploaded_file):
    reader = PdfReader(uploaded_file)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def read_docx_file(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([p.text for p in doc.paragraphs])


def read_eml_file(uploaded_file):
    uploaded_file.seek(0)
    msg = BytesParser(policy=policy.default).parse(uploaded_file)
    parts = []
    for key in ["from", "to", "subject", "date"]:
        value = msg.get(key)
        if value:
            parts.append(f"{key}: {value}")

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() in ["text/plain", "text/html"]:
                try:
                    parts.append(part.get_content())
                except Exception:
                    continue
    else:
        try:
            parts.append(msg.get_content())
        except Exception:
            pass

    return "\n".join(parts)


def read_msg_file(uploaded_file):
    try:
        import extract_msg
    except ImportError:
        return ""

    try:
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".msg") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        msg = extract_msg.Message(temp_path)
        msg.sender = msg.sender or ""
        msg.to = msg.to or ""
        msg.subject = msg.subject or ""
        msg.body = msg.body or ""
        fields = [msg.sender, msg.to, msg.subject, msg.body]
        text = "\n".join([str(v) for v in fields if v])
        try:
            msg.close()
        except Exception:
            pass
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return text
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _get_easyocr_reader():
    try:
        import easyocr
    except ImportError:
        return None

    try:
        return easyocr.Reader(["en", "fr"], gpu=False)
    except Exception:
        return None


def read_image_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        reader = _get_easyocr_reader()
        if reader is None:
            return ""

        results = reader.readtext(temp_path, detail=0, paragraph=True)
        return "\n".join([str(text) for text in results])
    except Exception:
        return ""
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def _read_html_folder(folder_path, label):
    texts, labels, count = [], [], 0
    if not os.path.isdir(folder_path):
        return texts, labels, count

    for name in os.listdir(folder_path):
        if not name.lower().endswith((".html", ".htm")):
            continue
        full_path = os.path.join(folder_path, name)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                labels.append(label)
                count += 1
        except Exception:
            continue
    return texts, labels, count


def read_html_dataset_from_folders(base_dir):
    train_notphish = os.path.join(base_dir, "training", "NotPhish")
    train_phish = os.path.join(base_dir, "training", "Phish")
    val_notphish = os.path.join(base_dir, "validation", "NotPhish")
    val_phish = os.path.join(base_dir, "validation", "Phish")

    x_t0, y_t0, c_t0 = _read_html_folder(train_notphish, 0)
    x_t1, y_t1, c_t1 = _read_html_folder(train_phish, 1)
    x_v0, y_v0, c_v0 = _read_html_folder(val_notphish, 0)
    x_v1, y_v1, c_v1 = _read_html_folder(val_phish, 1)

    X_train_html = x_t0 + x_t1
    y_train_html = y_t0 + y_t1
    X_val_html = x_v0 + x_v1
    y_val_html = y_v0 + y_v1

    stats_counts = {
        "training/NotPhish": c_t0,
        "training/Phish": c_t1,
        "validation/NotPhish": c_v0,
        "validation/Phish": c_v1,
    }

    return X_train_html, y_train_html, X_val_html, y_val_html, stats_counts


def list_validation_html_files(base_dir):
    files = []
    for label_folder in ["NotPhish", "Phish"]:
        folder = os.path.join(base_dir, "validation", label_folder)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith((".html", ".htm")):
                files.append(os.path.join(folder, name))
    return sorted(files)
