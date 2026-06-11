import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

URL_CANDIDATES = ["url", "URL", "link", "website", "domain"]
LABEL_CANDIDATES = ["label", "status", "result", "class"]

POSITIVE_LABELS = {"phishing", "phish", "bad", "malicious", "1", "true", "yes"}
NEGATIVE_LABELS = {"legitimate", "legit", "notphish", "not_phish", "good", "benign", "safe", "0", "false", "no"}


def detect_url_column(df: pd.DataFrame):
    for col in df.columns:
        if str(col).lower() in {c.lower() for c in URL_CANDIDATES}:
            return col
    return None


def detect_label_column(df: pd.DataFrame):
    for col in df.columns:
        if str(col).lower() in {c.lower() for c in LABEL_CANDIDATES}:
            return col
    return None


def normalize_labels(series: pd.Series):
    def _map(v):
        s = str(v).strip().lower()
        if s in POSITIVE_LABELS:
            return 1
        if s in NEGATIVE_LABELS:
            return 0
        return None

    return series.apply(_map)


def clean_text(text):
    if text is None:
        return ""
    return str(text).replace("\x00", " ").strip()


def extract_urls_from_text(text):
    text = clean_text(text)
    pattern = r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+"
    return re.findall(pattern, text)


def create_and_fit_tokenizer(texts, max_words, char_level=True):
    tokenizer = Tokenizer(num_words=max_words, char_level=char_level, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts([clean_text(t) for t in texts])
    return tokenizer


def texts_to_padded_sequences(texts, tokenizer, max_len, batch_size=500):
    """Convert texts to padded sequences in batches to avoid MemoryError."""
    all_cleaned = [clean_text(t) for t in texts]
    n = len(all_cleaned)
    result = np.zeros((n, max_len), dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = all_cleaned[start:end]
        seq = tokenizer.texts_to_sequences(batch)
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        result[start:end] = padded
    return result
