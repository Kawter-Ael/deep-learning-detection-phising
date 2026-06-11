import os

# Racine du projet = dossier parent de "pages/"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemins absolus
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PGHTML_DIR = os.path.join(RAW_DATA_DIR, "pghtml")

# Fichiers modèles
URL_MODEL_PATH = os.path.join(MODELS_DIR, "url_cnn_model.keras")
URL_TOKENIZER_PATH = os.path.join(MODELS_DIR, "url_tokenizer.pkl")
URL_METRICS_PATH = os.path.join(MODELS_DIR, "url_metrics.json")
HTML_MODEL_PATH = os.path.join(MODELS_DIR, "html_cnn_model.keras")
HTML_TOKENIZER_PATH = os.path.join(MODELS_DIR, "html_tokenizer.pkl")
HTML_METRICS_PATH = os.path.join(MODELS_DIR, "html_metrics.json")

# Datasets
URLS_DATASET_PATH = os.path.join(RAW_DATA_DIR, "urls_dataset.csv")

# Outputs
PREDICTIONS_URLS_CSV = os.path.join(OUTPUTS_DIR, "predictions_urls.csv")
PREDICTIONS_HTML_CSV = os.path.join(OUTPUTS_DIR, "predictions_html.csv")

# S'assurer que les dossiers existent
for d in [MODELS_DIR, OUTPUTS_DIR, RAW_DATA_DIR]:
    os.makedirs(d, exist_ok=True)
