import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, Dense


def build_url_cnn_model(max_words, max_len):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        Conv1D(filters=64, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_html_cnn_model(max_words, max_len):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        Conv1D(filters=64, kernel_size=7, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=7, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def save_tokenizer(tokenizer, path):
    joblib.dump(tokenizer, path)


def load_tokenizer(path):
    return joblib.load(path)


def load_cnn_model(path):
    return load_model(path)
