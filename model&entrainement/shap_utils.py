import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.preprocessing import texts_to_padded_sequences


def _token_name(tokenizer, token_id):
    token_id = int(token_id)
    if token_id == 0:
        return "PAD"

    token = tokenizer.index_word.get(token_id, f"id_{token_id}")
    if token == " ":
        return "espace"
    if token == "\n":
        return "\\n"
    if token == "\t":
        return "\\t"
    return str(token)


def _readable_label(position, token):
    if token == "PAD":
        return "PAD"
    if token == "espace":
        shown = "espace"
    else:
        shown = f'"{token}"'
    return f"Position {position} : {shown}"


def _normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    return shap_values


def _clean_sequence_features(values, encoded_values, tokenizer, top_k=20, row_index=None):
    values = np.array(values)
    encoded_values = np.array(encoded_values)

    if values.ndim == 1:
        values = values.reshape(1, -1)
    if encoded_values.ndim == 1:
        encoded_values = encoded_values.reshape(1, -1)

    rows = [row_index] if row_index is not None else range(values.shape[0])
    rows = [r for r in rows if r < values.shape[0]]
    if not rows:
        return pd.DataFrame(columns=["position", "caractere", "feature", "shap_value", "importance"])

    rows_values = values[rows]
    rows_encoded = encoded_values[rows]

    # PAD correspond au token 0 ajoute par pad_sequences. On le retire car il ne
    # represente aucun caractere reel de l'URL ou du HTML analyse.
    non_pad_mask = np.any(rows_encoded != 0, axis=0)
    if not np.any(non_pad_mask):
        return pd.DataFrame(columns=["position", "caractere", "feature", "shap_value", "importance"])

    mean_values = np.mean(rows_values, axis=0)
    mean_abs = np.mean(np.abs(rows_values), axis=0)
    candidate_positions = np.where(non_pad_mask)[0]
    order = candidate_positions[np.argsort(mean_abs[candidate_positions])[::-1]]
    selected = order[:top_k]

    rows_out = []
    for pos in selected:
        token_ids = rows_encoded[:, pos]
        token_ids = token_ids[token_ids != 0]
        if len(token_ids) == 0:
            continue

        # On reconvertit le token numerique vers son caractere pour que SHAP soit
        # lisible par l'utilisateur au lieu d'afficher un id ou un libelle brut.
        token = _token_name(tokenizer, token_ids[0])
        if token == "PAD":
            continue
        rows_out.append(
            {
                "position": int(pos),
                "caractere": token,
                "feature": _readable_label(pos, token),
                "shap_value": float(mean_values[pos]),
                "importance": float(mean_abs[pos]),
            }
        )

    return pd.DataFrame(rows_out)


def calculate_shap_values(model, tokenizer, texts, max_len, max_features=80, background_texts=None, nsamples=80):
    import shap

    clean_texts = [str(t) for t in texts if str(t).strip()]
    if not clean_texts:
        raise ValueError("Aucun texte valide pour SHAP.")

    X_full = texts_to_padded_sequences(clean_texts, tokenizer, max_len)
    feature_count = min(max_features, max_len)
    X_partial = X_full[:, :feature_count]

    if background_texts:
        bg_texts = [str(t) for t in background_texts if str(t).strip()][:5]
        background_full = texts_to_padded_sequences(bg_texts, tokenizer, max_len) if bg_texts else np.zeros((1, max_len))
        background = background_full[:, :feature_count]
    else:
        background = np.zeros((1, feature_count), dtype=np.int32)

    template = np.zeros(max_len, dtype=np.int32)

    def predict_from_partial(partial_sequences):
        partial_sequences = np.rint(partial_sequences).astype(np.int32)
        full_sequences = np.tile(template, (partial_sequences.shape[0], 1))
        full_sequences[:, :feature_count] = partial_sequences
        return model.predict(full_sequences, verbose=0).reshape(-1)

    explainer = shap.KernelExplainer(predict_from_partial, background)
    shap_values = explainer.shap_values(X_partial, nsamples=nsamples)
    shap_values = _normalize_shap_values(shap_values)

    return {
        "values": shap_values,
        "encoded_values": X_partial,
        "tokenizer": tokenizer,
    }


def get_top_shap_features(shap_result, top_k=20, row_index=None):
    return _clean_sequence_features(
        shap_result["values"],
        shap_result["encoded_values"],
        shap_result["tokenizer"],
        top_k=top_k,
        row_index=row_index,
    )


def plot_shap_bar(shap_result, title="Explication SHAP de l'URL analysee", top_k=20, top_n=None):
    if top_n is not None:
        top_k = top_n

    features = get_top_shap_features(shap_result, top_k=top_k, row_index=0)
    if features.empty:
        raise ValueError("Aucune caracteristique non-PAD disponible pour SHAP.")

    features = features.sort_values("shap_value")
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in features["shap_value"]]

    # top_k limite l'affichage aux caracteres vraiment influents pour garder un
    # graphe lisible dans Streamlit.
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(features))))
    ax.barh(features["feature"], features["shap_value"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Valeur SHAP : impact sur le score phishing", fontsize=11)
    ax.set_ylabel("Caracteristiques", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    return fig


def plot_shap_summary(
    shap_result,
    title="Influence globale des caracteristiques sur la detection phishing",
    top_k=20,
    top_n=None,
):
    if top_n is not None:
        top_k = top_n

    values = np.array(shap_result["values"])
    encoded = np.array(shap_result["encoded_values"])
    features = get_top_shap_features(shap_result, top_k=top_k)
    if features.empty:
        raise ValueError("Aucune caracteristique non-PAD disponible pour SHAP.")

    positions = features["position"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(positions))))
    scatter = None
    for row, feature_idx in enumerate(positions):
        x = values[:, feature_idx]
        y = np.full_like(x, row, dtype=float)
        color_values = encoded[:, feature_idx]
        scatter = ax.scatter(x, y, c=color_values, cmap="cool", s=34, alpha=0.9)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(features["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Valeur SHAP : impact sur le score phishing", fontsize=11)
    ax.set_ylabel("Caracteristiques", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Valeur encodee du caractere")
    fig.tight_layout()
    return fig
