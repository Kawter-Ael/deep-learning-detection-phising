import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["y_prob_preview"] = [float(v) for v in np.array(y_prob).flatten()[:10]]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["loss"] = float(log_loss(y_true, np.clip(np.array(y_prob).astype(float), 1e-7, 1 - 1e-7)))
        except ValueError:
            metrics["loss"] = None
    return metrics


def calculate_eval_summary(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "predictions": y_pred,
    }


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
