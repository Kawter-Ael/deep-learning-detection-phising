import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predit")
    ax.set_ylabel("Reel")
    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    return fig


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history.get("accuracy", []), label="Train")
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="Train")
    axes[1].plot(history.history.get("val_loss", []), label="Validation")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_accuracy_history(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history.get("accuracy", []), marker="o", label="Train")
    ax.plot(history.history.get("val_accuracy", []), marker="o", label="Validation")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_loss_history(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history.get("loss", []), marker="o", label="Train")
    ax.plot(history.history.get("val_loss", []), marker="o", label="Validation")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
    ax.set_title("Courbe ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metrics_bar(metrics, title="Metriques du modele"):
    labels = []
    values = []
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        value = metrics.get(key)
        if value is not None:
            labels.append(key.upper() if key != "roc_auc" else "ROC AUC")
            values.append(float(value))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"][: len(labels)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig
