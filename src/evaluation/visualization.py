import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.metrics import auc, confusion_matrix, roc_curve

logger = logging.getLogger(__name__)

CLASS_NAMES = ["No Delay", "Current", "Past Due+"]


def plot_confusion_matrix(y_true, y_pred, save_path: Optional[Path] = None):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    _save_or_show(save_path)


def plot_roc_curves(y_true, y_prob, save_path: Optional[Path] = None):
    y_true_oh = np.zeros((len(y_true), 3))
    y_true_oh[np.arange(len(y_true)), y_true] = 1

    colors = ["steelblue", "darkorange", "crimson"]
    plt.figure(figsize=(9, 7))

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{name}  (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (OVR)")
    plt.legend(loc="lower right")
    _save_or_show(save_path)


def plot_embeddings_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
    sample_size: int = 5000,
):
    """Project embedding_dim → 2D via UMAP for visual cluster inspection."""
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    logger.info(f"Computing UMAP on {len(embeddings):,} embeddings…")
    reducer = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.1)
    emb_2d  = reducer.fit_transform(embeddings)

    plt.figure(figsize=(9, 7))
    cmap   = plt.cm.get_cmap("viridis", 3)
    scatter = plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=labels, cmap=cmap, alpha=0.6, s=8, vmin=-0.5, vmax=2.5,
    )
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(CLASS_NAMES)
    plt.title("UMAP Projection of Customer Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    _save_or_show(save_path)


def plot_training_curves(
    history: dict,
    save_path: Optional[Path] = None,
):
    """
    Plot train loss, val loss, and val Macro F1 across training epochs.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'val_f1'
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: losses
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="darkorange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: val F1
    ax2.plot(epochs, history["val_f1"], label="Val Macro F1", color="mediumseagreen")
    best_epoch = int(np.argmax(history["val_f1"])) + 1
    best_f1    = max(history["val_f1"])
    ax2.axvline(best_epoch, color="crimson", linestyle="--", alpha=0.6,
                label=f"Best epoch {best_epoch} (F1={best_f1:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path)


def _save_or_show(save_path: Optional[Path]):
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
