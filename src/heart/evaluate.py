"""Metrics and evaluation plots for a binary classifier.

The positive class (1) is always "disease", so:
  * sensitivity / recall = share of disease cases the model catches
  * specificity          = share of healthy cases the model correctly clears
  * a false negative     = a disease case the model missed (the costly error)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from heart import config


def specificity_score(y_true, y_pred) -> float:
    """TN / (TN + FP). scikit-learn has no built-in for this metric."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """All metrics required by the project, at a given decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_true, y_proba, target_sensitivity: float = config.TARGET_SENSITIVITY) -> float:
    """Highest threshold that still reaches the target sensitivity.

    Lowering the threshold flags more people as "disease": sensitivity goes
    up, specificity goes down. Taking the *highest* threshold that meets the
    sensitivity target gives the best specificity we can get at that target.
    This must be called on training (out-of-fold) predictions, never on the
    test set.
    """
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    for t in np.round(np.arange(0.95, 0.04, -0.01), 2):
        if recall_score(y_true, (y_proba >= t).astype(int)) >= target_sensitivity:
            return float(t)
    return 0.05


def plot_confusion(y_true, y_proba, threshold: float, title: str, path):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1],
        display_labels=[config.NEGATIVE_CLASS_NAME, config.POSITIVE_CLASS_NAME],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_curves(curves: dict[str, tuple], path, title="ROC curves"):
    """curves: {name: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_threshold_tradeoff(y_true, y_proba, chosen: float, path):
    ts = np.linspace(0.05, 0.95, 91)
    sens = [recall_score(y_true, (y_proba >= t).astype(int)) for t in ts]
    spec = [specificity_score(y_true, (y_proba >= t).astype(int)) for t in ts]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(ts, sens, label="Sensitivity (recall on disease)")
    ax.plot(ts, spec, label="Specificity")
    ax.axvline(chosen, color="black", ls="--", lw=1, label=f"Chosen threshold = {chosen:.2f}")
    ax.axhline(config.TARGET_SENSITIVITY, color="grey", ls=":", lw=1)
    ax.set_xlabel("Decision threshold on predicted probability")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity vs specificity (out-of-fold, training data)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
