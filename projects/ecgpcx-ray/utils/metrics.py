"""Evaluation metrics for binary chest X-ray classification."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix


def compute_accuracy(y_true, y_scores, threshold=0.5):
    """Binary accuracy given sigmoid scores and a decision threshold."""
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    return accuracy_score(np.asarray(y_true), y_pred)


def compute_roc_auc(y_true, y_scores):
    """Area under the ROC curve (pneumonia = positive class)."""
    return roc_auc_score(np.asarray(y_true), np.asarray(y_scores))

def compute_pr_auc(y_true, y_scores):
    """Area under the Precision-Recall curve (pneumonia = positive class)."""
    from sklearn.metrics import average_precision_score
    return average_precision_score(np.asarray(y_true), np.asarray(y_scores))

def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve and return the figure."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_scores))
    auc = roc_auc_score(np.asarray(y_true), np.asarray(y_scores))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Pneumonia Classifier")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def compute_confusion_matrix(y_true, y_scores, threshold=0.5):
    """Return confusion matrix as [[TN, FP], [FN, TP]]."""
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true), y_pred)


def print_metrics(y_true, y_scores, threshold=0.5):
    """Print accuracy, AUC-ROC, and confusion matrix to stdout."""
    acc = compute_accuracy(y_true, y_scores, threshold)
    auc = compute_roc_auc(y_true, y_scores)
    cm = compute_confusion_matrix(y_true, y_scores, threshold)
    pr_auc = compute_pr_auc(y_true, y_scores)

    print(f"Accuracy : {acc:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Confusion matrix (threshold={threshold}):")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
