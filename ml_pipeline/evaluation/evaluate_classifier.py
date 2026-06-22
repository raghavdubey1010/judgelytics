# JUDGELYTICS - ML Pipeline: Classifier Evaluation Module
# Purpose: Evaluate classification models
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Comprehensive evaluation metrics for classification models.

Computes accuracy, precision, recall, F1-score, and confusion matrices.
"""

import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true,
    y_pred,
    model_name: str = "Model"
) -> dict:
    """
    Evaluate classification model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of model for logging

    Returns:
        dict: Dictionary with evaluation metrics
    """

    logger.info(f"\n{'='*60}")
    logger.info(f"Classification Evaluation: {model_name}")
    logger.info(f"{'='*60}")

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Log metrics
    logger.info(f"\nMetrics (Macro Average):")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"{cm}")

    # Per-class metrics
    logger.info(f"\nPer-Class Metrics:")
    class_names = ["Allowed", "Dismissed", "Partially Allowed"]
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, class_name in enumerate(class_names[:len(precisions)]):
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {precisions[i]:.4f}, Recall: {recalls[i]:.4f}, F1: {f1s[i]:.4f}")

    # Detailed report
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "Model",
    save_path: str = None
):
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name for title
        save_path (str): Path to save figure (optional)
    """

    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Allowed", "Dismissed", "Partially Allowed"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    logger.info("Classifier evaluation module")
