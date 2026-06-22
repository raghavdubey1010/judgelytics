# JUDGELYTICS - ML Pipeline: Regressor Evaluation Module
# Purpose: Evaluate regression models
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Comprehensive evaluation metrics for regression models.

Computes RMSE, MAE, R² score, and prediction analysis.
"""

import logging
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_regressor(
    y_true,
    y_pred,
    model_name: str = "Model"
) -> dict:
    """
    Evaluate regression model performance.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name (str): Name of model for logging

    Returns:
        dict: Dictionary with evaluation metrics
    """

    logger.info(f"\n{'='*60}")
    logger.info(f"Regression Evaluation: {model_name}")
    logger.info(f"{'='*60}")

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Log metrics
    logger.info(f"\nMetrics:")
    logger.info(f"  RMSE:  {rmse:.4f}")
    logger.info(f"  MAE:   {mae:.4f}")
    logger.info(f"  R²:    {r2:.4f}")
    logger.info(f"  MAPE:  {mape:.4f}")

    # Prediction analysis
    residuals = y_true - y_pred

    logger.info(f"\nPrediction Analysis:")
    logger.info(f"  Min error: {residuals.min():.4f}")
    logger.info(f"  Max error: {residuals.max():.4f}")
    logger.info(f"  Mean error: {residuals.mean():.4f}")
    logger.info(f"  Std error: {residuals.std():.4f}")

    # Distribution
    logger.info(f"\nPrediction Range:")
    logger.info(f"  Min prediction: {y_pred.min():.4f}")
    logger.info(f"  Max prediction: {y_pred.max():.4f}")
    logger.info(f"  Mean prediction: {y_pred.mean():.4f}")
    logger.info(f"  Std prediction: {y_pred.std():.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "residuals": residuals
    }


def plot_predictions(
    y_true,
    y_pred,
    model_name: str = "Model",
    save_path: str = None
):
    """
    Plot predictions vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name (str): Name for title
        save_path (str): Path to save figure (optional)
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    axes[0].set_xlabel("True Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title(f"{model_name} - Predictions vs Actual")
    axes[0].grid(True, alpha=0.3)

    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"{model_name} - Residuals")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Prediction plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    logger.info("Regressor evaluation module")
