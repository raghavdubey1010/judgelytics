# JUDGELYTICS - ML Pipeline: XGBoost Regressor Training Module
# Purpose: Train XGBoost regressor to predict win probability (0.0-1.0)
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
XGBoost regressor model training for predicting case win probability.

Win probability derived from outcome labels with bonuses based on
evidence quality and case characteristics.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from .model_utils import save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def derive_win_probability(
    labels,
    tier2_scores,
    has_legal_notices,
    claim_buckets
) -> np.ndarray:
    """
    Derive continuous win probability from outcome labels.

    Win probability derivation:
    - Allowed: base 0.75 + bonuses (up to 0.95)
    - Partially Allowed: base 0.50 + bonuses (up to 0.75)
    - Dismissed: base 0.20 - deductions (down to 0.05)

    Bonuses applied per factor:
    - Tier 2 score > 6: +0.05
    - Has legal notice: +0.05
    - Claim in District/State bucket: +0.05

    Args:
        labels: Outcome labels (0=Allowed, 1=Dismissed, 2=Partially Allowed)
        tier2_scores: Tier 2 keyword match scores
        has_legal_notices: Bool array for legal notice presence
        claim_buckets: Claim jurisdiction buckets

    Returns:
        np.ndarray: Win probability values (0.0-1.0)
    """

    win_probs = np.zeros(len(labels))

    logger.info("Deriving win probabilities from labels...")

    for i in range(len(labels)):
        base_prob = 0.5

        # Set base probability by outcome
        if labels[i] == 0:  # Allowed
            base_prob = 0.75
        elif labels[i] == 1:  # Dismissed
            base_prob = 0.20
        elif labels[i] == 2:  # Partially Allowed
            base_prob = 0.50

        prob = base_prob

        # Apply bonuses
        if tier2_scores[i] > 6:
            prob += 0.05
        if has_legal_notices[i]:
            prob += 0.05
        if claim_buckets[i] in ["District", "State"]:
            prob += 0.05

        # Apply caps
        if labels[i] == 0:  # Allowed - cap at 0.95
            prob = min(prob, 0.95)
        elif labels[i] == 2:  # Partially Allowed - cap at 0.75
            prob = min(prob, 0.75)
        else:  # Dismissed - floor at 0.05
            prob = max(prob, 0.05)

        win_probs[i] = prob

    logger.info(f"Win probabilities derived:")
    logger.info(f"  Min: {win_probs.min():.4f}, Max: {win_probs.max():.4f}")
    logger.info(f"  Mean: {win_probs.mean():.4f}, Std: {win_probs.std():.4f}")

    return win_probs


def train_xgboost_regressor(
    X_train,
    y_train,
    X_val,
    y_val,
    output_dir: str = "saved_models"
) -> xgb.XGBRegressor:
    """
    Train and tune XGBoost regressor for win probability prediction.

    Args:
        X_train: Training feature matrix (dense, structured features only)
        y_train: Training win probability labels
        X_val: Validation feature matrix
        y_val: Validation win probability labels
        output_dir (str): Directory to save model. Default: "saved_models"

    Returns:
        xgb.XGBRegressor: Trained and tuned model
    """

    logger.info("Starting XGBoost Regressor training...")
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Validation set size: {X_val.shape}")

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1]
    }

    logger.info("Hyperparameter grid:")
    for key, values in param_grid.items():
        logger.info(f"  {key}: {values}")

    # Initialize base model
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0
    )

    # Grid search with cross-validation
    logger.info("Running GridSearchCV with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Log best parameters and score
    logger.info(f"\nBest parameters found:")
    for key, value in grid_search.best_params_.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"Best cross-validation R² score: {grid_search.best_score_:.4f}")

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    logger.info("\nValidation Set Performance:")
    y_val_pred = best_model.predict(X_val)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    logger.info(f"  RMSE: {val_rmse:.4f}")
    logger.info(f"  MAE:  {val_mae:.4f}")
    logger.info(f"  R²:   {val_r2:.4f}")

    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost_regressor.pkl")
    save_model(best_model, model_path)

    logger.info(f"XGBoost Regressor training completed!")

    return best_model


def main(
    preprocessed_path: str = "data/processed/consumer_cases_preprocessed.csv",
    labels_path: str = "data/processed/labels.npz",
    output_dir: str = "saved_models"
) -> xgb.XGBRegressor:
    """
    Main entry point for training XGBoost regressor.

    Args:
        preprocessed_path (str): Path to preprocessed data
        labels_path (str): Path to labels file
        output_dir (str): Output directory for saved model

    Returns:
        xgb.XGBRegressor: Trained model
    """

    try:
        logger.info(f"Loading labels from {labels_path}")
        labels = np.load(labels_path, allow_pickle=True)

        y_train_labels = labels["y_train"]
        y_val_labels = labels["y_val"]

        # Load combined matrices and use structured feature slice (last 8 columns).
        feature_dir = os.path.dirname(preprocessed_path).replace("\\", "/") or "data/processed"
        X_train_path = os.path.join(feature_dir, "X_train_combined.npz")
        X_val_path = os.path.join(feature_dir, "X_val_combined.npz")

        X_train_combined = sparse.load_npz(X_train_path).tocsr()
        X_val_combined = sparse.load_npz(X_val_path).tocsr()

        X_train = X_train_combined[:, -8:].toarray()
        X_val = X_val_combined[:, -8:].toarray()

        # Derive target probabilities from labels with neutral defaults for metadata bonuses.
        y_train = derive_win_probability(
            y_train_labels,
            np.zeros(len(y_train_labels)),
            np.zeros(len(y_train_labels), dtype=bool),
            np.array(["Unknown"] * len(y_train_labels), dtype=object)
        )

        y_val = derive_win_probability(
            y_val_labels,
            np.zeros(len(y_val_labels)),
            np.zeros(len(y_val_labels), dtype=bool),
            np.array(["Unknown"] * len(y_val_labels), dtype=object)
        )

        logger.info(f"Feature matrices: Train shape={X_train.shape}, Val shape={X_val.shape}")

        # Train model
        model = train_xgboost_regressor(X_train, y_train, X_val, y_val, output_dir)

        return model

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        model = main()
        logger.info("XGBoost Regressor training completed successfully!")
    except Exception as e:
        logger.error(f"Failed to train XGBoost Regressor: {str(e)}")
        exit(1)
