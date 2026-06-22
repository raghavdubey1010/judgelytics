# JUDGELYTICS - ML Pipeline: Naive Bayes Training Module
# Purpose: Train Naive Bayes classifier for outcome prediction
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Naive Bayes model training for consumer case outcome prediction.

Uses TF-IDF features only (MultinomialNB requires non-negative inputs).
Includes hyperparameter tuning and comprehensive evaluation.
"""

import logging
import numpy as np
import joblib
import os
from pathlib import Path
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from .model_utils import save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_naive_bayes(
    X_train,
    y_train,
    X_val,
    y_val,
    output_dir: str = "saved_models"
) -> MultinomialNB:
    """
    Train and tune Multinomial Naive Bayes classifier.

    Uses GridSearchCV for alpha hyperparameter optimization with 5-fold cross-validation.

    Args:
        X_train: Training TF-IDF feature matrix (sparse)
        y_train: Training labels
        X_val: Validation TF-IDF feature matrix
        y_val: Validation labels
        output_dir (str): Directory to save model. Default: "saved_models"

    Returns:
        MultinomialNB: Trained and tuned model
    """

    logger.info("Starting Naive Bayes training...")
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Validation set size: {X_val.shape}")

    # Hyperparameter grid for alpha
    param_grid = {
        "alpha": [0.01, 0.1, 0.5, 1.0]
    }

    logger.info("Hyperparameter grid:")
    for key, values in param_grid.items():
        logger.info(f"  {key}: {values}")

    # Initialize base model
    base_model = MultinomialNB()

    # Grid search with cross-validation
    logger.info("Running GridSearchCV with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Log best parameters and score
    logger.info(f"\nBest parameters found:")
    for key, value in grid_search.best_params_.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"Best cross-validation F1-Macro score: {grid_search.best_score_:.4f}")

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    logger.info("\nValidation Set Performance:")
    y_val_pred = best_model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average="macro", zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average="macro", zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

    logger.info(f"  Accuracy:  {val_accuracy:.4f}")
    logger.info(f"  Precision: {val_precision:.4f}")
    logger.info(f"  Recall:    {val_recall:.4f}")
    logger.info(f"  F1-Macro:  {val_f1:.4f}")

    logger.info("\nDetailed Classification Report:")
    # Determine number of classes for target names
    num_classes = len(np.unique(y_val))
    if num_classes == 2:
        target_names = ["Allowed", "Dismissed"]
    else:
        target_names = ["Allowed", "Dismissed", "Partially Allowed"]
    logger.info(classification_report(y_val, y_val_pred, target_names=target_names[:num_classes]))

    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(output_dir, "naive_bayes.pkl")
    save_model(best_model, model_path)

    logger.info(f"Naive Bayes training completed!")

    return best_model


def main(
    feature_matrices_path: str = "data/processed/feature_matrices.npz",
    labels_path: str = "data/processed/labels.npz",
    output_dir: str = "saved_models"
) -> MultinomialNB:
    """
    Main entry point for training Naive Bayes.

    Args:
        feature_matrices_path (str): Path to feature matrices file
        labels_path (str): Path to labels file
        output_dir (str): Output directory for saved model

    Returns:
        MultinomialNB: Trained model
    """

    try:
        # Load feature matrices
        logger.info(f"Loading features...")
        
        # Load individual sparse matrices
        feature_dir = os.path.dirname(feature_matrices_path) or "data/processed"
        
        X_train_path = os.path.join(feature_dir, "X_train_combined.npz")
        X_val_path = os.path.join(feature_dir, "X_val_combined.npz")
        
        X_train_combined = sparse.load_npz(X_train_path).tocsr()  # Convert to CSR for slicing
        X_val_combined = sparse.load_npz(X_val_path).tocsr()  # Convert to CSR for slicing

        logger.info(f"Loading labels from {labels_path}")
        labels_data = np.load(labels_path, allow_pickle=True)

        # For Naive Bayes, use only TF-IDF features (first 15000 columns)
        # Assuming TF-IDF is first 15000 features, structured features are last 8
        X_train_tfidf = X_train_combined[:, :15000]
        X_val_tfidf = X_val_combined[:, :15000]

        y_train = labels_data["y_train"]
        y_val = labels_data["y_val"]

        # Train model
        model = train_naive_bayes(X_train_tfidf, y_train, X_val_tfidf, y_val, output_dir)

        return model

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        model = main()
        logger.info("Naive Bayes training completed successfully!")
    except Exception as e:
        logger.error(f"Failed to train Naive Bayes: {str(e)}")
        exit(1)
