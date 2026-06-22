# JUDGELYTICS - ML Pipeline: Model Comparison Module
# Purpose: Compare performance of all trained models
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Comprehensive model comparison and performance analysis.

Evaluates all trained models on the test set and provides side-by-side comparison.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
import time
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .evaluate_classifier import evaluate_classifier
from .evaluate_regressor import evaluate_regressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(
    feature_matrices_path: str = "data/processed/feature_matrices.npz",
    labels_path: str = "data/processed/labels.npz"
) -> tuple:
    """
    Load test feature matrices and labels.

    Args:
        feature_matrices_path (str): Path to feature matrices
        labels_path (str): Path to labels

    Returns:
        tuple: (X_test, X_test_tfidf, y_test)
    """

    logger.info("Loading test data...")

    # Load generated test split matrix.
    feature_dir = os.path.dirname(feature_matrices_path) or "data/processed"
    x_test_path = os.path.join(feature_dir, "X_test_combined.npz")
    X_test = sparse.load_npz(x_test_path).tocsr()
    X_test_tfidf = X_test[:, :15000]  # First 15000 columns are TF-IDF

    labels = np.load(labels_path)

    y_test = labels["y_test"]

    logger.info(f"Test data loaded: features shape={X_test.shape}, labels shape={y_test.shape}")

    return X_test, X_test_tfidf, y_test


def load_models(models_dir: str = "saved_models") -> dict:
    """
    Load all trained models.

    Args:
        models_dir (str): Directory containing saved models

    Returns:
        dict: Dictionary with model names and loaded models
    """

    logger.info(f"Loading models from {models_dir}...")

    models = {}

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "XGBoost Regressor": "xgboost_regressor.pkl"
    }

    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                logger.info(f"  ✓ {name} loaded")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {name}: {str(e)}")

    # Try to load LSTM model
    lstm_path = os.path.join(models_dir, "lstm_model.keras")
    if os.path.exists(lstm_path):
        try:
            from tensorflow import keras
            models["Bi-LSTM"] = keras.models.load_model(lstm_path)
            logger.info(f"  ✓ Bi-LSTM loaded")
        except Exception as e:
            logger.warning(f"  ✗ Failed to load Bi-LSTM: {str(e)}")

    logger.info(f"Loaded {len(models)} models")
    return models


def compare_models(
    X_test,
    X_test_tfidf,
    y_test,
    models_dir: str = "saved_models"
) -> pd.DataFrame:
    """
    Compare all models on test set and create comparison table.

    Args:
        X_test: Combined feature matrix
        X_test_tfidf: TF-IDF feature matrix
        y_test: Test labels
        models_dir (str): Directory with saved models

    Returns:
        pd.DataFrame: Comparison table
    """

    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON ON TEST SET")
    logger.info("="*80)

    models = load_models(models_dir)

    results = []

    # Evaluate each model
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")

        try:
            start_time = time.time()

            # Make predictions based on model type
            if model_name == "Naive Bayes":
                # Naive Bayes uses TF-IDF only
                y_pred = model.predict(X_test_tfidf)
            elif model_name == "Bi-LSTM":
                # LSTM requires tokenized and padded text
                logger.warning(f"  Skipping {model_name} - requires tokenized input")
                continue
            elif model_name == "XGBoost Regressor":
                # Regressor is trained on structured features only (last 8 columns)
                y_pred = model.predict(X_test[:, -8:].toarray())
            else:
                # Other models use combined features
                y_pred = model.predict(X_test)

            inference_time = time.time() - start_time

            # Calculate metrics
            if model_name != "XGBoost Regressor":
                # Classification metrics
                metrics = evaluate_classifier(y_test, y_pred, model_name)

                result = {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Macro": metrics["f1"],
                    "Inference Time (s)": inference_time
                }
            else:
                # Regression metrics
                metrics = evaluate_regressor(y_test, y_pred, model_name)

                result = {
                    "Model": model_name,
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"],
                    "R²": metrics["r2"],
                    "MAPE": metrics["mape"],
                    "Inference Time (s)": inference_time
                }

            results.append(result)

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            continue

    # Create comparison table
    if results:
        comparison_df = pd.DataFrame(results)
        logger.info("\n" + "="*80)
        logger.info("COMPARISON TABLE")
        logger.info("="*80)
        logger.info("\n" + comparison_df.to_string(index=False))

        # Find and recommend best model
        logger.info("\n" + "="*80)
        logger.info("MODEL RECOMMENDATIONS")
        logger.info("="*80)

        # Best classifier
        classifiers = comparison_df[comparison_df["Model"] != "XGBoost Regressor"]
        if not classifiers.empty:
            best_classifier_idx = classifiers["F1-Macro"].idxmax()
            best_classifier = comparison_df.loc[best_classifier_idx]
            logger.info(f"\nBest Classifier: {best_classifier['Model']}")
            logger.info(f"  F1-Macro Score: {best_classifier['F1-Macro']:.4f}")
            logger.info(f"  Accuracy: {best_classifier['Accuracy']:.4f}")
            logger.info(f"  Inference Time: {best_classifier['Inference Time (s)']:.4f}s")

        # Best regressor
        regressors = comparison_df[comparison_df["Model"] == "XGBoost Regressor"]
        if not regressors.empty:
            best_regressor = regressors.iloc[0]
            logger.info(f"\nBest Regressor: {best_regressor['Model']}")
            logger.info(f"  R² Score: {best_regressor['R²']:.4f}")
            logger.info(f"  RMSE: {best_regressor['RMSE']:.4f}")
            logger.info(f"  Inference Time: {best_regressor['Inference Time (s)']:.4f}s")

        return comparison_df

    else:
        logger.warning("No valid results to compare")
        return pd.DataFrame()


def main(
    feature_matrices_path: str = "data/processed/feature_matrices.npz",
    labels_path: str = "data/processed/labels.npz",
    models_dir: str = "saved_models"
) -> pd.DataFrame:
    """
    Main entry point for model comparison.

    Args:
        feature_matrices_path (str): Path to feature matrices
        labels_path (str): Path to labels
        models_dir (str): Directory with saved models

    Returns:
        pd.DataFrame: Comparison table
    """

    try:
        # Load test data
        X_test, X_test_tfidf, y_test = load_test_data(feature_matrices_path, labels_path)

        # Compare models
        comparison_df = compare_models(X_test, X_test_tfidf, y_test, models_dir)

        return comparison_df

    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        comparison_df = main()
        logger.info("Model comparison completed successfully!")
    except Exception as e:
        logger.error(f"Failed to compare models: {str(e)}")
        exit(1)
