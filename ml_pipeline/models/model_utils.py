# JUDGELYTICS - ML Pipeline: Model Utilities Module
# Purpose: Helper functions for model save/load operations
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Utility functions for model persistence.

Handles saving and loading of trained models using joblib and Keras.
"""

import logging
import joblib
import os
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_model(model: Any, filepath: str) -> bool:
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained model object
        filepath (str): Path to save the model

    Returns:
        bool: True if successful, False otherwise
    """

    try:
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, filepath)
        logger.info(f"Model saved to: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {str(e)}")
        return False


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.

    Args:
        filepath (str): Path to the saved model

    Returns:
        Any: Loaded model object

    Raises:
        FileNotFoundError: If model file not found
        Exception: If loading fails
    """

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {str(e)}")
        raise


def save_keras_model(model, filepath: str) -> bool:
    """
    Save a Keras/TensorFlow model to disk.

    Args:
        model: Keras model object
        filepath (str): Path to save the model

    Returns:
        bool: True if successful, False otherwise
    """

    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        model.save(filepath)
        logger.info(f"Keras model saved to: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save Keras model to {filepath}: {str(e)}")
        return False


def load_keras_model(filepath: str):
    """
    Load a Keras/TensorFlow model from disk.

    Args:
        filepath (str): Path to the saved model

    Returns:
        Keras model object

    Raises:
        FileNotFoundError: If model file not found
        Exception: If loading fails
    """

    try:
        from tensorflow import keras

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Keras model file not found: {filepath}")

        model = keras.models.load_model(filepath)
        logger.info(f"Keras model loaded from: {filepath}")
        return model

    except Exception as e:
        logger.error(f"Failed to load Keras model from {filepath}: {str(e)}")
        raise


def get_model_info(model: Any) -> dict:
    """
    Extract basic information about a model.

    Args:
        model: Model object

    Returns:
        dict: Model information
    """

    info = {
        "type": type(model).__name__,
        "params": getattr(model, "get_params", lambda: {})()
    }

    return info
