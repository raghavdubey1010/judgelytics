# JUDGELYTICS - ML Pipeline: LSTM Training Module
# Purpose: Train Bidirectional LSTM neural network for outcome prediction
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Bidirectional LSTM model training for consumer case outcome prediction.

Deep learning model with embedding layer, bidirectional LSTM cells, and dropout
for regularization. Includes early stopping and model checkpointing.
"""

import logging
import numpy as np
import random
import os
from pathlib import Path
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score

from .model_utils import save_keras_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Set random seeds for reproducibility
def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seeds set to {seed} for reproducibility")


def build_lstm_model(vocab_size: int = 20000, max_length: int = 512) -> keras.Model:
    """
    Build Bidirectional LSTM neural network.

    Architecture:
    - Embedding(20000, 128)
    - Bidirectional(LSTM(128, return_sequences=True))
    - Dropout(0.3)
    - Bidirectional(LSTM(64))
    - Dropout(0.3)
    - Dense(64, relu)
    - Dense(3, softmax)

    Args:
        vocab_size (int): Size of vocabulary. Default: 20000
        max_length (int): Maximum sequence length. Default: 512

    Returns:
        keras.Model: Compiled LSTM model
    """

    logger.info("Building Bidirectional LSTM model...")
    logger.info(f"  Vocabulary size: {vocab_size}")
    logger.info(f"  Max sequence length: {max_length}")

    model = keras.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_length),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)

    return model


def train_lstm(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 20,
    batch_size: int = 32,
    output_dir: str = "saved_models"
) -> keras.Model:
    """
    Train Bidirectional LSTM model with early stopping.

    Args:
        X_train: Training sequences (padded)
        y_train: Training labels
        X_val: Validation sequences (padded)
        y_val: Validation labels
        epochs (int): Maximum epochs. Default: 20
        batch_size (int): Batch size. Default: 32
        output_dir (str): Directory to save model. Default: "saved_models"

    Returns:
        keras.Model: Trained model
    """

    logger.info("Starting LSTM training...")
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_lstm_model()

    # Setup callbacks
    model_checkpoint_path = os.path.join(output_dir, "lstm_model.keras")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    logger.info("Callbacks configured:")
    logger.info(f"  - EarlyStopping: patience=3")
    logger.info(f"  - ModelCheckpoint: {model_checkpoint_path}")

    # Train model
    logger.info("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Training completed!")

    # Evaluate on validation set
    logger.info("\nValidation Set Performance:")
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    logger.info(f"  Loss: {val_loss:.4f}")
    logger.info(f"  Accuracy: {val_accuracy:.4f}")

    # Get predictions for detailed report
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    logger.info("\nDetailed Classification Report:")
    # Dynamic target names based on number of classes
    num_classes = len(np.unique(y_val))
    if num_classes == 2:
        target_names = ["Allowed", "Dismissed"]
    else:
        target_names = ["Allowed", "Dismissed", "Partially Allowed"]
    logger.info(classification_report(
        y_val, y_val_pred,
        target_names=target_names[:num_classes]
    ))

    return model


def tokenize_texts(
    train_texts,
    val_texts,
    test_texts,
    vocab_size: int = 20000,
    max_length: int = 512,
    output_dir: str = "saved_models"
) -> tuple:
    """
    Tokenize and pad text sequences.

    Args:
        train_texts: Training texts
        val_texts: Validation texts
        test_texts: Test texts
        vocab_size (int): Maximum vocabulary size. Default: 20000
        max_length (int): Maximum sequence length. Default: 512
        output_dir (str): Directory to save tokenizer

    Returns:
        tuple: (X_train_padded, X_val_padded, X_test_padded, tokenizer)
    """

    logger.info("Tokenizing texts...")
    logger.info(f"  Vocab size: {vocab_size}, Max length: {max_length}")

    # Create tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    # Tokenize and pad
    X_train_seq = tokenizer.texts_to_sequences(train_texts)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')

    X_val_seq = tokenizer.texts_to_sequences(val_texts)
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post')

    X_test_seq = tokenizer.texts_to_sequences(test_texts)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

    logger.info(f"Tokenization complete:")
    logger.info(f"  Train shape: {X_train_padded.shape}")
    logger.info(f"  Val shape: {X_val_padded.shape}")
    logger.info(f"  Test shape: {X_test_padded.shape}")
    logger.info(f"  Vocabulary size: {len(tokenizer.word_index)}")

    # Save tokenizer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "lstm_tokenizer.pkl")
    joblib.dump(tokenizer, tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    return X_train_padded, X_val_padded, X_test_padded, tokenizer


def main(
    clean_text_path: str = "data/processed/consumer_cases_preprocessed.csv",
    labels_path: str = "data/processed/labels.npz",
    output_dir: str = "saved_models"
) -> keras.Model:
    """
    Main entry point for training LSTM.

    Args:
        clean_text_path (str): Path to preprocessed texts
        labels_path (str): Path to labels file
        output_dir (str): Output directory for saved model

    Returns:
        keras.Model: Trained model
    """

    import pandas as pd

    try:
        # Set random seeds
        set_random_seeds(42)

        # Load data
        logger.info(f"Loading preprocessed texts from {clean_text_path}")
        df = pd.read_csv(clean_text_path)

        logger.info(f"Loading labels from {labels_path}")
        labels = np.load(labels_path, allow_pickle=True)

        y_train = labels["y_train"]
        y_val = labels["y_val"]
        y_test = labels["y_test"]

        # Since feature engineering creates train/val/test split, we need to use those splits
        # The CSV maintains all data, so we just take the first N samples matching the split sizes
        total_train = len(y_train)
        total_val = len(y_val)
        total_test = len(y_test)

        # Get texts in the same order they were split
        # We'll recreate the split from feature engineering (70% train, 15% val, 15% test)
        train_texts = df["clean_text"].values[:total_train]
        val_texts = df["clean_text"].values[total_train:total_train+total_val]
        test_texts = df["clean_text"].values[total_train+total_val:total_train+total_val+total_test]

        logger.info(f"Data loaded: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

        # Tokenize and pad
        X_train_padded, X_val_padded, X_test_padded, tokenizer = tokenize_texts(
            train_texts, val_texts, test_texts,
            output_dir=output_dir
        )

        # Train model
        model = train_lstm(X_train_padded, y_train, X_val_padded, y_val, output_dir=output_dir)

        # Evaluate on test set
        logger.info("\nTest Set Performance:")
        test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")

        y_test_pred = np.argmax(model.predict(X_test_padded), axis=1)
        logger.info("\nDetailed Test Set Classification Report:")
        # Dynamic target names based on number of classes
        num_classes = len(np.unique(y_test))
        if num_classes == 2:
            target_names = ["Allowed", "Dismissed"]
        else:
            target_names = ["Allowed", "Dismissed", "Partially Allowed"]
        logger.info(classification_report(
            y_test, y_test_pred,
            target_names=target_names[:num_classes]
        ))

        return model

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        model = main()
        logger.info("LSTM training completed successfully!")
    except Exception as e:
        logger.error(f"Failed to train LSTM: {str(e)}")
        exit(1)
