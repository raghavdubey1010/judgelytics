# JUDGELYTICS - ML Pipeline: Feature Engineering Module
# Purpose: Build TF-IDF and structured features for model training
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Feature engineering for ML model training.

Builds TF-IDF features from text and engineered features from structured data.
Includes text statistics, claim analysis, and legal metadata extraction.
"""

import logging
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTOR CLASSIFICATION PATTERNS
# ============================================================================
SECTOR_PATTERNS = {
    "E-commerce": [r"amazon|flipkart|e-commerce|online shopping|online purchase|website"],
    "Insurance": [r"insurance|claim|policy|premium"],
    "Real Estate": [r"property|land|building|housing|flat|apartment|registration"],
    "Automobile": [r"car|vehicle|bike|motorcycle|two-wheeler|scooter"],
    "Banking": [r"bank|loan|credit|debit|atm|overdraft"],
    "Telecom": [r"mobile|telephone|sim|recharge|bill|network"],
    "Consumer Goods": [r"product|defective|appliance|quality|manufacturer"],
    "Education": [r"school|college|university|course|fee|admission"],
    "Healthcare": [r"doctor|hospital|medical|medicine|treatment|surgery"],
    "Travel": [r"flight|hotel|travel|booking|resort|airline"],
    "Food & Beverage": [r"restaurant|food|meal|drink|cafe|hotel"],
    "Utilities": [r"electricity|water|supply|bill|meter"]
}

# CPA 2019 Consumer Forum Thresholds
CPA_THRESHOLDS = {
    "District": (0, 1_00_00_000),          # ₹0 to ₹1 crore
    "State": (1_00_00_001, 10_00_00_000),   # ₹1 crore to ₹10 crores
    "National": (10_00_00_001, float('inf'))  # ₹10 crores and above
}


def classify_sector(text: str) -> str:
    """
    Classify case sector based on text analysis.

    Args:
        text (str): Judgment text

    Returns:
        str: Sector name or "General" if no match
    """

    if not isinstance(text, str):
        return "General"

    text_lower = text.lower()

    for sector, patterns in SECTOR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return sector

    return "General"


def extract_claim_amount(text: str) -> float:
    """
    Extract monetary claim amount from text.

    Looks for Indian Rupee amounts in various formats (₹, Rs, Rupees)

    Args:
        text (str): Judgment text

    Returns:
        float: Extracted amount or 0.0 if not found
    """

    if not isinstance(text, str):
        return 0.0

    # Pattern: ₹ 5,00,000 or Rs 5,00,000 or Rupees 500000
    patterns = [
        r'₹\s*([\d,]+)',
        r'Rs\.?\s*([\d,]+)',
        r'rupees?\s*([\d,]+)',
        r'crore.*?₹\s*([\d,]+)',
        r'lac.*?₹\s*([\d,]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Extract and convert to numbers
            amount_str = match.group(1).replace(",", "")
            try:
                return float(amount_str)
            except ValueError:
                continue

    return 0.0


def classify_claim_bucket(amount: float) -> str:
    """
    Classify claim into jurisdiction buckets based on CPA 2019.

    Args:
        amount (float): Claim amount in rupees

    Returns:
        str: Bucket name: "Low", "District", "State", "National"
    """

    if amount <= 0:
        return "Low"
    elif amount <= CPA_THRESHOLDS["District"][1]:
        return "District"
    elif amount <= CPA_THRESHOLDS["State"][1]:
        return "State"
    else:
        return "National"


def calculate_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate text-based features.

    Args:
        df (pd.DataFrame): DataFrame with "clean_text" column

    Returns:
        pd.DataFrame: DataFrame with added feature columns
    """

    logger.info("Calculating text features...")

    # Text length (word count)
    df["text_length"] = df["clean_text"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )

    # Legal notice presence
    df["has_legal_notice"] = df["judgement"].apply(
        lambda x: bool(re.search(r'legal notice', str(x), re.IGNORECASE))
        if isinstance(x, str) else False
    )

    return df


def build_tfidf_matrix(
    train_texts,
    val_texts,
    test_texts,
    output_dir: str = "saved_models"
) -> tuple:
    """
    Build TF-IDF feature matrices.

    Fits vectorizer on training data only (no data leakage).

    Args:
        train_texts: Training text samples
        val_texts: Validation text samples
        test_texts: Test text samples
        output_dir (str): Directory to save vectorizer

    Returns:
        tuple: (X_train_tfidf, X_val_tfidf, X_test_tfidf) as sparse matrices
    """

    logger.info("Building TF-IDF matrices...")
    logger.info(f"  Vectorizer config: max_features=15000, ngram_range=(1,2)")

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        stop_words='english'
    )

    # Fit on training data
    logger.info("Fitting vectorizer on training data...")
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    logger.info(f"  Training TF-IDF shape: {X_train_tfidf.shape}")

    # Transform validation and test
    X_val_tfidf = vectorizer.transform(val_texts)
    logger.info(f"  Validation TF-IDF shape: {X_val_tfidf.shape}")

    X_test_tfidf = vectorizer.transform(test_texts)
    logger.info(f"  Test TF-IDF shape: {X_test_tfidf.shape}")

    # Save vectorizer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"TF-IDF vectorizer saved to {vectorizer_path}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf


def build_structured_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build structured feature matrix from engineered features.

    Features: [sector_encoded, claim_amount, claim_bucket_encoded, text_length,
               has_legal_notice, is_district_forum, tier2_score, court_tier_encoded]

    Args:
        df (pd.DataFrame): Dataframe with required columns

    Returns:
        np.ndarray: Feature matrix of shape (n_samples, 8)
    """

    logger.info("Building structured features...")

    features = []

    # 1. Sector (encoded)
    sector_encoder = LabelEncoder()
    sector_encoded = sector_encoder.fit_transform(df["sector"])
    features.append(sector_encoded.reshape(-1, 1))

    # 2. Claim amount (normalized)
    claim_amount = df["claim_amount"].values
    claim_amount_normalized = np.log1p(claim_amount) / np.log1p(claim_amount.max())
    features.append(claim_amount_normalized.reshape(-1, 1))

    # 3. Claim bucket (encoded)
    bucket_encoder = LabelEncoder()
    bucket_encoded = bucket_encoder.fit_transform(df["claim_bucket"])
    features.append(bucket_encoded.reshape(-1, 1))

    # 4. Text length (normalized)
    text_length = df["text_length"].values
    text_length_normalized = text_length / text_length.max()
    features.append(text_length_normalized.reshape(-1, 1))

    # 5. Has legal notice (bool to int)
    has_legal_notice = df["has_legal_notice"].astype(int).values
    features.append(has_legal_notice.reshape(-1, 1))

    # 6. Is district forum (bool to int)
    is_district_forum = (df["court_tier"] == "District Court").astype(int).values
    features.append(is_district_forum.reshape(-1, 1))

    # 7. Tier 2 score (normalized)
    tier2_score = df["tier2_score"].values
    tier2_score_normalized = tier2_score / (tier2_score.max() + 1e-8)
    features.append(tier2_score_normalized.reshape(-1, 1))

    # 8. Court tier (encoded)
    court_encoder = LabelEncoder()
    court_encoded = court_encoder.fit_transform(df["court_tier"])
    features.append(court_encoded.reshape(-1, 1))

    # Combine all features
    structured_features = np.hstack(features)
    logger.info(f"Structured feature matrix shape: {structured_features.shape}")

    return structured_features


def encode_labels(
    train_labels,
    val_labels,
    test_labels,
    output_dir: str = "saved_models"
) -> tuple:
    """
    Encode outcome labels to numeric values.

    Mapping: Allowed=0, Dismissed=1, Partially Allowed=2

    Args:
        train_labels: Training labels
        val_labels: Validation labels
        test_labels: Test labels
        output_dir (str): Directory to save encoder

    Returns:
        tuple: (y_train_encoded, y_val_encoded, y_test_encoded, label_encoder)
    """

    logger.info("Encoding labels...")

    # Convert labels to strings to ensure consistent handling
    train_labels = [str(label) for label in train_labels]
    val_labels = [str(label) for label in val_labels]
    test_labels = [str(label) for label in test_labels]

    # Check if labels are already numeric (0/1) or text
    if train_labels[0] in ['0', '1']:
        # Numeric labels: 0=Allowed, 1=Dismissed
        label_mapping = {"0": 0, "1": 1}
        unique_labels = ["0", "1"]
    else:
        # Text labels
        label_mapping = {
            "Allowed": 0,
            "Dismissed": 1,
            "Partially Allowed": 2
        }
        unique_labels = list(label_mapping.keys())

    # Create encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)

    # Encode
    y_train_encoded = label_encoder.transform(train_labels)
    y_val_encoded = label_encoder.transform(val_labels)
    y_test_encoded = label_encoder.transform(test_labels)

    logger.info(f"Label encoding: Using detected label format")

    # Save encoder
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"Label encoder saved to {encoder_path}")

    return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder


def main(input_path: str, output_dir: str = "data/processed") -> dict:
    """
    Main entry point: Load preprocessed data and build features.

    Args:
        input_path (str): Path to preprocessed CSV
        output_dir (str): Directory for output. Default: "data/processed"

    Returns:
        dict: Dictionary with feature matrices and metadata
    """

    logger.info(f"Loading preprocessed data from: {input_path}")
    df = pd.read_csv(input_path)

    # Add engineered features
    logger.info("Engineering features...")
    df["sector"] = df["judgement"].apply(classify_sector)
    df["claim_amount"] = df["judgement"].apply(extract_claim_amount)
    df["claim_bucket"] = df["claim_amount"].apply(classify_claim_bucket)

    df = calculate_text_features(df)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Separate by split
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    # If validation/test splits are empty, create them from training data
    if len(val_df) == 0 or len(test_df) == 0:
        from sklearn.model_selection import train_test_split
        logger.info("Validation/Test splits empty. Creating them from training data...")
        
        # Split: 70% train, 15% val, 15% test
        train_df, temp_df = train_test_split(train_df, test_size=0.30, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Build TF-IDF matrices
    X_train_tfidf, X_val_tfidf, X_test_tfidf = build_tfidf_matrix(
        train_df["clean_text"].values,
        val_df["clean_text"].values,
        test_df["clean_text"].values
    )

    # Build structured features
    X_train_struct = build_structured_features(train_df)
    X_val_struct = build_structured_features(val_df)
    X_test_struct = build_structured_features(test_df)

    # Combine TF-IDF and structured features
    X_train_combined = sparse.hstack([X_train_tfidf, X_train_struct])
    X_val_combined = sparse.hstack([X_val_tfidf, X_val_struct])
    X_test_combined = sparse.hstack([X_test_tfidf, X_test_struct])

    logger.info(f"Combined feature shapes:")
    logger.info(f"  Train: {X_train_combined.shape}")
    logger.info(f"  Val: {X_val_combined.shape}")
    logger.info(f"  Test: {X_test_combined.shape}")

    # Encode labels
    y_train, y_val, y_test, label_encoder = encode_labels(
        train_df["label"].values,
        val_df["label"].values,
        test_df["label"].values
    )

    # Save feature matrices (as separate sparse files for compatibility)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    feature_train_path = os.path.join(output_dir, "X_train_combined.npz")
    sparse.save_npz(feature_train_path, X_train_combined)
    logger.info(f"Train features saved to {feature_train_path}")
    
    feature_val_path = os.path.join(output_dir, "X_val_combined.npz")
    sparse.save_npz(feature_val_path, X_val_combined)
    logger.info(f"Validation features saved to {feature_val_path}")
    
    feature_test_path = os.path.join(output_dir, "X_test_combined.npz")
    sparse.save_npz(feature_test_path, X_test_combined)
    logger.info(f"Test features saved to {feature_test_path}")

    # Save labels
    labels_output = os.path.join(output_dir, "labels.npz")
    np.savez(labels_output, y_train=y_train, y_val=y_val, y_test=y_test)
    logger.info(f"Labels saved to {labels_output}")

    return {
        "X_train": X_train_combined,
        "X_val": X_val_combined,
        "X_test": X_test_combined,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": label_encoder
    }


if __name__ == "__main__":
    input_path = "data/processed/consumer_cases_preprocessed.csv"
    try:
        result = main(input_path)
        logger.info("Feature engineering completed successfully!")
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        exit(1)
