# JUDGELYTICS - ML Pipeline: Dataset Download Module
# Purpose: Download and merge NyayaAnumana datasets from HuggingFace
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: April 2026 (Updated)

"""
Dataset download and preprocessing module for Judgelytics.

Downloads the NyayaAnumana dataset from HuggingFace and prepares it
for consumer case filtering and ML training.

Dataset source:
    - L-NLProc/NyayaAnumana-Transformers-Results (Unified dataset)

The dataset includes judgement text and outcome labels suitable for
Indian legal case outcome prediction.
"""

import logging
import pandas as pd
from datasets import load_dataset
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _load_nyaya_dataset() -> pd.DataFrame:
    """Load and normalize NyayaAnumana dataset."""

    logger.info("Loading L-NLProc/NyayaAnumana-Transformers-Results...")
    dataset = load_dataset("L-NLProc/NyayaAnumana-Transformers-Results", trust_remote_code=True)

    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    data = dataset[split_name]
    df = data.to_pandas()

    # Identify and normalize column names
    text_columns = ["judgement", "judgment", "text", "content", "case_text"]
    label_columns = ["label", "outcome", "decision", "result"]

    text_col = next((col for col in text_columns if col in df.columns), df.columns[0])
    label_col = next((col for col in label_columns if col in df.columns), df.columns[1])

    logger.info(f"Nyaya columns mapped -> text: '{text_col}', label: '{label_col}'")

    df = df[[text_col, label_col]].copy()
    df = df.rename(columns={text_col: "judgement", label_col: "label"})

    # Normalize numeric labels if present
    label_map = {"0": "Allowed", "1": "Dismissed", "2": "Partially Allowed"}
    label_as_str = df["label"].astype(str)
    if set(label_as_str.unique()).issubset(set(label_map.keys())):
        df["label"] = label_as_str.map(label_map)

    df["source"] = "nyaya_transformers_results"
    df["court_tier"] = "all"
    df["split"] = "train"

    return df


def _load_cfpb_dataset(max_rows: int = 30000) -> pd.DataFrame:
    """
    Load and map CFPB consumer complaints dataset into outcome labels.

    Label mapping:
    - Closed with monetary relief -> Allowed
    - Closed with non-monetary relief -> Partially Allowed
    - Closed with explanation/Closed/Untimely response -> Dismissed
    """

    logger.info("Loading milesbutler/consumer_complaints for consumer-domain augmentation...")
    dataset = load_dataset("milesbutler/consumer_complaints")
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    df = dataset[split_name].to_pandas()

    required_cols = ["Consumer Complaint", "Company Response to Consumer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CFPB dataset missing required column: {col}")

    response_to_label = {
        "Closed with monetary relief": "Allowed",
        "Closed with non-monetary relief": "Partially Allowed",
        "Closed with explanation": "Dismissed",
        "Closed": "Dismissed",
        "Untimely response": "Dismissed",
    }

    df = df[df["Company Response to Consumer"].isin(response_to_label.keys())].copy()
    df = df[["Consumer Complaint", "Company Response to Consumer"]].rename(
        columns={
            "Consumer Complaint": "judgement",
            "Company Response to Consumer": "label",
        }
    )

    df["label"] = df["label"].map(response_to_label)
    df = df.dropna(subset=["judgement", "label"])

    # Keep training runtime manageable while preserving class proportions.
    if max_rows > 0 and len(df) > max_rows:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(frac=max_rows / len(df), random_state=42))
            .reset_index(drop=True)
        )

    df["source"] = "cfpb_consumer_complaints"
    df["court_tier"] = "consumer_finance"
    df["split"] = "train"

    logger.info(f"CFPB rows retained: {len(df)}")
    return df


def download_and_merge_datasets(output_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download NyayaAnumana dataset from HuggingFace and prepare for pipeline.

    Downloads the unified judicial decision dataset from L-NLProc/NyayaAnumana-Transformers-Results,
    uses the train split, and adds metadata columns for pipeline compatibility.

    Args:
        output_dir (str): Directory to save the merged CSV file. Default: "data/raw"

    Returns:
        pd.DataFrame: DataFrame with columns: judgement, label, source, court_tier, split

    Raises:
        Exception: If dataset download or processing fails
    """

    logger.info("Starting dataset download from HuggingFace...")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        nyaya_df = _load_nyaya_dataset()
        cfpb_df = _load_cfpb_dataset(max_rows=30000)

        df = pd.concat([nyaya_df, cfpb_df], ignore_index=True)

        logger.info(f"Combined dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Source distribution: {df['source'].value_counts().to_dict()}")

        # Log label distribution
        logger.info("\nLabel distribution:")
        label_counts = df["label"].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  Label {label}: {count} rows ({percentage:.2f}%)")

        # Verify data quality
        missing_judgement = df["judgement"].isna().sum()
        missing_label = df["label"].isna().sum()

        if missing_judgement > 0 or missing_label > 0:
            logger.warning(f"Missing values detected - judgement: {missing_judgement}, label: {missing_label}")
            logger.info("Removing rows with missing values...")
            df = df.dropna()
            logger.info(f"Dataset shape after cleaning: {df.shape}")

        # Save to CSV
        output_path = os.path.join(output_dir, "nyaya_combined.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"\nDataset saved to: {output_path}")

        return df

    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        logger.error(
            "Ensure both datasets are reachable: "
            "L-NLProc/NyayaAnumana-Transformers-Results and milesbutler/consumer_complaints"
        )
        raise


if __name__ == "__main__":
    # Execute download when run as script
    try:
        df = download_and_merge_datasets()
        logger.info("Download completed successfully!")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        exit(1)
