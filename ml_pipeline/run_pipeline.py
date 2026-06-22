# JUDGELYTICS - ML Pipeline: Master Orchestration Script
# Purpose: Execute complete ML pipeline from download to model evaluation
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Master orchestration script for Judgelytics ML pipeline.

Executes all pipeline stages in sequence:
1. Dataset download and merge
2. Consumer case filtering
3. Text preprocessing
4. Feature engineering
5. Model training (all variants)
6. Model evaluation and comparison

Supports CLI flags for partial pipeline execution.
"""

import logging
import sys
import argparse
import time
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print project banner."""
    banner = """
╔════════════════════════════════════════════════════════════════════╗
║         JUDGELYTICS - AI-Powered Legal Judgment Prediction        ║
║          ML Pipeline Orchestration - Complete Training            ║
║    Project: B.Tech Minor | Medicaps University, Indore (2026)     ║
║    Authors: Rakhi Tiwari & Raghav Dubey                           ║
╚════════════════════════════════════════════════════════════════════╝
    """
    logger.info(banner)


def log_step(step_num: int, step_name: str):
    """Log pipeline step."""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP {step_num}: {step_name}")
    logger.info(f"{'='*70}")


def run_step(step_func, step_name: str, *args, **kwargs):
    """
    Execute a pipeline step with timing and error handling.

    Args:
        step_func: Function to execute
        step_name (str): Name for logging
        *args, **kwargs: Arguments for step_func

    Returns:
        Result from step_func or None if failed
    """

    try:
        logger.info(f"Starting {step_name}...")
        start_time = time.time()

        result = step_func(*args, **kwargs)

        elapsed = time.time() - start_time
        logger.info(f"✓ {step_name} completed in {elapsed:.2f}s")

        return result

    except Exception as e:
        logger.error(f"✗ {step_name} failed: {str(e)}")
        raise


def main(args):
    """
    Execute complete ML pipeline.

    Args:
        args: Command line arguments
    """

    print_banner()
    logger.info(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        pipeline_start = time.time()

        # ====================================================================
        # STEP 1: Download Dataset
        # ====================================================================
        if not args.skip_download and not args.models_only:
            log_step(1, "Dataset Download and Merge")

            from preprocessing.download_dataset import download_and_merge_datasets

            dataset_df = run_step(
                download_and_merge_datasets,
                "Dataset download and merge",
                output_dir="data/raw"
            )

        # ====================================================================
        # STEP 2: Filter Consumer Cases
        # ====================================================================
        if not args.skip_filter and not args.models_only:
            log_step(2, "Consumer Case Filtering")

            from preprocessing.filter_consumer import main as filter_consumer

            filtered_df = run_step(
                filter_consumer,
                "Consumer case filtering",
                input_path="data/raw/nyaya_combined.csv",
                output_dir="data/processed"
            )

        # ====================================================================
        # STEP 3: Text Preprocessing
        # ====================================================================
        if not args.skip_preprocess and not args.models_only:
            log_step(3, "Text Preprocessing")

            from preprocessing.preprocess_text import main as preprocess_text

            preprocessed_df = run_step(
                preprocess_text,
                "Text preprocessing",
                input_path="data/processed/consumer_cases_raw.csv",
                output_dir="data/processed"
            )

        # ====================================================================
        # STEP 4: Feature Engineering
        # ====================================================================
        if not args.skip_preprocess and not args.models_only:
            log_step(4, "Feature Engineering")

            from preprocessing.build_features import main as build_features

            features = run_step(
                build_features,
                "Feature engineering",
                input_path="data/processed/consumer_cases_preprocessed.csv",
                output_dir="data/processed"
            )

        # ====================================================================
        # STEP 5: Train Logistic Regression
        # ====================================================================
        log_step(5, "Train Logistic Regression")

        from models.train_logistic_regression import main as train_lr

        lr_model = run_step(
            train_lr,
            "Logistic Regression training",
            feature_matrices_path="data/processed/feature_matrices.npz",
            labels_path="data/processed/labels.npz",
            output_dir="saved_models"
        )

        # ====================================================================
        # STEP 6: Train Naive Bayes
        # ====================================================================
        log_step(6, "Train Naive Bayes")

        from models.train_naive_bayes import main as train_nb

        nb_model = run_step(
            train_nb,
            "Naive Bayes training",
            feature_matrices_path="data/processed/feature_matrices.npz",
            labels_path="data/processed/labels.npz",
            output_dir="saved_models"
        )

        # ====================================================================
        # STEP 7: Train Bi-LSTM
        # ====================================================================
        log_step(7, "Train Bidirectional LSTM")

        from models.train_lstm import main as train_lstm

        lstm_model = run_step(
            train_lstm,
            "LSTM training",
            clean_text_path="data/processed/consumer_cases_preprocessed.csv",
            labels_path="data/processed/labels.npz",
            output_dir="saved_models"
        )

        # ====================================================================
        # STEP 8: Train XGBoost Regressor
        # ====================================================================
        log_step(8, "Train XGBoost Regressor (Win Probability)")

        from models.train_xgboost_regressor import main as train_xgb

        xgb_model = run_step(
            train_xgb,
            "XGBoost Regressor training",
            preprocessed_path="data/processed/consumer_cases_preprocessed.csv",
            labels_path="data/processed/labels.npz",
            output_dir="saved_models"
        )

        # ====================================================================
        # STEP 9: Model Comparison and Evaluation
        # ====================================================================
        log_step(9, "Model Comparison and Evaluation")

        from evaluation.compare_models import main as compare_models

        comparison_df = run_step(
            compare_models,
            "Model comparison",
            feature_matrices_path="data/processed/feature_matrices.npz",
            labels_path="data/processed/labels.npz",
            models_dir="saved_models"
        )

        # ====================================================================
        # Pipeline Complete
        # ====================================================================
        pipeline_elapsed = time.time() - pipeline_start

        logger.info(f"\n{'='*70}")
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total execution time: {pipeline_elapsed:.2f}s ({pipeline_elapsed/60:.2f} minutes)")
        logger.info(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\nAll models saved to: saved_models/")
        logger.info(f"Processed data saved to: data/processed/")

        return 0

    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error("PIPELINE EXECUTION FAILED")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judgelytics ML Pipeline Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py

  # Skip download, run from filtering onward
  python run_pipeline.py --skip_download

  # Only train models (skip all preprocessing)
  python run_pipeline.py --models_only

  # Skip preprocessing and filtering
  python run_pipeline.py --skip_download --skip_filter --skip_preprocess
        """
    )

    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip dataset download step"
    )

    parser.add_argument(
        "--skip_filter",
        action="store_true",
        help="Skip consumer case filtering step"
    )

    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip text preprocessing and feature engineering steps"
    )

    parser.add_argument(
        "--models_only",
        action="store_true",
        help="Only train models (skip all preprocessing)"
    )

    args = parser.parse_args()

    try:
        exit_code = main(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
