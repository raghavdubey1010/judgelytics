# JUDGELYTICS - ML Pipeline: Consumer Case Filter Module
# Purpose: Two-tier keyword filtering for consumer law cases
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Consumer case filtering based on two-tier keyword matching.

Applies both Tier 1 (high confidence) and Tier 2 (supporting) keywords
to identify consumer law cases from judicial decisions.
"""

import logging
import pandas as pd
import re
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TIER 1 KEYWORDS (High Confidence - any single match = include case)
# ============================================================================
TIER_1_KEYWORDS = [
    r"consumer protection act",
    r"consumer forum",
    r"district forum",
    r"ncdrc",
    r"scdrc",
    r"complainant",
    r"opposite party",
    r"section 35",
    r"section 47",
    r"section 58",
    r"section 2\(7\)",
    r"section 2\(11\)",
    r"deficiency of service"
]

# ============================================================================
# TIER 2 KEYWORDS (Supporting - 4+ matches = include even without Tier 1)
# ============================================================================
TIER_2_KEYWORDS = [
    r"insurance claim",
    r"refund",
    r"defective product",
    r"unfair trade practice",
    r"builder",
    r"bank charges",
    r"telecom",
    r"e-commerce",
    r"online purchase",
    r"warranty",
    r"compensation awarded",
    r"mental agony",
    r"litigation cost"
]

# Court-related keywords as secondary filter
COURT_KEYWORDS = [
    r"consumer",
    r"cdrc",
    r"ncdrc",
    r"district forum",
    r"consumer protection"
]


def compile_regex_patterns() -> tuple:
    """
    Compile regex patterns for efficient matching.

    Returns:
        tuple: (tier1_pattern, tier2_patterns_list)
    """
    # Tier 1: Single combined pattern with alternation
    tier1_pattern = re.compile("|".join(TIER_1_KEYWORDS), re.IGNORECASE)

    # Tier 2: Individual patterns for counting matches
    tier2_patterns = [re.compile(keyword, re.IGNORECASE) for keyword in TIER_2_KEYWORDS]

    return tier1_pattern, tier2_patterns


def score_row(text: str, tier1_pattern, tier2_patterns: list) -> tuple:
    """
    Score a case text for consumer law relevance using two-tier keyword matching.

    Args:
        text (str): Judgment text to analyze
        tier1_pattern: Compiled Tier 1 regex pattern
        tier2_patterns (list): List of compiled Tier 2 regex patterns

    Returns:
        tuple: (is_consumer: bool, tier2_score: int)
            - is_consumer: True if case qualifies as consumer law case
            - tier2_score: Count of Tier 2 keyword matches
    """

    if not isinstance(text, str) or len(text.strip()) == 0:
        return False, 0

    # Count Tier 1 matches
    tier1_match = tier1_pattern.search(text)

    # Count Tier 2 matches
    tier2_score = 0
    for pattern in tier2_patterns:
        if pattern.search(text):
            tier2_score += 1

    # Determine if case is consumer-related
    # Qualified if: Tier 1 match OR (Tier 2 score >= 4)
    is_consumer = bool(tier1_match) or (tier2_score >= 4)

    return is_consumer, tier2_score


def filter_dataframe(
    df: pd.DataFrame,
    text_col: str = "judgement"
) -> pd.DataFrame:
    """
    Filter DataFrame to identify consumer law cases.

    Applies two-tier keyword filtering and adds metadata columns:
    - is_consumer (bool): True if case qualifies
    - tier2_score (int): Count of Tier 2 keywords matched
    - confidence (str): "high", "medium", or "low" based on evidence

    Args:
        df (pd.DataFrame): Input DataFrame with judgment text
        text_col (str): Name of column containing judgment text. Default: "judgement"

    Returns:
        pd.DataFrame: Filtered DataFrame with consumer law cases only
    """

    logger.info("Compiling regex patterns...")
    tier1_pattern, tier2_patterns = compile_regex_patterns()

    logger.info(f"Total rows before filtering: {len(df)}")
    logger.info("Analyzing each case for consumer law relevance...")

    # Rows from curated consumer-domain datasets are already consumer cases.
    prequalified_sources = {"cfpb_consumer_complaints"}
    if "source" in df.columns:
        prequalified_mask = df["source"].isin(prequalified_sources)
    else:
        prequalified_mask = pd.Series(False, index=df.index)

    # Apply keyword scoring to remaining rows.
    scored = df[text_col].apply(lambda x: pd.Series(score_row(x, tier1_pattern, tier2_patterns)))
    scored.columns = ["is_consumer", "tier2_score"]

    df["is_consumer"] = scored["is_consumer"] | prequalified_mask
    df["tier2_score"] = scored["tier2_score"]
    df.loc[prequalified_mask, "tier2_score"] = df.loc[prequalified_mask, "tier2_score"].clip(lower=5)

    # Determine confidence level
    def assign_confidence(row):
        """Assign confidence level based on evidence strength."""
        if row["tier2_score"] >= 8:
            return "high"
        elif row["tier2_score"] >= 4:
            return "medium"
        else:
            return "low"

    df["confidence"] = df.apply(assign_confidence, axis=1)

    # Filter to consumer cases only
    consumer_cases = df[df["is_consumer"]].copy()

    # Log results
    logger.info(f"Consumer cases identified: {len(consumer_cases)}")
    logger.info(f"Percentage retained: {(len(consumer_cases)/len(df)*100):.2f}%")
    logger.info(f"\nConfidence breakdown:")
    for conf_level in ["high", "medium", "low"]:
        count = len(consumer_cases[consumer_cases["confidence"] == conf_level])
        pct = (count / len(consumer_cases) * 100) if len(consumer_cases) > 0 else 0
        logger.info(f"  {conf_level.capitalize()}: {count} ({pct:.2f}%)")

    return consumer_cases


def main(input_path: str, output_dir: str = "data/processed") -> pd.DataFrame:
    """
    Main entry point: Load raw dataset and filter for consumer cases.

    Args:
        input_path (str): Path to input CSV file
        output_dir (str): Directory for output. Default: "data/processed"

    Returns:
        pd.DataFrame: Filtered consumer cases DataFrame
    """

    logger.info(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    # Filter for consumer cases
    filtered_df = filter_dataframe(df)

    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save filtered dataset
    output_path = os.path.join(output_dir, "consumer_cases_raw.csv")
    filtered_df.to_csv(output_path, index=False)
    logger.info(f"Filtered dataset saved to: {output_path}")

    return filtered_df


if __name__ == "__main__":
    input_path = "data/raw/nyaya_combined.csv"
    try:
        filtered_df = main(input_path)
        logger.info("Filtering completed successfully!")
    except Exception as e:
        logger.error(f"Filtering failed: {str(e)}")
        exit(1)
