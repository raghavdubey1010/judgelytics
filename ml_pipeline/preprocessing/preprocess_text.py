# JUDGELYTICS - ML Pipeline: Text Preprocessing Module
# Purpose: Clean and normalize judgment text using NLP techniques
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Text preprocessing for judicial documents.

Handles text cleaning, tokenization, lemmatization, and stopword removal
optimized for Indian legal judgment analysis.
"""

import logging
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pathlib import Path
import os
import ssl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK wordnet...")
    try:
        nltk.download('wordnet', quiet=True)
    except:
        pass


# Custom legal stopwords specific to Indian legal documents
LEGAL_STOPWORDS = [
    "aforesaid",
    "hereinabove",
    "hereinafter",
    "thereof",
    "wherein",
    "whereby",
    "therein",
    "petitioner",
    "respondent",
    "appellant",
    "hon'ble",
    "learned",
    "ld.",
    "counsel",
    "bench",
    "coram",
    "vs",
    "v.",
    "vs.",
]

# Initialize reusable NLP resources once to avoid per-row overhead.
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(LEGAL_STOPWORDS)
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean and normalize judgment text.

    Removes citations, section references, special characters, and extra whitespace.

    Args:
        text (str): Raw judgment text

    Returns:
        str: Cleaned text
    """

    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove case citations like [2023] 4 SCC 123, AIR 2020 SC 456
    text = re.sub(r'\[\d{4}\]', '', text)
    text = re.sub(r'\d+ \w{2,3} \d+', '', text)
    text = re.sub(r'\d+ SCC \d+', '', text)
    text = re.sub(r'AIR \d+ [A-Z]{2,3} \d+', '', text)

    # Remove page numbers and references
    text = re.sub(r'page \d+', '', text)
    text = re.sub(r'p\. \d+', '', text)

    # Remove generic section references but preserve important ones
    text = re.sub(r'(?<!section )?\bs\. ?\d+\b(?!/)', '', text, flags=re.IGNORECASE)

    # Remove special characters except letters, numbers, spaces, and hyphens
    text = re.sub(r'[^a-z0-9\s\-]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_and_lemmatize(text: str) -> str:
    """
    Tokenize, remove stopwords, and lemmatize text.

    Args:
        text (str): Cleaned judgment text

    Returns:
        str: Preprocessed text with tokens lemmatized and stopwords removed
    """

    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Tokenize
    tokens = word_tokenize(text)

    # Process tokens: remove stopwords and lemmatize
    processed_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return " ".join(processed_tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "judgement"
) -> pd.DataFrame:
    """
    Preprocess all texts in DataFrame.

    Adds "clean_text" column with preprocessed judgment text.

    Args:
        df (pd.DataFrame): DataFrame with judgment texts
        text_col (str): Name of column containing text. Default: "judgement"

    Returns:
        pd.DataFrame: DataFrame with added "clean_text" column
    """

    logger.info(f"Preprocessing {len(df)} texts...")

    # Create copy to avoid modifying original
    df = df.copy()

    # Clean text
    logger.info("Step 1/2: Cleaning text...")
    df['clean_text'] = df[text_col].apply(clean_text)

    # Tokenize and lemmatize
    logger.info("Step 2/2: Tokenizing and lemmatizing...")
    df['clean_text'] = df['clean_text'].apply(tokenize_and_lemmatize)

    # Remove rows with empty clean_text
    initial_rows = len(df)
    df = df[df['clean_text'].str.len() > 0].copy()
    removed_rows = initial_rows - len(df)

    logger.info(f"Preprocessing complete!")
    logger.info(f"  Rows with empty text after preprocessing: {removed_rows}")
    logger.info(f"  Rows retained: {len(df)}")

    return df


def main(input_path: str, output_dir: str = "data/processed") -> pd.DataFrame:
    """
    Main entry point: Load filtered dataset and preprocess texts.

    Args:
        input_path (str): Path to input CSV file
        output_dir (str): Directory for output. Default: "data/processed"

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """

    logger.info(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    # Preprocess texts
    df = preprocess_dataframe(df)

    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save preprocessed dataset
    output_path = os.path.join(output_dir, "consumer_cases_preprocessed.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    input_path = "data/processed/consumer_cases_raw.csv"
    try:
        df = main(input_path)
        logger.info("Text preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        exit(1)
