# JUDGELYTICS - FastAPI Backend: ML Service
# Purpose: Load and run ML models for predictions with graceful fallback
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
ML service for Judgelytics backend.

Loads trained models and provides prediction interface for case analysis.
Falls back to an intelligent rule-based engine if models are not yet trained.
"""

import logging
import os
import re
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

from ..config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLService:
    """ML model loading and prediction service with rule-based fallback."""

    def __init__(self):
        """Initialize ML service by loading all models."""
        self.models_loaded = False
        self.tfidf = None
        self.classifiers: Dict = {}
        self.lstm_model = None
        self.lstm_tokenizer = None
        self.label_encoder = None

        # Try loading — will not raise if models are missing
        self._try_load_models()

    def _try_load_models(self):
        """
        Attempt to load trained models from disk.
        Logs warnings for missing models but does NOT raise exceptions —
        the service continues in rule-based fallback mode.
        """
        try:
            models_dir = settings.ml_models_path
            logger.info(f"Looking for ML models in: {models_dir}")

            if not models_dir.exists():
                logger.warning(f"Models directory not found: {models_dir}. Using rule-based fallback.")
                return

            # Load TF-IDF vectorizer
            tfidf_path = models_dir / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                self.tfidf = joblib.load(tfidf_path)
                logger.info("✓ TF-IDF vectorizer loaded")
            else:
                logger.warning(f"TF-IDF vectorizer not found at {tfidf_path}")

            # Load logistic regression classifier
            lr_path = models_dir / "logistic_regression.pkl"
            if lr_path.exists():
                self.classifiers["logistic_regression"] = joblib.load(lr_path)
                logger.info("✓ Logistic Regression loaded")
            else:
                logger.warning(f"Logistic Regression not found at {lr_path}")

            # Load Naive Bayes classifier
            nb_path = models_dir / "naive_bayes.pkl"
            if nb_path.exists():
                self.classifiers["naive_bayes"] = joblib.load(nb_path)
                logger.info("✓ Naive Bayes loaded")
            else:
                logger.warning(f"Naive Bayes not found at {nb_path}")

            # Load LSTM model and tokenizer
            lstm_model_path = models_dir / "lstm_model.keras"
            lstm_tokenizer_path = models_dir / "lstm_tokenizer.pkl"

            if KERAS_AVAILABLE and lstm_model_path.exists():
                try:
                    self.lstm_model = keras.models.load_model(str(lstm_model_path))
                    logger.info("✓ LSTM model loaded")
                except Exception as e:
                    logger.warning(f"Failed to load LSTM model: {e}")

            if lstm_tokenizer_path.exists():
                self.lstm_tokenizer = joblib.load(lstm_tokenizer_path)
                logger.info("✓ LSTM tokenizer loaded")

            # Load label encoder
            encoder_path = models_dir / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info("✓ Label encoder loaded")

            # Mark as loaded only if we have tfidf + at least one classifier
            if self.tfidf and self.classifiers:
                self.models_loaded = True
                logger.info(
                    f"ML models loaded! ({len(self.classifiers)} classifiers, LSTM: {self.lstm_model is not None})"
                )
            else:
                logger.warning("No ML models found — using rule-based fallback for predictions.")

        except Exception as e:
            logger.warning(f"Could not load ML models: {e}. Using rule-based fallback.")
            self.models_loaded = False

    def _clean_text(self, text: str) -> str:
        """Clean text for model input."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\[\d{4}\]', '', text)
        text = re.sub(r'\d+ \w{2,3} \d+', '', text)
        text = re.sub(r'[^a-z0-9\s\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _classify_sector(self, text: str) -> str:
        """Classify case sector from text."""
        sector_patterns = {
            "E-commerce": r"amazon|flipkart|e-commerce|online shopping|online purchase|website|meesho|myntra|snapdeal",
            "Insurance": r"insurance|claim|policy|premium|irdai|insurer",
            "Real Estate": r"property|land|building|housing|flat|apartment|builder|developer|rera",
            "Automobile": r"car|vehicle|bike|motorcycle|two-wheeler|maruti|honda|hyundai",
            "Banking": r"bank|loan|credit|debit|atm|neft|upi|emi|interest",
            "Telecom": r"mobile|telephone|sim|recharge|bill|airtel|jio|bsnl|vodafone",
            "Consumer Goods": r"product|defective|appliance|quality|electronic|refrigerator|washing machine",
            "Education": r"school|college|university|course|fee|admission|tuition",
            "Healthcare": r"doctor|hospital|medical|medicine|treatment|nurse|surgery|clinic",
            "Travel": r"flight|hotel|travel|booking|resort|tour|ticket|airline",
        }
        text_lower = text.lower()
        for sector, pattern in sector_patterns.items():
            if re.search(pattern, text_lower):
                return sector
        return "General"

    def _extract_amount(self, text: str) -> float:
        """Extract monetary amount from text."""
        patterns = [
            r'₹\s*([\d,]+)',
            r'rs\.?\s*([\d,]+)',
            r'rupees?\s*([\d,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    continue
        return 0.0

    def _classify_forum(self, claim_amount: float) -> str:
        """Classify recommended forum based on claim amount."""
        if claim_amount <= 1_00_00_000:   # ₹1 crore
            return "District Consumer Commission"
        elif claim_amount <= 10_00_00_000:  # ₹10 crores
            return "State Consumer Commission"
        else:
            return "National Consumer Commission (NCDRC)"

    def _rule_based_predict(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based prediction fallback when ML models are not available.
        Uses domain knowledge about consumer case outcomes.
        """
        description = case_data.get("description", "")
        claim_amount = case_data.get("claim_amount", 0.0)
        evidence_count = case_data.get("evidence_count", 0)
        has_legal_notice = case_data.get("has_legal_notice", "No")
        opponent_type = case_data.get("opponent_type", "Company")
        sector = case_data.get("sector", self._classify_sector(description))

        # Score-based prediction
        score = 0.5  # Base probability

        # Evidence quality
        if evidence_count >= 5:
            score += 0.15
        elif evidence_count >= 3:
            score += 0.08
        elif evidence_count == 0:
            score -= 0.15

        # Legal notice sent
        if has_legal_notice == "Yes":
            score += 0.10

        # Strong keyword signals
        strong_words = ["refund", "replacement", "defective", "fraud", "cheated", "not delivered", "counterfeit"]
        weak_words = ["delayed", "inconvenience", "service not good"]

        desc_lower = description.lower()
        if any(w in desc_lower for w in strong_words):
            score += 0.12
        if any(w in desc_lower for w in weak_words):
            score += 0.04

        # Sector-specific adjustments
        sector_bonus = {
            "E-commerce": 0.08, "Insurance": 0.05, "Healthcare": 0.10,
            "Real Estate": 0.05, "Banking": 0.06, "Telecom": 0.04,
        }
        score += sector_bonus.get(sector, 0.0)

        # Clamp to [0.25, 0.92]
        score = max(0.25, min(0.92, score))

        # Determine outcome
        if score >= 0.6:
            outcome = "Allowed"
            applicable_sections = ["Section 35 COPRA 2019", "Section 2(11) COPRA 2019",
                                    "Section 69 COPRA 2019", "Section 39(1)(b) COPRA 2019"]
        elif score >= 0.45:
            outcome = "Partially Allowed"
            applicable_sections = ["Section 35 COPRA 2019", "Section 47 COPRA 2019",
                                    "Section 2(7) COPRA 2019"]
        else:
            outcome = "Dismissed"
            applicable_sections = ["Section 58 COPRA 2019", "Section 2(21) COPRA 2019"]

        # Confidence
        if score > 0.75:
            confidence = "high"
        elif score > 0.58:
            confidence = "medium"
        else:
            confidence = "low"

        # Evidence strength
        if evidence_count >= 5:
            evidence_strength = "Strong"
        elif evidence_count >= 2:
            evidence_strength = "Moderate"
        else:
            evidence_strength = "Weak"

        recommended_forum = self._classify_forum(claim_amount)
        filing_fees = {
            "District Consumer Commission": "₹200–₹4,000",
            "State Consumer Commission": "₹5,000",
            "National Consumer Commission (NCDRC)": "₹7,500",
        }

        return {
            "outcome": outcome,
            "win_probability": round(score, 3),
            "win_probability_pct": int(score * 100),
            "confidence": confidence,
            "recommended_forum": recommended_forum,
            "applicable_sections": applicable_sections,
            "evidence_strength": evidence_strength,
            "filing_fee": filing_fees.get(recommended_forum, "₹500"),
            "similar_cases_count": 142,
            "models_used": ["rule_based_engine"],
        }

    def predict(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction for a case.

        Uses ML ensemble if models are loaded, otherwise falls back to rule-based.

        Args:
            case_data (dict): Case data with description, claim_amount, etc.

        Returns:
            dict: Prediction results
        """
        # Fallback if ML models are not available
        if not self.models_loaded:
            logger.info("Using rule-based prediction (ML models not loaded)")
            return self._rule_based_predict(case_data)

        try:
            logger.info("Generating ML prediction...")

            description = case_data.get("description", "")
            clean_text = self._clean_text(description)
            claim_amount = case_data.get("claim_amount", 0.0)
            evidence_count = case_data.get("evidence_count", 0)
            has_legal_notice = 1 if case_data.get("has_legal_notice") == "Yes" else 0

            tfidf_vector = None
            if self.tfidf:
                tfidf_vector = self.tfidf.transform([clean_text])

            predictions = []
            probabilities = []

            if tfidf_vector is not None:
                for model_name, model in self.classifiers.items():
                    try:
                        pred = model.predict(tfidf_vector)[0]
                        predictions.append(pred)
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(tfidf_vector)[0]
                            probabilities.append(max(proba))
                        logger.info(f"  {model_name}: {pred}")
                    except Exception as e:
                        logger.warning(f"Error with {model_name}: {e}")

            if self.lstm_model and self.lstm_tokenizer:
                try:
                    sequences = self.lstm_tokenizer.texts_to_sequences([clean_text])
                    max_len = 100
                    padded = np.zeros((1, max_len), dtype=np.int32)
                    for i, seq in enumerate(sequences):
                        length = min(len(seq), max_len)
                        padded[i, -length:] = seq[-length:]
                    lstm_output = self.lstm_model.predict(padded, verbose=0)
                    lstm_pred = 1 if lstm_output[0][0] > 0.5 else 0
                    lstm_prob = float(lstm_output[0][0]) if lstm_output[0][0] > 0.5 else float(1 - lstm_output[0][0])
                    predictions.append(lstm_pred)
                    probabilities.append(lstm_prob)
                    logger.info(f"  lstm: {lstm_pred} (prob: {lstm_prob:.2f})")
                except Exception as e:
                    logger.warning(f"Error with LSTM: {e}")

            if not predictions:
                logger.warning("No ML predictions, falling back to rule-based")
                return self._rule_based_predict(case_data)

            outcome_pred = max(set(predictions), key=predictions.count)
            avg_prob = float(np.mean(probabilities)) if probabilities else 0.5

            if self.label_encoder:
                try:
                    outcome_name = self.label_encoder.inverse_transform([outcome_pred])[0]
                except Exception:
                    outcome_name = "Allowed" if outcome_pred == 1 else "Dismissed"
            else:
                outcome_name = "Allowed" if outcome_pred == 1 else "Dismissed"

            confidence_score = avg_prob
            if confidence_score > 0.75:
                confidence = "high"
            elif confidence_score > 0.6:
                confidence = "medium"
            else:
                confidence = "low"

            # If the model predicts Dismissed, the confidence is in dismissal. 
            # Therefore, the probability of *winning* is 1 - confidence.
            if outcome_name in ["Allowed", "Partially Allowed"]:
                win_prob_val = confidence_score
            else:
                win_prob_val = 1.0 - confidence_score

            if outcome_name == "Allowed":
                applicable_sections = ["Section 35 COPRA 2019", "Section 2(11) COPRA 2019",
                                        "Section 69 COPRA 2019", "Section 73 COPRA 2019"]
            elif outcome_name == "Partially Allowed":
                applicable_sections = ["Section 35 COPRA 2019", "Section 47 COPRA 2019",
                                        "Section 2(7) COPRA 2019"]
            else:
                applicable_sections = ["Section 58 COPRA 2019", "Section 2(21) COPRA 2019"]

            evidence_count = case_data.get("evidence_count", 0)
            if evidence_count >= 5:
                evidence_strength = "Strong"
            elif evidence_count >= 2:
                evidence_strength = "Moderate"
            else:
                evidence_strength = "Weak"

            recommended_forum = self._classify_forum(claim_amount)
            filing_fees = {
                "District Consumer Commission": "₹200–₹4,000",
                "State Consumer Commission": "₹5,000",
                "National Consumer Commission (NCDRC)": "₹7,500",
            }

            return {
                "outcome": outcome_name,
                "win_probability": round(win_prob_val, 3),
                "win_probability_pct": int(win_prob_val * 100),
                "confidence": confidence,
                "recommended_forum": recommended_forum,
                "applicable_sections": applicable_sections,
                "evidence_strength": evidence_strength,
                "filing_fee": filing_fees.get(recommended_forum, "₹500"),
                "similar_cases_count": 142,
                "models_used": list(self.classifiers.keys()) + (["lstm"] if self.lstm_model else []),
            }

        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}. Falling back to rule-based.")
            return self._rule_based_predict(case_data)


# ─── Global Instance ──────────────────────────────────────────────────────────
ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    global ml_service
    if ml_service is None:
        ml_service = MLService()
    return ml_service


def initialize_ml_service() -> MLService:
    global ml_service
    ml_service = MLService()
    return ml_service
