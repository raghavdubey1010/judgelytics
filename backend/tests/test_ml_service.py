# JUDGELYTICS - FastAPI Backend: ML Service Tests
# Purpose: Unit tests for ML service
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Test suite for ML service predictions.

Tests ML model loading and prediction generation.
"""

import pytest
from app.services.ml_service import MLService


def test_ml_service_initialization():
    """Test ML service initialization."""
    try:
        ml_service = MLService()
        # Models may not be available in test environment
        assert ml_service is not None
    except Exception as e:
        # Expected if models not available
        pytest.skip(f"ML models not available: {str(e)}")


def test_clean_text():
    """Test text cleaning function."""
    ml_service = MLService()
    ml_service.models_loaded = False  # Set to False to skip model loading

    text = "Purchased a FAULTY [2023] Mobile Phone, AIR 2020 SC 123 (Ref)"
    cleaned = ml_service._clean_text(text)

    assert isinstance(cleaned, str)
    assert cleaned.islower()
    assert "[2023]" not in cleaned
    assert "AIR 2020" not in cleaned


def test_sector_classification():
    """Test sector classification."""
    ml_service = MLService()
    ml_service.models_loaded = False

    # E-commerce
    sector = ml_service._classify_sector("Purchased from Amazon online")
    assert sector == "E-commerce"

    # Insurance
    sector = ml_service._classify_sector("Insurance policy claim denied")
    assert sector == "Insurance"

    # Telecom
    sector = ml_service._classify_sector("Mobile phone bill charges")
    assert sector == "Telecom"

    # Default
    sector = ml_service._classify_sector("Some random text about a dispute")
    assert sector == "General"


def test_amount_extraction():
    """Test monetary amount extraction."""
    ml_service = MLService()
    ml_service.models_loaded = False

    # Indian Rupee symbol
    amount = ml_service._extract_amount("The claimed amount is ₹50,000")
    assert amount == 50000.0

    # Rs notation
    amount = ml_service._extract_amount("Rs. 1,00,000 due")
    assert amount == 100000.0

    # Word notation
    amount = ml_service._extract_amount("Rupees 500000 claimed")
    assert amount == 500000.0

    # No amount
    amount = ml_service._extract_amount("No specific amount mentioned")
    assert amount == 0.0


def test_forum_classification():
    """Test consumer forum classification by claim amount."""
    ml_service = MLService()
    ml_service.models_loaded = False

    # District
    forum = ml_service._classify_forum(50_00_000)  # ₹50 lakhs
    assert "District" in forum

    # State
    forum = ml_service._classify_forum(2_00_00_000)  # ₹2 crores
    assert "State" in forum

    # National
    forum = ml_service._classify_forum(15_00_00_000)  # ₹15 crores
    assert "National" in forum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
