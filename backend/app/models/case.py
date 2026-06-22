# JUDGELYTICS - FastAPI Backend: Case Model
# Purpose: SQLAlchemy model for case analyses
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
SQLAlchemy Case model for Judgelytics backend.

Represents case analyses with predictions and metadata.
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from ..database import Base


class Case(Base):
    """
    Case analysis model for storing predictions and metadata.

    Attributes:
        case_id (str): Unique case identifier
        user_uid (str): Foreign key to User
        description (str): Case description/facts
        category (str): Case category
        sector (str): Business sector
        claim_amount (float): Claim amount in rupees
        evidence_count (int): Number of evidence items
        has_legal_notice (bool): Whether legal notice issued
        opponent_type (str): Type of opponent
        predicted_outcome (str): Predicted outcome
        win_probability (float): Win probability (0.0-1.0)
        confidence_level (str): Confidence: "high", "medium", "low"
        recommended_forum (str): Recommended consumer forum
        applicable_sections (list): Applicable legal sections
        filing_fee (str): Filing fee amount
        evidence_strength (str): Evidence strength assessment
        similar_cases_count (int): Similar cases in training data
        created_at (datetime): Analysis timestamp
        updated_at (datetime): Last update timestamp
    """

    __tablename__ = "cases"

    case_id = Column(String(50), primary_key=True, unique=True, index=True)
    user_uid = Column(String(20), ForeignKey("users.uid"), nullable=False, index=True)
    
    # Input fields
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    sector = Column(String(100), nullable=False)
    claim_amount = Column(Float, nullable=False)
    evidence_count = Column(Integer, default=0)
    has_legal_notice = Column(String(10), nullable=False)
    opponent_type = Column(String(50), nullable=False)
    
    # Prediction fields
    predicted_outcome = Column(String(50), nullable=False)
    win_probability = Column(Float, nullable=False)
    confidence_level = Column(String(20), nullable=False)
    recommended_forum = Column(String(100), nullable=False)
    applicable_sections = Column(JSON, nullable=True)  # List of sections
    filing_fee = Column(String(50), nullable=False)
    evidence_strength = Column(String(20), nullable=False)
    similar_cases_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="cases")
    reports = relationship("Report", back_populates="case", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Case case_id={self.case_id} outcome={self.predicted_outcome}>"
