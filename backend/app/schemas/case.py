# JUDGELYTICS - FastAPI Backend: Case Schemas
# Purpose: Pydantic validation schemas for case analysis
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Pydantic schemas for case analysis endpoints.

Validates request/response data for case predictions.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class CaseRequest(BaseModel):
    """Schema for case analysis request."""

    description: str = Field(..., min_length=50, max_length=10000, description="Case facts and description")
    category: str = Field(..., description="Case category (e.g., 'Product Quality', 'Banking')")
    sector: str = Field(..., description="Business sector affected")
    claim_amount: float = Field(..., gt=0, description="Claimed amount in rupees")
    evidence_count: int = Field(default=0, ge=0, description="Number of evidence items")
    has_legal_notice: str = Field(default="No", description="'Yes' or 'No'")
    opponent_type: str = Field(..., description="Type of opponent (e.g., 'Company', 'Individual')")

    class Config:
        json_schema_extra = {
            "example": {
                "description": "Purchased a faulty mobile phone from an e-commerce platform. Despite requesting a replacement within warranty period, the company refused and also denied refund.",
                "category": "Product Quality",
                "sector": "E-commerce",
                "claim_amount": 45000.00,
                "evidence_count": 5,
                "has_legal_notice": "Yes",
                "opponent_type": "Company"
            }
        }


class PredictionResult(BaseModel):
    """Schema for prediction results."""

    outcome: str = Field(..., description="Predicted outcome: Allowed|Dismissed|Partially Allowed")
    win_probability: float = Field(..., ge=0.0, le=1.0, description="Win probability (0-1)")
    win_probability_pct: int = Field(..., ge=0, le=100, description="Win probability as percentage")
    confidence: str = Field(..., description="Confidence level: high|medium|low")
    recommended_forum: str = Field(..., description="Recommended consumer forum")
    applicable_sections: List[str] = Field(..., description="Applicable legal sections")
    evidence_strength: str = Field(..., description="Evidence strength: Strong|Moderate|Weak")
    filing_fee: str = Field(..., description="Filing fee amount (e.g., '₹500')")
    similar_cases_count: int = Field(..., description="Similar cases in training data")

    class Config:
        json_schema_extra = {
            "example": {
                "outcome": "Allowed",
                "win_probability": 0.78,
                "win_probability_pct": 78,
                "confidence": "high",
                "recommended_forum": "District Consumer Commission",
                "applicable_sections": ["Section 35", "Section 2(11)"],
                "evidence_strength": "Strong",
                "filing_fee": "₹500",
                "similar_cases_count": 142
            }
        }


class CaseResponse(BaseModel):
    """Schema for case analysis response."""

    case_id: str = Field(..., description="Unique case ID")
    prediction: PredictionResult = Field(..., description="Prediction results")
    created_at: datetime = Field(..., description="Analysis timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "CSE-2024-001",
                "prediction": {
                    "outcome": "Allowed",
                    "win_probability": 0.78,
                    "win_probability_pct": 78,
                    "confidence": "high",
                    "recommended_forum": "District Consumer Commission",
                    "applicable_sections": ["Section 35", "Section 2(11)"],
                    "evidence_strength": "Strong",
                    "filing_fee": "₹500",
                    "similar_cases_count": 142
                },
                "created_at": "2024-01-15T10:30:00"
            }
        }


class CaseListItem(BaseModel):
    """Schema for case list item."""

    case_id: str = Field(..., description="Case ID")
    category: str = Field(..., description="Case category")
    sector: str = Field(..., description="Sector")
    claim_amount: float = Field(..., description="Claim amount")
    predicted_outcome: str = Field(..., description="Predicted outcome")
    win_probability: float = Field(..., description="Win probability")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "CSE-2024-001",
                "category": "Product Quality",
                "sector": "E-commerce",
                "claim_amount": 45000.00,
                "predicted_outcome": "Allowed",
                "win_probability": 0.78,
                "created_at": "2024-01-15T10:30:00"
            }
        }
