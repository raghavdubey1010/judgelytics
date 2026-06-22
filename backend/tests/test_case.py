# JUDGELYTICS - FastAPI Backend: Case Tests
# Purpose: Unit tests for case analysis endpoints
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Test suite for case analysis endpoints.

Tests case creation, retrieval, and history.
"""

import pytest
from httpx import AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_case_analysis_requires_auth():
    """Test that case analysis requires authentication."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/case/analyze",
            json={
                "description": "Purchased a faulty mobile phone. Refused replacement and refund.",
                "category": "Product Quality",
                "sector": "E-commerce",
                "claim_amount": 45000.00,
                "evidence_count": 5,
                "has_legal_notice": "Yes",
                "opponent_type": "Company"
            }
        )

        assert response.status_code == 403  # Forbidden (no auth)


@pytest.mark.asyncio
async def test_case_analysis_validation():
    """Test case analysis request validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/case/analyze",
            json={
                "description": "Short",  # Too short
                "category": "Product Quality",
                "sector": "E-commerce",
                "claim_amount": 45000.00,
                "evidence_count": 5,
                "has_legal_notice": "Yes",
                "opponent_type": "Company"
            }
        )

        # Should fail validation
        assert response.status_code in [422, 403]


@pytest.mark.asyncio
async def test_case_history_requires_auth():
    """Test that case history requires authentication."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/case/history")

        assert response.status_code == 403  # Forbidden (no auth)


@pytest.mark.asyncio
async def test_case_detail_requires_auth():
    """Test that case detail requires authentication."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/case/CSE-20240115-ABCDEF")

        assert response.status_code == 403  # Forbidden (no auth)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
