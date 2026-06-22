# JUDGELYTICS - FastAPI Backend: Case Router
# Purpose: Case analysis endpoints
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Case analysis endpoints for Judgelytics backend.

Provides case prediction, history, and detail endpoints.
"""

import logging
import uuid
from datetime import datetime
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, status, Depends

from ..schemas.case import CaseRequest, CaseResponse, CaseListItem, PredictionResult
from ..models.user import User
from ..models.case import Case
from ..database import get_db
from ..services.ml_service import get_ml_service
from ..core.security import get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Cases"])


@router.post(
    "/analyze",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze a new case"
)
async def analyze_case(
    request: CaseRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> CaseResponse:
    """
    Analyze a new legal case and generate prediction.

    Args:
        request: Case information
        current_user: Authenticated user
        db: Database session

    Returns:
        CaseResponse: Prediction results and case ID
    """

    try:
        logger.info(f"Analyzing case for user: {current_user.uid}")

        # Get ML service (falls back to rule-based engine if ML models not loaded)
        ml_service = get_ml_service()

        # Generate prediction
        case_data = {
            "description": request.description,
            "category": request.category,
            "sector": request.sector,
            "claim_amount": request.claim_amount,
            "evidence_count": request.evidence_count,
            "has_legal_notice": request.has_legal_notice,
            "opponent_type": request.opponent_type
        }

        prediction_result = ml_service.predict(case_data)

        # Generate case ID
        case_id = f"CSE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        # Save case to database
        new_case = Case(
            case_id=case_id,
            user_uid=current_user.uid,
            description=request.description,
            category=request.category,
            sector=request.sector,
            claim_amount=request.claim_amount,
            evidence_count=request.evidence_count,
            has_legal_notice=request.has_legal_notice,
            opponent_type=request.opponent_type,
            predicted_outcome=prediction_result["outcome"],
            win_probability=prediction_result["win_probability"],
            confidence_level=prediction_result["confidence"],
            recommended_forum=prediction_result["recommended_forum"],
            applicable_sections=prediction_result["applicable_sections"],
            filing_fee=prediction_result["filing_fee"],
            evidence_strength=prediction_result["evidence_strength"],
            similar_cases_count=prediction_result["similar_cases_count"]
        )

        db.add(new_case)
        await db.commit()
        await db.refresh(new_case)

        logger.info(f"Case analyzed and saved: {case_id}")

        # Create response
        prediction = PredictionResult(**prediction_result)

        return CaseResponse(
            case_id=case_id,
            prediction=prediction,
            created_at=new_case.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Case analysis failed: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze case"
        )


@router.get(
    "/history",
    response_model=List[CaseListItem],
    summary="Get user's case history"
)
async def get_case_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0
) -> List[CaseListItem]:
    """
    Get list of cases analyzed by user.

    Args:
        current_user: Authenticated user
        db: Database session
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        List[CaseListItem]: List of cases for the user
    """

    try:
        logger.info(f"Fetching case history for user: {current_user.uid}")

        # Query cases
        result = await db.execute(
            select(Case)
            .where(Case.user_uid == current_user.uid)
            .order_by(Case.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        cases = result.scalars().all()

        logger.info(f"Found {len(cases)} cases for user: {current_user.uid}")

        return [
            CaseListItem(
                case_id=case.case_id,
                category=case.category,
                sector=case.sector,
                claim_amount=case.claim_amount,
                predicted_outcome=case.predicted_outcome,
                win_probability=case.win_probability,
                created_at=case.created_at
            )
            for case in cases
        ]

    except Exception as e:
        logger.error(f"Failed to fetch case history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch case history"
        )


@router.get(
    "/{case_id}",
    response_model=CaseResponse,
    summary="Get case details"
)
async def get_case_detail(
    case_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> CaseResponse:
    """
    Get detailed information about a specific case.

    Args:
        case_id: Case identifier
        current_user: Authenticated user
        db: Database session

    Returns:
        CaseResponse: Full case details and prediction

    Raises:
        HTTPException: If case not found or not authorized
    """

    try:
        logger.info(f"Fetching case details: {case_id}")

        # Find case
        result = await db.execute(
            select(Case).where(Case.case_id == case_id)
        )
        case = result.scalar_one_or_none()

        if not case:
            logger.warning(f"Case not found: {case_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found"
            )

        # Verify ownership
        if case.user_uid != current_user.uid:
            logger.warning(f"Unauthorized access to case: {case_id} by user: {current_user.uid}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this case"
            )

        # Build response
        prediction = PredictionResult(
            outcome=case.predicted_outcome,
            win_probability=case.win_probability,
            win_probability_pct=int(case.win_probability * 100),
            confidence=case.confidence_level,
            recommended_forum=case.recommended_forum,
            applicable_sections=case.applicable_sections or [],
            evidence_strength=case.evidence_strength,
            filing_fee=case.filing_fee,
            similar_cases_count=case.similar_cases_count
        )

        logger.info(f"Case details fetched: {case_id}")

        return CaseResponse(
            case_id=case.case_id,
            prediction=prediction,
            created_at=case.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch case details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch case details"
        )
