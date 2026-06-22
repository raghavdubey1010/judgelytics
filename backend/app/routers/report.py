# JUDGELYTICS - FastAPI Backend: Report Router
# Purpose: Report generation endpoints
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Report generation endpoints for Judgelytics backend.

Provides PDF report generation and download endpoints.
"""

import logging
import uuid
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import Response

from ..schemas.report import ReportRequest, ReportResponse
from ..models.user import User
from ..models.case import Case
from ..models.report import Report
from ..database import get_db
from ..services.pdf_service import get_pdf_service
from ..core.security import get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Reports"])


@router.post(
    "/generate",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate PDF report for a case"
)
async def generate_report(
    request: ReportRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ReportResponse:
    """
    Generate a PDF report for a case analysis.

    Args:
        request: Report generation request
        current_user: Authenticated user
        db: Database session

    Returns:
        ReportResponse: Report details with download URL
    """

    try:
        logger.info(f"Generating report for case: {request.case_id}")

        # Find case
        result = await db.execute(
            select(Case).where(Case.case_id == request.case_id)
        )
        case = result.scalar_one_or_none()

        if not case:
            logger.warning(f"Case not found: {request.case_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found"
            )

        # Verify ownership
        if case.user_uid != current_user.uid:
            logger.warning(f"Unauthorized report generation for case: {request.case_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to generate report for this case"
            )

        # Prepare data for PDF generation
        case_data = {
            "case_id": case.case_id,
            "category": case.category,
            "sector": case.sector,
            "claim_amount": case.claim_amount,
            "description": case.description,
        }

        prediction_data = {
            "outcome": case.predicted_outcome,
            "win_probability": case.win_probability,
            "win_probability_pct": int(case.win_probability * 100),
            "confidence": case.confidence_level,
            "recommended_forum": case.recommended_forum,
            "applicable_sections": case.applicable_sections or [],
            "evidence_strength": case.evidence_strength,
            "filing_fee": case.filing_fee,
            "similar_cases_count": case.similar_cases_count
        }

        user_data = {
            "uid": current_user.uid,
            "name": current_user.name,
            "email": current_user.email
        }

        # Generate PDF
        pdf_service = get_pdf_service()
        pdf_bytes = pdf_service.generate_report_pdf(case_data, prediction_data, user_data)

        # Generate report ID
        report_id = f"RPT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        file_name = f"Judgelytics_Report_{case.case_id}.pdf"

        # Save report to database
        new_report = Report(
            report_id=report_id,
            user_uid=current_user.uid,
            case_id=request.case_id,
            pdf_content=pdf_bytes,
            file_name=file_name
        )

        db.add(new_report)
        await db.commit()
        await db.refresh(new_report)

        logger.info(f"Report generated and saved: {report_id}")

        return ReportResponse(
            report_id=report_id,
            case_id=request.case_id,
            file_name=file_name,
            download_url=f"/api/v1/report/download/{report_id}",
            file_size=len(pdf_bytes)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get(
    "/download/{report_id}",
    summary="Download PDF report"
)
async def download_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Download a generated PDF report.

    Args:
        report_id: Report identifier
        current_user: Authenticated user
        db: Database session

    Returns:
        Response: PDF file bytes with correct content-type
    """

    try:
        logger.info(f"Downloading report: {report_id}")

        # Find report
        result = await db.execute(
            select(Report).where(Report.report_id == report_id)
        )
        report = result.scalar_one_or_none()

        if not report:
            logger.warning(f"Report not found: {report_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found"
            )

        # Verify ownership
        if report.user_uid != current_user.uid:
            logger.warning(f"Unauthorized download of report: {report_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to download this report"
            )

        logger.info(f"Report downloaded: {report_id}")

        # Return PDF bytes directly — fixes FileResponse(path=None) bug
        return Response(
            content=report.pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{report.file_name}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report download failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download report"
        )
