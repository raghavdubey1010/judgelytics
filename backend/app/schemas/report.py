# JUDGELYTICS - FastAPI Backend: Report Schemas
# Purpose: Pydantic validation schemas for reports
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Pydantic schemas for report generation endpoints.

Validates request/response data for PDF report generation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ReportRequest(BaseModel):
    """Schema for report generation request."""

    case_id: str = Field(..., description="Case ID to generate report for")
    include_similar_cases: bool = Field(default=True, description="Include similar cases analysis")
    include_timeline: bool = Field(default=True, description="Include filing process timeline")
    include_checklist: bool = Field(default=True, description="Include document checklist")

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "CSE-2024-001",
                "include_similar_cases": True,
                "include_timeline": True,
                "include_checklist": True
            }
        }


class ReportResponse(BaseModel):
    """Schema for report generation response."""

    report_id: str = Field(..., description="Unique report ID")
    case_id: str = Field(..., description="Associated case ID")
    file_name: str = Field(..., description="Generated filename")
    download_url: str = Field(..., description="URL to download report")
    file_size: Optional[int] = Field(None, description="File size in bytes")

    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "RPT-2024-001",
                "case_id": "CSE-2024-001",
                "file_name": "Judgelytics_Report_CSE_2024_001.pdf",
                "download_url": "/api/v1/report/download/RPT-2024-001",
                "file_size": 125000
            }
        }
