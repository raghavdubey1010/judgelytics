# JUDGELYTICS - FastAPI Backend: Report Model
# Purpose: SQLAlchemy model for generated reports
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
SQLAlchemy Report model for Judgelytics backend.

Represents PDF reports generated for cases.
"""

from datetime import datetime
from sqlalchemy import Column, String, LargeBinary, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from ..database import Base


class Report(Base):
    """
    Report model for storing generated PDF reports.

    Attributes:
        report_id (str): Unique report identifier
        user_uid (str): Foreign key to User
        case_id (str): Foreign key to Case
        pdf_content (bytes): PDF file content (BLOB)
        file_name (str): Original filename
        created_at (datetime): Report generation timestamp
    """

    __tablename__ = "reports"

    report_id = Column(String(50), primary_key=True, unique=True, index=True)
    user_uid = Column(String(20), ForeignKey("users.uid"), nullable=False, index=True)
    case_id = Column(String(50), ForeignKey("cases.case_id"), nullable=False, index=True)
    
    pdf_content = Column(LargeBinary, nullable=True)  # PDF as BLOB
    file_name = Column(String(255), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="reports")
    case = relationship("Case", back_populates="reports")

    def __repr__(self) -> str:
        return f"<Report report_id={self.report_id} case_id={self.case_id}>"
