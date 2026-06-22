# JUDGELYTICS - FastAPI Backend: User Model
# Purpose: SQLAlchemy model for users
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
SQLAlchemy User model for Judgelytics backend.

Represents platform users with authentication credentials.
"""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import relationship

from ..database import Base


class User(Base):
    """
    User model for authentication and account management.

    Attributes:
        uid (str): Unique identifier (JDG-XXXXXX)
        name (str): User's full name
        email (str): Email address (unique)
        phone (str): Phone number
        hashed_password (str): Bcrypt hashed password
        is_active (bool): Whether account is active
        created_at (datetime): Account creation timestamp
        updated_at (datetime): Last update timestamp
    """

    __tablename__ = "users"

    uid = Column(String(20), primary_key=True, unique=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    cases = relationship("Case", back_populates="user", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User uid={self.uid} email={self.email}>"
