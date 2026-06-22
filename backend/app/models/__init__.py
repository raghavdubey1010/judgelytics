"""
SQLAlchemy models for Judgelytics backend.
Database models for users, cases, reports, and chat history.
"""

# Import models so SQLAlchemy class registry can resolve relationship strings
# like "Case", "Report", and "ChatMessage" from User.
from .user import User
from .case import Case
from .report import Report
from .chat import ChatMessage

__all__ = ["User", "Case", "Report", "ChatMessage"]
