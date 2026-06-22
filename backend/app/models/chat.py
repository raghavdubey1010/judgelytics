# JUDGELYTICS - FastAPI Backend: ChatMessage Model
# Purpose: SQLAlchemy model for chat history
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
SQLAlchemy ChatMessage model for Judgelytics backend.

Represents chat conversation history between users and the system.
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from ..database import Base


class ChatMessage(Base):
    """
    Chat message model for storing conversation history.

    Attributes:
        message_id (str): Unique message identifier
        user_uid (str): Foreign key to User
        case_id (str): Optional reference to Case
        sender (str): "user" or "assistant"
        content (str): Message content
        created_at (datetime): Message timestamp
    """

    __tablename__ = "chat_messages"

    message_id = Column(String(50), primary_key=True, unique=True, index=True)
    user_uid = Column(String(20), ForeignKey("users.uid"), nullable=False, index=True)
    case_id = Column(String(50), ForeignKey("cases.case_id"), nullable=True)
    
    sender = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="chat_messages")

    def __repr__(self) -> str:
        return f"<ChatMessage message_id={self.message_id} sender={self.sender}>"
