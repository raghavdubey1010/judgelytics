# JUDGELYTICS - FastAPI Backend: Auth Schemas
# Purpose: Pydantic validation schemas for authentication
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Pydantic schemas for authentication endpoints.

Validates request/response data for user registration and login.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class RegisterRequest(BaseModel):
    """Schema for user registration request."""

    name: str = Field(..., min_length=2, max_length=255, description="User's full name")
    email: EmailStr = Field(..., description="User's email")
    phone: Optional[str] = Field(None, pattern=r'^[+]?[0-9]{10,}$', description="Phone number")
    password: str = Field(..., min_length=8, max_length=255, description="Password (min 8 chars)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Rakhi Tiwari",
                "email": "rakhi@example.com",
                "phone": "+919876543210",
                "password": "secure_password_123"
            }
        }


class LoginRequest(BaseModel):
    """Schema for user login request."""

    email: EmailStr = Field(..., description="User's email")
    password: str = Field(..., description="User's password")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "rakhi@example.com",
                "password": "secure_password_123"
            }
        }


class UserResponse(BaseModel):
    """Schema for user profile response."""

    uid: str = Field(..., description="Unique user identifier")
    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email")
    phone: Optional[str] = Field(None, description="Phone number")
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    case_count: Optional[int] = Field(0, description="Number of cases analyzed")

    class Config:
        json_schema_extra = {
            "example": {
                "uid": "JDG-ABC123",
                "name": "Rakhi Tiwari",
                "email": "rakhi@example.com",
                "phone": "+919876543210",
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "case_count": 5
            }
        }


class TokenResponse(BaseModel):
    """Schema for token response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }
