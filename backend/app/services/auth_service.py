# JUDGELYTICS - FastAPI Backend: Authentication Service
# Purpose: JWT and password management
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Authentication service for Judgelytics backend.

Handles JWT token generation/validation and password hashing.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import uuid

from jose import JWTError, jwt
from passlib.context import CryptContext

from ..config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Password hashing context
# Use pbkdf2_sha256 to avoid platform-specific bcrypt backend issues.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


class AuthService:
    """Authentication and token management service."""

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using configured passlib context.

        Args:
            password (str): Plain text password

        Returns:
            str: Hashed password
        """
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain text password against a hash.

        Args:
            plain_password (str): Plain text password
            hashed_password (str): Hashed password

        Returns:
            bool: True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(
        user_uid: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.

        Args:
            user_uid (str): User unique identifier
            expires_delta (timedelta): Token expiration time. Default: 24 hours

        Returns:
            str: Encoded JWT token
        """

        if expires_delta is None:
            expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        expire = datetime.utcnow() + expires_delta

        to_encode = {
            "user_uid": user_uid,
            "exp": expire,
            "iat": datetime.utcnow()
        }

        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )

        logger.debug(f"Access token created for user {user_uid}")

        return encoded_jwt

    @staticmethod
    def decode_access_token(token: str) -> Optional[str]:
        """
        Decode and validate JWT token.

        Args:
            token (str): JWT token

        Returns:
            Optional[str]: User UID if valid, None if invalid

        Raises:
            JWTError: If token is invalid or expired
        """

        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )

            user_uid = payload.get("user_uid")

            if user_uid is None:
                return None

            logger.debug(f"Token decoded for user {user_uid}")

            return user_uid

        except JWTError as e:
            logger.warning(f"Token decoding failed: {str(e)}")
            raise

    @staticmethod
    def generate_user_uid() -> str:
        """
        Generate unique user identifier.

        Format: JDG-XXXXXX (6 alphanumeric characters)

        Returns:
            str: Generated UID
        """

        # Generate 6 random alphanumeric characters
        random_part = str(uuid.uuid4())[:6].upper()
        uid = f"JDG-{random_part}"

        logger.debug(f"Generated user UID: {uid}")

        return uid


# Create global auth service instance
auth_service = AuthService()


def get_auth_service() -> AuthService:
    """
    Get authentication service instance.

    Returns:
        AuthService: Authentication service
    """
    return auth_service
