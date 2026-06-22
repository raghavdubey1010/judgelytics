# JUDGELYTICS - FastAPI Backend: Auth Router
# Purpose: Authentication endpoints
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Authentication endpoints for Judgelytics backend.

Provides user registration, login, and profile endpoints.
"""

import logging
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, status, Depends

from ..schemas.auth import RegisterRequest, LoginRequest, UserResponse, TokenResponse
from ..models.user import User
from ..database import get_db
from ..services.auth_service import auth_service
from ..core.security import get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user"
)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Register a new user account.

    Args:
        request: Registration request with name, email, phone, password
        db: Database session

    Returns:
        UserResponse: New user profile with access token
    """

    try:
        logger.info(f"Registration attempt for email: {request.email}")

        # Check if email already exists
        result = await db.execute(select(User).where(User.email == request.email))
        existing_user = result.scalar_one_or_none()

        if existing_user:
            logger.warning(f"Email already registered: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Generate unique UID
        uid = auth_service.generate_user_uid()

        # Hash password
        hashed_password = auth_service.hash_password(request.password)

        # Create new user
        new_user = User(
            uid=uid,
            name=request.name,
            email=request.email,
            phone=request.phone,
            hashed_password=hashed_password,
            is_active=True
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # Generate access token
        access_token = auth_service.create_access_token(uid)

        logger.info(f"User registered successfully: {uid}")

        return UserResponse(
            uid=new_user.uid,
            name=new_user.name,
            email=new_user.email,
            phone=new_user.phone,
            access_token=access_token,
            token_type="bearer",
            case_count=0
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post(
    "/login",
    response_model=UserResponse,
    summary="Login user"
)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Login with email and password.

    Args:
        request: Login request with email and password
        db: Database session

    Returns:
        UserResponse: User profile with access token
    """

    try:
        logger.info(f"Login attempt for email: {request.email}")

        # Find user by email
        result = await db.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()

        if not user:
            logger.warning(f"User not found: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Verify password
        if not auth_service.verify_password(request.password, user.hashed_password):
            logger.warning(f"Invalid password for user: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        if not user.is_active:
            logger.warning(f"Inactive user login attempt: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )

        # Generate access token
        access_token = auth_service.create_access_token(user.uid)

        # Count cases
        from ..models.case import Case
        result = await db.execute(select(Case).where(Case.user_uid == user.uid))
        cases = result.scalars().all()
        case_count = len(cases)

        logger.info(f"User logged in successfully: {user.uid}")

        return UserResponse(
            uid=user.uid,
            name=user.name,
            email=user.email,
            phone=user.phone,
            access_token=access_token,
            token_type="bearer",
            case_count=case_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile"
)
async def get_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Get current authenticated user's profile.

    Args:
        current_user: Current authenticated user
        db: Database session

    Returns:
        UserResponse: User profile information
    """

    try:
        # Count cases
        from ..models.case import Case
        result = await db.execute(select(Case).where(Case.user_uid == current_user.uid))
        cases = result.scalars().all()
        case_count = len(cases)

        logger.info(f"Profile retrieved for user: {current_user.uid}")

        return UserResponse(
            uid=current_user.uid,
            name=current_user.name,
            email=current_user.email,
            phone=current_user.phone,
            access_token="",  # Not needed for profile endpoint
            token_type="bearer",
            case_count=case_count
        )

    except Exception as e:
        logger.error(f"Failed to get user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )
