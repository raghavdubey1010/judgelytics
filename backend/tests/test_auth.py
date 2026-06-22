# JUDGELYTICS - FastAPI Backend: Auth Tests
# Purpose: Unit tests for authentication endpoints
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Test suite for authentication endpoints.

Tests user registration, login, and profile retrieval.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.models.user import User

# Test database URL (in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSession = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with TestSession() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    yield TestSession

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_register_user(test_db):
    """Test user registration."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User",
                "email": "test@example.com",
                "phone": "+919876543210",
                "password": "secure_password_123"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["access_token"]
        assert data["uid"].startswith("JDG-")


@pytest.mark.asyncio
async def test_register_duplicate_email(test_db):
    """Test registration with duplicate email."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # First registration
        await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User",
                "email": "test@example.com",
                "phone": "+919876543210",
                "password": "secure_password_123"
            }
        )

        # Duplicate registration
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User 2",
                "email": "test@example.com",
                "phone": "+919876543211",
                "password": "another_password_123"
            }
        )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_user(test_db):
    """Test user login."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Register
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User",
                "email": "test@example.com",
                "phone": "+919876543210",
                "password": "secure_password_123"
            }
        )

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "secure_password_123"
            }
        )

        assert login_response.status_code == 200
        data = login_response.json()
        assert data["email"] == "test@example.com"
        assert data["access_token"]


@pytest.mark.asyncio
async def test_login_wrong_password(test_db):
    """Test login with wrong password."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Register
        await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User",
                "email": "test@example.com",
                "phone": "+919876543210",
                "password": "secure_password_123"
            }
        )

        # Login with wrong password
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "wrong_password"
            }
        )

        assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
