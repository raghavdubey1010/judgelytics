# JUDGELYTICS - FastAPI Backend: Database Module
# Purpose: SQLAlchemy database configuration and session management
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Database configuration and session management for Judgelytics backend.

Implements async SQLAlchemy with SQLite (aiosqlite) for zero-setup development.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create declarative base for models
Base = declarative_base()

# Detect database type
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")


class Database:
    """
    Database connection and session management.

    Handles async database operations. SQLite uses StaticPool,
    PostgreSQL uses connection pooling.
    """

    def __init__(self, database_url: str):
        """
        Initialize database connection.

        Args:
            database_url (str): Database connection URL
        """
        self.database_url = database_url

        if _is_sqlite:
            # SQLite: use StaticPool for async compatibility
            from sqlalchemy.pool import StaticPool
            self.engine = create_async_engine(
                database_url,
                echo=settings.DEBUG,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            # PostgreSQL / production
            self.engine = create_async_engine(
                database_url,
                echo=settings.DEBUG,
                pool_size=20,
                max_overflow=0,
                pool_pre_ping=True,
                pool_recycle=3600,
            )

        # Create async session factory
        self.SessionLocal = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        logger.info(f"Database initialized: {database_url[:60]}...")

    async def create_tables(self):
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    async def drop_tables(self):
        """Drop all database tables (use with caution)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")

    async def close(self):
        """Close database connection pool."""
        await self.engine.dispose()
        logger.info("Database connection closed")

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session.

        Yields:
            AsyncSession: Database session for operations
        """
        async with self.SessionLocal() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {str(e)}")
                raise
            finally:
                await session.close()


# Create global database instance
db = Database(settings.DATABASE_URL)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to inject database session.

    Yields:
        AsyncSession: Database session
    """
    async for session in db.get_session():
        yield session
