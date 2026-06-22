# JUDGELYTICS - FastAPI Backend: Main Application
# Purpose: FastAPI application factory and initialization
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Judgelytics FastAPI backend application.

Creates FastAPI app instance with all routers, middleware, and startup events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .database import db
from . import models  # Ensure ORM models are registered before DB usage.
from .services.ml_service import initialize_ml_service
from .routers import auth, case, report, chat, legal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events.
    """

    # Startup
    logger.info("Starting Judgelytics API...")

    try:
        # Initialize database tables
        await db.create_tables()
        logger.info("Database tables created")

        # Initialize ML models
        ml_service = initialize_ml_service()
        app.state.ml_service = ml_service
        app.state.models_loaded = ml_service.models_loaded

        if ml_service.models_loaded:
            logger.info("ML models loaded successfully")
        else:
            logger.warning("ML models not loaded - predictions will fail")

        app.state.started_at = __import__('datetime').datetime.utcnow()

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

    logger.info("Judgelytics API started successfully!")

    yield

    # Shutdown
    logger.info("Shutting down Judgelytics API...")

    try:
        await db.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

    logger.info("Judgelytics API shut down")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """

    # Create app
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_CREDENTIALS,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )

    logger.info("CORS middleware configured")

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Health check endpoint.

        Returns:
            dict: Health status including models loaded state
        """
        return {
            "status": "ok",
            "models_loaded": getattr(app.state, "models_loaded", False),
            "version": settings.APP_VERSION
        }

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """
        Root endpoint with basic information.

        Returns:
            dict: API information
        """
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": settings.APP_DESCRIPTION,
            "docs": "/docs",
            "redoc": "/redoc"
        }

    # Include routers with prefix
    app.include_router(auth.router, prefix="/api/v1/auth")
    app.include_router(case.router, prefix="/api/v1/case")
    app.include_router(report.router, prefix="/api/v1/report")
    app.include_router(chat.router, prefix="/api/v1/chat")
    app.include_router(legal.router, prefix="/api/v1/legal")

    logger.info("API routers registered")

    return app


# Create app instance
app = create_app()

# Expose app for uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.RELOAD,
        debug=settings.DEBUG
    )
