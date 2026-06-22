# JUDGELYTICS - FastAPI Backend: Configuration Module
# Purpose: Application settings and configuration management
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Configuration management for Judgelytics backend.

Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from typing import List

# Load .env file automatically
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Get project root (judgelyctics/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings:
    """
    Application settings from environment variables.

    Default values are provided for development. Customize via .env file
    or environment variables in production.
    """

    # Application metadata
    APP_NAME: str = os.getenv("APP_NAME", "Judgelytics API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    APP_DESCRIPTION: str = "AI-Powered Legal Judgment Prediction for Indian Consumer Courts"

    # Database — defaults to SQLite for zero-setup development
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{PROJECT_ROOT / 'judgelytics.db'}"
    )

    # Security
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "judgelytics-super-secret-jwt-key-2026"
    )
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

    # CORS
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500,http://localhost:5173,http://127.0.0.1:3000"
        ).split(",")
    ] + ["*"]  # Allow all for development

    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]

    # ML Models — resolve relative to project root
    ML_MODELS_DIR: str = os.getenv(
        "ML_MODELS_DIR",
        str(PROJECT_ROOT / "ml_pipeline" / "saved_models")
    )

    # Resolve relative paths to absolute
    @property
    def ml_models_path(self) -> Path:
        p = Path(self.ML_MODELS_DIR)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    # Groq API
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # OpenAI API (takes priority over Groq if key is set)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Anthropic API (Claude) (takes highest priority if set)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    # API Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "True").lower() == "true"

    @property
    def models_loaded(self) -> bool:
        """Check if at least TF-IDF and one classifier exist."""
        required_models = [
            "tfidf_vectorizer.pkl",
            "logistic_regression.pkl",
        ]
        for model in required_models:
            model_path = self.ml_models_path / model
            if not model_path.exists():
                return False
        return True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get current settings.

    Returns:
        Settings: Application settings
    """
    return settings
