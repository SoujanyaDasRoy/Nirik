"""
Configuration management for the Nirikhshon backend.
Centralizes all configuration settings to avoid hardcoded values.
"""
import os
from datetime import timedelta
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration class."""
    # Flask settings
    SECRET_KEY = os.environ.get("FLASK_SECRET", "dev-secret-key-change-in-production")
    SESSION_COOKIE_SAMESITE = "None"
    SESSION_COOKIE_SECURE = True
    MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB
    # Application version
    VERSION = os.environ.get("APP_VERSION", "1.0.0")

    # Database settings
    DATABASE_PATH = os.environ.get(
        "DATABASE_PATH",
        str(BASE_DIR / "backend" / "patients.db")
    )

    # Model settings (paths are set but models are not loaded yet)
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        str(BASE_DIR / "results" / "student_best.weights.h5")
    )
    UNET_MODEL_PATH = os.environ.get(
        "UNET_MODEL_PATH",
        str(BASE_DIR / "results" / "attention_unet.keras")
    )
    QUANTIZED_MODEL_PATH = os.environ.get("QUANTIZED_MODEL_PATH")

    # Upload settings
    UPLOAD_FOLDER = os.environ.get(
        "UPLOAD_FOLDER",
        str(BASE_DIR / "backend" / "uploads")
    )
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm", "dicom"}

    # CORS settings
    ALLOWED_ORIGINS_ENV = os.environ.get(
        "ALLOWED_ORIGIN",
        "http://localhost:3000,http://127.0.0.1:3000"
    )

    # Prediction settings
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
    BATCH_TIMEOUT = float(os.environ.get("BATCH_TIMEOUT", "0.01"))
    OPTIMAL_THRESHOLD = float(os.environ.get("OPTIMAL_THRESHOLD", "0.5"))

    # ROI Extraction Settings
    ROI_MIN_CONTOUR_AREA = int(os.environ.get("ROI_MIN_CONTOUR_AREA", "15"))
    ROI_APPROXIMATION_EPSILON = float(os.environ.get("ROI_APPROXIMATION_EPSILON", "0.015"))
    ROI_MAX_REGIONS = int(os.environ.get("ROI_MAX_REGIONS", "6"))
    ROI_THRESHOLD_TB_FACTOR = float(os.environ.get("ROI_THRESHOLD_TB_FACTOR", "0.60"))
    ROI_THRESHOLD_NON_TB_FACTOR = float(os.environ.get("ROI_THRESHOLD_NON_TB_FACTOR", "0.70"))
    ROI_MIN_THRESHOLD = float(os.environ.get("ROI_MIN_THRESHOLD", "0.15"))

    # Anatomical Localization Zones (normalized coordinates 0-1)
    ROI_UPPER_ZONE_BOUNDARY = float(os.environ.get("ROI_UPPER_ZONE_BOUNDARY", "0.40"))
    ROI_MIDDLE_ZONE_BOUNDARY = float(os.environ.get("ROI_MIDDLE_ZONE_BOUNDARY", "0.68"))

    # Shape Analysis Filters
    MIN_ASPECT_RATIO = float(os.environ.get("MIN_ASPECT_RATIO", "0.1"))
    MAX_ASPECT_RATIO = float(os.environ.get("MAX_ASPECT_RATIO", "10.0"))
    MIN_SOLIDITY = float(os.environ.get("MIN_SOLIDITY", "0.1"))
    MIN_CONVEXITY = float(os.environ.get("MIN_CONVEXITY", "0.1"))

    # API versioning
    API_VERSION = "v1"

    # Logging settings
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    LOG_FILE = os.environ.get(
        "LOG_FILE",
        str(BASE_DIR / "logs" / "backend.log")
    )

    # Security settings
    RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
    SESSION_TIMEOUT = timedelta(minutes=30)

    # Feature flags
    ENABLE_WEBSOCKETS = os.environ.get("ENABLE_WEBSOCKETS", "true").lower() == "true"
    ENABLE_DEMO_MODE = os.environ.get("ENABLE_DEMO_MODE", "false").lower() == "true"

    @staticmethod
    def init_app(app):
        """Initialize application with configuration-specific setup."""
        pass


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = "WARNING"


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DATABASE_PATH = ":memory:"
    LOG_LEVEL = "ERROR"


def get_config() -> Config:
    """
    Get configuration based on environment.

    Returns:
        Config instance
    """
    env = os.environ.get("FLASK_ENV", "production").lower()
    if env == "development":
        return DevelopmentConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return ProductionConfig()