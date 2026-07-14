"""
Logging configuration for the Nirikhshon backend.
Sets up structured logging to avoid logging secrets and provide consistent format.
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from .config import Config


def setup_logging(config: Config) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        config: Configuration object

    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    log_file_path = Path(config.LOG_FILE)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Get application logger
    logger = logging.getLogger("nirikshon")
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(console_handler)

    # Log startup message (without secrets)
    logger.info(
        "Logging initialized",
        extra={
            "log_level": config.LOG_LEVEL,
            "log_file": str(log_file_path)
        }
    )

    # Reduce noise from some libraries
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.WARNING)

    return logger