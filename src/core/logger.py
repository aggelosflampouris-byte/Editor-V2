"""Centralized logging infrastructure for the Antigravity Video Engine."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import os


def setup_logger() -> logging.Logger:
    """Initialize and configure the root logger with console and rotating file output."""
    logger = logging.getLogger("antigravity")
    
    # Avoid duplicate handlers if already initialized
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # Ensure log directory exists
    log_dir = Path(os.getcwd()) / "workspace" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    # Formatting
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler (10 MB max size, keep 5 backups)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Singleton logger instance
logger = setup_logger()
