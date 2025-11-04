"""
Logging utility for the application
"""
import os
import sys
from loguru import logger
from config.settings import settings


def setup_logger():
    """Setup logger with file and console handlers"""
    
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL
    )
    
    # Add file handler
    logger.add(
        os.path.join(settings.LOG_DIR, "app_{time}.log"),
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=settings.LOG_LEVEL
    )
    
    return logger


# Initialize logger
app_logger = setup_logger()
