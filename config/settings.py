"""
Configuration settings for the Math Routing Agent
"""
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings"""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    # Available Models
    GROQ_MODELS: List[str] = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile", 
        "llama3-70b-8192",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    
    GEMINI_MODELS: List[str] = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = "logs"
    
    # Data Directories
    DATA_DIR: str = "data"
    FEEDBACK_DIR: str = os.path.join(DATA_DIR, "feedback")
    HISTORY_DIR: str = os.path.join(DATA_DIR, "history")
    VECTOR_DB_PATH: str = os.path.join(DATA_DIR, "vector_db")
    UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
    
    # Temperature settings for different problem types
    TEMPERATURE_CONFIG: Dict[str, float] = {
        "algebra": 0.3,
        "calculus": 0.3,
        "geometry": 0.4,
        "statistics": 0.4,
        "general": 0.5
    }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available models"""
        return cls.GROQ_MODELS + cls.GEMINI_MODELS
    
    @classmethod
    def validate_model(cls, model: str) -> bool:
        """Validate if model is available"""
        return model in cls.get_available_models()


settings = Settings()
