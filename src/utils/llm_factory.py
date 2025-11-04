"""
LLM Factory for creating LLM instances based on provider
"""
from typing import Optional
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from src.utils.logger import app_logger


class LLMFactory:
    """Factory class for creating LLM instances"""
    
    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs
    ):
        """
        Create LLM instance based on provider
        
        Args:
            provider: LLM provider (groq/gemini)
            model: Model name
            temperature: Temperature for generation
            **kwargs: Additional arguments for LLM
            
        Returns:
            LLM instance
        """
        provider = provider or settings.LLM_PROVIDER
        model = model or settings.LLM_MODEL
        
        app_logger.info(f"Creating LLM with provider: {provider}, model: {model}")
        
        if provider.lower() == "groq":
            return LLMFactory._create_groq_llm(model, temperature, **kwargs)
        elif provider.lower() in ["gemini", "google"]:
            return LLMFactory._create_gemini_llm(model, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def _create_groq_llm(model: str, temperature: float, **kwargs):
        """Create Groq LLM instance"""
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=model,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _create_gemini_llm(model: str, temperature: float, **kwargs):
        """Create Gemini LLM instance"""
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GOOGLE_API_KEY,
            model=model,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def get_model_list(provider: str) -> list:
        """Get list of available models for a provider"""
        if provider.lower() == "groq":
            return settings.GROQ_MODELS
        elif provider.lower() in ["gemini", "google"]:
            return settings.GEMINI_MODELS
        else:
            return []
