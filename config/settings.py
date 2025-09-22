"""
Configuration management for the healthcare bot.
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings using Pydantic Settings."""
    
    # Application Configuration
    app_name: str = "Healthcare Bot"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"
    secret_key: str = "your_secret_key_here_change_this_in_production"
    
    # OpenAI Configuration
    openai_api_key: str
    gpt_model: str = "gpt-4o"
    gpt_temperature: float = 0.3
    max_tokens: int = 2000
    
    # MongoDB Configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "healthcare_bot"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "healthcare-bot-index"
    
    # Twilio Configuration
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    
    # Serper API Configuration
    serper_api_key: str
    
    # File Upload Configuration
    max_file_size: int = 10485760  # 10MB in bytes
    allowed_file_types: List[str] = ["pdf", "doc", "docx", "jpg", "jpeg", "png", "webp"]
    upload_dir: str = "uploads"
    
    # Database Paths
    sqlite_db_path: str = "data/chat_memory.db"
    
    # Security Configuration
    cors_origins: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure necessary directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.upload_dir,
        Path(settings.sqlite_db_path).parent,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()