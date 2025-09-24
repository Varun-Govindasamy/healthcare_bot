"""
Settings and configuration management for the Healthcare Bot.
Uses Pydantic Settings for environment variable management.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Configuration
    app_name: str = "Healthcare Bot"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # OpenAI Configuration
    openai_api_key: str
    gpt_model: str = "gpt-4o"
    gpt_temperature: float = 0.3
    max_tokens: int = 2000
    
    # MongoDB Configuration
    mongodb_url: str
    mongodb_database: str = "healthcare_bot"
    
    # SQLite Configuration
    sqlite_db_path: str = "data/healthcare_bot.db"
    
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
    
    # Security Configuration
    secret_key: str = "healthcare_bot_secret_key_change_in_production"
    
    # File Upload Configuration
    max_file_size: int = 10485760  # 10MB in bytes
    allowed_file_types: str = "pdf,doc,docx,jpg,jpeg,png,webp"
    upload_dir: str = "uploads"
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # Medical Configuration
    include_medical_disclaimer: bool = True
    emergency_contact_info: str = "Call 911 for emergencies"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"
    
    @property
    def allowed_file_types_list(self) -> List[str]:
        """Get allowed file types as a list."""
        return [ext.strip().lower() for ext in self.allowed_file_types.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()