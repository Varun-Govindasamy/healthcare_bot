"""
Database connection manager.
"""
import logging
from typing import Optional

from ..database.mongodb import MongoDB, UserRepository, MedicalDocumentRepository
from ..database.sqlite import SQLiteDB, ChatRepository, SessionRepository
from ..database.redis_cache import RedisCache, FAQCache, UserCache
from ..config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Central database connection manager."""
    
    def __init__(self):
        # MongoDB
        self.mongodb: Optional[MongoDB] = None
        self.user_repo: Optional[UserRepository] = None
        self.document_repo: Optional[MedicalDocumentRepository] = None
        
        # SQLite
        self.sqlite_db: Optional[SQLiteDB] = None
        self.chat_repo: Optional[ChatRepository] = None
        self.session_repo: Optional[SessionRepository] = None
        
        # Redis
        self.redis_cache: Optional[RedisCache] = None
        self.faq_cache: Optional[FAQCache] = None
        self.user_cache: Optional[UserCache] = None
    
    async def initialize(self):
        """Initialize all database connections."""
        try:
            # Initialize MongoDB
            self.mongodb = MongoDB(settings.mongodb_url, settings.mongodb_database)
            await self.mongodb.connect()
            self.user_repo = UserRepository(self.mongodb)
            self.document_repo = MedicalDocumentRepository(self.mongodb)
            
            # Initialize SQLite
            self.sqlite_db = SQLiteDB(settings.sqlite_db_path)
            await self.sqlite_db.initialize()
            self.chat_repo = ChatRepository(settings.sqlite_db_path)
            self.session_repo = SessionRepository(settings.sqlite_db_path)
            
            # Initialize Redis
            self.redis_cache = RedisCache(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db
            )
            await self.redis_cache.connect()
            self.faq_cache = FAQCache(self.redis_cache)
            self.user_cache = UserCache(self.redis_cache)
            
            logger.info("All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            raise
    
    async def cleanup(self):
        """Clean up all database connections."""
        try:
            if self.mongodb:
                await self.mongodb.disconnect()
            
            if self.redis_cache:
                await self.redis_cache.disconnect()
            
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
    
    async def close(self):
        """Alias for cleanup method."""
        await self.cleanup()


# Global database manager instance
db_manager = DatabaseManager()