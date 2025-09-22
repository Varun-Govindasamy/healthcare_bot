"""
Redis cache for FAQ and quick responses.
"""
import redis.asyncio as redis
import json
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache manager."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, password: Optional[str] = None, db: int = 0):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set a key-value pair with optional expiration."""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            result = await self.redis_client.set(key, value, ex=expire)
            return result
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            value = await self.redis_client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to check key {key}: {e}")
            return False


class FAQCache:
    """FAQ caching service."""
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
        self.faq_prefix = "faq:"
        self.cache_expire = 3600  # 1 hour
    
    async def get_faq_response(self, question: str, language: str = "en") -> Optional[str]:
        """Get cached FAQ response."""
        cache_key = f"{self.faq_prefix}{language}:{hash(question.lower())}"
        return await self.cache.get(cache_key)
    
    async def cache_faq_response(self, question: str, response: str, language: str = "en"):
        """Cache FAQ response."""
        cache_key = f"{self.faq_prefix}{language}:{hash(question.lower())}"
        await self.cache.set(cache_key, response, expire=self.cache_expire)
    
    async def get_popular_faqs(self, language: str = "en", limit: int = 10) -> List[Dict[str, str]]:
        """Get popular FAQ questions and answers."""
        # This would typically be implemented with Redis sorted sets
        # For now, return empty list
        return []


class UserCache:
    """User-specific caching service."""
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
        self.user_prefix = "user:"
        self.session_expire = 1800  # 30 minutes
    
    async def cache_user_context(self, user_id: str, context: Dict[str, Any]):
        """Cache user conversation context."""
        cache_key = f"{self.user_prefix}{user_id}:context"
        await self.cache.set(cache_key, context, expire=self.session_expire)
    
    async def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user context."""
        cache_key = f"{self.user_prefix}{user_id}:context"
        return await self.cache.get(cache_key)
    
    async def cache_user_language(self, user_id: str, language: str):
        """Cache user's preferred language."""
        cache_key = f"{self.user_prefix}{user_id}:language"
        await self.cache.set(cache_key, language, expire=86400)  # 24 hours
    
    async def get_user_language(self, user_id: str) -> Optional[str]:
        """Get cached user language."""
        cache_key = f"{self.user_prefix}{user_id}:language"
        return await self.cache.get(cache_key)