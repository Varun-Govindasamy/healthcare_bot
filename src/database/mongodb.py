"""
MongoDB database models and operations.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from ..models.schemas import UserProfile, MedicalDocument

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB database manager."""
    
    def __init__(self, connection_string: str, database_name: str):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.connection_string = connection_string
        self.database_name = database_name
        
    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            self.database = self.client[self.database_name]
            # Test connection
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    @property
    def users(self) -> AsyncIOMotorCollection:
        """Get users collection."""
        return self.database.users
    
    @property
    def medical_documents(self) -> AsyncIOMotorCollection:
        """Get medical documents collection."""
        return self.database.medical_documents


class UserRepository:
    """User profile repository."""
    
    def __init__(self, db: MongoDB):
        self.db = db
    
    async def create_user(self, user_profile: UserProfile) -> str:
        """Create a new user profile."""
        try:
            user_dict = user_profile.dict()
            result = await self.db.users.insert_one(user_dict)
            logger.info(f"Created user profile for {user_profile.user_id}")
            return str(result.inserted_id)
        except DuplicateKeyError:
            logger.warning(f"User {user_profile.user_id} already exists")
            raise ValueError(f"User {user_profile.user_id} already exists")
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by WhatsApp ID."""
        user_doc = await self.db.users.find_one({"user_id": user_id})
        if user_doc:
            user_doc.pop('_id', None)  # Remove MongoDB ObjectId
            return UserProfile(**user_doc)
        return None
    
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user profile."""
        update_data['updated_at'] = datetime.utcnow()
        result = await self.db.users.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        if result.modified_count > 0:
            logger.info(f"Updated user profile for {user_id}")
            return True
        return False
    
    async def check_profile_completion(self, user_id: str) -> bool:
        """Check if user profile is complete."""
        user = await self.get_user_by_id(user_id)
        if user:
            return user.check_profile_completeness()
        return False
    
    async def get_users_by_location(self, district: str, state: str) -> List[UserProfile]:
        """Get users by location for outbreak alerts."""
        cursor = self.db.users.find({"district": district, "state": state})
        users = []
        async for user_doc in cursor:
            user_doc.pop('_id', None)
            users.append(UserProfile(**user_doc))
        return users


class MedicalDocumentRepository:
    """Medical document repository."""
    
    def __init__(self, db: MongoDB):
        self.db = db
    
    async def save_document(self, document: MedicalDocument) -> str:
        """Save medical document metadata."""
        doc_dict = document.dict()
        result = await self.db.medical_documents.insert_one(doc_dict)
        logger.info(f"Saved medical document for user {document.user_id}")
        return str(result.inserted_id)
    
    async def get_user_documents(self, user_id: str) -> List[MedicalDocument]:
        """Get all documents for a user."""
        cursor = self.db.medical_documents.find({"user_id": user_id})
        documents = []
        async for doc in cursor:
            doc.pop('_id', None)
            documents.append(MedicalDocument(**doc))
        return documents
    
    async def update_document_data(self, document_id: str, extracted_data: Dict[str, Any]) -> bool:
        """Update extracted data for a document."""
        result = await self.db.medical_documents.update_one(
            {"_id": document_id},
            {"$set": {"extracted_data": extracted_data}}
        )
        return result.modified_count > 0