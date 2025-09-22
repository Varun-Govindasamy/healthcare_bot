"""
SQLite database for chat memory and session management.
"""
import sqlite3
import aiosqlite
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import logging

from ..models.schemas import ChatMessage, MessageType

logger = logging.getLogger(__name__)


class SQLiteDB:
    """SQLite database manager for chat memory."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def initialize(self):
        """Initialize database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    response TEXT,
                    language_detected TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_user_id ON chat_messages(user_id);
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_user_id ON user_sessions(user_id);
            """)
            
            await db.commit()
            logger.info("SQLite database initialized successfully")


class ChatRepository:
    """Chat message repository."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def save_message(self, message: ChatMessage) -> int:
        """Save a chat message."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO chat_messages 
                (user_id, message_type, content, response, language_detected, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                message.user_id,
                message.message_type.value,
                message.content,
                message.response,
                message.language_detected,
                message.timestamp.isoformat(),
                message.session_id
            ))
            
            await db.commit()
            message_id = cursor.lastrowid
            logger.info(f"Saved chat message {message_id} for user {message.user_id}")
            return message_id
    
    async def get_user_messages(self, user_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get recent messages for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, user_id, message_type, content, response, 
                       language_detected, timestamp, session_id
                FROM chat_messages 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, limit))
            
            rows = await cursor.fetchall()
            messages = []
            
            for row in rows:
                message = ChatMessage(
                    id=row[0],
                    user_id=row[1],
                    message_type=MessageType(row[2]),
                    content=row[3],
                    response=row[4],
                    language_detected=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    session_id=row[7]
                )
                messages.append(message)
            
            return messages
    
    async def get_session_context(self, user_id: str, session_id: str) -> List[ChatMessage]:
        """Get messages from current session for context."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, user_id, message_type, content, response, 
                       language_detected, timestamp, session_id
                FROM chat_messages 
                WHERE user_id = ? AND session_id = ?
                ORDER BY timestamp ASC
            """, (user_id, session_id))
            
            rows = await cursor.fetchall()
            messages = []
            
            for row in rows:
                message = ChatMessage(
                    id=row[0],
                    user_id=row[1],
                    message_type=MessageType(row[2]),
                    content=row[3],
                    response=row[4],
                    language_detected=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    session_id=row[7]
                )
                messages.append(message)
            
            return messages


class SessionRepository:
    """User session repository."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def create_session(self, user_id: str, session_id: str) -> int:
        """Create a new user session."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO user_sessions (user_id, session_id)
                VALUES (?, ?)
            """, (user_id, session_id))
            
            await db.commit()
            return cursor.lastrowid
    
    async def update_session_activity(self, user_id: str, session_id: str):
        """Update session last activity."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE user_sessions 
                SET last_activity = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
            """, (user_id, session_id))
            
            await db.commit()
    
    async def get_active_session(self, user_id: str) -> Optional[str]:
        """Get active session for user."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT session_id FROM user_sessions 
                WHERE user_id = ? AND is_active = TRUE
                ORDER BY last_activity DESC 
                LIMIT 1
            """, (user_id,))
            
            row = await cursor.fetchone()
            return row[0] if row else None