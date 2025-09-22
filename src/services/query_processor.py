"""
Main query processing pipeline with CrewAI router and agent coordination.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import os

from ..models.schemas import ChatMessage, MessageType, HealthcareResponse
from ..database.manager import db_manager
from ..services.onboarding_service import onboarding_service
from ..services.language_processor import language_processor
from ..services.twilio_service import twilio_service
from ..agents.conversation_agent import conversation_agent
from ..agents.vision_agent import vision_agent
from ..agents.medical_data_agent import medical_data_agent

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Main query processing pipeline."""
    
    def __init__(self):
        self.active_sessions = {}  # In-memory session storage
    
    async def process_whatsapp_message(self, webhook_data: Dict[str, Any]) -> str:
        """Process incoming WhatsApp message and return response."""
        try:
            # Parse incoming message
            message = twilio_service.parse_incoming_message(webhook_data)
            
            # Extract user information
            phone_number = twilio_service.extract_phone_number(message.From)
            user_id = twilio_service.get_user_id_from_phone(message.From)
            
            logger.info(f"Processing message from {user_id}: {message.Body[:50]}...")
            
            # Check if user profile is complete
            is_profile_complete = await onboarding_service.check_profile_completion(user_id)
            
            # Handle onboarding if profile not complete
            if not is_profile_complete:
                return await self._handle_onboarding(user_id, message, phone_number)
            
            # Process regular healthcare query
            return await self._handle_healthcare_query(user_id, message)
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp message: {e}")
            return "I'm sorry, I'm having trouble processing your message right now. Please try again."
    
    async def _handle_onboarding(self, user_id: str, message, phone_number: str) -> str:
        """Handle onboarding flow."""
        try:
            # Check if this is the first message from user
            progress = await onboarding_service.get_onboarding_progress(user_id)
            
            if not progress.get("in_progress", False):
                # Start onboarding
                response = await onboarding_service.start_onboarding(user_id, phone_number)
            else:
                # Continue onboarding with user response
                response, is_complete = await onboarding_service.process_onboarding_response(
                    user_id, message.Body or ""
                )
                
                if is_complete:
                    logger.info(f"Onboarding completed for user {user_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling onboarding: {e}")
            return "Sorry, there was an error during setup. Please try again."
    
    async def _handle_healthcare_query(self, user_id: str, message) -> str:
        """Handle healthcare query for users with complete profiles."""
        try:
            # Determine message type
            message_type = self._determine_message_type(message)
            
            # Handle different message types
            if message_type == MessageType.IMAGE:
                return await self._handle_image_message(user_id, message)
            elif message_type == MessageType.DOCUMENT:
                return await self._handle_document_message(user_id, message)
            else:
                return await self._handle_text_message(user_id, message)
                
        except Exception as e:
            logger.error(f"Error handling healthcare query: {e}")
            return "I'm sorry, I'm having trouble processing your query right now. Please try again or consult a healthcare professional."
    
    def _determine_message_type(self, message) -> MessageType:
        """Determine the type of incoming message."""
        try:
            num_media = int(message.NumMedia or "0")
            
            if num_media > 0 and message.MediaContentType0:
                content_type = message.MediaContentType0.lower()
                
                if content_type.startswith('image/'):
                    return MessageType.IMAGE
                elif content_type in ['application/pdf', 'application/msword', 
                                     'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    return MessageType.DOCUMENT
            
            return MessageType.TEXT
            
        except Exception as e:
            logger.error(f"Error determining message type: {e}")
            return MessageType.TEXT
    
    async def _handle_text_message(self, user_id: str, message) -> str:
        """Handle text-based healthcare queries."""
        try:
            query_text = message.Body or ""
            
            if not query_text.strip():
                return "Please send me your health question or concern, and I'll do my best to help you."
            
            # Save message to chat history
            chat_message = ChatMessage(
                user_id=user_id,
                message_type=MessageType.TEXT,
                content=query_text,
                session_id=self._get_or_create_session(user_id)
            )
            
            # Process with conversation agent
            healthcare_response = await conversation_agent.process_healthcare_query(
                user_id, query_text, "text"
            )
            
            # Update chat message with response
            chat_message.response = healthcare_response.translated_response
            chat_message.language_detected = healthcare_response.detected_language
            
            # Save to database
            await db_manager.chat_repo.save_message(chat_message)
            
            return healthcare_response.translated_response
            
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
            return "I'm sorry, I couldn't process your message. Please try rephrasing your question."
    
    async def _handle_image_message(self, user_id: str, message) -> str:
        """Handle image uploads (skin conditions, reports, etc.)."""
        try:
            if not message.MediaUrl0:
                return "I didn't receive any image. Please try uploading your image again."
            
            # Download the image
            image_path = await twilio_service.download_media(message.MediaUrl0, user_id)
            
            if not image_path:
                return "Sorry, I couldn't download your image. Please try uploading it again."
            
            # Validate file
            validation = await twilio_service.validate_file_upload(
                image_path, message.MediaContentType0 or ""
            )
            
            if not validation["valid"]:
                return f"❌ {validation['error']} Please upload a valid image file (JPG, PNG, WEBP) under 10MB."
            
            # Determine analysis type based on user's message or context
            analysis_context = message.Body or "general medical image analysis"
            
            # Get user profile for context
            user_profile = await db_manager.user_repo.get_user_by_id(user_id)
            user_context = user_profile.dict() if user_profile else None
            
            # Analyze with vision agent
            if "skin" in analysis_context.lower() or "rash" in analysis_context.lower():
                analysis_result = await vision_agent.analyze_skin_condition(image_path, user_context)
            else:
                # Try document parsing first, then general image analysis
                analysis_result = await vision_agent.parse_medical_document(image_path, "medical_report")
                
                if "error" in analysis_result:
                    # Fallback to skin analysis
                    analysis_result = await vision_agent.analyze_skin_condition(image_path, user_context)
            
            # Format response
            if "error" in analysis_result:
                response = f"I had trouble analyzing your image: {analysis_result['error']}\n\nPlease try uploading a clearer image or consult a healthcare professional."
            else:
                response = self._format_vision_analysis_response(analysis_result)
            
            # Save to chat history
            chat_message = ChatMessage(
                user_id=user_id,
                message_type=MessageType.IMAGE,
                content=f"[Image uploaded: {os.path.basename(image_path)}] {analysis_context}",
                response=response,
                session_id=self._get_or_create_session(user_id)
            )
            await db_manager.chat_repo.save_message(chat_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling image message: {e}")
            return "Sorry, I had trouble analyzing your image. Please try again or consult a healthcare professional."
    
    async def _handle_document_message(self, user_id: str, message) -> str:
        """Handle document uploads (PDFs, medical reports)."""
        try:
            if not message.MediaUrl0:
                return "I didn't receive any document. Please try uploading your document again."
            
            # Download the document
            doc_path = await twilio_service.download_media(message.MediaUrl0, user_id)
            
            if not doc_path:
                return "Sorry, I couldn't download your document. Please try uploading it again."
            
            # Validate file
            validation = await twilio_service.validate_file_upload(
                doc_path, message.MediaContentType0 or ""
            )
            
            if not validation["valid"]:
                return f"❌ {validation['error']} Please upload a valid document (PDF, DOC, DOCX) under 10MB."
            
            # Process document with medical data agent
            processing_result = await medical_data_agent.process_onboarding_response(
                user_id, "document_upload", doc_path
            )
            
            if processing_result:
                response = f"""📄 Document processed successfully!

I've analyzed your medical document and extracted the relevant information. This data will help me provide more personalized healthcare advice.

You can now ask me questions about your medical history, current medications, or any health concerns.

⚠️ This is AI analysis only. Please consult a doctor for confirmation."""
            else:
                response = "Sorry, I had trouble processing your document. Please ensure it's a clear, readable medical document."
            
            # Save to chat history
            chat_message = ChatMessage(
                user_id=user_id,
                message_type=MessageType.DOCUMENT,
                content=f"[Document uploaded: {os.path.basename(doc_path)}] {message.Body or ''}",
                response=response,
                session_id=self._get_or_create_session(user_id)
            )
            await db_manager.chat_repo.save_message(chat_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling document message: {e}")
            return "Sorry, I had trouble processing your document. Please try again or consult a healthcare professional."
    
    def _format_vision_analysis_response(self, analysis_result: Dict[str, Any]) -> str:
        """Format vision analysis result into user-friendly response."""
        try:
            if analysis_result.get("raw_response"):
                return analysis_result.get("analysis_text", "Analysis completed.")
            
            # Format structured response
            response_parts = []
            
            if analysis_result.get("description"):
                response_parts.append(f"🔍 **What I can see:**\n{analysis_result['description']}")
            
            if analysis_result.get("possible_conditions"):
                conditions = analysis_result["possible_conditions"]
                if isinstance(conditions, list):
                    conditions_text = "\n".join([f"• {condition}" for condition in conditions])
                else:
                    conditions_text = str(conditions)
                response_parts.append(f"🩺 **Possible conditions:**\n{conditions_text}")
            
            if analysis_result.get("immediate_care"):
                response_parts.append(f"🏥 **Immediate care:**\n{analysis_result['immediate_care']}")
            
            if analysis_result.get("when_to_see_doctor"):
                response_parts.append(f"⚠️ **When to see a doctor:**\n{analysis_result['when_to_see_doctor']}")
            
            if analysis_result.get("prevention"):
                response_parts.append(f"🛡️ **Prevention:**\n{analysis_result['prevention']}")
            
            # Add disclaimer
            response_parts.append("⚠️ This is AI analysis only. Please consult a doctor for confirmation.")
            
            return "\n\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error formatting vision response: {e}")
            return "Analysis completed. Please consult a healthcare professional for proper diagnosis."
    
    def _get_or_create_session(self, user_id: str) -> str:
        """Get or create session ID for user."""
        try:
            if user_id not in self.active_sessions:
                self.active_sessions[user_id] = {
                    "session_id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow(),
                    "last_activity": datetime.utcnow()
                }
            else:
                self.active_sessions[user_id]["last_activity"] = datetime.utcnow()
            
            return self.active_sessions[user_id]["session_id"]
            
        except Exception as e:
            logger.error(f"Error managing session: {e}")
            return str(uuid.uuid4())
    
    async def get_user_chat_history(self, user_id: str, limit: int = 10) -> List[ChatMessage]:
        """Get recent chat history for user."""
        try:
            return await db_manager.chat_repo.get_user_messages(user_id, limit)
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    async def clear_user_session(self, user_id: str) -> bool:
        """Clear user session."""
        try:
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False


# Global query processor instance
query_processor = QueryProcessor()