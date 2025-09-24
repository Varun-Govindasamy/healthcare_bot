"""
Twilio WhatsApp integration service.
"""
import logging
from typing import Optional, Dict, Any, List
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import aiofiles
import os
from urllib.parse import urlparse
import httpx
from datetime import datetime

from ..config.settings import settings
from ..models.schemas import WhatsAppMessage

logger = logging.getLogger(__name__)


class TwilioService:
    """Service for Twilio WhatsApp integration."""
    
    def __init__(self):
        self.client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        self.from_number = f"whatsapp:{settings.twilio_phone_number}"
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """Send WhatsApp message to user."""
        try:
            formatted_to = f"whatsapp:{to_number}"
            
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=formatted_to
            )
            
            logger.info(f"Sent WhatsApp message to {to_number}: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            return False
    
    async def send_message_with_media(self, to_number: str, message: str, media_url: str) -> bool:
        """Send WhatsApp message with media attachment."""
        try:
            formatted_to = f"whatsapp:{to_number}"
            
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=formatted_to,
                media_url=[media_url]
            )
            
            logger.info(f"Sent WhatsApp message with media to {to_number}: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message with media: {e}")
            return False
    
    async def download_media(self, media_url: str, user_id: str) -> Optional[str]:
        """Download media file from Twilio with proper redirect handling."""
        try:
            if not media_url:
                logger.warning("No media URL provided")
                return None
            
            logger.info(f"Attempting to download media from: {media_url[:100]}...")
            
            # Create user upload directory
            user_upload_dir = os.path.join(settings.upload_dir, user_id)
            os.makedirs(user_upload_dir, exist_ok=True)
            
            # Get file extension from URL or default to jpg
            parsed_url = urlparse(media_url)
            file_extension = os.path.splitext(parsed_url.path)[1] or '.jpg'
            
            # Generate filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"media_{timestamp}{file_extension}"
            file_path = os.path.join(user_upload_dir, filename)
            
            # Download file with proper redirect handling
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            ) as client:
                # Add Twilio auth for media download
                auth = (settings.twilio_account_sid, settings.twilio_auth_token)
                
                try:
                    # First attempt: Direct download with auth
                    response = await client.get(media_url, auth=auth)
                    
                    # Handle redirects manually if needed
                    if response.status_code in [301, 302, 307, 308]:
                        redirect_url = response.headers.get('location')
                        if redirect_url:
                            logger.info(f"Following redirect from Twilio to: {redirect_url[:100]}...")
                            # Try without auth for the final URL (common for CDN redirects)
                            response = await client.get(redirect_url)
                    
                    if response.status_code == 200:
                        # Validate it's actually an image
                        content_type = response.headers.get('content-type', '').lower()
                        if not any(img_type in content_type for img_type in ['image', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                            logger.warning(f"Downloaded content may not be an image. Content-Type: {content_type}")
                        
                        async with aiofiles.open(file_path, 'wb') as f:
                            await f.write(response.content)
                        
                        file_size = len(response.content)
                        logger.info(f"✅ Successfully downloaded media: {file_path} (Size: {file_size} bytes, Type: {content_type})")
                        return file_path
                    else:
                        logger.error(f"❌ Failed to download media: HTTP {response.status_code}")
                        logger.error(f"Response headers: {dict(response.headers)}")
                        if response.text:
                            logger.error(f"Response body: {response.text[:200]}")
                        return None
                        
                except httpx.HTTPStatusError as http_err:
                    logger.error(f"HTTP status error: {http_err}")
                    return None
                except httpx.RequestError as req_err:
                    logger.error(f"Request error: {req_err}")
                    return None
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading media: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error downloading media: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading media: {e}")
            return None
    
    def create_response(self, message: str) -> str:
        """Create TwiML response for webhook."""
        try:
            response = MessagingResponse()
            response.message(message)
            return str(response)
        except Exception as e:
            logger.error(f"Error creating TwiML response: {e}")
            return ""
    
    def validate_webhook(self, request_data: Dict[str, Any]) -> bool:
        """Validate incoming webhook request."""
        try:
            # Check required fields
            required_fields = ['From', 'Body', 'MessageSid']
            
            for field in required_fields:
                if field not in request_data:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate phone number format
            from_number = request_data.get('From', '')
            if not from_number.startswith('whatsapp:'):
                logger.warning(f"Invalid phone number format: {from_number}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating webhook: {e}")
            return False
    
    def parse_incoming_message(self, request_data: Dict[str, Any]) -> WhatsAppMessage:
        """Parse incoming WhatsApp message from webhook."""
        try:
            return WhatsAppMessage(
                From=request_data.get('From', ''),
                Body=request_data.get('Body', ''),
                MediaUrl0=request_data.get('MediaUrl0'),
                MediaContentType0=request_data.get('MediaContentType0'),
                NumMedia=request_data.get('NumMedia', '0')
            )
        except Exception as e:
            logger.error(f"Error parsing incoming message: {e}")
            return WhatsAppMessage(From='', Body='Error parsing message')
    
    def extract_phone_number(self, whatsapp_id: str) -> str:
        """Extract phone number from WhatsApp ID."""
        try:
            if whatsapp_id.startswith('whatsapp:'):
                return whatsapp_id.replace('whatsapp:', '')
            return whatsapp_id
        except Exception as e:
            logger.error(f"Error extracting phone number: {e}")
            return whatsapp_id
    
    def get_user_id_from_phone(self, phone_number: str) -> str:
        """Generate user ID from phone number."""
        try:
            # Remove whatsapp: prefix and any special characters
            clean_phone = phone_number.replace('whatsapp:', '').replace('+', '').replace('-', '').replace(' ', '')
            return f"user_{clean_phone}"
        except Exception as e:
            logger.error(f"Error generating user ID: {e}")
            return f"user_unknown_{datetime.utcnow().timestamp()}"
    
    async def send_typing_indicator(self, to_number: str) -> bool:
        """Send typing indicator (not directly supported by Twilio, but we can send a placeholder)."""
        try:
            # Twilio doesn't support typing indicators, so we'll skip this
            # In a real implementation, you might send a quick "Thinking..." message
            return True
        except Exception as e:
            logger.error(f"Error sending typing indicator: {e}")
            return False
    
    async def validate_file_upload(self, file_path: str, content_type: str) -> Dict[str, Any]:
        """Validate uploaded file."""
        try:
            validation_result = {
                "valid": False,
                "file_size": 0,
                "file_type": "",
                "error": ""
            }
            
            # Check if file exists
            if not os.path.exists(file_path):
                validation_result["error"] = "File not found"
                return validation_result
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > settings.max_file_size:
                validation_result["error"] = f"File too large: {file_size} bytes"
                return validation_result
            
            # Check file type
            file_extension = os.path.splitext(file_path)[1].lower().replace('.', '')
            if file_extension not in settings.allowed_file_types:
                validation_result["error"] = f"File type not allowed: {file_extension}"
                return validation_result
            
            validation_result.update({
                "valid": True,
                "file_size": file_size,
                "file_type": file_extension
            })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def get_message_status(self, message_sid: str) -> Optional[Dict[str, Any]]:
        """Get status of sent message."""
        try:
            message = self.client.messages(message_sid).fetch()
            
            return {
                "sid": message.sid,
                "status": message.status,
                "error_code": message.error_code,
                "error_message": message.error_message,
                "date_sent": message.date_sent,
                "date_updated": message.date_updated
            }
            
        except Exception as e:
            logger.error(f"Error getting message status: {e}")
            return None
    
    async def send_template_message(self, to_number: str, template_data: Dict[str, Any]) -> bool:
        """Send WhatsApp template message (for business accounts)."""
        try:
            # This would be used for approved WhatsApp Business templates
            # For now, we'll send a regular message
            formatted_to = f"whatsapp:{to_number}"
            template_body = template_data.get("body", "")
            
            message = self.client.messages.create(
                body=template_body,
                from_=self.from_number,
                to=formatted_to
            )
            
            logger.info(f"Sent template message to {to_number}: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send template message: {e}")
            return False


# Global Twilio service instance
twilio_service = TwilioService()