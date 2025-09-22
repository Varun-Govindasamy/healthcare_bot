"""
WhatsApp AI Healthcare Chatbot - Main FastAPI Application

This is the main entry point for the healthcare chatbot that integrates:
- WhatsApp webhook handling via Twilio
- Multi-language processing with GPT
- 5 CrewAI agents for medical assistance
- Vector database for healthcare knowledge
- Safety validation and emergency detection
- Comprehensive user onboarding
"""

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

# Internal imports
from src.config.settings import Settings
from src.database.manager import DatabaseManager
from src.services.twilio_service import TwilioService
from src.services.query_processor import QueryProcessor
from src.services.onboarding_service import OnboardingService
from src.utils.safety_validator import MedicalSafetyValidator
from src.models.schemas import WhatsAppMessage, HealthcareResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('healthcare_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for services
db_manager: Optional[DatabaseManager] = None
twilio_service: Optional[TwilioService] = None
query_processor: Optional[QueryProcessor] = None
onboarding_service: Optional[OnboardingService] = None
safety_validator: Optional[MedicalSafetyValidator] = None
settings: Settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    Initializes all services and database connections.
    """
    global db_manager, twilio_service, query_processor, onboarding_service, safety_validator
    
    logger.info("Starting Healthcare Chatbot Application...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("✅ Database connections established")
        
        # Initialize Twilio service
        twilio_service = TwilioService(settings)
        logger.info("✅ Twilio service initialized")
        
        # Initialize safety validator
        safety_validator = MedicalSafetyValidator()
        logger.info("✅ Safety validator initialized")
        
        # Initialize onboarding service
        onboarding_service = OnboardingService(db_manager, safety_validator)
        logger.info("✅ Onboarding service initialized")
        
        # Initialize query processor
        query_processor = QueryProcessor(db_manager, safety_validator)
        logger.info("✅ Query processor initialized")
        
        logger.info("🚀 Healthcare Chatbot is ready to serve!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down Healthcare Chatbot...")
        if db_manager:
            await db_manager.close()
        logger.info("✅ Cleanup completed")

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="WhatsApp AI Healthcare Chatbot",
    description="""
    An AI-powered healthcare chatbot that provides medical information and assistance through WhatsApp.
    
    Features:
    - Multi-language support (13 Indian languages + English)
    - 5 specialized AI agents for comprehensive medical assistance
    - Safety validation and emergency detection
    - Vector database for healthcare knowledge
    - Comprehensive user onboarding
    - Integration with WhatsApp via Twilio
    """,
    version="1.0.0",
    contact={
        "name": "Healthcare Bot Team",
        "email": "support@healthcarebot.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure for production
)

# Dependency to get services
async def get_services():
    """Dependency to provide access to initialized services."""
    if not all([db_manager, twilio_service, query_processor, onboarding_service, safety_validator]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    return {
        "db_manager": db_manager,
        "twilio_service": twilio_service,
        "query_processor": query_processor,
        "onboarding_service": onboarding_service,
        "safety_validator": safety_validator
    }

@app.get("/", response_class=PlainTextResponse)
async def root():
    """Health check endpoint."""
    return "WhatsApp AI Healthcare Chatbot is running! 🏥🤖"

@app.get("/health")
async def health_check():
    """Detailed health check with service status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": db_manager is not None,
            "twilio": twilio_service is not None,
            "query_processor": query_processor is not None,
            "onboarding": onboarding_service is not None,
            "safety_validator": safety_validator is not None
        }
    }
    
    # Check database connections
    if db_manager:
        try:
            health_status["services"]["mongodb"] = await db_manager.test_mongodb_connection()
            health_status["services"]["redis"] = await db_manager.test_redis_connection()
            health_status["services"]["sqlite"] = await db_manager.test_sqlite_connection()
        except Exception as e:
            logger.error(f"Health check database error: {e}")
            health_status["services"]["database_error"] = str(e)
    
    # Determine overall health
    all_services_ok = all(health_status["services"].values())
    if not all_services_ok:
        health_status["status"] = "unhealthy"
    
    status_code = 200 if all_services_ok else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    services: Dict[str, Any] = Depends(get_services)
):
    """
    Main WhatsApp webhook endpoint for receiving messages from Twilio.
    Processes incoming messages through the healthcare pipeline.
    """
    try:
        # Parse incoming webhook
        form_data = await request.form()
        logger.info(f"Received WhatsApp webhook: {dict(form_data)}")
        
        # Parse message using Twilio service
        whatsapp_message = services["twilio_service"].parse_incoming_message(form_data)
        
        if not whatsapp_message:
            logger.warning("Failed to parse WhatsApp message")
            return PlainTextResponse("OK", status_code=200)
        
        # Process message in background to avoid webhook timeout
        background_tasks.add_task(
            process_whatsapp_message,
            whatsapp_message,
            services
        )
        
        # Return success immediately to Twilio
        return PlainTextResponse("OK", status_code=200)
        
    except Exception as e:
        logger.error(f"Error processing WhatsApp webhook: {e}")
        # Still return 200 to avoid Twilio retries
        return PlainTextResponse("OK", status_code=200)

async def process_whatsapp_message(
    message: WhatsAppMessage,
    services: Dict[str, Any]
):
    """
    Background task to process WhatsApp messages through the healthcare pipeline.
    """
    try:
        logger.info(f"Processing message from {message.from_number}: {message.body[:100]}...")
        
        # Check if user exists and needs onboarding
        user_profile = await services["db_manager"].user_repository.get_user(message.from_number)
        
        if not user_profile or not user_profile.onboarding_completed:
            # Handle onboarding flow
            response = await services["onboarding_service"].process_onboarding_message(
                message.from_number,
                message.body
            )
        else:
            # Process through main query pipeline
            response = await services["query_processor"].process_message(message)
        
        # Send response back via WhatsApp
        if response and response.message:
            await services["twilio_service"].send_message(
                to=message.from_number,
                message=response.message,
                media_url=response.media_url
            )
            
            logger.info(f"Sent response to {message.from_number}")
        
    except Exception as e:
        logger.error(f"Error processing message from {message.from_number}: {e}")
        
        # Send error message to user
        try:
            error_msg = (
                "I apologize, but I encountered an error processing your message. "
                "Please try again in a few moments. If this persists, please contact support."
            )
            await services["twilio_service"].send_message(
                to=message.from_number,
                message=error_msg
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")

@app.post("/api/send-message")
async def send_message_api(
    to: str,
    message: str,
    media_url: Optional[str] = None,
    services: Dict[str, Any] = Depends(get_services)
):
    """
    API endpoint to send messages programmatically (for testing/admin purposes).
    """
    try:
        message_sid = await services["twilio_service"].send_message(
            to=to,
            message=message,
            media_url=media_url
        )
        
        return {
            "success": True,
            "message_sid": message_sid,
            "sent_to": to
        }
        
    except Exception as e:
        logger.error(f"Failed to send message via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/{phone_number}")
async def get_user_profile(
    phone_number: str,
    services: Dict[str, Any] = Depends(get_services)
):
    """
    Get user profile information (for admin/debugging purposes).
    """
    try:
        user_profile = await services["db_manager"].user_repository.get_user(phone_number)
        
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Remove sensitive information
        user_data = user_profile.model_dump()
        if 'medical_history' in user_data:
            user_data['medical_history'] = "***REDACTED***"
        
        return user_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/user/{phone_number}")
async def delete_user_data(
    phone_number: str,
    services: Dict[str, Any] = Depends(get_services)
):
    """
    Delete all user data (GDPR compliance).
    """
    try:
        # Delete from MongoDB
        await services["db_manager"].user_repository.delete_user(phone_number)
        
        # Delete chat history from SQLite
        await services["db_manager"].chat_repository.delete_user_chats(phone_number)
        
        # Clear Redis cache
        await services["db_manager"].redis_cache.clear_user_data(phone_number)
        
        return {"success": True, "message": "User data deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_bot_statistics(
    services: Dict[str, Any] = Depends(get_services)
):
    """
    Get bot usage statistics.
    """
    try:
        # Get user count
        total_users = await services["db_manager"].user_repository.get_user_count()
        
        # Get completed onboarding count
        completed_onboarding = await services["db_manager"].user_repository.get_completed_onboarding_count()
        
        # Get message count (approximate)
        total_messages = await services["db_manager"].chat_repository.get_total_message_count()
        
        return {
            "total_users": total_users,
            "completed_onboarding": completed_onboarding,
            "total_messages": total_messages,
            "onboarding_completion_rate": round((completed_onboarding / total_users * 100) if total_users > 0 else 0, 2)
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )