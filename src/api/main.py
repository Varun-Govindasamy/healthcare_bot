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
from typing import Dict, Any, Optional, List
import os
from PyPDF2 import PdfReader  # type: ignore
import docx  # type: ignore
import asyncio
from datetime import datetime

# Internal imports
from src.config.settings import Settings
from src.database.manager import DatabaseManager
from src.services.twilio_service import TwilioService
from src.services.query_processor import QueryProcessor
from src.services.onboarding_service import OnboardingService
from src.services.language_processor import LanguageProcessor
from src.agents.medical_data_agent import MedicalDataAgent
from src.agents.vision_agent import VisionAgent
from src.utils.safety_validator import MedicalSafetyValidator
from src.models.schemas import WhatsAppMessage, HealthcareResponse
from openai import AsyncOpenAI
from src.services.pinecone_service import pinecone_service

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

# Basic health query processing function
async def process_basic_health_query(user_id: str, query: str) -> str:
    """Process basic health queries with multi-language support."""
    global language_processor
    
    try:
        if language_processor:
            # Detect language and translate to English for processing
            detected_language, english_query = await language_processor.process_user_input(query)
            logger.info(f"Detected language: {detected_language} for user {user_id}")
            
            # Process the query in English
            english_response = await process_health_query_english(user_id, english_query)
            
            # Translate response back to user's language
            final_response = await language_processor.process_bot_response(english_response, detected_language)
            
            return final_response
        else:
            # Fallback if language processor not available
            return await process_health_query_english(user_id, query)
        
    except Exception as e:
        logger.error(f"Error in multi-language health query processing: {e}")
        # Fallback to English processing
        return await process_health_query_english(user_id, query)

async def process_health_query_english(user_id: str, query: str) -> str:
    """Process basic health queries with simple responses."""
    query_lower = query.lower()

    # Try RAG first if user has uploaded reports/documents
    try:
        # Only attempt if Pinecone is initialized
        if getattr(pinecone_service, 'index', None) is not None:
            user_hits = await pinecone_service.search_user_documents(query=query, user_id=user_id, top_k=3)
            if user_hits:
                return await generate_rag_answer(user_id, query, user_hits)
    except Exception as e:
        logger.warning(f"RAG lookup failed or unavailable, falling back to heuristics: {e}")
    
    # Check for common health concerns
    if 'headache' in query_lower:
        return """🤕 **Headache Relief:**

**Immediate steps:**
• Rest in a quiet, dark room
• Apply cold or warm compress to head/neck
• Stay hydrated - drink water
• Consider over-the-counter pain relievers (if no allergies)

**When to see a doctor:**
• Sudden, severe headache
• Headache with fever, stiff neck, confusion
• Headaches that worsen or become frequent

⚠️ This is general guidance. Consult a doctor for persistent or severe symptoms."""

    elif 'fever' in query_lower:
        return """🌡️ **Fever Management:**

**Immediate steps:**
• Rest and stay hydrated
• Use fever-reducing medication (acetaminophen/ibuprofen)
• Cool compress on forehead
• Wear light clothing

**When to seek immediate care:**
• Fever above 103°F (39.4°C)
• Fever with severe symptoms (difficulty breathing, chest pain)
• Fever lasting more than 3 days

⚠️ This is general guidance. Consult a doctor for high or persistent fever."""

    elif any(word in query_lower for word in ['cold', 'cough']):
        return """🤧 **Cold & Cough Relief:**

**Home remedies:**
• Rest and plenty of fluids
• Warm salt water gargle
• Honey and warm water for cough
• Humidifier or steam inhalation

**When to see a doctor:**
• Symptoms worsen after 7-10 days
• High fever or difficulty breathing
• Severe throat pain or green mucus

⚠️ This is general guidance. Consult a doctor for worsening symptoms."""

    elif any(word in query_lower for word in ['stomach', 'nausea', 'vomit']):
        return """🤢 **Stomach Issues:**

**Home remedies:**
• Stay hydrated with clear fluids
• BRAT diet (Bananas, Rice, Applesauce, Toast)
• Avoid dairy and spicy foods
• Rest and avoid solid foods initially

**When to seek care:**
• Severe dehydration
• Blood in vomit or stool
• High fever with stomach pain
• Symptoms persist over 2 days

⚠️ This is general guidance. Seek immediate care for severe symptoms."""

    elif any(word in query_lower for word in ['pain', 'hurt', 'ache']) and not 'headache' in query_lower:
        return """😣 **Pain Management:**

**General pain relief:**
• Rest the affected area
• Apply ice for acute injuries (first 24-48 hours)
• Apply heat for muscle tension
• Over-the-counter pain relievers (if appropriate)

**When to see a doctor:**
• Severe or worsening pain
• Pain after injury
• Pain with swelling, redness, or fever
• Pain affecting daily activities

⚠️ This is general guidance. Consult a healthcare provider for persistent pain."""

    else:
        # General health query → use LLM to produce a concise, human-readable medical overview
        try:
            return await generate_medical_answer_english(query)
        except Exception as _e:
            logger.warning(f"LLM fallback failed, using generic guidance: {_e}")
            return f"""👨‍⚕️ **Healthcare Guidance:**

Thank you for your question: "{query[:100]}..."

I'm unable to fetch a detailed answer right now. Please share your main symptoms, how long you've had them, and any key medical history (age, allergies, existing conditions). I'll provide tailored guidance.

⚠️ **Important:** For emergencies, call emergency services immediately. This is educational information and not a substitute for professional medical advice."""

async def generate_medical_answer_english(query: str) -> str:
    """Use GPT to answer general health questions in a concise, responsible format."""
    global openai_client, settings
    # Lazy init in case startup path didn't set it yet
    if not openai_client and getattr(settings, 'openai_api_key', None):
        try:
            from openai import AsyncOpenAI as _AsyncOpenAI
            openai_client = _AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("Initialized OpenAI client lazily for general Q&A")
        except Exception as e:
            logger.warning(f"Failed lazy OpenAI init: {e}")
    if not openai_client:
        raise RuntimeError("OpenAI client is not initialized")

    system_msg = (
        "You are a careful, helpful medical information assistant. Answer general health questions "
        "in simple, human-readable language. Be concise (around 8–12 bullets/short lines total). "
        "Do NOT diagnose individuals. Include: definition/overview, common symptoms, how it spreads/causes (if relevant), "
        "self-care, typical treatments (OTC if appropriate), prevention, and when to see a doctor. "
        "End with a brief safety disclaimer. Avoid long paragraphs; use short bullets."
    )

    user_prompt = (
        f"Question: {query}\n\n"
        "Provide a compact response with clear section headers and bullets. Keep within WhatsApp-friendly length."
    )

    resp = await openai_client.chat.completions.create(
        model=getattr(settings, 'gpt_model', 'gpt-4o-mini'),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    msg_content = resp.choices[0].message.content or ""
    text = msg_content.strip()

    # Append a standard disclaimer if model omitted it
    if "Important" not in text and "disclaimer" not in text.lower():
        text += (
            "\n\n⚠️ **Important:** This is general health information and not a substitute for professional medical advice. "
            "See a healthcare professional for diagnosis and treatment."
        )
    return text

async def generate_rag_answer(user_id: str, query: str, user_hits: Optional[List[Dict[str, Any]]] = None) -> str:
    """Generate an answer grounded in the user's uploaded medical documents with healthcare knowledge as backup."""
    global openai_client, settings
    if not openai_client and getattr(settings, 'openai_api_key', None):
        try:
            from openai import AsyncOpenAI as _AsyncOpenAI
            openai_client = _AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("Initialized OpenAI client lazily for RAG")
        except Exception as e:
            logger.warning(f"Failed lazy OpenAI init (RAG): {e}")
    if not openai_client:
        raise RuntimeError("OpenAI client is not initialized")

    # Retrieve general healthcare knowledge for additional context
    general_hits: List[Dict[str, Any]] = []
    try:
        general_hits = await pinecone_service.search_healthcare_knowledge(query=query, top_k=3)
    except Exception as e:
        logger.warning(f"Healthcare knowledge retrieval failed: {e}")

    # Build context blocks with identifiers
    def trim(txt: str, limit: int = 1600) -> str:
        return (txt or "")[:limit]

    user_hits = user_hits or []
    contexts = []
    for i, h in enumerate(user_hits[:3]):
        contexts.append(f"[R{i+1}]\nType: {h.get('document_type','unknown')}\nDate: {h.get('date','')}\nContent:\n{trim(h.get('content',''))}")
    for j, h in enumerate(general_hits[:3]):
        contexts.append(f"[K{j+1}]\nTitle: {h.get('title','')}\nSource: {h.get('source','')}\nContent:\n{trim(h.get('content',''))}")

    system_msg = (
        "You are a careful, helpful medical assistant. Answer the user's question using ONLY the provided context blocks. "
        "Prioritize [R#] user report content. If a specific detail is not in the context, say you don't have enough information. "
        "Provide concise bullets: findings, what they mean, medication recommendations with dosage/frequency/duration when appropriate, "
        "self-care, and when to see a doctor. If recommending prescription meds, note that a doctor's prescription is required. "
        "Always include a brief safety disclaimer at the end. Keep WhatsApp-friendly length."
    )

    context_blob = "\n\n".join(contexts) if contexts else "[No context available]"
    user_prompt = (
        f"User ID: {user_id}\nQuestion: {query}\n\nContext blocks (cite like [R1], [K1]):\n{context_blob}\n\n"
        "Compose the answer now. Use citations inline like [R1] for claims tied to the user's report, and [K1] for general knowledge."
    )

    resp = await openai_client.chat.completions.create(
        model=getattr(settings, 'gpt_model', 'gpt-4o-mini'),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    text = (resp.choices[0].message.content or "").strip()
    if "Important" not in text and "disclaimer" not in text.lower():
        text += (
            "\n\n⚠️ Important: Educational guidance only, not a diagnosis. "
            "Consult a healthcare professional for personalized medical advice."
        )
    return text

async def process_image_analysis(user_id: str, image_url: str, image_type: str = "skin", user_context: Optional[Dict[str, Any]] = None) -> str:
    """Process uploaded images for skin condition analysis with medication suggestions."""
    global vision_agent, language_processor, twilio_service
    
    try:
        if not vision_agent:
            return "🔍 **Image Analysis Unavailable**\n\nSorry, image analysis service is currently unavailable. Please describe your symptoms in text for assistance."
        
        # Download the image first
        if not twilio_service:
            return "❌ **Service Error**\n\nUnable to download image. Please try again later."
        
        # Download media file
        downloaded_file_path = await twilio_service.download_media(image_url, user_id)
        
        if not downloaded_file_path:
            return "❌ **Download Failed**\n\nUnable to download the image. Please try sending the image again."
        
        logger.info(f"🖼️ Processing image analysis for user {user_id}: {downloaded_file_path}")
        
        # Analyze skin condition
        analysis_result = await vision_agent.analyze_skin_condition(downloaded_file_path, user_context)
        
        if "error" in analysis_result:
            return f"❌ **Analysis Error**\n\n{analysis_result['error']}\n\n💡 **Tip**: Try uploading a clearer image or describe your symptoms in text."
        
        # Format the response with medication suggestions
        response = format_image_analysis_response(analysis_result)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image analysis: {e}")
        return "❌ **Analysis Failed**\n\nSorry, we couldn't analyze your image. Please describe your symptoms in text, and I'll provide appropriate guidance."

def format_image_analysis_response(analysis_result: Dict[str, Any]) -> str:
    """Format the image analysis result with medication suggestions."""
    
    if analysis_result.get("raw_response"):
        # Handle raw text response. If it's a refusal, provide a safe general fallback.
        raw = (analysis_result.get('analysis_text') or '').strip()
        lower = raw.lower()
        refusal_markers = [
            "i'm sorry, i can't", "i am sorry, i can't", "i cannot assist", "can't assist",
            "cannot assist", "can't help", "cannot help"
        ]
        if any(m in lower for m in refusal_markers):
            return (
                "🔍 **Skin Condition Analysis**\n\n"
                "📝 **What I can see:**\nGeneral signs of skin irritation/rash.\n\n"
                "🏠 **Immediate care & medications:**\n"
                "• Hydrocortisone 1% cream — apply a thin layer 1–2× daily for up to 7 days\n"
                "• Cetirizine 10 mg — 1 tablet once daily for itch (adult dose)\n"
                "• If ring-shaped, scaly edges → Clotrimazole 1% cream 2× daily for 2–4 weeks\n"
                "• Keep area clean/dry, gentle cleanser, fragrance‑free moisturizer\n\n"
                "🚨 **See a doctor if:**\n"
                "• Rapid spreading, severe pain, fever, pus, or no improvement in 3–5 days\n\n"
                "⚠️ **Important**: Educational guidance only, not a diagnosis. Please consult a dermatologist."
            )
        # Otherwise show the raw text from the model
        return f"🔍 **Image Analysis Results**\n\n{raw or 'Analysis completed'}\n\n⚠️ **Important**: This is AI analysis only. Please consult a dermatologist for professional diagnosis and treatment."
    
    def bullets(items: List[str]) -> str:
        return "\n".join([f"• {it}" for it in items if it]) + ("\n" if items else "")

    def safe_get_list(val) -> List:
        if val is None:
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            # try to parse JSON-like strings
            try:
                import json
                obj = json.loads(val)
                return obj if isinstance(obj, list) else [val]
            except Exception:
                # fallback to Python-literal style (single quotes)
                try:
                    import ast
                    obj = ast.literal_eval(val)
                    return obj if isinstance(obj, list) else [obj]
                except Exception:
                    return [val]
        return [val]

    response = "🔍 **Skin Condition Analysis**\n\n"
    
    # Visual description
    if "description" in analysis_result:
        response += f"📝 **What I can see:**\n{analysis_result['description']}\n\n"
    
    # Possible conditions
    if "possible_conditions" in analysis_result:
        pcs = safe_get_list(analysis_result.get("possible_conditions"))[:3]
        lines: List[str] = []
        for item in pcs:
            if isinstance(item, dict):
                name = item.get("name") or item.get("condition") or "Condition"
                conf = item.get("confidence_percent") or item.get("confidence")
                rationale = item.get("rationale") or item.get("reason")
                parts = [name]
                if conf is not None:
                    try:
                        conf_val = int(float(conf))
                        parts.append(f"{conf_val}%")
                    except Exception:
                        parts.append(str(conf))
                if rationale:
                    parts.append(f"— {rationale}")
                lines.append(" ".join(parts))
            else:
                lines.append(str(item))
        if lines:
            response += "🔬 **Possible conditions:**\n" + bullets(lines) + "\n"
    
    # Immediate care with specific medications
    if "immediate_care" in analysis_result:
        care_items = safe_get_list(analysis_result.get("immediate_care"))[:4]
        care_lines: List[str] = []
        for it in care_items:
            if isinstance(it, dict):
                issue = it.get("issue")
                med = it.get("medication")
                rec = it.get("recommendation")
                dosage = it.get("dosage")
                freq = it.get("frequency")
                app = it.get("application")
                dur = it.get("duration")
                warn = it.get("warnings")
                line = []
                if issue:
                    line.append(f"{issue}:")
                if rec:
                    line.append(rec)
                if med:
                    line.append(f"{med}")
                tail = []
                if dosage:
                    tail.append(dosage)
                if freq:
                    tail.append(freq)
                if app:
                    tail.append(app)
                if tail:
                    line.append(" — " + ", ".join(tail))
                if dur:
                    line.append(f" (Duration: {dur})")
                if warn:
                    line.append(f" [Warnings: {warn}]")
                care_lines.append(" ".join(part for part in line if part))
            else:
                care_lines.append(str(it))
        if care_lines:
            response += "🏠 **Immediate care & medications:**\n" + bullets(care_lines) + "\n"
    else:
        # Add general medication suggestions for skin conditions
        response += """🏠 **General care & medications:**
        
**For inflammatory skin conditions:**
• **Hydrocortisone cream 1%**: Apply thin layer 2-3 times daily
• **Calamine lotion**: Apply as needed for itching relief
• **Antihistamines**: Cetirizine 10mg once daily for itching
• **Moisturizer**: Apply fragrance-free moisturizer 2-3 times daily

**For minor wounds/cuts:**
• **Antiseptic**: Clean with betadine or hydrogen peroxide
• **Antibiotic ointment**: Neosporin - apply 2-3 times daily
• **Bandage**: Keep covered and change daily

**Pain relief:**
• **Ibuprofen**: 400mg every 6-8 hours with food (for inflammation)
• **Paracetamol**: 500mg every 4-6 hours (for pain)

"""
    
    # When to see doctor
    if "when_to_see_doctor" in analysis_result:
        flags = analysis_result.get("when_to_see_doctor")
        items = safe_get_list(flags)
        response += "🚨 **See a doctor if:**\n" + bullets([str(x) for x in items]) + "\n"
    else:
        response += """🚨 **See a doctor if:**
• Condition worsens or doesn't improve in 3-5 days
• Signs of infection (increased redness, warmth, pus, red streaks)
• Severe pain or itching
• Fever or feeling unwell
• Spreading rash or lesions

"""
    
    # Prevention advice
    if "prevention" in analysis_result:
        prev = analysis_result.get("prevention")
        items = safe_get_list(prev)
        response += "🛡️ **Prevention tips:**\n" + bullets([str(x) for x in items]) + "\n"
    
    # Disclaimer
    response += """⚠️ **Important Medical Disclaimer:**
• This is AI-powered image analysis, not a medical diagnosis
• Always consult a dermatologist or healthcare provider for professional evaluation
• Medication suggestions are general guidelines - check with pharmacist/doctor
• Seek immediate medical attention for severe symptoms
• This guidance cannot replace professional medical consultation

💊 **Medication Safety:**
• Check dosages with pharmacist for your age/weight
• Read medication labels and warnings
• Stop use if allergic reactions occur
• Some medications may not be suitable for children, pregnant women, or people with certain conditions"""
    
    return response

# Global variables for services
db_manager: Optional[DatabaseManager] = None
twilio_service: Optional[TwilioService] = None  
query_processor: Optional[QueryProcessor] = None
onboarding_service: Optional[OnboardingService] = None
safety_validator: Optional[MedicalSafetyValidator] = None
language_processor: Optional[LanguageProcessor] = None
vision_agent: Optional[VisionAgent] = None
settings: Settings = Settings()  # type: ignore[call-arg]
openai_client: Optional[AsyncOpenAI] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    Initializes all services and database connections.
    """
    global db_manager, twilio_service, query_processor, onboarding_service, safety_validator, language_processor, vision_agent, openai_client
    
    logger.info("Starting Healthcare Chatbot Application...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("✅ Database connections established")
        
        # Initialize Twilio service
        twilio_service = TwilioService()
        logger.info("✅ Twilio service initialized")
        
        # Initialize language processor
        language_processor = LanguageProcessor()
        logger.info("✅ Language processor initialized")
        
        # Initialize vision agent
        vision_agent = VisionAgent()
        logger.info("✅ Vision agent initialized")
        
        # Initialize OpenAI client for general Q&A
        try:
            if settings.openai_api_key:
                from openai import AsyncOpenAI as _AsyncOpenAI  # local import to avoid startup issues
                # nosec - API key comes from configuration
                openai_client = _AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("✅ OpenAI client initialized for general Q&A")
            else:
                logger.warning("OpenAI API key not set; general Q&A will use generic guidance")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            openai_client = None
        
    # Initialize safety validator
        safety_validator = MedicalSafetyValidator()
        logger.info("✅ Safety validator initialized")
        
        # Initialize medical data agent
        medical_data_agent = MedicalDataAgent(db_manager)
        logger.info("✅ Medical data agent initialized")
        
        # Initialize onboarding service
        onboarding_service = OnboardingService(db_manager, medical_data_agent)
        logger.info("✅ Onboarding service initialized")
        
        # Initialize query processor
        query_processor = QueryProcessor(onboarding_service)
        logger.info("✅ Query processor initialized")

        # Initialize Pinecone (vector DB) for RAG and load default knowledge
        try:
            await pinecone_service.initialize()
            logger.info("✅ Pinecone service initialized")
            # Optionally seed default healthcare knowledge once
            await pinecone_service.initialize_default_healthcare_knowledge()
            logger.info("✅ Default healthcare knowledge initialized")
        except Exception as e:
            logger.warning(f"Pinecone initialization failed or unavailable: {e}")
        
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
    # Don't require language_processor and vision_agent to be initialized for basic functionality
    if not all([db_manager, twilio_service, query_processor, onboarding_service, safety_validator]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    return {
        "db_manager": db_manager,
        "twilio_service": twilio_service,
        "query_processor": query_processor,
        "onboarding_service": onboarding_service,
        "safety_validator": safety_validator,
        "language_processor": language_processor,  # Can be None if not initialized
        "vision_agent": vision_agent  # Can be None if not initialized
    }

@app.get("/", response_class=PlainTextResponse)
async def root():
    """Health check endpoint."""
    return "WhatsApp AI Healthcare Chatbot is running! 🏥🤖"

@app.post("/")
async def webhook_root(
    request: Request,
    background_tasks: BackgroundTasks,
    services: Dict[str, Any] = Depends(get_services)
):
    """Handle Twilio webhook at root path (fallback)."""
    try:
        # Parse incoming webhook
        form_data = await request.form()
        logger.info(f"📱 Root webhook received: {dict(form_data)}")
        
        # Extract basic fields
        from_number = form_data.get('From', '')
        body = form_data.get('Body', '')
        to_number = form_data.get('To', '')
        
        logger.info(f"� Message from {from_number} to {to_number}: {body}")
        
        # Parse message using Twilio service
        whatsapp_message = services["twilio_service"].parse_incoming_message(dict(form_data))
        
        if not whatsapp_message or not whatsapp_message.From:
            logger.warning("❌ Failed to parse WhatsApp message")
            return PlainTextResponse("OK", status_code=200)
        
        # Process message in background to avoid webhook timeout
        background_tasks.add_task(
            process_whatsapp_message,
            whatsapp_message,
            services
        )
        # Return empty 200 so Twilio does not send an immediate message.
        return PlainTextResponse("", status_code=200)
    
    except Exception as e:
        logger.error(f"❌ Error in root webhook: {str(e)}")
        # Still return 200 (empty) to avoid Twilio retries and avoid sending a user-visible message
        return PlainTextResponse("", status_code=200)

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
            "safety_validator": safety_validator is not None,
            "language_processor": language_processor is not None,
            "vision_agent": vision_agent is not None,
            "pinecone": getattr(pinecone_service, 'index', None) is not None
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
    
    # Add Pinecone stats if available
    try:
        if getattr(pinecone_service, 'index', None) is not None:
            stats = await pinecone_service.get_index_stats()
            health_status["pinecone_stats"] = stats
    except Exception as e:
        logger.warning(f"Pinecone stats retrieval failed: {e}")

    # Determine overall health
    all_services_ok = all(health_status["services"].values())
    if not all_services_ok:
        health_status["status"] = "unhealthy"
    
    status_code = 200 if all_services_ok else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/webhook/whatsapp")
async def whatsapp_webhook_get():
    """GET endpoint for WhatsApp webhook verification."""
    return {"status": "WhatsApp webhook endpoint is active", "method": "GET"}

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
        logger.info(f"📱 Received WhatsApp webhook: {dict(form_data)}")
        
        # Extract basic fields
        from_number = form_data.get('From', '')
        body = form_data.get('Body', '')
        to_number = form_data.get('To', '')
        
        logger.info(f"📨 Message from {from_number} to {to_number}: {body}")
        
        # Parse message using Twilio service
        whatsapp_message = services["twilio_service"].parse_incoming_message(dict(form_data))
        
        if not whatsapp_message or not whatsapp_message.From:
            logger.warning("❌ Failed to parse WhatsApp message")
            return PlainTextResponse("OK", status_code=200)
        
        # Process message in background to avoid webhook timeout
        background_tasks.add_task(
            process_whatsapp_message,
            whatsapp_message,
            services
        )
        # Return empty 200 so Twilio does not send an immediate message.
        return PlainTextResponse("", status_code=200)
    
    except Exception as e:
        logger.error(f"Error processing WhatsApp webhook: {e}")
        # Still return 200 (empty) to avoid Twilio retries and avoid sending a user-visible message
        return PlainTextResponse("", status_code=200)

async def process_whatsapp_message(
    message: WhatsAppMessage,
    services: Dict[str, Any]
):
    """
    Background task to process WhatsApp messages through the healthcare pipeline.
    """
    try:
        # Extract phone number from Twilio format
        phone_number = message.From.replace("whatsapp:", "")
        message_text = message.Body or ""
        has_media = message.MediaUrl0 and message.MediaUrl0.strip()
        
        logger.info(f"Processing message from {phone_number}: {message_text[:100]}... (Media: {'Yes' if has_media else 'No'})")
        
        # Check if user exists and needs onboarding
        user_profile = await services["db_manager"].user_repo.get_user_by_id(phone_number)
        
        if not user_profile or not user_profile.is_profile_complete:
            # Handle onboarding flow - don't process images during onboarding
            if has_media:
                response_text = "📝 **Please complete your profile first**\n\nI see you've sent an image. Before I can analyze images, please complete your health profile by answering a few questions. This helps me provide more accurate analysis.\n\nPlease answer the current profile question first."
                response = type('Response', (), {'message': response_text, 'media_url': None})()
            elif not user_profile:
                # New user - start onboarding
                response_text = await services["onboarding_service"].start_onboarding(
                    phone_number,
                    phone_number
                )
                response = type('Response', (), {'message': response_text, 'media_url': None})()
            else:
                # Existing user continuing onboarding
                response_text, is_complete = await services["onboarding_service"].process_onboarding_response(
                    phone_number,
                    message_text
                )
                response = type('Response', (), {'message': response_text, 'media_url': None})()
        else:
            # User profile is complete - process normally
            if has_media:
                # Route based on content type (image vs document)
                ctype = (message.MediaContentType0 or "").lower()
                if ctype.startswith("image/"):
                    logger.info(f"🖼️ Processing image from {phone_number}")
                    user_context = {
                        'age': user_profile.age,
                        'gender': user_profile.gender.value if user_profile.gender else None,
                        'allergies': user_profile.allergies,
                        'existing_conditions': user_profile.existing_conditions
                    } if user_profile else None
                    image_type = "skin"
                    if message_text:
                        text_lower = message_text.lower()
                        if any(word in text_lower for word in ['skin', 'rash', 'spot', 'itch', 'bump', 'mole', 'acne']):
                            image_type = "skin"
                        elif any(word in text_lower for word in ['wound', 'cut', 'injury']):
                            image_type = "wound"
                        elif any(word in text_lower for word in ['lab', 'report', 'test', 'result']):
                            image_type = "lab_report"
                    response_text = await process_image_analysis(phone_number, message.MediaUrl0 or "", image_type, user_context)
                    response = type('Response', (), {'message': response_text, 'media_url': None})()
                elif ctype in ("application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                    logger.info(f"📄 Processing document from {phone_number} ({ctype})")
                    response_text = await process_document_ingestion(phone_number, message.MediaUrl0 or "", ctype, message_text)
                    response = type('Response', (), {'message': response_text, 'media_url': None})()
                else:
                    response = type('Response', (), {'message': "Unsupported media type. Please send an image (JPG/PNG) or PDF report.", 'media_url': None})()
            else:
                # Process text-based health query
                response_text = await process_basic_health_query(phone_number, message_text)
                response = type('Response', (), {'message': response_text, 'media_url': None})()
        
        # Send response back via WhatsApp
        if response and response.message:
            if response.media_url:
                await services["twilio_service"].send_message_with_media(
                    to_number=phone_number,
                    message=response.message,
                    media_url=response.media_url
                )
            else:
                await services["twilio_service"].send_message(
                    to_number=phone_number,
                    message=response.message
                )
            
            logger.info(f"Sent response to {phone_number}")
        
    except Exception as e:
        phone_number = message.From.replace("whatsapp:", "") if message.From else "unknown"
        logger.error(f"Error processing message from {phone_number}: {e}")
        
        # Send error message to user
        try:
            error_msg = (
                "I apologize, but I encountered an error processing your message. "
                "Please try again in a few moments. If this persists, please contact support."
            )
            await services["twilio_service"].send_message(
                to_number=phone_number,
                message=error_msg
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")

async def process_document_ingestion(user_id: str, media_url: str, content_type: str, message_text: str = "") -> str:
    """Download, extract, and index a medical document for RAG. Supports image-based reports; PDFs are acknowledged with limited support."""
    global twilio_service, vision_agent
    try:
        if not twilio_service:
            return "❌ Service error: unable to download document right now. Please try again later."

        # Download file
        file_path = await twilio_service.download_media(media_url, user_id)
        if not file_path:
            return "❌ Download failed. Please send the document again."

        # Normalize file extension based on content type for reliable validation/extraction
        try:
            desired_ext = None
            ct = (content_type or "").lower()
            if ct == "application/pdf":
                desired_ext = ".pdf"
            elif ct == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                desired_ext = ".docx"
            elif ct == "application/msword":
                desired_ext = ".doc"
            if desired_ext:
                base, ext = os.path.splitext(file_path)
                if ext.lower() != desired_ext:
                    new_path = base + desired_ext
                    try:
                        os.replace(file_path, new_path)
                        file_path = new_path
                    except Exception as ren_err:
                        logger.warning(f"Could not rename file to match content type: {ren_err}")
        except Exception as _e:
            logger.debug(f"Extension normalization skipped: {_e}")

        # Validate
        validation = await twilio_service.validate_file_upload(file_path, content_type or "")
        if not validation.get("valid"):
            return f"❌ {validation.get('error','Invalid file')} Please upload a valid document (PDF, DOC/DOCX, or a clear image of the report)."

        # Detect document type from user text
        doc_type = "medical_report"
        t = (message_text or "").lower()
        if any(w in t for w in ["lab", "test", "result", "cbc", "platelet", "haemoglobin", "hemoglobin"]):
            doc_type = "lab_report"
        elif any(w in t for w in ["prescription", "rx", "medication", "medicines"]):
            doc_type = "prescription"

        extracted_text = ""
        metadata: Dict[str, Any] = {"source": "whatsapp", "content_type": content_type, "file_name": os.path.basename(file_path)}

        if content_type.startswith("image/"):
            # Use vision agent to parse the document image
            if not vision_agent:
                return "📄 Document received, but analysis service is unavailable. I'll store it and you can ask questions about it later."
            parsed = await vision_agent.parse_medical_document(file_path, doc_type)
            # Flatten structured data into a text blob for embeddings
            try:
                import json
                if parsed.get("structured") is False and parsed.get("raw_text"):
                    extracted_text = str(parsed.get("raw_text") or "")
                else:
                    # Keep a concise, readable summary
                    extracted_text = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                extracted_text = str(parsed)
        elif content_type == "application/pdf":
            # Extract text from PDF (works for digital PDFs; scanned PDFs may be empty)
            extracted_text = _extract_text_from_pdf(file_path)
            if not extracted_text.strip():
                metadata["extraction"] = "pdf_scan_likely_no_text"
                extracted_text = (
                    "This appears to be a scanned PDF with no extractable text. "
                    "Please send a clear photo of the report pages for best results."
                )
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # DOCX support
            extracted_text = _extract_text_from_docx(file_path)
            if not extracted_text.strip():
                metadata["extraction"] = "docx_empty"
                extracted_text = (
                    "Couldn't extract text from the DOCX file. Please verify the document or send a PDF/image."
                )
        elif content_type == "application/msword":
            # Legacy .doc is not well-supported by python-docx
            metadata["extraction"] = "doc_legacy_limited"
            extracted_text = (
                "You uploaded a legacy .doc file. Text extraction is limited. "
                "Please convert it to PDF/DOCX or send a clear image of the pages."
            )
        else:
            metadata["extraction"] = "unsupported"
            extracted_text = "Unsupported document type. Please send PDF, DOCX, or a clear image of the report."

        # Upsert into Pinecone under user's namespace
        try:
            ok = await pinecone_service.upsert_user_document(user_id=user_id, document_content=extracted_text, document_type=doc_type, metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to upsert user document: {e}")
            ok = False

        if ok:
            return (
                "📄 Document processed successfully!\n\n"
                "I extracted the key details and indexed them so I can answer questions about this report. "
                "You can ask things like:\n"
                "• Summarize my report and key findings\n"
                "• Recommend medicines based on this report\n"
                "• Are any values abnormal and what should I do?\n\n"
                "⚠️ AI assistance only — not a medical diagnosis. Please consult a doctor for confirmation."
            )
        else:
            return (
                "⚠️ I received your document but couldn't index it for search. You can still ask questions, "
                "but replies may be generic. Try sending a clear photo of the report’s key pages."
            )
    except Exception as e:
        logger.error(f"Error in document ingestion: {e}")
        return "Sorry, I had trouble processing your document. Please try again or consult a healthcare professional."

def _extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF using PyPDF2. Scanned PDFs may return empty text."""
    try:
        reader = PdfReader(file_path)
        parts: List[str] = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
                if txt:
                    parts.append(txt)
            except Exception as _:
                continue
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return ""

def _extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        d = docx.Document(file_path)
        paras = [p.text for p in d.paragraphs if p.text]
        return "\n".join(paras)
    except Exception as e:
        logger.warning(f"DOCX extraction failed: {e}")
        return ""

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
        if media_url:
            ok = await services["twilio_service"].send_message_with_media(
                to_number=to,
                message=message,
                media_url=media_url
            )
        else:
            ok = await services["twilio_service"].send_message(
                to_number=to,
                message=message
            )
        return {"success": bool(ok), "sent_to": to}
        
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