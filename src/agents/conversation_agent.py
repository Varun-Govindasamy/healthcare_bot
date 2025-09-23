"""
Conversation Agent - Main GPT-4o agent for healthcare conversations and coordination.
"""
import logging
from typing import Dict, Any, Optional, List
from crewai import Agent, Task, Crew
from datetime import datetime
from openai import AsyncOpenAI

from ..models.schemas import UserProfile, HealthcareResponse, SafetyCheck
from ..database.manager import DatabaseManager
from ..services.language_processor import language_processor
from .medical_data_agent import MedicalDataAgent
from .rag_agent import RAGAgent
from .search_agent import SearchAgent
from .vision_agent import VisionAgent
from ..config.settings import settings

logger = logging.getLogger(__name__)


def safety_validator(advice: str, user_age: Optional[int] = None, existing_conditions: Optional[List[str]] = None) -> str:
    """Validates medical advice for safety concerns and age-appropriateness."""
    try:
        warnings = []
        
        # Check for emergency keywords
        emergency_keywords = [
            "chest pain", "difficulty breathing", "severe bleeding", 
            "unconscious", "stroke", "heart attack", "seizure"
        ]
        
        if any(keyword in advice.lower() for keyword in emergency_keywords):
            warnings.append("EMERGENCY: Seek immediate medical attention")
        
        # Age-based validation
        if user_age and user_age < 18:
            warnings.append("Pediatric case: Consult pediatrician")
        
        return f"Safety check completed. Warnings: {'; '.join(warnings) if warnings else 'None'}"
        
    except Exception as e:
        return f"Safety validation error: {str(e)}"


class ConversationAgent:
    """Main conversation agent for healthcare interactions."""
    
    def __init__(self):
        self.tools = []
        
        self.agent = Agent(
            role="Healthcare Conversation Specialist",
            goal="Provide empathetic, accurate, and safe healthcare guidance while coordinating with specialized agents",
            backstory="""You are a compassionate healthcare conversation specialist with expertise in 
            medical communication. You coordinate with specialized agents to gather information and provide 
            comprehensive, personalized healthcare guidance. You maintain a warm, empathetic tone while 
            ensuring medical accuracy and safety. You always include appropriate disclaimers and know 
            when to recommend professional medical consultation.""",
            tools=self.tools,
            verbose=True,
            allow_delegation=True
        )
        
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    async def process_healthcare_query(self, user_id: str, query: str, message_type: str = "text") -> HealthcareResponse:
        """Process a healthcare query with full agent coordination."""
        try:
            # Step 1: Check user profile completeness
            user_profile = await db_manager.user_repo.get_user_by_id(user_id)
            
            if not user_profile or not user_profile.is_profile_complete:
                # User needs to complete onboarding
                missing_questions = await medical_data_agent.get_onboarding_questions(user_id)
                if missing_questions:
                    next_question = missing_questions[0]
                    return HealthcareResponse(
                        user_id=user_id,
                        original_message=query,
                        detected_language="en",  # Will be detected properly
                        translated_response=f"Before I can help you, I need some information. {next_question.question}",
                        safety_check=SafetyCheck(),
                        sources=[]
                    )
            
            # Step 2: Language processing
            detected_language, english_query = await language_processor.process_user_input(query)
            
            # Step 3: Determine query type and route to appropriate agents
            query_context = await self._analyze_query_context(english_query, user_profile)
            
            # Step 4: Gather information from relevant agents
            gathered_info = await self._gather_agent_responses(english_query, user_id, query_context)
            
            # Step 5: Generate comprehensive response
            response_data = await self._generate_healthcare_response(
                english_query, user_profile, gathered_info, query_context
            )
            
            # Step 6: Safety validation
            safety_check = await self._perform_safety_check(response_data, user_profile)
            
            # Step 7: Translate response back to user's language
            final_response = await language_processor.process_bot_response(
                response_data["response"], detected_language
            )
            
            return HealthcareResponse(
                user_id=user_id,
                original_message=query,
                detected_language=detected_language,
                translated_response=final_response,
                safety_check=safety_check,
                sources=response_data.get("sources", [])
            )
            
        except Exception as e:
            logger.error(f"Error processing healthcare query: {e}")
            return HealthcareResponse(
                user_id=user_id,
                original_message=query,
                detected_language="en",
                translated_response="I'm sorry, I'm having trouble processing your request right now. Please try again or consult a healthcare professional.",
                safety_check=SafetyCheck(has_emergency_symptoms=False),
                sources=[]
            )
    
    async def _analyze_query_context(self, query: str, user_profile: Optional[UserProfile]) -> Dict[str, Any]:
        """Analyze query to determine which agents to involve."""
        try:
            analysis_prompt = f"""
            Analyze this healthcare query and determine the appropriate response strategy:
            Query: "{query}"
            
            Determine if the query involves:
            1. General medical knowledge (needs RAG agent)
            2. Location-specific health info/outbreaks (needs search agent)
            3. Image analysis (needs vision agent)
            4. Document processing (needs vision agent)
            5. Emergency/urgent care (needs immediate attention)
            6. Medication interactions (needs RAG + safety check)
            7. Symptoms analysis (needs RAG + possible search)
            
            Return JSON with:
            - query_type: primary type of query
            - agents_needed: list of agents to involve
            - urgency_level: low/medium/high
            - requires_location: boolean
            - keywords: relevant medical keywords
            """
            
            response = await self.openai_client.chat.completions.create(
                model=settings.gpt_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=500,
                temperature=0.2
            )
            
            try:
                import json
                context = json.loads(response.choices[0].message.content)
            except:
                # Fallback context
                context = {
                    "query_type": "general_medical",
                    "agents_needed": ["rag"],
                    "urgency_level": "medium",
                    "requires_location": False,
                    "keywords": []
                }
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing query context: {e}")
            return {
                "query_type": "general_medical",
                "agents_needed": ["rag"],
                "urgency_level": "medium",
                "requires_location": False,
                "keywords": []
            }
    
    async def _gather_agent_responses(self, query: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather responses from relevant agents based on context."""
        responses = {}
        
        try:
            agents_needed = context.get("agents_needed", ["rag"])
            
            # RAG Agent - Healthcare knowledge
            if "rag" in agents_needed:
                rag_response = await rag_agent.get_healthcare_knowledge(query, user_id)
                responses["healthcare_knowledge"] = rag_response
            
            # Search Agent - Current outbreaks/news
            if "search" in agents_needed:
                user_profile = await db_manager.user_repo.get_user_by_id(user_id)
                if user_profile and user_profile.district and user_profile.state:
                    search_response = await search_agent.search_disease_outbreak(
                        context.get("keywords", ["health"])[0] if context.get("keywords") else "health",
                        user_profile.district,
                        user_profile.state
                    )
                    responses["outbreak_info"] = search_response
            
            # Medical Data Agent - User-specific info
            if user_id:
                user_profile = await db_manager.user_repo.get_user_by_id(user_id)
                if user_profile:
                    responses["user_profile"] = user_profile.dict()
            
            return responses
            
        except Exception as e:
            logger.error(f"Error gathering agent responses: {e}")
            return {}
    
    async def _generate_healthcare_response(self, query: str, user_profile: Optional[UserProfile], 
                                          gathered_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive healthcare response using gathered information."""
        try:
            # Build context for GPT
            user_context = ""
            if user_profile:
                user_context = f"""
                User Profile:
                - Age: {user_profile.age}
                - Gender: {user_profile.gender}
                - Location: {user_profile.district}, {user_profile.state}
                - Allergies: {', '.join(user_profile.allergies) if user_profile.allergies else 'None'}
                - Existing conditions: {', '.join(user_profile.existing_conditions) if user_profile.existing_conditions else 'None'}
                - Current medications: {', '.join(user_profile.current_medications) if user_profile.current_medications else 'None'}
                - Medication preference: {user_profile.medication_preference}
                """
            
            knowledge_context = ""
            if gathered_info.get("healthcare_knowledge"):
                knowledge_context = f"Healthcare Knowledge: {gathered_info['healthcare_knowledge']}"
            
            outbreak_context = ""
            if gathered_info.get("outbreak_info"):
                outbreak_context = f"Local Health Information: {gathered_info['outbreak_info']}"
            
            response_prompt = f"""
            You are a compassionate healthcare AI assistant. Provide helpful, accurate, and empathetic medical guidance.
            
            User Query: "{query}"
            
            {user_context}
            
            {knowledge_context}
            
            {outbreak_context}
            
            Provide a comprehensive response that:
            1. Addresses the user's specific concern
            2. Considers their personal medical profile
            3. Includes relevant prevention/treatment advice
            4. Mentions when to seek professional help
            5. Uses appropriate tone (empathetic, reassuring, informative)
            6. Includes medication recommendations based on their preference
            7. Adds location-specific information if relevant
            
            Important guidelines:
            - Be empathetic and understanding
            - Provide practical, actionable advice
            - Consider age-appropriate recommendations
            - Mention contraindications based on allergies/conditions
            - Always include medical disclaimer
            - Use simple, understandable language
            
            End with: "⚠️ This is AI guidance only. Please consult a doctor for confirmation."
            """
            
            response = await self.openai_client.chat.completions.create(
                model=settings.gpt_model,
                messages=[{"role": "user", "content": response_prompt}],
                max_tokens=settings.max_tokens,
                temperature=settings.gpt_temperature
            )
            
            healthcare_response = response.choices[0].message.content
            
            return {
                "response": healthcare_response,
                "sources": self._extract_sources(gathered_info),
                "confidence": "high" if gathered_info else "medium"
            }
            
        except Exception as e:
            logger.error(f"Error generating healthcare response: {e}")
            return {
                "response": "I'm sorry, I'm having trouble generating a proper response right now. Please consult a healthcare professional for assistance.",
                "sources": [],
                "confidence": "low"
            }
    
    async def _perform_safety_check(self, response_data: Dict[str, Any], user_profile: Optional[UserProfile]) -> SafetyCheck:
        """Perform safety validation on the generated response."""
        try:
            response_text = response_data.get("response", "")
            
            # Emergency keywords check
            emergency_keywords = [
                "emergency", "urgent", "immediately", "call doctor", "hospital",
                "chest pain", "difficulty breathing", "severe", "blood", "unconscious"
            ]
            
            has_emergency = any(keyword in response_text.lower() for keyword in emergency_keywords)
            
            # Age appropriateness
            age_appropriate = True
            if user_profile and user_profile.age:
                if user_profile.age < 18 and "adult dosage" in response_text.lower():
                    age_appropriate = False
            
            # Check for contraindications
            contraindications = []
            if user_profile and user_profile.allergies:
                for allergy in user_profile.allergies:
                    if allergy.lower() in response_text.lower():
                        contraindications.append(f"Patient allergic to {allergy}")
            
            return SafetyCheck(
                has_emergency_symptoms=has_emergency,
                requires_immediate_attention=has_emergency,
                age_appropriate=age_appropriate,
                contraindications=contraindications,
                warning_message="Consult healthcare professional" if not age_appropriate or contraindications else None
            )
            
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            return SafetyCheck(
                has_emergency_symptoms=False,
                requires_immediate_attention=False,
                age_appropriate=True,
                contraindications=[],
                warning_message="Unable to perform safety check"
            )
    
    def _extract_sources(self, gathered_info: Dict[str, Any]) -> List[str]:
        """Extract source information from gathered data."""
        sources = []
        
        if gathered_info.get("healthcare_knowledge"):
            sources.append("Medical Knowledge Base")
        
        if gathered_info.get("outbreak_info"):
            sources.append("Current Health Alerts")
        
        if gathered_info.get("user_profile"):
            sources.append("Personal Medical Profile")
        
        return sources
    
    def create_conversation_task(self, query: str, user_context: Dict[str, Any]) -> Task:
        """Create a conversation task for the agent."""
        return Task(
            description=f"""
            Provide comprehensive healthcare guidance for the following query:
            Query: "{query}"
            
            User Context: {user_context}
            
            Generate a response that:
            1. Addresses the specific medical concern
            2. Considers user's personal health profile
            3. Provides actionable advice
            4. Includes appropriate safety warnings
            5. Maintains empathetic tone
            6. Recommends professional consultation when needed
            
            Always include medical disclaimer.
            """,
            agent=self.agent,
            expected_output="Comprehensive, empathetic healthcare guidance with safety considerations"
        )