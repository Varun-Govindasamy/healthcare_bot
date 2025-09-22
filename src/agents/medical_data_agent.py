"""
Medical Data Agent - Handles onboarding and medical document processing.
"""
import logging
from typing import Dict, Any, List, Optional
from crewai import Agent, Task
from pydantic import BaseModel, Field
import json
import PyPDF2
import docx
from PIL import Image
import io
import base64

from ..models.schemas import UserProfile, OnboardingQuestion, MedicalDocument
from ..database.manager import db_manager
from ..config.settings import settings

logger = logging.getLogger(__name__)


def onboarding_collector(user_id: str, field_name: str, value: str) -> str:
    """Collects and validates user medical profile information during onboarding."""
    try:
        # This would be called asynchronously in practice
        return f"Collected {field_name}: {value} for user {user_id}"
    except Exception as e:
        return f"Error collecting {field_name}: {str(e)}"


def parse_pdf(file_path: str) -> str:
    """Parse PDF document."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"

def parse_docx(file_path: str) -> str:
    """Parse DOCX document."""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error parsing DOCX: {str(e)}"

def parse_image(file_path: str) -> str:
    """Parse image file (this would use GPT-4o Vision in practice)."""
    try:
        # In practice, this would send the image to GPT-4o Vision
        with open(file_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
            return f"Image parsed successfully. Base64 length: {len(image_data)}"
    except Exception as e:
        return f"Error parsing image: {str(e)}"

def document_parser(file_path: str, file_type: str) -> str:
    """Parses PDF, DOC, and image files to extract medical information."""
    try:
        if file_type.lower() == 'pdf':
            return parse_pdf(file_path)
        elif file_type.lower() in ['doc', 'docx']:
            return parse_docx(file_path)
        elif file_type.lower() in ['jpg', 'jpeg', 'png', 'webp']:
            return parse_image(file_path)
        else:
            return f"Unsupported file type: {file_type}"
    except Exception as e:
        return f"Error parsing document: {str(e)}"


class MedicalDataAgent:
    """Agent responsible for collecting and processing medical data."""
    
    def __init__(self):
        self.tools = []
        
        self.agent = Agent(
            role="Medical Data Specialist",
            goal="Collect complete and accurate medical profiles and process medical documents",
            backstory="""You are a specialized medical data collection expert. Your role is to gather 
            comprehensive user medical profiles during onboarding and process uploaded medical documents. 
            You ensure all required information is collected and validated before allowing users to proceed.""",
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
    
    def create_onboarding_task(self, user_id: str, missing_fields: List[str]) -> Task:
        """Create task for collecting missing onboarding information."""
        return Task(
            description=f"""
            Collect missing medical profile information for user {user_id}.
            Missing fields: {', '.join(missing_fields)}
            
            For each missing field, determine the appropriate question to ask the user.
            Validate the responses and update the user profile.
            
            Required fields:
            - Name (string)
            - Age (1-120)
            - Gender (male/female/other)
            - District (string)
            - State (string)
            - Medication preference (english/ayurvedic/home_remedies)
            
            Optional fields:
            - Allergies (list)
            - Existing conditions (list) 
            - Current medications (list)
            """,
            agent=self.agent,
            expected_output="List of questions to ask the user for missing profile information"
        )
    
    def create_document_processing_task(self, file_path: str, file_type: str, user_id: str) -> Task:
        """Create task for processing uploaded medical documents."""
        return Task(
            description=f"""
            Process uploaded medical document for user {user_id}.
            File: {file_path}
            Type: {file_type}
            
            Extract the following information:
            1. Patient information (name, age, gender if mentioned)
            2. Medical conditions and diagnoses
            3. Medications and prescriptions
            4. Test results and values
            5. Doctor recommendations
            6. Allergies or contraindications
            7. Appointment dates and follow-ups
            
            Return structured data in JSON format.
            """,
            agent=self.agent,
            expected_output="Structured medical data extracted from the document in JSON format"
        )
    
    async def get_onboarding_questions(self, user_id: str) -> List[OnboardingQuestion]:
        """Get list of onboarding questions for incomplete profile."""
        try:
            # Get current user profile
            user = await db_manager.user_repo.get_user_by_id(user_id)
            
            questions = []
            
            if not user or not user.name:
                questions.append(OnboardingQuestion(
                    field_name="name",
                    question="What is your full name?",
                    is_required=True
                ))
            
            if not user or not user.age:
                questions.append(OnboardingQuestion(
                    field_name="age",
                    question="What is your age?",
                    is_required=True
                ))
            
            if not user or not user.gender:
                questions.append(OnboardingQuestion(
                    field_name="gender",
                    question="What is your gender?",
                    options=["male", "female", "other"],
                    is_required=True
                ))
            
            if not user or not user.district:
                questions.append(OnboardingQuestion(
                    field_name="district",
                    question="Which district are you from?",
                    is_required=True
                ))
            
            if not user or not user.state:
                questions.append(OnboardingQuestion(
                    field_name="state",
                    question="Which state are you from?",
                    is_required=True
                ))
            
            if not user or not user.medication_preference:
                questions.append(OnboardingQuestion(
                    field_name="medication_preference",
                    question="What type of medication do you prefer?",
                    options=["english", "ayurvedic", "home_remedies"],
                    is_required=True
                ))
            
            # Optional questions
            if not user or not user.allergies:
                questions.append(OnboardingQuestion(
                    field_name="allergies",
                    question="Do you have any allergies? (Please list them, or type 'none')",
                    is_required=False
                ))
            
            if not user or not user.existing_conditions:
                questions.append(OnboardingQuestion(
                    field_name="existing_conditions",
                    question="Do you have any existing medical conditions? (e.g., diabetes, hypertension, asthma)",
                    is_required=False
                ))
            
            if not user or not user.current_medications:
                questions.append(OnboardingQuestion(
                    field_name="current_medications",
                    question="Are you currently taking any medications? (Please list them, or type 'none')",
                    is_required=False
                ))
            
            return questions
            
        except Exception as e:
            logger.error(f"Error getting onboarding questions: {e}")
            return []
    
    async def process_onboarding_response(self, user_id: str, field_name: str, response: str) -> bool:
        """Process user response to onboarding question."""
        try:
            # Get or create user profile
            user = await db_manager.user_repo.get_user_by_id(user_id)
            if not user:
                user = UserProfile(user_id=user_id)
                await db_manager.user_repo.create_user(user)
            
            # Process the response based on field type
            update_data = {}
            
            if field_name == "name":
                update_data["name"] = response.strip()
            elif field_name == "age":
                try:
                    age = int(response.strip())
                    if 1 <= age <= 120:
                        update_data["age"] = age
                    else:
                        return False
                except ValueError:
                    return False
            elif field_name == "gender":
                if response.lower() in ["male", "female", "other"]:
                    update_data["gender"] = response.lower()
                else:
                    return False
            elif field_name == "district":
                update_data["district"] = response.strip()
            elif field_name == "state":
                update_data["state"] = response.strip()
            elif field_name == "medication_preference":
                if response.lower() in ["english", "ayurvedic", "home_remedies"]:
                    update_data["medication_preference"] = response.lower()
                else:
                    return False
            elif field_name == "allergies":
                if response.lower() != "none":
                    allergies = [allergy.strip() for allergy in response.split(",")]
                    update_data["allergies"] = allergies
                else:
                    update_data["allergies"] = []
            elif field_name == "existing_conditions":
                if response.lower() != "none":
                    conditions = [condition.strip() for condition in response.split(",")]
                    update_data["existing_conditions"] = conditions
                else:
                    update_data["existing_conditions"] = []
            elif field_name == "current_medications":
                if response.lower() != "none":
                    medications = [med.strip() for med in response.split(",")]
                    update_data["current_medications"] = medications
                else:
                    update_data["current_medications"] = []
            
            # Update user profile
            success = await db_manager.user_repo.update_user(user_id, update_data)
            
            # Check if profile is now complete
            if success:
                await self._check_and_update_profile_completion(user_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing onboarding response: {e}")
            return False
    
    async def _check_and_update_profile_completion(self, user_id: str):
        """Check if profile is complete and update the flag."""
        try:
            user = await db_manager.user_repo.get_user_by_id(user_id)
            if user and user.check_profile_completeness():
                await db_manager.user_repo.update_user(user_id, {"is_profile_complete": True})
                logger.info(f"Profile completed for user {user_id}")
        except Exception as e:
            logger.error(f"Error checking profile completion: {e}")


# Global medical data agent instance
medical_data_agent = MedicalDataAgent()