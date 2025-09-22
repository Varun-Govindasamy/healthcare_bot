"""
Pydantic models for data validation and serialization.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class MedicationPreference(str, Enum):
    ENGLISH = "english"
    AYURVEDIC = "ayurvedic"
    HOME_REMEDIES = "home_remedies"


class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"


class UserProfile(BaseModel):
    """User profile model for MongoDB storage."""
    user_id: str = Field(..., description="WhatsApp user ID")
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=1, le=120)
    gender: Optional[Gender] = None
    district: Optional[str] = None
    state: Optional[str] = None
    allergies: List[str] = Field(default_factory=list)
    medication_preference: Optional[MedicationPreference] = None
    existing_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_profile_complete: bool = False

    @validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 1 or v > 120):
            raise ValueError('Age must be between 1 and 120')
        return v

    def check_profile_completeness(self) -> bool:
        """Check if all required fields are filled."""
        required_fields = [
            self.name, self.age, self.gender, 
            self.district, self.state, self.medication_preference
        ]
        self.is_profile_complete = all(field is not None for field in required_fields)
        return self.is_profile_complete


class OnboardingQuestion(BaseModel):
    """Model for onboarding questions."""
    field_name: str
    question: str
    options: Optional[List[str]] = None
    is_required: bool = True


class ChatMessage(BaseModel):
    """Chat message model for SQLite storage."""
    id: Optional[int] = None
    user_id: str
    message_type: MessageType
    content: str
    response: Optional[str] = None
    language_detected: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None


class WhatsAppMessage(BaseModel):
    """Incoming WhatsApp message model."""
    From: str
    Body: Optional[str] = None
    MediaUrl0: Optional[str] = None
    MediaContentType0: Optional[str] = None
    NumMedia: str = "0"


class MedicalDocument(BaseModel):
    """Medical document upload model."""
    user_id: str
    document_type: str  # pdf, image, etc.
    file_path: str
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    upload_date: datetime = Field(default_factory=datetime.utcnow)


class SearchQuery(BaseModel):
    """Search query model for outbreak/disease information."""
    query: str
    location: Optional[str] = None
    user_id: str


class AgentResponse(BaseModel):
    """Response from CrewAI agents."""
    agent_name: str
    response: str
    confidence: Optional[float] = None
    sources: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SafetyCheck(BaseModel):
    """Safety validation model."""
    has_emergency_symptoms: bool = False
    requires_immediate_attention: bool = False
    age_appropriate: bool = True
    contraindications: List[str] = Field(default_factory=list)
    warning_message: Optional[str] = None


class HealthcareResponse(BaseModel):
    """Final healthcare response model."""
    user_id: str
    original_message: str
    detected_language: str
    translated_response: str
    safety_check: SafetyCheck
    sources: List[str] = Field(default_factory=list)
    disclaimer: str = "⚠️ This is AI guidance only. Please consult a doctor for confirmation."
    timestamp: datetime = Field(default_factory=datetime.utcnow)