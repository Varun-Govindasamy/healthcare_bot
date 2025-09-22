"""
User onboarding flow service with mandatory profile completion.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import uuid

from ..models.schemas import UserProfile, OnboardingQuestion, Gender, MedicationPreference
from ..database.manager import db_manager
from ..agents.medical_data_agent import medical_data_agent
from ..services.twilio_service import twilio_service

logger = logging.getLogger(__name__)


class OnboardingService:
    """Service for managing user onboarding flow."""
    
    def __init__(self):
        self.onboarding_states = {}  # In-memory state storage (use Redis in production)
    
    async def start_onboarding(self, user_id: str, phone_number: str) -> str:
        """Start onboarding process for new user."""
        try:
            # Check if user already exists
            existing_user = await db_manager.user_repo.get_user_by_id(user_id)
            
            if existing_user and existing_user.is_profile_complete:
                return "Welcome back! Your profile is already complete. How can I help you today?"
            
            # Create new user profile if doesn't exist
            if not existing_user:
                new_user = UserProfile(user_id=user_id)
                await db_manager.user_repo.create_user(new_user)
            
            # Initialize onboarding state
            self.onboarding_states[user_id] = {
                "step": 0,
                "phone_number": phone_number,
                "started_at": datetime.utcnow(),
                "completed_fields": []
            }
            
            # Start with welcome message and first question
            welcome_message = """🩺 Welcome to Healthcare Bot! 

I'm here to provide you with personalized medical guidance. Before I can help you, I need to collect some important information about you.

This will only take a few minutes and will help me give you better, safer advice.

Let's start:"""
            
            first_question = await self._get_next_question(user_id)
            
            return f"{welcome_message}\n\n{first_question}"
            
        except Exception as e:
            logger.error(f"Error starting onboarding: {e}")
            return "Sorry, I'm having trouble getting started. Please try again."
    
    async def process_onboarding_response(self, user_id: str, response: str) -> Tuple[str, bool]:
        """
        Process user response during onboarding.
        Returns: (response_message, is_onboarding_complete)
        """
        try:
            # Check if user is in onboarding
            if user_id not in self.onboarding_states:
                # User might have completed onboarding, check profile
                user_profile = await db_manager.user_repo.get_user_by_id(user_id)
                if user_profile and user_profile.is_profile_complete:
                    return "Your profile is already complete! How can I help you today?", True
                else:
                    # Restart onboarding
                    phone_number = twilio_service.extract_phone_number(user_id)
                    restart_message = await self.start_onboarding(user_id, phone_number)
                    return restart_message, False
            
            # Get current onboarding state
            state = self.onboarding_states[user_id]
            current_step = state["step"]
            
            # Get the question sequence
            questions = await self._get_onboarding_questions()
            
            if current_step >= len(questions):
                return await self._complete_onboarding(user_id)
            
            # Process current response
            current_question = questions[current_step]
            validation_result = await self._validate_response(current_question, response)
            
            if not validation_result["valid"]:
                # Invalid response, ask again
                error_message = validation_result.get("error", "Please provide a valid response.")
                return f"❌ {error_message}\n\n{current_question['question']}", False
            
            # Save the response
            success = await medical_data_agent.process_onboarding_response(
                user_id, current_question["field_name"], response
            )
            
            if not success:
                return f"❌ There was an error saving your response. Please try again.\n\n{current_question['question']}", False
            
            # Mark field as completed
            state["completed_fields"].append(current_question["field_name"])
            
            # Move to next step
            state["step"] += 1
            
            # Check if onboarding is complete
            if state["step"] >= len(questions):
                return await self._complete_onboarding(user_id)
            
            # Get next question
            next_question = await self._get_next_question(user_id)
            progress = f"({state['step']}/{len(questions)})"
            
            return f"✅ Thank you!\n\n{progress} {next_question}", False
            
        except Exception as e:
            logger.error(f"Error processing onboarding response: {e}")
            return "Sorry, there was an error processing your response. Please try again.", False
    
    async def _get_onboarding_questions(self) -> List[Dict[str, Any]]:
        """Get the sequence of onboarding questions."""
        return [
            {
                "field_name": "name",
                "question": "What is your full name?",
                "type": "text",
                "required": True,
                "validation": "name"
            },
            {
                "field_name": "age",
                "question": "What is your age? (Please enter a number between 1 and 120)",
                "type": "number",
                "required": True,
                "validation": "age"
            },
            {
                "field_name": "gender",
                "question": "What is your gender?\nPlease type: male, female, or other",
                "type": "choice",
                "required": True,
                "validation": "gender",
                "options": ["male", "female", "other"]
            },
            {
                "field_name": "district",
                "question": "Which district are you from? (e.g., Mumbai, Delhi, Bangalore)",
                "type": "text",
                "required": True,
                "validation": "district"
            },
            {
                "field_name": "state",
                "question": "Which state are you from? (e.g., Maharashtra, Delhi, Karnataka)",
                "type": "text",
                "required": True,
                "validation": "state"
            },
            {
                "field_name": "medication_preference",
                "question": "What type of medication do you prefer?\nPlease type: english, ayurvedic, or home_remedies",
                "type": "choice",
                "required": True,
                "validation": "medication_preference",
                "options": ["english", "ayurvedic", "home_remedies"]
            },
            {
                "field_name": "allergies",
                "question": "Do you have any allergies? (Please list them separated by commas, or type 'none')\nExample: peanuts, shellfish, penicillin",
                "type": "list",
                "required": False,
                "validation": "allergies"
            },
            {
                "field_name": "existing_conditions",
                "question": "Do you have any existing medical conditions? (Please list them separated by commas, or type 'none')\nExample: diabetes, hypertension, asthma",
                "type": "list",
                "required": False,
                "validation": "existing_conditions"
            },
            {
                "field_name": "current_medications",
                "question": "Are you currently taking any medications? (Please list them separated by commas, or type 'none')\nExample: metformin, lisinopril, inhaler",
                "type": "list",
                "required": False,
                "validation": "current_medications"
            }
        ]
    
    async def _get_next_question(self, user_id: str) -> str:
        """Get the next question for the user."""
        try:
            questions = await self._get_onboarding_questions()
            state = self.onboarding_states.get(user_id, {"step": 0})
            current_step = state["step"]
            
            if current_step < len(questions):
                question_data = questions[current_step]
                question = question_data["question"]
                
                # Add options if it's a choice question
                if question_data.get("options"):
                    options_text = "\n".join([f"• {option}" for option in question_data["options"]])
                    question += f"\n\nOptions:\n{options_text}"
                
                return question
            
            return "No more questions available."
            
        except Exception as e:
            logger.error(f"Error getting next question: {e}")
            return "Sorry, there was an error getting the next question."
    
    async def _validate_response(self, question: Dict[str, Any], response: str) -> Dict[str, Any]:
        """Validate user response based on question type."""
        try:
            field_name = question["field_name"]
            response = response.strip()
            
            if question["required"] and not response:
                return {"valid": False, "error": "This field is required."}
            
            validation_type = question.get("validation", "text")
            
            if validation_type == "name":
                if len(response) < 2:
                    return {"valid": False, "error": "Please enter a valid name (at least 2 characters)."}
                if any(char.isdigit() for char in response):
                    return {"valid": False, "error": "Name should not contain numbers."}
            
            elif validation_type == "age":
                try:
                    age = int(response)
                    if age < 1 or age > 120:
                        return {"valid": False, "error": "Please enter an age between 1 and 120."}
                except ValueError:
                    return {"valid": False, "error": "Please enter a valid number for age."}
            
            elif validation_type == "gender":
                if response.lower() not in ["male", "female", "other"]:
                    return {"valid": False, "error": "Please choose: male, female, or other"}
            
            elif validation_type == "medication_preference":
                if response.lower() not in ["english", "ayurvedic", "home_remedies"]:
                    return {"valid": False, "error": "Please choose: english, ayurvedic, or home_remedies"}
            
            elif validation_type == "district":
                if len(response) < 2:
                    return {"valid": False, "error": "Please enter a valid district name."}
            
            elif validation_type == "state":
                if len(response) < 2:
                    return {"valid": False, "error": "Please enter a valid state name."}
            
            elif validation_type in ["allergies", "existing_conditions", "current_medications"]:
                # These can be 'none' or a list
                if response.lower() == "none":
                    return {"valid": True}
                # Validate list format
                items = [item.strip() for item in response.split(",")]
                if any(len(item) < 2 for item in items if item):
                    return {"valid": False, "error": "Please enter valid items separated by commas."}
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return {"valid": False, "error": "There was an error validating your response."}
    
    async def _complete_onboarding(self, user_id: str) -> Tuple[str, bool]:
        """Complete the onboarding process."""
        try:
            # Update profile completion status
            await db_manager.user_repo.update_user(user_id, {"is_profile_complete": True})
            
            # Clean up onboarding state
            if user_id in self.onboarding_states:
                del self.onboarding_states[user_id]
            
            # Get user profile for personalized message
            user_profile = await db_manager.user_repo.get_user_by_id(user_id)
            
            completion_message = f"""🎉 Congratulations {user_profile.name if user_profile else 'there'}! 

Your health profile is now complete. I can now provide you with personalized medical guidance based on your:
• Age and gender
• Location ({user_profile.district if user_profile else 'your area'})
• Medical preferences ({user_profile.medication_preference if user_profile else 'your preferences'})
• Health conditions and allergies

💬 You can now ask me about:
• Symptoms and health concerns
• Medication advice
• Local health alerts
• General medical questions
• Upload medical reports for analysis

How can I help you today?

⚠️ Remember: This is AI guidance only. Please consult a doctor for confirmation."""
            
            logger.info(f"Completed onboarding for user {user_id}")
            return completion_message, True
            
        except Exception as e:
            logger.error(f"Error completing onboarding: {e}")
            return "Your profile setup is complete! How can I help you today?", True
    
    async def check_profile_completion(self, user_id: str) -> bool:
        """Check if user profile is complete."""
        try:
            user_profile = await db_manager.user_repo.get_user_by_id(user_id)
            return user_profile and user_profile.is_profile_complete
        except Exception as e:
            logger.error(f"Error checking profile completion: {e}")
            return False
    
    async def get_onboarding_progress(self, user_id: str) -> Dict[str, Any]:
        """Get current onboarding progress."""
        try:
            if user_id not in self.onboarding_states:
                return {"in_progress": False, "completed": await self.check_profile_completion(user_id)}
            
            state = self.onboarding_states[user_id]
            questions = await self._get_onboarding_questions()
            
            return {
                "in_progress": True,
                "current_step": state["step"],
                "total_steps": len(questions),
                "completed_fields": state["completed_fields"],
                "progress_percentage": (state["step"] / len(questions)) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting onboarding progress: {e}")
            return {"in_progress": False, "completed": False, "error": str(e)}
    
    async def reset_onboarding(self, user_id: str) -> str:
        """Reset onboarding process for a user."""
        try:
            # Clear onboarding state
            if user_id in self.onboarding_states:
                del self.onboarding_states[user_id]
            
            # Reset profile completion status
            await db_manager.user_repo.update_user(user_id, {"is_profile_complete": False})
            
            # Start fresh onboarding
            phone_number = twilio_service.extract_phone_number(user_id)
            return await self.start_onboarding(user_id, phone_number)
            
        except Exception as e:
            logger.error(f"Error resetting onboarding: {e}")
            return "Sorry, there was an error resetting your profile. Please try again."


# Global onboarding service instance
onboarding_service = OnboardingService()