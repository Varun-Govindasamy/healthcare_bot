"""
Safety validation layer for medical advice and recommendations.
"""
import logging
from typing import Dict, Any, List, Optional, Set
import re
from datetime import datetime

from ..models.schemas import UserProfile, SafetyCheck
from ..config.settings import settings

logger = logging.getLogger(__name__)


class MedicalSafetyValidator:
    """Comprehensive medical safety validation system."""
    
    def __init__(self):
        # Emergency keywords that require immediate medical attention
        self.emergency_keywords = {
            "chest pain", "heart attack", "stroke", "difficulty breathing",
            "severe bleeding", "unconscious", "seizure", "anaphylaxis",
            "severe allergic reaction", "overdose", "poisoning",
            "severe burns", "broken bone", "head injury", "severe pain",
            "cannot breathe", "choking", "cardiac arrest", "coma",
            "severe dehydration", "high fever in infant", "severe vomiting",
            "severe diarrhea", "blood in vomit", "blood in stool",
            "sudden vision loss", "sudden hearing loss", "paralysis",
            "severe abdominal pain", "appendicitis", "meningitis"
        }
        
        # Critical symptoms requiring urgent care
        self.urgent_symptoms = {
            "high fever", "persistent vomiting", "severe headache",
            "shortness of breath", "rapid heartbeat", "dizziness",
            "fainting", "confusion", "severe fatigue", "jaundice",
            "swelling", "unusual bleeding", "severe rash",
            "difficulty swallowing", "severe cough", "wheezing"
        }
        
        # Age-specific medication warnings
        self.pediatric_warnings = {
            "aspirin", "ibuprofen under 6 months", "honey under 1 year",
            "adult dosage", "prescription medication without doctor"
        }
        
        # Dangerous medication combinations
        self.medication_interactions = {
            ("warfarin", "aspirin"): "Increased bleeding risk",
            ("metformin", "alcohol"): "Risk of lactic acidosis",
            ("lithium", "nsaid"): "Lithium toxicity risk",
            ("digoxin", "diuretic"): "Electrolyte imbalance risk"
        }
        
        # Contraindications by condition
        self.condition_contraindications = {
            "pregnancy": ["aspirin", "ibuprofen", "alcohol", "smoking"],
            "hypertension": ["nsaids", "decongestants", "high sodium"],
            "diabetes": ["high sugar", "steroids", "certain antibiotics"],
            "kidney disease": ["nsaids", "certain antibiotics", "potassium"],
            "liver disease": ["acetaminophen high dose", "alcohol", "certain herbs"]
        }
    
    async def validate_medical_advice(self, advice_text: str, user_profile: Optional[UserProfile] = None) -> SafetyCheck:
        """Comprehensive validation of medical advice."""
        try:
            safety_check = SafetyCheck()
            
            # Check for emergency situations
            emergency_check = self._check_emergency_keywords(advice_text)
            safety_check.has_emergency_symptoms = emergency_check["has_emergency"]
            safety_check.requires_immediate_attention = emergency_check["requires_immediate"]
            
            # Age-based validation
            if user_profile and user_profile.age:
                age_check = self._validate_age_appropriateness(advice_text, user_profile.age)
                safety_check.age_appropriate = age_check["appropriate"]
                if not age_check["appropriate"]:
                    safety_check.contraindications.extend(age_check["warnings"])
            
            # Check medication interactions and contraindications
            if user_profile:
                interaction_check = self._check_medication_interactions(advice_text, user_profile)
                safety_check.contraindications.extend(interaction_check["contraindications"])
                
                # Check condition-specific contraindications
                condition_check = self._check_condition_contraindications(advice_text, user_profile)
                safety_check.contraindications.extend(condition_check["contraindications"])
                
                # Check allergies
                allergy_check = self._check_allergy_contraindications(advice_text, user_profile)
                safety_check.contraindications.extend(allergy_check["contraindications"])
            
            # Generate warning message if needed
            warning_parts = []
            
            if safety_check.has_emergency_symptoms:
                warning_parts.append("🚨 EMERGENCY: Seek immediate medical attention")
            
            if not safety_check.age_appropriate:
                warning_parts.append("⚠️ Age-specific precautions apply")
            
            if safety_check.contraindications:
                warning_parts.append("⚠️ Medical contraindications detected")
            
            if warning_parts:
                safety_check.warning_message = " | ".join(warning_parts)
            
            logger.info(f"Safety validation completed. Warnings: {len(safety_check.contraindications)}")
            return safety_check
            
        except Exception as e:
            logger.error(f"Error in safety validation: {e}")
            return SafetyCheck(
                has_emergency_symptoms=False,
                requires_immediate_attention=False,
                age_appropriate=True,
                contraindications=[],
                warning_message="Unable to perform complete safety check"
            )
    
    def _check_emergency_keywords(self, text: str) -> Dict[str, Any]:
        """Check for emergency keywords in the text."""
        text_lower = text.lower()
        
        # Check for emergency keywords
        emergency_found = any(keyword in text_lower for keyword in self.emergency_keywords)
        
        # Check for urgent symptoms
        urgent_found = any(keyword in text_lower for keyword in self.urgent_symptoms)
        
        # Check for emergency phrases
        emergency_phrases = [
            "call 911", "emergency room", "immediate medical attention",
            "life threatening", "critical condition", "urgent care"
        ]
        
        emergency_phrase_found = any(phrase in text_lower for phrase in emergency_phrases)
        
        return {
            "has_emergency": emergency_found or emergency_phrase_found,
            "requires_immediate": emergency_found or emergency_phrase_found,
            "urgent_symptoms": urgent_found,
            "keywords_found": [kw for kw in self.emergency_keywords if kw in text_lower]
        }
    
    def _validate_age_appropriateness(self, text: str, age: int) -> Dict[str, Any]:
        """Validate advice based on user's age."""
        warnings = []
        appropriate = True
        text_lower = text.lower()
        
        # Pediatric checks (under 18)
        if age < 18:
            if any(warning in text_lower for warning in self.pediatric_warnings):
                appropriate = False
                warnings.append("Pediatric medication precautions apply")
            
            # Specific age checks
            if age < 2 and ("adult dose" in text_lower or "standard dose" in text_lower):
                appropriate = False
                warnings.append("Infant dosing requires pediatric consultation")
            
            if age < 12 and "aspirin" in text_lower:
                appropriate = False
                warnings.append("Aspirin not recommended for children under 12")
        
        # Elderly checks (over 65)
        elif age > 65:
            elderly_warnings = ["kidney function", "liver function", "drug interactions"]
            if any(warning in text_lower for warning in elderly_warnings):
                warnings.append("Elderly patients require special consideration")
        
        return {
            "appropriate": appropriate,
            "warnings": warnings,
            "age_group": "pediatric" if age < 18 else "elderly" if age > 65 else "adult"
        }
    
    def _check_medication_interactions(self, text: str, user_profile: UserProfile) -> Dict[str, Any]:
        """Check for potential medication interactions."""
        contraindications = []
        text_lower = text.lower()
        
        current_meds = [med.lower() for med in user_profile.current_medications]
        
        # Check mentioned medications against current medications
        for interaction, warning in self.medication_interactions.items():
            med1, med2 = interaction
            
            # Check if one medication is currently taken and another is mentioned
            if (med1 in current_meds and med2 in text_lower) or \
               (med2 in current_meds and med1 in text_lower):
                contraindications.append(f"Interaction warning: {warning}")
        
        # Generic interaction warnings
        if current_meds and any(med in text_lower for med in ["medication", "drug", "pill"]):
            contraindications.append("Check with doctor before adding new medications")
        
        return {
            "contraindications": contraindications,
            "interactions_checked": len(self.medication_interactions)
        }
    
    def _check_condition_contraindications(self, text: str, user_profile: UserProfile) -> Dict[str, Any]:
        """Check contraindications based on existing conditions."""
        contraindications = []
        text_lower = text.lower()
        
        for condition in user_profile.existing_conditions:
            condition_lower = condition.lower()
            
            # Check specific condition contraindications
            if condition_lower in self.condition_contraindications:
                contraindicated_items = self.condition_contraindications[condition_lower]
                
                for item in contraindicated_items:
                    if item in text_lower:
                        contraindications.append(
                            f"Caution: {item} may not be suitable for {condition}"
                        )
        
        return {
            "contraindications": contraindications,
            "conditions_checked": user_profile.existing_conditions
        }
    
    def _check_allergy_contraindications(self, text: str, user_profile: UserProfile) -> Dict[str, Any]:
        """Check for allergy contraindications."""
        contraindications = []
        text_lower = text.lower()
        
        for allergy in user_profile.allergies:
            allergy_lower = allergy.lower()
            
            if allergy_lower in text_lower:
                contraindications.append(f"ALLERGY ALERT: Patient allergic to {allergy}")
            
            # Check for related substances
            allergy_related = {
                "penicillin": ["amoxicillin", "ampicillin", "antibiotics"],
                "aspirin": ["nsaids", "ibuprofen", "naproxen"],
                "sulfa": ["sulfamethoxazole", "trimethoprim"],
                "latex": ["rubber", "gloves"]
            }
            
            if allergy_lower in allergy_related:
                for related_item in allergy_related[allergy_lower]:
                    if related_item in text_lower:
                        contraindications.append(
                            f"CAUTION: {related_item} may cross-react with {allergy} allergy"
                        )
        
        return {
            "contraindications": contraindications,
            "allergies_checked": user_profile.allergies
        }
    
    def generate_emergency_response(self, emergency_type: str = "general") -> str:
        """Generate appropriate emergency response."""
        emergency_responses = {
            "chest_pain": """🚨 EMERGENCY - CHEST PAIN
Call emergency services immediately (112/108 in India, 911 in US)
- Chew aspirin if not allergic
- Sit upright, loosen tight clothing
- Stay calm, help is coming""",
            
            "breathing": """🚨 EMERGENCY - BREATHING DIFFICULTY
Call emergency services immediately (112/108)
- Sit upright, lean forward slightly
- Remove tight clothing
- If using inhaler, use as prescribed""",
            
            "bleeding": """🚨 EMERGENCY - SEVERE BLEEDING
Call emergency services immediately (112/108)
- Apply direct pressure to wound
- Elevate injured area if possible
- Do not remove embedded objects""",
            
            "general": """🚨 EMERGENCY SITUATION DETECTED
Call emergency services immediately:
- India: 112 (National Emergency) or 108 (Ambulance)
- US: 911
- Stay calm and follow dispatcher instructions"""
        }
        
        return emergency_responses.get(emergency_type, emergency_responses["general"])
    
    def add_medical_disclaimer(self, advice_text: str) -> str:
        """Add appropriate medical disclaimer to advice."""
        disclaimer = "\n\n⚠️ This is AI guidance only. Please consult a doctor for confirmation."
        
        # Don't add duplicate disclaimers
        if "AI guidance only" in advice_text or "consult a doctor" in advice_text:
            return advice_text
        
        return advice_text + disclaimer
    
    def validate_dosage_recommendations(self, text: str, age: Optional[int] = None) -> List[str]:
        """Validate dosage recommendations in text."""
        warnings = []
        
        # Look for dosage patterns
        dosage_patterns = [
            r'\d+\s*mg',  # milligrams
            r'\d+\s*ml',  # milliliters
            r'\d+\s*tablets?',  # tablets
            r'\d+\s*times?\s*(daily|per day)'  # frequency
        ]
        
        dosages_found = []
        for pattern in dosage_patterns:
            matches = re.findall(pattern, text.lower())
            dosages_found.extend(matches)
        
        if dosages_found:
            if age and age < 18:
                warnings.append("Pediatric dosing requires medical supervision")
            elif age and age > 65:
                warnings.append("Elderly patients may require dose adjustment")
            else:
                warnings.append("Verify dosage with healthcare provider")
        
        return warnings


# Global safety validator instance
safety_validator = MedicalSafetyValidator()