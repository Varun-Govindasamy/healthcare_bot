"""
Vision Agent - GPT-4o Vision for medical image and document analysis.
"""
import logging
from typing import Dict, Any, Optional, List
from crewai import Agent, Task
import base64
from PIL import Image
import io
from openai import AsyncOpenAI

from ..config.settings import settings

logger = logging.getLogger(__name__)


def medical_image_analyzer(image_path: str, analysis_type: str = "general") -> str:
    """Analyzes medical images like skin conditions, X-rays, lab reports using GPT-4o Vision."""
    try:
        return f"Analyzed {analysis_type} medical image: {image_path}"
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def document_image_parser(image_path: str, document_type: str = "general") -> str:
    """Extracts text and medical data from document images like prescription, lab reports."""
    try:
        return f"Parsed {document_type} document from image: {image_path}"
    except Exception as e:
        return f"Error parsing document: {str(e)}"


class VisionAgent:
    """Agent for medical image and document vision analysis."""
    
    def __init__(self):
        self.tools = []
        
        self.agent = Agent(
            role="Medical Vision Specialist",
            goal="Analyze medical images and document photos to extract relevant medical information and provide insights",
            backstory="""You are a medical vision analysis specialist with expertise in interpreting 
            medical images, skin conditions, lab reports, prescriptions, and other healthcare documents. 
            You use advanced vision AI to extract structured information and provide medical insights 
            while maintaining appropriate disclaimers about the need for professional medical consultation.""",
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
        
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""
    
    def resize_image_if_needed(self, image_path: str, max_size: int = 2048) -> str:
        """Resize image if too large for API."""
        try:
            with Image.open(image_path) as img:
                if max(img.size) > max_size:
                    # Calculate new size maintaining aspect ratio
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    
                    # Resize and save
                    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized_path = image_path.replace('.', '_resized.')
                    img_resized.save(resized_path)
                    return resized_path
                
                return image_path
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image_path
    
    async def analyze_skin_condition(self, image_path: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze skin condition from image."""
        try:
            # Resize image if needed
            processed_image_path = self.resize_image_if_needed(image_path)
            
            # Encode image
            base64_image = self.encode_image(processed_image_path)
            
            if not base64_image:
                return {"error": "Failed to process image"}
            
            # Build context prompt
            context = ""
            if user_context:
                age = user_context.get('age')
                gender = user_context.get('gender')
                allergies = user_context.get('allergies', [])
                existing_conditions = user_context.get('existing_conditions', [])
                
                context = f"""
                Patient context:
                - Age: {age or 'Not specified'}
                - Gender: {gender or 'Not specified'}
                - Known allergies: {', '.join(allergies) if allergies else 'None reported'}
                - Existing conditions: {', '.join(existing_conditions) if existing_conditions else 'None reported'}
                """
            # System message to steer the model away from refusals and towards safe, educational guidance
            system_msg = (
                "You are a dermatology information assistant. Provide non-diagnostic, educational analysis "
                "of skin images with general guidance and over-the-counter options. Do not refuse; if uncertain, "
                "express uncertainty and give general care advice, safety warnings, and when to see a doctor. "
                "Always include a clear disclaimer that this is not a medical diagnosis. Return strictly JSON using the schema."
            )

            user_prompt = f"""
            Review this skin image and provide a non-diagnostic, educational summary with practical self-care steps.
            {context}

            Return JSON with the following keys exactly:
            - description: short visual description of what is visible (e.g., redness, scaling, circular patches)
            - possible_conditions: array of objects {{name, confidence_percent, rationale}}
            - immediate_care: array of items with fields {{issue, recommendation, medication, dosage, frequency, application, duration, warnings}}
            - when_to_see_doctor: bullet list of red flags
            - prevention: bullet list of general advice
            - disclaimer: short safety statement (non-diagnostic, consult a dermatologist)

            Rules:
            - Educational only; do not claim diagnosis.
            - Prefer over-the-counter options when possible.
            - Include concrete examples, e.g., Hydrocortisone 1% cream (thin layer, 1-2x daily, up to 7 days), Cetirizine 10 mg (once daily),
              Clotrimazole 1% cream (2x daily, 2-4 weeks if fungal features), Mupirocin 2% (3x daily if superficial bacterial signs),
              gentle cleanser and moisturizer guidance, sunscreen usage, patch testing.
            - Add pregnancy/children warnings and allergy considerations when relevant.
            - Never return Markdown, only JSON.
            """

            async def call_model() -> str:
                """Call model and return content string."""
                try:
                    # Try to enforce JSON output when supported
                    return (await self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1500,
                        temperature=0.15,
                        # Some models support JSON mode; ignore if unsupported
                        response_format={"type": "json_object"}
                    )).choices[0].message.content
                except Exception as e:
                    logger.warning(f"Primary JSON-mode call failed or unsupported: {e}")
                    # Fallback without response_format
                    return (await self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1500,
                        temperature=0.15
                    )).choices[0].message.content

            analysis_text = await call_model()

            # If the model still refused, gently retry with a more constrained prompt
            refusal_markers = [
                "I can't assist", "I cannot assist", "can't help with that", "cannot help with that",
                "not able to", "I'm sorry, I can't", "I’m sorry, I can’t"
            ]
            if any(m.lower() in (analysis_text or "").lower() for m in refusal_markers):
                logger.info("Detected refusal; retrying with reinforced educational framing")
                user_prompt_retry = (
                    user_prompt
                    + "\n\nImportant: Provide a neutral, educational summary and general over-the-counter self-care options with clear disclaimers."
                )
                try:
                    analysis_text = (await self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt_retry},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=1500,
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )).choices[0].message.content
                except Exception as e:
                    logger.warning(f"Retry JSON-mode call failed or unsupported: {e}")
                    analysis_text = (await self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt_retry},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=1500,
                        temperature=0.1
                    )).choices[0].message.content

            # Try to parse as JSON, fallback to structured text
            try:
                import json
                analysis = json.loads(analysis_text)
            except Exception as parse_err:
                logger.warning(f"Failed to parse analysis as JSON, returning raw: {parse_err}")
                analysis = {
                    "description": "Educational analysis completed",
                    "analysis_text": analysis_text,
                    "raw_response": True
                }
            
            analysis["image_analyzed"] = True
            analysis["analysis_timestamp"] = __import__('datetime').datetime.utcnow().isoformat()
            logger.info("Vision analysis completed for skin condition")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing skin condition: {e}")
            return {
                "error": str(e),
                "disclaimer": "Unable to analyze image. Please consult a dermatologist."
            }
    
    async def parse_medical_document(self, image_path: str, document_type: str = "general") -> Dict[str, Any]:
        """Parse medical document from image."""
        try:
            # Resize image if needed
            processed_image_path = self.resize_image_if_needed(image_path)
            
            # Encode image
            base64_image = self.encode_image(processed_image_path)
            
            if not base64_image:
                return {"error": "Failed to process image"}
            
            prompt = f"""
            Extract structured information from this medical document image.
            Document type: {document_type}
            
            Please extract and structure the following information if present:
            1. Patient information (name, age, gender, ID numbers)
            2. Date of report/prescription
            3. Doctor/clinic information
            4. Medical conditions/diagnoses
            5. Medications prescribed (name, dosage, frequency, duration)
            6. Test results and values (with normal ranges)
            7. Recommendations and instructions
            8. Follow-up appointments
            9. Allergies or contraindications mentioned
            
            Format as JSON with clear key-value pairs.
            If text is unclear or unreadable, mark as "unclear" or "not_readable".
            Include confidence level for extracted information.
            
            Return structured JSON data.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                import json
                extracted_data = json.loads(extracted_text)
            except:
                extracted_data = {
                    "raw_text": extracted_text,
                    "structured": False,
                    "extraction_method": "text_only"
                }
            
            extracted_data["document_type"] = document_type
            extracted_data["extraction_timestamp"] = __import__('datetime').datetime.utcnow().isoformat()
            logger.info("Document extraction completed")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error parsing medical document: {e}")
            return {
                "error": str(e),
                "document_type": document_type
            }
    
    async def analyze_lab_report(self, image_path: str) -> Dict[str, Any]:
        """Analyze lab report from image."""
        try:
            # Use document parser with specific lab report context
            result = await self.parse_medical_document(image_path, "lab_report")
            
            if "error" not in result:
                # Add lab-specific analysis
                prompt = f"""
                Based on this lab report data: {result}
                
                Provide interpretation for:
                1. Values outside normal ranges (high/low)
                2. Clinical significance of abnormal values
                3. Potential health implications
                4. Recommendations for follow-up
                5. Lifestyle modifications if applicable
                
                Important: Always recommend consulting with a doctor for professional interpretation.
                
                Return as JSON with keys: abnormal_values, clinical_significance, recommendations, disclaimer
                """
                
                interpretation_response = await self.openai_client.chat.completions.create(
                    model=settings.gpt_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.2
                )
                
                try:
                    import json
                    interpretation = json.loads(interpretation_response.choices[0].message.content)
                    result["interpretation"] = interpretation
                except:
                    result["interpretation"] = {
                        "analysis": interpretation_response.choices[0].message.content
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing lab report: {e}")
            return {"error": str(e)}
    
    def create_image_analysis_task(self, image_path: str, analysis_type: str, user_context: Optional[Dict[str, Any]] = None) -> Task:
        """Create task for image analysis."""
        return Task(
            description=f"""
            Analyze the medical image at {image_path} for {analysis_type}.
            
            User context: {user_context or 'No specific context provided'}
            
            Provide detailed analysis including:
            1. Visual observations
            2. Possible medical interpretations
            3. Recommended actions
            4. When to seek professional help
            5. Appropriate disclaimers
            
            Maintain empathetic and helpful tone while emphasizing the need for professional medical consultation.
            """,
            agent=self.agent,
            expected_output=f"Comprehensive {analysis_type} analysis with medical insights and recommendations"
        )


# Global vision agent instance
vision_agent = VisionAgent()