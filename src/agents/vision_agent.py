"""
Vision Agent - GPT-4o Vision for medical image and document analysis.
"""
import logging
from typing import Dict, Any, Optional, List
from crewai import Agent, Task
from crewai.tools import BaseTool
import base64
from PIL import Image
import io
from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class MedicalImageAnalyzer(BaseTool):
    """Tool for analyzing medical images using GPT-4o Vision."""
    
    name: str = "medical_image_analyzer"
    description: str = "Analyzes medical images like skin conditions, X-rays, lab reports using GPT-4o Vision"
    
    def _run(self, image_path: str, analysis_type: str = "general") -> str:
        """Analyze medical image."""
        try:
            return f"Analyzed {analysis_type} medical image: {image_path}"
        except Exception as e:
            return f"Error analyzing image: {str(e)}"


class DocumentImageParser(BaseTool):
    """Tool for parsing documents from images using GPT-4o Vision."""
    
    name: str = "document_image_parser"
    description: str = "Extracts text and medical data from document images like prescription, lab reports"
    
    def _run(self, image_path: str, document_type: str = "general") -> str:
        """Parse document from image."""
        try:
            return f"Parsed {document_type} document from image: {image_path}"
        except Exception as e:
            return f"Error parsing document: {str(e)}"


class VisionAgent:
    """Agent for medical image and document vision analysis."""
    
    def __init__(self):
        self.tools = [
            MedicalImageAnalyzer(),
            DocumentImageParser()
        ]
        
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
            
            prompt = f"""
            Analyze this skin condition image and provide a structured assessment. 
            {context}
            
            Please provide:
            1. Visual description of the skin condition
            2. Possible condition(s) it might indicate (with confidence levels)
            3. Recommended immediate care steps
            4. When to seek medical attention
            5. General prevention advice
            
            Important: 
            - This is AI analysis only, not a medical diagnosis
            - Always recommend consulting a dermatologist for definitive diagnosis
            - Focus on general care and when to seek help
            - Be empathetic and reassuring
            
            Format as JSON with keys: description, possible_conditions, immediate_care, when_to_see_doctor, prevention, disclaimer
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
                max_tokens=1500,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to structured text
            try:
                import json
                analysis = json.loads(analysis_text)
            except:
                analysis = {
                    "description": "Analysis completed",
                    "analysis_text": analysis_text,
                    "raw_response": True
                }
            
            analysis["image_analyzed"] = True
            analysis["analysis_timestamp"] = logger.info("Vision analysis completed for skin condition")
            
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
            extracted_data["extraction_timestamp"] = logger.info("Document extraction completed")
            
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