"""
GPT-powered language detection and translation service.
"""
import logging
from typing import Optional, Dict, Any, Tuple
import re
from openai import AsyncOpenAI

from ..config.settings import settings

logger = logging.getLogger(__name__)


class LanguageProcessor:
    """GPT-powered language detection and translation."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.supported_languages = {
            "en": "English",
            "hi": "Hindi", 
            "ta": "Tamil",
            "te": "Telugu",
            "bn": "Bengali",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "mr": "Marathi",
            "pa": "Punjabi",
            "or": "Odia",
            "as": "Assamese",
            "ur": "Urdu"
        }
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of input text using GPT."""
        try:
            prompt = f"""
            Detect the language of the following text and return only the language code from this list:
            {', '.join([f'{code}: {name}' for code, name in self.supported_languages.items()])}
            
            If the language is not in the list, return 'en' for English.
            Return only the 2-letter language code, nothing else.
            
            Text: "{text}"
            """
            
            response = await self.client.chat.completions.create(
                model=settings.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            detected_lang = response.choices[0].message.content.strip().lower()
            
            # Validate the detected language
            if detected_lang in self.supported_languages:
                logger.info(f"Detected language: {detected_lang}")
                return detected_lang
            else:
                logger.warning(f"Unknown language detected: {detected_lang}, defaulting to English")
                return "en"
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"  # Default to English
    
    async def translate_to_english(self, text: str, source_language: str) -> str:
        """Translate text from source language to English."""
        if source_language == "en":
            return text
        
        try:
            source_lang_name = self.supported_languages.get(source_language, "Unknown")
            
            prompt = f"""
            Translate the following {source_lang_name} text to English. 
            Preserve the medical context and intent. If medical terms are used, 
            maintain their accuracy in translation.
            
            {source_lang_name} text: "{text}"
            
            English translation:
            """
            
            response = await self.client.chat.completions.create(
                model=settings.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.max_tokens,
                temperature=0.2
            )
            
            translated_text = response.choices[0].message.content.strip()
            logger.info(f"Translated from {source_language} to English")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation to English failed: {e}")
            return text  # Return original text if translation fails
    
    async def translate_from_english(self, text: str, target_language: str) -> str:
        """Translate text from English to target language."""
        if target_language == "en":
            return text
        
        try:
            target_lang_name = self.supported_languages.get(target_language, "English")
            
            prompt = f"""
            Translate the following English medical advice/response to {target_lang_name}.
            Maintain the medical accuracy and empathetic tone. Keep medical disclaimers clear.
            Use simple, understandable language that a common person can understand.
            
            English text: "{text}"
            
            {target_lang_name} translation:
            """
            
            response = await self.client.chat.completions.create(
                model=settings.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.max_tokens,
                temperature=0.2
            )
            
            translated_text = response.choices[0].message.content.strip()
            logger.info(f"Translated from English to {target_language}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation from English failed: {e}")
            return text  # Return original text if translation fails
    
    async def process_user_input(self, text: str) -> Tuple[str, str]:
        """
        Process user input: detect language and translate to English.
        Returns: (detected_language, english_text)
        """
        detected_language = await self.detect_language(text)
        english_text = await self.translate_to_english(text, detected_language)
        return detected_language, english_text
    
    async def process_bot_response(self, english_response: str, target_language: str) -> str:
        """
        Process bot response: translate from English to target language.
        Returns: translated_response
        """
        translated_response = await self.translate_from_english(english_response, target_language)
        return translated_response
    
    def is_supported_language(self, language_code: str) -> bool:
        """Check if language is supported."""
        return language_code in self.supported_languages
    
    def get_language_name(self, language_code: str) -> str:
        """Get language name from code."""
        return self.supported_languages.get(language_code, "Unknown")


# Global language processor instance
language_processor = LanguageProcessor()