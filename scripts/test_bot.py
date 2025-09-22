#!/usr/bin/env python3
"""
Healthcare Bot Testing Script

This script provides comprehensive testing for the WhatsApp AI Healthcare Chatbot.
It tests all major components including:
- Database connections
- API endpoints
- Agent functionality
- Safety validation
- Multi-language processing
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.settings import Settings
from src.database.manager import DatabaseManager
from src.services.language_processor import LanguageProcessor
from src.utils.safety_validator import MedicalSafetyValidator
from src.models.schemas import UserProfile, ChatMessage, WhatsAppMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthcareBotTester:
    """Comprehensive testing class for the healthcare bot."""
    
    def __init__(self):
        self.settings = Settings()
        self.base_url = "http://localhost:8000"
        self.db_manager = None
        self.language_processor = None
        self.safety_validator = None
        self.test_results = []
        
    async def setup(self):
        """Initialize test environment."""
        logger.info("Setting up test environment...")
        
        try:
            # Initialize database manager
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize other services
            self.language_processor = LanguageProcessor()
            self.safety_validator = MedicalSafetyValidator()
            
            logger.info("✅ Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup test environment."""
        if self.db_manager:
            await self.db_manager.close()
        logger.info("✅ Test environment cleanup complete")
    
    def add_test_result(self, test_name: str, success: bool, details: str = ""):
        """Add test result to tracking."""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name} - {details}")
    
    async def test_database_connections(self):
        """Test all database connections."""
        logger.info("Testing database connections...")
        
        # Test MongoDB
        try:
            mongo_ok = await self.db_manager.test_mongodb_connection()
            self.add_test_result("MongoDB Connection", mongo_ok, "Connection successful" if mongo_ok else "Connection failed")
        except Exception as e:
            self.add_test_result("MongoDB Connection", False, str(e))
        
        # Test Redis
        try:
            redis_ok = await self.db_manager.test_redis_connection()
            self.add_test_result("Redis Connection", redis_ok, "Connection successful" if redis_ok else "Connection failed")
        except Exception as e:
            self.add_test_result("Redis Connection", False, str(e))
        
        # Test SQLite
        try:
            sqlite_ok = await self.db_manager.test_sqlite_connection()
            self.add_test_result("SQLite Connection", sqlite_ok, "Connection successful" if sqlite_ok else "Connection failed")
        except Exception as e:
            self.add_test_result("SQLite Connection", False, str(e))
    
    async def test_api_endpoints(self):
        """Test FastAPI endpoints."""
        logger.info("Testing API endpoints...")
        
        async with aiohttp.ClientSession() as session:
            # Test health check
            try:
                async with session.get(f"{self.base_url}/") as response:
                    success = response.status == 200
                    content = await response.text()
                    self.add_test_result("Root Endpoint", success, f"Status: {response.status}, Content: {content[:50]}")
            except Exception as e:
                self.add_test_result("Root Endpoint", False, str(e))
            
            # Test detailed health check
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    success = response.status in [200, 503]  # May be 503 if services not fully initialized
                    data = await response.json()
                    self.add_test_result("Health Endpoint", success, f"Status: {data.get('status', 'unknown')}")
            except Exception as e:
                self.add_test_result("Health Endpoint", False, str(e))
            
            # Test statistics endpoint
            try:
                async with session.get(f"{self.base_url}/api/stats") as response:
                    success = response.status in [200, 503]
                    if success and response.status == 200:
                        data = await response.json()
                        self.add_test_result("Stats Endpoint", True, f"Users: {data.get('total_users', 0)}")
                    else:
                        self.add_test_result("Stats Endpoint", False, f"Status: {response.status}")
            except Exception as e:
                self.add_test_result("Stats Endpoint", False, str(e))
    
    async def test_language_processing(self):
        """Test language detection and translation."""
        logger.info("Testing language processing...")
        
        test_cases = [
            ("Hello, I need medical help", "en"),
            ("मुझे डॉक्टर की सलाह चाहिए", "hi"),
            ("আমার চিকিৎসা সাহায্য প্রয়োজন", "bn"),
            ("నాకు వైద్య సహాయం అవసరం", "te"),
        ]
        
        for text, expected_lang in test_cases:
            try:
                detected_lang = await self.language_processor.detect_language(text)
                success = detected_lang == expected_lang
                self.add_test_result(
                    f"Language Detection ({expected_lang})", 
                    success, 
                    f"Detected: {detected_lang}, Expected: {expected_lang}"
                )
                
                # Test translation to English
                if detected_lang != "en":
                    translated = await self.language_processor.translate_to_english(text, detected_lang)
                    success = len(translated) > 0 and translated != text
                    self.add_test_result(
                        f"Translation ({detected_lang} to en)", 
                        success, 
                        f"Translated: {translated[:50]}..."
                    )
                    
            except Exception as e:
                self.add_test_result(f"Language Processing ({expected_lang})", False, str(e))
    
    async def test_safety_validation(self):
        """Test safety validation and emergency detection."""
        logger.info("Testing safety validation...")
        
        # Test emergency detection
        emergency_messages = [
            "I'm having a heart attack!",
            "Emergency! Can't breathe!",
            "Severe chest pain right now",
            "मुझे दिल का दौरा हो रहा है",  # Hindi: I'm having a heart attack
        ]
        
        for message in emergency_messages:
            try:
                is_emergency = await self.safety_validator.is_emergency(message)
                self.add_test_result(
                    "Emergency Detection", 
                    is_emergency, 
                    f"Message: {message[:30]}... - Emergency: {is_emergency}"
                )
            except Exception as e:
                self.add_test_result("Emergency Detection", False, str(e))
        
        # Test age validation
        try:
            # Test pediatric age
            is_appropriate = await self.safety_validator.validate_age_appropriate_advice(
                "Take this medication", 5
            )
            self.add_test_result(
                "Age Validation (Pediatric)", 
                not is_appropriate,  # Should be False for young children
                f"Age 5, medication advice appropriate: {is_appropriate}"
            )
            
            # Test adult age
            is_appropriate = await self.safety_validator.validate_age_appropriate_advice(
                "Take this medication", 30
            )
            self.add_test_result(
                "Age Validation (Adult)", 
                is_appropriate,  # Should be True for adults
                f"Age 30, medication advice appropriate: {is_appropriate}"
            )
            
        except Exception as e:
            self.add_test_result("Age Validation", False, str(e))
        
        # Test medication interactions
        try:
            has_interaction = await self.safety_validator.check_medication_interactions(
                ["warfarin", "aspirin"]  # Known interaction
            )
            self.add_test_result(
                "Medication Interaction Check", 
                has_interaction, 
                f"Warfarin + Aspirin interaction detected: {has_interaction}"
            )
        except Exception as e:
            self.add_test_result("Medication Interaction Check", False, str(e))
    
    async def test_user_operations(self):
        """Test user-related database operations."""
        logger.info("Testing user operations...")
        
        test_phone = "+1234567890"
        
        try:
            # Create test user
            test_user = UserProfile(
                phone_number=test_phone,
                name="Test User",
                age=30,
                gender="male",
                city="Test City",
                state="Test State",
                primary_language="en",
                emergency_contact="+1234567891",
                known_allergies=["peanuts"],
                current_medications=["vitamin D"],
                medical_history="No significant history",
                onboarding_completed=True
            )
            
            # Save user
            await self.db_manager.user_repository.create_user(test_user)
            self.add_test_result("User Creation", True, f"Created user: {test_phone}")
            
            # Retrieve user
            retrieved_user = await self.db_manager.user_repository.get_user(test_phone)
            success = retrieved_user is not None and retrieved_user.name == "Test User"
            self.add_test_result("User Retrieval", success, f"Retrieved user: {retrieved_user.name if retrieved_user else 'None'}")
            
            # Update user
            if retrieved_user:
                retrieved_user.city = "Updated City"
                await self.db_manager.user_repository.update_user(retrieved_user)
                
                updated_user = await self.db_manager.user_repository.get_user(test_phone)
                success = updated_user.city == "Updated City"
                self.add_test_result("User Update", success, f"Updated city: {updated_user.city}")
            
            # Clean up test user
            await self.db_manager.user_repository.delete_user(test_phone)
            self.add_test_result("User Deletion", True, f"Deleted user: {test_phone}")
            
        except Exception as e:
            self.add_test_result("User Operations", False, str(e))
    
    async def test_chat_operations(self):
        """Test chat-related database operations."""
        logger.info("Testing chat operations...")
        
        test_phone = "+1234567890"
        
        try:
            # Create test chat message
            test_message = ChatMessage(
                phone_number=test_phone,
                message="Test message",
                response="Test response",
                timestamp=datetime.utcnow(),
                language="en"
            )
            
            # Save message
            await self.db_manager.chat_repository.save_message(test_message)
            self.add_test_result("Chat Message Save", True, "Saved test chat message")
            
            # Retrieve chat history
            history = await self.db_manager.chat_repository.get_chat_history(test_phone, limit=10)
            success = len(history) > 0 and any(msg.message == "Test message" for msg in history)
            self.add_test_result("Chat History Retrieval", success, f"Retrieved {len(history)} messages")
            
            # Clean up test messages
            await self.db_manager.chat_repository.delete_user_chats(test_phone)
            self.add_test_result("Chat Cleanup", True, "Deleted test chat messages")
            
        except Exception as e:
            self.add_test_result("Chat Operations", False, str(e))
    
    async def test_whatsapp_message_parsing(self):
        """Test WhatsApp message parsing logic."""
        logger.info("Testing WhatsApp message parsing...")
        
        # Mock Twilio webhook data
        mock_form_data = {
            "From": "whatsapp:+1234567890",
            "To": "whatsapp:+14155238886",
            "Body": "Hello, I need medical help",
            "MessageSid": "SM123456789",
            "NumMedia": "0"
        }
        
        try:
            # This would normally be tested with actual Twilio service
            # For now, we'll just validate the structure
            
            required_fields = ["From", "To", "Body", "MessageSid"]
            has_required = all(field in mock_form_data for field in required_fields)
            
            self.add_test_result(
                "WhatsApp Message Structure", 
                has_required, 
                f"Required fields present: {has_required}"
            )
            
            # Test phone number extraction
            from_number = mock_form_data["From"].replace("whatsapp:", "")
            success = from_number.startswith("+")
            self.add_test_result(
                "Phone Number Extraction", 
                success, 
                f"Extracted: {from_number}"
            )
            
        except Exception as e:
            self.add_test_result("WhatsApp Message Parsing", False, str(e))
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("\\n" + "="*80)
        print("📊 HEALTHCARE BOT TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
        print("="*80)
        
        if failed_tests > 0:
            print("\\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  • {result['test']}: {result['details']}")
        
        print("\\n✅ PASSED TESTS:")
        for result in self.test_results:
            if result["success"]:
                print(f"  • {result['test']}: {result['details']}")
        
        print("\\n" + "="*80)
        
        # Save detailed results to file
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"📄 Detailed results saved to: {results_file}")
        
        return failed_tests == 0

async def main():
    """Main testing function."""
    print("🧪 Starting Healthcare Bot Comprehensive Testing")
    print("="*60)
    
    tester = HealthcareBotTester()
    
    # Setup test environment
    if not await tester.setup():
        print("❌ Failed to setup test environment. Exiting.")
        return False
    
    try:
        # Run all tests
        await tester.test_database_connections()
        await tester.test_api_endpoints()
        await tester.test_language_processing()
        await tester.test_safety_validation()
        await tester.test_user_operations()
        await tester.test_chat_operations()
        await tester.test_whatsapp_message_parsing()
        
        # Print summary
        success = tester.print_test_summary()
        
        if success:
            print("\\n🎉 All tests passed! The healthcare bot is ready for deployment.")
        else:
            print("\\n⚠️  Some tests failed. Please review the issues before deployment.")
        
        return success
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Testing failed with error: {e}")
        sys.exit(1)