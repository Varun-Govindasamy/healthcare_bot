#!/usr/bin/env python3
"""
WhatsApp Healthcare Bot - Example Usage Script

This script demonstrates how to use the healthcare bot APIs
and provides examples of common operations.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareBotClient:
    """Client for interacting with the Healthcare Bot API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def health_check(self) -> Dict[str, Any]:
        """Check bot health status."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def send_message(self, to: str, message: str, media_url: str = None) -> Dict[str, Any]:
        """Send a message via the bot."""
        data = {
            "to": to,
            "message": message
        }
        
        if media_url:
            data["media_url"] = media_url
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/send-message", json=data) as response:
                return await response.json()
    
    async def get_user_profile(self, phone_number: str) -> Dict[str, Any]:
        """Get user profile information."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/user/{phone_number}") as response:
                if response.status == 404:
                    return {"error": "User not found"}
                return await response.json()
    
    async def get_bot_statistics(self) -> Dict[str, Any]:
        """Get bot usage statistics."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/stats") as response:
                return await response.json()

async def example_health_check():
    """Example: Check bot health."""
    print("\\n🏥 Checking Bot Health...")
    print("-" * 40)
    
    client = HealthcareBotClient()
    
    try:
        health = await client.health_check()
        print(f"Status: {health.get('status', 'unknown')}")
        print(f"Timestamp: {health.get('timestamp', 'unknown')}")
        
        services = health.get('services', {})
        for service, status in services.items():
            emoji = "✅" if status else "❌"
            print(f"{emoji} {service}: {status}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")

async def example_bot_statistics():
    """Example: Get bot statistics."""
    print("\\n📊 Bot Usage Statistics...")
    print("-" * 40)
    
    client = HealthcareBotClient()
    
    try:
        stats = await client.get_bot_statistics()
        print(f"Total Users: {stats.get('total_users', 0)}")
        print(f"Completed Onboarding: {stats.get('completed_onboarding', 0)}")
        print(f"Total Messages: {stats.get('total_messages', 0)}")
        print(f"Onboarding Rate: {stats.get('onboarding_completion_rate', 0)}%")
        
    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")

async def example_send_test_message():
    """Example: Send a test message."""
    print("\\n💬 Sending Test Message...")
    print("-" * 40)
    
    client = HealthcareBotClient()
    
    # Note: This will only work if you have a valid test phone number
    # and the Twilio service is properly configured
    test_phone = "whatsapp:+1234567890"  # Replace with your test number
    test_message = "Hello! This is a test message from the healthcare bot."
    
    try:
        result = await client.send_message(test_phone, test_message)
        
        if result.get('success'):
            print(f"✅ Message sent successfully!")
            print(f"Message SID: {result.get('message_sid')}")
            print(f"Sent to: {result.get('sent_to')}")
        else:
            print(f"❌ Failed to send message: {result}")
            
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        print("Note: This requires proper Twilio configuration and valid phone number")

def example_onboarding_flow():
    """Example: Show the onboarding question flow."""
    print("\\n📝 User Onboarding Flow Example...")
    print("-" * 40)
    
    onboarding_questions = [
        {
            "step": 1,
            "question": "What's your full name?",
            "example": "John Doe",
            "validation": "Must be at least 2 characters"
        },
        {
            "step": 2,
            "question": "How old are you?",
            "example": "25",
            "validation": "Must be between 1 and 120"
        },
        {
            "step": 3,
            "question": "What's your gender?",
            "example": "male/female/other",
            "validation": "Must be male, female, or other"
        },
        {
            "step": 4,
            "question": "Which city are you from?",
            "example": "Mumbai",
            "validation": "Required field"
        },
        {
            "step": 5,
            "question": "Which state are you from?",
            "example": "Maharashtra",
            "validation": "Required field"
        },
        {
            "step": 6,
            "question": "What's your preferred language?",
            "example": "English/Hindi/Bengali/etc.",
            "validation": "Must be supported language"
        },
        {
            "step": 7,
            "question": "Emergency contact number?",
            "example": "+91-9876543210",
            "validation": "Valid phone number format"
        },
        {
            "step": 8,
            "question": "Do you have any known allergies?",
            "example": "Peanuts, Shellfish (or 'None')",
            "validation": "Optional, comma-separated list"
        },
        {
            "step": 9,
            "question": "Current medications?",
            "example": "Vitamin D, Aspirin (or 'None')",
            "validation": "Optional, comma-separated list"
        }
    ]
    
    for q in onboarding_questions:
        print(f"Step {q['step']}: {q['question']}")
        print(f"   Example: {q['example']}")
        print(f"   Validation: {q['validation']}")
        print()

def example_supported_languages():
    """Example: Show supported languages."""
    print("\\n🌐 Supported Languages...")
    print("-" * 40)
    
    languages = [
        {"code": "en", "name": "English", "native": "English"},
        {"code": "hi", "name": "Hindi", "native": "हिन्दी"},
        {"code": "bn", "name": "Bengali", "native": "বাংলা"},
        {"code": "te", "name": "Telugu", "native": "తెలుగు"},
        {"code": "mr", "name": "Marathi", "native": "मराठी"},
        {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
        {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી"},
        {"code": "ur", "name": "Urdu", "native": "اردو"},
        {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ"},
        {"code": "or", "name": "Odia", "native": "ଓଡ଼ିଆ"},
        {"code": "pa", "name": "Punjabi", "native": "ਪੰਜਾਬੀ"},
        {"code": "ml", "name": "Malayalam", "native": "മലയാളം"},
        {"code": "as", "name": "Assamese", "native": "অসমীয়া"},
        {"code": "mai", "name": "Maithili", "native": "मैथिली"}
    ]
    
    for lang in languages:
        print(f"{lang['code']}: {lang['name']} ({lang['native']})")

def example_safety_features():
    """Example: Show safety features."""
    print("\\n🛡️ Safety Features...")
    print("-" * 40)
    
    safety_features = [
        {
            "feature": "Emergency Detection",
            "description": "Automatically detects emergency keywords in 30+ terms",
            "examples": ["heart attack", "can't breathe", "severe pain", "emergency"]
        },
        {
            "feature": "Age-based Validation", 
            "description": "Validates medical advice appropriateness by age group",
            "examples": ["Pediatric considerations", "Elderly care modifications", "Adult dosages"]
        },
        {
            "feature": "Medication Interactions",
            "description": "Checks for dangerous drug interactions",
            "examples": ["Warfarin + Aspirin", "MAOIs + SSRIs", "Blood thinners combinations"]
        },
        {
            "feature": "Allergy Contraindications",
            "description": "Cross-checks allergies against medications",
            "examples": ["Penicillin allergy", "Sulfa drug reactions", "Food-drug interactions"]
        },
        {
            "feature": "Medical Disclaimers",
            "description": "Automatically adds appropriate medical disclaimers",
            "examples": ["Consult healthcare provider", "Emergency contact info", "Not medical advice"]
        }
    ]
    
    for feature in safety_features:
        print(f"• {feature['feature']}")
        print(f"  {feature['description']}")
        print(f"  Examples: {', '.join(feature['examples'])}")
        print()

def example_api_endpoints():
    """Example: Show available API endpoints."""
    print("\\n🔗 Available API Endpoints...")
    print("-" * 40)
    
    endpoints = [
        {
            "method": "GET",
            "path": "/",
            "description": "Basic health check"
        },
        {
            "method": "GET", 
            "path": "/health",
            "description": "Detailed health status with service checks"
        },
        {
            "method": "POST",
            "path": "/webhook/whatsapp",
            "description": "Main WhatsApp webhook for Twilio"
        },
        {
            "method": "POST",
            "path": "/api/send-message", 
            "description": "Send message programmatically"
        },
        {
            "method": "GET",
            "path": "/api/user/{phone_number}",
            "description": "Get user profile information"
        },
        {
            "method": "DELETE",
            "path": "/api/user/{phone_number}",
            "description": "Delete user data (GDPR compliance)"
        },
        {
            "method": "GET",
            "path": "/api/stats",
            "description": "Get bot usage statistics"
        }
    ]
    
    for endpoint in endpoints:
        print(f"{endpoint['method']} {endpoint['path']}")
        print(f"   {endpoint['description']}")
        print()

async def main():
    """Main example function."""
    print("🤖 WhatsApp AI Healthcare Chatbot - Usage Examples")
    print("=" * 60)
    
    # Show static examples
    example_supported_languages()
    example_onboarding_flow()
    example_safety_features()
    example_api_endpoints()
    
    # Try dynamic examples (these require the bot to be running)
    print("\\n🔄 Dynamic Examples (requires running bot)...")
    print("=" * 60)
    
    try:
        await example_health_check()
        await example_bot_statistics()
        
        # Uncomment the next line to test message sending (requires proper setup)
        # await example_send_test_message()
        
    except Exception as e:
        print(f"⚠️  Dynamic examples skipped: {e}")
        print("Make sure the bot is running on http://localhost:8000")
    
    print("\\n" + "=" * 60)
    print("📚 For more information, see README.md")
    print("🧪 Run 'python scripts/test_bot.py' for comprehensive testing")
    print("🚀 Start the bot with 'python main.py'")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n🛑 Examples interrupted by user")
    except Exception as e:
        print(f"\\n💥 Examples failed: {e}")