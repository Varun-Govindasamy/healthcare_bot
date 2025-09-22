# API Documentation - WhatsApp AI Healthcare Chatbot

This document provides comprehensive API documentation for the WhatsApp AI Healthcare Chatbot.

## Base URL

```
Production: https://your-domain.com
Development: http://localhost:8000
```

## Authentication

Currently, the API uses basic authentication. For production deployment, implement proper JWT or API key authentication.

## Endpoints

### Health & Monitoring

#### GET /
Basic health check endpoint.

**Response:**
```
WhatsApp AI Healthcare Chatbot is running! 🏥🤖
```

**Status Codes:**
- `200`: Service is running

---

#### GET /health
Detailed health check with service status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "services": {
    "database": true,
    "twilio": true,
    "query_processor": true,
    "onboarding": true,
    "safety_validator": true,
    "mongodb": true,
    "redis": true,
    "sqlite": true
  }
}
```

**Status Codes:**
- `200`: All services healthy
- `503`: Some services unhealthy

---

### WhatsApp Integration

#### POST /webhook/whatsapp
Main webhook endpoint for receiving WhatsApp messages from Twilio.

**Headers:**
```
Content-Type: application/x-www-form-urlencoded
```

**Request Body (Form Data):**
```
From=whatsapp:+1234567890
To=whatsapp:+14155238886
Body=Hello, I need medical help
MessageSid=SM123456789
NumMedia=0
```

**Response:**
```
OK
```

**Status Codes:**
- `200`: Message received and queued for processing

**Notes:**
- This endpoint is called by Twilio when messages are received
- Processing happens in background to avoid webhook timeouts
- Always returns 200 to prevent Twilio retries

---

### Admin APIs

#### POST /api/send-message
Send a message programmatically (for testing/admin purposes).

**Request:**
```json
{
  "to": "whatsapp:+1234567890",
  "message": "Hello from the healthcare bot!",
  "media_url": "https://example.com/image.jpg" // optional
}
```

**Response:**
```json
{
  "success": true,
  "message_sid": "SM123456789",
  "sent_to": "whatsapp:+1234567890"
}
```

**Status Codes:**
- `200`: Message sent successfully
- `500`: Failed to send message

---

#### GET /api/user/{phone_number}
Get user profile information (admin/debugging purposes).

**Parameters:**
- `phone_number` (path): User's phone number (e.g., +1234567890)

**Response:**
```json
{
  "phone_number": "+1234567890",
  "name": "John Doe",
  "age": 30,
  "gender": "male",
  "city": "Mumbai",
  "state": "Maharashtra",
  "primary_language": "en",
  "emergency_contact": "+1234567891",
  "known_allergies": ["peanuts"],
  "current_medications": ["vitamin D"],
  "medical_history": "***REDACTED***",
  "onboarding_completed": true,
  "created_at": "2024-01-01T12:00:00.000Z",
  "updated_at": "2024-01-01T12:00:00.000Z"
}
```

**Status Codes:**
- `200`: User found
- `404`: User not found
- `500`: Internal error

**Notes:**
- Sensitive medical information is redacted in API responses
- Only use for admin/debugging purposes

---

#### DELETE /api/user/{phone_number}
Delete all user data (GDPR compliance).

**Parameters:**
- `phone_number` (path): User's phone number

**Response:**
```json
{
  "success": true,
  "message": "User data deleted successfully"
}
```

**Status Codes:**
- `200`: Data deleted successfully
- `500`: Failed to delete data

**Notes:**
- Removes data from MongoDB, SQLite, and Redis
- Irreversible operation
- Use for GDPR compliance requests

---

#### GET /api/stats
Get bot usage statistics.

**Response:**
```json
{
  "total_users": 1250,
  "completed_onboarding": 1100,
  "total_messages": 15750,
  "onboarding_completion_rate": 88.0
}
```

**Status Codes:**
- `200`: Statistics retrieved successfully
- `500`: Failed to get statistics

---

## Error Handling

### Standard Error Response

```json
{
  "error": "Error description",
  "message": "User-friendly error message"
}
```

### Common HTTP Status Codes

- `200`: Success
- `400`: Bad Request - Invalid input
- `404`: Not Found - Resource doesn't exist
- `500`: Internal Server Error - Server-side error
- `503`: Service Unavailable - Services not initialized

---

## Webhook Configuration

### Twilio Setup

1. **Webhook URL**: `https://your-domain.com/webhook/whatsapp`
2. **HTTP Method**: POST
3. **Content Type**: application/x-www-form-urlencoded

### Webhook Security

For production, implement webhook signature verification:

```python
from twilio.request_validator import RequestValidator

def verify_twilio_signature(request):
    validator = RequestValidator(auth_token)
    return validator.validate(
        uri=request.url,
        params=request.form,
        signature=request.headers.get('X-Twilio-Signature', '')
    )
```

---

## Data Models

### UserProfile

```json
{
  "phone_number": "string",
  "name": "string",
  "age": "integer",
  "gender": "male|female|other",
  "city": "string", 
  "state": "string",
  "primary_language": "string",
  "emergency_contact": "string",
  "known_allergies": ["string"],
  "current_medications": ["string"],
  "medical_history": "string",
  "onboarding_completed": "boolean",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### ChatMessage

```json
{
  "phone_number": "string",
  "message": "string",
  "response": "string",
  "timestamp": "datetime",
  "language": "string",
  "agent_used": "string",
  "safety_checked": "boolean"
}
```

### WhatsAppMessage

```json
{
  "from_number": "string",
  "to_number": "string", 
  "body": "string",
  "message_sid": "string",
  "media_urls": ["string"],
  "media_content_types": ["string"]
}
```

### HealthcareResponse

```json
{
  "message": "string",
  "media_url": "string",
  "safety_warnings": ["string"],
  "emergency_detected": "boolean",
  "language": "string",
  "agent_used": "string"
}
```

---

## Rate Limiting

Default rate limits (configurable via environment variables):

- **Per Minute**: 60 requests
- **Per Hour**: 1000 requests
- **Per User**: 10 messages per minute

Rate limit headers in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

---

## Multi-language Support

### Supported Languages

| Code | Language | Native Name |
|------|----------|-------------|
| en   | English  | English     |
| hi   | Hindi    | हिन्दी        |
| bn   | Bengali  | বাংলা        |
| te   | Telugu   | తెలుగు      |
| mr   | Marathi  | मराठी       |
| ta   | Tamil    | தமிழ்       |
| gu   | Gujarati | ગુજરાતી     |
| ur   | Urdu     | اردو        |
| kn   | Kannada  | ಕನ್ನಡ      |
| or   | Odia     | ଓଡ଼ିଆ       |
| pa   | Punjabi  | ਪੰਜਾਬੀ     |
| ml   | Malayalam| മലയാളം     |
| as   | Assamese | অসমীয়া     |
| mai  | Maithili | मैथिली      |

### Language Detection

The bot automatically detects user language using GPT-4o and translates messages accordingly.

---

## Safety & Validation

### Emergency Detection

The bot detects emergency keywords in multiple languages:

**English**: heart attack, can't breathe, severe pain, emergency, help me, dying, unconscious, overdose, poison, bleeding heavily, chest pain, stroke, choking, severe allergy, anaphylaxis

**Hindi**: दिल का दौरा, सांस नहीं आ रही, तेज दर्द, इमरजेंसी, मदद करो

**And more in other supported languages...**

### Safety Validations

1. **Age-based Validation**: Ensures medical advice is appropriate for user's age
2. **Medication Interactions**: Checks for dangerous drug combinations
3. **Allergy Contraindications**: Cross-checks medications against known allergies
4. **Emergency Response**: Immediate response with emergency contact information

### Medical Disclaimers

All medical responses include appropriate disclaimers:
- "This is not professional medical advice"
- "Consult with a healthcare provider"
- "For emergencies, call local emergency services"

---

## SDKs and Examples

### Python Client Example

```python
import aiohttp
import asyncio

class HealthcareBotClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def send_message(self, to, message):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/send-message", 
                                   json={"to": to, "message": message}) as response:
                return await response.json()

# Usage
async def main():
    client = HealthcareBotClient()
    result = await client.send_message(
        "whatsapp:+1234567890", 
        "Hello from the bot!"
    )
    print(result)

asyncio.run(main())
```

### cURL Examples

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Send Message:**
```bash
curl -X POST http://localhost:8000/api/send-message \
  -H "Content-Type: application/json" \
  -d '{
    "to": "whatsapp:+1234567890",
    "message": "Hello from the healthcare bot!"
  }'
```

**Get User Profile:**
```bash
curl http://localhost:8000/api/user/+1234567890
```

**Get Statistics:**
```bash
curl http://localhost:8000/api/stats
```

---

## Deployment

### Environment Variables

Required for production:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
PINECONE_API_KEY=your_pinecone_key
SERPER_API_KEY=your_serper_key

# Database URLs
MONGODB_URL=mongodb://user:pass@host:port/db
REDIS_URL=redis://user:pass@host:port
```

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Security**: Implement proper authentication and HTTPS
2. **Scaling**: Use multiple workers and load balancing
3. **Monitoring**: Set up logging, metrics, and alerting
4. **Backup**: Regular database backups
5. **Rate Limiting**: Implement per-user rate limiting
6. **Webhook Security**: Verify Twilio signatures

---

## Support

- **Documentation**: README.md
- **Testing**: `python scripts/test_bot.py`
- **Examples**: `python scripts/examples.py`
- **Setup**: `python scripts/setup_environment.py`

For issues and support, check the application logs in the `logs/` directory.