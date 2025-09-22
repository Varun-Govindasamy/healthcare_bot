# WhatsApp AI Healthcare Chatbot 🏥🤖

A comprehensive AI-powered healthcare chatbot that provides medical information and assistance through WhatsApp, featuring multi-language support, specialized AI agents, and comprehensive safety validation.

## 🌟 Features

- **Multi-language Support**: 13 Indian languages + English with GPT-powered translation
- **5 Specialized AI Agents**: Medical Data, RAG, Search, Vision, and Conversation agents
- **Safety Validation**: Emergency detection, age-based validation, medication interactions
- **Vector Database**: Healthcare knowledge base with user document management
- **Comprehensive Onboarding**: Mandatory profile collection with validation
- **WhatsApp Integration**: Full Twilio integration with file upload support
- **Database Architecture**: MongoDB, SQLite, Redis, and Pinecone integration

## 🏗️ Architecture

### Core Components

1. **FastAPI Backend** (`src/api/main.py`)
   - WhatsApp webhook handling
   - RESTful API endpoints
   - Background task processing
   - Health checks and monitoring

2. **CrewAI Agents** (`src/agents/`)
   - Medical Data Agent: Handles medical information requests
   - RAG Agent: Retrieves relevant healthcare documents
   - Search Agent: Performs web searches for medical information
   - Vision Agent: Analyzes medical images and reports
   - Conversation Agent: Orchestrates agent interactions

3. **Database Layer** (`src/database/`)
   - MongoDB: User profiles and medical history
   - SQLite: Chat history and session management
   - Redis: Caching and temporary data
   - Pinecone: Vector storage for healthcare knowledge

4. **Services** (`src/services/`)
   - Language Processor: Multi-language detection and translation
   - Twilio Service: WhatsApp message handling
   - Onboarding Service: User profile collection
   - Query Processor: Main message processing pipeline
   - Pinecone Service: Vector database operations

5. **Safety & Validation** (`src/utils/`)
   - Emergency detection (30+ keywords)
   - Age-based medical advice validation
   - Medication interaction checking
   - Contraindication analysis

## 📋 Prerequisites

- Python 3.12.6 or higher
- MongoDB instance
- Redis instance
- Twilio account with WhatsApp Business API
- OpenAI API key
- Pinecone API key
- Serper API key (for web search)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sih_proj
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\\Scripts\\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key
   
   # Twilio Configuration
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
   
   # Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=healthcare-knowledge
   
   # Serper API (for web search)
   SERPER_API_KEY=your_serper_api_key
   
   # Database Configuration
   MONGODB_URL=mongodb://localhost:27017
   MONGODB_DATABASE=healthcare_bot
   REDIS_URL=redis://localhost:6379
   SQLITE_DATABASE=./data/healthcare_bot.db
   
   # Application Settings
   DEBUG=true
   LOG_LEVEL=INFO
   MAX_FILE_SIZE_MB=10
   ```

5. **Initialize databases**
   ```bash
   # Start MongoDB and Redis services
   # Create necessary directories
   mkdir -p data logs
   ```

## 🏃‍♂️ Running the Application

### Development Mode

```bash
python main.py
```

The application will start on `http://localhost:8000`

### Production Mode

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📱 WhatsApp Setup

1. **Configure Twilio Webhook**
   - Set webhook URL to: `https://your-domain.com/webhook/whatsapp`
   - Enable WhatsApp sandbox or configure production WhatsApp Business API

2. **Test the Bot**
   - Send a message to your Twilio WhatsApp number
   - The bot will start the onboarding process for new users

## 🔧 API Endpoints

### Health & Monitoring

- `GET /` - Basic health check
- `GET /health` - Detailed health status with service checks

### WhatsApp Integration

- `POST /webhook/whatsapp` - Main WhatsApp webhook endpoint

### Admin APIs

- `POST /api/send-message` - Send message programmatically
- `GET /api/user/{phone_number}` - Get user profile
- `DELETE /api/user/{phone_number}` - Delete user data (GDPR)
- `GET /api/stats` - Bot usage statistics

## 🧪 Testing

### Manual Testing

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Send Test Message**
   ```bash
   curl -X POST http://localhost:8000/api/send-message \\
     -H "Content-Type: application/json" \\
     -d '{
       "to": "whatsapp:+1234567890",
       "message": "Hello from the healthcare bot!"
     }'
   ```

### Automated Testing

Run the test suite:
```bash
python scripts/test_bot.py
```

## 🔄 User Flow

### New User Journey

1. **First Contact**: User sends any message to WhatsApp number
2. **Onboarding Initiation**: Bot detects new user and starts profile collection
3. **Profile Collection**: 9-step mandatory onboarding process:
   - Name
   - Age
   - Gender
   - Location (City/State)
   - Primary language preference
   - Emergency contact
   - Known allergies
   - Current medications
   - Medical history
4. **Validation**: Each step is validated before proceeding
5. **Completion**: User profile is saved and bot is ready for medical queries

### Existing User Journey

1. **Message Processing**: Incoming message is processed through language detection
2. **Safety Check**: Message is validated for emergency keywords and safety
3. **Agent Coordination**: Appropriate agents are selected based on query type
4. **Response Generation**: Agents collaborate to generate comprehensive response
5. **Translation**: Response is translated back to user's preferred language
6. **Delivery**: Response is sent via WhatsApp with appropriate disclaimers

## 🛡️ Safety Features

### Emergency Detection
- 30+ emergency keywords in multiple languages
- Immediate emergency response with local contact information
- Escalation protocols for critical situations

### Medical Safety
- Age-based advice validation (pediatric/elderly considerations)
- Medication interaction checking
- Allergy contraindication analysis
- Medical disclaimer inclusion

### Data Privacy
- User data encryption
- GDPR compliance with data deletion
- Secure API endpoints
- Audit logging

## 🌐 Multi-language Support

### Supported Languages
- English
- Hindi
- Bengali
- Telugu
- Marathi
- Tamil
- Gujarati
- Urdu
- Kannada
- Odia
- Punjabi
- Malayalam
- Assamese
- Maithili

### Language Processing
- Automatic language detection using GPT-4o
- Context-aware translation
- Culturally appropriate medical terminology
- Regional dialect support

## 📊 Monitoring & Logging

### Application Logs
- Structured logging with timestamps
- Error tracking and alerting
- Performance metrics
- User interaction analytics

### Health Monitoring
- Database connection status
- Service availability checks
- API response times
- Error rates and patterns

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes |
| `TWILIO_ACCOUNT_SID` | Twilio account identifier | Yes |
| `TWILIO_AUTH_TOKEN` | Twilio authentication token | Yes |
| `PINECONE_API_KEY` | Pinecone vector database key | Yes |
| `SERPER_API_KEY` | Serper web search API key | Yes |
| `MONGODB_URL` | MongoDB connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `DEBUG` | Enable debug mode | No |
| `MAX_FILE_SIZE_MB` | Maximum file upload size | No |

### Database Configuration

The application uses multiple databases for different purposes:

- **MongoDB**: User profiles, medical history
- **SQLite**: Chat sessions, conversation history
- **Redis**: Temporary caching, session management
- **Pinecone**: Vector embeddings, knowledge base

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check MongoDB status
   mongosh --eval "db.adminCommand('ismaster')"
   
   # Check Redis status
   redis-cli ping
   ```

2. **API Key Issues**
   ```bash
   # Verify environment variables
   python -c "from src.config.settings import Settings; print(Settings().openai_api_key[:10])"
   ```

3. **Webhook Not Receiving Messages**
   - Verify Twilio webhook URL configuration
   - Check firewall and port accessibility
   - Validate SSL certificate if using HTTPS

### Debug Mode

Enable debug logging:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

## 📝 Development

### Project Structure

```
sih_proj/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables
├── src/
│   ├── api/
│   │   ├── main.py        # FastAPI application
│   │   └── __init__.py
│   ├── agents/            # CrewAI agents
│   │   ├── medical_data_agent.py
│   │   ├── rag_agent.py
│   │   ├── search_agent.py
│   │   ├── vision_agent.py
│   │   └── conversation_agent.py
│   ├── database/          # Database layers
│   │   ├── mongodb.py
│   │   ├── sqlite.py
│   │   ├── redis_cache.py
│   │   └── manager.py
│   ├── services/          # Business logic
│   │   ├── language_processor.py
│   │   ├── twilio_service.py
│   │   ├── onboarding_service.py
│   │   ├── query_processor.py
│   │   └── pinecone_service.py
│   ├── models/            # Data models
│   │   └── schemas.py
│   ├── config/            # Configuration
│   │   └── settings.py
│   └── utils/             # Utilities
│       └── safety_validator.py
├── scripts/               # Testing and utility scripts
├── data/                  # SQLite database files
└── logs/                  # Application logs
```

### Adding New Features

1. **New Agent**: Create in `src/agents/` and register in query processor
2. **New Service**: Add to `src/services/` and update dependency injection
3. **New API Endpoint**: Add to `src/api/main.py` with proper validation
4. **New Database Model**: Update `src/models/schemas.py` and repositories

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For technical support or questions:
- Email: support@healthcarebot.com
- Documentation: [Project Wiki]
- Issues: [GitHub Issues]

## ⚖️ Disclaimer

This healthcare chatbot is designed to provide general medical information and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

## 🎯 Roadmap

- [ ] Voice message support
- [ ] Integration with electronic health records
- [ ] Advanced symptom checker
- [ ] Appointment scheduling
- [ ] Medication reminders
- [ ] Telemedicine integration