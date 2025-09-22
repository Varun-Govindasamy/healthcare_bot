#!/usr/bin/env python3
"""
Environment Setup Script for WhatsApp AI Healthcare Chatbot

This script helps set up the development environment by:
- Creating necessary directories
- Generating example .env file
- Validating Python version
- Checking required system dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print setup banner."""
    print("🏥🤖 WhatsApp AI Healthcare Chatbot - Environment Setup")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Setup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    print("\\n🐍 Checking Python version...")
    
    major, minor = sys.version_info[:2]
    
    if major != 3 or minor < 10:
        print(f"❌ Python 3.10+ required. Current: {major}.{minor}")
        print("Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {major}.{minor} is compatible")
    return True

def create_directories():
    """Create necessary project directories."""
    print("\\n📁 Creating project directories...")
    
    directories = [
        "data",
        "logs", 
        "uploads",
        "scripts",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}/")
        else:
            print(f"📁 Exists: {directory}/")

def create_env_template():
    """Create example .env file."""
    print("\\n⚙️  Creating environment template...")
    
    env_template = '''# WhatsApp AI Healthcare Chatbot Configuration
# Copy this file to .env and fill in your actual values

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Twilio Configuration (WhatsApp Business API)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token  
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# Pinecone Configuration (Vector Database)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=healthcare-knowledge

# Serper API (Web Search)
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
API_HOST=0.0.0.0
API_PORT=8000

# Security (Generate strong secrets for production)
SECRET_KEY=your_secret_key_for_jwt_tokens
WEBHOOK_VERIFY_TOKEN=your_webhook_verification_token

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Medical Disclaimer Settings
INCLUDE_MEDICAL_DISCLAIMER=true
EMERGENCY_CONTACT_INFO=Call 911 for emergencies
'''

    env_file = Path(".env.example")
    
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("✅ Created: .env.example")
        print("📝 Copy .env.example to .env and add your API keys")
    else:
        print("📁 Exists: .env.example")

def check_system_dependencies():
    """Check for required system dependencies."""
    print("\\n🔧 Checking system dependencies...")
    
    dependencies = {
        "git": "git --version",
        "curl": "curl --version" if platform.system() != "Windows" else None
    }
    
    for name, command in dependencies.items():
        if command is None:
            continue
            
        try:
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ {name} is available")
            else:
                print(f"⚠️  {name} check failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ {name} not found")

def check_python_packages():
    """Check if required Python packages can be imported."""
    print("\\n📦 Checking Python packages...")
    
    # Core packages that should be available
    core_packages = [
        "json",
        "os", 
        "sys",
        "datetime",
        "pathlib",
        "asyncio",
        "logging"
    ]
    
    for package in core_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Core package missing!")

def create_gitignore():
    """Create .gitignore file."""
    print("\\n🔒 Creating .gitignore...")
    
    gitignore_content = '''# Environment variables
.env
.env.local
.env.production

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log
healthcare_bot.log

# Database files
data/
*.db
*.sqlite

# Uploads
uploads/
temp/

# Test results
test_results_*.json
coverage/
.coverage
.pytest_cache/

# API keys and secrets
secrets/
keys/
certificates/

# Temporary files
tmp/
temp/
.tmp/
'''

    gitignore_file = Path(".gitignore")
    
    if not gitignore_file.exists():
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content)
        print("✅ Created: .gitignore")
    else:
        print("📁 Exists: .gitignore")

def create_startup_script():
    """Create startup script for different platforms."""
    print("\\n🚀 Creating startup scripts...")
    
    # Windows startup script
    windows_script = '''@echo off
echo Starting WhatsApp AI Healthcare Chatbot...
echo.

REM Activate virtual environment if it exists
if exist venv\\Scripts\\activate.bat (
    echo Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Check if .env exists
if not exist .env (
    echo WARNING: .env file not found!
    echo Please copy .env.example to .env and configure your API keys.
    pause
    exit /b 1
)

REM Start the application
echo Starting FastAPI server...
python main.py

pause
'''

    # Unix startup script  
    unix_script = '''#!/bin/bash

echo "Starting WhatsApp AI Healthcare Chatbot..."
echo

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please copy .env.example to .env and configure your API keys."
    exit 1
fi

# Start the application
echo "Starting FastAPI server..."
python main.py
'''

    # Create Windows script
    with open("start.bat", 'w') as f:
        f.write(windows_script)
    print("✅ Created: start.bat (Windows)")
    
    # Create Unix script
    with open("start.sh", 'w') as f:
        f.write(unix_script)
    
    # Make Unix script executable
    if platform.system() != "Windows":
        os.chmod("start.sh", 0o755)
    
    print("✅ Created: start.sh (Unix/Linux/Mac)")

def print_next_steps():
    """Print next steps for the user."""
    print("\\n" + "=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)
    
    steps = [
        "1. Install dependencies: pip install -r requirements.txt",
        "2. Copy .env.example to .env and configure your API keys",
        "3. Set up MongoDB and Redis services",
        "4. Run tests: python scripts/test_bot.py",
        "5. Start the bot: python main.py",
        "",
        "📚 For detailed setup instructions, see README.md",
        "🔧 For troubleshooting, check the logs/ directory",
        "🧪 Run tests regularly during development"
    ]
    
    for step in steps:
        if step:
            print(f"   {step}")
        else:
            print()

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create project structure
    create_directories()
    create_env_template()
    create_gitignore()
    create_startup_scripts()
    
    # System checks
    check_system_dependencies()
    check_python_packages()
    
    print("\\n✅ Environment setup completed successfully!")
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Setup failed: {e}")
        sys.exit(1)