#!/usr/bin/env python3
"""
Healthcare Bot - WhatsApp AI Healthcare Chatbot
Main application entry point.
"""
import uvicorn
from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )