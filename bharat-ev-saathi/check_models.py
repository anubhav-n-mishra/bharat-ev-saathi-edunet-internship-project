"""
Check available Gemini models
"""
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if api_key:
    genai.configure(api_key=api_key)
    print("✅ API Key configured")
    print("\n📋 Available models:\n")
    
    for model in genai.list_models():
        print(f"Model: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        print()
else:
    print("❌ No API key found")
