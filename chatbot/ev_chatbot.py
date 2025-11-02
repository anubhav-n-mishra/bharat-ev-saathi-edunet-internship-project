"""
EV Chatbot - Professional AI Assistant for Electric Vehicles
=============================================================
Powered by Google Gemini Pro API
Answers only EV-related questions

Author: Bharat EV Saathi Project
Date: November 2025
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EVChatbot:
    """Professional EV Chatbot with strict topic filtering"""
    
    def __init__(self, api_key=None):
        """
        Initialize the EV Chatbot
        
        Args:
            api_key: Google Gemini API key (optional, reads from env if not provided)
        """
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            print("⚠️  Warning: No Gemini API key found!")
            print("   Set GEMINI_API_KEY in .env file or pass as parameter")
            self.model = None
            return
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ EV Chatbot initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing chatbot: {e}")
            self.model = None
        
        # System prompt for strict EV focus
        self.system_prompt = """You are an expert AI assistant specializing in Electric Vehicles (EVs) in India. 

YOUR ROLE:
- Answer ONLY questions related to electric vehicles, EVs, charging, batteries, subsidies, and related topics
- Provide accurate, helpful, and professional responses
- Support both English and Hindi queries
- Be concise but informative

STRICT RULES:
1. If question is NOT about EVs/electric vehicles - politely decline and redirect to EV topics
2. Do NOT answer questions about: politics, entertainment, sports, cooking, travel (unless EV-related), general topics
3. ALWAYS stay professional and helpful
4. Focus on Indian EV market when relevant

EV-RELATED TOPICS YOU CAN DISCUSS:
✅ EV models and specifications (cars, bikes, scooters, buses)
✅ FAME-II and state subsidies
✅ Charging infrastructure and stations
✅ Battery technology and range
✅ EV vs Petrol/Diesel comparison
✅ Total Cost of Ownership (TCO)
✅ Indian EV policies and regulations
✅ EV manufacturers and brands
✅ Maintenance and servicing
✅ Environmental benefits
✅ Future of EVs in India

NON-EV TOPICS - POLITELY DECLINE:
❌ General knowledge questions
❌ Other domains (unless related to EVs)

EXAMPLE RESPONSES:
Question: "What is the capital of France?"
Response: "I'm an EV specialist chatbot. I can only answer questions about electric vehicles. Would you like to know about EV adoption in Europe or charging infrastructure instead?"

Question: "Tell me about Ather 450X"
Response: "The Ather 450X is an excellent electric scooter! [provide detailed EV information]"

Remember: Always be polite, professional, and helpful within your EV expertise domain."""
        
        self.conversation_history = []
    
    def is_ev_related(self, question):
        """
        Quick check if question is EV-related (basic filtering)
        
        Args:
            question: User's question
            
        Returns:
            bool: True if likely EV-related
        """
        ev_keywords = [
            'ev', 'electric vehicle', 'electric car', 'electric bike', 'electric scooter',
            'battery', 'charging', 'range', 'subsidy', 'fame',
            'tesla', 'tata nexon', 'ather', 'ola', 'mahindra', 'mg',
            'बिजली', 'इलेक्ट्रिक', 'गाड़ी', 'सब्सिडी', 'चार्जिंग',
            'kwh', 'range anxiety', 'charger', 'tco', 'emission'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in ev_keywords)
    
    def chat(self, user_message):
        """
        Send message to chatbot and get response
        
        Args:
            user_message: User's question/message
            
        Returns:
            str: Chatbot's response
        """
        if not self.model:
            return "❌ Chatbot not initialized. Please check your API key."
        
        try:
            # Create full prompt with system instructions
            full_prompt = f"{self.system_prompt}\n\nUser Question: {user_message}\n\nYour Response:"
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Extract text
            bot_response = response.text
            
            # Store in history
            self.conversation_history.append({
                'user': user_message,
                'assistant': bot_response
            })
            
            return bot_response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            return f"Sorry, I encountered an error. Please try again.\n\nError: {error_msg}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")
    
    def get_history(self):
        """Get conversation history"""
        return self.conversation_history


def main():
    """Main function to run the chatbot in terminal"""
    print("\n" + "="*60)
    print("🤖 EV EXPERT CHATBOT - Bharat EV Saathi")
    print("="*60)
    print("Powered by Google Gemini Pro")
    print("I answer questions about Electric Vehicles in India")
    print("\nCommands:")
    print("  • Type your question and press Enter")
    print("  • Type 'quit' or 'exit' to end")
    print("  • Type 'clear' to clear conversation history")
    print("="*60 + "\n")
    
    # Initialize chatbot
    chatbot = EVChatbot()
    
    if not chatbot.model:
        print("\n❌ Failed to initialize chatbot. Exiting...")
        return
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thank you for using EV Expert Chatbot! Drive electric! ⚡")
                break
            
            # Check for clear command
            if user_input.lower() == 'clear':
                chatbot.clear_history()
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Get response
            print("\n🤖 Assistant: ", end="")
            response = chatbot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chatbot stopped. Drive electric! ⚡")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
