"""
EV Chatbot powered by Google Gemini
====================================
Intelligent conversational AI assistant for answering EV-related queries.
Supports bilingual conversations (English & Hindi).

Author: Bharat EV Saathi Project
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from backend.data_loader import ev_data

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVChatbot:
    """
    AI-powered chatbot for EV queries using Google Gemini
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the chatbot with Gemini API
        
        Args:
            api_key: Google Gemini API key (optional, can use env variable)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            logger.warning("No Gemini API key found. Chatbot will run in demo mode.")
            self.demo_mode = True
        else:
            self.demo_mode = False
            genai.configure(api_key=self.api_key)
            
            # Initialize the model (using latest stable Gemini 2.5 Flash)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Gemini chatbot initialized successfully with gemini-2.5-flash")
        
        # Load EV data for context
        self.data_loader = ev_data
        self.conversation_history = []
        
        # System prompt for the chatbot
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self):
        """
        Create a comprehensive system prompt with EV context
        
        Returns:
            String with system instructions
        """
        # Get data statistics for context
        try:
            stats = self.data_loader.get_dataset_stats()
            ev_df = self.data_loader.load_ev_vehicles()
            
            # Sample EV models
            popular_evs = ev_df.nlargest(5, 'range_km')[['brand', 'model', 'price_inr', 'range_km']].to_dict('records')
            
            ev_list = "\n".join([
                f"- {ev['brand']} {ev['model']}: ₹{ev['price_inr']:,}, Range: {ev['range_km']} km"
                for ev in popular_evs
            ])
        except:
            ev_list = ""
            stats = {}
        
        prompt = f"""You are a highly professional EV (Electric Vehicle) expert consultant for the Indian market, 
specifically working for the "Bharat EV Saathi" platform. You are India's most trusted EV advisor.

**CRITICAL INSTRUCTIONS - STRICTLY FOLLOW:**
1. ONLY answer questions related to Electric Vehicles, EVs, and related topics
2. Topics you CAN discuss:
   - Electric vehicles (2-wheelers, 3-wheelers, 4-wheelers)
   - EV specifications, range, battery, charging
   - FAME-II and state subsidy schemes
   - Charging infrastructure and stations
   - EV vs Petrol/Diesel comparisons
   - Total cost of ownership (TCO)
   - Indian EV policies and regulations
   - EV maintenance and service
   - Battery technology and lifespan
   - Environmental benefits of EVs
   - EV manufacturers and models in India

3. If asked about NON-EV topics (politics, movies, sports, general knowledge, etc.):
   Respond EXACTLY: "I am an EV specialist and can only assist with Electric Vehicle related queries. Please ask me about EVs, charging stations, subsidies, or EV comparisons. How can I help you with EVs today?"

**Your Knowledge Base:**
- Electric vehicles available in India: {stats.get('total_evs', 60)}+ models
- Charging infrastructure: {stats.get('total_charging_stations', 500)}+ stations across {stats.get('cities_with_stations', 15)} cities
- FAME-II subsidy coverage: 2-wheelers, 3-wheelers, and eligible 4-wheelers
- State subsidies: Delhi, Maharashtra, Gujarat, Karnataka, Tamil Nadu, Telangana, and more

**Top EV Models in India:**
{ev_list}

**Your Professional Communication Style:**
- Authoritative yet approachable
- Data-driven responses with specific numbers
- Use ₹ for pricing (Indian Rupees)
- Provide accurate, verified information only
- Be concise but comprehensive
- Support both English and Hindi (respond in user's language)
- Always stay on topic - ELECTRIC VEHICLES ONLY
- Always mention prices in Indian Rupees (₹)
- Compare with petrol/diesel vehicles when relevant

**Important Guidelines:**
1. If asked about specific EV models, provide accurate information about Indian market
2. When discussing subsidies, mention both FAME-II (central) and state subsidies
3. For charging queries, reference our 500+ station database across major cities
4. If you don't know something, be honest and suggest where to find the information
5. Encourage EV adoption but be realistic about limitations (range, charging time)
6. Always consider Indian context: traffic, climate, infrastructure

**Example Responses:**
- For "Best EV under 15 lakhs?": Suggest Tata Nexon EV Max, MG ZS EV with specific reasons
- For "Charging stations in Mumbai?": Mention 45+ stations, major networks like Tata Power
- For "Subsidy in Delhi?": Explain FAME-II + Delhi's ₹1.5L incentive
- For Hindi query "EV kharidne ka sahi samay?": Respond in Hindi with 2025 benefits

Be enthusiastic about EVs while being practical and helpful!
"""
        return prompt
    
    def chat(self, user_message, language='auto'):
        """
        Process user message and generate response
        
        Args:
            user_message: User's query/message
            language: 'auto', 'english', or 'hindi'
            
        Returns:
            Chatbot response string
        """
        if self.demo_mode:
            return self._demo_response(user_message)
        
        try:
            # Prepare the full prompt with context
            full_prompt = f"{self.system_prompt}\n\nUser Query: {user_message}\n\nAssistant:"
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Store in conversation history
            self.conversation_history.append({
                'user': user_message,
                'assistant': response.text
            })
            
            return response.text
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(user_message)
    
    def chat_with_context(self, user_message):
        """
        Chat with conversation history for context
        
        Args:
            user_message: User's query
            
        Returns:
            Chatbot response string
        """
        if self.demo_mode:
            return self._demo_response(user_message)
        
        try:
            # Build conversation context
            conversation_text = self.system_prompt + "\n\n"
            
            # Add conversation history
            for turn in self.conversation_history[-5:]:  # Last 5 turns
                conversation_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
            
            # Add current message
            conversation_text += f"User: {user_message}\nAssistant:"
            
            # Generate response
            response = self.model.generate_content(conversation_text)
            
            # Store in history
            self.conversation_history.append({
                'user': user_message,
                'assistant': response.text
            })
            
            return response.text
        
        except Exception as e:
            logger.error(f"Error in contextual chat: {e}")
            return self._fallback_response(user_message)
    
    def _demo_response(self, user_message):
        """
        Provide demo responses when API is not available
        
        Args:
            user_message: User query
            
        Returns:
            Pre-configured demo response
        """
        message_lower = user_message.lower()
        
        # Demo responses for common queries
        if any(word in message_lower for word in ['tata', 'nexon']):
            return """**Tata Nexon EV** is one of India's most popular electric SUVs! 

**Key Specs:**
- Price: ₹14.99 - ₹17.99 Lakhs (ex-showroom)
- Range: 325-437 km (depending on variant)
- Battery: 30.2 kWh (standard) / 40.5 kWh (Max)
- Charging: 0-80% in 56 minutes (fast charging)

**Why it's popular:**
✅ Affordable price point
✅ Good range for city + highway
✅ 5-star safety rating
✅ Established service network

**Available Subsidies:** Up to ₹1.5 lakh in states like Delhi, Maharashtra!"""
        
        elif any(word in message_lower for word in ['subsidy', 'fame']):
            return """**EV Subsidies in India (2025):**

**FAME-II (Central Government):**
- 2-Wheelers: Up to ₹15,000
- 3-Wheelers: Up to ₹50,000
- 4-Wheelers: Up to ₹1.5 lakh (mainly for commercial)

**Top State Subsidies:**
1. **Delhi**: Up to ₹1.5L + road tax waiver + registration waiver
2. **Maharashtra**: Up to ₹1L + tax exemptions
3. **Gujarat**: Up to ₹1.5L + 100% road tax exemption

**How to claim:** Most subsidies are applied at the time of purchase. Some require post-purchase application."""
        
        elif any(word in message_lower for word in ['charging', 'station']):
            return """**EV Charging in India:**

**Our Database:** 500+ stations across 15 major cities!

**Top Networks:**
- Tata Power EZ Charge
- Ather Grid
- Fortum Charge & Drive
- ChargeZone
- Statiq

**Charging Types:**
- **AC Charging**: 3-8 hours (home/office)
- **DC Fast Charging**: 30-60 minutes (0-80%)

**Cost:** ₹8-15 per kWh (much cheaper than petrol!)

Use our Charging Station Finder to locate stations in your city!"""
        
        elif 'best' in message_lower or 'recommend' in message_lower:
            return """**Top EVs in India by Category (2025):**

**Budget 2-Wheelers (₹80K-₹1.2L):**
- Ola S1 Air
- TVS iQube
- Ather 450S

**Premium 2-Wheelers (₹1.3L-₹1.5L):**
- Ola S1 Pro
- Ather 450X

**Affordable 4-Wheelers (₹8L-₹13L):**
- Tata Tiago EV
- Citroen eC3
- MG Comet EV

**Mid-Range SUVs (₹15L-₹25L):**
- Tata Nexon EV Max
- MG ZS EV
- Mahindra XUV400 EV

Use our **Recommendation Tool** for personalized suggestions based on your budget and usage!"""
        
        else:
            return """I'm an EV assistant for Bharat EV Saathi! I can help you with:

🚗 **EV Recommendations** - Find the best EV for your needs
💰 **Subsidy Information** - FAME-II and state subsidies
🔌 **Charging Stations** - Locations across India
📊 **Comparisons** - Compare different EV models
💵 **Cost Analysis** - EV vs Petrol/Diesel savings

**Note:** I'm currently in demo mode. Please add your Gemini API key to enable full AI capabilities.

What would you like to know about EVs in India?"""
    
    def _fallback_response(self, user_message):
        """
        Fallback response when API fails
        
        Args:
            user_message: User query
            
        Returns:
            Error message
        """
        return """I apologize, but I'm having trouble processing your request right now. 

This could be due to:
- API connection issues
- Rate limiting
- Invalid API key

**In the meantime, you can:**
- Use our EV Recommendation tool
- Check the Subsidy Calculator
- Browse the Charging Station Finder

Please try again in a moment, or contact support if the issue persists."""
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history


# Create global instance
chatbot = EVChatbot()


if __name__ == '__main__':
    # Test the chatbot
    print("🤖 Testing EV Chatbot...")
    
    bot = EVChatbot()
    
    # Test queries
    test_queries = [
        "What are the best EVs under 15 lakhs in India?",
        "Tell me about Tata Nexon EV",
        "What subsidies are available in Maharashtra?",
        "Where can I find charging stations in Mumbai?",
    ]
    
    print("\n" + "="*60)
    for query in test_queries:
        print(f"\n👤 User: {query}")
        response = bot.chat(query)
        print(f"🤖 Bot: {response}")
        print("="*60)
    
    print("\n✅ Chatbot test completed!")
