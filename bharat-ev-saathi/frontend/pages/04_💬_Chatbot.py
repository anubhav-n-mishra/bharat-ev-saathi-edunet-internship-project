"""
EV Chatbot Page - Bharat EV Saathi
===================================
Professional AI-powered chatbot for EV queries
Powered by Google Gemini Pro
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.chatbot import chatbot
from utils.config import is_api_configured

# Page config
st.set_page_config(
    page_title="EV Chatbot - Bharat EV Saathi",
    page_icon="💬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s;
        color: #000000;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1e88e5;
        color: #000000;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
        color: #000000;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border: 1px solid #1e88e5;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%;
        background-color: #1e88e5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .example-question {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
        color: #000000;
    }
    .example-question:hover {
        background-color: #e9ecef;
        border-color: #1e88e5;
        color: #000000;
    }
    /* Fix Streamlit default text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #000000 !important;
    }
    /* Fix chat input text */
    .stChatInput textarea {
        color: #000000 !important;
    }
    /* Fix chat messages */
    .stChatMessage {
        color: #000000 !important;
    }
    .stChatMessage p {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 EV Expert Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask me anything about Electric Vehicles in India!</div>', unsafe_allow_html=True)

# Check API configuration
if not is_api_configured():
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ API Key Not Configured</h4>
        <p>The chatbot is running in <strong>demo mode</strong> with limited responses.</p>
        <p>To unlock full AI capabilities, add your Gemini API key to the <code>.env</code> file.</p>
        <p>Get a free API key from: <a href="https://ai.google.dev/" target="_blank">https://ai.google.dev/</a></p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="success-box">
        <h4>✅ AI Chatbot Active</h4>
        <p>Powered by Google Gemini Pro - Ask me anything about EVs!</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar with instructions
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Type your question** in the chat box below
    2. **Click example questions** for quick queries
    3. **Ask in English or Hindi** - I understand both!
    4. I **only answer EV-related questions**
    
    ---
    
    ### ✅ What I Can Help With:
    - EV models and specifications
    - FAME-II & state subsidies
    - Charging stations location
    - EV vs Petrol comparisons
    - Battery technology
    - Total cost of ownership
    - Indian EV policies
    
    ### ❌ What I Can't Help With:
    - Non-EV topics (politics, movies, etc.)
    - Personal advice unrelated to EVs
    - General knowledge questions
    
    ---
    
    ### 🌟 Pro Tips:
    - Be specific in your questions
    - Mention your budget for better recommendations
    - Ask about subsidies in your state
    - Inquire about charging infrastructure
    """)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        chatbot.conversation_history = []
        st.rerun()

# Main chat area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 EV Expert:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me about EVs... (e.g., 'Best EV under 15 lakhs?')")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get chatbot response
        with st.spinner("🤔 Thinking..."):
            response = chatbot.chat_with_context(user_input)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })
        
        # Rerun to display new messages
        st.rerun()

with col2:
    st.markdown("### 💡 Example Questions")
    
    # Example questions
    examples = [
        "What is the best EV under ₹15 lakhs?",
        "How much subsidy can I get in Delhi?",
        "Tata Nexon EV vs MG ZS EV comparison",
        "Where are charging stations in Mumbai?",
        "Is it worth buying an EV in 2025?",
        "Best electric scooter under ₹1.5 lakhs?",
        "What is FAME-II subsidy scheme?",
        "EV maintenance cost vs petrol car",
        "How long does EV battery last?",
        "Ather 450X vs Ola S1 Pro comparison"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            # Add to chat as if user typed it
            st.session_state.chat_history.append({
                'role': 'user',
                'content': example
            })
            
            # Get response
            with st.spinner("🤔 Thinking..."):
                response = chatbot.chat_with_context(example)
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>🚗⚡ Bharat EV Saathi</strong> - Your Trusted EV Companion</p>
    <p>Powered by Google Gemini Pro | Data covers 60+ EV models & 500+ charging stations</p>
    <p style="font-size: 0.9rem;">
        <em>Note: This chatbot provides information based on available data. 
        Always verify with official sources and dealers before making purchase decisions.</em>
    </p>
</div>
""", unsafe_allow_html=True)

# Display some stats
if st.session_state.chat_history:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Chat Stats:**")
    st.sidebar.markdown(f"- Total messages: {len(st.session_state.chat_history)}")
    st.sidebar.markdown(f"- Questions asked: {len([m for m in st.session_state.chat_history if m['role'] == 'user'])}")
