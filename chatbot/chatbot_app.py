"""
Streamlit Web Interface for EV Chatbot
=======================================
Professional web-based chatbot interface
Powered by Google Gemini Pro

Run: streamlit run chatbot_app.py
"""

import streamlit as st
from ev_chatbot import EVChatbot
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="EV Expert Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better visibility
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
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #000000;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #000000;
    }
    .info-box {
        background-color: #e3f2fd;
        border: 2px solid #1e88e5;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #000000;
    }
    /* Ensure all text is black and visible */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #000000 !important;
    }
    .stChatMessage {
        color: #000000 !important;
    }
    .stChatMessage p, .stChatMessage span, .stChatMessage div {
        color: #000000 !important;
    }
    /* Sidebar text */
    .css-1d391kg, .css-1d391kg p, .css-1d391kg li {
        color: #000000 !important;
    }
    /* Buttons */
    .stButton>button {
        background-color: #1e88e5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    /* Example questions */
    .example-btn {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
        color: #000000;
        font-weight: 500;
    }
    .example-btn:hover {
        background-color: #e3f2fd;
        border-color: #1e88e5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 EV Expert Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask me anything about Electric Vehicles in India!</div>', unsafe_allow_html=True)

# Check API key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ API Key Not Configured</h3>
        <p><strong>To use the chatbot:</strong></p>
        <ol>
            <li>Get a free API key from: <a href="https://ai.google.dev/" target="_blank">https://ai.google.dev/</a></li>
            <li>Create a <code>.env</code> file in this directory</li>
            <li>Add: <code>GEMINI_API_KEY=your_api_key_here</code></li>
            <li>Restart the app</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
else:
    st.markdown("""
    <div class="success-box">
        <h4>✅ AI Chatbot Active</h4>
        <p>Powered by Google Gemini Pro - Ready to answer your EV questions!</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize chatbot in session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = EVChatbot(api_key=api_key)
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.markdown("""
    **I can help with:**
    - ✅ EV models & specs
    - ✅ FAME-II subsidies
    - ✅ Charging stations
    - ✅ Battery technology
    - ✅ EV vs Petrol comparison
    - ✅ Indian EV policies
    
    **I only answer EV questions!**
    """)
    
    st.markdown("---")
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    
    examples = [
        "Where are charging stations in Mumbai?",
        "Is it worth buying an EV?",
        "Best electric scooter under 1 lakh?",
        "What is FAME-II subsidy?",
        "EV maintenance costs?",
        "How long does EV battery last?",
        "Ather 450X vs Ola S1 Pro?"
    ]
    
    for example in examples:
        if st.button(f"💬 {example}", key=example, use_container_width=True):
            # Add to chat
            st.session_state.messages.append({"role": "user", "content": example})
            response = st.session_state.chatbot.chat(example)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    st.markdown("---")
    
    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chatbot.clear_history()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Bharat EV Saathi**  
    Your AI companion for Electric Vehicles
    
    Powered by:  
    🤖 Google Gemini Pro  
    ⚡ Streamlit
    """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Electric Vehicles..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.chat(prompt)
            st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚗⚡ Bharat EV Saathi - Drive Electric, Drive Smart!</p>
    <p style="font-size: 0.9rem;">Powered by Google Gemini Pro | Made with ❤️ for India's EV Revolution</p>
</div>
""", unsafe_allow_html=True)
