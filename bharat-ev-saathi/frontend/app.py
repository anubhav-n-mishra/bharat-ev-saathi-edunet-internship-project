"""
Bharat EV Saathi - Main Streamlit Application
==============================================
India's Smart EV Companion Platform

Author: Bharat EV Saathi Project
Date: November 2025
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import STREAMLIT_CONFIG, APP_NAME, APP_VERSION
from backend.data_loader import ev_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(**STREAMLIT_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #00C853;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #333;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #00C853;
        margin: 1rem 0;
        background-color: #f8f9fa;
        color: #000000;
    }
    .stat-card {
        padding: 1rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
    }
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        color: white;
    }
    /* Ensure all text is visible */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {
        color: #000000 !important;
    }
    /* Fix sidebar text */
    .css-1d391kg, .css-1d391kg p {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown(f'<h1 class="main-header">🚗⚡ {APP_NAME}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">भारत EV साथी - Your Intelligent EV Companion</p>', unsafe_allow_html=True)
    
    # Welcome message
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🤖 **AI-Powered Recommendations**\nGet personalized EV suggestions")
    
    with col2:
        st.success("💰 **FAME Subsidy Calculator**\nKnow your savings instantly")
    
    with col3:
        st.warning("💬 **Smart Chatbot**\nAsk anything about EVs")
    
    st.write("---")
    
    # Statistics Dashboard
    st.header("📊 India's EV Ecosystem at a Glance")
    
    try:
        stats = ev_data.get_dataset_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['total_evs']}</div>
                <div class="stat-label">EV Models</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['total_charging_stations']}</div>
                <div class="stat-label">Charging Stations</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['cities_with_stations']}</div>
                <div class="stat-label">Cities Covered</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['states_with_subsidies']}</div>
                <div class="stat-label">States with Subsidies</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("---")
        
        # EV Distribution Chart
        st.subheader("🚗 EV Models by Type")
        
        ev_df = ev_data.load_ev_vehicles()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            type_counts = ev_df['type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Distribution by Vehicle Type",
                color_discrete_sequence=px.colors.sequential.Greens
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price vs Range scatter
            fig = px.scatter(
                ev_df,
                x='price_inr',
                y='range_km',
                size='battery_kwh',
                color='type',
                hover_data=['brand', 'model'],
                title="Price vs Range (Size = Battery Capacity)",
                labels={'price_inr': 'Price (₹)', 'range_km': 'Range (km)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top EVs by Range
        st.subheader("🏆 Top 5 EVs by Range")
        top_evs = ev_df.nlargest(5, 'range_km')[['brand', 'model', 'type', 'price_inr', 'range_km', 'battery_kwh']]
        top_evs['price_inr'] = top_evs['price_inr'].apply(lambda x: f"₹{x:,}")
        top_evs.columns = ['Brand', 'Model', 'Type', 'Price', 'Range (km)', 'Battery (kWh)']
        st.dataframe(top_evs, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("👉 Please generate the datasets first by running the data generation scripts.")
    
    # Features Section
    st.write("---")
    st.header("✨ What Can Bharat EV Saathi Do For You?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Smart EV Recommendation</h3>
            <p>Tell us your budget, daily usage, and preferences. Our AI will suggest the perfect EV for you from 25+ models!</p>
            <ul>
                <li>Budget-based filtering</li>
                <li>Range calculation based on usage</li>
                <li>Personalized scoring system</li>
                <li>Compare top 3 matches</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>💰 FAME & State Subsidy Calculator</h3>
            <p>Know exactly how much you can save! Calculate central FAME-II and state subsidies.</p>
            <ul>
                <li>10+ states covered</li>
                <li>Scrapping bonus calculation</li>
                <li>Tax benefit information</li>
                <li>State-wise comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 AI Chatbot (Gemini Powered)</h3>
            <p>Ask anything about EVs in India! Our chatbot knows it all.</p>
            <ul>
                <li>Bilingual support (English & Hindi)</li>
                <li>EV comparisons & specs</li>
                <li>Charging infrastructure info</li>
                <li>Policy & subsidy queries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🗺️ Charging Station Finder</h3>
            <p>Locate 500+ charging stations across 15 Indian cities!</p>
            <ul>
                <li>City-wise search</li>
                <li>Network information</li>
                <li>Charger types & power</li>
                <li>Operating hours & costs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.write("---")
    st.header("🚀 Quick Start Guide")
    
    st.markdown("""
    ### Phase 1 (Current - 30% Complete):
    
    1. **🤖 Get EV Recommendations** - Use the sidebar to navigate to "Recommendation" page
       - Enter your budget and daily driving distance
       - Select vehicle type
       - Get top 3 personalized suggestions!
    
    2. **💰 Calculate Subsidies** - Navigate to "Subsidy Calculator"
       - Select your state and vehicle
       - See total savings with breakdown
       - Compare across states
    
    3. **💬 Chat with AI** - Ask our chatbot
       - Type queries in English or Hindi
       - Get instant, accurate answers
       - Learn about EVs interactively
    
    4. **🗺️ Find Charging Stations** - Locate nearby stations
       - Search by city
       - View station details
       - Plan your charging stops
    
    ### Coming in Phase 2 & 3:
    - 📊 Market analytics & trends
    - 🧮 Total Cost of Ownership calculator
    - 🗺️ Route optimizer with charging stops
    - 🎓 Interactive EV learning modules
    - 🔋 Battery health predictor
    """)
    
    # API Setup Info
    st.write("---")
    st.header("⚙️ Setup Instructions")
    
    with st.expander("🔑 Configure Gemini API (Required for Chatbot)"):
        st.markdown("""
        ### Get Your Free Gemini API Key:
        
        1. Visit [Google AI Studio](https://ai.google.dev/)
        2. Click "Get API Key"
        3. Sign in with Google account
        4. Create new API key
        5. Copy the key
        
        ### Add to Application:
        
        1. Open `.env` file in project root
        2. Add: `GEMINI_API_KEY=your_api_key_here`
        3. Save and restart the app
        
        **Free Tier Limits:**
        - ✅ 15 requests per minute
        - ✅ 1,500 requests per day
        - ✅ Perfect for this project!
        """)
    
    with st.expander("📊 Generate Datasets (First Time Setup)"):
        st.markdown("""
        ### Generate All Required Datasets:
        
        ```powershell
        cd data/raw
        python generate_charging_stations.py
        python generate_indian_ev_data.py
        python generate_subsidy_data.py
        Move-Item *.csv ../processed/
        ```
        
        This will create:
        - ✅ Indian EV vehicles data (25+ models)
        - ✅ Charging stations (500+ locations)
        - ✅ FAME & state subsidies
        - ✅ Sales data for analytics
        """)
    
    # Footer
    st.write("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>{APP_NAME}</strong> v{APP_VERSION}</p>
        <p>Made with ❤️ for India's EV Revolution 🇮🇳</p>
        <p>Internship Project | Skills4Future Program | AICTE & Shell</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
