"""
Configuration Management
========================
Central configuration for Bharat EV Saathi application.

Author: Bharat EV Saathi Project
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Application settings
APP_NAME = "Bharat EV Saathi"
APP_VERSION = "1.0.0"
DEBUG = os.getenv('DEBUG', 'False') == 'True'

# UI Settings
PAGE_ICON = "🚗"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Theme colors
PRIMARY_COLOR = "#00C853"  # Green for EV
SECONDARY_COLOR = "#FF6F00"  # Orange/Saffron
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': APP_NAME,
    'page_icon': PAGE_ICON,
    'layout': LAYOUT,
    'initial_sidebar_state': SIDEBAR_STATE
}

# Model settings
ML_MODEL_NAME = 'ev_recommender.pkl'
USE_CACHE = True
CACHE_TTL = 3600  # seconds

# External APIs
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '')
MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')

# Feature flags
ENABLE_HINDI = True
ENABLE_VOICE = False  # Future feature
ENABLE_ANALYTICS = True

# Data validation
VALID_VEHICLE_TYPES = ['2-Wheeler', '3-Wheeler', '4-Wheeler']
MIN_BUDGET = 50000
MAX_BUDGET = 50000000

# Supported states
INDIAN_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    'Delhi', 'Chandigarh', 'Puducherry'
]

def get_config(key, default=None):
    """Get configuration value"""
    return globals().get(key, default)

def is_api_configured(api_name='gemini'):
    """Check if API key is configured"""
    if api_name.lower() == 'gemini':
        return bool(GEMINI_API_KEY)
    elif api_name.lower() == 'openai':
        return bool(OPENAI_API_KEY)
    return False
