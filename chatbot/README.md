# 🤖 EV Expert Chatbot

Professional AI-powered chatbot specialized in Electric Vehicles in India.

## ✨ Features

- 🎯 **EV-Focused**: Only answers questions about Electric Vehicles
- 🤖 **AI-Powered**: Uses Google Gemini Pro for intelligent responses
- 🌐 **Bilingual**: Understands English and Hindi
- 💬 **Professional**: Polite and helpful responses
- 🚫 **Smart Filtering**: Declines non-EV questions gracefully

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Your API key is already configured in `.env`:
```
GEMINI_API_KEY=AIzaSyBBXyNPfNypTIlKg2DVYvtAjq653m0o0FA
```

### 3. Run the Chatbot

**Option A: Web Interface (Recommended)**
```bash
streamlit run chatbot_app.py
```

**Option B: Terminal Interface**
```bash
python ev_chatbot.py
```

## 📁 Files

- `chatbot_app.py` - Streamlit web interface
- `ev_chatbot.py` - Core chatbot logic + terminal interface
- `.env` - API key configuration
- `requirements.txt` - Python dependencies

## 💡 What Can I Ask?

### ✅ EV-Related Questions:
- "What is the range of Tata Nexon EV?"
- "Where are charging stations in Delhi?"
- "What is FAME-II subsidy?"
- "Best electric scooter under 1 lakh?"
- "EV vs Petrol comparison"
- "How long does EV battery last?"
- "Ather 450X specifications"

### ❌ Non-EV Questions:
- General knowledge, politics, sports, entertainment, etc.
- The chatbot will politely decline and redirect to EV topics

## 🎨 Web Interface Features

- Clean, professional UI with black text
- Example question buttons for quick queries
- Chat history with user/assistant distinction
- Clear conversation option
- Responsive design
- Real-time responses

## 🔧 Technical Details

- **AI Model**: Google Gemini 1.5 Flash
- **Language**: Python 3.9+
- **Framework**: Streamlit
- **API**: Google Generative AI

## 📊 Usage

### Terminal Mode:
```python
from ev_chatbot import EVChatbot

chatbot = EVChatbot()
response = chatbot.chat("Tell me about Ola S1 Pro")
print(response)
```

### Web Mode:
Just run `streamlit run chatbot_app.py` and use the browser interface!

## 🛡️ Safety Features

- Strict topic filtering (EV-only)
- Professional tone enforcement
- Error handling
- API rate limit awareness
- Secure API key management

## 📝 Notes

- Free Gemini API: 1500 requests/day
- Responses typically take 1-3 seconds
- Works best with specific EV questions
- Supports both English and Hindi

## 🎯 Use Cases

1. **EV Buyers**: Get answers before purchasing
2. **EV Owners**: Maintenance and troubleshooting
3. **Students**: Learn about EV technology
4. **Research**: Indian EV market insights
5. **Comparisons**: Compare different EV models

## 🚗 Supported Topics

- EV Models (2W, 3W, 4W)
- FAME-II & State Subsidies
- Charging Infrastructure
- Battery Technology
- Total Cost of Ownership
- Indian EV Policies
- Manufacturers & Brands
- Environmental Benefits

---

**Made with ❤️ for India's EV Revolution** 🇮🇳⚡

**Version**: 1.0  
**Last Updated**: November 2025
