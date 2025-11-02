# 🔑 API Setup Guide

## Google Gemini API (Recommended - FREE)

### Why Gemini?
- ✅ **Generous free tier**: 15 requests/minute, 1,500/day
- ✅ **Easy setup**: Get API key in 2 minutes
- ✅ **Multilingual**: Supports Hindi natively
- ✅ **Reliable**: Google's infrastructure
- ✅ **Perfect for this project**: Handles all our chatbot needs

### Step-by-Step Setup:

#### 1. Get Your API Key

1. **Visit Google AI Studio**
   - Go to: [https://ai.google.dev/](https://ai.google.dev/)
   - Or: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

2. **Sign In**
   - Use your Google account
   - Accept terms of service

3. **Create API Key**
   - Click "Get API Key" or "Create API Key"
   - Choose "Create API key in new project" (recommended)
   - Your key will be generated instantly

4. **Copy the Key**
   - It looks like: `AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`
   - **IMPORTANT**: Save it securely!

#### 2. Add to Project

**Option A: Using .env file (Recommended)**

1. Open `.env` file in project root
2. Add your API key:
   ```
   GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```
3. Save the file

**Option B: Using Streamlit Secrets (for deployment)**

1. Create `.streamlit/secrets.toml` file
2. Add:
   ```toml
   GEMINI_API_KEY = "AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
   ```

#### 3. Verify Setup

Run this test:

```powershell
cd backend
python chatbot.py
```

If successful, you'll see:
```
🤖 Testing EV Chatbot...
Gemini chatbot initialized successfully
```

---

## Alternative: Hugging Face (Free)

If you prefer open-source models:

### Setup:

1. **Get Token**
   - Visit: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name it: "Bharat-EV-Saathi"
   - Role: "Read"
   - Copy token

2. **Add to .env**
   ```
   HF_API_KEY=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```

3. **Modify Code** (Optional)
   - In `backend/chatbot.py`, you can add Hugging Face Inference API support
   - Use models like: `mistralai/Mistral-7B-Instruct-v0.1`

---

## Groq API (Fast & Free)

For ultra-fast inference:

### Setup:

1. **Get API Key**
   - Visit: [https://console.groq.com/](https://console.groq.com/)
   - Sign up (free)
   - Go to API Keys
   - Create new key

2. **Add to .env**
   ```
   GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```

---

## Free Tier Limits Comparison

| Provider | Requests/Min | Requests/Day | Best For |
|----------|--------------|--------------|----------|
| **Gemini** | 15 | 1,500 | ✅ **Recommended** - Best overall |
| Hugging Face | 10 | 1,000 | Open-source models |
| Groq | 30 | 14,400 | Fastest response |

---

## Testing Your Setup

### 1. Check API Key Configuration

```python
# Run this in Python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if api_key:
    print(f"✅ API Key found: {api_key[:10]}...")
else:
    print("❌ API Key not found")
```

### 2. Test Chatbot

```powershell
python backend/chatbot.py
```

Expected output:
```
🤖 Testing EV Chatbot...
Gemini chatbot initialized successfully

👤 User: What are the best EVs under 15 lakhs in India?
🤖 Bot: [AI Response here]
```

### 3. Test in Streamlit App

```powershell
streamlit run frontend/app.py
```

Navigate to Chatbot page and ask: "Tell me about Tata Nexon EV"

---

## Troubleshooting

### Issue: "No Gemini API key found"

**Solution:**
1. Check if `.env` file exists in project root
2. Verify API key is correct (no extra spaces)
3. Restart the application after adding key

### Issue: "API key is invalid"

**Solution:**
1. Regenerate API key in Google AI Studio
2. Make sure you're using Gemini API, not PaLM API
3. Check if API key is enabled for your project

### Issue: "Rate limit exceeded"

**Solution:**
1. Free tier: 15 requests/minute
2. Wait 60 seconds before retrying
3. Consider implementing rate limiting in code

### Issue: "Module 'google.generativeai' not found"

**Solution:**
```powershell
pip install google-generativeai
```

---

## Security Best Practices

### ✅ DO:
- Store API keys in `.env` file
- Add `.env` to `.gitignore`
- Use environment variables
- Rotate keys regularly

### ❌ DON'T:
- Commit API keys to Git
- Share keys publicly
- Hardcode keys in source code
- Use same key for multiple projects

---

## For Deployment (Future)

When deploying to cloud:

### Streamlit Cloud:
1. Go to app settings
2. Add secrets in "Secrets" section
3. Format:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```

### Heroku:
```bash
heroku config:set GEMINI_API_KEY=your_key_here
```

### AWS/GCP:
Use respective secret management services:
- AWS: Secrets Manager
- GCP: Secret Manager

---

## Cost Monitoring

### Gemini Free Tier:
- **Limit**: 1,500 requests/day
- **Project Usage**: ~50-100 requests/day (estimated)
- **Safety Margin**: You're safe with free tier! ✅

### Track Usage:
- Visit: [Google Cloud Console](https://console.cloud.google.com/)
- Go to: APIs & Services → Credentials
- Monitor quota usage

---

## Need Help?

### Official Documentation:
- Gemini API: [https://ai.google.dev/docs](https://ai.google.dev/docs)
- Python SDK: [https://ai.google.dev/tutorials/python_quickstart](https://ai.google.dev/tutorials/python_quickstart)

### Community:
- Stack Overflow: Tag `google-gemini`
- Reddit: r/GoogleCloud

---

## Summary Checklist

- [ ] Got Gemini API key from [ai.google.dev](https://ai.google.dev/)
- [ ] Added key to `.env` file
- [ ] Installed `google-generativeai` package
- [ ] Tested with `python backend/chatbot.py`
- [ ] Verified in Streamlit app
- [ ] `.env` file is in `.gitignore`

**All set? Let's revolutionize EV adoption in India! 🚗⚡🇮🇳**
