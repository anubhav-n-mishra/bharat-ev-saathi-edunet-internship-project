# 🚀 Quick Start Guide

## Get Your Project Running in 5 Minutes!

### Prerequisites
- ✅ Windows 10/11
- ✅ Python 3.9 or higher
- ✅ PowerShell (built-in)
- ✅ Internet connection

---

## 🎯 Option 1: Automated Setup (Recommended)

### Just run this single command:

```powershell
.\setup_and_run.ps1
```

**That's it!** The script will:
1. ✅ Create virtual environment
2. ✅ Install all dependencies
3. ✅ Generate datasets
4. ✅ Launch the application

**Time: ~3-5 minutes**

---

## 🛠️ Option 2: Manual Setup (Step by Step)

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 2: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Download & Generate Datasets

**Option A: Use Real Kaggle Data (Recommended)**
```powershell
# Install Kaggle hub
pip install kagglehub

# Download real datasets + generate supplementary data
python data\raw\download_kaggle_datasets.py
```

**Option B: Generate Only (No Kaggle)**
```powershell
cd data\raw
python generate_charging_stations.py
python generate_indian_ev_data.py
python generate_subsidy_data.py
python process_fame_bus_data.py
Move-Item *.csv ..\processed\
cd ..\..
```

### Step 5: Configure API Keys (Optional but Recommended)
```powershell
# Copy the template
Copy-Item .env.example .env

# Edit .env file and add your Gemini API key
# Get free key from: https://ai.google.dev/
notepad .env
```

Add this line:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### Step 6: Run the Application
```powershell
streamlit run frontend\app.py
```

**The app will open automatically in your browser at `http://localhost:8501`**

---

## 🔑 Getting Gemini API Key (2 Minutes)

### Free & Easy Setup:

1. **Visit**: [https://ai.google.dev/](https://ai.google.dev/)
2. **Click**: "Get API Key"
3. **Sign in** with Google account
4. **Create** new API key
5. **Copy** the key (starts with `AIzaSy...`)
6. **Paste** in `.env` file

**Free Tier**: 1,500 requests/day - Perfect for this project! ✅

---

## 📊 What Datasets Are Generated?

### Automatically Created:

1. **indian_ev_vehicles.csv** (25+ models)
   - Tata Nexon EV, Ola S1 Pro, Ather 450X, MG ZS EV, etc.
   - Specs: Price, Range, Battery, Efficiency

2. **india_ev_charging_stations.csv** (500+ stations)
   - 15 major cities: Mumbai, Delhi, Bangalore, Pune, etc.
   - Networks: Tata Power, Ather Grid, Fortum, ChargeZone

3. **state_ev_subsidies.csv** (10+ states)
   - FAME-II + state policies
   - Delhi, Maharashtra, Gujarat, Karnataka, Tamil Nadu

4. **indian_ev_sales.csv** (20K+ records)
   - Monthly sales 2023-2024
   - State-wise breakdown

**All data is realistic and based on actual Indian market!**

---

## 🎨 Features You'll See

### 1. Home Page
- 📊 Statistics dashboard
- 📈 Interactive charts
- 🏆 Top EVs by range

### 2. Recommendation Engine
- 🎯 Enter budget & requirements
- 🤖 Get AI-powered suggestions
- 💡 Detailed reasoning for each recommendation

### 3. Subsidy Calculator
- 💰 Calculate FAME + state subsidies
- 🔍 Compare across states
- 💵 See effective price after benefits

### 4. Chatbot
- 💬 Ask questions in English or Hindi
- 🧠 Powered by Google Gemini AI
- ⚡ Instant, accurate answers

### 5. Charging Station Finder
- 🗺️ Search by city
- 🔌 Filter by network/type
- 📍 View station details

---

## 🐛 Troubleshooting

### Issue: "Execution of scripts is disabled"

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Python not found"

**Solution:**
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Check "Add Python to PATH" during installation
3. Restart PowerShell

### Issue: "Module 'streamlit' not found"

**Solution:**
```powershell
pip install streamlit
```

### Issue: "Datasets not generated"

**Solution:**
```powershell
cd data\raw
python generate_charging_stations.py
python generate_indian_ev_data.py
python generate_subsidy_data.py
Move-Item *.csv ..\processed\
```

### Issue: "Chatbot not working"

**Cause:** Gemini API key not configured

**Solution:**
1. The chatbot works in "demo mode" without API key
2. For full AI features, add Gemini API key to `.env`
3. Get free key from [ai.google.dev](https://ai.google.dev/)

---

## 📁 Project Structure

```
bharat-ev-saathi/
├── data/
│   ├── raw/              # Data generation scripts
│   └── processed/        # Generated CSV files
├── models/               # ML recommendation engine
├── backend/              # Business logic
│   ├── data_loader.py    # Data management
│   ├── subsidy_calculator.py
│   ├── chatbot.py        # Gemini AI integration
│   └── ev_recommender.py # ML model
├── frontend/
│   └── app.py            # Main Streamlit app
├── docs/                 # Documentation
├── utils/                # Configuration
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
└── README.md             # Project overview
```

---

## 🎓 Learning Resources

### Understand the Code:

1. **Data Generation** (`data/raw/*.py`)
   - How realistic datasets are created
   - Indian EV market structure

2. **ML Model** (`models/ev_recommender.py`)
   - Random Forest classifier
   - Feature engineering
   - Scoring system

3. **Subsidy Calculator** (`backend/subsidy_calculator.py`)
   - FAME-II rules implementation
   - State policy integration

4. **Chatbot** (`backend/chatbot.py`)
   - Gemini API integration
   - Context management
   - Bilingual support

5. **UI** (`frontend/app.py`)
   - Streamlit components
   - Interactive visualizations
   - User experience design

**Every file has detailed comments explaining the logic!**

---

## 📱 Using the App

### Recommendation Tool:
1. Navigate to "Recommendation" from sidebar
2. Enter your budget (e.g., ₹15,00,000)
3. Enter daily driving distance (e.g., 50 km)
4. Select vehicle type (2/3/4-Wheeler)
5. Click "Get Recommendations"
6. View top 3 personalized suggestions!

### Subsidy Calculator:
1. Go to "Subsidy Calculator"
2. Select your state
3. Choose an EV model
4. Check "Old vehicle" if scrapping
5. See total subsidy breakdown
6. Compare across states

### Chatbot:
1. Open "Chatbot" page
2. Type your question (English/Hindi)
3. Examples:
   - "Best EV under 15 lakhs?"
   - "Tata Nexon EV vs MG ZS EV comparison"
   - "Charging stations in Delhi?"
4. Get instant AI-powered answers

---

## 🚀 Next Steps

### Phase 1 Complete ✅
You now have:
- ✅ AI recommendation engine
- ✅ Subsidy calculator
- ✅ Gemini chatbot
- ✅ Charging station finder
- ✅ Interactive UI

### Phase 2 & 3 (Coming Soon):
- 📊 Advanced analytics
- 🧮 TCO calculator
- 🗺️ Route optimizer
- 🎓 Learning modules
- 🔋 Battery health predictor

---

## 💡 Tips for Best Experience

1. **Use Chrome/Edge** for best Streamlit performance
2. **Maximize browser window** to see all charts
3. **Enable JavaScript** for interactive features
4. **Add Gemini API key** for full chatbot capabilities
5. **Explore all pages** using sidebar navigation

---

## 🆘 Need Help?

### Documentation:
- 📖 **README.md** - Project overview
- 📋 **PROBLEM_STATEMENT.md** - Detailed problem analysis
- 🔑 **API_SETUP.md** - API configuration guide

### Code Questions:
- All Python files have extensive comments
- Each function has docstrings explaining purpose
- Check `__main__` sections for usage examples

### Issues:
- Check troubleshooting section above
- Review error messages carefully
- Verify all dependencies installed

---

## ✅ Quick Checklist

Before submitting/presenting:

- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Datasets generated (4 CSV files in `data/processed/`)
- [ ] `.env` file created
- [ ] Gemini API key added (optional but recommended)
- [ ] Application runs without errors
- [ ] All features tested
- [ ] README.md reviewed
- [ ] Documentation complete

---

## 🎉 You're All Set!

**Your award-winning EV project is ready!**

### What makes it special:
✨ Solves real Indian problem
✨ Uses actual market data
✨ AI/ML integration
✨ Professional code quality
✨ Comprehensive documentation
✨ Unique India-first approach

**Go revolutionize India's EV adoption! 🚗⚡🇮🇳**

---

**Questions? Check the docs folder or review the inline code comments!**
