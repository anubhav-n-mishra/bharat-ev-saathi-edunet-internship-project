# 🚗 Bharat EV Saathi - भारत EV साथी

**Your Complete Electric Vehicle Assistant for India**

A comprehensive full-stack web application that helps users make informed decisions about electric vehicles in India through AI-powered recommendations, subsidy calculations, interactive maps, and market analytics.

![React](https://img.shields.io/badge/React-18-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue) ![Python](https://img.shields.io/badge/Python-3.x-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.121-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

### 🎯 **EV Recommender**
- ML-powered vehicle recommendations using ensemble model (72% accuracy)
- Filter by budget, daily usage, and vehicle type
- Intelligent scoring based on ML predictions, range, value, and efficiency
- Detailed vehicle specifications and comparisons

### 💰 **FAME-II Subsidy Calculator**
- Calculate FAME-II subsidies for all vehicle types
- State-level subsidy integration (22 states covered)
- Real-time savings calculation with percentage breakdown
- Comprehensive eligibility checking

### 📊 **Analytics Dashboard**
- Interactive sales trends visualization
- Top brands and states performance analysis
- Vehicle type distribution charts
- Monthly sales patterns with filters
- Key market insights

### 🗺️ **Charging Stations Map**
- Interactive Leaflet map with 458+ charging stations
- City-wise filtering and live search
- Station details with operator and charger types
- Dual view: Map + scrollable list

### 💬 **AI Chatbot**
- Google Gemini-powered conversational AI
- Context-aware responses about EVs
- Beautiful message formatting with markdown support
- Real-time typing indicators

---

## 🛠️ Tech Stack

### **Frontend**
- **React 18** with TypeScript
- **Vite 7.2** - Lightning-fast build tool
- **Tailwind CSS v4** - Modern utility-first CSS
- **React Router v6** - Client-side routing
- **Leaflet** - Interactive maps (no API key needed)
- **Lucide React** - Beautiful icons

### **Backend**
- **FastAPI 0.121** - Modern Python web framework
- **Uvicorn** - ASGI server
- **scikit-learn** - Machine learning models
- **Google Gemini API** - AI chatbot (gemini-2.5-flash)
- **pandas & numpy** - Data processing

### **Machine Learning**
- **EV Recommender**: Voting ensemble (RandomForest + GradientBoosting + RandomForest)
  - 72.06% cross-validation accuracy
  - 15 engineered features
  - Real-time predictions
  
- **Sales Predictor**: Ensemble model (RandomForest + GradientBoosting)
  - R² Score: 0.9968
  - MAE: 23.87 units
  - 6-month trend forecasting

---

## 📂 Project Structure

```
bharat-ev-saathi/
├── api_server.py              # FastAPI backend server
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
│
├── backend/                  # Backend logic
│   ├── chatbot.py           # Gemini AI chatbot
│   ├── data_loader.py       # Data processing
│   └── subsidy_calculator.py # Subsidy logic
│
├── models/                   # Machine learning models
│   ├── ev_recommender.py    # ML recommender class
│   ├── sales_predictor.py   # Sales prediction class
│   ├── fame_calculator.py   # FAME-II calculator
│   ├── train_final_model.py # Training script
│   └── saved/               # Trained model files
│       ├── ev_recommender_production.pkl
│       ├── sales_predictor.pkl
│       └── *.json
│
├── data/                     # Datasets
│   ├── processed/
│   │   ├── indian_ev_vehicles.csv (58 EVs)
│   │   ├── indian_ev_sales.csv (1,218 records)
│   │   ├── india_ev_charging_stations.csv (458 stations)
│   │   └── state_ev_subsidies.csv (22 states)
│   └── raw/                 # Data generation scripts
│
├── react-frontend/          # React application
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   └── Footer.tsx
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── EVRecommender.tsx
│   │   │   ├── SubsidyCalculator.tsx
│   │   │   ├── Analytics.tsx
│   │   │   ├── Chatbot.tsx
│   │   │   └── ChargingStations.tsx
│   │   └── services/
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── notebooks/               # Jupyter notebooks
│   └── 01_EV_Sales_Analysis.ipynb
│
└── docs/                    # Documentation
    ├── AI_ML_IMPLEMENTATION.md
    ├── API_SETUP.md
    ├── DATA_SOURCES.md
    └── PROBLEM_STATEMENT.md
```

---

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8+ (Anaconda recommended)
- Node.js 18+ and npm
- Git

### **1. Clone Repository**
```bash
git clone https://github.com/anubhav-n-mishra/bharat-ev-saathi-edunet-internship-project.git
cd bharat-ev-saathi-edunet-internship-project
```

### **2. Backend Setup**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env

# Add your Google Gemini API key to .env
# GEMINI_API_KEY=your_api_key_here
```

### **3. Frontend Setup**
```bash
cd react-frontend
npm install
```

---

## ▶️ Running the Application

### **Start Backend (Terminal 1)**
```bash
cd bharat-ev-saathi
python api_server.py
```
Backend runs on: **http://localhost:8000**  
API docs: **http://localhost:8000/docs**

### **Start Frontend (Terminal 2)**
```bash
cd react-frontend
npm run dev
```
Frontend runs on: **http://localhost:5173**

### **Access Application**
Open your browser and navigate to: **http://localhost:5173**

---

## 📡 API Endpoints

### **EV Recommendations**
- `POST /api/recommend` - Get ML-powered EV recommendations
  ```json
  {
    "budget": 150000,
    "daily_km": 50,
    "vehicle_type": "2-Wheeler"
  }
  ```

### **Subsidy Calculator**
- `POST /api/subsidy/calculate` - Calculate FAME-II + state subsidies
- `GET /api/subsidy/states` - List all states with subsidy programs

### **Sales Prediction**
- `POST /api/sales/predict` - Predict sales for a vehicle
- `POST /api/sales/trend` - Get 6-month sales forecast

### **Chatbot**
- `POST /api/chat` - Send message to AI chatbot

### **Charging Stations**
- `GET /api/charging-stations` - Get all charging stations
- `GET /api/charging-stations?city={city}` - Filter by city

Full API documentation: http://localhost:8000/docs

---

## 🤖 Machine Learning Models

### **EV Recommender (ev_recommender_production.pkl)**
- **Algorithm**: Voting Ensemble (3 models)
- **Accuracy**: 72.06% ± 1.8% (cross-validation)
- **Features**: 15 engineered features including price, range, efficiency, type
- **Training Data**: 58 EV models with specifications

### **Sales Predictor (sales_predictor.pkl)**
- **Algorithm**: RandomForest + GradientBoosting ensemble
- **Performance**: 
  - R² Score: 0.9968 (excellent fit)
  - MAE: 23.87 units
  - RMSE: 48.13 units
- **Features**: Brand, model, type, state, year, month, rolling averages
- **Training Data**: 1,218 sales records

---

## 📊 Datasets

| Dataset | Records | Description |
|---------|---------|-------------|
| **indian_ev_vehicles.csv** | 58 | EV specifications (price, range, battery, efficiency) |
| **indian_ev_sales.csv** | 1,218 | Monthly sales data by brand, model, state (2021-2024) |
| **india_ev_charging_stations.csv** | 458 | Charging stations with locations, operators, types |
| **state_ev_subsidies.csv** | 22 | State-level EV subsidy programs |
| **fame_ii_subsidy.csv** | - | FAME-II scheme rules and eligibility |

---

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Smooth Animations**: Professional transitions and hover effects
- **Dark/Light Elements**: Modern gradient cards and backgrounds
- **Interactive Charts**: CSS-based visualizations (no external libraries)
- **Real-time Updates**: Live data fetching and filtering
- **Error Handling**: User-friendly error messages

---

## 🧪 Testing

### **Test EV Recommender**
1. Navigate to `/recommender`
2. Set budget: ₹1,50,000
3. Daily km: 50
4. Type: 2-Wheeler
5. Click "Get AI Recommendations"

### **Test Subsidy Calculator**
1. Navigate to `/subsidy`
2. Vehicle type: 4-Wheeler
3. Battery: 40 kWh
4. Price: ₹14,00,000
5. State: Delhi
6. Click "Calculate Subsidy"

### **Test Chatbot**
1. Navigate to `/chatbot`
2. Ask: "What are the best 2-wheelers under 2 lakhs?"
3. Get AI-powered response with formatting

---

## 📝 Development Commands

```bash
# Frontend development
cd react-frontend
npm run dev          # Start dev server
npm run build        # Production build
npm run preview      # Preview production build

# Backend development
python api_server.py # Start FastAPI server

# Model training
python models/train_final_model.py        # Train EV recommender
python models/train_sales_predictor.py    # Train sales predictor
```

---

## 🌟 Key Highlights

✅ **Full-stack application** with modern tech stack  
✅ **Production-ready ML models** with 72%+ accuracy  
✅ **Real-time AI chatbot** using Google Gemini  
✅ **Interactive maps** with 458+ charging stations  
✅ **FAME-II subsidy calculator** with state integration  
✅ **Comprehensive analytics** with interactive charts  
✅ **RESTful API** with FastAPI and automatic docs  
✅ **TypeScript** for type-safe frontend development  
✅ **Responsive design** for all screen sizes  

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Contributors

**Anubhav N. Mishra**  
GitHub: [@anubhav-n-mishra](https://github.com/anubhav-n-mishra)

---

## 🙏 Acknowledgments

- **EDUNET Foundation** - For the internship opportunity
- **AICTE** - For supporting skill development
- **Google Gemini** - AI chatbot capabilities
- **OpenStreetMap** - Free map tiles for charging stations
- **Kaggle Community** - For EV dataset inspiration

---

## 📞 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/anubhav-n-mishra/bharat-ev-saathi-edunet-internship-project/issues)
- Check the `/docs` folder for detailed documentation

---

**Made with ❤️ for the Indian EV Revolution** 🇮🇳⚡
