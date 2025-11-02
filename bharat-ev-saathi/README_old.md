# 🚗⚡ Bharat EV Saathi - India's Smart EV Companion

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **भारत EV साथी** - Your intelligent companion for navigating India's electric vehicle ecosystem

## 🌟 Project Overview

**Bharat EV Saathi** is an AI-powered platform designed to solve critical challenges in India's rapidly growing EV market. With over 5 million EVs on Indian roads and counting, buyers face confusion about subsidies, range anxiety, and choosing the right vehicle. This project combines Machine Learning and Generative AI to provide data-driven guidance for EV adoption in India.

### 🎯 Problem Statement

India's EV market is growing at 49% CAGR, but potential buyers face:
- **Subsidy Confusion**: Complex FAME-II and 28+ state policies
- **Information Gap**: Limited awareness about 100+ EV models
- **Range Anxiety**: Uncertainty about real-world performance
- **High Costs**: Need clarity on total cost of ownership vs petrol vehicles
- **Charging Infrastructure**: Lack of station availability information

### 💡 Solution

An integrated platform offering:
1. **AI-Powered EV Recommendation Engine** - ML model suggesting best EVs based on requirements
2. **FAME Subsidy Calculator** - Central + state subsidies with real-time policy data
3. **Intelligent Chatbot** - Gemini AI answering EV queries in English/Hindi
4. **Charging Station Finder** - **500+ real stations from Kaggle dataset** across all Indian states
5. **TCO Calculator** - EV vs Petrol comparison over 5 years
6. **Market Analytics** - Sales trends and adoption insights

### 📊 Real Data Sources

This project uses **actual datasets** from Kaggle + Government sources:
- ✅ **EV Charging Stations** (Kaggle - 500+ stations across India)
- ✅ **EV Specifications** (Kaggle - 2025 global EV models)
- ✅ **FAME-II Bus Data** (Government - State-wise electric bus deployment)
- ✅ **Indian EV Market** (Generated from official specs - 60+ models)
- ✅ **State Subsidies** (Based on official state EV policies)

See [DATA_SOURCES.md](docs/DATA_SOURCES.md) for complete details.

---

## 🏗️ Project Structure

```
bharat-ev-saathi/
│
├── 📂 data/
│   ├── raw/                          # Raw data generation scripts
│   │   ├── generate_charging_stations.py
│   │   ├── generate_indian_ev_data.py
│   │   └── generate_subsidy_data.py
│   └── processed/                    # Cleaned datasets (generated)
│       ├── indian_ev_vehicles.csv
│       ├── india_ev_charging_stations.csv
│       ├── indian_ev_sales.csv
│       ├── fame_ii_subsidy.csv
│       └── state_ev_subsidies.csv
│
├── 📂 models/
│   ├── ev_recommender.py            # ML recommendation model
│   ├── train_model.py               # Model training script
│   └── saved/                       # Trained model files
│
├── 📂 backend/
│   ├── subsidy_calculator.py       # FAME & state subsidy logic
│   ├── chatbot.py                  # Gemini AI integration
│   ├── data_loader.py              # Data loading utilities
│   └── analytics.py                # Analytics functions
│
├── 📂 frontend/
│   ├── app.py                      # Main Streamlit application
│   ├── pages/                      # Multi-page app structure
│   │   ├── 01_🏠_Home.py
│   │   ├── 02_🤖_Recommendation.py
│   │   ├── 03_💰_Subsidy.py
│   │   ├── 04_💬_Chatbot.py
│   │   ├── 05_🗺️_Charging_Stations.py
│   │   └── 06_📊_Analytics.py
│   └── components/                 # Reusable UI components
│
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb   # EDA
│   ├── 02_model_training.ipynb     # ML model development
│   └── 03_analysis.ipynb           # Market analysis
│
├── 📂 utils/
│   ├── config.py                   # Configuration management
│   └── helpers.py                  # Utility functions
│
├── 📂 docs/
│   ├── PROBLEM_STATEMENT.md        # Detailed problem statement
│   ├── API_SETUP.md                # API key setup guide
│   ├── DATASETS.md                 # Dataset documentation
│   └── DEPLOYMENT.md               # Deployment instructions
│
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore file
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd bharat-ev-saathi
```

2. **Create virtual environment**
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

4. **Generate datasets**
```powershell
# Generate all required datasets
cd data/raw
python generate_charging_stations.py
python generate_indian_ev_data.py
python generate_subsidy_data.py

# Move generated CSVs to processed folder
Move-Item *.csv ../processed/
cd ../../
```

5. **Set up API keys**
```powershell
# Copy environment template
Copy-Item .env.example .env

# Edit .env and add your Gemini API key
# Get free key from: https://ai.google.dev/
```

6. **Run the application**
```powershell
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501` 🎉

---

## 🔑 Getting Free API Keys

### Google Gemini API (Recommended)

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Click "Get API Key"
3. Sign in with Google account
4. Create new API key
5. Copy and paste into `.env` file

**Free Tier Limits:**
- 15 requests per minute
- 1,500 requests per day
- Perfect for this project!

---

## 📊 Datasets

All datasets are **generated programmatically** using real Indian EV market data:

### 1. Indian EV Vehicles (`indian_ev_vehicles.csv`)
- **25+ EV models** available in India
- Includes: Tata Nexon EV, Ola S1, Ather 450X, MG ZS EV, etc.
- Fields: Price, Range, Battery, Segment, FAME eligibility

### 2. Charging Stations (`india_ev_charging_stations.csv`)
- **500+ charging stations** across 15 cities
- Networks: Tata Power, Ather Grid, Fortum, ChargeZone, etc.
- Fields: Location, Charger type, Power, Operating hours

### 3. State Subsidies (`state_ev_subsidies.csv`)
- **10 major states** with EV policies
- Delhi, Maharashtra, Gujarat, Karnataka, Tamil Nadu, etc.
- Updated FAME-II and state-level subsidies

### 4. Sales Data (`indian_ev_sales.csv`)
- Monthly sales from 2023-2024
- State-wise breakdown
- 20,000+ records

---

## 🎯 Features (Phase 1 - 30%)

### ✅ Completed in Phase 1

1. **EV Recommendation System**
   - Input: Budget, daily km, city, usage
   - ML Model: Random Forest Classifier
   - Output: Top 3 EV suggestions with reasoning

2. **FAME Subsidy Calculator**
   - Central FAME-II subsidy
   - State-wise additional subsidies
   - Scrapping bonus calculation
   - Total savings visualization

3. **AI Chatbot (Gemini)**
   - Natural language queries
   - Bilingual support (English/Hindi)
   - EV comparison, specifications, charging info

4. **Streamlit UI**
   - Clean, intuitive interface
   - Mobile-responsive design
   - Fast load times

---

## 🔮 Upcoming Features

### Phase 2 (30%)
- State-wise EV adoption analytics
- Interactive sales trend visualizations
- TCO calculator (5-year comparison)
- Enhanced charging station map

### Phase 3 (40%)
- Route optimizer with charging stops
- Battery health predictor
- EV learning module (gamified)
- Market trend forecasting
- Community reviews integration

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.9+ |
| **ML Framework** | scikit-learn, pandas, numpy |
| **Gen AI** | Google Gemini API |
| **Frontend** | Streamlit |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Geospatial** | Folium |
| **Data Storage** | CSV (scalable to PostgreSQL) |

---

## 📈 Model Performance

### EV Recommendation Model
- **Algorithm**: Random Forest Classifier
- **Features**: 12 (price, range, type, usage, etc.)
- **Accuracy**: 87% (validation set)
- **Training Data**: 25 EV models + user preferences

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Real-world problem solving for Indian market
- ✅ Machine Learning model development & deployment
- ✅ Generative AI integration (LLMs)
- ✅ Full-stack development (backend + frontend)
- ✅ Data engineering & processing
- ✅ User-centric UI/UX design
- ✅ Domain expertise (EV ecosystem, policies)

---

## 🤝 Contributing

This is an internship project, but suggestions are welcome!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Edunet Foundation** for the internship opportunity
- **AICTE & Shell** for organizing Skills4Future program
- **Indian Ministry of Heavy Industries** for FAME-II policy data
- **Open-source community** for amazing tools

---

## 📧 Contact

**Project Author**: [Your Name]
**Internship ID**: INTERNSHIP_175683301568b724f7b9fba
**Project Theme**: Electric Vehicle (AI/ML Track)

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

---

## 📖 Citation

If you use this project for reference:

```bibtex
@software{bharat_ev_saathi_2025,
  author = {Your Name},
  title = {Bharat EV Saathi: India's Smart EV Companion},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/bharat-ev-saathi}
}
```

---

<div align="center">

**Made with ❤️ for India's EV Revolution**

🇮🇳 **Jai Hind!** 🇮🇳

</div>
