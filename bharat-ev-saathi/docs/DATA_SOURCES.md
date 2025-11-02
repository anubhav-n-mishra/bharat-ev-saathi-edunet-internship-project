# 📊 Data Sources - Bharat EV Saathi

This document explains all data sources used in the project, including both **real datasets from Kaggle** and **programmatically generated data**.

---

## 🎯 Real Datasets (Kaggle)

### 1. **EV Charging Stations in India (2025)** ⚡
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/pranjal9091/ev-charging-stations-in-india-simplified-2025)
- **Author**: pranjal9091
- **Size**: 500+ charging stations
- **Coverage**: All states across India
- **Update Frequency**: 2025 data
- **Usage in Project**: 
  - Charging station locator feature
  - Range anxiety calculation
  - Route planning (future enhancement)

**How to Download**:
```python
import kagglehub
path = kagglehub.dataset_download("pranjal9091/ev-charging-stations-in-india-simplified-2025")
```

---

### 2. **Electric Vehicle Specifications Dataset (2025)** 🚗
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/urvishahir/electric-vehicle-specifications-dataset-2025)
- **Author**: urvishahir
- **Coverage**: Latest EV models available globally
- **Update Frequency**: 2025 data
- **Usage in Project**:
  - EV recommendation engine
  - Price comparison
  - Range analysis
  - Efficiency calculations

**How to Download**:
```python
import kagglehub
path = kagglehub.dataset_download("urvishahir/electric-vehicle-specifications-dataset-2025")
```

---

### 3. **FAME-II Bus Deployment Data** 🚌
- **Source**: RS_Session_265_AU_2154_A_and_B_2.csv (Government data)
- **Type**: FAME-II electric bus sanctioning and deployment
- **Coverage**: 
  - State-wise bus sanctioned
  - Buses received and deployed
  - All Indian states/UTs
- **Usage in Project**:
  - State EV adoption analysis
  - Government initiative tracking
  - Infrastructure readiness indicator

**Sample Data**:
```
State/UT,Number of Buses Sanctioned,Number of Buses Received and Deployed
Andhra Pradesh,100,100
Delhi,1321,1321
Gujarat,800,625
```

---

## 🤖 Generated/Programmatic Data

### 4. **Indian EV Market Data** 🇮🇳
- **Source**: `generate_indian_ev_data.py`
- **Type**: Programmatically generated with real specifications
- **Coverage**: 60+ Indian EV models
  - 21 Two-Wheelers (Ather, Ola, TVS, Bajaj, etc.)
  - 26 Four-Wheelers (Tata, Mahindra, MG, BYD, Mercedes, BMW, etc.)
  - 7 Three-Wheelers (Mahindra, Piaggio, Euler, etc.)
- **Data Points**: 
  - Price, Range, Battery capacity
  - Charging time, Top speed
  - FAME eligibility, Subsidies
  - User ratings, Efficiency metrics

**Why Generated?**:
The Kaggle EV dataset is global. We created India-specific data with:
- Accurate Indian pricing (₹)
- FAME-II subsidy calculations
- Popular Indian models
- State-wise availability

---

### 5. **State EV Subsidy Database** 💰
- **Source**: `generate_subsidy_data.py`
- **Type**: Programmatically generated from official policies
- **Coverage**: 
  - Central FAME-II subsidies
  - 10+ state policies (Delhi, Maharashtra, Gujarat, Karnataka, etc.)
  - Scrapping bonuses
- **Data Points**:
  - Subsidy amount by vehicle type
  - Maximum caps
  - Eligibility criteria
  - State-specific incentives

**Based On**:
- Official FAME-II policy documents (2024)
- State EV policy announcements
- Ministry of Heavy Industries guidelines

---

### 6. **EV Sales Data** 📈
- **Source**: `generate_indian_ev_data.py` (generate_sales_data function)
- **Type**: Realistic sales patterns
- **Coverage**: 
  - Monthly sales (2023-2024)
  - 20,000+ records
  - State-wise distribution
  - Model-wise trends

**Methodology**:
- Base sales volumes derived from market reports
- Growth trends matching industry data (30-40% YoY)
- Seasonal variations
- State-wise popularity patterns

---

## 📥 How to Get All Data

### Option 1: Automated Script (Recommended)
```powershell
cd "c:\Users\anubh\Downloads\internship\bharat-ev-saathi"
python data\raw\download_kaggle_datasets.py
```

This will:
1. Download Kaggle datasets (if kagglehub installed)
2. Fall back to generated data if Kaggle unavailable
3. Load FAME-II bus data
4. Generate supplementary datasets
5. Save everything to `data/processed/`

### Option 2: Manual Download
1. **Install kagglehub**: `pip install kagglehub`
2. **Authenticate Kaggle**: Get API key from kaggle.com/settings
3. **Run individual scripts**:
   ```powershell
   python data\raw\generate_charging_stations.py
   python data\raw\generate_indian_ev_data.py
   python data\raw\generate_subsidy_data.py
   ```

---

## 🎓 Why This Hybrid Approach?

### Advantages:
1. **Real World Data**: Actual charging stations and global EV specs from Kaggle
2. **India-Specific Context**: Generated data captures FAME-II, state subsidies, Indian models
3. **No API Limits**: Works offline after initial download
4. **Educational Value**: Shows data engineering + external data integration
5. **Reproducibility**: Anyone can generate the same datasets

### Data Quality:
- **Kaggle datasets**: Verified by community, regular updates
- **Generated data**: Based on official sources (govt websites, manufacturer specs)
- **FAME-II data**: Direct government data (RS Session document)

---

## 📚 Citations & Credits

### Kaggle Datasets:
```
1. Pranjal9091. (2025). EV Charging Stations in India Simplified 2025. Kaggle.
   https://www.kaggle.com/datasets/pranjal9091/ev-charging-stations-in-india-simplified-2025

2. Urvishahir. (2025). Electric Vehicle Specifications Dataset 2025. Kaggle.
   https://www.kaggle.com/datasets/urvishahir/electric-vehicle-specifications-dataset-2025
```

### Official Sources:
- Ministry of Heavy Industries - FAME-II Policy
- State EV Policy Documents (Delhi, Maharashtra, Gujarat, Karnataka, etc.)
- Manufacturer websites for EV specifications

---

## 🔄 Data Updates

### How Often to Update:
- **Kaggle datasets**: Monthly (check dataset update dates)
- **Generated data**: When new models launch or policies change
- **FAME-II data**: Quarterly (government releases)

### Update Command:
```powershell
python data\raw\download_kaggle_datasets.py --force-download
```

---

## 🛡️ Data Privacy & Ethics

- **No Personal Data**: All datasets are aggregated statistics
- **Public Sources**: Only publicly available data used
- **Compliance**: Follows Kaggle terms of service
- **Attribution**: All sources properly credited

---

## 📧 Questions?

If you have questions about the data sources or want to contribute new datasets, please:
1. Check if dataset is India-specific
2. Verify data accuracy
3. Ensure proper licensing
4. Submit a pull request with documentation

---

**Last Updated**: November 2025
**Data Version**: v1.0
