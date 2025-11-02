# Bharat EV Saathi - Setup and Run Script
# =========================================
# This script will set up and run the complete application

Write-Host "🚗⚡ Bharat EV Saathi - Setup Script" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""

# Check Python version
Write-Host "📌 Checking Python version..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found! Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Step 1: Create virtual environment
Write-Host ""
Write-Host "📦 Step 1: Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists. Skipping..." -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Step 2: Activate virtual environment
Write-Host ""
Write-Host "🔌 Step 2: Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Step 3: Install dependencies
Write-Host ""
Write-Host "📥 Step 3: Installing dependencies..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Step 4: Download & Generate datasets
Write-Host ""
Write-Host "📊 Step 4: Downloading & Generating datasets..." -ForegroundColor Cyan
$dataGenerated = $false

if (Test-Path "data\processed\indian_ev_vehicles.csv") {
    Write-Host "⚠️  Datasets already exist. Skipping generation..." -ForegroundColor Yellow
    $dataGenerated = $true
} else {
    Write-Host "🌐 Attempting to download real Kaggle datasets..." -ForegroundColor Yellow
    Write-Host "   (If kagglehub not available, will use generated data)" -ForegroundColor Gray
    
    # Try the unified download script first
    python data\raw\download_kaggle_datasets.py
    
    # Check if data was created
    if (Test-Path "data\processed\indian_ev_vehicles.csv") {
        Write-Host "✅ Datasets ready!" -ForegroundColor Green
        $dataGenerated = $true
    } else {
        Write-Host "⚠️  Kaggle download failed. Using generated data..." -ForegroundColor Yellow
        
        Write-Host "Generating EV vehicles data..." -ForegroundColor Yellow
        python data\raw\generate_indian_ev_data.py
        
        Write-Host "Generating charging stations data..." -ForegroundColor Yellow
        python data\raw\generate_charging_stations.py
        
        Write-Host "Generating subsidy data..." -ForegroundColor Yellow
        python data\raw\generate_subsidy_data.py
        
        Write-Host "Processing FAME-II bus data..." -ForegroundColor Yellow
        python data\raw\process_fame_bus_data.py
        
        # Move CSV files to processed directory
        Write-Host "Moving datasets to processed folder..." -ForegroundColor Yellow
        if (Test-Path "data\raw\*.csv") {
            Move-Item -Path "data\raw\*.csv" -Destination "data\processed\" -Force
        }
        
        Write-Host "✅ Datasets generated successfully" -ForegroundColor Green
        $dataGenerated = $true
    }
}

# Step 5: Check API configuration
Write-Host ""
Write-Host "🔑 Step 5: Checking API configuration..." -ForegroundColor Cyan
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "GEMINI_API_KEY=.+") {
        Write-Host "✅ Gemini API key found in .env file" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Gemini API key not configured" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To enable the chatbot:" -ForegroundColor Yellow
        Write-Host "1. Get free API key from: https://ai.google.dev/" -ForegroundColor White
        Write-Host "2. Open .env file and add: GEMINI_API_KEY=your_key_here" -ForegroundColor White
        Write-Host "3. Restart the application" -ForegroundColor White
        Write-Host ""
        Write-Host "The app will work without it (chatbot in demo mode)" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  .env file not found. Creating from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✅ .env file created. Please add your Gemini API key." -ForegroundColor Green
    Write-Host "   Get it from: https://ai.google.dev/" -ForegroundColor Cyan
}

# Step 6: Summary
Write-Host ""
Write-Host "=======================================" -ForegroundColor Green
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Generated Datasets:" -ForegroundColor Cyan
Write-Host "   ✅ Indian EV Vehicles (25+ models)" -ForegroundColor White
Write-Host "   ✅ Charging Stations (500+ locations)" -ForegroundColor White
Write-Host "   ✅ FAME & State Subsidies" -ForegroundColor White
Write-Host "   ✅ Sales Data (2023-2024)" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Starting Streamlit Application..." -ForegroundColor Cyan
Write-Host ""

# Step 7: Run the application
streamlit run frontend\app.py
