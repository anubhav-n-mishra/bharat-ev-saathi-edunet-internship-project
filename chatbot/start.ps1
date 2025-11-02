# EV Chatbot - Quick Start Script
# ================================

Write-Host ""
Write-Host "🤖 EV EXPERT CHATBOT - QUICK START" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""

# Check Python
Write-Host "📌 Checking Python..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found! Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists. Skipping..." -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "📥 Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Check API key
Write-Host ""
Write-Host "🔑 Checking API configuration..." -ForegroundColor Cyan
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "GEMINI_API_KEY=AIza") {
        Write-Host "✅ Gemini API key found and configured!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  API key not found in .env file" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ .env file not found!" -ForegroundColor Red
    exit 1
}

# Launch chatbot
Write-Host ""
Write-Host "===================================" -ForegroundColor Green
Write-Host "🚀 LAUNCHING CHATBOT..." -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""
Write-Host "Choose interface:" -ForegroundColor Cyan
Write-Host "1. Web Interface (Streamlit) - Recommended" -ForegroundColor White
Write-Host "2. Terminal Interface" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host ""
    Write-Host "🌐 Starting web interface..." -ForegroundColor Green
    Write-Host "The app will open in your browser at http://localhost:8501" -ForegroundColor Cyan
    Write-Host ""
    streamlit run chatbot_app.py
} elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "💻 Starting terminal interface..." -ForegroundColor Green
    Write-Host ""
    python ev_chatbot.py
} else {
    Write-Host "❌ Invalid choice. Exiting..." -ForegroundColor Red
    exit 1
}
