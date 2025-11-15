"""
FastAPI Backend for Bharat EV Saathi
=====================================
Serves ML model recommendations via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ev_recommender import recommender
from backend.chatbot import EVChatbot
from models.fame_calculator import fame_calculator
from models.sales_predictor import sales_predictor

app = FastAPI(
    title="Bharat EV Saathi API",
    description="ML-powered EV recommendation API",
    version="1.0.0"
)

# Initialize chatbot
chatbot = EVChatbot()

# FAME calculator is already initialized globally

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    budget: int = 1500000
    daily_km: int = 50
    vehicle_type: str = "4-Wheeler"
    top_n: int = 5

class RecommendationResponse(BaseModel):
    recommendations: list
    count: int
    model_loaded: bool

@app.get("/")
async def root():
    return {
        "message": "Bharat EV Saathi API",
        "version": "1.0.0",
        "model_loaded": recommender.model is not None,
        "endpoints": {
            "recommendations": "/api/recommendations",
            "stats": "/api/stats",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": recommender.model is not None,
        "ev_count": len(recommender.ev_df) if recommender.ev_df is not None else 0
    }

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get ML-powered EV recommendations based on user preferences
    """
    try:
        # Check if model is loaded
        if recommender.model is None:
            raise HTTPException(
                status_code=503,
                detail="ML model not loaded. Please ensure production model is trained."
            )
        
        # Get recommendations
        recommendations = recommender.recommend_ml(
            budget=request.budget,
            daily_km=request.daily_km,
            vehicle_type=request.vehicle_type,
            top_n=request.top_n
        )
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "model_loaded": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """
    Get dataset statistics
    """
    try:
        from backend.data_loader import ev_data
        stats = ev_data.get_dataset_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charging-stations")
async def get_charging_stations(city: str = None):
    """
    Get all charging stations or filter by city
    """
    try:
        from backend.data_loader import ev_data
        import pandas as pd
        
        # Load charging stations data
        stations_df = ev_data.load_charging_stations()
        
        if city:
            stations_df = stations_df[stations_df['city'].str.lower() == city.lower()]
        
        # Convert to list of dictionaries
        stations = []
        for idx, row in stations_df.iterrows():
            stations.append({
                'id': int(idx),
                'name': str(row.get('station_name', 'Unknown Station')),
                'city': str(row.get('city', 'Unknown')),
                'state': str(row.get('state', 'Unknown')),
                'address': str(row.get('location_type', '')) if pd.notna(row.get('location_type')) else None,
                'latitude': float(row.get('latitude', 0)),
                'longitude': float(row.get('longitude', 0)),
                'operator': str(row.get('network', '')) if pd.notna(row.get('network')) else None,
                'charger_type': str(row.get('charger_type', '')) if pd.notna(row.get('charger_type')) else None,
                'num_chargers': int(row.get('num_chargers', 1)) if pd.notna(row.get('num_chargers')) else 1,
            })
        
        return {
            'stations': stations,
            'count': len(stations),
            'cities': sorted(stations_df['city'].unique().tolist()) if not city else [city]
        }
    except Exception as e:
        logger.error(f"Error in charging stations endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    is_ev_related: bool = True

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the EV AI assistant
    """
    try:
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get response from chatbot
        response = chatbot.chat(request.message)
        
        return {
            "response": response,
            "is_ev_related": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/clear")
async def clear_chat_history():
    """
    Clear chat history
    """
    try:
        chatbot.reset_conversation()
        return {"status": "success", "message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SubsidyRequest(BaseModel):
    vehicle_type: str
    state: str
    battery_capacity_kwh: float = None
    ex_showroom_price: float = None

@app.post("/api/subsidy/calculate")
async def calculate_subsidy(request: SubsidyRequest):
    """
    Calculate FAME-II and state subsidies
    """
    try:
        result = fame_calculator.calculate_total_subsidy(
            vehicle_type=request.vehicle_type,
            state=request.state,
            battery_capacity_kwh=request.battery_capacity_kwh,
            ex_showroom_price=request.ex_showroom_price
        )
        return result
    except Exception as e:
        logger.error(f"Subsidy calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/subsidy/states")
async def get_subsidy_states():
    """
    Get list of states with subsidy programs
    """
    try:
        states = fame_calculator.get_available_states()
        return {"states": states, "count": len(states)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SalesPredictionRequest(BaseModel):
    brand: str
    model: str
    vehicle_type: str
    state: str
    year: int = 2024
    month: int = 12

@app.post("/api/sales/predict")
async def predict_sales(request: SalesPredictionRequest):
    """
    Predict sales for a specific vehicle
    """
    try:
        result = sales_predictor.predict_sales(
            brand=request.brand,
            model=request.model,
            vehicle_type=request.vehicle_type,
            state=request.state,
            year=request.year,
            month=request.month
        )
        return result
    except Exception as e:
        logger.error(f"Sales prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sales/trend")
async def predict_sales_trend(request: SalesPredictionRequest):
    """
    Predict sales trend for next 6 months
    """
    try:
        result = sales_predictor.predict_trend(
            brand=request.brand,
            model=request.model,
            vehicle_type=request.vehicle_type,
            state=request.state,
            months=6
        )
        return result
    except Exception as e:
        logger.error(f"Trend prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Bharat EV Saathi API...")
    print("📊 Loading ML model...")
    
    if recommender.model is None:
        print("⚠️  Warning: Production model not loaded!")
        print("💡 Run: python models/train_final_model.py")
    else:
        print("✅ ML model loaded successfully!")
    
    print("📡 Server running at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
