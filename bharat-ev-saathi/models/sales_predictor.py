"""
EV Sales Predictor
==================
Class for making sales predictions using the trained model
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_names = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load trained model and artifacts"""
        try:
            model_dir = Path(__file__).parent / 'saved'
            
            # Load model
            self.model = joblib.load(model_dir / 'sales_predictor.pkl')
            logger.info("✅ Sales prediction model loaded")
            
            # Load scaler
            self.scaler = joblib.load(model_dir / 'sales_scaler.pkl')
            logger.info("✅ Scaler loaded")
            
            # Load encoders
            with open(model_dir / 'sales_encoders.json', 'r') as f:
                self.encoders = json.load(f)
            self.feature_names = self.encoders['feature_names']
            logger.info("✅ Encoders loaded")
            
        except Exception as e:
            logger.error(f"Error loading sales model: {e}")
            self.model = None
    
    def predict_sales(self, brand, model, vehicle_type, state, year=2024, month=12):
        """
        Predict sales for a specific vehicle
        
        Args:
            brand: Brand name
            model: Model name
            vehicle_type: Vehicle type (2-Wheeler, 3-Wheeler, 4-Wheeler)
            state: State name
            year: Year for prediction
            month: Month for prediction (1-12)
            
        Returns:
            dict with predicted sales and details
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Encode categorical variables
            brand_encoded = self.encoders['brand_encoder'].index(brand) if brand in self.encoders['brand_encoder'] else 0
            model_encoded = self.encoders['model_encoder'].index(model) if model in self.encoders['model_encoder'] else 0
            type_encoded = self.encoders['type_encoder'].index(vehicle_type) if vehicle_type in self.encoders['type_encoder'] else 0
            state_encoded = self.encoders['state_encoder'].index(state) if state in self.encoders['state_encoder'] else 0
            
            quarter = (month - 1) // 3 + 1
            
            # Create feature vector (using dummy values for rolling stats)
            features = np.array([[
                brand_encoded,
                model_encoded,
                type_encoded,
                state_encoded,
                year,
                month,
                quarter,
                1500,  # sales_rolling_3 (dummy)
                1500   # sales_rolling_6 (dummy)
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            prediction = max(0, prediction)  # Ensure non-negative
            
            return {
                'predicted_sales': int(prediction),
                'brand': brand,
                'model': model,
                'type': vehicle_type,
                'state': state,
                'year': year,
                'month': month,
                'confidence': 'medium'
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    def predict_trend(self, brand, model, vehicle_type, state, months=6):
        """Predict sales trend for next N months"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        import datetime
        today = datetime.date.today()
        
        predictions = []
        for i in range(months):
            future_date = today + datetime.timedelta(days=30 * i)
            pred = self.predict_sales(
                brand, model, vehicle_type, state,
                year=future_date.year,
                month=future_date.month
            )
            if 'error' not in pred:
                predictions.append(pred)
        
        return {
            'predictions': predictions,
            'total_predicted': sum(p['predicted_sales'] for p in predictions),
            'average_monthly': int(np.mean([p['predicted_sales'] for p in predictions]))
        }

# Create global instance
sales_predictor = SalesPredictor()

if __name__ == '__main__':
    # Test predictions
    print("Testing Sales Predictor...")
    
    result = sales_predictor.predict_sales(
        brand='Tata',
        model='Nexon EV',
        vehicle_type='4-Wheeler',
        state='Maharashtra',
        year=2024,
        month=12
    )
    
    print(f"\nPrediction Result:")
    print(json.dumps(result, indent=2))
    
    print("\n6-month trend:")
    trend = sales_predictor.predict_trend(
        brand='Tata',
        model='Nexon EV',
        vehicle_type='4-Wheeler',
        state='Maharashtra',
        months=6
    )
    print(json.dumps(trend, indent=2))
