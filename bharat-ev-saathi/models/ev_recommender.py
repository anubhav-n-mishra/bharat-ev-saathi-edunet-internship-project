"""
EV Recommendation Engine
=========================
Machine Learning model to recommend best EVs based on user requirements.
Uses ensemble learning with feature engineering for Indian market.

Author: Bharat EV Saathi Project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path
import logging

from backend.data_loader import ev_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / 'models' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class EVRecommender:
    """
    Intelligent EV recommendation system using machine learning
    """
    
    def __init__(self):
        """Initialize the recommender"""
        self.data_loader = ev_data
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        logger.info("EV Recommender initialized")
    
    def prepare_training_data(self):
        """
        Prepare and engineer features for training
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, full_data)
        """
        # Load EV data
        df = self.data_loader.load_ev_vehicles().copy()
        
        # Feature Engineering
        # 1. Price category
        df['price_category'] = pd.cut(
            df['price_inr'], 
            bins=[0, 200000, 500000, 1500000, 5000000, float('inf')],
            labels=['Budget', 'Affordable', 'Mid-Range', 'Premium', 'Luxury']
        )
        
        # 2. Range category
        df['range_category'] = pd.cut(
            df['range_km'],
            bins=[0, 100, 200, 300, 400, float('inf')],
            labels=['Very Short', 'Short', 'Medium', 'Good', 'Excellent']
        )
        
        # 3. Efficiency score (km per kWh per ₹1000)
        df['value_score'] = (df['range_km'] / df['battery_kwh']) / (df['price_inr'] / 100000)
        
        # 4. Fast charging capability (based on battery size)
        df['fast_charge'] = (df['battery_kwh'] > 10).astype(int)
        
        # 5. Segment encoding
        df['is_premium'] = df['segment'].str.contains('Premium|Luxury', case=False, na=False).astype(int)
        
        # Create target variable (vehicle ID for classification)
        df['vehicle_id'] = df['brand'] + '_' + df['model']
        
        # Select features for the model
        feature_cols = [
            'price_inr', 'range_km', 'battery_kwh', 'top_speed', 
            'charging_time', 'efficiency_km_per_kwh', 'seating_capacity',
            'value_score', 'fast_charge', 'is_premium'
        ]
        
        # Encode categorical columns
        categorical_cols = ['type', 'price_category', 'range_category']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                feature_cols.append(f'{col}_encoded')
        
        self.feature_columns = feature_cols
        
        # Prepare X and y
        X = df[feature_cols].fillna(0)
        y = df['vehicle_id']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df['type']
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training data prepared: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df
    
    def train(self):
        """
        Train the recommendation model
        
        Returns:
            Dictionary with training metrics
        """
        X_train, X_test, y_train, y_test, full_data = self.prepare_training_data()
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Model trained - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance
        }
    
    def recommend(self, budget, daily_km, vehicle_type='4-Wheeler', 
                  city='Mumbai', usage='personal', top_n=3):
        """
        Recommend EVs based on user requirements
        
        Args:
            budget: Maximum budget in INR
            daily_km: Daily driving distance in km
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            city: City of residence
            usage: 'personal' or 'commercial'
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended EVs with scores and reasons
        """
        # Load all EVs
        all_evs = self.data_loader.load_ev_vehicles()
        
        # Filter by type and budget
        filtered_evs = all_evs[
            (all_evs['type'] == vehicle_type) & 
            (all_evs['price_inr'] <= budget)
        ].copy()
        
        if len(filtered_evs) == 0:
            return []
        
        # Calculate required range (daily_km * 3 for buffer)
        required_range = daily_km * 3
        
        # Scoring system
        scores = []
        for _, ev in filtered_evs.iterrows():
            score = 0
            reasons = []
            
            # 1. Range score (40% weight)
            if ev['range_km'] >= required_range:
                range_score = min(100, (ev['range_km'] / required_range) * 40)
                score += range_score
                reasons.append(f"Range: {ev['range_km']} km covers your {daily_km} km/day needs")
            else:
                range_score = (ev['range_km'] / required_range) * 40
                score += range_score
                reasons.append(f"Range: {ev['range_km']} km (may need frequent charging)")
            
            # 2. Price-value score (30% weight)
            price_ratio = 1 - (ev['price_inr'] / budget)
            price_score = price_ratio * 30
            score += price_score
            if price_ratio > 0.5:
                reasons.append(f"Great value at ₹{ev['price_inr']:,} (within budget)")
            
            # 3. Efficiency score (20% weight)
            efficiency_score = min(20, ev['efficiency_km_per_kwh'] * 2)
            score += efficiency_score
            if ev['efficiency_km_per_kwh'] > 8:
                reasons.append(f"Excellent efficiency: {ev['efficiency_km_per_kwh']:.1f} km/kWh")
            
            # 4. User rating (10% weight)
            if 'user_rating' in ev:
                rating_score = (ev['user_rating'] / 5) * 10
                score += rating_score
                if ev['user_rating'] >= 4.0:
                    reasons.append(f"High user rating: {ev['user_rating']}/5")
            
            # Bonus for FAME eligibility
            if ev.get('fame_eligible') == 'Yes':
                score += 5
                reasons.append(f"FAME subsidy eligible (save ₹{ev.get('central_subsidy_inr', 0):,})")
            
            scores.append({
                'ev': ev,
                'score': score,
                'reasons': reasons
            })
        
        # Sort by score
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Prepare recommendations
        recommendations = []
        for i, item in enumerate(scores[:top_n], 1):
            ev = item['ev']
            recommendations.append({
                'rank': i,
                'brand': ev['brand'],
                'model': ev['model'],
                'price': ev['price_inr'],
                'range': ev['range_km'],
                'battery_kwh': ev['battery_kwh'],
                'type': ev['type'],
                'segment': ev['segment'],
                'score': round(item['score'], 2),
                'reasons': item['reasons'],
                'user_rating': ev.get('user_rating', 0),
                'efficiency': ev['efficiency_km_per_kwh'],
                'top_speed': ev['top_speed'],
                'charging_time': ev['charging_time'],
                'fame_eligible': ev.get('fame_eligible', 'No'),
                'subsidy': ev.get('central_subsidy_inr', 0)
            })
        
        return recommendations
    
    def compare_evs(self, ev_list):
        """
        Compare multiple EVs side by side
        
        Args:
            ev_list: List of (brand, model) tuples
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for brand, model in ev_list:
            ev = self.data_loader.get_ev_by_id(brand, model)
            if ev is not None:
                comparison_data.append(ev)
        
        if not comparison_data:
            return None
        
        df = pd.DataFrame(comparison_data)
        
        # Select key columns for comparison
        compare_cols = [
            'brand', 'model', 'price_inr', 'range_km', 'battery_kwh',
            'efficiency_km_per_kwh', 'top_speed', 'charging_time',
            'user_rating', 'fame_eligible'
        ]
        
        return df[compare_cols]
    
    def save_model(self, filename='ev_recommender.pkl'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = MODEL_DIR / filename
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename='ev_recommender.pkl'):
        """Load a trained model"""
        model_path = MODEL_DIR / filename
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        
        logger.info(f"Model loaded from {model_path}")
        return True


# Create global instance
recommender = EVRecommender()


if __name__ == '__main__':
    # Test the recommender
    print("🧪 Testing EV Recommender...")
    
    rec = EVRecommender()
    
    # Test 1: Train model
    print("\n📊 Training model...")
    metrics = rec.train()
    print(f"Train Score: {metrics['train_score']:.3f}")
    print(f"Test Score: {metrics['test_score']:.3f}")
    print("\nTop 5 Important Features:")
    print(metrics['feature_importance'].head())
    
    # Test 2: Get recommendations
    print("\n🎯 Getting recommendations for:")
    print("Budget: ₹15 lakhs, Daily: 50 km, Type: 4-Wheeler")
    
    recommendations = rec.recommend(
        budget=1500000,
        daily_km=50,
        vehicle_type='4-Wheeler',
        top_n=3
    )
    
    for rec_item in recommendations:
        print(f"\n{rec_item['rank']}. {rec_item['brand']} {rec_item['model']}")
        print(f"   Price: ₹{rec_item['price']:,}, Range: {rec_item['range']} km")
        print(f"   Score: {rec_item['score']}/100")
        print(f"   Reasons: {'; '.join(rec_item['reasons'][:2])}")
    
    print("\n✅ Recommender test completed!")
