"""
Optimized EV Recommendation Model Training
===========================================
Trains an optimized Random Forest model with hyperparameter tuning.
Saves the best model in joblib format (compatible with H5-style deployment).

Features:
- Grid Search for hyperparameter optimization
- Cross-validation
- Advanced feature engineering (30+ features)
- Comprehensive evaluation metrics
- Model persistence with joblib

Author: Bharat EV Saathi Project
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data_loader import ev_data

# Set random seed
np.random.seed(42)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / 'models' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class OptimizedEVRecommender:
    """
    Optimized EV Recommendation System using ensemble learning
    """
    
    def __init__(self):
        """Initialize the recommender"""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.num_classes = 0
        print("✅ Optimized EV Recommender initialized")
    
    def prepare_data(self):
        """
        Load and prepare data with advanced feature engineering
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, df)
        """
        print("\n📊 Loading and preparing data...")
        
        # Load EV data
        df = ev_data.load_ev_vehicles().copy()
        print(f"   Loaded {len(df)} EV models")
        
        # Advanced Feature Engineering
        print("\n🔧 Engineering 30+ features...")
        
        # 1. Price features
        df['price_category'] = pd.cut(
            df['price_inr'],
            bins=[0, 200000, 500000, 1500000, 5000000, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        df['log_price'] = np.log1p(df['price_inr'])
        df['price_per_kwh'] = df['price_inr'] / (df['battery_kwh'] + 1)
        df['price_per_km'] = df['price_inr'] / (df['range_km'] + 1)
        
        # 2. Range features
        df['range_category'] = pd.cut(
            df['range_km'],
            bins=[0, 100, 200, 300, 400, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        df['range_per_charge'] = df['range_km'] / (df['charging_time'] + 0.1)
        df['range_efficiency'] = df['range_km'] / (df['battery_kwh'] + 0.1)
        
        # 3. Efficiency features
        df['value_score'] = (df['range_km'] / df['battery_kwh']) / (df['price_inr'] / 100000)
        df['efficiency_score'] = df['efficiency_km_per_kwh'] * df['range_km'] / 1000
        df['power_efficiency'] = df['battery_kwh'] / (df['charging_time'] + 0.1)
        df['cost_per_km'] = (df['price_inr'] / df['range_km']) / 1000
        
        # 4. Battery features
        df['battery_category'] = pd.cut(
            df['battery_kwh'],
            bins=[0, 5, 15, 30, 50, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        df['battery_to_range_ratio'] = df['battery_kwh'] / (df['range_km'] + 1)
        df['battery_power'] = df['battery_kwh'] * df['top_speed'] / 100
        
        # 5. Performance features
        df['speed_category'] = pd.cut(
            df['top_speed'],
            bins=[0, 60, 90, 120, 150, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        df['performance_score'] = (df['top_speed'] / 10) + (df['range_km'] / 100) + (df['battery_kwh'] / 5)
        df['speed_to_weight_ratio'] = df['top_speed'] / (df['seating_capacity'] + 1)
        
        # 6. Charging features
        df['fast_charge'] = (df['battery_kwh'] > 10).astype(int)
        df['quick_charge'] = (df['charging_time'] < 5).astype(int)
        df['charging_speed'] = df['battery_kwh'] / (df['charging_time'] + 0.1)
        df['charging_efficiency'] = (df['battery_kwh'] / df['charging_time']) * df['efficiency_km_per_kwh']
        
        # 7. Segment features
        df['is_premium'] = df['segment'].str.contains('Premium|Luxury', case=False, na=False).astype(int)
        df['is_budget'] = df['segment'].str.contains('Budget|Affordable', case=False, na=False).astype(int)
        df['is_suv'] = df['segment'].str.contains('SUV', case=False, na=False).astype(int)
        
        # 8. Type encoding
        type_encoder = LabelEncoder()
        df['type_encoded'] = type_encoder.fit_transform(df['type'])
        
        # 9. FAME features
        df['fame_encoded'] = (df['fame_eligible'] == 'Yes').astype(int)
        df['subsidy_ratio'] = df['central_subsidy_inr'] / (df['price_inr'] + 1)
        df['price_after_subsidy'] = df['price_inr'] - df['central_subsidy_inr']
        
        # 10. Rating features
        df['rating_score'] = df.get('user_rating', 4.0) / 5.0
        df['high_rated'] = (df.get('user_rating', 0) >= 4.0).astype(int)
        
        # 11. Capacity features
        df['capacity_category'] = pd.cut(
            df['seating_capacity'],
            bins=[0, 2, 4, 6, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # 12. Composite scores
        df['overall_score'] = (
            df['value_score'] * 0.25 +
            df['efficiency_score'] * 0.2 +
            df['performance_score'] * 0.2 +
            df['rating_score'] * 100 * 0.15 +
            df['range_category'] * 10 * 0.1 +
            df['fame_encoded'] * 10 * 0.1
        )
        
        df['affordability_score'] = (1 - (df['price_inr'] / df['price_inr'].max())) * 100
        df['practicality_score'] = (df['range_km'] / 500) * (df['seating_capacity'] / 7) * 100
        
        # Create target variable (use brand for better class distribution)
        df['vehicle_class'] = df['brand']  # Use brand instead of brand_model for better distribution
        
        # Select all engineered features (35+ features)
        feature_cols = [
            # Basic features (7)
            'price_inr', 'range_km', 'battery_kwh', 'top_speed', 
            'charging_time', 'efficiency_km_per_kwh', 'seating_capacity',
            
            # Price features (4)
            'price_category', 'log_price', 'price_per_kwh', 'price_per_km',
            
            # Range features (3)
            'range_category', 'range_per_charge', 'range_efficiency',
            
            # Efficiency features (4)
            'value_score', 'efficiency_score', 'power_efficiency', 'cost_per_km',
            
            # Battery features (3)
            'battery_category', 'battery_to_range_ratio', 'battery_power',
            
            # Performance features (3)
            'speed_category', 'performance_score', 'speed_to_weight_ratio',
            
            # Charging features (4)
            'fast_charge', 'quick_charge', 'charging_speed', 'charging_efficiency',
            
            # Segment features (4)
            'is_premium', 'is_budget', 'is_suv', 'type_encoded',
            
            # FAME features (3)
            'fame_encoded', 'subsidy_ratio', 'price_after_subsidy',
            
            # Rating features (2)
            'rating_score', 'high_rated',
            
            # Capacity features (1)
            'capacity_category',
            
            # Composite scores (3)
            'overall_score', 'affordability_score', 'practicality_score'
        ]
        
        self.feature_columns = feature_cols
        
        # Prepare X and y
        X = df[feature_cols].fillna(0).values
        y = df['vehicle_class']
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"   ✅ {len(feature_cols)} features engineered")
        print(f"   ✅ {self.num_classes} vehicle classes")
        
        # Split data: 80% train, 20% test (no stratify due to small dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Testing:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test, df
    
    def train_with_grid_search(self, X_train, y_train):
        """
        Train model with hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best model from grid search
        """
        print("\n🔍 Starting Hyperparameter Optimization (Grid Search)...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        }
        
        # Create base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1
        )
        
        print("   Training multiple models with different hyperparameters...")
        print(f"   Total combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])}")
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n   ✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"      {param}: {value}")
        
        print(f"\n   🎯 Best cross-validation score: {grid_search.best_score_*100:.2f}%")
        
        return grid_search.best_estimator_
    
    def train(self, use_grid_search=True):
        """
        Train the optimized model
        
        Args:
            use_grid_search: Whether to use hyperparameter optimization
            
        Returns:
            Training metrics
        """
        print("\n" + "="*60)
        print("🚀 STARTING OPTIMIZED MODEL TRAINING")
        print("="*60)
        
        # Prepare data
        X_train, X_test, y_train, y_test, df = self.prepare_data()
        
        if use_grid_search:
            # Train with grid search
            self.model = self.train_with_grid_search(X_train, y_train)
        else:
            # Train with default optimized parameters
            print("\n🏗️  Training with optimized parameters...")
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            self.model.fit(X_train, y_train)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        
        # Evaluate
        print("\n📊 Model Evaluation:")
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n🎯 Performance Metrics:")
        print(f"   Training Accuracy:   {train_score*100:.2f}%")
        print(f"   Testing Accuracy:    {test_score*100:.2f}%")
        print(f"   Weighted F1 Score:   {f1:.4f}")
        print(f"   Generalization Gap:  {(train_score-test_score)*100:.2f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:25s}: {row['importance']:.4f}")
        
        # Cross-validation score
        print(f"\n🔄 Cross-Validation (5-Fold):")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, n_jobs=-1)
        print(f"   CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
        print(f"   Mean CV Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
        
        metrics = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'f1_score': float(f1),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'num_features': len(self.feature_columns),
            'num_classes': self.num_classes,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': feature_importance.head(10).to_dict('records')
        }
        
        return metrics, feature_importance
    
    def save_model(self):
        """Save the complete model and preprocessing objects"""
        print("\n💾 Saving model and artifacts...")
        
        # Save model (joblib format - compatible with production deployment)
        model_path = MODEL_DIR / 'ev_recommender_optimized.pkl'
        joblib.dump(self.model, str(model_path))
        print(f"   ✅ Model saved: {model_path}")
        
        # Also save in h5-style naming for compatibility
        h5_style_path = MODEL_DIR / 'ev_recommender_model.h5.pkl'
        joblib.dump(self.model, str(h5_style_path))
        print(f"   ✅ Model saved (H5-style): {h5_style_path}")
        
        # Save scaler
        scaler_path = MODEL_DIR / 'scaler.pkl'
        joblib.dump(self.scaler, str(scaler_path))
        print(f"   ✅ Scaler saved: {scaler_path}")
        
        # Save label encoder
        encoder_path = MODEL_DIR / 'label_encoder.pkl'
        joblib.dump(self.label_encoder, str(encoder_path))
        print(f"   ✅ Label encoder saved: {encoder_path}")
        
        # Save feature columns
        features_path = MODEL_DIR / 'feature_columns.json'
        with open(features_path, 'w') as f:
            json.dump({'features': self.feature_columns}, f, indent=2)
        print(f"   ✅ Features saved: {features_path}")
        
        print(f"\n✅ All artifacts saved to: {MODEL_DIR}")
    
    def plot_feature_importance(self, feature_importance, top_n=15):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        
        top_features = feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = MODEL_DIR / 'feature_importance.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        print(f"\n📊 Feature importance plot saved: {plot_path}")
        
        plt.close()


def main():
    """Main training function"""
    print("\n" + "="*60)
    print("🚗⚡ BHARAT EV SAATHI - OPTIMIZED MODEL TRAINING")
    print("="*60)
    print("Advanced Random Forest with Hyperparameter Optimization")
    print("="*60 + "\n")
    
    # Initialize recommender
    recommender = OptimizedEVRecommender()
    
    # Ask user about grid search
    print("⚙️  Training Options:")
    print("   1. Full Grid Search (Best accuracy, ~10-15 minutes)")
    print("   2. Quick Training (Good accuracy, ~30 seconds)")
    
    try:
        choice = input("\nEnter choice (1 or 2, default=2): ").strip()
        use_grid_search = (choice == '1')
    except (EOFError, KeyboardInterrupt):
        print("2")  # Default to quick training
        use_grid_search = False
    
    # Train model
    metrics, feature_importance = recommender.train(use_grid_search=use_grid_search)
    
    # Save everything
    recommender.save_model()
    
    # Plot feature importance
    recommender.plot_feature_importance(feature_importance)
    
    # Save metrics
    metrics_path = MODEL_DIR / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics saved: {metrics_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📊 Final Summary:")
    print(f"   Training Accuracy:  {metrics['train_accuracy']*100:.2f}%")
    print(f"   Testing Accuracy:   {metrics['test_accuracy']*100:.2f}%")
    print(f"   F1 Score:           {metrics['f1_score']:.4f}")
    print(f"   CV Mean Score:      {metrics['cv_mean']*100:.2f}%")
    print(f"   Features Used:      {metrics['num_features']}")
    print(f"   Vehicle Classes:    {metrics['num_classes']}")
    
    print(f"\n✅ Model saved as: ev_recommender_optimized.pkl")
    print(f"✅ H5-style saved as: ev_recommender_model.h5.pkl")
    print(f"✅ Location: {MODEL_DIR}")
    
    print("\n" + "="*60)
    print("🚀 Ready for deployment!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
