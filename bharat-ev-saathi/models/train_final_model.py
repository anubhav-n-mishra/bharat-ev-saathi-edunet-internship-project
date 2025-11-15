"""
Production-Ready EV Recommendation Model
========================================
Advanced model with proper handling for small datasets.
Uses ensemble methods with strong regularization.

Author: Bharat EV Saathi Project
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.data_loader import ev_data

np.random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / 'models' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ProductionEVRecommender:
    """Production-ready EV Recommendation System"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.num_classes = 0
        print("✅ Production EV Recommender initialized")
    
    def engineer_features(self, df):
        """Engineer high-quality features for small dataset"""
        print("\n🔧 Engineering features...")
        
        # Core features (keep simple for small dataset)
        df['price_per_kwh'] = df['price_inr'] / (df['battery_kwh'] + 1)
        df['range_per_kwh'] = df['range_km'] / (df['battery_kwh'] + 1)
        df['efficiency_score'] = df['range_km'] * df['efficiency_km_per_kwh'] / 100
        df['value_score'] = df['range_km'] / (df['price_inr'] / 100000)
        df['charging_speed'] = df['battery_kwh'] / (df['charging_time'] + 0.1)
        df['power_to_weight'] = df['battery_kwh'] / (df['seating_capacity'] + 1)
        
        # Category features
        df['price_category'] = pd.cut(df['price_inr'], bins=[0, 300000, 800000, 2000000, np.inf], labels=[0,1,2,3]).astype(int)
        df['range_category'] = pd.cut(df['range_km'], bins=[0, 150, 250, 350, np.inf], labels=[0,1,2,3]).astype(int)
        df['battery_category'] = pd.cut(df['battery_kwh'], bins=[0, 10, 25, 50, np.inf], labels=[0,1,2,3]).astype(int)
        
        # Boolean features
        df['is_premium'] = (df['price_inr'] > 1000000).astype(int)
        df['is_long_range'] = (df['range_km'] > 250).astype(int)
        df['fame_encoded'] = (df['fame_eligible'] == 'Yes').astype(int)
        
        # Type encoding
        type_map = {'2-Wheeler': 0, '3-Wheeler': 1, '4-Wheeler': 2}
        df['type_encoded'] = df['type'].map(type_map).fillna(1).astype(int)
        
        # Use simplified vehicle segments (broader categories)
        segment_map = {
            'Tata Motors': 0, 'Mahindra': 0, 'MG Motor': 0,  # Large OEMs
            'Ola Electric': 1, 'Ather': 1, 'Hero Electric': 1, 'TVS': 1,  # 2-Wheeler brands
            'BYD': 2, 'Hyundai': 2,  # International brands
        }
        df['brand_segment'] = df['brand'].map(segment_map).fillna(3).astype(int)
        
        return df
    
    def prepare_data(self):
        """Load and prepare data"""
        print("\n📊 Loading and preparing data...")
        df = ev_data.load_ev_vehicles().copy()
        print(f"   Loaded {len(df)} EV models")
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Use brand_segment as target (broader categories, better for small dataset)
        df['vehicle_class'] = df['brand_segment']
        
        # Select features (keep it simple - 15 key features)
        self.feature_columns = [
            'price_inr', 'range_km', 'battery_kwh', 'top_speed', 'charging_time',
            'efficiency_km_per_kwh', 'seating_capacity',
            'price_per_kwh', 'range_per_kwh', 'efficiency_score', 'value_score',
            'charging_speed', 'type_encoded', 'fame_encoded', 'is_premium'
        ]
        
        X = df[self.feature_columns].fillna(0).values
        y = df['vehicle_class']
        
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"   ✅ {len(self.feature_columns)} features")
        print(f"   ✅ {self.num_classes} vehicle segments")
        
        # Split: 75/25 for small dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, random_state=42
        )
        
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing:  {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, df
    
    def train(self):
        """Train ensemble model with proper regularization"""
        print("\n" + "="*60)
        print("🚀 STARTING PRODUCTION MODEL TRAINING")
        print("="*60)
        
        X_train, X_test, y_train, y_test, df = self.prepare_data()
        
        print("\n🏗️  Training ensemble model...")
        print("   Using 3 complementary models:")
        
        # Model 1: Random Forest (strong regularization for small dataset)
        rf = RandomForestClassifier(
            n_estimators=100,  # Reduced from 300
            max_depth=5,       # Reduced from 20 to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        print("   ✅ Random Forest (100 trees, depth=5)")
        
        # Model 2: Gradient Boosting (gentle boosting)
        gb = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        print("   ✅ Gradient Boosting (50 trees, depth=3)")
        
        # Model 3: Another RF with different parameters
        rf2 = RandomForestClassifier(
            n_estimators=80,
            max_depth=4,
            min_samples_split=6,
            min_samples_leaf=4,
            max_features='log2',
            random_state=43,
            n_jobs=-1
        )
        print("   ✅ Random Forest #2 (80 trees, depth=4)")
        
        # Create voting ensemble
        print("\n   Combining models with soft voting...")
        self.model = VotingClassifier(
            estimators=[('rf1', rf), ('gb', gb), ('rf2', rf2)],
            voting='soft',
            n_jobs=-1
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
        
        # Get feature importance from the trained estimator
        rf_trained = self.model.estimators_[0]  # Get first estimator (RF)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_trained.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:25s}: {row['importance']:.4f}")
        
        # Cross-validation
        print(f"\n🔄 Cross-Validation (3-Fold for small dataset):")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=3, n_jobs=-1)
        print(f"   CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
        print(f"   Mean CV Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
        
        metrics = {
            'model_type': 'Voting Ensemble (RF + GB + RF)',
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'f1_score': float(f1),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'num_features': len(self.feature_columns),
            'num_classes': self.num_classes,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': self.feature_columns,
            'feature_importance': feature_importance.to_dict('records')
        }
        
        return metrics, feature_importance
    
    def save_model(self):
        """Save complete model"""
        print("\n💾 Saving model and artifacts...")
        
        # Main model files
        model_path = MODEL_DIR / 'ev_recommender_production.pkl'
        joblib.dump(self.model, str(model_path))
        print(f"   ✅ Model saved: {model_path}")
        
        # H5-style naming for compatibility
        h5_path = MODEL_DIR / 'ev_recommender_model.h5.pkl'
        joblib.dump(self.model, str(h5_path))
        print(f"   ✅ Model saved (H5-style): {h5_path}")
        
        # Preprocessing artifacts
        joblib.dump(self.scaler, str(MODEL_DIR / 'scaler.pkl'))
        joblib.dump(self.label_encoder, str(MODEL_DIR / 'label_encoder.pkl'))
        print(f"   ✅ Scaler & encoder saved")
        
        # Feature info
        with open(MODEL_DIR / 'feature_columns.json', 'w') as f:
            json.dump({'features': self.feature_columns}, f, indent=2)
        print(f"   ✅ Features saved")
        
        print(f"\n✅ All artifacts saved to: {MODEL_DIR}")
    
    def plot_results(self, feature_importance):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        
        top_n = min(15, len(feature_importance))
        top_features = feature_importance.head(top_n)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Features\nEnsemble Model (RF + GB)', 
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plot_path = MODEL_DIR / 'feature_importance.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        print(f"\n📊 Feature importance plot saved: {plot_path}")
        plt.close()


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🚗⚡ BHARAT EV SAATHI - PRODUCTION MODEL TRAINING")
    print("="*60)
    print("Ensemble Model with Proper Regularization")
    print("Optimized for Small Datasets")
    print("="*60 + "\n")
    
    # Initialize
    recommender = ProductionEVRecommender()
    
    # Train
    metrics, feature_importance = recommender.train()
    
    # Save
    recommender.save_model()
    
    # Plot
    recommender.plot_results(feature_importance)
    
    # Save metrics
    with open(MODEL_DIR / 'training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics saved")
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"   Model Type:         {metrics['model_type']}")
    print(f"   Training Accuracy:  {metrics['train_accuracy']*100:.2f}%")
    print(f"   Testing Accuracy:   {metrics['test_accuracy']*100:.2f}%")
    print(f"   F1 Score:           {metrics['f1_score']:.4f}")
    print(f"   CV Mean Score:      {metrics['cv_mean']*100:.2f}%")
    print(f"   Features Used:      {metrics['num_features']}")
    print(f"   Vehicle Segments:   {metrics['num_classes']}")
    
    print(f"\n✅ Models saved:")
    print(f"   • ev_recommender_production.pkl")
    print(f"   • ev_recommender_model.h5.pkl (H5-compatible)")
    print(f"   • scaler.pkl, label_encoder.pkl")
    print(f"   • feature_columns.json")
    print(f"   • training_metrics.json")
    print(f"   • feature_importance.png")
    
    print(f"\n📂 Location: {MODEL_DIR}")
    
    print("\n" + "="*60)
    print("🚀 Ready for production deployment!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
