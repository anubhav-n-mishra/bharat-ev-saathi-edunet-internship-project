"""
EV Sales Prediction Model Training
===================================
Trains an ensemble model to predict future EV sales based on historical data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def prepare_sales_data():
    """Load and prepare sales data for training"""
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'indian_ev_sales.csv')
    
    # Convert month to datetime
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    
    # Encode categorical variables
    brand_encoder = LabelEncoder()
    model_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    
    df['brand_encoded'] = brand_encoder.fit_transform(df['brand'])
    df['model_encoded'] = model_encoder.fit_transform(df['model'])
    df['type_encoded'] = type_encoder.fit_transform(df['type'])
    df['state_encoded'] = state_encoder.fit_transform(df['state'])
    
    # Calculate rolling statistics
    df = df.sort_values(['brand', 'model', 'state', 'month'])
    df['sales_rolling_3'] = df.groupby(['brand', 'model', 'state'])['units_sold'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['sales_rolling_6'] = df.groupby(['brand', 'model', 'state'])['units_sold'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()
    )
    
    # Feature engineering
    features = [
        'brand_encoded', 'model_encoded', 'type_encoded', 'state_encoded',
        'year', 'month_num', 'quarter',
        'sales_rolling_3', 'sales_rolling_6'
    ]
    
    X = df[features].fillna(0).values
    y = df['units_sold'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, {
        'brand_encoder': brand_encoder,
        'model_encoder': model_encoder,
        'type_encoder': type_encoder,
        'state_encoder': state_encoder,
        'feature_names': features
    }

def build_model():
    """Build ensemble model for sales prediction"""
    # Create individual models
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Voting ensemble
    model = VotingRegressor([
        ('rf', rf),
        ('gb', gb)
    ])
    
    return model

def train_model():
    """Train and save the sales prediction model"""
    print("🚀 Training EV Sales Prediction Model...")
    print("=" * 60)
    
    # Prepare data
    print("\n📊 Loading and preparing data...")
    X, y, scaler, encoders = prepare_sales_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build model
    print("\n🏗️  Building ensemble model...")
    model = build_model()
    
    # Train model
    print("\n🎯 Training model...")
    model.fit(X_train, y_train)
    
    # Cross-validation
    print("\n📊 Cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    print(f"CV MAE: {cv_mae:.2f} ± {cv_scores.std():.2f} units")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test MAE: {test_mae:.2f} units")
    print(f"Test RMSE: {test_rmse:.2f} units")
    print(f"Test R² Score: {test_r2:.4f}")
    
    # Save model
    save_dir = PROJECT_ROOT / 'models' / 'saved'
    save_dir.mkdir(exist_ok=True)
    
    print("\n💾 Saving model and artifacts...")
    joblib.dump(model, save_dir / 'sales_predictor.pkl')
    joblib.dump(scaler, save_dir / 'sales_scaler.pkl')
    
    # Save encoders
    encoders_dict = {}
    for key, encoder in encoders.items():
        if key != 'feature_names':
            encoders_dict[key] = encoder.classes_.tolist()
        else:
            encoders_dict[key] = encoder
    
    with open(save_dir / 'sales_encoders.json', 'w') as f:
        json.dump(encoders_dict, f, indent=2)
    
    # Save training metrics
    metrics = {
        'cv_mae': float(cv_mae),
        'cv_std': float(cv_scores.std()),
        'test_mae': float(test_mae),
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': encoders['feature_names']
    }
    
    with open(save_dir / 'sales_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Model training complete!")
    print(f"📁 Model saved to: {save_dir / 'sales_predictor.pkl'}")
    print("=" * 60)
    
    return model

if __name__ == '__main__':
    train_model()

def prepare_sales_data():
    """Load and prepare sales data for training"""
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'indian_ev_sales.csv')
    
    # Convert month to datetime
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    
    # Encode categorical variables
    brand_encoder = LabelEncoder()
    model_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    
    df['brand_encoded'] = brand_encoder.fit_transform(df['brand'])
    df['model_encoded'] = model_encoder.fit_transform(df['model'])
    df['type_encoded'] = type_encoder.fit_transform(df['type'])
    df['state_encoded'] = state_encoder.fit_transform(df['state'])
    
    # Calculate rolling statistics
    df = df.sort_values(['brand', 'model', 'state', 'month'])
    df['sales_rolling_3'] = df.groupby(['brand', 'model', 'state'])['units_sold'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['sales_rolling_6'] = df.groupby(['brand', 'model', 'state'])['units_sold'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()
    )
    
    # Feature engineering
    features = [
        'brand_encoded', 'model_encoded', 'type_encoded', 'state_encoded',
        'year', 'month_num', 'quarter',
        'sales_rolling_3', 'sales_rolling_6'
    ]
    
    X = df[features].fillna(0).values
    y = df['units_sold'].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, {
        'brand_encoder': brand_encoder,
        'model_encoder': model_encoder,
        'type_encoder': type_encoder,
        'state_encoder': state_encoder,
        'feature_names': features
    }

def build_model(input_dim):
    """Build deep learning model for sales prediction"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')  # Regression output
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )
    
    return model

def train_model():
    """Train and save the sales prediction model"""
    print("🚀 Training EV Sales Prediction Model...")
    print("=" * 60)
    
    # Prepare data
    print("\n📊 Loading and preparing data...")
    X, y, scaler, encoders = prepare_sales_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build model
    print("\n🏗️  Building neural network...")
    model = build_model(X.shape[1])
    model.summary()
    
    # Train model
    print("\n🎯 Training model...")
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    print("\n📈 Evaluating model...")
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {test_mae:.2f} units")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.2f} units")
    
    # Save model
    save_dir = PROJECT_ROOT / 'models' / 'saved'
    save_dir.mkdir(exist_ok=True)
    
    print("\n💾 Saving model and artifacts...")
    model.save(save_dir / 'sales_predictor.h5')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, save_dir / 'sales_scaler.pkl')
    
    # Save encoders
    encoders_dict = {}
    for key, encoder in encoders.items():
        if key != 'feature_names':
            encoders_dict[key] = encoder.classes_.tolist()
        else:
            encoders_dict[key] = encoder
    
    with open(save_dir / 'sales_encoders.json', 'w') as f:
        json.dump(encoders_dict, f, indent=2)
    
    # Save training metrics
    metrics = {
        'test_mae': float(test_mae),
        'test_mse': float(test_mse),
        'test_rmse': float(np.sqrt(test_mse)),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': encoders['feature_names']
    }
    
    with open(save_dir / 'sales_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Model training complete!")
    print(f"📁 Model saved to: {save_dir / 'sales_predictor.h5'}")
    print("=" * 60)
    
    return model, history

if __name__ == '__main__':
    train_model()
