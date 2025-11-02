"""
Download and Prepare Real EV Datasets from Kaggle
==================================================
This script downloads actual datasets from Kaggle and prepares them for the project.

Datasets Used:
1. EV Charging Stations in India (Kaggle)
2. Electric Vehicle Specifications Dataset 2025 (Kaggle)

Author: Bharat EV Saathi Project
Date: November 2025
"""

import pandas as pd
import os
from pathlib import Path

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
    print("✅ kagglehub module found")
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️  kagglehub not installed. Install with: pip install kagglehub")

# Paths
CURRENT_DIR = Path(__file__).parent
PROCESSED_DIR = CURRENT_DIR.parent / 'processed'
PROCESSED_DIR.mkdir(exist_ok=True)

def download_charging_stations():
    """
    Download real EV charging stations data from Kaggle
    Dataset: pranjal9091/ev-charging-stations-in-india-simplified-2025
    """
    print("\n🔌 Downloading EV Charging Stations Dataset from Kaggle...")
    
    if not KAGGLE_AVAILABLE:
        print("❌ kagglehub not available. Using generated data instead.")
        return None
    
    try:
        # Download from Kaggle
        path = kagglehub.dataset_download("pranjal9091/ev-charging-stations-in-india-simplified-2025")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Find CSV file in downloaded path
        path_obj = Path(path)
        csv_files = list(path_obj.glob("*.csv"))
        
        if not csv_files:
            print("❌ No CSV files found in downloaded dataset")
            return None
        
        # Load the CSV
        df = pd.read_csv(csv_files[0])
        print(f"✅ Loaded {len(df)} charging stations")
        
        # Standardize column names if needed
        # Map to our expected format
        column_mapping = {
            # Add mappings based on actual dataset columns
            # Example: 'Station Name': 'station_name'
        }
        
        # Print original columns for debugging
        print(f"\n📋 Original columns: {df.columns.tolist()}")
        
        # Save to processed directory
        output_path = PROCESSED_DIR / 'india_ev_charging_stations.csv'
        df.to_csv(output_path, index=False)
        print(f"✅ Saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading charging stations: {e}")
        print("Will use generated data as fallback")
        return None

def download_ev_specifications():
    """
    Download real EV specifications data from Kaggle
    Dataset: urvishahir/electric-vehicle-specifications-dataset-2025
    """
    print("\n🚗 Downloading EV Specifications Dataset from Kaggle...")
    
    if not KAGGLE_AVAILABLE:
        print("❌ kagglehub not available. Using generated data instead.")
        return None
    
    try:
        # Download from Kaggle
        path = kagglehub.dataset_download("urvishahir/electric-vehicle-specifications-dataset-2025")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Find CSV file
        path_obj = Path(path)
        csv_files = list(path_obj.glob("*.csv"))
        
        if not csv_files:
            print("❌ No CSV files found in downloaded dataset")
            return None
        
        # Load the CSV
        df = pd.read_csv(csv_files[0])
        print(f"✅ Loaded {len(df)} EV specifications")
        
        # Print columns for debugging
        print(f"\n📋 Original columns: {df.columns.tolist()}")
        
        # Standardize and enhance the dataset
        df = process_ev_specifications(df)
        
        # Save to processed directory
        output_path = PROCESSED_DIR / 'indian_ev_vehicles.csv'
        df.to_csv(output_path, index=False)
        print(f"✅ Saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading EV specifications: {e}")
        print("Will use generated data as fallback")
        return None

def process_ev_specifications(df):
    """
    Process and enhance the EV specifications dataset
    Adds calculated fields and standardizes format
    """
    print("\n🔧 Processing EV specifications...")
    
    # Add calculated fields if not present
    if 'efficiency_km_per_kwh' not in df.columns and 'range_km' in df.columns and 'battery_kwh' in df.columns:
        df['efficiency_km_per_kwh'] = (df['range_km'] / df['battery_kwh']).round(2)
    
    # Add FAME eligibility
    if 'fame_eligible' not in df.columns:
        df['fame_eligible'] = df.apply(lambda x:
            'Yes' if (x.get('type') == '2-Wheeler' and x.get('price_inr', 0) <= 150000) or
                     (x.get('type') == '4-Wheeler' and x.get('price_inr', 0) <= 1500000) or
                     (x.get('type') == '3-Wheeler')
            else 'No', axis=1)
    
    # Add central subsidy calculation
    if 'central_subsidy_inr' not in df.columns:
        def calculate_fame_subsidy(row):
            if row.get('fame_eligible') == 'No':
                return 0
            vehicle_type = row.get('type', '')
            battery_kwh = row.get('battery_kwh', 0)
            
            if vehicle_type == '2-Wheeler':
                return min(15000, battery_kwh * 15000)
            elif vehicle_type == '3-Wheeler':
                return min(50000, battery_kwh * 10000)
            elif vehicle_type == '4-Wheeler':
                return 0  # FAME-II primarily for 2W/3W
            return 0
        
        df['central_subsidy_inr'] = df.apply(calculate_fame_subsidy, axis=1).astype(int)
    
    # Add price after subsidy
    if 'price_after_subsidy_inr' not in df.columns:
        df['price_after_subsidy_inr'] = df['price_inr'] - df['central_subsidy_inr']
    
    # Add user ratings if not present
    if 'user_rating' not in df.columns:
        import random
        df['user_rating'] = [round(random.uniform(3.5, 4.9), 1) for _ in range(len(df))]
    
    print(f"✅ Processing complete. Final columns: {df.columns.tolist()}")
    
    return df

def load_subsidy_data(csv_path):
    """
    Load and validate subsidy data from RS_Session CSV
    
    Args:
        csv_path: Path to RS_Session_265_AU_2154_A_and_B_2.csv
    """
    print(f"\n💰 Loading Subsidy Data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} subsidy records")
        print(f"\n📋 Columns: {df.columns.tolist()}")
        print(f"\n👀 First few rows:")
        print(df.head())
        
        # Validate if it's subsidy data
        # Check for expected columns
        expected_keywords = ['state', 'subsidy', 'vehicle', 'amount', 'policy', 'incentive']
        found_keywords = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            for keyword in expected_keywords:
                if keyword in col_lower:
                    found_keywords.append(keyword)
                    break
        
        if found_keywords:
            print(f"\n✅ Looks like subsidy data! Found keywords: {found_keywords}")
            
            # Save to processed directory
            output_path = PROCESSED_DIR / 'state_ev_subsidies.csv'
            df.to_csv(output_path, index=False)
            print(f"✅ Saved to: {output_path}")
            
            return df
        else:
            print("\n⚠️  Dataset doesn't clearly match subsidy structure")
            print("Will use generated subsidy data as fallback")
            return None
            
    except Exception as e:
        print(f"❌ Error loading subsidy data: {e}")
        return None

def main():
    """
    Main function to download and prepare all datasets
    """
    print("="*60)
    print("🎯 Bharat EV Saathi - Real Data Integration")
    print("="*60)
    
    # Download charging stations
    charging_df = download_charging_stations()
    if charging_df is None:
        print("\n⚠️  Using fallback: Running generate_charging_stations.py...")
        from generate_charging_stations import generate_charging_stations
        charging_df = generate_charging_stations()
        charging_df.to_csv(PROCESSED_DIR / 'india_ev_charging_stations.csv', index=False)
    
    # Download EV specifications
    ev_df = download_ev_specifications()
    if ev_df is None:
        print("\n⚠️  Using fallback: Running generate_indian_ev_data.py...")
        from generate_indian_ev_data import generate_ev_dataset
        ev_df = generate_ev_dataset()
        ev_df.to_csv(PROCESSED_DIR / 'indian_ev_vehicles.csv', index=False)
    
    # Check for subsidy data in project folder
    subsidy_path = Path(__file__).parent.parent.parent / 'RS_Session_265_AU_2154_A_and_B_2.csv'
    subsidy_df = load_subsidy_data(subsidy_path)
    
    if subsidy_df is None:
        print("\n⚠️  Using fallback: Running generate_subsidy_data.py...")
        from generate_subsidy_data import STATE_SUBSIDIES
        import pandas as pd
        subsidy_df = pd.DataFrame(STATE_SUBSIDIES)
        subsidy_df.to_csv(PROCESSED_DIR / 'state_ev_subsidies.csv', index=False)
    
    # Generate sales data (derived from EV data)
    print("\n📊 Generating sales data...")
    from generate_indian_ev_data import generate_sales_data
    sales_df = generate_sales_data(ev_df)
    sales_df.to_csv(PROCESSED_DIR / 'indian_ev_sales.csv', index=False)
    print(f"✅ Sales data created: {len(sales_df)} records")
    
    # Summary
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\n📊 Datasets Created:")
    print(f"  1. EV Vehicles: {len(ev_df)} models")
    if charging_df is not None:
        print(f"  2. Charging Stations: {len(charging_df)} stations")
    print(f"  3. State Subsidies: Available")
    print(f"  4. Sales Data: {len(sales_df)} records")
    print(f"\n💾 All files saved to: {PROCESSED_DIR}")
    print("\n🚀 Ready to run the application!")

if __name__ == '__main__':
    main()
