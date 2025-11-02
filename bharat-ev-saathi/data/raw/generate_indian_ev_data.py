"""
Indian Electric Vehicles Dataset Generator
==========================================
This script creates a comprehensive dataset of EVs available in the Indian market.
Includes popular models with accurate specifications based on real data.

Author: Bharat EV Saathi Project
Date: November 2025
"""

import pandas as pd
import random

# Real Indian EV Market Data (as of 2025) - EXPANDED TO 50+ MODELS
INDIAN_EVS = [
    # Two-Wheelers - Premium Segment
    {'brand': 'Ather', 'model': '450X', 'type': '2-Wheeler', 'price_inr': 145000, 'range_km': 105, 'battery_kwh': 3.7, 'top_speed': 90, 'charging_time': 5.5, 'segment': 'Premium Scooter'},
    {'brand': 'Ather', 'model': '450S', 'type': '2-Wheeler', 'price_inr': 130000, 'range_km': 90, 'battery_kwh': 3.0, 'top_speed': 90, 'charging_time': 4.5, 'segment': 'Premium Scooter'},
    {'brand': 'Ather', 'model': '450 Apex', 'type': '2-Wheeler', 'price_inr': 195000, 'range_km': 115, 'battery_kwh': 3.7, 'top_speed': 100, 'charging_time': 5.5, 'segment': 'Premium Scooter'},
    {'brand': 'Ola Electric', 'model': 'S1 Pro', 'type': '2-Wheeler', 'price_inr': 139999, 'range_km': 135, 'battery_kwh': 3.97, 'top_speed': 115, 'charging_time': 6.5, 'segment': 'Premium Scooter'},
    {'brand': 'Ola Electric', 'model': 'S1 Air', 'type': '2-Wheeler', 'price_inr': 104999, 'range_km': 101, 'battery_kwh': 2.98, 'top_speed': 85, 'charging_time': 5.0, 'segment': 'Mid-Range Scooter'},
    {'brand': 'Ola Electric', 'model': 'S1 X', 'type': '2-Wheeler', 'price_inr': 89999, 'range_km': 91, 'battery_kwh': 2.5, 'top_speed': 85, 'charging_time': 4.5, 'segment': 'Budget Scooter'},
    
    # Two-Wheelers - Mid-Range Segment
    {'brand': 'TVS', 'model': 'iQube', 'type': '2-Wheeler', 'price_inr': 112000, 'range_km': 100, 'battery_kwh': 3.4, 'top_speed': 78, 'charging_time': 4.5, 'segment': 'Mid-Range Scooter'},
    {'brand': 'TVS', 'model': 'iQube ST', 'type': '2-Wheeler', 'price_inr': 135000, 'range_km': 125, 'battery_kwh': 4.0, 'top_speed': 82, 'charging_time': 5.0, 'segment': 'Mid-Range Scooter'},
    {'brand': 'Bajaj', 'model': 'Chetak', 'type': '2-Wheeler', 'price_inr': 125000, 'range_km': 95, 'battery_kwh': 3.0, 'top_speed': 73, 'charging_time': 5.0, 'segment': 'Mid-Range Scooter'},
    {'brand': 'Bajaj', 'model': 'Chetak Premium', 'type': '2-Wheeler', 'price_inr': 145000, 'range_km': 108, 'battery_kwh': 3.5, 'top_speed': 77, 'charging_time': 5.5, 'segment': 'Mid-Range Scooter'},
    {'brand': 'Simple Energy', 'model': 'One', 'type': '2-Wheeler', 'price_inr': 110000, 'range_km': 236, 'battery_kwh': 4.8, 'top_speed': 105, 'charging_time': 4.0, 'segment': 'Mid-Range Scooter'},
    {'brand': 'Bounce', 'model': 'Infinity E1', 'type': '2-Wheeler', 'price_inr': 68000, 'range_km': 85, 'battery_kwh': 1.9, 'top_speed': 65, 'charging_time': 3.5, 'segment': 'Budget Scooter'},
    
    # Two-Wheelers - Budget Segment
    {'brand': 'Hero Electric', 'model': 'Optima', 'type': '2-Wheeler', 'price_inr': 75000, 'range_km': 80, 'battery_kwh': 1.8, 'top_speed': 45, 'charging_time': 4.0, 'segment': 'Budget Scooter'},
    {'brand': 'Hero Electric', 'model': 'Photon', 'type': '2-Wheeler', 'price_inr': 82000, 'range_km': 90, 'battery_kwh': 2.1, 'top_speed': 50, 'charging_time': 4.5, 'segment': 'Budget Scooter'},
    {'brand': 'Hero Electric', 'model': 'Nyx', 'type': '2-Wheeler', 'price_inr': 95000, 'range_km': 100, 'battery_kwh': 2.4, 'top_speed': 60, 'charging_time': 5.0, 'segment': 'Budget Scooter'},
    {'brand': 'Ampere', 'model': 'Magnus Pro', 'type': '2-Wheeler', 'price_inr': 85000, 'range_km': 100, 'battery_kwh': 2.3, 'top_speed': 55, 'charging_time': 5.0, 'segment': 'Budget Scooter'},
    {'brand': 'Ampere', 'model': 'Primus', 'type': '2-Wheeler', 'price_inr': 95000, 'range_km': 120, 'battery_kwh': 2.9, 'top_speed': 55, 'charging_time': 5.5, 'segment': 'Budget Scooter'},
    {'brand': 'Okinawa', 'model': 'iPraise', 'type': '2-Wheeler', 'price_inr': 110000, 'range_km': 139, 'battery_kwh': 3.3, 'top_speed': 58, 'charging_time': 5.0, 'segment': 'Mid-Range Scooter'},
    {'brand': 'Okinawa', 'model': 'PraisePro', 'type': '2-Wheeler', 'price_inr': 88000, 'range_km': 110, 'battery_kwh': 2.7, 'top_speed': 52, 'charging_time': 4.5, 'segment': 'Budget Scooter'},
    {'brand': 'Revolt', 'model': 'RV400', 'type': '2-Wheeler', 'price_inr': 135000, 'range_km': 150, 'battery_kwh': 3.24, 'top_speed': 85, 'charging_time': 4.5, 'segment': 'Premium Scooter'},
    {'brand': 'Pure EV', 'model': 'ePluto 7G', 'type': '2-Wheeler', 'price_inr': 95000, 'range_km': 116, 'battery_kwh': 2.9, 'top_speed': 60, 'charging_time': 4.0, 'segment': 'Budget Scooter'},
    
    # Four-Wheelers - Budget/Compact Segment
    {'brand': 'Tata', 'model': 'Tiago EV', 'type': '4-Wheeler', 'price_inr': 849000, 'range_km': 315, 'battery_kwh': 24, 'top_speed': 110, 'charging_time': 8.0, 'segment': 'Compact Hatchback'},
    {'brand': 'Tata', 'model': 'Tigor EV', 'type': '4-Wheeler', 'price_inr': 1249000, 'range_km': 315, 'battery_kwh': 26, 'top_speed': 120, 'charging_time': 8.5, 'segment': 'Compact Sedan'},
    {'brand': 'Tata', 'model': 'Punch EV', 'type': '4-Wheeler', 'price_inr': 1099000, 'range_km': 365, 'battery_kwh': 28.5, 'top_speed': 125, 'charging_time': 9.0, 'segment': 'Micro SUV'},
    {'brand': 'Citroen', 'model': 'eC3', 'type': '4-Wheeler', 'price_inr': 1199000, 'range_km': 320, 'battery_kwh': 29.2, 'top_speed': 107, 'charging_time': 10.5, 'segment': 'Compact Hatchback'},
    {'brand': 'MG', 'model': 'Comet EV', 'type': '4-Wheeler', 'price_inr': 799000, 'range_km': 230, 'battery_kwh': 17.3, 'top_speed': 105, 'charging_time': 7.0, 'segment': 'Micro Car'},
    {'brand': 'Mahindra', 'model': 'e2o', 'type': '4-Wheeler', 'price_inr': 650000, 'range_km': 140, 'battery_kwh': 15, 'top_speed': 85, 'charging_time': 6.0, 'segment': 'Micro Car'},
    
    # Four-Wheelers - SUV Segment
    {'brand': 'Tata', 'model': 'Nexon EV', 'type': '4-Wheeler', 'price_inr': 1499000, 'range_km': 325, 'battery_kwh': 30.2, 'top_speed': 120, 'charging_time': 8.5, 'segment': 'Compact SUV'},
    {'brand': 'Tata', 'model': 'Nexon EV Max', 'type': '4-Wheeler', 'price_inr': 1799000, 'range_km': 437, 'battery_kwh': 40.5, 'top_speed': 140, 'charging_time': 15.0, 'segment': 'Compact SUV'},
    {'brand': 'Tata', 'model': 'Curvv EV', 'type': '4-Wheeler', 'price_inr': 1899000, 'range_km': 500, 'battery_kwh': 45, 'top_speed': 150, 'charging_time': 16.0, 'segment': 'Mid-Size SUV'},
    {'brand': 'Mahindra', 'model': 'XUV400 EV', 'type': '4-Wheeler', 'price_inr': 1699000, 'range_km': 456, 'battery_kwh': 39.4, 'top_speed': 150, 'charging_time': 13.0, 'segment': 'Compact SUV'},
    {'brand': 'Mahindra', 'model': 'XUV700 EV', 'type': '4-Wheeler', 'price_inr': 2499000, 'range_km': 550, 'battery_kwh': 60, 'top_speed': 160, 'charging_time': 18.0, 'segment': 'Mid-Size SUV'},
    {'brand': 'Mahindra', 'model': 'BE 6e', 'type': '4-Wheeler', 'price_inr': 2299000, 'range_km': 535, 'battery_kwh': 59, 'top_speed': 155, 'charging_time': 17.0, 'segment': 'Mid-Size SUV'},
    {'brand': 'MG', 'model': 'ZS EV', 'type': '4-Wheeler', 'price_inr': 2199000, 'range_km': 461, 'battery_kwh': 50.3, 'top_speed': 140, 'charging_time': 15.0, 'segment': 'Mid-Size SUV'},
    {'brand': 'MG', 'model': 'Windsor EV', 'type': '4-Wheeler', 'price_inr': 1499000, 'range_km': 331, 'battery_kwh': 38, 'top_speed': 130, 'charging_time': 12.0, 'segment': 'Compact SUV'},
    {'brand': 'BYD', 'model': 'Atto 3', 'type': '4-Wheeler', 'price_inr': 3399000, 'range_km': 521, 'battery_kwh': 60.48, 'top_speed': 160, 'charging_time': 16.0, 'segment': 'Mid-Size SUV'},
    {'brand': 'BYD', 'model': 'e6', 'type': '4-Wheeler', 'price_inr': 2999000, 'range_km': 415, 'battery_kwh': 71.7, 'top_speed': 130, 'charging_time': 19.0, 'segment': 'MPV'},
    {'brand': 'Hyundai', 'model': 'Kona Electric', 'type': '4-Wheeler', 'price_inr': 2399000, 'range_km': 452, 'battery_kwh': 39.2, 'top_speed': 155, 'charging_time': 9.5, 'segment': 'Mid-Size SUV'},
    {'brand': 'Hyundai', 'model': 'Ioniq 5', 'type': '4-Wheeler', 'price_inr': 4599000, 'range_km': 631, 'battery_kwh': 72.6, 'top_speed': 185, 'charging_time': 18.0, 'segment': 'Premium SUV'},
    {'brand': 'Kia', 'model': 'EV6', 'type': '4-Wheeler', 'price_inr': 6099000, 'range_km': 708, 'battery_kwh': 77.4, 'top_speed': 185, 'charging_time': 18.0, 'segment': 'Premium SUV'},
    {'brand': 'Volvo', 'model': 'XC40 Recharge', 'type': '4-Wheeler', 'price_inr': 6095000, 'range_km': 418, 'battery_kwh': 78, 'top_speed': 180, 'charging_time': 19.0, 'segment': 'Luxury SUV'},
    
    # Four-Wheelers - Luxury Segment
    {'brand': 'Mercedes-Benz', 'model': 'EQS', 'type': '4-Wheeler', 'price_inr': 15500000, 'range_km': 677, 'battery_kwh': 107.8, 'top_speed': 210, 'charging_time': 31.0, 'segment': 'Luxury Sedan'},
    {'brand': 'Mercedes-Benz', 'model': 'EQB', 'type': '4-Wheeler', 'price_inr': 7899000, 'range_km': 423, 'battery_kwh': 66.5, 'top_speed': 160, 'charging_time': 19.0, 'segment': 'Luxury SUV'},
    {'brand': 'Mercedes-Benz', 'model': 'EQE', 'type': '4-Wheeler', 'price_inr': 13990000, 'range_km': 550, 'battery_kwh': 90.6, 'top_speed': 210, 'charging_time': 25.0, 'segment': 'Luxury Sedan'},
    {'brand': 'BMW', 'model': 'iX', 'type': '4-Wheeler', 'price_inr': 12500000, 'range_km': 635, 'battery_kwh': 111.5, 'top_speed': 200, 'charging_time': 35.0, 'segment': 'Luxury SUV'},
    {'brand': 'BMW', 'model': 'i4', 'type': '4-Wheeler', 'price_inr': 7990000, 'range_km': 590, 'battery_kwh': 83.9, 'top_speed': 225, 'charging_time': 22.0, 'segment': 'Luxury Sedan'},
    {'brand': 'BMW', 'model': 'i7', 'type': '4-Wheeler', 'price_inr': 21000000, 'range_km': 625, 'battery_kwh': 101.7, 'top_speed': 240, 'charging_time': 28.0, 'segment': 'Luxury Sedan'},
    {'brand': 'Audi', 'model': 'e-tron', 'type': '4-Wheeler', 'price_inr': 10200000, 'range_km': 484, 'battery_kwh': 95, 'top_speed': 200, 'charging_time': 29.0, 'segment': 'Luxury SUV'},
    {'brand': 'Audi', 'model': 'e-tron GT', 'type': '4-Wheeler', 'price_inr': 17900000, 'range_km': 488, 'battery_kwh': 93.4, 'top_speed': 245, 'charging_time': 22.0, 'segment': 'Luxury Sports'},
    {'brand': 'Porsche', 'model': 'Taycan', 'type': '4-Wheeler', 'price_inr': 18500000, 'range_km': 484, 'battery_kwh': 93.4, 'top_speed': 260, 'charging_time': 22.0, 'segment': 'Luxury Sports'},
    {'brand': 'Jaguar', 'model': 'I-Pace', 'type': '4-Wheeler', 'price_inr': 11800000, 'range_km': 470, 'battery_kwh': 90, 'top_speed': 200, 'charging_time': 25.0, 'segment': 'Luxury SUV'},
    
    # Three-Wheelers (Auto-rickshaw)
    {'brand': 'Mahindra', 'model': 'Treo', 'type': '3-Wheeler', 'price_inr': 245000, 'range_km': 130, 'battery_kwh': 7.37, 'top_speed': 55, 'charging_time': 3.5, 'segment': 'Passenger Auto'},
    {'brand': 'Mahindra', 'model': 'Treo Zor', 'type': '3-Wheeler', 'price_inr': 285000, 'range_km': 125, 'battery_kwh': 8.0, 'top_speed': 50, 'charging_time': 3.8, 'segment': 'Cargo Auto'},
    {'brand': 'Piaggio', 'model': 'Ape E-Xtra', 'type': '3-Wheeler', 'price_inr': 275000, 'range_km': 110, 'battery_kwh': 8.1, 'top_speed': 45, 'charging_time': 4.0, 'segment': 'Cargo Auto'},
    {'brand': 'Piaggio', 'model': 'Ape E-City', 'type': '3-Wheeler', 'price_inr': 295000, 'range_km': 120, 'battery_kwh': 8.5, 'top_speed': 50, 'charging_time': 4.2, 'segment': 'Passenger Auto'},
    {'brand': 'Euler Motors', 'model': 'HiLoad', 'type': '3-Wheeler', 'price_inr': 325000, 'range_km': 150, 'battery_kwh': 12.6, 'top_speed': 50, 'charging_time': 4.5, 'segment': 'Cargo Auto'},
    {'brand': 'Omega Seiki', 'model': 'Rage+', 'type': '3-Wheeler', 'price_inr': 315000, 'range_km': 140, 'battery_kwh': 10.9, 'top_speed': 55, 'charging_time': 4.0, 'segment': 'Cargo Auto'},
    {'brand': 'Altigreen', 'model': 'neEV', 'type': '3-Wheeler', 'price_inr': 580000, 'range_km': 170, 'battery_kwh': 16.2, 'top_speed': 60, 'charging_time': 5.0, 'segment': 'Cargo Auto'},
]

def generate_ev_dataset():
    """
    Generate comprehensive Indian EV dataset with additional calculated fields
    
    Returns:
        DataFrame with vehicle specifications and calculated metrics
    """
    df = pd.DataFrame(INDIAN_EVS)
    
    # Add calculated fields
    df['year'] = 2024  # Current model year
    
    # Calculate efficiency (km per kWh)
    df['efficiency_km_per_kwh'] = (df['range_km'] / df['battery_kwh']).round(2)
    
    # Estimate acceleration (0-60 km/h in seconds) - simplified formula
    df['acceleration_0_60'] = df.apply(lambda x: 
        round(3.5 + (x['battery_kwh'] / 10) - (x['type'] == '2-Wheeler') * 1.5, 1), axis=1)
    
    # FAME-II subsidy eligibility
    df['fame_eligible'] = df.apply(lambda x:
        'Yes' if (x['type'] == '2-Wheeler' and x['price_inr'] <= 150000) or
                 (x['type'] == '4-Wheeler' and x['price_inr'] <= 1500000) or
                 (x['type'] == '3-Wheeler')
        else 'No', axis=1)
    
    # Calculate central subsidy amount (simplified FAME-II rules)
    def calculate_fame_subsidy(row):
        if row['fame_eligible'] == 'No':
            return 0
        if row['type'] == '2-Wheeler':
            return min(15000, row['battery_kwh'] * 15000)  # ₹15,000 per kWh, max ₹15,000
        elif row['type'] == '3-Wheeler':
            return min(50000, row['battery_kwh'] * 15000)  # Max ₹50,000
        elif row['type'] == '4-Wheeler':
            return 0  # FAME-II primarily for 2W/3W
        return 0
    
    df['central_subsidy_inr'] = df.apply(calculate_fame_subsidy, axis=1).astype(int)
    
    # Effective price after central subsidy
    df['price_after_subsidy_inr'] = df['price_inr'] - df['central_subsidy_inr']
    
    # Seating capacity
    seating_map = {'2-Wheeler': 2, '3-Wheeler': 4, '4-Wheeler': 5}
    df['seating_capacity'] = df['type'].map(seating_map)
    
    # Warranty (years)
    df['battery_warranty_years'] = df.apply(lambda x: 
        8 if x['type'] == '4-Wheeler' else 3, axis=1)
    
    # Available in India
    df['available_in_india'] = 'Yes'
    
    # Popular states (randomly assign for demo)
    states = ['Maharashtra', 'Karnataka', 'Delhi', 'Tamil Nadu', 'Gujarat', 'Telangana']
    df['popular_in_states'] = [', '.join(random.sample(states, 3)) for _ in range(len(df))]
    
    # Sales rank (approximate)
    df['sales_rank'] = range(1, len(df) + 1)
    
    # User rating (out of 5)
    df['user_rating'] = [round(random.uniform(3.5, 4.9), 1) for _ in range(len(df))]
    
    # Add maintenance cost per year (estimated)
    df['annual_maintenance_inr'] = df.apply(lambda x:
        8000 if x['type'] == '2-Wheeler' else
        12000 if x['type'] == '3-Wheeler' else
        25000 if x['price_inr'] < 2000000 else
        50000, axis=1)
    
    return df

def generate_sales_data(ev_df):
    """
    Generate monthly sales data for Indian EVs (2023-2024)
    
    Returns:
        DataFrame with sales records by model and month
    """
    sales_data = []
    months = pd.date_range('2023-01', '2024-10', freq='M')
    
    for _, ev in ev_df.iterrows():
        # Base sales volume (higher for cheaper models)
        if ev['type'] == '2-Wheeler':
            base_sales = random.randint(500, 3000)
        elif ev['type'] == '3-Wheeler':
            base_sales = random.randint(100, 500)
        else:  # 4-Wheeler
            if ev['price_inr'] < 1500000:
                base_sales = random.randint(200, 800)
            else:
                base_sales = random.randint(10, 100)
        
        for month in months:
            # Add some variation and growth trend
            growth_factor = 1 + (month.month / 100)  # Slight upward trend
            seasonal_factor = 1 + 0.2 * random.random() - 0.1  # Random variation
            sales = int(base_sales * growth_factor * seasonal_factor)
            
            sales_data.append({
                'month': month.strftime('%Y-%m'),
                'brand': ev['brand'],
                'model': ev['model'],
                'type': ev['type'],
                'units_sold': max(0, sales),
                'state': random.choice(['Maharashtra', 'Karnataka', 'Delhi', 'Tamil Nadu', 'Gujarat'])
            })
    
    return pd.DataFrame(sales_data)

if __name__ == '__main__':
    print("🚗 Generating Indian EV Market Dataset...")
    
    # Generate EV specifications dataset
    ev_df = generate_ev_dataset()
    ev_df.to_csv('indian_ev_vehicles.csv', index=False)
    print(f"✅ EV Vehicles dataset created: {len(ev_df)} models")
    
    # Generate sales data
    sales_df = generate_sales_data(ev_df)
    sales_df.to_csv('indian_ev_sales.csv', index=False)
    print(f"✅ Sales data created: {len(sales_df)} records")
    
    print("\n📊 Dataset Summary:")
    print(f"Total EV Models: {len(ev_df)}")
    print(f"2-Wheelers: {len(ev_df[ev_df['type'] == '2-Wheeler'])}")
    print(f"3-Wheelers: {len(ev_df[ev_df['type'] == '3-Wheeler'])}")
    print(f"4-Wheelers: {len(ev_df[ev_df['type'] == '4-Wheeler'])}")
    print(f"\nPrice Range: ₹{ev_df['price_inr'].min():,} - ₹{ev_df['price_inr'].max():,}")
    print(f"Range: {ev_df['range_km'].min()} - {ev_df['range_km'].max()} km")
    
    print("\n🔝 Top 5 EVs by Range:")
    print(ev_df.nlargest(5, 'range_km')[['brand', 'model', 'range_km', 'price_inr']])
