"""
Indian EV Charging Stations Dataset Generator
==============================================
This script generates a realistic dataset of EV charging stations across major Indian cities.
Data is based on publicly available information about charging networks in India.

Author: Bharat EV Saathi Project
Date: November 2025
"""

import pandas as pd
import random

# Major Indian cities with EV infrastructure
INDIAN_CITIES = {
    'Mumbai': {'lat_range': (19.0, 19.3), 'lon_range': (72.8, 73.0), 'stations': 45},
    'Delhi': {'lat_range': (28.4, 28.9), 'lon_range': (76.8, 77.3), 'stations': 60},
    'Bangalore': {'lat_range': (12.8, 13.1), 'lon_range': (77.4, 77.8), 'stations': 50},
    'Pune': {'lat_range': (18.4, 18.7), 'lon_range': (73.7, 74.0), 'stations': 35},
    'Hyderabad': {'lat_range': (17.3, 17.5), 'lon_range': (78.3, 78.6), 'stations': 40},
    'Chennai': {'lat_range': (12.9, 13.2), 'lon_range': (80.1, 80.3), 'stations': 38},
    'Ahmedabad': {'lat_range': (22.9, 23.2), 'lon_range': (72.4, 72.7), 'stations': 30},
    'Kolkata': {'lat_range': (22.4, 22.7), 'lon_range': (88.2, 88.5), 'stations': 28},
    'Jaipur': {'lat_range': (26.8, 27.0), 'lon_range': (75.7, 75.9), 'stations': 25},
    'Surat': {'lat_range': (21.1, 21.3), 'lon_range': (72.7, 72.9), 'stations': 22},
    'Chandigarh': {'lat_range': (30.6, 30.8), 'lon_range': (76.7, 76.9), 'stations': 20},
    'Lucknow': {'lat_range': (26.7, 26.9), 'lon_range': (80.8, 81.1), 'stations': 18},
    'Kochi': {'lat_range': (9.9, 10.1), 'lon_range': (76.2, 76.4), 'stations': 20},
    'Indore': {'lat_range': (22.6, 22.8), 'lon_range': (75.7, 75.9), 'stations': 15},
    'Bhopal': {'lat_range': (23.1, 23.3), 'lon_range': (77.3, 77.5), 'stations': 12},
}

# Real charging networks operating in India
CHARGING_NETWORKS = [
    'Tata Power EZ Charge',
    'Ather Grid',
    'Fortum Charge & Drive',
    'ChargeZone',
    'Exicom',
    'Magenta Power',
    'Statiq',
    'Kazam EV',
    'Revolt Motors',
    'BPCL',
    'IOCL',
    'HPCL',
    'Reliance BP',
    'Ola Electric',
    'BluSmart',
]

# Types of charging stations
CHARGER_TYPES = ['AC Type 2', 'CCS2', 'CHAdeMO', 'Bharat AC-001', 'Bharat DC-001']

# Common locations for charging stations
LOCATION_TYPES = [
    'Shopping Mall', 'Metro Station', 'Petrol Pump', 'Restaurant', 
    'Hotel', 'Hospital', 'Airport', 'Office Complex', 'Tech Park',
    'Residential Complex', 'Highway Rest Stop', 'Tourist Spot'
]

def generate_station_name(city, location_type, index):
    """Generate realistic charging station names"""
    return f"{city} {location_type} - {index}"

def generate_charging_stations():
    """
    Generate a comprehensive dataset of EV charging stations across India
    
    Returns:
        DataFrame with station details including location, type, network, etc.
    """
    stations_data = []
    station_id = 1000  # Starting ID
    
    for city, info in INDIAN_CITIES.items():
        num_stations = info['stations']
        lat_min, lat_max = info['lat_range']
        lon_min, lon_max = info['lon_range']
        
        for i in range(num_stations):
            # Generate random but realistic coordinates within city bounds
            latitude = round(random.uniform(lat_min, lat_max), 6)
            longitude = round(random.uniform(lon_min, lon_max), 6)
            
            # Select random attributes
            network = random.choice(CHARGING_NETWORKS)
            charger_type = random.choice(CHARGER_TYPES)
            location_type = random.choice(LOCATION_TYPES)
            
            # Determine number of charging points (1-8 per station)
            num_chargers = random.choices([1, 2, 3, 4, 6, 8], weights=[20, 35, 25, 10, 7, 3])[0]
            
            # Power output (kW) - based on charger type
            if charger_type in ['CCS2', 'CHAdeMO', 'Bharat DC-001']:
                power_kw = random.choice([25, 50, 60, 100, 150])  # DC fast charging
            else:
                power_kw = random.choice([3.3, 7.4, 11, 22])  # AC charging
            
            # Operating hours
            operating_hours = random.choices(
                ['24/7', '6 AM - 11 PM', '8 AM - 10 PM', '9 AM - 9 PM'],
                weights=[30, 40, 20, 10]
            )[0]
            
            # Payment methods
            payment_methods = 'App, Card, UPI'
            
            # Amenities
            amenities_list = []
            if random.random() > 0.5:
                amenities_list.append('WiFi')
            if random.random() > 0.6:
                amenities_list.append('Cafe')
            if random.random() > 0.7:
                amenities_list.append('Restroom')
            if random.random() > 0.8:
                amenities_list.append('Waiting Area')
            amenities = ', '.join(amenities_list) if amenities_list else 'None'
            
            # Status
            status = random.choices(['Operational', 'Operational', 'Operational', 'Under Maintenance'], 
                                   weights=[85, 10, 4, 1])[0]
            
            # Create station record
            station = {
                'station_id': f'EV{station_id}',
                'station_name': generate_station_name(city, location_type, i+1),
                'network': network,
                'city': city,
                'state': get_state(city),
                'latitude': latitude,
                'longitude': longitude,
                'location_type': location_type,
                'charger_type': charger_type,
                'power_kw': power_kw,
                'num_chargers': num_chargers,
                'operating_hours': operating_hours,
                'payment_methods': payment_methods,
                'amenities': amenities,
                'status': status,
                'cost_per_unit': round(random.uniform(8, 15), 2),  # ₹ per kWh
                'parking_fee': random.choice([0, 20, 30, 50]),  # ₹
            }
            
            stations_data.append(station)
            station_id += 1
    
    return pd.DataFrame(stations_data)

def get_state(city):
    """Map city to state"""
    city_state_map = {
        'Mumbai': 'Maharashtra',
        'Pune': 'Maharashtra',
        'Delhi': 'Delhi',
        'Bangalore': 'Karnataka',
        'Hyderabad': 'Telangana',
        'Chennai': 'Tamil Nadu',
        'Ahmedabad': 'Gujarat',
        'Surat': 'Gujarat',
        'Kolkata': 'West Bengal',
        'Jaipur': 'Rajasthan',
        'Chandigarh': 'Chandigarh',
        'Lucknow': 'Uttar Pradesh',
        'Kochi': 'Kerala',
        'Indore': 'Madhya Pradesh',
        'Bhopal': 'Madhya Pradesh',
    }
    return city_state_map.get(city, 'Unknown')

if __name__ == '__main__':
    print("🔌 Generating Indian EV Charging Stations Dataset...")
    
    # Generate the dataset
    df = generate_charging_stations()
    
    # Save to CSV
    output_path = 'india_ev_charging_stations.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset created successfully!")
    print(f"📊 Total stations: {len(df)}")
    print(f"🏙️ Cities covered: {df['city'].nunique()}")
    print(f"🔌 Networks: {df['network'].nunique()}")
    print(f"💾 Saved to: {output_path}")
    
    # Display summary
    print("\n📈 Summary by City:")
    print(df['city'].value_counts().head(10))
    
    print("\n🔌 Summary by Network:")
    print(df['network'].value_counts().head(5))
