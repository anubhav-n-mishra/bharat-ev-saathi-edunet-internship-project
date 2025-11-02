"""
Data Loader Module
==================
Handles loading and preprocessing of all datasets for Bharat EV Saathi.
Provides clean interfaces for accessing EV data, charging stations, and subsidies.

Author: Bharat EV Saathi Project
"""

import pandas as pd
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

class EVDataLoader:
    """
    Central data loader class for all EV-related datasets.
    Implements caching for better performance.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data loader
        
        Args:
            data_dir: Path to processed data directory (optional)
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._cache = {}  # Simple caching mechanism
        logger.info(f"Data loader initialized with directory: {self.data_dir}")
    
    def _load_csv(self, filename, use_cache=True):
        """
        Load CSV file with caching
        
        Args:
            filename: Name of CSV file
            use_cache: Whether to use cached data
            
        Returns:
            pandas DataFrame
        """
        if use_cache and filename in self._cache:
            logger.debug(f"Using cached data for {filename}")
            return self._cache[filename].copy()
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Dataset {filename} not found in {self.data_dir}")
        
        try:
            df = pd.read_csv(filepath)
            self._cache[filename] = df.copy()
            logger.info(f"Loaded {filename}: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    
    def load_ev_vehicles(self):
        """
        Load Indian EV vehicles dataset
        
        Returns:
            DataFrame with EV specifications and details
        """
        df = self._load_csv('indian_ev_vehicles.csv')
        
        # Ensure correct data types
        df['price_inr'] = pd.to_numeric(df['price_inr'], errors='coerce')
        df['range_km'] = pd.to_numeric(df['range_km'], errors='coerce')
        df['battery_kwh'] = pd.to_numeric(df['battery_kwh'], errors='coerce')
        
        return df
    
    def load_charging_stations(self):
        """
        Load EV charging stations dataset
        
        Returns:
            DataFrame with charging station details
        """
        df = self._load_csv('india_ev_charging_stations.csv')
        
        # Ensure correct data types
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['power_kw'] = pd.to_numeric(df['power_kw'], errors='coerce')
        
        return df
    
    def load_sales_data(self):
        """
        Load EV sales data
        
        Returns:
            DataFrame with monthly sales records
        """
        df = self._load_csv('indian_ev_sales.csv')
        
        # Convert month to datetime
        df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
        
        return df
    
    def load_state_subsidies(self):
        """
        Load state-wise EV subsidies
        
        Returns:
            DataFrame with state subsidy policies
        """
        return self._load_csv('state_ev_subsidies.csv')
    
    def load_fame_subsidies(self):
        """
        Load FAME-II subsidy rules
        
        Returns:
            DataFrame with central subsidy details
        """
        return self._load_csv('fame_ii_subsidy.csv')
    
    def get_ev_by_id(self, brand, model):
        """
        Get specific EV details by brand and model
        
        Args:
            brand: EV brand name
            model: EV model name
            
        Returns:
            Series with EV details or None if not found
        """
        df = self.load_ev_vehicles()
        result = df[(df['brand'] == brand) & (df['model'] == model)]
        
        if len(result) == 0:
            logger.warning(f"EV not found: {brand} {model}")
            return None
        
        return result.iloc[0]
    
    def get_evs_by_type(self, vehicle_type):
        """
        Get all EVs of a specific type
        
        Args:
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            
        Returns:
            DataFrame with filtered EVs
        """
        df = self.load_ev_vehicles()
        return df[df['type'] == vehicle_type].copy()
    
    def get_evs_by_price_range(self, min_price, max_price):
        """
        Get EVs within a price range
        
        Args:
            min_price: Minimum price in INR
            max_price: Maximum price in INR
            
        Returns:
            DataFrame with filtered EVs
        """
        df = self.load_ev_vehicles()
        return df[(df['price_inr'] >= min_price) & 
                  (df['price_inr'] <= max_price)].copy()
    
    def get_charging_stations_by_city(self, city):
        """
        Get all charging stations in a specific city
        
        Args:
            city: City name
            
        Returns:
            DataFrame with filtered charging stations
        """
        df = self.load_charging_stations()
        return df[df['city'] == city].copy()
    
    def get_state_subsidy(self, state, vehicle_type):
        """
        Get subsidy information for a specific state and vehicle type
        
        Args:
            state: Indian state name
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            
        Returns:
            Series with subsidy details or None
        """
        df = self.load_state_subsidies()
        result = df[(df['state'] == state) & (df['vehicle_type'] == vehicle_type)]
        
        if len(result) == 0:
            logger.warning(f"No subsidy data found for {state}, {vehicle_type}")
            return None
        
        return result.iloc[0]
    
    def get_all_states_with_subsidies(self):
        """
        Get list of all states offering EV subsidies
        
        Returns:
            List of state names
        """
        df = self.load_state_subsidies()
        return sorted(df['state'].unique().tolist())
    
    def get_all_cities_with_stations(self):
        """
        Get list of all cities with charging stations
        
        Returns:
            List of city names
        """
        df = self.load_charging_stations()
        return sorted(df['city'].unique().tolist())
    
    def get_dataset_stats(self):
        """
        Get statistics about all datasets
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_evs': len(self.load_ev_vehicles()),
            'total_charging_stations': len(self.load_charging_stations()),
            'total_sales_records': len(self.load_sales_data()),
            'states_with_subsidies': len(self.get_all_states_with_subsidies()),
            'cities_with_stations': len(self.get_all_cities_with_stations()),
        }
        
        # EV breakdown
        ev_df = self.load_ev_vehicles()
        stats['evs_by_type'] = ev_df['type'].value_counts().to_dict()
        
        # Price ranges
        stats['price_range'] = {
            'min': int(ev_df['price_inr'].min()),
            'max': int(ev_df['price_inr'].max()),
            'average': int(ev_df['price_inr'].mean())
        }
        
        return stats
    
    def clear_cache(self):
        """Clear the data cache"""
        self._cache = {}
        logger.info("Cache cleared")


# Create global instance for easy importing
ev_data = EVDataLoader()


if __name__ == '__main__':
    # Test the data loader
    print("🧪 Testing Data Loader...")
    
    loader = EVDataLoader()
    
    # Test loading all datasets
    print("\n📊 Loading datasets...")
    vehicles = loader.load_ev_vehicles()
    print(f"✅ Vehicles: {len(vehicles)} records")
    
    stations = loader.load_charging_stations()
    print(f"✅ Charging Stations: {len(stations)} records")
    
    sales = loader.load_sales_data()
    print(f"✅ Sales Data: {len(sales)} records")
    
    # Test specific queries
    print("\n🔍 Testing queries...")
    two_wheelers = loader.get_evs_by_type('2-Wheeler')
    print(f"✅ 2-Wheelers: {len(two_wheelers)} models")
    
    mumbai_stations = loader.get_charging_stations_by_city('Mumbai')
    print(f"✅ Mumbai Stations: {len(mumbai_stations)} stations")
    
    # Get stats
    print("\n📈 Dataset Statistics:")
    stats = loader.get_dataset_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ All tests passed!")
