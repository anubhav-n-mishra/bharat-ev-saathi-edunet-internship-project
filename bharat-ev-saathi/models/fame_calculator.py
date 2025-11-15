"""
FAME-II Subsidy Calculator
===========================
Calculate FAME-II and state subsidies for EV purchases
"""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FAMESubsidyCalculator:
    def __init__(self):
        self.fame_data = None
        self.state_data = None
        self.load_data()
    
    def load_data(self):
        """Load FAME and state subsidy data"""
        try:
            project_root = Path(__file__).parent.parent
            
            # Load FAME-II data
            self.fame_data = pd.read_csv(project_root / 'data' / 'processed' / 'fame_ii_subsidy.csv')
            logger.info("✅ FAME-II subsidy data loaded")
            
            # Load state subsidy data
            self.state_data = pd.read_csv(project_root / 'data' / 'processed' / 'state_ev_subsidies.csv')
            logger.info("✅ State subsidy data loaded")
            
        except Exception as e:
            logger.error(f"Error loading subsidy data: {e}")
    
    def calculate_fame_subsidy(self, vehicle_type, battery_capacity_kwh=None, ex_showroom_price=None):
        """
        Calculate FAME-II subsidy
        
        Args:
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            battery_capacity_kwh: Battery capacity in kWh
            ex_showroom_price: Ex-showroom price in rupees
            
        Returns:
            dict with subsidy details
        """
        if self.fame_data is None:
            return {"error": "FAME data not loaded"}
        
        try:
            # Get FAME-II details for vehicle type
            fame_row = self.fame_data[self.fame_data['vehicle_type'] == vehicle_type]
            
            if fame_row.empty:
                return {
                    "eligible": False,
                    "reason": f"No FAME-II subsidy available for {vehicle_type}"
                }
            
            fame_row = fame_row.iloc[0]
            subsidy_per_kwh = fame_row['subsidy_per_kwh']
            max_subsidy = fame_row['max_subsidy']
            conditions = fame_row['conditions']
            valid_until = fame_row['valid_until']
            
            # Calculate subsidy
            if vehicle_type == '2-Wheeler':
                # For 2-wheelers: Fixed subsidy if conditions met
                if battery_capacity_kwh and battery_capacity_kwh >= 2:
                    if ex_showroom_price and ex_showroom_price <= 150000:
                        fame_subsidy = min(subsidy_per_kwh, max_subsidy)
                        eligible = True
                    else:
                        fame_subsidy = 0
                        eligible = False
                        conditions = "Ex-showroom price must be ≤ ₹1.5 lakh"
                else:
                    fame_subsidy = 0
                    eligible = False
                    conditions = "Battery capacity must be ≥ 2 kWh"
            
            elif vehicle_type == '3-Wheeler':
                # For 3-wheelers: ₹10,000 per kWh up to max
                if battery_capacity_kwh:
                    fame_subsidy = min(battery_capacity_kwh * subsidy_per_kwh, max_subsidy)
                    eligible = True
                else:
                    fame_subsidy = 0
                    eligible = False
                    conditions = "Battery capacity required"
            
            else:  # 4-Wheeler
                # For 4-wheelers: ₹10,000 per kWh up to ₹1.5 lakh (minimal for personal use)
                if battery_capacity_kwh:
                    if ex_showroom_price and ex_showroom_price <= 1500000:
                        fame_subsidy = min(battery_capacity_kwh * subsidy_per_kwh, max_subsidy)
                        eligible = True
                    else:
                        fame_subsidy = 0
                        eligible = False
                        conditions = "Ex-showroom price must be ≤ ₹15 lakh"
                else:
                    fame_subsidy = 0
                    eligible = False
                    conditions = "Battery capacity required"
            
            return {
                "eligible": eligible,
                "fame_subsidy": int(fame_subsidy) if eligible else 0,
                "vehicle_type": vehicle_type,
                "conditions": conditions,
                "valid_until": valid_until,
                "subsidy_breakdown": {
                    "per_kwh": int(subsidy_per_kwh),
                    "max_allowed": int(max_subsidy),
                    "battery_capacity_kwh": battery_capacity_kwh
                }
            }
        
        except Exception as e:
            logger.error(f"FAME calculation error: {e}")
            return {"error": str(e)}
    
    def calculate_state_subsidy(self, state, vehicle_type):
        """
        Calculate state subsidy
        
        Args:
            state: State name
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            
        Returns:
            dict with state subsidy details
        """
        if self.state_data is None:
            return {"error": "State data not loaded"}
        
        try:
            # Filter for state and vehicle type
            state_row = self.state_data[
                (self.state_data['state'].str.lower() == state.lower()) &
                (self.state_data['vehicle_type'] == vehicle_type)
            ]
            
            if state_row.empty:
                return {
                    "eligible": False,
                    "state_subsidy": 0,
                    "reason": f"No state subsidy available for {vehicle_type} in {state}"
                }
            
            state_row = state_row.iloc[0]
            
            return {
                "eligible": True,
                "state": state,
                "vehicle_type": vehicle_type,
                "state_subsidy": int(state_row['subsidy_amount']),
                "additional_benefits": state_row.get('additional_benefits', 'N/A'),
                "valid_until": state_row.get('valid_until', 'N/A')
            }
        
        except Exception as e:
            logger.error(f"State subsidy calculation error: {e}")
            return {"error": str(e)}
    
    def calculate_total_subsidy(self, vehicle_type, state, battery_capacity_kwh=None, ex_showroom_price=None):
        """
        Calculate total subsidy (FAME + State)
        
        Returns:
            dict with complete subsidy breakdown
        """
        fame_result = self.calculate_fame_subsidy(vehicle_type, battery_capacity_kwh, ex_showroom_price)
        state_result = self.calculate_state_subsidy(state, vehicle_type)
        
        fame_amount = fame_result.get('fame_subsidy', 0) if fame_result.get('eligible') else 0
        state_amount = state_result.get('state_subsidy', 0) if state_result.get('eligible') else 0
        
        total_subsidy = fame_amount + state_amount
        
        # Calculate effective price
        effective_price = None
        savings_percent = None
        if ex_showroom_price:
            effective_price = ex_showroom_price - total_subsidy
            savings_percent = (total_subsidy / ex_showroom_price) * 100
        
        return {
            "vehicle_type": vehicle_type,
            "state": state,
            "ex_showroom_price": ex_showroom_price,
            "fame_subsidy": fame_amount,
            "state_subsidy": state_amount,
            "total_subsidy": total_subsidy,
            "effective_price": effective_price,
            "savings_percent": round(savings_percent, 2) if savings_percent else None,
            "fame_details": fame_result,
            "state_details": state_result
        }
    
    def get_available_states(self):
        """Get list of states with subsidy programs"""
        if self.state_data is None:
            return []
        return sorted(self.state_data['state'].unique().tolist())

# Create global instance
fame_calculator = FAMESubsidyCalculator()

if __name__ == '__main__':
    print("Testing FAME Subsidy Calculator...")
    
    # Test 2-wheeler
    result = fame_calculator.calculate_total_subsidy(
        vehicle_type='2-Wheeler',
        state='Delhi',
        battery_capacity_kwh=3.24,
        ex_showroom_price=125000
    )
    
    print("\n2-Wheeler Subsidy (Delhi):")
    print(f"Ex-showroom: ₹{result['ex_showroom_price']:,}")
    print(f"FAME-II: ₹{result['fame_subsidy']:,}")
    print(f"State: ₹{result['state_subsidy']:,}")
    print(f"Total Subsidy: ₹{result['total_subsidy']:,}")
    print(f"Effective Price: ₹{result['effective_price']:,}")
    print(f"Savings: {result['savings_percent']}%")
    
    # Test 4-wheeler
    result = fame_calculator.calculate_total_subsidy(
        vehicle_type='4-Wheeler',
        state='Maharashtra',
        battery_capacity_kwh=40.5,
        ex_showroom_price=1400000
    )
    
    print("\n4-Wheeler Subsidy (Maharashtra):")
    print(f"Ex-showroom: ₹{result['ex_showroom_price']:,}")
    print(f"FAME-II: ₹{result['fame_subsidy']:,}")
    print(f"State: ₹{result['state_subsidy']:,}")
    print(f"Total Subsidy: ₹{result['total_subsidy']:,}")
    print(f"Effective Price: ₹{result['effective_price']:,}")
    print(f"Savings: {result['savings_percent']}%")
    
    print("\nAvailable states:", fame_calculator.get_available_states())
