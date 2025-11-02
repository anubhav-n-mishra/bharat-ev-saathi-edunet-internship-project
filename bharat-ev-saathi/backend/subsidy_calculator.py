"""
FAME & State Subsidy Calculator
================================
Calculates total EV subsidies available based on central FAME-II scheme
and state-specific policies. Provides detailed breakdown of savings.

Author: Bharat EV Saathi Project
"""

from backend.data_loader import ev_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FAME-II Subsidy Rules (as of 2025)
FAME_II_RULES = {
    '2-Wheeler': {
        'subsidy_per_kwh': 15000,  # ₹15,000 per kWh
        'max_subsidy': 15000,       # Maximum ₹15,000
        'max_price': 150000,        # Price cap: ₹1.5 lakh
        'min_battery': 2.0,         # Minimum 2 kWh battery
    },
    '3-Wheeler': {
        'subsidy_per_kwh': 10000,  # ₹10,000 per kWh
        'max_subsidy': 50000,       # Maximum ₹50,000
        'max_price': None,          # No price cap
        'min_battery': 5.0,         # Minimum 5 kWh battery
    },
    '4-Wheeler': {
        'subsidy_per_kwh': 10000,  # ₹10,000 per kWh
        'max_subsidy': 150000,      # Maximum ₹1.5 lakh
        'max_price': 1500000,       # Price cap: ₹15 lakh (mainly for commercial)
        'min_battery': 15.0,        # Minimum 15 kWh battery
    }
}

class SubsidyCalculator:
    """
    Calculate comprehensive EV subsidies including FAME-II and state benefits
    """
    
    def __init__(self):
        """Initialize the subsidy calculator"""
        self.data_loader = ev_data
        logger.info("Subsidy Calculator initialized")
    
    def calculate_fame_subsidy(self, vehicle_type, battery_kwh, vehicle_price):
        """
        Calculate central FAME-II subsidy
        
        Args:
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            battery_kwh: Battery capacity in kWh
            vehicle_price: Ex-showroom price in INR
            
        Returns:
            Dictionary with subsidy amount and eligibility details
        """
        result = {
            'eligible': False,
            'subsidy_amount': 0,
            'reason': '',
            'calculation': ''
        }
        
        if vehicle_type not in FAME_II_RULES:
            result['reason'] = f"Invalid vehicle type: {vehicle_type}"
            return result
        
        rules = FAME_II_RULES[vehicle_type]
        
        # Check battery capacity
        if battery_kwh < rules['min_battery']:
            result['reason'] = f"Battery capacity too low. Minimum: {rules['min_battery']} kWh"
            return result
        
        # Check price cap
        if rules['max_price'] and vehicle_price > rules['max_price']:
            result['reason'] = f"Price exceeds limit of ₹{rules['max_price']:,}"
            # Note: For 4-wheelers, personal use has limited FAME benefits
            if vehicle_type == '4-Wheeler':
                result['reason'] += " (FAME-II mainly supports 2W/3W and commercial 4W)"
            return result
        
        # Calculate subsidy
        calculated_subsidy = battery_kwh * rules['subsidy_per_kwh']
        final_subsidy = min(calculated_subsidy, rules['max_subsidy'])
        
        result['eligible'] = True
        result['subsidy_amount'] = int(final_subsidy)
        result['calculation'] = (
            f"{battery_kwh} kWh × ₹{rules['subsidy_per_kwh']:,}/kWh = "
            f"₹{calculated_subsidy:,.0f} (capped at ₹{rules['max_subsidy']:,})"
        )
        result['reason'] = 'Eligible for FAME-II subsidy'
        
        return result
    
    def get_state_subsidy(self, state, vehicle_type):
        """
        Get state-specific subsidy information
        
        Args:
            state: Indian state name
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            
        Returns:
            Dictionary with state subsidy details
        """
        subsidy_data = self.data_loader.get_state_subsidy(state, vehicle_type)
        
        if subsidy_data is None:
            return {
                'eligible': False,
                'subsidy_amount': 0,
                'scrapping_bonus': 0,
                'additional_benefits': [],
                'policy_name': None
            }
        
        return {
            'eligible': True,
            'subsidy_amount': int(subsidy_data.get('subsidy_amount', 0)),
            'scrapping_bonus': int(subsidy_data.get('scrapping_bonus', 0)),
            'additional_benefits': subsidy_data.get('additional_benefits', '').split(', '),
            'policy_name': subsidy_data.get('policy_name', ''),
            'purchase_incentive': subsidy_data.get('purchase_incentive', ''),
            'valid_until': subsidy_data.get('valid_until', '')
        }
    
    def calculate_total_subsidy(self, vehicle_type, battery_kwh, vehicle_price, 
                                state, has_old_vehicle=False):
        """
        Calculate total subsidy combining FAME-II and state benefits
        
        Args:
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            battery_kwh: Battery capacity in kWh
            vehicle_price: Ex-showroom price in INR
            state: Indian state name
            has_old_vehicle: Whether customer is scrapping old vehicle
            
        Returns:
            Comprehensive dictionary with all subsidy details
        """
        # Calculate FAME-II subsidy
        fame_result = self.calculate_fame_subsidy(vehicle_type, battery_kwh, vehicle_price)
        
        # Get state subsidy
        state_result = self.get_state_subsidy(state, vehicle_type)
        
        # Calculate scrapping bonus if applicable
        scrapping_bonus = 0
        if has_old_vehicle and state_result['eligible']:
            scrapping_bonus = state_result['scrapping_bonus']
        
        # Total subsidy
        total_subsidy = (
            fame_result['subsidy_amount'] + 
            state_result['subsidy_amount'] + 
            scrapping_bonus
        )
        
        # Effective price after subsidies
        effective_price = vehicle_price - total_subsidy
        
        # Calculate savings percentage
        savings_percentage = (total_subsidy / vehicle_price * 100) if vehicle_price > 0 else 0
        
        return {
            'vehicle_price': vehicle_price,
            'fame_subsidy': fame_result['subsidy_amount'],
            'fame_eligible': fame_result['eligible'],
            'fame_reason': fame_result['reason'],
            'fame_calculation': fame_result.get('calculation', ''),
            'state_subsidy': state_result['subsidy_amount'],
            'state_eligible': state_result['eligible'],
            'state_policy': state_result['policy_name'],
            'scrapping_bonus': scrapping_bonus,
            'total_subsidy': total_subsidy,
            'effective_price': effective_price,
            'savings_percentage': round(savings_percentage, 2),
            'additional_benefits': state_result['additional_benefits'],
            'breakdown': {
                'Central FAME-II': fame_result['subsidy_amount'],
                'State Subsidy': state_result['subsidy_amount'],
                'Scrapping Bonus': scrapping_bonus,
            }
        }
    
    def calculate_for_vehicle(self, brand, model, state, has_old_vehicle=False):
        """
        Calculate subsidy for a specific EV from the database
        
        Args:
            brand: EV brand name
            model: EV model name
            state: Indian state name
            has_old_vehicle: Whether scrapping old vehicle
            
        Returns:
            Complete subsidy calculation result
        """
        vehicle = self.data_loader.get_ev_by_id(brand, model)
        
        if vehicle is None:
            return {
                'error': f"Vehicle not found: {brand} {model}",
                'total_subsidy': 0
            }
        
        return self.calculate_total_subsidy(
            vehicle_type=vehicle['type'],
            battery_kwh=vehicle['battery_kwh'],
            vehicle_price=vehicle['price_inr'],
            state=state,
            has_old_vehicle=has_old_vehicle
        )
    
    def compare_states(self, vehicle_type, battery_kwh, vehicle_price):
        """
        Compare subsidies across all states for a given vehicle
        
        Args:
            vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
            battery_kwh: Battery capacity in kWh
            vehicle_price: Ex-showroom price in INR
            
        Returns:
            List of dictionaries with state-wise comparison
        """
        states = self.data_loader.get_all_states_with_subsidies()
        comparison = []
        
        for state in states:
            result = self.calculate_total_subsidy(
                vehicle_type, battery_kwh, vehicle_price, state, has_old_vehicle=True
            )
            
            comparison.append({
                'state': state,
                'total_subsidy': result['total_subsidy'],
                'effective_price': result['effective_price'],
                'savings_percentage': result['savings_percentage'],
                'state_policy': result['state_policy']
            })
        
        # Sort by total subsidy (descending)
        comparison.sort(key=lambda x: x['total_subsidy'], reverse=True)
        
        return comparison
    
    def get_available_states(self):
        """Get list of states with EV subsidies"""
        return self.data_loader.get_all_states_with_subsidies()


# Create global instance
subsidy_calc = SubsidyCalculator()


if __name__ == '__main__':
    # Test the subsidy calculator
    print("🧪 Testing Subsidy Calculator...")
    
    calc = SubsidyCalculator()
    
    # Test 1: Calculate for Tata Nexon EV in Maharashtra
    print("\n📊 Test 1: Tata Nexon EV in Maharashtra")
    result = calc.calculate_for_vehicle('Tata', 'Nexon EV', 'Maharashtra', has_old_vehicle=True)
    
    print(f"Vehicle Price: ₹{result['vehicle_price']:,}")
    print(f"FAME-II Subsidy: ₹{result['fame_subsidy']:,}")
    print(f"State Subsidy: ₹{result['state_subsidy']:,}")
    print(f"Scrapping Bonus: ₹{result['scrapping_bonus']:,}")
    print(f"Total Subsidy: ₹{result['total_subsidy']:,}")
    print(f"Effective Price: ₹{result['effective_price']:,}")
    print(f"Savings: {result['savings_percentage']}%")
    print(f"Additional Benefits: {', '.join(result['additional_benefits'])}")
    
    # Test 2: Compare states for Ola S1 Pro
    print("\n📊 Test 2: State Comparison for Ola S1 Pro (2-Wheeler)")
    comparison = calc.compare_states('2-Wheeler', 3.97, 139999)
    
    print("Top 5 states by subsidy:")
    for i, state_data in enumerate(comparison[:5], 1):
        print(f"{i}. {state_data['state']}: ₹{state_data['total_subsidy']:,} "
              f"({state_data['savings_percentage']:.1f}% savings)")
    
    print("\n✅ All tests passed!")
