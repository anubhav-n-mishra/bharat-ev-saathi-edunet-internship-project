"""
Indian EV Subsidy Database
==========================
Comprehensive database of central (FAME-II) and state-level EV subsidies in India.
Updated as of 2025.

Author: Bharat EV Saathi Project
"""

import pandas as pd

# Central FAME-II Scheme Details
FAME_II_SUBSIDY = {
    '2-Wheeler': {
        'subsidy_per_kwh': 15000,
        'max_subsidy': 15000,
        'conditions': 'Battery capacity ≥ 2 kWh, Ex-showroom price ≤ ₹1.5 lakh',
        'valid_until': '2027-03-31'
    },
    '3-Wheeler': {
        'subsidy_per_kwh': 10000,
        'max_subsidy': 50000,
        'conditions': 'Public transport or commercial use',
        'valid_until': '2027-03-31'
    },
    '4-Wheeler': {
        'subsidy_per_kwh': 10000,
        'max_subsidy': 150000,
        'conditions': 'Ex-showroom price ≤ ₹15 lakh (minimal for personal use)',
        'valid_until': '2027-03-31'
    }
}

# State-wise EV Subsidy Schemes
STATE_SUBSIDIES = [
    {
        'state': 'Delhi',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 5000,
        'additional_benefits': 'Road tax waiver, Registration fee waiver',
        'purchase_incentive': 'Up to ₹5,000 for first 10,000 vehicles',
        'scrapping_bonus': 7500,
        'policy_name': 'Delhi EV Policy 2020',
        'valid_until': '2024-12-31'
    },
    {
        'state': 'Delhi',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 150000,
        'additional_benefits': 'Road tax waiver, Registration fee waiver',
        'purchase_incentive': 'Up to ₹1.5 lakh for first 30,000 vehicles',
        'scrapping_bonus': 0,
        'policy_name': 'Delhi EV Policy 2020',
        'valid_until': '2024-12-31'
    },
    {
        'state': 'Maharashtra',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 10000,
        'additional_benefits': 'Road tax exemption, Registration fee waiver',
        'purchase_incentive': '₹10,000 for first 1 lakh vehicles',
        'scrapping_bonus': 7000,
        'policy_name': 'Maharashtra EV Policy 2021',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'Maharashtra',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 100000,
        'additional_benefits': 'Road tax exemption, Registration fee waiver',
        'purchase_incentive': 'Up to ₹1 lakh (price-based)',
        'scrapping_bonus': 25000,
        'policy_name': 'Maharashtra EV Policy 2021',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'Gujarat',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 10000,
        'additional_benefits': 'Registration fee waiver, 100% road tax exemption',
        'purchase_incentive': '₹10,000 per vehicle',
        'scrapping_bonus': 5000,
        'policy_name': 'Gujarat EV Policy 2021',
        'valid_until': '2025-06-30'
    },
    {
        'state': 'Gujarat',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 150000,
        'additional_benefits': 'Registration fee waiver, 100% road tax exemption',
        'purchase_incentive': 'Up to ₹1.5 lakh',
        'scrapping_bonus': 20000,
        'policy_name': 'Gujarat EV Policy 2021',
        'valid_until': '2025-06-30'
    },
    {
        'state': 'Karnataka',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 10000,
        'additional_benefits': 'Road tax exemption, Registration fee waiver',
        'purchase_incentive': '₹10,000 for first 20,000 vehicles',
        'scrapping_bonus': 0,
        'policy_name': 'Karnataka EV Policy 2017',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'Karnataka',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 0,
        'additional_benefits': 'Road tax exemption, Registration fee waiver',
        'purchase_incentive': 'Manufacturing incentives only',
        'scrapping_bonus': 0,
        'policy_name': 'Karnataka EV Policy 2017',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'Tamil Nadu',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 5000,
        'additional_benefits': 'Registration fee waiver, Road tax exemption',
        'purchase_incentive': '₹5,000 per vehicle',
        'scrapping_bonus': 0,
        'policy_name': 'Tamil Nadu EV Policy 2019',
        'valid_until': '2024-12-31'
    },
    {
        'state': 'Tamil Nadu',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 100000,
        'additional_benefits': 'Registration fee waiver, Road tax exemption',
        'purchase_incentive': 'Up to ₹1 lakh for first 50,000 vehicles',
        'scrapping_bonus': 0,
        'policy_name': 'Tamil Nadu EV Policy 2019',
        'valid_until': '2024-12-31'
    },
    {
        'state': 'Telangana',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 0,
        'additional_benefits': 'Road tax exemption',
        'purchase_incentive': 'Focus on manufacturing incentives',
        'scrapping_bonus': 0,
        'policy_name': 'Telangana EV Policy 2020',
        'valid_until': '2030-12-31'
    },
    {
        'state': 'Telangana',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 0,
        'additional_benefits': 'Road tax exemption',
        'purchase_incentive': 'Focus on manufacturing incentives',
        'scrapping_bonus': 0,
        'policy_name': 'Telangana EV Policy 2020',
        'valid_until': '2030-12-31'
    },
    {
        'state': 'Rajasthan',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 10000,
        'additional_benefits': 'Registration fee waiver',
        'purchase_incentive': '₹10,000 per vehicle',
        'scrapping_bonus': 0,
        'policy_name': 'Rajasthan EV Policy 2019',
        'valid_until': '2024-12-31'
    },
    {
        'state': 'Rajasthan',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 50000,
        'additional_benefits': 'Registration fee waiver',
        'purchase_incentive': '₹50,000 per vehicle',
        'scrapping_bonus': 0,
        'policy_name': 'Rajasthan EV Policy 2019',
        'valid_until': '2024-12-31'
    },
    {
        'state': 'Uttar Pradesh',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 5000,
        'additional_benefits': 'Road tax exemption, Registration fee waiver',
        'purchase_incentive': '₹5,000 per vehicle',
        'scrapping_bonus': 0,
        'policy_name': 'UP EV Policy 2022',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'Uttar Pradesh',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 100000,
        'additional_benefits': 'Road tax exemption, Registration fee waiver',
        'purchase_incentive': 'Up to ₹1 lakh',
        'scrapping_bonus': 0,
        'policy_name': 'UP EV Policy 2022',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'West Bengal',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 0,
        'additional_benefits': 'Road tax exemption',
        'purchase_incentive': 'Manufacturing & infrastructure focus',
        'scrapping_bonus': 0,
        'policy_name': 'West Bengal EV Policy 2021',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'West Bengal',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 0,
        'additional_benefits': 'Road tax exemption',
        'purchase_incentive': 'Manufacturing & infrastructure focus',
        'scrapping_bonus': 0,
        'policy_name': 'West Bengal EV Policy 2021',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'Kerala',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 10000,
        'additional_benefits': 'Motor vehicle tax exemption',
        'purchase_incentive': '₹10,000 per vehicle',
        'scrapping_bonus': 0,
        'policy_name': 'Kerala EV Policy 2022',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'Kerala',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 50000,
        'additional_benefits': 'Motor vehicle tax exemption',
        'purchase_incentive': '₹50,000 per vehicle',
        'scrapping_bonus': 0,
        'policy_name': 'Kerala EV Policy 2022',
        'valid_until': '2025-12-31'
    },
    {
        'state': 'Madhya Pradesh',
        'vehicle_type': '2-Wheeler',
        'subsidy_amount': 0,
        'additional_benefits': 'Road tax exemption',
        'purchase_incentive': 'Limited state incentives',
        'scrapping_bonus': 0,
        'policy_name': 'MP EV Policy 2019',
        'valid_until': '2024-12-31'
    },
    {
        'state': 'Madhya Pradesh',
        'vehicle_type': '4-Wheeler',
        'subsidy_amount': 0,
        'additional_benefits': 'Road tax exemption',
        'purchase_incentive': 'Limited state incentives',
        'scrapping_bonus': 0,
        'policy_name': 'MP EV Policy 2019',
        'valid_until': '2024-12-31'
    },
]

def calculate_total_subsidy(vehicle_type, vehicle_price, battery_kwh, state, old_vehicle=False):
    """
    Calculate total subsidy available for an EV purchase
    
    Args:
        vehicle_type: '2-Wheeler', '3-Wheeler', or '4-Wheeler'
        vehicle_price: Ex-showroom price in INR
        battery_kwh: Battery capacity in kWh
        state: Indian state name
        old_vehicle: Whether scrapping old vehicle
    
    Returns:
        Dictionary with subsidy breakdown
    """
    subsidy_breakdown = {
        'central_fame': 0,
        'state_subsidy': 0,
        'scrapping_bonus': 0,
        'total_subsidy': 0,
        'effective_price': vehicle_price,
        'tax_benefits': []
    }
    
    # Calculate FAME-II subsidy
    if vehicle_type in FAME_II_SUBSIDY:
        fame_rules = FAME_II_SUBSIDY[vehicle_type]
        calculated_subsidy = battery_kwh * fame_rules['subsidy_per_kwh']
        
        # Check eligibility
        if vehicle_type == '2-Wheeler' and vehicle_price <= 150000:
            subsidy_breakdown['central_fame'] = min(calculated_subsidy, fame_rules['max_subsidy'])
        elif vehicle_type == '3-Wheeler':
            subsidy_breakdown['central_fame'] = min(calculated_subsidy, fame_rules['max_subsidy'])
        elif vehicle_type == '4-Wheeler' and vehicle_price <= 1500000:
            subsidy_breakdown['central_fame'] = min(calculated_subsidy, fame_rules['max_subsidy'])
    
    # Add state subsidy
    state_data = next((s for s in STATE_SUBSIDIES 
                      if s['state'] == state and s['vehicle_type'] == vehicle_type), None)
    
    if state_data:
        subsidy_breakdown['state_subsidy'] = state_data['subsidy_amount']
        if old_vehicle and state_data['scrapping_bonus'] > 0:
            subsidy_breakdown['scrapping_bonus'] = state_data['scrapping_bonus']
        if state_data['additional_benefits']:
            subsidy_breakdown['tax_benefits'].append(state_data['additional_benefits'])
    
    # Calculate totals
    subsidy_breakdown['total_subsidy'] = (
        subsidy_breakdown['central_fame'] + 
        subsidy_breakdown['state_subsidy'] + 
        subsidy_breakdown['scrapping_bonus']
    )
    subsidy_breakdown['effective_price'] = vehicle_price - subsidy_breakdown['total_subsidy']
    
    return subsidy_breakdown

if __name__ == '__main__':
    print("💰 Generating Indian EV Subsidy Database...")
    
    # Save state subsidies to CSV
    df_subsidies = pd.DataFrame(STATE_SUBSIDIES)
    df_subsidies.to_csv('state_ev_subsidies.csv', index=False)
    print(f"✅ State subsidies data saved: {len(df_subsidies)} records")
    
    # Create FAME-II summary
    fame_data = []
    for vtype, details in FAME_II_SUBSIDY.items():
        fame_data.append({
            'vehicle_type': vtype,
            'subsidy_per_kwh': details['subsidy_per_kwh'],
            'max_subsidy': details['max_subsidy'],
            'conditions': details['conditions'],
            'valid_until': details['valid_until']
        })
    
    df_fame = pd.DataFrame(fame_data)
    df_fame.to_csv('fame_ii_subsidy.csv', index=False)
    print(f"✅ FAME-II data saved: {len(df_fame)} records")
    
    print("\n📊 Subsidy Summary by State:")
    print(df_subsidies.groupby('state')['subsidy_amount'].sum().sort_values(ascending=False))
    
    # Test calculation
    print("\n🧮 Example Calculation:")
    print("Vehicle: Tata Nexon EV (₹15,00,000, 30.2 kWh) in Maharashtra")
    result = calculate_total_subsidy('4-Wheeler', 1500000, 30.2, 'Maharashtra', old_vehicle=True)
    print(f"Central FAME: ₹{result['central_fame']:,}")
    print(f"State Subsidy: ₹{result['state_subsidy']:,}")
    print(f"Scrapping Bonus: ₹{result['scrapping_bonus']:,}")
    print(f"Total Subsidy: ₹{result['total_subsidy']:,}")
    print(f"Effective Price: ₹{result['effective_price']:,}")
    print(f"Tax Benefits: {', '.join(result['tax_benefits'])}")
