"""
Process FAME-II Bus Deployment Data
====================================
Loads and analyzes the government FAME-II electric bus deployment data.

Source: RS_Session_265_AU_2154_A_and_B_2.csv
Type: State-wise electric bus sanctioning and deployment statistics

Author: Bharat EV Saathi Project
Date: November 2025
"""

import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FAME_CSV = PROJECT_ROOT / 'RS_Session_265_AU_2154_A_and_B_2.csv'
PROCESSED_DIR = Path(__file__).parent.parent / 'processed'
PROCESSED_DIR.mkdir(exist_ok=True)

def load_fame_bus_data():
    """
    Load FAME-II bus deployment data
    
    Returns:
        DataFrame with state-wise bus data
    """
    print("\n🚌 Loading FAME-II Bus Deployment Data...")
    
    if not FAME_CSV.exists():
        print(f"❌ File not found: {FAME_CSV}")
        print("This is government data showing electric bus deployment under FAME-II scheme")
        return None
    
    try:
        # Load CSV
        df = pd.read_csv(FAME_CSV)
        print(f"✅ Loaded data for {len(df)} states/UTs")
        
        # Print columns
        print(f"\n📋 Columns: {df.columns.tolist()}")
        
        # Calculate additional metrics
        df['deployment_percentage'] = (
            df['Number of Buses Received and Deployed'] / 
            df['Number of Buses Sanctioned'] * 100
        ).round(2)
        
        df['pending_buses'] = (
            df['Number of Buses Sanctioned'] - 
            df['Number of Buses Received and Deployed']
        )
        
        # Rename for easier access
        df.rename(columns={
            'State/UT': 'state',
            'Number of Buses Sanctioned': 'buses_sanctioned',
            'Number of Buses Received and Deployed': 'buses_deployed'
        }, inplace=True)
        
        # Add EV readiness score (based on deployment %)
        df['ev_readiness_score'] = df['deployment_percentage'].apply(
            lambda x: 'High' if x >= 90 else 'Medium' if x >= 70 else 'Low'
        )
        
        # Save processed data
        output_path = PROCESSED_DIR / 'fame_ii_bus_deployment.csv'
        df.to_csv(output_path, index=False)
        print(f"✅ Saved processed data to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def analyze_fame_data(df):
    """
    Analyze FAME-II bus deployment statistics
    
    Args:
        df: FAME-II bus deployment DataFrame
    """
    if df is None:
        return
    
    print("\n" + "="*60)
    print("📊 FAME-II Bus Deployment Analysis")
    print("="*60)
    
    # Overall statistics
    total_sanctioned = df['buses_sanctioned'].sum()
    total_deployed = df['buses_deployed'].sum()
    overall_deployment = (total_deployed / total_sanctioned * 100)
    
    print(f"\n🇮🇳 National Summary:")
    print(f"  Total Buses Sanctioned: {total_sanctioned:,}")
    print(f"  Total Buses Deployed: {total_deployed:,}")
    print(f"  Deployment Rate: {overall_deployment:.2f}%")
    print(f"  Pending Deployment: {total_sanctioned - total_deployed:,} buses")
    
    # Top performers
    print(f"\n🏆 Top 5 States by Deployment:")
    top_5 = df.nlargest(5, 'buses_deployed')[['state', 'buses_deployed', 'deployment_percentage']]
    for idx, row in top_5.iterrows():
        print(f"  {row['state']}: {row['buses_deployed']:,} buses ({row['deployment_percentage']:.1f}% deployed)")
    
    # 100% deployment states
    fully_deployed = df[df['deployment_percentage'] == 100]
    if len(fully_deployed) > 0:
        print(f"\n✅ States with 100% Deployment ({len(fully_deployed)}):")
        for state in fully_deployed['state']:
            print(f"  • {state}")
    
    # States with pending buses
    pending = df[df['pending_buses'] > 0].sort_values('pending_buses', ascending=False)
    if len(pending) > 0:
        print(f"\n⏳ Top 5 States with Pending Buses:")
        top_pending = pending.head(5)
        for idx, row in top_pending.iterrows():
            print(f"  {row['state']}: {row['pending_buses']:,} pending ({row['deployment_percentage']:.1f}% done)")
    
    # EV Readiness Distribution
    print(f"\n🎯 EV Infrastructure Readiness:")
    readiness = df['ev_readiness_score'].value_counts()
    for score, count in readiness.items():
        print(f"  {score}: {count} states")
    
    # Correlation with EV adoption potential
    print(f"\n💡 Insights:")
    high_deployment = df[df['deployment_percentage'] >= 90]['state'].tolist()
    if high_deployment:
        print(f"  • High EV readiness states (90%+ deployment): {', '.join(high_deployment[:5])}")
        print(f"  • These states likely have better charging infrastructure")
        print(f"  • Good targets for EV expansion and marketing")

def main():
    """
    Main function to load and analyze FAME-II data
    """
    print("="*60)
    print("🎯 FAME-II Bus Deployment Data Processing")
    print("="*60)
    
    # Load data
    df = load_fame_bus_data()
    
    # Analyze
    if df is not None:
        analyze_fame_data(df)
        
        print("\n" + "="*60)
        print("✅ FAME-II Data Processing Complete!")
        print("="*60)
        print(f"\n📊 Key Insights:")
        print("  1. This data shows government's FAME-II electric bus program")
        print("  2. High deployment % indicates strong EV infrastructure")
        print("  3. Can be used to predict EV adoption potential by state")
        print("  4. Useful for targeting marketing and expansion efforts")
        print(f"\n💾 Processed data saved to: {PROCESSED_DIR}")
    else:
        print("\n⚠️  Could not process FAME-II data")
        print("Make sure RS_Session_265_AU_2154_A_and_B_2.csv is in project root")

if __name__ == '__main__':
    main()
