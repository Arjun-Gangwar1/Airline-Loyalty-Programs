"""
STAGE 1: DATA EXPLORATION & QUALITY ASSESSMENT
==============================================

This script performs comprehensive data exploration and documents all findings.

CHECKPOINT: After this stage, you'll have:
✅ Cleaned dataset loaded
✅ Data quality report
✅ Initial visualizations
✅ Understanding of the data structure

Can safely stop after this stage and resume with Stage 2.

Expected Time: 30-45 minutes
Expected Outputs: 
- Data quality report (JSON)
- Exploration visualizations (PNG)
- Summary statistics (CSV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class DataExplorer:
    """
    Comprehensive data exploration and quality assessment
    """
    
    def __init__(self, raw_data_path="data/raw"):
        self.raw_path = Path(raw_data_path)
        self.output_path = Path("outputs/figures/exploration")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.loyalty = None
        self.activity = None
        self.calendar = None
        self.data_dict = None
        
        self.issues = []
        self.summary = {}
        
    def load_data(self):
        """Load all datasets"""
        
        print("\n" + "=" * 60)
        print("STAGE 1: DATA EXPLORATION")
        print("=" * 60)
        
        print("\n� Loading datasets...")
        
        try:
            # Load Customer Loyalty History
            loyalty_path = self.raw_path / "Customer Loyalty History.csv"
            if not loyalty_path.exists():
                print(f"❌ ERROR: {loyalty_path} not found!")
                print("   Please download the dataset first.")
                print("   See DATA_DOWNLOAD_GUIDE.md for instructions.")
                return False
                
            self.loyalty = pd.read_csv(loyalty_path)
            print(f"   ✓ Loyalty History: {self.loyalty.shape}")
            
            # Load Customer Flight Activity
            activity_path = self.raw_path / "Customer Flight Activity.csv"
            self.activity = pd.read_csv(activity_path)
            print(f"   ✓ Flight Activity: {self.activity.shape}")
            
            # Load Calendar (optional)
            calendar_path = self.raw_path / "Calendar.csv"
            if calendar_path.exists():
                self.calendar = pd.read_csv(calendar_path)
                print(f"   ✓ Calendar: {self.calendar.shape}")
            
            # Load Data Dictionary (optional)
            dict_path = self.raw_path / "Airline Loyalty Data Dictionary.csv"
            if dict_path.exists():
                self.data_dict = pd.read_csv(dict_path)
                print(f"   ✓ Data Dictionary: {self.data_dict.shape}")
            
            print("\n✅ All datasets loaded successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR loading data: {e}")
            return False
    
    def explore_loyalty_data(self):
        """Deep exploration of loyalty history"""
        
        print("\n" + "=" * 60)
        print("1. CUSTOMER LOYALTY HISTORY ANALYSIS")
        print("=" * 60)
        
        df = self.loyalty
        
        # Basic info
        print(f"\n� Dataset Shape: {df.shape}")
        print(f"� Unique Customers: {df['Loyalty Number'].nunique()}")
        print(f"� Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Store summary
        self.summary['total_customers'] = len(df)
        self.summary['unique_customers'] = df['Loyalty Number'].nunique()
        
        # Column analysis
        print("\n� Columns Found:")
        for col in df.columns:
            print(f"   - {col}")
        
        # Missing values
        print("\n� Missing Values Analysis:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing': missing,
            'Percentage': missing_pct
        }).sort_values('Missing', ascending=False)
        
        if missing_df['Missing'].sum() > 0:
            print(missing_df[missing_df['Missing'] > 0])
            self.issues.append({
                'dataset': 'loyalty',
                'issue': 'missing_values',
                'severity': 'medium',
                'details': missing_df[missing_df['Missing'] > 0].to_dict()
            })
        else:
            print("   ✓ No missing values!")
        
        # Data types
        print("\n� Data Types:")
        print(df.dtypes)
        
        # Numerical statistics
        print("\n� Numerical Summary Statistics:")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numerical_cols].describe())
        
        # Cancellation analysis
        cancel_cols = [col for col in df.columns if 'cancel' in col.lower()]
        if cancel_cols:
            print(f"\n⚠️  Cancellation Analysis:")
            cancel_col = cancel_cols[0]
            cancelled = df[cancel_col].notna().sum()
            cancel_rate = (cancelled / len(df)) * 100
            print(f"   Cancelled members: {cancelled} ({cancel_rate:.2f}%)")
            print(f"   Active members: {len(df) - cancelled} ({100-cancel_rate:.2f}%)")
            
            self.summary['cancellation_rate'] = cancel_rate
            
            # Visualize
            fig, ax = plt.subplots(figsize=(10, 6))
            cancel_data = df[cancel_col].value_counts().sort_index()
            cancel_data.plot(kind='bar', ax=ax)
            ax.set_title('Cancellations by Year', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Cancellations')
            plt.tight_layout()
            plt.savefig(self.output_path / 'cancellations_by_year.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: cancellations_by_year.png")
        
        # CLV analysis
        clv_cols = [col for col in df.columns if 'clv' in col.lower()]
        if clv_cols:
            clv_col = clv_cols[0]
            print(f"\n� Customer Lifetime Value ({clv_col}):")
            print(f"   Mean: ${df[clv_col].mean():,.2f}")
            print(f"   Median: ${df[clv_col].median():,.2f}")
            print(f"   Min: ${df[clv_col].min():,.2f}")
            print(f"   Max: ${df[clv_col].max():,.2f}")
            
            self.summary['avg_clv'] = float(df[clv_col].mean())
            
            # Check for anomalies
            negative_clv = (df[clv_col] <= 0).sum()
            if negative_clv > 0:
                print(f"   ⚠️  {negative_clv} customers with CLV ≤ 0")
                self.issues.append({
                    'dataset': 'loyalty',
                    'issue': 'negative_clv',
                    'severity': 'high',
                    'count': int(negative_clv)
                })
            
            # Visualize CLV
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram
            axes[0].hist(df[clv_col].clip(lower=df[clv_col].quantile(0.01), 
                                          upper=df[clv_col].quantile(0.99)), 
                        bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_title('CLV Distribution', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('CLV ($)')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Box plot by tier
            tier_cols = [col for col in df.columns if 'tier' in col.lower() or 'card' in col.lower()]
            if tier_cols:
                tier_col = tier_cols[0]
                df.boxplot(column=clv_col, by=tier_col, ax=axes[1])
                axes[1].set_title(f'CLV by {tier_col}', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(tier_col)
                axes[1].set_ylabel('CLV ($)')
                plt.suptitle('')  # Remove default title
            
            plt.tight_layout()
            plt.savefig(self.output_path / 'clv_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: clv_analysis.png")
        
        # Age distribution
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        if age_cols:
            age_col = age_cols[0]
            print(f"\n� Age Distribution:")
            print(f"   Mean: {df[age_col].mean():.1f} years")
            print(f"   Range: {df[age_col].min():.0f} - {df[age_col].max():.0f} years")
            
            suspicious_ages = ((df[age_col] < 18) | (df[age_col] > 100)).sum()
            if suspicious_ages > 0:
                print(f"   ⚠️  {suspicious_ages} customers with suspicious age (<18 or >100)")
                self.issues.append({
                    'dataset': 'loyalty',
                    'issue': 'suspicious_ages',
                    'severity': 'low',
                    'count': int(suspicious_ages)
                })
    
    def explore_activity_data(self):
        """Deep exploration of flight activity"""
        
        print("\n" + "=" * 60)
        print("2. CUSTOMER FLIGHT ACTIVITY ANALYSIS")
        print("=" * 60)
        
        df = self.activity
        
        print(f"\n� Dataset Shape: {df.shape}")
        print(f"� Unique Customers: {df['Loyalty Number'].nunique()}")
        
        # Time period
        date_cols = [col for col in df.columns if 'month' in col.lower() or 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            print(f"� Time Period: {df[date_col].min()} to {df[date_col].max()}")
            
            self.summary['data_start_date'] = str(df[date_col].min())
            self.summary['data_end_date'] = str(df[date_col].max())
        
        # Missing values
        print("\n� Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("   ✓ No missing values!")
        
        # Activity metrics
        print("\n� Activity Metrics:")
        flight_cols = [col for col in df.columns if 'flight' in col.lower()]
        if flight_cols:
            flight_col = flight_cols[0]
            print(f"   Total Flights: {df[flight_col].sum():,}")
            print(f"   Avg per Record: {df[flight_col].mean():.2f}")
            
            self.summary['total_flights'] = int(df[flight_col].sum())
        
        distance_cols = [col for col in df.columns if 'distance' in col.lower() or 'mile' in col.lower()]
        if distance_cols:
            distance_col = distance_cols[0]
            print(f"   Total Distance: {df[distance_col].sum():,.0f} miles")
        
        # Temporal analysis
        if date_cols:
            print("\n� Temporal Patterns:")
            monthly_activity = df.groupby(date_col)[flight_col].sum()
            print(f"   Most Active Month: {monthly_activity.idxmax().strftime('%Y-%m')}")
            print(f"   Least Active Month: {monthly_activity.idxmin().strftime('%Y-%m')}")
            
            # Visualize
            fig, ax = plt.subplots(figsize=(15, 6))
            monthly_activity.plot(ax=ax, linewidth=2, color='steelblue')
            ax.set_title('Total Flights Booked Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Flights Booked')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_path / 'monthly_flight_activity.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: monthly_flight_activity.png")
        
        # Customer activity distribution
        print("\n� Customer Activity Distribution:")
        customer_totals = df.groupby('Loyalty Number').agg({
            flight_col: 'sum'
        })
        
        print(f"   Avg Flights per Customer: {customer_totals[flight_col].mean():.2f}")
        print(f"   Median Flights per Customer: {customer_totals[flight_col].median():.0f}")
        print(f"   Max Flights (Single Customer): {customer_totals[flight_col].max():.0f}")
        
        self.summary['avg_flights_per_customer'] = float(customer_totals[flight_col].mean())
    
    def check_data_leakage(self):
        """Check for potential data leakage issues"""
        
        print("\n" + "=" * 60)
        print("3. DATA LEAKAGE RISK CHECK")
        print("=" * 60)
        
        print("\n⚠️  Checking for activity after cancellation...")
        
        # Find cancellation column
        cancel_cols = [col for col in self.loyalty.columns if 'cancel' in col.lower()]
        if not cancel_cols:
            print("   ⚠️  No cancellation column found")
            return
        
        cancel_col = cancel_cols[0]
        cancelled_customers = self.loyalty[self.loyalty[cancel_col].notna()]
        
        if len(cancelled_customers) == 0:
            print("   ✓ No cancelled customers to check")
            return
        
        # Check sample
        leakage_count = 0
        for _, customer in cancelled_customers.head(100).iterrows():
            customer_id = customer['Loyalty Number']
            
            # Get customer activity
            customer_activity = self.activity[self.activity['Loyalty Number'] == customer_id]
            
            # Simple check: customer with cancellation but has activity records
            if len(customer_activity) > 0:
                leakage_count += 1
        
        if leakage_count > 0:
            print(f"   ⚠️  {leakage_count}/100 cancelled customers have activity records")
            print(f"   This is expected - we need to filter by date later")
            self.issues.append({
                'dataset': 'both',
                'issue': 'potential_temporal_leakage',
                'severity': 'high',
                'count': leakage_count,
                'action': 'Must implement strict temporal filtering in feature engineering'
            })
        else:
            print("   ✓ No obvious leakage detected")
    
    def generate_summary_report(self):
        """Generate comprehensive summary"""
        
        print("\n" + "=" * 60)
        print("4. DATA QUALITY SUMMARY")
        print("=" * 60)
        
        if len(self.issues) == 0:
            print("\n✅ No major data quality issues detected!")
        else:
            print(f"\n⚠️  {len(self.issues)} data quality issues identified:\n")
            for i, issue in enumerate(self.issues, 1):
                severity_icon = "�" if issue['severity'] == 'high' else "�" if issue['severity'] == 'medium' else "�"
                print(f"{i}. {severity_icon} {issue['dataset'].upper()}: {issue['issue']}")
                if 'count' in issue:
                    print(f"   Affected records: {issue['count']}")
                if 'action' in issue:
                    print(f"   Action needed: {issue['action']}")
                print()
        
        # Save issues
        issues_file = self.processed_path / 'data_quality_issues.json'
        with open(issues_file, 'w') as f:
            json.dump(self.issues, f, indent=2)
        print(f"✓ Data quality report saved: {issues_file}")
        
        # Save summary
        summary_file = self.processed_path / 'data_summary.json'
        self.summary['exploration_date'] = datetime.now().isoformat()
        self.summary['total_issues'] = len(self.issues)
        with open(summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2)
        print(f"✓ Summary statistics saved: {summary_file}")
    
    def save_checkpoint(self):
        """Save data for next stage"""
        
        print("\n� Saving checkpoint for Stage 2...")
        
        # Save datasets
        self.loyalty.to_csv(self.processed_path / 'loyalty_raw.csv', index=False)
        self.activity.to_csv(self.processed_path / 'activity_raw.csv', index=False)
        
        print("   ✓ Data saved to checkpoints")
        
        # Update progress
        progress_file = Path("checkpoints/progress.json")
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        progress['current_stage'] = 1
        progress['completed_stages'].append(1)
        progress['last_updated'] = datetime.now().isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("   ✓ Progress updated")
    
    def run_full_exploration(self):
        """Execute complete exploration pipeline"""
        
        # Load data
        if not self.load_data():
            return False
        
        # Run exploration
        self.explore_loyalty_data()
        self.explore_activity_data()
        self.check_data_leakage()
        self.generate_summary_report()
        self.save_checkpoint()
        
        print("\n" + "=" * 60)
        print("✅ STAGE 1 COMPLETE")
        print("=" * 60)
        print(f"\n� Summary:")
        print(f"   Total Customers: {self.summary.get('total_customers', 'N/A')}")
        print(f"   Cancellation Rate: {self.summary.get('cancellation_rate', 0):.2f}%")
        print(f"   Average CLV: ${self.summary.get('avg_clv', 0):,.2f}")
        print(f"   Total Flights: {self.summary.get('total_flights', 0):,}")
        print(f"\n� Outputs:")
        print(f"   - Visualizations: {self.output_path}")
        print(f"   - Quality Report: {self.processed_path / 'data_quality_issues.json'}")
        print(f"   - Summary: {self.processed_path / 'data_summary.json'}")
        print(f"\n� Next Step: Run 02_data_cleaning.py")
        
        return True

def main():
    """Run Stage 1"""
    
    explorer = DataExplorer()
    success = explorer.run_full_exploration()
    
    if not success:
        print("\n❌ Stage 1 failed. Please check errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())