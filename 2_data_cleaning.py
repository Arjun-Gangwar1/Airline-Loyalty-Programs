"""
STAGE 2: DATA CLEANING & PREPARATION
====================================

This script cleans data based on issues found in Stage 1.

CHECKPOINT: After this stage, you'll have:
✅ Clean loyalty dataset
✅ Clean activity dataset
✅ Proper data types
✅ Handled missing values
✅ Removed duplicates

Can safely stop after this stage and resume with Stage 3.

Expected Time: 20-30 minutes
Expected Outputs:
- loyalty_clean.csv
- activity_clean.csv
- cleaning_report.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    Comprehensive data cleaning pipeline
    """
    
    def __init__(self):
        self.processed_path = Path("data/processed")
        self.cleaning_log = []
        
    def load_raw_data(self):
        """Load data from Stage 1 checkpoint"""
        
        print("\n" + "=" * 60)
        print("STAGE 2: DATA CLEANING")
        print("=" * 60)
        
        print("\n� Loading data from Stage 1 checkpoint...")
        
        raw_path = Path("data/raw")
        try:
            # Load directly from raw CSVs to avoid any Stage 1 type conversion issues
            self.loyalty = pd.read_csv(raw_path / 'Customer Loyalty History.csv')
            self.activity = pd.read_csv(raw_path / 'Customer Flight Activity.csv')
            print(f"   ✓ Loyalty: {self.loyalty.shape}")
            print(f"   ✓ Activity: {self.activity.shape}")
            return True
        except FileNotFoundError:
            print("\n❌ ERROR: Raw data files not found in data/raw/!")
            print("   Expected: 'Customer Loyalty History.csv' and 'Customer Flight Activity.csv'")
            return False
    
    def clean_loyalty_data(self):
        """Clean customer loyalty history"""
        
        print("\n" + "=" * 60)
        print("1. CLEANING LOYALTY HISTORY")
        print("=" * 60)
        
        df = self.loyalty.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        print("\n� Checking for duplicates...")
        duplicates = df.duplicated(subset=['Loyalty Number']).sum()
        if duplicates > 0:
            print(f"   ⚠️  Found {duplicates} duplicate customers")
            df = df.drop_duplicates(subset=['Loyalty Number'], keep='first')
            self.cleaning_log.append({
                'step': 'remove_duplicates',
                'dataset': 'loyalty',
                'rows_removed': duplicates
            })
        else:
            print("   ✓ No duplicates found")
        
        # Handle missing values
        print("\n� Handling missing values...")
        
        # For categorical columns, fill with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Loyalty Number' and df[col].isnull().sum() > 0:
                missing_count = df[col].isnull().sum()
                df[col].fillna('Unknown', inplace=True)
                print(f"   ✓ Filled {missing_count} missing values in '{col}' with 'Unknown'")
                self.cleaning_log.append({
                    'step': 'fill_missing',
                    'dataset': 'loyalty',
                    'column': col,
                    'method': 'Unknown',
                    'count': int(missing_count)
                })
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                missing_count = df[col].isnull().sum()
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   ✓ Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
                self.cleaning_log.append({
                    'step': 'fill_missing',
                    'dataset': 'loyalty',
                    'column': col,
                    'method': 'median',
                    'count': int(missing_count)
                })
        
        # Clean CLV
        clv_cols = [col for col in df.columns if 'clv' in col.lower()]
        if clv_cols:
            clv_col = clv_cols[0]
            print(f"\n� Cleaning CLV column: {clv_col}")
            
            # Handle negative/zero CLV
            negative_clv = (df[clv_col] <= 0).sum()
            if negative_clv > 0:
                print(f"   ⚠️  Found {negative_clv} customers with CLV ≤ 0")
                # Set to small positive value
                df.loc[df[clv_col] <= 0, clv_col] = 1.0
                print(f"   ✓ Set negative/zero CLV to 1.0")
                self.cleaning_log.append({
                    'step': 'fix_negative_clv',
                    'dataset': 'loyalty',
                    'count': int(negative_clv)
                })
            
            # Create log-transformed CLV for modeling
            df['CLV_Log'] = np.log1p(df[clv_col])
            print(f"   ✓ Created CLV_Log column")
        
        # Clean age
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        if age_cols:
            age_col = age_cols[0]
            print(f"\n� Cleaning age column: {age_col}")
            
            # Cap suspicious ages
            suspicious_low = (df[age_col] < 18).sum()
            suspicious_high = (df[age_col] > 100).sum()
            
            if suspicious_low > 0:
                df.loc[df[age_col] < 18, age_col] = 18
                print(f"   ✓ Capped {suspicious_low} ages < 18 to 18")
            
            if suspicious_high > 0:
                df.loc[df[age_col] > 100, age_col] = 100
                print(f"   ✓ Capped {suspicious_high} ages > 100 to 100")
        
        # Standardize column names (remove spaces, lowercase)
        print("\n� Standardizing column names...")
        old_names = df.columns.tolist()
        new_names = [col.strip().replace(' ', '_').lower() for col in df.columns]
        df.columns = new_names
        print("   ✓ Column names standardized")
        
        # Save cleaned data
        self.loyalty_clean = df
        
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        
        print(f"\n� Loyalty Cleaning Summary:")
        print(f"   Initial rows: {initial_rows:,}")
        print(f"   Final rows: {final_rows:,}")
        print(f"   Rows removed: {rows_removed:,}")
    
    def clean_activity_data(self):
        """Clean flight activity data"""
        
        print("\n" + "=" * 60)
        print("2. CLEANING FLIGHT ACTIVITY")
        print("=" * 60)
        
        df = self.activity.copy()
        initial_rows = len(df)
        
        # Standardize column names
        print("\n� Standardizing column names...")
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        print("   ✓ Column names standardized")
        
        # Convert date columns
        # Convert year + month integers into a proper date column
        print(f"\n Converting date columns...")
        if 'year' in df.columns and 'month' in df.columns:
            df['activity_date'] = pd.to_datetime(
                {'year': df['year'], 'month': df['month'], 'day': 1}
            )
            print(f"   Converted: activity_date from year+month")
        
        # Remove duplicates
        print("\n� Checking for duplicates...")
        # Consider duplicate if same customer, same month
        id_col = [col for col in df.columns if 'loyalty' in col and 'number' in col][0]
        month_col = "activity_date" if "activity_date" in df.columns else None
        
        if month_col:
            duplicates = df.duplicated(subset=[id_col, month_col]).sum()
            if duplicates > 0:
                print(f"   ⚠️  Found {duplicates} duplicate records")
                df = df.drop_duplicates(subset=[id_col, month_col], keep='first')
                self.cleaning_log.append({
                    'step': 'remove_duplicates',
                    'dataset': 'activity',
                    'rows_removed': duplicates
                })
            else:
                print("   ✓ No duplicates found")
        
        # Handle missing values
        print("\n� Handling missing values...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    # Fill numerical with 0 (assumes missing activity = no activity)
                    df[col].fillna(0, inplace=True)
                    print(f"   ✓ Filled missing {col} with 0")
        
        # Ensure no negative values in activity metrics
        print("\n� Checking for negative values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != id_col:  # Don't modify ID column
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    df.loc[df[col] < 0, col] = 0
                    print(f"   ✓ Fixed {negative_count} negative values in '{col}'")
        
        # Save cleaned data
        self.activity_clean = df
        
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        
        print(f"\n� Activity Cleaning Summary:")
        print(f"   Initial rows: {initial_rows:,}")
        print(f"   Final rows: {final_rows:,}")
        print(f"   Rows removed: {rows_removed:,}")
    
    def validate_cleaned_data(self):
        """Validate cleaning was successful"""
        
        print("\n" + "=" * 60)
        print("3. VALIDATION")
        print("=" * 60)
        
        print("\n✅ Checking cleaned data quality...")
        
        # Check for remaining missing values
        loyalty_missing = self.loyalty_clean.isnull().sum().sum()
        activity_missing = self.activity_clean.isnull().sum().sum()
        
        if loyalty_missing == 0:
            print("   ✓ Loyalty: No missing values")
        else:
            print(f"   ⚠️  Loyalty: {loyalty_missing} missing values remain")
        
        if activity_missing == 0:
            print("   ✓ Activity: No missing values")
        else:
            print(f"   ⚠️  Activity: {activity_missing} missing values remain")
        
        # Check for duplicates
        loyalty_dupes = self.loyalty_clean.duplicated().sum()
        activity_dupes = self.activity_clean.duplicated().sum()
        
        if loyalty_dupes == 0:
            print("   ✓ Loyalty: No duplicates")
        else:
            print(f"   ⚠️  Loyalty: {loyalty_dupes} duplicates remain")
        
        if activity_dupes == 0:
            print("   ✓ Activity: No duplicates")
        else:
            print(f"   ⚠️  Activity: {activity_dupes} duplicates remain")
        
        # Check data types
        print("\n� Data Types Check:")
        print("   Loyalty columns:")
        for col, dtype in self.loyalty_clean.dtypes.items():
            print(f"      {col}: {dtype}")
        
        print("\n✅ Data validation complete!")
    
    def save_cleaned_data(self):
        """Save cleaned data for Stage 3"""
        
        print("\n" + "=" * 60)
        print("4. SAVING CLEANED DATA")
        print("=" * 60)
        
        # Save cleaned datasets
        loyalty_file = self.processed_path / 'loyalty_clean.csv'
        activity_file = self.processed_path / 'activity_clean.csv'
        
        self.loyalty_clean.to_csv(loyalty_file, index=False)
        self.activity_clean.to_csv(activity_file, index=False)
        
        print(f"\n✓ Saved: {loyalty_file}")
        print(f"✓ Saved: {activity_file}")
        
        # Save cleaning log
        log_file = self.processed_path / 'cleaning_report.json'
        report = {
            'cleaning_date': datetime.now().isoformat(),
            'loyalty_shape': list(self.loyalty_clean.shape),
            'activity_shape': list(self.activity_clean.shape),
            'cleaning_steps': self.cleaning_log
        }
        
        def convert_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        with open(log_file, 'w') as f:
            json.dump(convert_types(report), f, indent=2)
        
        print(f"✓ Saved: {log_file}")
        
        # Update progress
        progress_file = Path("checkpoints/progress.json")
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        progress['current_stage'] = 2
        if 2 not in progress['completed_stages']:
            progress['completed_stages'].append(2)
        progress['last_updated'] = datetime.now().isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✓ Progress updated")
    
    def run_cleaning_pipeline(self):
        """Execute complete cleaning pipeline"""
        
        # Load data
        if not self.load_raw_data():
            return False
        
        # Clean
        self.clean_loyalty_data()
        self.clean_activity_data()
        self.validate_cleaned_data()
        self.save_cleaned_data()
        
        print("\n" + "=" * 60)
        print("✅ STAGE 2 COMPLETE")
        print("=" * 60)
        print(f"\n� Cleaned Data:")
        print(f"   Loyalty: {self.loyalty_clean.shape}")
        print(f"   Activity: {self.activity_clean.shape}")
        print(f"\n� Outputs:")
        print(f"   - loyalty_clean.csv")
        print(f"   - activity_clean.csv")
        print(f"   - cleaning_report.json")
        print(f"\n� Next Step: Run 03_churn_definition.py")
        
        return True

def main():
    """Run Stage 2"""
    
    cleaner = DataCleaner()
    success = cleaner.run_cleaning_pipeline()
    
    if not success:
        print("\n❌ Stage 2 failed. Please check errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())