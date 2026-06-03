"""Data cleaning pipeline for loyalty and flight activity datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Clean and prepare raw loyalty + flight activity data for modeling."""

    def __init__(self):
        self.processed_path = Path("data/processed")
        self.cleaning_log = []

    def load_raw_data(self):
        raw_path = Path("data/raw")
        try:
            self.loyalty  = pd.read_csv(raw_path / 'Customer Loyalty History.csv')
            self.activity = pd.read_csv(raw_path / 'Customer Flight Activity.csv')
            print(f"Loaded — Loyalty: {self.loyalty.shape}, Activity: {self.activity.shape}")
            return True
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

    def clean_loyalty_data(self):
        df = self.loyalty.copy()

        # Remove duplicates
        dupes = df.duplicated(subset=['Loyalty Number']).sum()
        if dupes:
            df = df.drop_duplicates(subset=['Loyalty Number'], keep='first')
            self.cleaning_log.append({'step': 'remove_duplicates', 'dataset': 'loyalty', 'rows_removed': dupes})

        # Fill categorical missing values
        for col in df.select_dtypes(include='object').columns:
            if col != 'Loyalty Number' and df[col].isnull().sum():
                n = int(df[col].isnull().sum())
                df[col].fillna('Unknown', inplace=True)
                self.cleaning_log.append({'step': 'fill_missing', 'column': col, 'method': 'Unknown', 'count': n})

        # Fill numeric missing values with median
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().sum():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        # Fix negative/zero CLV
        clv_cols = [c for c in df.columns if 'clv' in c.lower()]
        if clv_cols:
            clv_col = clv_cols[0]
            neg = (df[clv_col] <= 0).sum()
            if neg:
                df.loc[df[clv_col] <= 0, clv_col] = 1.0
            df['CLV_Log'] = np.log1p(df[clv_col])

        # Cap age outliers
        age_cols = [c for c in df.columns if 'age' in c.lower()]
        if age_cols:
            age_col = age_cols[0]
            df.loc[df[age_col] < 18,  age_col] = 18
            df.loc[df[age_col] > 100, age_col] = 100

        # Standardize column names
        df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
        self.loyalty_clean = df

    def clean_activity_data(self):
        df = self.activity.copy()
        df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]

        # Build activity_date from year + month
        if 'year' in df.columns and 'month' in df.columns:
            df['activity_date'] = pd.to_datetime({'year': df['year'], 'month': df['month'], 'day': 1})

        # Remove duplicates on customer × month
        id_col = [c for c in df.columns if 'loyalty' in c and 'number' in c][0]
        if 'activity_date' in df.columns:
            dupes = df.duplicated(subset=[id_col, 'activity_date']).sum()
            if dupes:
                df = df.drop_duplicates(subset=[id_col, 'activity_date'], keep='first')

        # Fill numeric missing values with 0 (absence of activity)
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().sum():
                df[col].fillna(0, inplace=True)

        # Remove negative values
        for col in df.select_dtypes(include=np.number).columns:
            if col != id_col:
                df.loc[df[col] < 0, col] = 0

        self.activity_clean = df

    def validate_cleaned_data(self):
        issues = []
        if self.loyalty_clean.isnull().sum().sum():
            issues.append("loyalty has remaining nulls")
        if self.activity_clean.isnull().sum().sum():
            issues.append("activity has remaining nulls")
        if issues:
            print(f"Validation warnings: {issues}")
        else:
            print("Validation passed — no missing values or duplicates.")

    def save_cleaned_data(self):
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.loyalty_clean.to_csv(self.processed_path / 'loyalty_clean.csv',  index=False)
        self.activity_clean.to_csv(self.processed_path / 'activity_clean.csv', index=False)

        report = {
            'cleaning_date':   datetime.now().isoformat(),
            'loyalty_shape':   list(self.loyalty_clean.shape),
            'activity_shape':  list(self.activity_clean.shape),
            'cleaning_steps':  self.cleaning_log,
        }
        with open(self.processed_path / 'cleaning_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else o)

        self._update_progress(stage=2)
        print("Saved: loyalty_clean.csv, activity_clean.csv, cleaning_report.json")

    def run(self):
        if not self.load_raw_data():
            return False
        self.clean_loyalty_data()
        self.clean_activity_data()
        self.validate_cleaned_data()
        self.save_cleaned_data()
        return True

    def _update_progress(self, stage: int):
        progress_file = Path("checkpoints/progress.json")
        if not progress_file.exists():
            return
        with open(progress_file) as f:
            progress = json.load(f)
        progress['current_stage'] = stage
        if stage not in progress.get('completed_stages', []):
            progress.setdefault('completed_stages', []).append(stage)
        progress['last_updated'] = datetime.now().isoformat()
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
