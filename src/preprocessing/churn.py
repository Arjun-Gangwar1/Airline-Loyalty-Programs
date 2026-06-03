"""Churn definition framework — creates and validates churn labels."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ChurnDefinitionFramework:
    """
    Define and validate customer churn labels.

    Supports three definitions:
      1. Hard churn  — explicit membership cancellation
      2. Activity churn — no flights for N months
      3. Combined churn (recommended) — cancellation OR inactivity
    """

    def __init__(self, prediction_date: str = '2017-12-31'):
        self.processed_path  = Path("data/processed")
        self.output_path     = Path("outputs/figures/churn")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.prediction_date = pd.Timestamp(prediction_date)
        self.definitions     = {}
        self.comparison      = []

    # ── Data loading ────────────────────────────────────────────────────────

    def load_cleaned_data(self):
        try:
            self.loyalty  = pd.read_csv(self.processed_path / 'loyalty_clean.csv')
            self.activity = pd.read_csv(self.processed_path / 'activity_clean.csv')
        except FileNotFoundError:
            raise FileNotFoundError("Stage 2 outputs not found. Run DataCleaner first.")

        if 'activity_date' not in self.activity.columns:
            self.activity['activity_date'] = pd.to_datetime(
                {'year': self.activity['year'], 'month': self.activity['month'], 'day': 1}
            )
        self.activity['activity_date'] = pd.to_datetime(self.activity['activity_date'])
        self.date_col = 'activity_date'

        id_cols = [c for c in self.loyalty.columns if 'loyalty' in c and 'number' in c]
        self.id_col = id_cols[0] if id_cols else 'loyalty_number'

    # ── Churn definitions ────────────────────────────────────────────────────

    def define_hard_churn(self):
        cancel_cols = [c for c in self.loyalty.columns if 'cancel' in c.lower()]
        if not cancel_cols:
            return None
        cancel_col = cancel_cols[0]
        self.loyalty['hard_churn'] = self.loyalty[cancel_col].notna().astype(int)
        rate = self.loyalty['hard_churn'].mean() * 100
        self._register('hard_churn', self.loyalty['hard_churn'], 'Explicit cancellation')
        print(f"Hard churn rate: {rate:.2f}%")
        return self.loyalty['hard_churn']

    def define_activity_churn(self, months_inactive: int = 12):
        historical = self.activity[self.activity[self.date_col] <= self.prediction_date]
        last_activity = historical.groupby(self.id_col)[self.date_col].max()
        cutoff = self.prediction_date - pd.DateOffset(months=months_inactive)

        last_df = pd.DataFrame({self.id_col: last_activity.index, 'last_activity_date': last_activity.values})
        if 'last_activity_date' in self.loyalty.columns:
            self.loyalty = self.loyalty.drop(columns=['last_activity_date'])
        self.loyalty = self.loyalty.merge(last_df, on=self.id_col, how='left')

        col = f'activity_churn_{months_inactive}m'
        self.loyalty[col] = (
            (self.loyalty['last_activity_date'] < cutoff) |
            (self.loyalty['last_activity_date'].isna())
        ).astype(int)

        rate = self.loyalty[col].mean() * 100
        self._register(col, self.loyalty[col], f'No activity for {months_inactive} months')
        print(f"Activity churn ({months_inactive}m) rate: {rate:.2f}%")
        return self.loyalty[col]

    def define_combined_churn(self):
        hard     = self.loyalty.get('hard_churn', pd.Series(0, index=self.loyalty.index))
        activity = self.loyalty.get('activity_churn_12m', pd.Series(0, index=self.loyalty.index))
        self.loyalty['churn'] = ((hard == 1) | (activity == 1)).astype(int)

        rate = self.loyalty['churn'].mean() * 100
        self._register('combined_churn', self.loyalty['churn'], 'Cancelled OR 12-month inactive')
        print(f"Combined churn rate: {rate:.2f}%")
        return self.loyalty['churn']

    # ── Validation ───────────────────────────────────────────────────────────

    def check_temporal_leakage(self):
        churned = self.loyalty[self.loyalty['churn'] == 1][self.id_col]
        issues  = 0
        for cid in churned.head(100):
            future = self.activity[
                (self.activity[self.id_col] == cid) &
                (self.activity[self.date_col] > self.prediction_date)
            ]
            if len(future):
                issues += 1
        if issues:
            print(f"Warning: {issues}/100 churned customers have post-prediction activity (expected for combined definition).")
        else:
            print("Temporal validation passed — no leakage detected in sample.")

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_labels(self):
        self.processed_path.mkdir(parents=True, exist_ok=True)
        labels = self.loyalty[[self.id_col, 'churn']].copy()
        labels.to_csv(self.processed_path / 'churn_labels.csv', index=False)
        self.loyalty.to_csv(self.processed_path / 'loyalty_with_churn.csv', index=False)
        pd.DataFrame(self.comparison).to_csv(self.processed_path / 'churn_comparison.csv', index=False)
        print(f"Saved churn labels — rate: {labels['churn'].mean()*100:.2f}%")

    def run(self, prediction_date: str = None):
        if prediction_date:
            self.prediction_date = pd.Timestamp(prediction_date)
        self.load_cleaned_data()
        self.define_hard_churn()
        self.define_activity_churn(12)
        self.define_combined_churn()
        self.check_temporal_leakage()
        self.save_labels()
        return self.loyalty['churn']

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _register(self, name: str, label: pd.Series, description: str):
        count = int(label.sum())
        rate  = float(label.mean() * 100)
        self.definitions[name] = {'count': count, 'rate': rate, 'description': description}
        self.comparison.append({'Definition': name, 'Churned': count,
                                'Rate (%)': f"{rate:.2f}", 'Description': description})
