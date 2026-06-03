"""
Feature engineering — builds 30+ behavioral features for churn prediction.

Feature categories:
  RFM        — recency, frequency, monetary value
  Momentum   — short vs long-term flight trends
  Volatility — consistency and variance of behavior
  Temporal   — seasonal patterns and travel timing
  Psychology — points earn/burn ratio, redemption engagement
  Trajectory — year-over-year growth trends
  Engagement — composite health score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Build behavioral feature matrix from cleaned loyalty + activity data."""

    def __init__(self, prediction_date: str = '2017-12-31'):
        self.processed_path  = Path("data/processed")
        self.final_path      = Path("data/final")
        self.prediction_date = pd.Timestamp(prediction_date)
        self.features        = None

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_data(self):
        self.loyalty  = pd.read_csv(self.processed_path / 'loyalty_with_churn.csv')
        self.activity = pd.read_csv(self.processed_path / 'activity_clean.csv')

        if 'activity_date' not in self.activity.columns:
            self.activity['activity_date'] = pd.to_datetime(
                {'year': self.activity['year'], 'month': self.activity['month'], 'day': 1}
            )
        self.activity['activity_date'] = pd.to_datetime(self.activity['activity_date'])

        id_cols = [c for c in self.loyalty.columns if 'loyalty' in c and 'number' in c]
        self.id_col = id_cols[0] if id_cols else 'loyalty_number'

        # Historical activity only (no future leakage)
        self.hist = self.activity[self.activity['activity_date'] <= self.prediction_date].copy()
        print(f"Loaded {len(self.loyalty):,} customers, {len(self.hist):,} historical activity records.")

    # ── Feature groups ────────────────────────────────────────────────────────

    def _rfm_features(self) -> pd.DataFrame:
        grp  = self.hist.groupby(self.id_col)
        flight_col = [c for c in self.hist.columns if 'flight' in c and 'total' in c][0] \
                     if any('flight' in c and 'total' in c for c in self.hist.columns) \
                     else [c for c in self.hist.columns if 'flight' in c][0]

        last_date   = grp['activity_date'].max()
        recency     = (self.prediction_date - last_date).dt.days
        frequency   = grp[flight_col].sum()
        active_mo   = grp['activity_date'].nunique()
        avg_monthly = frequency / active_mo.clip(lower=1)

        rfm = pd.DataFrame({
            self.id_col:       recency.index,
            'recency_days':    recency.values,
            'total_flights':   frequency.values,
            'active_months':   active_mo.values,
            'avg_flights_mo':  avg_monthly.values,
        })

        # Merge CLV from loyalty
        clv_col = [c for c in self.loyalty.columns if 'clv' in c.lower() and 'log' not in c.lower()]
        if clv_col:
            rfm = rfm.merge(self.loyalty[[self.id_col, clv_col[0]]], on=self.id_col, how='left')
            rfm.rename(columns={clv_col[0]: 'clv'}, inplace=True)

        return rfm

    def _momentum_features(self) -> pd.DataFrame:
        flight_col = [c for c in self.hist.columns if 'flight' in c][0]
        grp        = self.hist.groupby(self.id_col)

        cutoff_3m = self.prediction_date - pd.DateOffset(months=3)
        cutoff_6m = self.prediction_date - pd.DateOffset(months=6)

        recent_3m = self.hist[self.hist['activity_date'] > cutoff_3m].groupby(self.id_col)[flight_col].sum()
        recent_6m = self.hist[self.hist['activity_date'] > cutoff_6m].groupby(self.id_col)[flight_col].sum()
        total     = grp[flight_col].sum()

        momentum = pd.DataFrame(index=total.index)
        momentum[self.id_col]       = momentum.index
        momentum['flights_3m']      = recent_3m.reindex(momentum.index, fill_value=0)
        momentum['flights_6m']      = recent_6m.reindex(momentum.index, fill_value=0)
        momentum['momentum_ratio']  = (momentum['flights_3m'] / momentum['flights_6m'].clip(lower=1)).clip(0, 5)
        momentum['trend_direction'] = (momentum['momentum_ratio'] > 1).astype(int)  # 1=growing, 0=declining
        return momentum.reset_index(drop=True)

    def _volatility_features(self) -> pd.DataFrame:
        flight_col  = [c for c in self.hist.columns if 'flight' in c][0]
        grp         = self.hist.groupby(self.id_col)[flight_col]
        std_flights = grp.std().fillna(0)
        mean_flights = grp.mean().clip(lower=0.01)
        cv          = (std_flights / mean_flights).rename('cv_flights')
        consistency = (1 / (1 + cv)).rename('consistency_score')

        vol = pd.concat([std_flights.rename('std_flights'), cv, consistency], axis=1)
        vol[self.id_col] = vol.index
        return vol.reset_index(drop=True)

    def _temporal_features(self) -> pd.DataFrame:
        flight_col    = [c for c in self.hist.columns if 'flight' in c][0]
        self.hist['month_num'] = self.hist['activity_date'].dt.month

        summer_months  = [6, 7, 8]
        holiday_months = [11, 12, 1]

        summer  = self.hist[self.hist['month_num'].isin(summer_months)].groupby(self.id_col)[flight_col].sum()
        holiday = self.hist[self.hist['month_num'].isin(holiday_months)].groupby(self.id_col)[flight_col].sum()
        total   = self.hist.groupby(self.id_col)[flight_col].sum().clip(lower=1)

        temp = pd.DataFrame(index=total.index)
        temp[self.id_col]        = temp.index
        temp['summer_ratio']     = (summer.reindex(temp.index, fill_value=0)  / total).values
        temp['holiday_ratio']    = (holiday.reindex(temp.index, fill_value=0) / total).values
        temp['seasonality_score'] = (temp['summer_ratio'] + temp['holiday_ratio'])
        return temp.reset_index(drop=True)

    def _psychology_features(self) -> pd.DataFrame:
        points_earned  = [c for c in self.hist.columns if 'points' in c and 'accumulated' in c]
        points_redeemed = [c for c in self.hist.columns if 'points' in c and 'redeemed' in c]

        psych = pd.DataFrame({self.id_col: self.hist[self.id_col].unique()})

        if points_earned and points_redeemed:
            grp  = self.hist.groupby(self.id_col)
            earn = grp[points_earned[0]].sum()
            burn = grp[points_redeemed[0]].sum()
            psych = psych.set_index(self.id_col)
            psych['total_earned']    = earn.reindex(psych.index, fill_value=0)
            psych['total_redeemed']  = burn.reindex(psych.index, fill_value=0)
            psych['redemption_rate'] = (psych['total_redeemed'] /
                                        psych['total_earned'].clip(lower=1)).clip(0, 1)
            psych['earn_burn_ratio'] = (psych['total_earned'] /
                                        psych['total_redeemed'].clip(lower=1)).clip(0, 100)
            psych[self.id_col]       = psych.index
            psych = psych.reset_index(drop=True)
        return psych

    def _trajectory_features(self) -> pd.DataFrame:
        flight_col   = [c for c in self.hist.columns if 'flight' in c][0]
        cutoff_1y    = self.prediction_date - pd.DateOffset(years=1)
        cutoff_2y    = self.prediction_date - pd.DateOffset(years=2)

        last_year  = self.hist[self.hist['activity_date'] > cutoff_1y].groupby(self.id_col)[flight_col].sum()
        prev_year  = self.hist[
            (self.hist['activity_date'] > cutoff_2y) &
            (self.hist['activity_date'] <= cutoff_1y)
        ].groupby(self.id_col)[flight_col].sum()

        traj = pd.DataFrame(index=last_year.index.union(prev_year.index))
        traj[self.id_col]    = traj.index
        traj['flights_ly']   = last_year.reindex(traj.index, fill_value=0)
        traj['flights_py']   = prev_year.reindex(traj.index, fill_value=0)
        traj['yoy_growth']   = ((traj['flights_ly'] - traj['flights_py']) /
                                 traj['flights_py'].clip(lower=1)).clip(-2, 5)
        return traj.reset_index(drop=True)

    def _engagement_score(self, features: pd.DataFrame) -> pd.Series:
        """Composite engagement health score 0–100."""
        score = pd.Series(50.0, index=features.index)

        if 'recency_days' in features.columns:
            recency_norm = (1 - features['recency_days'].clip(0, 365) / 365) * 20
            score += recency_norm

        if 'consistency_score' in features.columns:
            score += features['consistency_score'] * 20

        if 'redemption_rate' in features.columns:
            score += features['redemption_rate'] * 20

        if 'trend_direction' in features.columns:
            score += features['trend_direction'] * 10

        if 'yoy_growth' in features.columns:
            score += features['yoy_growth'].clip(-1, 1) * 10

        return score.clip(0, 100).rename('engagement_score')

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def build_features(self) -> pd.DataFrame:
        print("Building features...")
        rfm       = self._rfm_features()
        momentum  = self._momentum_features()
        volatility = self._volatility_features()
        temporal  = self._temporal_features()
        psych     = self._psychology_features()
        traj      = self._trajectory_features()

        # Merge all feature groups on id_col
        feat = rfm.copy()
        for df in [momentum, volatility, temporal, psych, traj]:
            overlap = [c for c in df.columns if c != self.id_col and c in feat.columns]
            df = df.drop(columns=overlap, errors='ignore')
            feat = feat.merge(df, on=self.id_col, how='left')

        feat['engagement_score'] = self._engagement_score(feat)
        feat = feat.fillna(0)

        # Attach churn label
        churn_df = self.loyalty[[self.id_col, 'churn']].drop_duplicates()
        feat = feat.merge(churn_df, on=self.id_col, how='left')

        self.features = feat
        print(f"Feature matrix: {feat.shape[0]:,} customers × {feat.shape[1]} features")
        return feat

    def save_features(self):
        self.final_path.mkdir(parents=True, exist_ok=True)
        self.features.to_csv(self.final_path / 'customer_features.csv', index=False)
        print(f"Saved: data/final/customer_features.csv")

    def run(self) -> pd.DataFrame:
        self.load_data()
        self.build_features()
        self.save_features()
        return self.features
