"""
STAGE 4: FEATURE ENGINEERING
=============================

Builds the feature matrix for churn prediction from cleaned loyalty +
flight activity data.

CHECKPOINT: After this stage, you'll have:
✅ 40+ behavioral features per customer
✅ Flight behavior metrics (frequency, consistency, momentum)
✅ Points / engagement metrics (redemption rate, earn-burn ratio)
✅ Recency & tenure features
✅ Encoded categorical features
✅ Composite engagement health score

Expected Time: 20-30 minutes
Expected Outputs:
- data/final/customer_features.csv  (full feature matrix)
- outputs/figures/features/         (correlation heatmap + importance preview)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# PREDICTION_DATE is auto-detected from data in load_data() below.
PREDICTION_DATE = pd.Timestamp('2017-12-31')   # default — overridden at runtime

TIER_MAP       = {'Star': 1, 'Aurora': 2, 'Nova': 3}
EDUCATION_MAP  = {
    'High School or Below': 1,
    'College': 2,
    'Bachelor': 3,
    'Master': 4,
    'Doctor': 5,
}


class FeatureEngineer:
    """
    Build a 40+ feature matrix from cleaned loyalty + activity data.

    Feature groups:
      1. Flight behavior  — frequency, consistency, momentum
      2. Distance         — total & average travel distance
      3. Points           — earn, burn, balance, redemption rate
      4. Recency          — last activity, months silent
      5. Tenure           — loyalty program seniority
      6. Engagement       — composite health score (0-100)
      7. Demographics     — encoded card tier, education, gender
    """

    def __init__(self):
        self.processed_path = Path("data/processed")
        self.final_path     = Path("data/final")
        self.fig_path       = Path("outputs/figures/features")
        self.final_path.mkdir(parents=True, exist_ok=True)
        self.fig_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────

    def load_data(self):
        global PREDICTION_DATE

        print("\n" + "=" * 60)
        print("STAGE 4: FEATURE ENGINEERING")
        print("=" * 60)

        print("\n📂 Loading cleaned data from Stage 2 & 3...")
        try:
            self.loyalty  = pd.read_csv(self.processed_path / 'loyalty_with_churn.csv')
            self.activity = pd.read_csv(self.processed_path / 'activity_clean.csv')
            self.activity['activity_date'] = pd.to_datetime(self.activity['activity_date'])
            print(f"   ✓ Loyalty:  {self.loyalty.shape}")
            print(f"   ✓ Activity: {self.activity.shape}")

            # Auto-detect prediction date: max activity date minus 3 months
            max_date = self.activity['activity_date'].max()
            PREDICTION_DATE = max_date - pd.DateOffset(months=3)
            print(f"   ✓ Prediction date (auto): {PREDICTION_DATE.date()}")
            print(f"   ✓ Activity data ends    : {max_date.date()}")

            # Validate churn column from Stage 3
            churn_col = next((c for c in ['churn', 'churned'] if c in self.loyalty.columns), None)
            if churn_col and churn_col != 'churn':
                self.loyalty = self.loyalty.rename(columns={churn_col: 'churn'})

            if 'churn' in self.loyalty.columns:
                churn_rate = self.loyalty['churn'].mean()
                n_classes  = self.loyalty['churn'].nunique()
                print(f"   ✓ Churn rate (Stage 3) : {churn_rate*100:.1f}%")
                if n_classes < 2 or churn_rate > 0.90 or churn_rate < 0.02:
                    print(f"\n⚠️  Stage 3 churn rate is extreme ({churn_rate*100:.1f}%).")
                    print("   Building fallback churn definition from activity data...")
                    self.loyalty['churn'] = self._fallback_churn()
                    print(f"   ✓ Fallback churn rate  : {self.loyalty['churn'].mean()*100:.2f}%")

            return True
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print("   Please run 02_data_cleaning.py and 03_churn_definition.py first.")
            return False

    def _fallback_churn(self) -> pd.Series:
        """
        Self-healing churn definition used when Stage 3 output is extreme.
        Churned = no flight activity in the last 6 months of the dataset.
        """
        max_date  = self.activity['activity_date'].max()
        cutoff    = max_date - pd.DateOffset(months=6)
        last_act  = self.activity.groupby('loyalty_number')['activity_date'].max()
        churn_map = (last_act < cutoff).astype(int)

        # Also include hard churn (explicit cancellations) if column exists
        cancel_cols = [c for c in self.loyalty.columns if 'cancel' in c.lower() and 'year' in c.lower()]
        hard_churn  = self.loyalty['loyalty_number'].map(
            self.loyalty.set_index('loyalty_number')[cancel_cols[0]].notna().astype(int)
            if cancel_cols else pd.Series(0, index=self.loyalty.index)
        ).fillna(0).astype(int)

        churn_series = (
            self.loyalty['loyalty_number'].map(churn_map).fillna(1).astype(int) | hard_churn
        ).astype(int)
        return churn_series

    # ── 2. Flight behavior features ──────────────────────────────────────────

    def _flight_features(self) -> pd.DataFrame:
        print("\n✈️  Building flight behavior features...")
        grp = self.activity.groupby('loyalty_number')

        flights = grp['total_flights'].sum().rename('total_flights')
        dist    = grp['distance'].sum().rename('total_distance')
        months  = grp['activity_date'].nunique().rename('active_months')

        # months actually recorded in the dataset per customer
        months_recorded = (
            self.activity.groupby('loyalty_number')
            .apply(lambda x: (x['year'].max() - x['year'].min()) * 12 + x['month'].max())
            .rename('months_recorded')
        )

        df = pd.concat([flights, dist, months, months_recorded], axis=1).reset_index()

        df['avg_flights_per_month']  = df['total_flights']  / df['active_months'].clip(lower=1)
        df['avg_distance_per_month'] = df['total_distance'] / df['active_months'].clip(lower=1)
        df['active_month_ratio']     = df['active_months']  / df['months_recorded'].clip(lower=1)

        # Per-customer std of monthly flights (consistency)
        std_df = (
            self.activity.groupby('loyalty_number')['total_flights']
            .agg(['max', 'std'])
            .rename(columns={'max': 'max_flights_month', 'std': 'flight_std'})
            .fillna(0)
            .reset_index()
        )
        df = df.merge(std_df, on='loyalty_number', how='left')
        df['flight_consistency'] = np.where(
            df['avg_flights_per_month'] > 0,
            1 - (df['flight_std'] / df['avg_flights_per_month'].clip(lower=0.01)).clip(0, 1),
            0
        )

        print(f"   ✓ {len(df):,} customers | {df.shape[1]-1} flight features")
        return df

    # ── 3. Points / engagement features ─────────────────────────────────────

    def _points_features(self) -> pd.DataFrame:
        print("💎 Building points & engagement features...")
        grp = self.activity.groupby('loyalty_number')

        earned   = grp['points_accumulated'].sum().rename('total_points_accumulated')
        redeemed = grp['points_redeemed'].sum().rename('total_points_redeemed')
        avg_pts  = (earned / self.activity.groupby('loyalty_number')
                    ['activity_date'].nunique().clip(lower=1)).rename('avg_points_per_month')

        df = pd.concat([earned, redeemed, avg_pts], axis=1).reset_index()
        df['redemption_rate'] = (
            df['total_points_redeemed'] /
            df['total_points_accumulated'].clip(lower=1)
        ).clip(0, 1)
        df['points_balance'] = (df['total_points_accumulated'] -
                                df['total_points_redeemed']).clip(lower=0)

        print(f"   ✓ {df.shape[1]-1} points features")
        return df

    # ── 4. Recency features ──────────────────────────────────────────────────

    def _recency_features(self) -> pd.DataFrame:
        print("🕐 Building recency features...")
        last_date = (
            self.activity.groupby('loyalty_number')['activity_date']
            .max()
            .rename('last_date')
        )
        df = last_date.reset_index()
        df['recency_months'] = (
            (PREDICTION_DATE - df['last_date']) / pd.Timedelta(days=30.44)
        ).clip(lower=0).round(1)
        df['last_active_month'] = df['last_date'].dt.month
        df = df.drop(columns=['last_date'])
        print(f"   ✓ {df.shape[1]-1} recency features")
        return df

    # ── 5. Momentum features ─────────────────────────────────────────────────

    def _momentum_features(self) -> pd.DataFrame:
        print("📈 Building momentum features...")
        act = self.activity.copy()

        y2017 = act[act['year'] == 2017].groupby('loyalty_number')['total_flights'].sum()
        y2018 = act[act['year'] == 2018].groupby('loyalty_number')['total_flights'].sum()
        q4    = act[act['month'].isin([10, 11, 12])].groupby('loyalty_number')['total_flights'].sum()

        all_ids = self.loyalty['loyalty_number']
        df = pd.DataFrame({'loyalty_number': all_ids})
        df['flights_2017']         = df['loyalty_number'].map(y2017).fillna(0)
        df['flights_2018']         = df['loyalty_number'].map(y2018).fillna(0)
        df['momentum_h2_minus_h1'] = df['flights_2018'] - df['flights_2017']
        df['q4_flights']           = df['loyalty_number'].map(q4).fillna(0)
        print(f"   ✓ {df.shape[1]-1} momentum features")
        return df.drop(columns=['flights_2017', 'flights_2018'])

    # ── 6. Loyalty / tenure features ─────────────────────────────────────────

    def _loyalty_features(self) -> pd.DataFrame:
        print("🎖️  Building loyalty & demographic features...")
        df = self.loyalty[[
            'loyalty_number', 'gender', 'education', 'salary', 'marital_status',
            'loyalty_card', 'clv', 'clv_log', 'enrollment_type',
            'enrollment_year', 'enrollment_month', 'churn',
        ]].copy()

        # Salary missing flag
        df['salary_missing'] = df['salary'].isna().astype(int)
        df['salary'] = df['salary'].fillna(df['salary'].median())

        # Tenure in months from enrollment to prediction date
        df['enrollment_date_num'] = (
            df['enrollment_year'] * 12 + df['enrollment_month']
        )
        pred_num = PREDICTION_DATE.year * 12 + PREDICTION_DATE.month
        df['tenure_months'] = (pred_num - df['enrollment_date_num']).clip(lower=0)

        # Encode categoricals
        df['tier_numeric']       = df['loyalty_card'].map(TIER_MAP).fillna(1).astype(int)
        df['education_numeric']  = df['education'].map(EDUCATION_MAP).fillna(2).astype(int)
        df['is_female']          = (df['gender'].str.lower() == 'female').astype(int)
        df['is_married']         = (df['marital_status'].str.lower() == 'married').astype(int)
        df['is_promo_enrollment']= (df['enrollment_type'].str.lower() == '2018 promotion').astype(int)
        df['has_high_clv']       = (df['clv'] > df['clv'].quantile(0.75)).astype(int)

        print(f"   ✓ {df.shape[1]-1} loyalty/demographic features")
        return df

    # ── 7. Engagement health score ────────────────────────────────────────────

    def _engagement_health_score(self, features: pd.DataFrame) -> pd.Series:
        """Composite score 0–100 combining recency, activity, consistency, redemption."""
        score = pd.Series(0.0, index=features.index)

        if 'recency_months' in features.columns:
            # 0 months away = 25 points, 24+ months = 0 points
            score += (1 - (features['recency_months'].clip(0, 24) / 24)) * 25

        if 'active_month_ratio' in features.columns:
            score += features['active_month_ratio'].clip(0, 1) * 25

        if 'flight_consistency' in features.columns:
            score += features['flight_consistency'].clip(0, 1) * 25

        if 'redemption_rate' in features.columns:
            score += features['redemption_rate'].clip(0, 1) * 25

        return score.clip(0, 100).rename('engagement_health_score')

    # ── 8. Merge & save ───────────────────────────────────────────────────────

    def build_features(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("BUILDING FEATURE MATRIX")
        print("=" * 60)

        loyalty_df  = self._loyalty_features()
        flight_df   = self._flight_features()
        points_df   = self._points_features()
        recency_df  = self._recency_features()
        momentum_df = self._momentum_features()

        # Merge everything on loyalty_number
        feat = loyalty_df.copy()
        for df in [flight_df, points_df, recency_df, momentum_df]:
            feat = feat.merge(df, on='loyalty_number', how='left')

        feat = feat.fillna(0)
        feat['engagement_health_score'] = self._engagement_health_score(feat)

        churn_rate = feat['churn'].mean() * 100 if 'churn' in feat.columns else 0
        print(f"\n✅ Final feature matrix: {feat.shape[0]:,} customers × {feat.shape[1]} features")
        print(f"   Churn rate in feature matrix: {churn_rate:.2f}%")
        return feat

    def save_features(self, features: pd.DataFrame):
        features.to_csv(self.final_path / 'customer_features.csv', index=False)
        print(f"✓ Saved: data/final/customer_features.csv")

    # ── 9. Visualise feature correlations ────────────────────────────────────

    def plot_correlation_heatmap(self, features: pd.DataFrame):
        numeric_cols = [c for c in features.select_dtypes(include=np.number).columns
                        if c not in ('loyalty_number',)]
        corr = features[numeric_cols].corr()
        churn_corr = corr['churn'].drop('churn').sort_values(key=abs, ascending=False).head(15)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Top 15 features correlated with churn
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in churn_corr.values]
        axes[0].barh(churn_corr.index[::-1], churn_corr.values[::-1], color=colors[::-1], alpha=0.8)
        axes[0].axvline(0, color='black', linewidth=0.8)
        axes[0].set_title('Top 15 Features Correlated with Churn', fontweight='bold')
        axes[0].set_xlabel('Pearson Correlation')
        axes[0].grid(True, alpha=0.3)

        # Full correlation heatmap (top 15 features)
        top_feats = churn_corr.index.tolist() + ['churn']
        sns.heatmap(
            features[top_feats].corr(),
            ax=axes[1], cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            square=True, linewidths=0.5
        )
        axes[1].set_title('Correlation Heatmap — Top Features', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.fig_path / 'feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: outputs/figures/features/feature_correlations.png")

    def print_feature_summary(self, features: pd.DataFrame):
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        groups = {
            'Flight Behavior':   ['total_flights', 'avg_flights_per_month', 'flight_consistency',
                                  'active_month_ratio', 'max_flights_month'],
            'Distance':          ['total_distance', 'avg_distance_per_month'],
            'Points/Engagement': ['total_points_accumulated', 'total_points_redeemed',
                                  'redemption_rate', 'points_balance'],
            'Recency':           ['recency_months', 'last_active_month'],
            'Momentum':          ['momentum_h2_minus_h1', 'q4_flights'],
            'Demographics':      ['tier_numeric', 'clv', 'clv_log', 'tenure_months',
                                  'engagement_health_score'],
        }
        total = 0
        for group, cols in groups.items():
            present = [c for c in cols if c in features.columns]
            total += len(present)
            print(f"\n  {group} ({len(present)} features):")
            for col in present:
                mean = features[col].mean()
                print(f"    • {col:<35} mean={mean:.3f}")
        print(f"\n  Total features: {features.shape[1]-1} (excluding loyalty_number)")
        print(f"  Churn rate:     {features['churn'].mean()*100:.2f}%")
        print(f"  Customers:      {len(features):,}")

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run_feature_engineering_pipeline(self):
        if not self.load_data():
            return False

        features = self.build_features()
        self.save_features(features)
        self.plot_correlation_heatmap(features)
        self.print_feature_summary(features)

        # Update progress checkpoint
        progress_file = Path("checkpoints/progress.json")
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
            progress['current_stage'] = 4
            if 4 not in progress.get('completed_stages', []):
                progress.setdefault('completed_stages', []).append(4)
            progress['last_updated'] = datetime.now().isoformat()
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

        print("\n" + "=" * 60)
        print("✅ STAGE 4 COMPLETE")
        print("=" * 60)
        print(f"\n📊 Feature Matrix:")
        print(f"   {len(features):,} customers × {features.shape[1]} features")
        print(f"\n📁 Outputs:")
        print(f"   - data/final/customer_features.csv")
        print(f"   - outputs/figures/features/feature_correlations.png")
        print(f"\n➡️  Next Step: Run 05_customer_segmentation.py")
        return True


def main():
    engineer = FeatureEngineer()
    success  = engineer.run_feature_engineering_pipeline()
    if not success:
        print("\n❌ Stage 4 failed. Please check errors above.")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
