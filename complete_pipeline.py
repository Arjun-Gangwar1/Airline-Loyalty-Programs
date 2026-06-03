"""
Complete Airline Loyalty Behavioral Intelligence Pipeline
=========================================================
Handles all stages: data loading, cleaning, churn definition,
feature engineering, segmentation, modeling, retention engine.

Data note: Activity data covers 2017-2018 only (not 2012-2018).
Prediction date = 2017-12-31. Features from 2017, labels from 2018.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                              f1_score, classification_report, confusion_matrix,
                              roc_curve)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ─── PATHS ───────────────────────────────────────────────────────────────────
RAW = Path("data/raw")
PROC = Path("data/processed")
FINAL = Path("data/final")
FIGS = Path("outputs/figures")
MODELS = Path("outputs/models")
REPORTS = Path("outputs/reports")

for p in [PROC, FINAL, FIGS/"exploration", FIGS/"models", FIGS/"segments",
          MODELS, REPORTS, Path("checkpoints"), Path("logs")]:
    p.mkdir(parents=True, exist_ok=True)

PREDICTION_DATE = pd.Timestamp("2017-12-31")

# ─── STEP 1: LOAD & CLEAN ────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 1: LOADING & CLEANING DATA")
print("="*65)

loyalty_raw = pd.read_csv(RAW / "Customer Loyalty History.csv")
activity_raw = pd.read_csv(RAW / "Customer Flight Activity.csv")

loyalty_raw.columns = [c.lower().replace(' ', '_') for c in loyalty_raw.columns]
activity_raw.columns = [c.lower().replace(' ', '_') for c in activity_raw.columns]

print(f"Loyalty History: {loyalty_raw.shape}")
print(f"Flight Activity: {activity_raw.shape}")
print(f"Activity years: {sorted(activity_raw['year'].unique())}")
print(f"Loyalty card dist:\n{loyalty_raw['loyalty_card'].value_counts()}")
print(f"Cancellation rate: {loyalty_raw['cancellation_year'].notna().mean():.1%}")

# Fix salary: fill missing with median
salary_median = loyalty_raw['salary'].median()
loyalty_raw['salary_missing'] = loyalty_raw['salary'].isna().astype(int)
loyalty_raw['salary'] = loyalty_raw['salary'].fillna(salary_median)

# Remove true duplicates from activity (keep first)
activity_dedup = activity_raw.drop_duplicates(
    subset=['loyalty_number', 'year', 'month'], keep='first'
)
print(f"Activity after dedup: {activity_dedup.shape} (removed {len(activity_raw)-len(activity_dedup)} true duplicates)")

# ─── STEP 2: CHURN LABELS ────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 2: DEFINING CHURN LABELS")
print("="*65)

# Hard churn: formal cancellation in 2018
loyalty_raw['hard_churn'] = (
    loyalty_raw['cancellation_year'].notna() &
    (loyalty_raw['cancellation_year'] >= 2018)
).astype(int)

# Activity churn: no flights in all of 2018
activity_2018 = activity_dedup[activity_dedup['year'] == 2018]
customers_with_2018_flights = set(
    activity_2018[activity_2018['total_flights'] > 0]['loyalty_number'].unique()
)
loyalty_raw['activity_churn'] = (~loyalty_raw['loyalty_number'].isin(
    customers_with_2018_flights)).astype(int)

# Combined churn: hard OR activity
loyalty_raw['churned'] = ((loyalty_raw['hard_churn'] == 1) |
                           (loyalty_raw['activity_churn'] == 1)).astype(int)

hard_rate = loyalty_raw['hard_churn'].mean()
act_rate = loyalty_raw['activity_churn'].mean()
combined_rate = loyalty_raw['churned'].mean()

print(f"Hard churn rate:     {hard_rate:.1%} ({loyalty_raw['hard_churn'].sum():,} customers)")
print(f"Activity churn rate: {act_rate:.1%} ({loyalty_raw['activity_churn'].sum():,} customers)")
print(f"Combined churn rate: {combined_rate:.1%} ({loyalty_raw['churned'].sum():,} customers)")

# Save churn labels
churn_labels = loyalty_raw[['loyalty_number', 'hard_churn', 'activity_churn', 'churned']].copy()
churn_labels.to_csv(PROC / "churn_labels.csv", index=False)
print(f"Saved churn labels: {PROC/'churn_labels.csv'}")

# ─── STEP 3: FEATURE ENGINEERING ─────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 3: FEATURE ENGINEERING (using 2017 data only)")
print("="*65)

# Use only 2017 data for features (before prediction date)
activity_2017 = activity_dedup[activity_dedup['year'] == 2017].copy()
activity_2017 = activity_2017.sort_values(['loyalty_number', 'month'])

print(f"Activity 2017 rows: {len(activity_2017):,}")
print(f"Customers with 2017 activity: {activity_2017['loyalty_number'].nunique():,}")

def compute_features(loyalty_df, activity_2017_df):
    """Compute behavioral features per customer from 2017 activity."""

    features_list = []

    # Per-customer 2017 stats
    cust_stats = activity_2017_df.groupby('loyalty_number').agg(
        total_flights_2017=('total_flights', 'sum'),
        avg_flights_per_month=('total_flights', 'mean'),
        max_flights_month=('total_flights', 'max'),
        flight_std=('total_flights', 'std'),
        total_distance=('distance', 'sum'),
        avg_distance_per_month=('distance', 'mean'),
        total_points_accumulated=('points_accumulated', 'sum'),
        total_points_redeemed=('points_redeemed', 'sum'),
        avg_points_per_month=('points_accumulated', 'mean'),
        active_months=('total_flights', lambda x: (x > 0).sum()),
        months_recorded=('total_flights', 'count'),
        last_active_month=('month', lambda x: x[x.map(
            activity_2017_df.loc[x.index, 'total_flights']) > 0].max()
                           if (activity_2017_df.loc[x.index, 'total_flights'] > 0).any()
                           else 0),
    ).reset_index()

    # Last active month as recency (distance from December 2017)
    cust_stats['recency_months'] = 12 - cust_stats['last_active_month'].fillna(0)
    cust_stats['recency_months'] = cust_stats['recency_months'].clip(lower=0)

    # Active month ratio
    cust_stats['active_month_ratio'] = (
        cust_stats['active_months'] / cust_stats['months_recorded'].clip(lower=1)
    )

    # Flight consistency (CV = std/mean)
    cust_stats['flight_consistency'] = np.where(
        cust_stats['avg_flights_per_month'] > 0,
        1 - (cust_stats['flight_std'] / (cust_stats['avg_flights_per_month'] + 1e-6)).clip(0, 1),
        0
    )

    # Redemption rate
    cust_stats['redemption_rate'] = np.where(
        cust_stats['total_points_accumulated'] > 0,
        cust_stats['total_points_redeemed'] / cust_stats['total_points_accumulated'],
        0
    ).clip(0, 2)

    # Momentum: H1 (Jan-Jun) vs H2 (Jul-Dec) 2017
    h1 = activity_2017_df[activity_2017_df['month'] <= 6].groupby('loyalty_number')['total_flights'].sum()
    h2 = activity_2017_df[activity_2017_df['month'] >= 7].groupby('loyalty_number')['total_flights'].sum()
    momentum = (h2 - h1).rename('momentum_h2_minus_h1')
    cust_stats = cust_stats.merge(momentum.reset_index(), on='loyalty_number', how='left')
    cust_stats['momentum_h2_minus_h1'] = cust_stats['momentum_h2_minus_h1'].fillna(0)

    # Recent quarter (Q4 2017)
    q4 = activity_2017_df[activity_2017_df['month'] >= 10].groupby('loyalty_number')['total_flights'].sum()
    q4 = q4.rename('q4_flights')
    cust_stats = cust_stats.merge(q4.reset_index(), on='loyalty_number', how='left')
    cust_stats['q4_flights'] = cust_stats['q4_flights'].fillna(0)

    # Merge with loyalty demographics
    demo_cols = ['loyalty_number', 'gender', 'education', 'salary', 'salary_missing',
                 'marital_status', 'loyalty_card', 'clv', 'enrollment_type',
                 'enrollment_year', 'enrollment_month']
    demo = loyalty_df[demo_cols].copy()

    features = demo.merge(cust_stats, on='loyalty_number', how='left')

    # Fill customers with no 2017 activity
    num_cols = cust_stats.columns.difference(['loyalty_number']).tolist()
    for col in num_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Tenure (months from enrollment to end of 2017)
    features['enrollment_date_num'] = (
        features['enrollment_year'] * 12 + features['enrollment_month']
    )
    features['tenure_months'] = (2017 * 12 + 12) - features['enrollment_date_num']
    features['tenure_months'] = features['tenure_months'].clip(lower=0)

    # Tier encoding
    tier_map = {'Star': 1, 'Nova': 2, 'Aurora': 3}
    features['tier_numeric'] = features['loyalty_card'].map(tier_map).fillna(1)

    # Engagement health score (0–100 composite)
    def min_max_norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    health_components = pd.DataFrame({
        'recency': 1 - min_max_norm(features['recency_months']),
        'frequency': min_max_norm(features['avg_flights_per_month']),
        'consistency': min_max_norm(features['active_month_ratio']),
        'redemption': min_max_norm(features['redemption_rate']),
        'tier': min_max_norm(features['tier_numeric']),
    })
    features['engagement_health_score'] = (health_components.mean(axis=1) * 100).round(1)

    # Education encoding
    edu_map = {'High School or Below': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5}
    features['education_numeric'] = features['education'].map(edu_map).fillna(2)

    # Binary flags
    features['is_female'] = (features['gender'] == 'Female').astype(int)
    features['is_married'] = (features['marital_status'] == 'Married').astype(int)
    features['is_promo_enrollment'] = (features['enrollment_type'] == '2018 Promotion').astype(int)
    features['has_high_clv'] = (features['clv'] > features['clv'].median()).astype(int)

    # log-transform CLV
    features['clv_log'] = np.log1p(features['clv'])

    # Points balance proxy (accumulated - redeemed)
    features['points_balance'] = (
        features['total_points_accumulated'] - features['total_points_redeemed']
    ).clip(lower=0)

    return features


features_df = compute_features(loyalty_raw, activity_2017)

# Merge churn labels
features_df = features_df.merge(
    churn_labels[['loyalty_number', 'churned', 'hard_churn', 'activity_churn']],
    on='loyalty_number', how='left'
)

print(f"Feature matrix shape: {features_df.shape}")
print(f"Feature columns: {list(features_df.select_dtypes(include=np.number).columns)}")

features_df.to_csv(FINAL / "customer_features.csv", index=False)
print(f"Saved: {FINAL/'customer_features.csv'}")

# ─── STEP 4: CUSTOMER SEGMENTATION ────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 4: CUSTOMER SEGMENTATION")
print("="*65)

seg_features = [
    'avg_flights_per_month', 'recency_months', 'active_month_ratio',
    'redemption_rate', 'clv_log', 'tier_numeric', 'tenure_months',
    'engagement_health_score', 'momentum_h2_minus_h1', 'total_distance'
]

X_seg = features_df[seg_features].copy()
scaler_seg = MinMaxScaler()
X_seg_scaled = scaler_seg.fit_transform(X_seg)

# Elbow method
inertias = []
silhouettes = []
from sklearn.metrics import silhouette_score
K_range = range(3, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_seg_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_seg_scaled, labels))

best_k = K_range[np.argmax(silhouettes)]
print(f"Best K by silhouette: {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
features_df['segment_id'] = km_final.fit_predict(X_seg_scaled)

# Name segments by their behavioral profile
seg_profiles = features_df.groupby('segment_id')[seg_features + ['churned']].mean()
print("\nSegment profiles:")
print(seg_profiles.round(2))

# Auto-name segments based on characteristics
def name_segment(row):
    if row['churned'] > 0.7 and row['recency_months'] > 6:
        return 'Silent Drifters'
    elif row['clv_log'] > seg_profiles['clv_log'].median() and row['recency_months'] < 3:
        return 'Premium Loyalists'
    elif row['engagement_health_score'] > seg_profiles['engagement_health_score'].median() and row['churned'] < 0.3:
        return 'Active Champions'
    elif row['redemption_rate'] < 0.1 and row['total_distance'] > seg_profiles['total_distance'].median():
        return 'Miles Hoarders'
    elif row['tier_numeric'] < 1.5 and row['churned'] > 0.5:
        return 'At-Risk Starters'
    elif row['tenure_months'] < seg_profiles['tenure_months'].median() and row['churned'] < 0.4:
        return 'Rising Stars'
    else:
        return 'Seasonal Travelers'

seg_names = {sid: name_segment(seg_profiles.loc[sid]) for sid in seg_profiles.index}

# Ensure unique names
used_names = {}
for sid, name in seg_names.items():
    if name in used_names.values():
        seg_names[sid] = name + f" {sid+1}"
    used_names[sid] = seg_names[sid]

features_df['segment_name'] = features_df['segment_id'].map(seg_names)
print("\nSegment assignment:")
print(features_df['segment_name'].value_counts())
print("\nChurn by segment:")
print(features_df.groupby('segment_name')['churned'].agg(['mean', 'count']).round(3))

features_df.to_csv(FINAL / "customer_segments.csv", index=False)
print(f"Saved: {FINAL/'customer_segments.csv'}")

# ─── STEP 5: MODEL TRAINING ────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 5: MODEL TRAINING & EVALUATION")
print("="*65)

model_features = [
    'avg_flights_per_month', 'recency_months', 'active_month_ratio',
    'redemption_rate', 'clv_log', 'tier_numeric', 'tenure_months',
    'engagement_health_score', 'momentum_h2_minus_h1', 'total_distance',
    'total_points_accumulated', 'active_months', 'q4_flights',
    'flight_consistency', 'salary', 'salary_missing', 'education_numeric',
    'is_female', 'is_married', 'is_promo_enrollment', 'points_balance',
]

X = features_df[model_features].copy()
y = features_df['churned'].copy()

# Impute any remaining NAs
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.1%}")
print(f"Test churn rate: {y_test.mean():.1%}")

# Scale for LR
scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)

# SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {X_train_sm.shape}, class dist: {pd.Series(y_train_sm).value_counts().to_dict()}")

results = {}

# Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
lr.fit(X_train_sm, y_train_sm)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = {
    'AUC': roc_auc_score(y_test, y_prob_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'F1': f1_score(y_test, y_pred_lr),
}
joblib.dump(lr, MODELS / "logistic_regression.pkl")

# Random Forest
print("Training Random Forest...")
X_train_rf, y_train_rf = smote.fit_resample(X_train, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                             max_depth=10, min_samples_leaf=5)
rf.fit(X_train_rf, y_train_rf)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = {
    'AUC': roc_auc_score(y_test, y_prob_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'F1': f1_score(y_test, y_pred_rf),
}
joblib.dump(rf, MODELS / "random_forest.pkl")

# XGBoost
print("Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric='logloss', use_label_encoder=False,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {
    'AUC': roc_auc_score(y_test, y_prob_xgb),
    'Recall': recall_score(y_test, y_pred_xgb),
    'Precision': precision_score(y_test, y_pred_xgb),
    'F1': f1_score(y_test, y_pred_xgb),
}
joblib.dump(xgb_model, MODELS / "xgboost.pkl")
joblib.dump(xgb_model, MODELS / "best_model.pkl")

# Save model comparison
model_comparison = pd.DataFrame(results).T.round(4)
model_comparison.index.name = 'Model'
model_comparison.to_csv(REPORTS / "model_comparison.csv")
print("\nModel Comparison:")
print(model_comparison)

# ─── STEP 6: FEATURE IMPORTANCE ───────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 6: FEATURE IMPORTANCE (XGBoost)")
print("="*65)

importance_df = pd.DataFrame({
    'feature': model_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.head(15).to_string())

importance_df.to_csv(REPORTS / "feature_importance.csv", index=False)

# ─── STEP 7: RETENTION ENGINE ─────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 7: RETENTION ENGINE")
print("="*65)

# Add churn probability to full dataset
features_df['churn_probability'] = xgb_model.predict_proba(
    imputer.transform(features_df[model_features])
)[:, 1]

def get_risk_level(prob):
    if prob >= 0.70:
        return 'High'
    elif prob >= 0.40:
        return 'Medium'
    else:
        return 'Low'

features_df['risk_level'] = features_df['churn_probability'].apply(get_risk_level)

# Revenue at risk
features_df['revenue_at_risk'] = (features_df['clv'] * features_df['churn_probability'] * 0.82).round(2)
features_df['potential_save'] = (features_df['revenue_at_risk'] * 0.27).round(2)

# Offer library per segment × risk level
offer_library = {
    'Silent Drifters': {
        'High':   {'action': 'Urgent Win-Back Campaign',
                   'offer': '8,000 bonus miles + waived change fees for 60 days',
                   'channel': 'SMS + Email + App Push',
                   'timing': 'Within 48 hours',
                   'incentive_cost': 80},
        'Medium': {'action': 'Re-engagement Campaign',
                   'offer': '5,000 bonus miles + 3-month tier extension',
                   'channel': 'Email + SMS',
                   'timing': 'Within 7 days',
                   'incentive_cost': 50},
        'Low':    {'action': 'Soft Re-engagement',
                   'offer': '2,000 bonus miles on next booking',
                   'channel': 'Email',
                   'timing': 'Within 21 days',
                   'incentive_cost': 20},
    },
    'Premium Loyalists': {
        'High':   {'action': 'VIP Retention Call',
                   'offer': 'Complimentary companion ticket + Aurora status guarantee',
                   'channel': 'Personal Call + Premium Email',
                   'timing': 'Within 24 hours',
                   'incentive_cost': 200},
        'Medium': {'action': 'Loyalty Appreciation',
                   'offer': '10,000 bonus miles + lounge access upgrade',
                   'channel': 'Email',
                   'timing': 'Within 3 days',
                   'incentive_cost': 100},
        'Low':    {'action': 'Thank You Offer',
                   'offer': '3,000 bonus miles as loyalty appreciation',
                   'channel': 'Email',
                   'timing': 'Within 30 days',
                   'incentive_cost': 30},
    },
    'At-Risk Starters': {
        'High':   {'action': 'Emergency Activation',
                   'offer': '5,000 bonus miles + first free seat upgrade',
                   'channel': 'SMS + Email',
                   'timing': 'Within 48 hours',
                   'incentive_cost': 60},
        'Medium': {'action': 'Tier Upgrade Incentive',
                   'offer': '3,000 bonus miles + fast-track to Nova card',
                   'channel': 'Email',
                   'timing': 'Within 7 days',
                   'incentive_cost': 40},
        'Low':    {'action': 'Onboarding Nudge',
                   'offer': 'How-to guide + 1,000 bonus miles',
                   'channel': 'Email',
                   'timing': 'Within 14 days',
                   'incentive_cost': 10},
    },
    'Miles Hoarders': {
        'High':   {'action': 'Redemption Activation',
                   'offer': '2x redemption value + flash sale on points',
                   'channel': 'Email + App Push',
                   'timing': 'Within 72 hours',
                   'incentive_cost': 50},
        'Medium': {'action': 'Redeem Reminder',
                   'offer': '50% bonus on next points redemption',
                   'channel': 'Email',
                   'timing': 'Within 10 days',
                   'incentive_cost': 30},
        'Low':    {'action': 'Points Expiry Warning',
                   'offer': 'Points expiry notice + 500 bonus miles',
                   'channel': 'Email',
                   'timing': 'Within 30 days',
                   'incentive_cost': 5},
    },
    'Active Champions': {
        'High':   {'action': 'Loyalty Lock-In',
                   'offer': '6,000 bonus miles + priority check-in for 6 months',
                   'channel': 'Email + App',
                   'timing': 'Within 3 days',
                   'incentive_cost': 70},
        'Medium': {'action': 'Engagement Boost',
                   'offer': '3,000 bonus miles on next flight',
                   'channel': 'Email',
                   'timing': 'Within 14 days',
                   'incentive_cost': 30},
        'Low':    {'action': 'Celebration Offer',
                   'offer': '1,500 bonus miles as appreciation',
                   'channel': 'Email',
                   'timing': 'Within 30 days',
                   'incentive_cost': 15},
    },
    'Rising Stars': {
        'High':   {'action': 'Early Churn Prevention',
                   'offer': '4,000 bonus miles + Nova tier trial for 2 months',
                   'channel': 'Email + SMS',
                   'timing': 'Within 48 hours',
                   'incentive_cost': 45},
        'Medium': {'action': 'Growth Incentive',
                   'offer': '2,000 bonus miles + referral program invite',
                   'channel': 'Email',
                   'timing': 'Within 14 days',
                   'incentive_cost': 20},
        'Low':    {'action': 'Welcome Journey',
                   'offer': 'Educational content + 500 bonus miles',
                   'channel': 'Email',
                   'timing': 'Within 30 days',
                   'incentive_cost': 5},
    },
    'Seasonal Travelers': {
        'High':   {'action': 'Off-Season Activation',
                   'offer': '4,000 bonus miles + exclusive off-peak deal',
                   'channel': 'Email + SMS',
                   'timing': 'Within 72 hours',
                   'incentive_cost': 40},
        'Medium': {'action': 'Seasonal Engagement',
                   'offer': 'Upcoming season preview + 2,000 bonus miles',
                   'channel': 'Email',
                   'timing': 'Within 14 days',
                   'incentive_cost': 20},
        'Low':    {'action': 'Seasonal Reminder',
                   'offer': 'Seasonal deals newsletter',
                   'channel': 'Email',
                   'timing': 'Within 30 days',
                   'incentive_cost': 5},
    },
}

default_offer = {
    'High':   {'action': 'Priority Retention',
               'offer': '5,000 bonus miles + special retention offer',
               'channel': 'Email + SMS',
               'timing': 'Within 48 hours',
               'incentive_cost': 50},
    'Medium': {'action': 'Engagement Campaign',
               'offer': '2,500 bonus miles',
               'channel': 'Email',
               'timing': 'Within 14 days',
               'incentive_cost': 25},
    'Low':    {'action': 'Retention Touch',
               'offer': '1,000 bonus miles',
               'channel': 'Email',
               'timing': 'Within 30 days',
               'incentive_cost': 10},
}

def get_retention_action(row):
    seg = row['segment_name']
    risk = row['risk_level']
    lib = offer_library.get(seg, default_offer)
    offer = lib.get(risk, default_offer[risk])
    return pd.Series({
        'recommended_action': offer['action'],
        'specific_offer': offer['offer'],
        'channel': offer['channel'],
        'timing': offer['timing'],
        'incentive_cost_cad': offer['incentive_cost'],
    })

action_cols = features_df.apply(get_retention_action, axis=1)
features_df = pd.concat([features_df, action_cols], axis=1)

# Retention actions report (all customers, sorted by revenue at risk)
at_risk = features_df[features_df['risk_level'].isin(['High', 'Medium'])].copy()
at_risk = at_risk.sort_values('revenue_at_risk', ascending=False)

report_cols = [
    'loyalty_number', 'loyalty_card', 'segment_name', 'risk_level',
    'churn_probability', 'clv', 'revenue_at_risk', 'potential_save',
    'recommended_action', 'specific_offer', 'channel', 'timing',
    'incentive_cost_cad', 'avg_flights_per_month', 'recency_months',
    'engagement_health_score', 'tenure_months'
]
at_risk[report_cols].to_csv(REPORTS / "retention_actions.csv", index=False)
print(f"At-risk customers: {len(at_risk):,}")
print(f"  High risk: {(at_risk['risk_level']=='High').sum():,}")
print(f"  Medium risk: {(at_risk['risk_level']=='Medium').sum():,}")
print(f"Total revenue at risk: ${at_risk['revenue_at_risk'].sum():,.0f}")
print(f"Total potential save: ${at_risk['potential_save'].sum():,.0f}")
print(f"Saved: {REPORTS/'retention_actions.csv'}")

# ─── STEP 8: VISUALIZATIONS ───────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 8: GENERATING VISUALIZATIONS")
print("="*65)

# --- ROC Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Performance", fontsize=14, fontweight='bold')

ax = axes[0]
colors = {'Logistic Regression': '#3498db', 'Random Forest': '#2ecc71', 'XGBoost': '#e74c3c'}
for name, prob in [('Logistic Regression', y_prob_lr),
                    ('Random Forest', y_prob_rf),
                    ('XGBoost', y_prob_xgb)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[name], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Feature importance
ax2 = axes[1]
top_features = importance_df.head(12)
ax2.barh(top_features['feature'][::-1], top_features['importance'][::-1], color='#e74c3c', alpha=0.8)
ax2.set_xlabel('Importance', fontsize=11)
ax2.set_title('Top 12 Feature Importances (XGBoost)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(FIGS / "models" / "roc_and_importance.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: roc_and_importance.png")

# --- Churn Distribution by Segment ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Customer Segmentation Analysis", fontsize=14, fontweight='bold')

seg_churn = features_df.groupby('segment_name').agg(
    churn_rate=('churned', 'mean'),
    count=('churned', 'count'),
    avg_clv=('clv', 'mean')
).reset_index().sort_values('churn_rate', ascending=False)

ax = axes[0]
bars = ax.barh(seg_churn['segment_name'], seg_churn['churn_rate'] * 100,
               color=['#e74c3c' if r > 0.5 else '#f39c12' if r > 0.3 else '#2ecc71'
                      for r in seg_churn['churn_rate']], alpha=0.85)
ax.set_xlabel('Churn Rate (%)', fontsize=11)
ax.set_title('Churn Rate by Segment', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, seg_churn['churn_rate']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.0%}', va='center', fontsize=9)

ax2 = axes[1]
sizes = seg_churn['count']
labels = [f"{row['segment_name']}\n({row['count']:,})" for _, row in seg_churn.iterrows()]
wedge_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
ax2.pie(sizes, labels=labels, colors=wedge_colors[:len(sizes)], autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 8})
ax2.set_title('Segment Distribution', fontsize=12)

plt.tight_layout()
plt.savefig(FIGS / "segments" / "segment_analysis.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: segment_analysis.png")

# --- Revenue at Risk ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Revenue at Risk Analysis", fontsize=14, fontweight='bold')

rev_by_seg = features_df.groupby('segment_name').agg(
    revenue_at_risk=('revenue_at_risk', 'sum'),
    customers=('loyalty_number', 'count')
).reset_index().sort_values('revenue_at_risk', ascending=False)

ax = axes[0]
ax.bar(rev_by_seg['segment_name'], rev_by_seg['revenue_at_risk'] / 1e6,
       color='#e74c3c', alpha=0.8)
ax.set_xticklabels(rev_by_seg['segment_name'], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Revenue at Risk (CAD Millions)', fontsize=11)
ax.set_title('Revenue at Risk by Segment', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
risk_dist = features_df['risk_level'].value_counts()
risk_colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
ax2.pie(risk_dist.values, labels=[f"{k}\n({v:,})" for k, v in risk_dist.items()],
        colors=[risk_colors.get(k, '#95a5a6') for k in risk_dist.index],
        autopct='%1.1f%%', startangle=90)
ax2.set_title('Customer Risk Distribution', fontsize=12)

plt.tight_layout()
plt.savefig(FIGS / "revenue_at_risk.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: revenue_at_risk.png")

# --- Behavioral Heatmap ---
fig, ax = plt.subplots(figsize=(12, 7))
heatmap_features = ['avg_flights_per_month', 'recency_months', 'active_month_ratio',
                     'redemption_rate', 'clv_log', 'engagement_health_score',
                     'tenure_months', 'tier_numeric']
heatmap_data = features_df.groupby('segment_name')[heatmap_features].mean()
heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-6)
sns.heatmap(heatmap_normalized, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Normalized Score'})
ax.set_title('Segment Behavioral Profiles (Normalized)', fontsize=14, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(FIGS / "segments" / "behavioral_heatmap.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: behavioral_heatmap.png")

# --- Exploration Visualizations ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Data Overview", fontsize=14, fontweight='bold')

ax = axes[0]
clv_data = features_df['clv'].clip(upper=features_df['clv'].quantile(0.99))
ax.hist(clv_data, bins=50, color='#3498db', alpha=0.8, edgecolor='white')
ax.axvline(features_df['clv'].mean(), color='red', linestyle='--', label=f"Mean: ${features_df['clv'].mean():,.0f}")
ax.axvline(features_df['clv'].median(), color='orange', linestyle='--', label=f"Median: ${features_df['clv'].median():,.0f}")
ax.set_xlabel('Customer Lifetime Value (CAD)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('CLV Distribution', fontsize=12)
ax.legend()

ax2 = axes[1]
card_churn = features_df.groupby('loyalty_card')['churned'].agg(['mean', 'count'])
ax2.bar(card_churn.index, card_churn['mean'] * 100,
        color=['#f39c12', '#3498db', '#2ecc71'], alpha=0.85, edgecolor='white')
ax2.set_xlabel('Loyalty Card Tier', fontsize=11)
ax2.set_ylabel('Churn Rate (%)', fontsize=11)
ax2.set_title('Churn Rate by Loyalty Tier', fontsize=12)
for i, (idx, row) in enumerate(card_churn.iterrows()):
    ax2.text(i, row['mean'] * 100 + 0.5, f"{row['mean']:.1%}\n(n={row['count']:,})",
             ha='center', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGS / "exploration" / "clv_and_tier_analysis.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: clv_and_tier_analysis.png")

# --- Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['AUC', 'Recall', 'Precision', 'F1']
x = np.arange(len(metrics))
width = 0.25
model_names = list(results.keys())
colors_bar = ['#3498db', '#2ecc71', '#e74c3c']
for i, (model_name, color) in enumerate(zip(model_names, colors_bar)):
    vals = [results[model_name][m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=model_name, color=color, alpha=0.85)
ax.set_xlabel('Metric', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend()
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis='y')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=7, padding=2)
plt.tight_layout()
plt.savefig(FIGS / "models" / "model_comparison.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: model_comparison.png")

# ─── STEP 9: FINAL SUMMARY STATS ──────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 9: FINAL SUMMARY STATISTICS")
print("="*65)

total_customers = len(features_df)
at_risk_high = (features_df['risk_level'] == 'High').sum()
at_risk_medium = (features_df['risk_level'] == 'Medium').sum()
total_revenue_at_risk = features_df['revenue_at_risk'].sum()
total_potential_save = features_df['potential_save'].sum()
best_auc = max(v['AUC'] for v in results.values())
best_recall = max(v['Recall'] for v in results.values())
xgb_auc = results['XGBoost']['AUC']
xgb_recall = results['XGBoost']['Recall']
xgb_precision = results['XGBoost']['Precision']

summary = {
    'total_customers': int(total_customers),
    'churn_rate_combined': float(loyalty_raw['churned'].mean()),
    'churn_rate_hard': float(loyalty_raw['hard_churn'].mean()),
    'churn_rate_activity': float(loyalty_raw['activity_churn'].mean()),
    'at_risk_high': int(at_risk_high),
    'at_risk_medium': int(at_risk_medium),
    'total_revenue_at_risk_cad': float(total_revenue_at_risk),
    'total_potential_save_cad': float(total_potential_save),
    'num_segments': int(best_k),
    'xgboost_auc': float(xgb_auc),
    'xgboost_recall': float(xgb_recall),
    'xgboost_precision': float(xgb_precision),
    'best_model_auc': float(best_auc),
    'analysis_date': datetime.now().isoformat(),
    'segment_profiles': {
        name: {
            'count': int(grp['loyalty_number'].count()),
            'churn_rate': float(grp['churned'].mean()),
            'avg_clv': float(grp['clv'].mean()),
            'avg_revenue_at_risk': float(grp['revenue_at_risk'].mean()),
        }
        for name, grp in features_df.groupby('segment_name')
    },
    'model_results': {k: {m: float(v) for m, v in vals.items()} for k, vals in results.items()},
    'top_features': importance_df.head(10)[['feature', 'importance']].to_dict(orient='records'),
}

with open(REPORTS / "pipeline_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

# Update progress
progress = {
    'completed_stages': [0, 1, 2, 3, 4, 5, 6, 7],
    'current_stage': 7,
    'last_updated': datetime.now().isoformat(),
    'summary': summary
}
with open(Path("checkpoints/progress.json"), 'w') as f:
    json.dump(progress, f, indent=2)

print("\n" + "="*65)
print("PIPELINE COMPLETE — KEY RESULTS")
print("="*65)
print(f"Total customers analyzed:     {total_customers:,}")
print(f"Combined churn rate:          {loyalty_raw['churned'].mean():.1%}")
print(f"  ├─ Hard churn:              {loyalty_raw['hard_churn'].mean():.1%}")
print(f"  └─ Activity churn:          {loyalty_raw['activity_churn'].mean():.1%}")
print(f"High-risk customers:          {at_risk_high:,}")
print(f"Medium-risk customers:        {at_risk_medium:,}")
print(f"Total revenue at risk (CAD):  ${total_revenue_at_risk:,.0f}")
print(f"Potential savings (CAD):      ${total_potential_save:,.0f}")
print(f"Number of segments:           {best_k}")
print(f"\nModel Performance (XGBoost):")
print(f"  AUC:       {xgb_auc:.4f}")
print(f"  Recall:    {xgb_recall:.4f}")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  F1:        {results['XGBoost']['F1']:.4f}")
print(f"\nAll outputs saved to outputs/")
print("To launch dashboard: venv/bin/streamlit run 8_dashboard.py")
