"""
Airline Loyalty Behavioral Intelligence — Complete Pipeline
============================================================
All 9 stages in one script. Run from the project root:

    venv/bin/python complete_pipeline.py

Outputs saved to:
    data/final/          customer_features.csv, customer_segments.csv
    outputs/models/      best_model.pkl, xgboost.pkl, random_forest.pkl, logistic_regression.pkl
    outputs/figures/     all PNG charts (exploration/, models/, segments/, churn/)
    outputs/reports/     retention_actions.csv, model_comparison.csv,
                         feature_importance.csv, pipeline_summary.json
    checkpoints/         progress.json

Data note:
    Activity data covers 2017–2018 only (not 2012–2018).
    Prediction date = 2017-12-31.
    Features = 2017 data only.  Labels = 2018 behaviour.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json, warnings, joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                              f1_score, roc_curve, confusion_matrix)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW     = Path("data/raw")
PROC    = Path("data/processed")
FINAL   = Path("data/final")
FIGS    = Path("outputs/figures")
MODELS  = Path("outputs/models")
REPORTS = Path("outputs/reports")

for p in [PROC, FINAL,
          FIGS/"exploration", FIGS/"models", FIGS/"segments", FIGS/"churn",
          MODELS, REPORTS, Path("checkpoints"), Path("logs")]:
    p.mkdir(parents=True, exist_ok=True)

PREDICTION_DATE = pd.Timestamp("2017-12-31")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 1: LOADING & CLEANING DATA")
print("="*65)

loyalty_raw  = pd.read_csv(RAW / "Customer Loyalty History.csv")
activity_raw = pd.read_csv(RAW / "Customer Flight Activity.csv")
calendar_raw = pd.read_csv(RAW / "Calendar.csv")

loyalty_raw.columns  = [c.lower().replace(' ', '_') for c in loyalty_raw.columns]
activity_raw.columns = [c.lower().replace(' ', '_') for c in activity_raw.columns]
calendar_raw.columns = [c.lower().replace(' ', '_') for c in calendar_raw.columns]

print(f"Loyalty History : {loyalty_raw.shape}")
print(f"Flight Activity : {activity_raw.shape}")
print(f"Calendar        : {calendar_raw.shape}")
print(f"Activity years  : {sorted(activity_raw['year'].unique())}")
print(f"Loyalty card dist:\n{loyalty_raw['loyalty_card'].value_counts()}")
print(f"Provinces        : {loyalty_raw['province'].nunique()} unique")

# ── Salary: median-fill + missing flag ───────────────────────────────────────
salary_median = loyalty_raw['salary'].median()
loyalty_raw['salary_missing'] = loyalty_raw['salary'].isna().astype(int)
loyalty_raw['salary'] = loyalty_raw['salary'].fillna(salary_median)

# Fix one known negative salary
loyalty_raw.loc[loyalty_raw['salary'] < 0, 'salary'] = salary_median

# ── Activity: remove true duplicates ─────────────────────────────────────────
activity = activity_raw.drop_duplicates(
    subset=['loyalty_number', 'year', 'month'], keep='first'
)
print(f"Activity after dedup: {activity.shape}  "
      f"(removed {len(activity_raw)-len(activity)} true duplicates)")

# ── Calendar: quarter mapping ─────────────────────────────────────────────────
calendar_raw['date'] = pd.to_datetime(calendar_raw['date'])
calendar_raw['month_num'] = calendar_raw['date'].dt.month
calendar_raw['year_num']  = calendar_raw['date'].dt.year
calendar_raw['quarter']   = calendar_raw['date'].dt.quarter
quarter_map = calendar_raw.drop_duplicates(subset=['year_num','month_num'])[
    ['year_num','month_num','quarter']
].rename(columns={'year_num':'year','month_num':'month'})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CHURN LABELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 2: DEFINING CHURN LABELS")
print("="*65)

# Hard churn: formal cancellation in 2018
loyalty_raw['hard_churn'] = (
    loyalty_raw['cancellation_year'].notna() &
    (loyalty_raw['cancellation_year'] >= 2018)
).astype(int)

# Activity churn: zero flights across all 2018 months
flights_2018 = (
    activity[activity['year'] == 2018]
    .groupby('loyalty_number')['total_flights'].sum()
    .reset_index(name='flights_2018')
)
loyalty_raw = loyalty_raw.merge(flights_2018, on='loyalty_number', how='left')
loyalty_raw['flights_2018'] = loyalty_raw['flights_2018'].fillna(0)
loyalty_raw['activity_churn'] = (loyalty_raw['flights_2018'] == 0).astype(int)

# Combined churn (adopted definition)
loyalty_raw['churned'] = (
    (loyalty_raw['hard_churn'] == 1) | (loyalty_raw['activity_churn'] == 1)
).astype(int)

print(f"Hard churn rate    : {loyalty_raw['hard_churn'].mean():.1%}  "
      f"({loyalty_raw['hard_churn'].sum()} customers)")
print(f"Activity churn rate: {loyalty_raw['activity_churn'].mean():.1%}  "
      f"({loyalty_raw['activity_churn'].sum()} customers)")
print(f"Combined churn rate: {loyalty_raw['churned'].mean():.1%}  "
      f"({loyalty_raw['churned'].sum()} customers)")

loyalty_raw[['loyalty_number','hard_churn','activity_churn','churned']].to_csv(
    PROC / "churn_labels.csv", index=False
)
print(f"Saved churn labels: {PROC/'churn_labels.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING  (strictly 2017 data)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 3: FEATURE ENGINEERING (using 2017 data only)")
print("="*65)

act17 = activity[activity['year'] == 2017].copy()
act17 = act17.merge(quarter_map, on=['year','month'], how='left')
print(f"Activity 2017 rows             : {len(act17):,}")
print(f"Customers with 2017 activity   : {act17['loyalty_number'].nunique():,}")

# ── Province encoding ─────────────────────────────────────────────────────────
province_map = {
    'Ontario': 1, 'British Columbia': 2, 'Quebec': 3,
    'Alberta': 4, 'Manitoba': 5, 'New Brunswick': 6,
    'Nova Scotia': 7, 'Saskatchewan': 8, 'Newfoundland': 9,
    'Yukon': 10, 'Prince Edward Island': 11
}
loyalty_raw['province_code'] = loyalty_raw['province'].map(province_map).fillna(0).astype(int)
# Binary flags for largest provinces (useful for XGBoost)
loyalty_raw['is_ontario']  = (loyalty_raw['province'] == 'Ontario').astype(int)
loyalty_raw['is_bc']       = (loyalty_raw['province'] == 'British Columbia').astype(int)
loyalty_raw['is_quebec']   = (loyalty_raw['province'] == 'Quebec').astype(int)
loyalty_raw['is_alberta']  = (loyalty_raw['province'] == 'Alberta').astype(int)

# ── Per-customer aggregations from 2017 activity ──────────────────────────────
agg = act17.groupby('loyalty_number').agg(
    total_flights_2017          = ('total_flights', 'sum'),
    avg_flights_per_month       = ('total_flights', 'mean'),
    max_flights_month           = ('total_flights', 'max'),
    flight_std                  = ('total_flights', 'std'),
    total_distance              = ('distance', 'sum'),
    avg_distance_per_month      = ('distance', 'mean'),
    total_points_accumulated    = ('points_accumulated', 'sum'),
    total_points_redeemed       = ('points_redeemed', 'sum'),
    avg_points_per_month        = ('points_accumulated', 'mean'),
    total_dollar_cost_redeemed  = ('dollar_cost_points_redeemed', 'sum'),
    active_months               = ('total_flights', lambda x: (x > 0).sum()),
    months_recorded             = ('total_flights', 'count'),
    last_active_month           = ('month', lambda x: x[x > 0].max() if (x > 0).any() else 0),
).reset_index()

agg['flight_std'] = agg['flight_std'].fillna(0)

# Recency (months since last flight before end of 2017)
agg['recency_months'] = (12 - agg['last_active_month']).clip(lower=0)

# Consistency & ratio
agg['active_month_ratio'] = agg['active_months'] / 12
agg['flight_consistency'] = 1 - (
    agg['flight_std'] / (agg['avg_flights_per_month'] + 1e-6)
).clip(0, 1)

# Redemption rate
agg['redemption_rate'] = (
    agg['total_points_redeemed'] / (agg['total_points_accumulated'] + 1e-6)
).clip(0, 1)

# Dollar cost per flight (economic value per trip)
agg['dollar_cost_per_flight'] = (
    agg['total_dollar_cost_redeemed'] / (agg['total_flights_2017'] + 1e-6)
).clip(0)

# Momentum: H2 2017 (Jul–Dec) vs H1 2017 (Jan–Jun)
h1 = act17[act17['month'] <= 6].groupby('loyalty_number')['total_flights'].sum()
h2 = act17[act17['month'] >= 7].groupby('loyalty_number')['total_flights'].sum()
momentum = (h2 - h1).reset_index().rename(columns={'total_flights': 'momentum_h2_minus_h1'})
agg = agg.merge(momentum, on='loyalty_number', how='left')
agg['momentum_h2_minus_h1'] = agg['momentum_h2_minus_h1'].fillna(0)

# Q4 flights (Oct–Dec) — strongest single predictor
q4 = act17[act17['month'] >= 10].groupby('loyalty_number')['total_flights'].sum()
q4 = q4.reset_index().rename(columns={'total_flights': 'q4_flights'})
agg = agg.merge(q4, on='loyalty_number', how='left')
agg['q4_flights'] = agg['q4_flights'].fillna(0)

# Q1 flights (Jan–Mar) — seasonal comparison
q1 = act17[act17['month'] <= 3].groupby('loyalty_number')['total_flights'].sum()
q1 = q1.reset_index().rename(columns={'total_flights': 'q1_flights'})
agg = agg.merge(q1, on='loyalty_number', how='left')
agg['q1_flights'] = agg['q1_flights'].fillna(0)

# Points balance (unredeemed equity)
agg['points_balance'] = (agg['total_points_accumulated'] - agg['total_points_redeemed']).clip(0)

# ── Merge demographics ────────────────────────────────────────────────────────
demo_cols = [
    'loyalty_number', 'gender', 'education', 'salary', 'salary_missing',
    'marital_status', 'loyalty_card', 'clv', 'enrollment_type',
    'enrollment_year', 'enrollment_month', 'province', 'province_code',
    'is_ontario', 'is_bc', 'is_quebec', 'is_alberta',
    'churned', 'hard_churn', 'activity_churn'
]
features_df = loyalty_raw[demo_cols].merge(agg, on='loyalty_number', how='left')

# Fill customers with no 2017 activity (stayed enrolled but didn't fly)
activity_cols = [c for c in agg.columns if c != 'loyalty_number']
features_df[activity_cols] = features_df[activity_cols].fillna(0)

# ── Derived demographic features ──────────────────────────────────────────────
features_df['enrollment_date_num'] = (
    features_df['enrollment_year'] * 12 + features_df['enrollment_month']
)
features_df['tenure_months'] = (
    2017 * 12 + 12 - features_df['enrollment_date_num']
).clip(lower=0)

tier_map = {'Star': 1, 'Nova': 2, 'Aurora': 3}
features_df['tier_numeric'] = features_df['loyalty_card'].map(tier_map).fillna(1)

edu_map = {
    'High School or Below': 1, 'College': 2,
    'Bachelor': 3, 'Master': 4, 'Doctor': 5
}
features_df['education_numeric'] = features_df['education'].map(edu_map).fillna(2)

features_df['is_female']          = (features_df['gender'] == 'Female').astype(int)
features_df['is_married']         = (features_df['marital_status'] == 'Married').astype(int)
features_df['is_promo_enrollment'] = (features_df['enrollment_type'] == 'Promotion').astype(int)
features_df['has_high_clv']        = (features_df['clv'] > features_df['clv'].quantile(0.75)).astype(int)
features_df['clv_log']             = np.log1p(features_df['clv'])

# Enrollment cohort flag
features_df['is_early_member']  = (features_df['enrollment_year'] <= 2014).astype(int)
features_df['is_recent_member'] = (features_df['enrollment_year'] >= 2017).astype(int)

# ── Engagement Health Score (0–100) ──────────────────────────────────────────
scaler_eng = MinMaxScaler()
eng_raw = features_df[[
    'recency_months', 'active_month_ratio', 'flight_consistency',
    'redemption_rate', 'tier_numeric'
]].copy()
eng_raw['recency_months'] = 1 - scaler_eng.fit_transform(
    eng_raw[['recency_months']]
)  # invert: lower recency = better
eng_norm = scaler_eng.fit_transform(eng_raw)
features_df['engagement_health_score'] = (eng_norm.mean(axis=1) * 100).round(1)

print(f"Feature matrix shape: {features_df.shape}")

features_df.to_csv(FINAL / "customer_features.csv", index=False)
print(f"Saved: {FINAL/'customer_features.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CUSTOMER SEGMENTATION  (K-Means)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 4: CUSTOMER SEGMENTATION")
print("="*65)

from sklearn.metrics import silhouette_score

seg_features = [
    'avg_flights_per_month', 'recency_months', 'active_month_ratio',
    'redemption_rate', 'clv_log', 'tier_numeric', 'tenure_months',
    'engagement_health_score', 'momentum_h2_minus_h1', 'total_distance',
    'dollar_cost_per_flight', 'q4_flights'
]
seg_data = features_df[seg_features].fillna(0)
scaler_seg = MinMaxScaler()
seg_scaled = scaler_seg.fit_transform(seg_data)

# Choose optimal K (4–8) via silhouette score — minimum 4 for business usefulness
best_k, best_score = 4, -1
for k in range(4, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(seg_scaled)
    score  = silhouette_score(seg_scaled, labels, sample_size=3000, random_state=42)
    if score > best_score:
        best_score, best_k = score, k

print(f"Best K by silhouette: {best_k}  (score={best_score:.4f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
features_df['segment_id'] = km_final.fit_predict(seg_scaled)

# ── Name segments by behaviour ────────────────────────────────────────────────
seg_profiles = features_df.groupby('segment_id').agg(
    avg_flights  = ('avg_flights_per_month', 'mean'),
    recency      = ('recency_months', 'mean'),
    active_ratio = ('active_month_ratio', 'mean'),
    churn_rate   = ('churned', 'mean'),
    avg_clv      = ('clv', 'mean'),
    avg_dist     = ('total_distance', 'mean'),
    count        = ('loyalty_number', 'count'),
).round(3)
print("\nCluster profiles:")
print(seg_profiles.to_string())

# Rank segments by avg_clv within each behaviour class to assign unique names
# Strategy: split on activity threshold (0.5 flights/month)
#   High-activity (≥0.5): rank by CLV → Champion (high), Hoarder (low)
#   Low-activity  (<0.5): rank by CLV → Dormant Premium (high), Seasonal (low)
high_act = seg_profiles[seg_profiles['avg_flights'] >= 0.5].sort_values('avg_clv', ascending=False)
low_act  = seg_profiles[seg_profiles['avg_flights'] < 0.5].sort_values('avg_clv', ascending=False)

high_names = ['Active Champions', 'Miles Hoarders', 'Engaged Regulars', 'Budget Fliers']
low_names  = ['Premium Dormant', 'Seasonal Travelers', 'Inactive Low-Value']

seg_names = {}
for i, sid in enumerate(high_act.index):
    seg_names[sid] = high_names[min(i, len(high_names)-1)]
for i, sid in enumerate(low_act.index):
    seg_names[sid] = low_names[min(i, len(low_names)-1)]

features_df['segment_name'] = features_df['segment_id'].map(seg_names)

print("\nSegment assignment:")
print(features_df['segment_name'].value_counts())
print("\nChurn by segment:")
print(features_df.groupby('segment_name')['churned'].agg(['mean','count']).round(3))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 5: MODEL TRAINING & EVALUATION")
print("="*65)

model_features = [
    'recency_months', 'active_month_ratio', 'avg_flights_per_month',
    'redemption_rate', 'clv_log', 'tenure_months', 'active_months',
    'q4_flights', 'total_distance', 'momentum_h2_minus_h1',
    'flight_consistency', 'total_points_accumulated', 'salary',
    'salary_missing', 'engagement_health_score', 'points_balance',
    'tier_numeric', 'education_numeric', 'is_female', 'is_married',
    'is_promo_enrollment', 'dollar_cost_per_flight',
    'is_early_member', 'is_recent_member',
    'is_ontario', 'is_bc', 'is_quebec'
]

X = features_df[model_features].values
y = features_df['churned'].values

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.1%}   Test churn rate: {y_test.mean():.1%}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_sm.shape}, class dist: "
      f"{{0: {(y_train_sm==0).sum()}, 1: {(y_train_sm==1).sum()}}}")

scaler_model = StandardScaler()
X_train_sc = scaler_model.fit_transform(X_train_sm)
X_test_sc  = scaler_model.transform(X_test)

results = {}
cv_scores = {}

# ── Logistic Regression ───────────────────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42, n_jobs=-1)
lr.fit(X_train_sc, y_train_sm)
y_pred_lr = lr.predict(X_test_sc)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]
results['Logistic Regression'] = {
    'AUC': roc_auc_score(y_test, y_prob_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'F1': f1_score(y_test, y_pred_lr),
}
joblib.dump(lr, MODELS / "logistic_regression.pkl")

# Cross-validation (on SMOTE data, 5-fold)
cv_lr = cross_val_score(lr, X_train_sc, y_train_sm, cv=5, scoring='roc_auc', n_jobs=-1)
cv_scores['Logistic Regression'] = cv_lr.mean()
print(f"  LR  AUC={results['Logistic Regression']['AUC']:.4f}  "
      f"CV-AUC={cv_lr.mean():.4f}±{cv_lr.std():.4f}")

# ── Random Forest ─────────────────────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
rf.fit(X_train_sm, y_train_sm)
y_pred_rf = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = {
    'AUC': roc_auc_score(y_test, y_prob_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'F1': f1_score(y_test, y_pred_rf),
}
joblib.dump(rf, MODELS / "random_forest.pkl")
print(f"  RF  AUC={results['Random Forest']['AUC']:.4f}")

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("Training XGBoost...")
spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=spw, random_state=42,
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

cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
cv_scores['XGBoost'] = cv_xgb.mean()
print(f"  XGB AUC={results['XGBoost']['AUC']:.4f}  "
      f"CV-AUC={cv_xgb.mean():.4f}±{cv_xgb.std():.4f}")

model_comparison_df = pd.DataFrame(results).T.round(4)
model_comparison_df.index.name = 'Model'
model_comparison_df.to_csv(REPORTS / "model_comparison.csv")
print("\nModel Comparison:")
print(model_comparison_df.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 6: FEATURE IMPORTANCE (XGBoost)")
print("="*65)

importance_df = pd.DataFrame({
    'feature': model_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)
print(importance_df.head(15).to_string())
importance_df.to_csv(REPORTS / "feature_importance.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — RETENTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 7: RETENTION ENGINE")
print("="*65)

# Add churn probability and risk level to full dataset
features_df['churn_probability'] = xgb_model.predict_proba(
    imputer.transform(features_df[model_features])
)[:, 1]

features_df['risk_level'] = pd.cut(
    features_df['churn_probability'],
    bins=[0, 0.40, 0.70, 1.0],
    labels=['Low', 'Medium', 'High']
).astype(str)

features_df['revenue_at_risk'] = (
    features_df['clv'] * features_df['churn_probability'] * 0.82
).round(2)
features_df['potential_save'] = (
    features_df['revenue_at_risk'] * 0.27
).round(2)

# ── Offer library: segment × risk level ──────────────────────────────────────
offer_library = {
    'Seasonal Travelers': {
        'High':   {'action': 'Off-Season Reactivation',
                   'offer': '4,000 bonus miles + exclusive off-peak fare (30% off)',
                   'channel': 'Email + SMS simultaneously',
                   'timing': 'Within 72 hours',
                   'incentive_cost': 40},
        'Medium': {'action': 'Seasonal Re-Engagement',
                   'offer': 'Upcoming season preview + 2,000 bonus miles',
                   'channel': 'Email',
                   'timing': 'Within 14 days',
                   'incentive_cost': 20},
        'Low':    {'action': 'Seasonal Newsletter',
                   'offer': 'Seasonal deals + destination inspiration',
                   'channel': 'Email',
                   'timing': 'Monthly',
                   'incentive_cost': 5},
    },
    'Active Champions': {
        'High':   {'action': 'Loyalty Lock-In',
                   'offer': '6,000 bonus miles + priority check-in for 6 months',
                   'channel': 'Email + App push',
                   'timing': 'Within 3 days',
                   'incentive_cost': 70},
        'Medium': {'action': 'Engagement Boost',
                   'offer': 'Double miles on next 3 flights',
                   'channel': 'Email',
                   'timing': 'Within 14 days',
                   'incentive_cost': 30},
        'Low':    {'action': 'Loyalty Appreciation',
                   'offer': '1,500 bonus miles as thank-you',
                   'channel': 'Email',
                   'timing': 'Quarterly',
                   'incentive_cost': 15},
    },
    'Premium Loyalists': {
        'High':   {'action': 'VIP Retention Call',
                   'offer': 'Complimentary companion ticket + status guarantee 12 months',
                   'channel': 'Personal phone call + premium email',
                   'timing': 'Within 24 hours',
                   'incentive_cost': 200},
        'Medium': {'action': 'Loyalty Appreciation',
                   'offer': '10,000 bonus miles + lounge access upgrade (3 visits)',
                   'channel': 'Email',
                   'timing': 'Within 3 days',
                   'incentive_cost': 100},
        'Low':    {'action': 'Tier Thank-You',
                   'offer': '3,000 bonus miles as loyalty recognition',
                   'channel': 'Email',
                   'timing': 'Monthly',
                   'incentive_cost': 30},
    },
    'Miles Hoarders': {
        'High':   {'action': 'Redemption Activation',
                   'offer': '2× redemption value + 60-day flash sale on points',
                   'channel': 'Email + App push',
                   'timing': 'Within 72 hours',
                   'incentive_cost': 50},
        'Medium': {'action': 'Redemption Reminder',
                   'offer': '50% bonus on any redemption within 60 days',
                   'channel': 'Email',
                   'timing': 'Within 10 days',
                   'incentive_cost': 30},
        'Low':    {'action': 'Points Expiry Warning',
                   'offer': 'Balance statement + expiry alert + 500 bonus miles',
                   'channel': 'Email',
                   'timing': 'Monthly',
                   'incentive_cost': 5},
    },
    'Engaged Regulars': {
        'High':   {'action': 'Urgent Retention Offer',
                   'offer': '5,000 bonus miles + waived change fees for 60 days',
                   'channel': 'SMS + Email',
                   'timing': 'Within 48 hours',
                   'incentive_cost': 55},
        'Medium': {'action': 'Engagement Campaign',
                   'offer': '3,000 bonus miles on next flight',
                   'channel': 'Email',
                   'timing': 'Within 7 days',
                   'incentive_cost': 30},
        'Low':    {'action': 'Loyalty Touch',
                   'offer': '1,000 bonus miles',
                   'channel': 'Email',
                   'timing': 'Quarterly',
                   'incentive_cost': 10},
    },
}

def get_retention_action(row):
    seg  = row['segment_name']
    risk = row['risk_level']
    lib  = offer_library.get(seg, offer_library['Engaged Regulars'])
    offer = lib.get(risk, lib['Low'])
    return pd.Series(offer)

action_cols = features_df.apply(get_retention_action, axis=1)
features_df = pd.concat([features_df, action_cols], axis=1)

at_risk = features_df[features_df['risk_level'].isin(['High','Medium'])].copy()
at_risk = at_risk.sort_values('revenue_at_risk', ascending=False)

report_cols = [
    'loyalty_number', 'loyalty_card', 'province', 'segment_name',
    'risk_level', 'churn_probability', 'clv', 'revenue_at_risk',
    'potential_save', 'action', 'offer', 'channel', 'timing',
    'incentive_cost', 'avg_flights_per_month', 'recency_months',
    'engagement_health_score', 'tenure_months'
]
# rename offer cols to avoid collision
at_risk_out = at_risk[report_cols].copy()
at_risk_out.columns = [
    'loyalty_number', 'loyalty_card', 'province', 'segment_name',
    'risk_level', 'churn_probability', 'clv', 'revenue_at_risk',
    'potential_save', 'recommended_action', 'specific_offer', 'channel',
    'timing', 'incentive_cost_cad', 'avg_flights_per_month',
    'recency_months', 'engagement_health_score', 'tenure_months'
]
at_risk_out.to_csv(REPORTS / "retention_actions.csv", index=False)

print(f"At-risk customers  : {len(at_risk):,}")
print(f"  High risk        : {(at_risk['risk_level']=='High').sum():,}")
print(f"  Medium risk      : {(at_risk['risk_level']=='Medium').sum():,}")
print(f"Total revenue at risk: ${at_risk['revenue_at_risk'].sum():,.0f}")
print(f"Total potential save : ${at_risk['potential_save'].sum():,.0f}")

# Save full segments file (includes churn_probability + risk_level for dashboard)
features_df.to_csv(FINAL / "customer_segments.csv", index=False)
print(f"Saved: {FINAL/'customer_segments.csv'} ({features_df.shape})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 8: GENERATING VISUALIZATIONS")
print("="*65)

COLORS = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}
SEG_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

# ── 1. ROC Curves + Feature Importance ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Model Performance", fontsize=14, fontweight='bold')

ax = axes[0]
for name, prob, color in [
    ('Logistic Regression', y_prob_lr, '#3498db'),
    ('Random Forest', y_prob_rf, '#2ecc71'),
    ('XGBoost', y_prob_xgb, '#e74c3c')
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
ax.plot([0,1],[0,1],'k--', alpha=0.5, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves — All Models', fontsize=12)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
top12 = importance_df.head(12)
ax2.barh(top12['feature'][::-1], top12['importance'][::-1], color='#e74c3c', alpha=0.8)
ax2.set_xlabel('Importance Score', fontsize=11)
ax2.set_title('Top 12 Feature Importances (XGBoost)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIGS/"models"/"roc_and_importance.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: roc_and_importance.png")

# ── 2. Confusion Matrix (XGBoost) ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("XGBoost Prediction Quality", fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Pred: Stay', 'Pred: Churn'],
            yticklabels=['Actual: Stay', 'Actual: Churn'])
axes[0].set_title('Confusion Matrix', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=11)
axes[0].set_xlabel('Predicted', fontsize=11)

# Model comparison bar
metrics_list = ['AUC', 'Recall', 'Precision', 'F1']
x = np.arange(len(metrics_list))
width = 0.25
for i, (mname, color) in enumerate(zip(
    ['Logistic Regression','Random Forest','XGBoost'],
    ['#3498db','#2ecc71','#e74c3c']
)):
    vals = [results[mname][m] for m in metrics_list]
    axes[1].bar(x + i*width, vals, width, label=mname, color=color, alpha=0.85)
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(metrics_list, fontsize=11)
axes[1].set_ylim(0, 1.15)
axes[1].set_title('Model Metric Comparison', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.3f', fontsize=7, padding=2)
plt.tight_layout()
plt.savefig(FIGS/"models"/"model_comparison.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: model_comparison.png")

# ── 3. Segment Analysis ───────────────────────────────────────────────────────
seg_summary = features_df.groupby('segment_name').agg(
    churn_rate=('churned','mean'),
    count=('churned','count'),
    avg_clv=('clv','mean'),
    revenue_at_risk=('revenue_at_risk','sum')
).reset_index().sort_values('churn_rate', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Customer Segmentation Analysis", fontsize=14, fontweight='bold')

ax = axes[0]
colors_bar = [SEG_COLORS[i % len(SEG_COLORS)] for i in range(len(seg_summary))]
bars = ax.barh(seg_summary['segment_name'], seg_summary['churn_rate']*100,
               color=colors_bar, alpha=0.85)
ax.set_xlabel('Churn Rate (%)', fontsize=11)
ax.set_title('Churn Rate by Segment', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, seg_summary['churn_rate']):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f'{val:.0%}', va='center', fontsize=9)

ax2 = axes[1]
labels_pie = [f"{r['segment_name']}\n({r['count']:,})" for _, r in seg_summary.iterrows()]
ax2.pie(seg_summary['count'], labels=labels_pie,
        colors=colors_bar, autopct='%1.1f%%', startangle=90, textprops={'fontsize':8})
ax2.set_title('Segment Size Distribution', fontsize=12)
plt.tight_layout()
plt.savefig(FIGS/"segments"/"segment_analysis.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: segment_analysis.png")

# ── 4. Revenue at Risk ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Revenue at Risk Analysis", fontsize=14, fontweight='bold')

ax = axes[0]
ax.bar(seg_summary['segment_name'], seg_summary['revenue_at_risk']/1e6,
       color=colors_bar, alpha=0.85)
ax.set_xticklabels(seg_summary['segment_name'], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Revenue at Risk (CAD Millions)', fontsize=11)
ax.set_title('Revenue at Risk by Segment', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
risk_dist = features_df['risk_level'].value_counts()
risk_order = ['High','Medium','Low']
risk_dist = risk_dist.reindex([r for r in risk_order if r in risk_dist.index])
ax2.pie(risk_dist.values,
        labels=[f"{k}\n({v:,})" for k, v in risk_dist.items()],
        colors=[COLORS.get(k,'#95a5a6') for k in risk_dist.index],
        autopct='%1.1f%%', startangle=90)
ax2.set_title('Customer Risk Distribution', fontsize=12)
plt.tight_layout()
plt.savefig(FIGS/"revenue_at_risk.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: revenue_at_risk.png")

# ── 5. Behavioral Heatmap ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
hmap_feats = ['avg_flights_per_month','recency_months','active_month_ratio',
              'redemption_rate','clv_log','engagement_health_score',
              'tenure_months','tier_numeric','dollar_cost_per_flight','q4_flights']
hmap_data = features_df.groupby('segment_name')[hmap_feats].mean()
hmap_norm = (hmap_data - hmap_data.min()) / (hmap_data.max() - hmap_data.min() + 1e-6)
sns.heatmap(hmap_norm, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
            linewidths=0.5, cbar_kws={'label':'Normalized Score (0=lowest, 1=highest)'})
ax.set_title('Segment Behavioral Profiles (Normalized 0–1)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS/"segments"/"behavioral_heatmap.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: behavioral_heatmap.png")

# ── 6. Geographic Analysis ────────────────────────────────────────────────────
prov_analysis = features_df.merge(
    loyalty_raw[['loyalty_number','province']].drop_duplicates(),
    on='loyalty_number', how='left', suffixes=('','_raw')
)
prov_col = 'province' if 'province' in prov_analysis.columns else 'province_raw'
prov_stats = prov_analysis.groupby(prov_col).agg(
    churn_rate=('churned','mean'),
    count=(prov_col,'count'),
    avg_clv=('clv','mean'),
    revenue_at_risk=('revenue_at_risk','sum')
).reset_index().rename(columns={prov_col:'province'}).sort_values('churn_rate', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Geographic Analysis — Churn by Province", fontsize=14, fontweight='bold')

ax = axes[0]
prov_plot = prov_stats[prov_stats['count'] >= 50]
bar_colors = ['#e74c3c' if r > 0.18 else '#f39c12' if r > 0.13 else '#27ae60'
              for r in prov_plot['churn_rate']]
bars = ax.barh(prov_plot['province'], prov_plot['churn_rate']*100,
               color=bar_colors, alpha=0.85)
ax.axvline(features_df['churned'].mean()*100, color='navy', linestyle='--', alpha=0.6,
           label=f"National avg {features_df['churned'].mean():.1%}")
ax.set_xlabel('Churn Rate (%)', fontsize=11)
ax.set_title('Churn Rate by Province', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, prov_plot['churn_rate']):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=9)

ax2 = axes[1]
ax2.bar(prov_plot['province'], prov_plot['revenue_at_risk']/1e6,
        color=bar_colors, alpha=0.85)
ax2.set_xticklabels(prov_plot['province'], rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Revenue at Risk (CAD Millions)', fontsize=11)
ax2.set_title('Revenue at Risk by Province', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIGS/"exploration"/"geographic_analysis.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: geographic_analysis.png")

# ── 7. Cohort Analysis — Churn by Enrollment Year ─────────────────────────────
cohort_stats = features_df.groupby('enrollment_year').agg(
    churn_rate=('churned','mean'),
    count=('loyalty_number','count'),
    avg_clv=('clv','mean'),
    avg_engagement=('engagement_health_score','mean')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Cohort Analysis — Enrollment Year", fontsize=14, fontweight='bold')

ax = axes[0]
bar_c = ['#e74c3c' if r > 0.20 else '#f39c12' if r > 0.13 else '#27ae60'
         for r in cohort_stats['churn_rate']]
ax.bar(cohort_stats['enrollment_year'].astype(str), cohort_stats['churn_rate']*100,
       color=bar_c, alpha=0.85)
ax.axhline(features_df['churned'].mean()*100, color='navy', linestyle='--', alpha=0.6,
           label=f"Overall avg {features_df['churned'].mean():.1%}")
ax.set_xlabel('Enrollment Year', fontsize=11)
ax.set_ylabel('Churn Rate (%)', fontsize=11)
ax.set_title('Churn Rate by Enrollment Cohort', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for i, (yr, rate) in enumerate(zip(cohort_stats['enrollment_year'], cohort_stats['churn_rate'])):
    ax.text(i, rate*100+0.5, f'{rate:.1%}', ha='center', fontsize=9)

ax2 = axes[1]
ax2.plot(cohort_stats['enrollment_year'].astype(str),
         cohort_stats['avg_engagement'], 'o-', color='#3498db', linewidth=2, markersize=8)
ax2.set_xlabel('Enrollment Year', fontsize=11)
ax2.set_ylabel('Avg Engagement Health Score', fontsize=11)
ax2.set_title('Engagement Score by Enrollment Cohort', fontsize=12)
ax2.grid(True, alpha=0.3)
for x, y in zip(cohort_stats['enrollment_year'].astype(str), cohort_stats['avg_engagement']):
    ax2.text(x, y+0.5, f'{y:.1f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGS/"exploration"/"cohort_analysis.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: cohort_analysis.png")

# ── 8. Demographic Analysis ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Demographic Analysis — Churn Patterns", fontsize=14, fontweight='bold')

# Churn by education
edu_churn = features_df.groupby('education')['churned'].agg(['mean','count']).reset_index()
edu_order = ['High School or Below','College','Bachelor','Master','Doctor']
edu_churn = edu_churn.set_index('education').reindex(
    [e for e in edu_order if e in edu_churn['education'].values]
).reset_index()
ax = axes[0,0]
ax.bar(edu_churn['education'], edu_churn['mean']*100,
       color='#3498db', alpha=0.85)
ax.set_xticklabels(edu_churn['education'], rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Churn Rate (%)', fontsize=11)
ax.set_title('Churn by Education Level', fontsize=12)
ax.axhline(features_df['churned'].mean()*100, color='red', linestyle='--', alpha=0.6)
ax.grid(True, alpha=0.3, axis='y')
for i, (_, row) in enumerate(edu_churn.iterrows()):
    ax.text(i, row['mean']*100+0.3, f"{row['mean']:.1%}\n(n={int(row['count'])})",
            ha='center', fontsize=8)

# Churn by gender
gen_churn = features_df.groupby('gender')['churned'].agg(['mean','count']).reset_index()
ax2 = axes[0,1]
ax2.bar(gen_churn['gender'], gen_churn['mean']*100,
        color=['#e91e8c','#2196F3'], alpha=0.85)
ax2.set_ylabel('Churn Rate (%)', fontsize=11)
ax2.set_title('Churn by Gender', fontsize=12)
ax2.axhline(features_df['churned'].mean()*100, color='red', linestyle='--', alpha=0.6)
ax2.grid(True, alpha=0.3, axis='y')
for i, (_, row) in enumerate(gen_churn.iterrows()):
    ax2.text(i, row['mean']*100+0.3, f"{row['mean']:.1%}\n(n={int(row['count'])})",
             ha='center', fontsize=9)

# Churn by marital status
mar_churn = features_df.groupby('marital_status')['churned'].agg(['mean','count']).reset_index()
ax3 = axes[1,0]
ax3.bar(mar_churn['marital_status'], mar_churn['mean']*100,
        color=['#9b59b6','#1abc9c'], alpha=0.85)
ax3.set_ylabel('Churn Rate (%)', fontsize=11)
ax3.set_title('Churn by Marital Status', fontsize=12)
ax3.axhline(features_df['churned'].mean()*100, color='red', linestyle='--', alpha=0.6)
ax3.grid(True, alpha=0.3, axis='y')
for i, (_, row) in enumerate(mar_churn.iterrows()):
    ax3.text(i, row['mean']*100+0.3, f"{row['mean']:.1%}\n(n={int(row['count'])})",
             ha='center', fontsize=9)

# CLV distribution by loyalty tier
ax4 = axes[1,1]
for tier, color in [('Star','#f39c12'),('Nova','#3498db'),('Aurora','#2ecc71')]:
    data = features_df[features_df['loyalty_card']==tier]['clv']
    ax4.hist(data.clip(upper=data.quantile(0.99)), bins=40, alpha=0.6,
             color=color, label=f"{tier} (n={len(data):,})")
ax4.set_xlabel('CLV (CAD)', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('CLV Distribution by Loyalty Tier', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGS/"exploration"/"demographic_analysis.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: demographic_analysis.png")

# ── 9. Churn Definition Comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Churn Definition Analysis", fontsize=14, fontweight='bold')

ax = axes[0]
churn_types = ['Hard Churn\n(Cancellation)', 'Activity Churn\n(No 2018 Flights)', 'Combined Churn\n(Adopted)']
churn_rates = [
    loyalty_raw['hard_churn'].mean(),
    loyalty_raw['activity_churn'].mean(),
    loyalty_raw['churned'].mean()
]
bars = ax.bar(churn_types, [r*100 for r in churn_rates],
              color=['#3498db','#f39c12','#e74c3c'], alpha=0.85)
ax.set_ylabel('Churn Rate (%)', fontsize=11)
ax.set_title('Three Churn Definitions Compared', fontsize=12)
for bar, rate, count in zip(bars, churn_rates, [645, 2123, 2728]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{rate:.1%}\n({count:,} customers)', ha='center', fontsize=9, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
ax2.hist(features_df['churn_probability'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
ax2.axvline(0.40, color='#f39c12', linestyle='--', linewidth=2, label='Medium threshold (0.40)')
ax2.axvline(0.70, color='#e74c3c', linestyle='--', linewidth=2, label='High threshold (0.70)')
ax2.set_xlabel('Predicted Churn Probability', fontsize=11)
ax2.set_ylabel('Number of Customers', fontsize=11)
ax2.set_title('Churn Probability Distribution', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS/"churn"/"churn_definition_comparison.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: churn_definition_comparison.png")

print(f"\nAll figures saved to {FIGS}/")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SUMMARY & CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 9: FINAL SUMMARY STATISTICS")
print("="*65)

at_risk_high   = int((features_df['risk_level'] == 'High').sum())
at_risk_medium = int((features_df['risk_level'] == 'Medium').sum())
total_rar      = float(features_df['revenue_at_risk'].sum())
total_save     = float(features_df['potential_save'].sum())

# Geographic summary
prov_stats_out = prov_stats.to_dict(orient='records')

# Cohort summary
cohort_out = {
    str(int(row['enrollment_year'])): {
        'count': int(row['count']),
        'churn_rate': float(row['churn_rate']),
        'avg_clv': float(row['avg_clv']),
    }
    for _, row in cohort_stats.iterrows()
}

summary = {
    'total_customers'          : int(len(features_df)),
    'churn_rate_combined'      : float(loyalty_raw['churned'].mean()),
    'churn_rate_hard'          : float(loyalty_raw['hard_churn'].mean()),
    'churn_rate_activity'      : float(loyalty_raw['activity_churn'].mean()),
    'at_risk_high'             : at_risk_high,
    'at_risk_medium'           : at_risk_medium,
    'total_revenue_at_risk_cad': total_rar,
    'total_potential_save_cad' : total_save,
    'num_segments'             : int(best_k),
    'xgboost_auc'              : float(results['XGBoost']['AUC']),
    'xgboost_recall'           : float(results['XGBoost']['Recall']),
    'xgboost_precision'        : float(results['XGBoost']['Precision']),
    'best_model_auc'           : float(max(v['AUC'] for v in results.values())),
    'analysis_date'            : datetime.now().isoformat(),
    'segment_profiles'         : {
        name: {
            'count'              : int(grp['loyalty_number'].count()),
            'churn_rate'         : float(grp['churned'].mean()),
            'avg_clv'            : float(grp['clv'].mean()),
            'avg_revenue_at_risk': float(grp['revenue_at_risk'].mean()),
        }
        for name, grp in features_df.groupby('segment_name')
    },
    'model_results' : {
        k: {m: float(v) for m, v in vals.items()}
        for k, vals in results.items()
    },
    'cv_scores'     : {k: float(v) for k, v in cv_scores.items()},
    'top_features'  : importance_df.head(10)[['feature','importance']].to_dict('records'),
    'geographic'    : prov_stats_out,
    'cohorts'       : cohort_out,
}

with open(REPORTS / "pipeline_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

progress = {
    'completed_stages': [0,1,2,3,4,5,6,7],
    'current_stage'   : 7,
    'last_updated'    : datetime.now().isoformat(),
    'summary'         : summary,
}
with open(Path("checkpoints/progress.json"), 'w') as f:
    json.dump(progress, f, indent=2)

print("\n" + "="*65)
print("PIPELINE COMPLETE — KEY RESULTS")
print("="*65)
print(f"Total customers analyzed  : {len(features_df):,}")
print(f"Combined churn rate       : {loyalty_raw['churned'].mean():.1%}")
print(f"  ├─ Hard churn           : {loyalty_raw['hard_churn'].mean():.1%}")
print(f"  └─ Activity churn       : {loyalty_raw['activity_churn'].mean():.1%}")
print(f"High-risk customers       : {at_risk_high:,}")
print(f"Medium-risk customers     : {at_risk_medium:,}")
print(f"Total revenue at risk     : ${total_rar:,.0f} CAD")
print(f"Potential savings         : ${total_save:,.0f} CAD")
print(f"Segments                  : {best_k}")
print(f"\nModel Performance:")
for model, res in results.items():
    star = " ★" if model == 'XGBoost' else ""
    print(f"  {model+star:<26} AUC={res['AUC']:.4f}  "
          f"Recall={res['Recall']:.4f}  F1={res['F1']:.4f}")
print(f"\nTop 5 features:")
for _, row in importance_df.head(5).iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}")
print(f"\nOutputs saved to outputs/")
print("Dashboard: venv/bin/streamlit run dashboard.py")
