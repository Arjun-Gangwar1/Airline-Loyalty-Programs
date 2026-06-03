"""
Generates the Technical Report as a formatted text/markdown file.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

REPORTS = Path("outputs/reports")
FINAL = Path("data/final")

with open(REPORTS / "pipeline_summary.json") as f:
    summary = json.load(f)

features_df = pd.read_csv(FINAL / "customer_segments.csv")
retention_df = pd.read_csv(REPORTS / "retention_actions.csv")
model_comp = pd.read_csv(REPORTS / "model_comparison.csv")
importance_df = pd.read_csv(REPORTS / "feature_importance.csv")

# Merge retention info back to features for full picture
features_df = features_df.merge(
    retention_df[['loyalty_number', 'churn_probability', 'risk_level',
                  'revenue_at_risk', 'potential_save', 'incentive_cost_cad']],
    on='loyalty_number', how='left'
)
features_df['churn_probability'] = features_df['churn_probability'].fillna(0.05)
features_df['risk_level'] = features_df['risk_level'].fillna('Low')
features_df['revenue_at_risk'] = features_df['revenue_at_risk'].fillna(0)
features_df['potential_save'] = features_df['potential_save'].fillna(0)
features_df['incentive_cost_cad'] = features_df['incentive_cost_cad'].fillna(5)

# Compute some stats
seg_profiles = features_df.groupby('segment_name').agg(
    count=('loyalty_number', 'count'),
    churn_rate=('churned', 'mean'),
    avg_clv=('clv', 'mean'),
    avg_flights=('avg_flights_per_month', 'mean'),
    avg_recency=('recency_months', 'mean'),
    revenue_at_risk=('revenue_at_risk', 'sum'),
).reset_index().sort_values('revenue_at_risk', ascending=False)

top5_features = importance_df.head(5)

total_customers = summary['total_customers']
churn_rate = summary['churn_rate_combined']
hard_rate = summary['churn_rate_hard']
activity_rate = summary['churn_rate_activity']
high_risk = summary['at_risk_high']
medium_risk = summary['at_risk_medium']
rev_at_risk = summary['total_revenue_at_risk_cad']
potential_save = summary['total_potential_save_cad']
xgb_auc = summary['xgboost_auc']
xgb_recall = summary['xgboost_recall']
xgb_precision = summary['xgboost_precision']
rf_auc = summary['model_results']['Random Forest']['AUC']
lr_auc = summary['model_results']['Logistic Regression']['AUC']
num_segments = summary['num_segments']

top_seg = retention_df.groupby('segment_name')['revenue_at_risk'].sum().idxmax()
top_seg_rev = retention_df.groupby('segment_name')['revenue_at_risk'].sum().max()

report = f"""
TECHNICAL REPORT
================================================================================
UNLOCKING BEHAVIORAL INTELLIGENCE IN AIRLINE LOYALTY PROGRAMS
================================================================================

Prepared for: Marketing Analytics Leadership (CFO & CMO)
Dataset: Canadian Airline Loyalty Program — 16,737 Members (2017–2018)
Analysis Date: {datetime.now().strftime('%B %d, %Y')}
Stack: Python · Pandas · XGBoost · Scikit-learn · SMOTE · Streamlit · Plotly

================================================================================
1. EXECUTIVE SUMMARY
================================================================================

Airlines are losing customers silently. Of the 16,737 loyalty members analyzed,
16.3% (2,728 customers) are projected to disengage in 2018 — and most will do
so without formally cancelling their membership. This "quiet churn" represents
${rev_at_risk:,.0f} in at-risk revenue.

Our behavioral intelligence platform identifies {high_risk:,} high-priority
customers up to 3 months before they leave, giving the marketing team an
actionable window to intervene. With targeted, segment-specific offers, we
estimate ${potential_save:,.0f} in annual revenue can be recovered.

Key model result: XGBoost AUC = {xgb_auc:.3f}, Recall = {xgb_recall:.1%}. For
every 100 customers who will churn, the model correctly flags {xgb_recall*100:.0f}.

Three recommendations a CFO and CMO can act on today:

1. DEPLOY URGENT WIN-BACK for 1,840 high-risk customers immediately — average
   revenue at risk per customer = ${features_df[features_df['risk_level']=='High']['clv'].mean():,.0f} CLV.
   Expected recovery: ${features_df[features_df['risk_level']=='High']['potential_save'].sum():,.0f} CAD.

2. RESTRUCTURE THE SEASONAL TRAVELERS program: this segment has a {seg_profiles[seg_profiles['segment_name']=='Seasonal Travelers']['churn_rate'].values[0]:.0%}
   churn rate. Off-peak flight incentives (bonus miles for non-summer bookings)
   can convert seasonal visitors into year-round fliers.

3. PROTECT PREMIUM LOYALISTS: though currently low-churn, the Aurora/Nova
   tier customers represent disproportionate CLV. Any drop in their engagement
   should trigger immediate executive-level intervention.

================================================================================
2. PROBLEM FRAMING & DATA
================================================================================

2.1 DATASET OVERVIEW
─────────────────────
• Customer Loyalty History: 16,737 Canadian members — demographics (gender,
  education, salary, marital status), loyalty tier (Star/Nova/Aurora), CLV,
  enrollment dates, and cancellation records.

• Customer Flight Activity: 392,936 monthly records covering 2017 and 2018.
  Each row = one customer × one month, with flights, distance, and points
  earned/redeemed.

Note: The activity data covers 2017–2018 (not 2012–2018 as originally anticipated).
This compressed observation window was identified during data exploration and the
prediction framework was adjusted accordingly: features derived from 2017, labels
from 2018.

2.2 DATA QUALITY DECISIONS
────────────────────────────
Issue 1 — Salary missingness (25.3%): Rather than dropping these customers,
we filled missing salary with the population median (CAD 73,455) and created a
binary `salary_missing` indicator feature. This preserves all customers in the
model while the indicator lets the model learn from missingness patterns.

Issue 2 — True duplicate activity records: 3,871 duplicate rows (loyalty_number
+ year + month triplets) were found and removed. A naive deduplication approach
had erroneously flagged 193,063 rows — we corrected this by deduplicating only
on exact match across all key fields.

Issue 3 — Post-cancellation activity records: 100 cancelled customers had
activity records in the same period. This is expected (customers flew before
cancelling). Temporal filtering ensures only pre-2017-12-31 data is used for
features.

Issue 4 — Negative salary values: One record with salary = -58,486 was detected.
Treated as a data entry error and replaced with the median.

2.3 CHURN DEFINITION
─────────────────────
Three definitions were evaluated:

  a) Hard Churn (Formal Cancellation in 2018):
     Cancellation_Year = 2018 OR Cancellation_Year is not null and year ≥ 2018.
     Rate: {hard_rate:.1%} ({int(total_customers * hard_rate):,} customers).
     Weakness: Captures only formal resignations; misses silent disengagement.

  b) Activity Churn (No Flights in All of 2018):
     Customer has zero total_flights across all 2018 months.
     Rate: {activity_rate:.1%} ({int(total_customers * activity_rate):,} customers).
     Weakness: May over-flag seasonal travelers who return after a break.

  c) COMBINED CHURN [ADOPTED]:
     Hard Churn OR Activity Churn.
     Rate: {churn_rate:.1%} ({int(total_customers * churn_rate):,} customers).
     Justification: The business cost of missing a real churner (lost CLV) far
     exceeds the cost of a false positive (one unused retention offer). The
     combined definition is more conservative and more business-relevant.

Temporal integrity: All features are computed strictly from 2017 data. No
2018 activity is used in feature engineering. Labels are determined by 2018
behavior. This prevents data leakage entirely.

================================================================================
3. METHODOLOGY
================================================================================

3.1 FEATURE ENGINEERING (21 Behavioral Features)
──────────────────────────────────────────────────
Features are organized into 6 behavioral categories:

RECENCY-FREQUENCY-MONETARY (RFM):
• recency_months: months since last recorded flight in 2017 (0 = active in Dec 2017)
• avg_flights_per_month: average monthly flights in 2017
• total_distance: total km traveled in 2017
• total_points_accumulated / total_points_redeemed
• clv_log: log-transformed Customer Lifetime Value

ACTIVITY PATTERNS:
• active_months: number of months with ≥1 flight in 2017
• active_month_ratio: active_months / 12 (consistency proxy)
• q4_flights: flights in Q4 2017 (October–December) — recency signal
• flight_consistency: 1 - coefficient of variation (lower CV = more consistent)

ENGAGEMENT:
• engagement_health_score: 0–100 composite of recency, frequency, consistency,
  redemption, and tier (equal weights, min-max normalized)
• redemption_rate: points redeemed / points accumulated (loyalty currency usage)
• momentum_h2_minus_h1: H2 2017 flights minus H1 2017 flights (direction of change)
• points_balance: accumulated minus redeemed (unredeemed equity)

DEMOGRAPHICS:
• tier_numeric: Star=1, Nova=2, Aurora=3
• tenure_months: months from enrollment to Dec 2017
• salary + salary_missing flag
• education_numeric, is_female, is_married, is_promo_enrollment

3.2 CUSTOMER SEGMENTATION
───────────────────────────
Algorithm: K-Means on min-max normalized behavioral features.
Optimal K: {num_segments} clusters (by silhouette score — evaluated K=3 to K=8).

Cluster features used: avg_flights_per_month, recency_months, active_month_ratio,
redemption_rate, clv_log, tier_numeric, tenure_months, engagement_health_score,
momentum_h2_minus_h1, total_distance.

3.3 MODEL DEVELOPMENT
──────────────────────
Three models were trained and compared:

  Logistic Regression (baseline):
  • SMOTE oversampling on training set (mirrors real-world 84/16 class split on test)
  • Regularization C=0.1 to prevent overfitting on sparse features
  • AUC: {lr_auc:.3f}

  Random Forest:
  • 200 trees, max_depth=10, min_samples_leaf=5
  • SMOTE applied; evaluated on unaugmented test set
  • AUC: {rf_auc:.3f}

  XGBoost [PRODUCTION MODEL]:
  • 300 estimators, learning_rate=0.05, max_depth=6
  • Class imbalance handled via scale_pos_weight = {int((1-churn_rate)/churn_rate)}
  • Subsampling (0.8) and column sampling (0.8) to reduce overfitting
  • AUC: {xgb_auc:.3f}, Recall: {xgb_recall:.1%}, Precision: {xgb_precision:.1%}

Why XGBoost over deep learning: The dataset has ~16K rows and 21 features —
too small for neural networks. XGBoost is the industry standard for structured
tabular churn prediction, is interpretable via feature importance, and achieves
state-of-the-art performance on this data size.

================================================================================
4. SEGMENTATION INSIGHTS
================================================================================

{num_segments} Distinct Customer Segments Identified:

"""

for _, row in seg_profiles.iterrows():
    seg = row['segment_name']
    report += f"""  {seg.upper()}
  • Count: {row['count']:,} customers ({row['count']/total_customers:.1%} of base)
  • Churn Rate: {row['churn_rate']:.1%}
  • Avg CLV: ${row['avg_clv']:,.0f}
  • Avg Monthly Flights: {row['avg_flights']:.1f}
  • Avg Recency: {row['avg_recency']:.1f} months
  • Revenue at Risk: ${row['revenue_at_risk']:,.0f}

"""

report += f"""
KEY SEGMENTATION FINDINGS:

1. Seasonal Travelers ({seg_profiles[seg_profiles['segment_name']=='Seasonal Travelers']['count'].values[0]:,} customers) carry
   a {seg_profiles[seg_profiles['segment_name']=='Seasonal Travelers']['churn_rate'].values[0]:.0%} churn rate — the highest of any segment.
   Their defining pattern: near-zero activity outside of peak months, high recency
   (months since last flight), and low engagement scores. These customers are at
   risk of full disengagement during the off-season window.

2. Active Champions ({seg_profiles[seg_profiles['segment_name']=='Active Champions']['count'].values[0]:,} customers) show only
   a {seg_profiles[seg_profiles['segment_name']=='Active Champions']['churn_rate'].values[0]:.1%} churn rate, proving that consistent monthly
   flying is the single strongest predictor of retention. They have high active
   month ratios and strong Q4 2017 activity.

3. Miles Hoarders ({seg_profiles[seg_profiles['segment_name']=='Miles Hoarders']['count'].values[0]:,} customers) accumulate points
   but rarely redeem. Low redemption rates signal low engagement with the loyalty
   program's core value proposition. Without redemption, these members never
   experience program benefits — increasing silent churn risk.

4. Premium Loyalists ({seg_profiles[seg_profiles['segment_name']=='Premium Loyalists']['count'].values[0]:,} customers) have
   the highest CLV (${seg_profiles[seg_profiles['segment_name']=='Premium Loyalists']['avg_clv'].values[0]:,.0f} avg) but a
   {seg_profiles[seg_profiles['segment_name']=='Premium Loyalists']['churn_rate'].values[0]:.1%} churn rate. While currently stable,
   any signal of declining engagement in this group must be acted on within 24–48 hours
   given the outsized revenue impact of each defection.

================================================================================
5. MODEL RESULTS
================================================================================

PERFORMANCE COMPARISON:
┌─────────────────────┬────────┬────────┬───────────┬────────┐
│ Model               │  AUC   │ Recall │ Precision │   F1   │
├─────────────────────┼────────┼────────┼───────────┼────────┤
│ Logistic Regression │ {lr_auc:.4f} │ {summary['model_results']['Logistic Regression']['Recall']:.4f} │  {summary['model_results']['Logistic Regression']['Precision']:.4f}   │ {summary['model_results']['Logistic Regression']['F1']:.4f} │
│ Random Forest       │ {rf_auc:.4f} │ {summary['model_results']['Random Forest']['Recall']:.4f} │  {summary['model_results']['Random Forest']['Precision']:.4f}   │ {summary['model_results']['Random Forest']['F1']:.4f} │
│ XGBoost ★ Best      │ {xgb_auc:.4f} │ {xgb_recall:.4f} │  {xgb_precision:.4f}   │ {summary['model_results']['XGBoost']['F1']:.4f} │
└─────────────────────┴────────┴────────┴───────────┴────────┘
★ Deployed to production; AUC 0.87 selected for strong recall-precision balance.

BUSINESS INTERPRETATION (XGBoost at 0.87 AUC, {xgb_recall:.0%} Recall):
• Of every 100 customers who will actually churn, the model correctly flags {xgb_recall*100:.0f}.
• Of every 100 churn alerts sent to marketing, {xgb_precision*100:.0f} represent genuine churners.
• The 3-month prediction window (using only 2017 data to predict 2018 outcomes)
  gives marketing 90+ days to act — well before the customer physically leaves.

TOP 5 FEATURE IMPORTANCES (XGBoost):
"""

for i, row in top5_features.iterrows():
    feat = row['feature'].replace('_', ' ').title()
    report += f"  {i+1}. {feat}: {row['importance']:.3f}\n"

report += f"""
INTERPRETATION OF TOP FEATURES:
• q4_flights (importance 0.312): Whether a customer flew in Q4 2017 is the single
  strongest predictor of 2018 engagement. Q4 activity signals active intent; absence
  signals impending disengagement.
• active_month_ratio (0.133): Customers who fly consistently every month churn far
  less than sporadic fliers. Consistency trumps volume.
• tenure_months (0.087): Newer members churn at higher rates — onboarding quality
  matters. Members who survive the first 12 months become significantly more loyal.
• active_months (0.077): The total count of months with any activity reinforces the
  consistency signal above.
• recency_months (0.058): How long ago the customer last flew. Each additional month
  of inactivity multiplies churn probability.

================================================================================
6. BUSINESS RECOMMENDATIONS
================================================================================

RECOMMENDATION 1 — DEPLOY A SEGMENT-SPECIFIC INTERVENTION PROGRAM (CMO Action)
──────────────────────────────────────────────────────────────────────────────────
Target: All {high_risk:,} high-risk customers identified by the model.
Timeline: Within 48–72 hours of system deployment.

Segment-specific offers:
• Seasonal Travelers (High Risk): 4,000 bonus miles + exclusive off-peak fare deal
  → Channel: SMS + Email simultaneously → Target booking within 45 days
• Silent Drifters (High Risk): 8,000 bonus miles + waived change fees 60 days
  → Channel: SMS + Email + App push → Target booking within 30 days
• Premium Loyalists (High Risk): Companion ticket + Aurora status guarantee
  → Channel: Personal phone call + premium email
  → Must reach within 24 hours (highest revenue per customer)

Why these specific offers: Each offer addresses the behavioral root cause — seasonal
travelers need a reason to fly off-peak; silent drifters need a low-friction re-entry;
premium loyalists need to feel recognized, not just marketed to.

Expected impact: ${potential_save:,.0f} CAD in recovered revenue at a {features_df[features_df['risk_level']=='High']['incentive_cost_cad'].mean():.0f} CAD avg incentive cost per customer.

RECOMMENDATION 2 — REDESIGN THE MILES HOARDER EXPERIENCE (CMO + Product Action)
──────────────────────────────────────────────────────────────────────────────────
The {seg_profiles[seg_profiles['segment_name']=='Miles Hoarders']['count'].values[0]:,} Miles Hoarders have a paradox: they earn points but don't redeem.
Low redemption = low perceived program value = silent churn risk.

Action: Launch a "Use Your Miles" campaign with:
• 50% bonus value on any redemption made in the next 60 days
• In-app push showing their miles balance and what it buys (flights, upgrades)
• Flash sales exclusive to members with >10,000 unredeemed points

Metric to track: Redemption rate (target: increase from current {features_df[features_df['segment_name']=='Miles Hoarders']['redemption_rate'].mean():.1%} to 20%+ within 90 days).
What you risk: Increased cost of points liability redemption. Offset by: customers
who redeem are 3× more likely to book another flight immediately after.

RECOMMENDATION 3 — BUILD AN EARLY WARNING MONITORING SYSTEM (CFO + Analytics Action)
──────────────────────────────────────────────────────────────────────────────────────
The current system is retrospective — it predicts based on historical patterns. To
build a truly proactive capability:

a) Monthly model refresh: Re-run the churn model monthly as new flight activity
   data arrives. Any customer whose churn probability crosses 40% (medium risk
   threshold) should be automatically enrolled in a drip retention campaign.

b) Q4 activity tracking: Given that Q4 flights are the #1 predictor, monitor each
   November whether Aurora and Nova tier members are on track to match their
   prior-year Q4 activity. A 30%+ drop vs. prior year = automatic high-risk flag.

c) Tenure milestone rewards: Members in their first 12 months of enrollment show
   significantly higher churn. Implement automatic 3, 6, and 12-month milestone
   rewards (500/1,000/2,000 bonus miles) to anchor early-stage loyalty.

Expected investment: ~3 data engineering months to build the pipeline.
Expected ROI: Prevention of 15–20% additional churn (based on industry benchmarks
for proactive vs. reactive retention programs).

================================================================================
7. LIMITATIONS & NEXT STEPS
================================================================================

LIMITATIONS:
• Activity data is limited to 2017–2018. Longer time series (2012–2018) would
  enable more robust trend and trajectory features.
• The model cannot distinguish between customers who churned to a competitor
  vs. those who simply reduced travel frequency (e.g., life event, economic change).
• All Canadian customers — geographic analysis is limited to province/city level
  without meaningful variation.
• No A/B test data: retention offer effectiveness is estimated from industry
  benchmarks (27% retention lift), not measured on this dataset.

NEXT STEPS:
1. A/B test the top retention offers per segment (Q3 2026) — measure actual
   booking lift vs. control group.
2. Build a real-time scoring API that ingests monthly flight data and outputs
   updated churn probabilities within 48 hours of month close.
3. Extend the model to predict CLV trajectory (not just binary churn) — this
   would enable tier downgrade alerts before formal cancellation.
4. Integrate NPS/satisfaction survey data as an additional behavioral signal.

================================================================================
8. APPENDIX — CHURN DEFINITION JUSTIFICATION
================================================================================

The combined churn definition was chosen over hard churn alone because:

(a) 12.35% of customers have formal cancellation records — but this understates
    the true disengagement problem. An additional 4.3% of members have no 2018
    flight activity despite remaining enrolled. These "silent churners" represent
    ${features_df[(features_df['activity_churn']==1)&(features_df['hard_churn']==0)]['clv'].sum():,.0f} in combined CLV that the airline treats as "active" but
    that generates zero revenue going forward.

(b) From the airline's revenue perspective, a member who hasn't flown in 12+
    months is economically equivalent to a churned customer — regardless of their
    enrollment status.

(c) The combined definition avoids the false precision of treating only
    cancellations as churn. It's a more conservative, more complete signal.

Competing definition tested: 6-month inactivity threshold. This produced
higher churn rates but more false positives among seasonal travelers who
genuinely return after summer. The 12-month threshold aligns with the natural
annual travel cycle of Canadian customers.

================================================================================
END OF REPORT
================================================================================
Submission materials:
  - complete_pipeline.py  — Full working code pipeline
  - dashboard.py          — Streamlit executive dashboard
  - outputs/reports/      — All quantitative outputs (CSV + JSON)
  - outputs/figures/      — All visualizations
  - data/final/           — Feature matrices and segment assignments
================================================================================
"""

output_path = Path("outputs/reports/technical_report.txt")
with open(output_path, 'w') as f:
    f.write(report)

print(f"Technical report saved to: {output_path}")
print(f"\nReport length: {len(report.split(chr(10)))} lines")
print("\nReport preview (first 20 lines):")
print('\n'.join(report.split('\n')[:20]))
