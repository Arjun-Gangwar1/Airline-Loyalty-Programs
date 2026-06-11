# Airline Loyalty Behavioral Intelligence — Complete Guide
## How to Run, What's Happening, and Key Concepts

---

## LIVE DASHBOARD (no setup needed)

**https://airline-loyalty-programs-dashboard.streamlit.app**

Open in any browser — publicly deployed on Streamlit Community Cloud.

---

## QUICK START (3 commands to run locally)

```bash
# Step 1 — Install packages (first time only)
pip install -r requirements.txt

# Step 2 — Run the full pipeline (all 9 stages, ~90 seconds)
python complete_pipeline.py

# Step 3 — Launch the dashboard locally
streamlit run dashboard.py
# Open browser → http://localhost:8501
```

---

## PROJECT STRUCTURE

```
airline_behavioral_intelligence/
│
├── data/
│   ├── raw/                         ← Original CSV files (never modify these)
│   │   ├── Customer Loyalty History.csv     (16,737 rows — one per member)
│   │   ├── Customer Flight Activity.csv     (392,936 rows — monthly activity)
│   │   ├── Calendar.csv
│   │   └── Airline Loyalty Data Dictionary.csv
│   │
│   ├── processed/                   ← Intermediate outputs (after cleaning)
│   │   ├── loyalty_clean.csv        ← Cleaned demographics
│   │   ├── activity_clean.csv       ← Cleaned flight records (with activity_date)
│   │   ├── churn_labels.csv         ← Who churned: 0 or 1 per customer
│   │   └── churn_definition_report.json
│   │
│   └── final/                       ← Model-ready datasets
│       ├── customer_features.csv    ← 43 features per customer
│       └── customer_segments.csv    ← Features + segment assignment + churn prob
│
├── outputs/
│   ├── models/                      ← Trained ML models (.pkl files)
│   │   ├── best_model.pkl           ← XGBoost (best performer)
│   │   ├── xgboost.pkl
│   │   ├── random_forest.pkl
│   │   └── logistic_regression.pkl
│   │
│   ├── figures/                     ← All charts and plots
│   │   ├── exploration/             ← EDA visualizations
│   │   ├── models/                  ← ROC curves, feature importance
│   │   └── segments/                ← Cluster heatmaps, segment breakdown
│   │
│   └── reports/                     ← Final outputs for submission
│       ├── retention_actions.csv    ← Per-customer action plan (4,868 customers)
│       ├── model_comparison.csv     ← LR vs RF vs XGBoost metrics
│       ├── feature_importance.csv   ← Top predictors ranked
│       ├── pipeline_summary.json    ← All key numbers in one file
│       └── technical_report.txt     ← Full 6-8 page written report
│
├── complete_pipeline.py             ← ★ MAIN SCRIPT — runs everything
├── dashboard.py                     ← ★ STREAMLIT DASHBOARD
├── generate_technical_report.py     ← Generates the written report
│
├── 1_data_exploration.py            ← Stage 1 (individual stage scripts)
├── 2_data_cleaning.py               ← Stage 2
├── 3_churn_definition.py            ← Stage 3
├── 4_feature_engineering.py         ← Stage 4
└── checkpoints/progress.json        ← Tracks which stages completed
```

---

## THE FULL PIPELINE — WHAT HAPPENS AT EACH STEP

### STEP 1 — DATA LOADING & CLEANING

**What happens:**
- Load raw CSVs: 16,737 loyalty members + 392,936 monthly flight records
- Fix missing salary values (25% of customers have no salary) → fill with median
- Create a `salary_missing` flag column so the model knows which values were imputed
- Build a proper `activity_date` column from `year` + `month` integer columns
- Remove 3,871 true duplicate records (same customer + same month twice)

**Key concept — Missing Data Imputation:**
> When data is missing, you have 3 options:
> 1. Drop the row → lose customers (bad for small datasets)
> 2. Fill with a stat (mean/median) → preserves all rows, adds noise
> 3. Fill + create a flag column → best of both: model can learn "this value was imputed"
>
> We use option 3. The `salary_missing` column (0 or 1) lets XGBoost
> learn: "customers who didn't report salary have different churn patterns."

---

### STEP 2 — DEFINING CHURN (the most critical design decision)

**What happens:**
Three definitions are tested and compared:

| Definition | How it works | Churn Rate | Problem |
|---|---|---|---|
| Hard Churn | Has a formal cancellation record | 3.9% | Misses silent quitters |
| Activity Churn | Zero flights in all of 2018 | 12.7% | May flag seasonal travelers |
| **Combined (used)** | Hard OR Activity | **16.3%** | Most complete picture |

**Why prediction date = 2017-12-31:**
- Features are built from **2017 data only**
- Labels (did they churn?) are determined by **2018 behavior**
- This creates a clean "predict the future from the past" setup
- Using 2018 data in features would be **data leakage** (cheating)

**Key concept — Data Leakage:**
> Leakage happens when your model accidentally "sees" the answer
> during training. Example: if you include "flights in Jan 2018" as a
> feature, the model learns "this person flew in 2018 → not churned."
> That's useless for real predictions — you won't have future data.
>
> Fix: strict temporal cutoff. ALL features = pre-2017-12-31.
> ALL labels = post-2017-12-31.

**Key concept — Class Imbalance:**
> Only 16.3% of customers churned. If you train naively, the model
> learns to just say "not churned" for everyone — gets 83.7% accuracy
> but catches zero actual churners. Useless.
>
> Fix 1: SMOTE (Synthetic Minority Oversampling) — creates artificial
> churn examples by interpolating between real ones.
> Fix 2: scale_pos_weight in XGBoost — tells the model "treat each
> churner as 5× more important than a non-churner."

---

### STEP 3 — FEATURE ENGINEERING (building behavioral signals)

**What happens:**
From 12 months of raw flight data (2017), we compute 27 behavioral features per customer across 6 groups.

**All 27 features explained:**

**RFM Group (core loyalty health):**
| Feature | How computed | What it captures |
|---|---|---|
| `recency_months` | 12 − last_month_with_flight | How long since last flight (0 = flew in Dec 2017) |
| `avg_flights_per_month` | total_flights / 12 | Typical activity level |
| `total_distance` | sum of monthly km | Volume of travel |
| `total_points_accumulated` | sum of points earned | Loyalty program engagement |
| `total_points_redeemed` | sum of points spent | Actually using the program |
| `clv_log` | log(CLV + 1) | Log-transform reduces skew from outliers |

**Activity Patterns:**
| Feature | How computed | What it captures |
|---|---|---|
| `active_months` | count of months with flights > 0 | How many months they actually flew |
| `active_month_ratio` | active_months / 12 | Consistency (0 = never flew, 1 = flew every month) |
| `q4_flights` | sum of flights in Oct–Dec 2017 | Most recent quarter — #1 predictor |
| `q1_flights` | sum of flights in Jan–Mar 2017 | Early-year baseline activity |
| `flight_consistency` | 1 − (std/mean) of monthly flights | Predictability; low = erratic flier |
| `momentum_h2_minus_h1` | (Jul–Dec flights) − (Jan–Jun flights) | Trending up or down in 2017? |
| `points_balance` | accumulated − redeemed | Unclaimed loyalty currency |

**Engagement & Economics:**
| Feature | How computed | What it captures |
|---|---|---|
| `engagement_health_score` | weighted composite 0–100 | Single number summarizing loyalty strength |
| `redemption_rate` | redeemed / accumulated | Using miles = engaged with program |
| `dollar_cost_per_flight` | total CAD value redeemed / flights [NEW] | Economic intensity of redemption |
| `salary` | from loyalty history | Income proxy |
| `salary_missing` | 0/1 flag | Was salary imputed? |

**Geographic [NEW — 11 provinces, 12.7%–24.2% churn variation]:**
| Feature | How computed | What it captures |
|---|---|---|
| `is_ontario` | province == 'Ontario' | Largest member base (5,404 members) |
| `is_bc` | province == 'British Columbia' | Second-largest province |
| `is_quebec` | province == 'Quebec' | Third-largest province |
| `is_alberta` | province == 'Alberta' | Fourth-largest province |

**Cohort [NEW — enrollment year churn patterns]:**
| Feature | How computed | What it captures |
|---|---|---|
| `is_recent_member` | enrolled in 2017+ | New members churn at 18% — #2 predictor |
| `is_early_member` | enrolled ≤ 2014 | Veteran members churn at 14% |

**Demographics:**
| Feature | How computed | What it captures |
|---|---|---|
| `tier_numeric` | Star=1, Nova=2, Aurora=3 | Loyalty tier as ordinal number |
| `tenure_months` | months since enrollment | How long they've been a member |
| `education_numeric` | High School=1 … Doctor=5 | Education level as number |
| `is_female`, `is_married`, `is_promo_enrollment` | binary flags | Demographic signals |

**Key concept — Why log-transform CLV?**
> CLV ranges from $1,898 to $83,325. Raw values would dominate the model.
> log(83325) ≈ 11.3, log(1898) ≈ 7.5 — much more compressed range.
> Linear models especially need this. Tree models (RF, XGBoost) are less
> sensitive but it still helps.

**Key concept — Why engagement_health_score?**
> Instead of feeding the model 10 separate variables, we combine them
> into one 0–100 score. Benefits:
> - Easier for humans to understand
> - Reduces correlated features going into the model
> - Can be monitored monthly as a single KPI

---

### STEP 4 — CUSTOMER SEGMENTATION (K-Means Clustering)

**What happens:**
Group 16,737 customers into behaviorally distinct clusters using K-Means.

**How K-Means works (step by step):**
1. Choose K = number of clusters (we test K=3 to K=8)
2. Randomly place K "centroids" (cluster centers) in feature space
3. Assign every customer to the nearest centroid
4. Recalculate each centroid as the mean of its assigned customers
5. Repeat steps 3–4 until assignments stop changing
6. Result: K groups where customers within a group are similar

**How we choose K — Silhouette Score:**
> Silhouette score measures "how well does each point fit its cluster
> vs. the next-nearest cluster?" Range: -1 to 1.
> - 1.0 = perfect separation
> - 0.0 = overlapping clusters
> - Negative = misclassified
>
> We test K=4 to K=8 (minimum 4 for business usefulness) and pick
> K=4 because it gives the highest silhouette score.

**The 4 segments found:**

| Segment | Size | Churn Rate | Avg CLV | Key Pattern | Priority |
|---|---|---|---|---|---|
| **Premium Dormant** | 3,447 (20.6%) | **36.8%** | $9,134 | High CLV, near-zero activity | CRITICAL |
| **Seasonal Travelers** | 2,843 (17.0%) | 35.4% | $6,742 | Low activity, low engagement | HIGH |
| **Active Champions** | 5,653 (33.8%) | 4.6% | $8,976 | Fly every month, consistent | Monitor |
| **Miles Hoarders** | 4,794 (28.6%) | 4.0% | $6,742 | Earn points, low redemption | Nurture |

**Key concept — Why segment before modeling?**
> Different customers churn for different reasons. A Seasonal Traveler
> needs "here's a deal to fly off-peak." A Miles Hoarder needs "use your
> points before they expire." A Premium Loyalist needs a phone call.
> Segmentation lets you send the RIGHT message, not a generic one.

---

### STEP 5 — MACHINE LEARNING MODELS

**What happens:**
Train 3 models, evaluate on a held-out test set, pick the best.

**The train/test split:**
- 80% of customers → training data (model learns from this)
- 20% of customers → test data (model never sees this during training)
- We use `stratify=y` to ensure both splits have the same 16.3% churn rate

**Model 1 — Logistic Regression (baseline):**
```
Learns: churn_probability = sigmoid(w1×recency + w2×flights + ... + b)
```
- Simple linear model — each feature gets one weight
- After SMOTE to balance classes
- AUC: 0.8435, Recall: 77.3%
- Good recall but poor precision (too many false alarms)

**Model 2 — Random Forest:**
```
Builds 200 decision trees on random subsets of data + features.
Final prediction = majority vote across all 200 trees.
```
- Better than LR because it captures non-linear relationships
- AUC: 0.8740 (best), Recall: 50.2%, Precision: 90.7%
- High precision (when it says churn, it's usually right)
- Lower recall (misses more real churners)

**Model 3 — XGBoost (chosen for production):**
```
Builds trees sequentially. Each new tree focuses on the mistakes
of the previous trees. Gradient boosting = gradient descent on trees.
```
- AUC: 0.8711, Recall: 67.2%, Precision: 52.9%
- Best balance of recall and precision
- 5-fold cross-validation AUC: 0.8771
- scale_pos_weight handles class imbalance without SMOTE

**Key concept — What is AUC-ROC?**
> AUC (Area Under the ROC Curve) measures how well the model RANKS
> customers by risk — regardless of the threshold you use.
>
> AUC = 0.87 means: pick any random churner and any random non-churner.
> There's an 87% chance the model gives the churner a higher churn score.
>
> AUC = 0.5 = random guessing (coin flip)
> AUC = 1.0 = perfect (impossible in practice)
> AUC > 0.80 = strong model, ready for business use

**Key concept — Recall vs Precision trade-off:**
> Recall = of all REAL churners, what % did we catch?
>   → "Don't miss any at-risk customer"
>   → False negatives are expensive (lost revenue)
>
> Precision = of all our ALERTS, what % are real churners?
>   → "Don't waste offers on people who weren't going to leave"
>   → False positives are cheap (unused offer voucher)
>
> For churn prevention: prioritize Recall. Missing a churner costs
> their entire CLV ($7,989 avg). A false alarm costs one offer (~$50).

**Key concept — How does XGBoost work?**
```
Round 1: Build a shallow tree. Predicts poorly.
Round 2: Build another tree that focuses on where Round 1 was wrong.
Round 3: Build another tree that focuses on Round 1+2 errors.
...
Round 300: Sum all 300 trees weighted by learning_rate=0.05.
```
> Each tree "corrects" the previous ensemble. This is why it's called
> gradient boosting — each step moves in the direction that reduces the
> loss function (log-loss for classification).

---

### STEP 6 — FEATURE IMPORTANCE

**Top 10 most important predictors (XGBoost):**

| Rank | Feature | Importance | Why it matters |
|---|---|---|---|
| 1 | `q4_flights` | 0.246 | Flew in Q4 2017? Best single signal of staying active in 2018 |
| 2 | `is_recent_member` | 0.198 | Members enrolled 2017+ churn at 18% — structurally at-risk |
| 3 | `active_month_ratio` | 0.087 | Consistency beats volume — flying every month > flying a lot once |
| 4 | `active_months` | 0.077 | Total months with activity — reinforces consistency signal |
| 5 | `tenure_months` | 0.062 | Long-tenure members are more loyal |
| 6 | `avg_flights_per_month` | 0.025 | Volume of activity |
| 7 | `flight_consistency` | 0.022 | Predictable fliers stay loyal |
| 8 | `points_balance` | 0.019 | Unspent miles = disengagement signal |
| 9 | `total_points_accumulated` | 0.019 | Total engagement with program |
| 10 | `total_distance` | 0.018 | Volume of travel covered |

**Key concept — Feature Importance in XGBoost:**
> Importance = how often a feature is used to split trees, weighted
> by the improvement in loss each split provides.
>
> High importance ≠ causal. q4_flights being #1 doesn't mean
> "make them fly in Q4." It means Q4 activity is the most reliable
> observable signal of future loyalty. The cause may be something else
> (job type, location, life stage).

---

### STEP 7 — RETENTION ENGINE

**What happens:**
For every at-risk customer, the system outputs a specific action:

```
Input:  customer ID
Output: risk level + specific offer + channel + timing + expected revenue saved
```

**The logic:**
```python
churn_probability = xgboost.predict(customer_features)

if probability >= 0.70:   risk = "High"
elif probability >= 0.40: risk = "Medium"
else:                     risk = "Low"

offer = offer_library[segment][risk]
revenue_at_risk = CLV × churn_probability × 0.82
potential_save  = revenue_at_risk × 0.27
```

**Revenue calculation explained:**
- `× 0.82` → only 82% of CLV is future revenue (18% already received)
- `× 0.27` → industry benchmark: targeted retention saves ~27% of at-risk revenue

**Example output (one customer):**
```
Customer #547821 | Aurora tier | Seasonal Travelers | HIGH RISK
Churn probability: 84.2%
CLV: $12,400
Revenue at risk: $8,579
Potential save:   $2,316

Action:  Urgent Win-Back Campaign
Offer:   4,000 bonus miles + exclusive off-peak fare deal
Channel: SMS + Email simultaneously
Timing:  Within 48 hours
```

---

### STEP 8 — DASHBOARD

**5 pages explained:**

**Page 1 — Executive Overview**
For CEOs and CMOs. Shows:
- 5 KPI cards: total members, churn rate, high-risk count, revenue at risk, potential savings
- Revenue at risk by segment (bar chart, colored by churn rate)
- Risk distribution donut (High / Medium / Low)
- Top 15 priority customers by revenue at risk
- Churn definition comparison table

**Page 2 — Churn Intelligence**
For data and analytics teams. Shows:
- Model comparison cards (LR vs RF vs XGBoost + CV scores)
- Feature importance horizontal bar chart (top 15)
- Churn probability distribution histogram
- Churn rate by loyalty tier
- CLV vs churn probability scatter plot

**Page 3 — Segment Analysis**
For marketing managers. Shows:
- Segment cards with key stats (count, churn rate, avg CLV)
- Risk vs Value scatter matrix
- Revenue at risk by segment
- Behavioral heatmap (what makes each segment different?)
- Segment drill-down selector

**Page 4 — Geography & Demographics [NEW]**
For regional managers and HR. Shows:
- Province churn rate bar chart (11 Canadian provinces)
- Province revenue at risk bar chart
- Province summary table
- Enrollment cohort analysis (churn by enrollment year 2012–2018)
- Demographic breakdown: education, gender, marital status

**Page 5 — Retention Actions**
For operations teams. Shows:
- 4 filters: risk level, segment, loyalty tier, province
- Full action table (4,883 at-risk customers)
- Download CSV button → ready for CRM import
- Segment playbooks (expandable per segment)

---

## ALL RESULTS — NUMBERS TO KNOW

### Dataset
| Fact | Value |
|---|---|
| Total customers | 16,737 |
| Activity data period | 2017–2018 |
| Loyalty tiers | Star (46%), Nova (34%), Aurora (20%) |
| Average CLV | $7,989 CAD |
| Salary missing | 25.3% → filled with median $73,455 |

### Churn
| Definition | Rate | Count |
|---|---|---|
| Hard churn (cancellation) | 3.9% | 645 |
| Activity churn (no 2018 flights) | 12.7% | 2,123 |
| **Combined (used in model)** | **16.3%** | **2,728** |

### Segments
| Segment | Count | Churn Rate | Avg CLV | Revenue at Risk |
|---|---|---|---|---|
| **Premium Dormant** | 3,447 (20.6%) | **36.8%** | $9,134 | $14.7M |
| **Seasonal Travelers** | 2,843 (17.0%) | 35.4% | $6,742 | $8.6M |
| Active Champions | 5,653 (33.8%) | 4.6% | $8,976 | $3.8M |
| Miles Hoarders | 4,794 (28.6%) | 4.0% | $6,742 | $2.3M |

### Models
| Model | AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8435 | 77.3% | 40.6% | 53.2% |
| Random Forest | 0.8740 | 50.2% | 90.7% | 64.6% |
| **XGBoost (chosen)** | **0.8711** | **67.2%** | **52.9%** | **59.2%** |
| XGBoost 5-fold CV | **0.8771** | — | — | — |

### Business Impact
| Metric | Value |
|---|---|
| High-risk customers | 1,862 |
| Medium-risk customers | 3,021 |
| Total revenue at risk | $29.4M CAD |
| Potential recoverable revenue | $7.9M CAD |

### Geographic Findings
| Province | Churn Rate | Members | Revenue at Risk |
|---|---|---|---|
| Prince Edward Island | 24.2% (highest) | 66 | $128K |
| Manitoba | 19.3% | 658 | $1.37M |
| Ontario | 16.2% | 5,404 | $9.31M |
| British Columbia | 15.7% | 4,409 | $7.52M |
| Yukon | 12.7% (lowest) | 110 | $123K |

---

## HOW TO RUN (STEP BY STEP OPTIONS)

### Option A — Full pipeline in one command (recommended)
```bash
venv/bin/python3 complete_pipeline.py
```
Runs all 9 stages, saves all outputs, takes ~60 seconds.

### Option B — Run individual original scripts
```bash
# Stage 1 — Explore raw data
venv/bin/python3 1_data_exploration.py

# Stage 2 — Clean data (FIXED: now loads from raw CSV directly)
venv/bin/python3 2_data_cleaning.py

# Stage 3 — Define churn labels (FIXED: proper date parsing)
venv/bin/python3 3_churn_definition.py

# Stage 4 — Feature engineering (FIXED: same date + merge fixes)
# Note: this file is a duplicate of stage 3 in the original project
venv/bin/python3 4_feature_engineering.py

# Stages 5-7 — Run via complete_pipeline.py (original files had bugs)
venv/bin/python3 complete_pipeline.py
```

### Option C — Just the dashboard (if pipeline already ran)
```bash
venv/bin/streamlit run dashboard.py
# Opens at http://localhost:8501
```

### Option D — Regenerate the technical report
```bash
venv/bin/python3 generate_technical_report.py
# Output: outputs/reports/technical_report.txt
```

---

## BUGS FIXED IN THE ORIGINAL SCRIPTS

Understanding these bugs helps you avoid them in your own projects:

### Bug 1 — `pd.to_datetime()` on integer month numbers
**File:** `2_data_cleaning.py` (line 197), same in `3_churn_definition.py`, `4_feature_engineering.py`

**What went wrong:**
```python
# WRONG — treats integer 6 as "6 nanoseconds since Unix epoch"
df['month'] = pd.to_datetime(df['month'])
# Result: "1970-01-01 00:00:00.000000006" for month=6
```

**The fix:**
```python
# RIGHT — build date from year + month columns
df['activity_date'] = pd.to_datetime(
    {'year': df['year'], 'month': df['month'], 'day': 1}
)
# Result: 2017-06-01 for year=2017, month=6
```

**Why it mattered:** With 1970 dates, ALL activity was "before" the 2017-12-31
prediction date cutoff, so every customer appeared to have no 2018 activity → 100% churn rate.

### Bug 2 — Column collision on second merge
**File:** `3_churn_definition.py` (line 177), same in `4_feature_engineering.py`

**What went wrong:**
```python
# Called with months_inactive=12 → adds 'last_activity_date' column to loyalty df
define_activity_churn(12)
# Called again with months_inactive=6 → tries to merge 'last_activity_date' AGAIN
# But the column already exists → KeyError
define_activity_churn(6)
```

**The fix:**
```python
# Drop the column before re-merging
if 'last_activity_date' in self.loyalty.columns:
    self.loyalty = self.loyalty.drop(columns=["last_activity_date"])
self.loyalty = self.loyalty.merge(last_activity_df, ...)
```

### Bug 3 — `int64`/`Series` not JSON serializable
**Files:** `2_data_cleaning.py`, `3_churn_definition.py`, `4_feature_engineering.py`

**What went wrong:**
```python
# pandas int64 and numpy types are not Python native types
# json.dump() only handles Python int, not numpy.int64
json.dump({'count': np.int64(500)}, f)  # TypeError!
```

**The fix:**
```python
def make_serializable(obj):
    if hasattr(obj, 'tolist'):   return obj.tolist()   # numpy arrays
    if hasattr(obj, 'item'):     return obj.item()     # numpy scalars
    if isinstance(obj, dict):    return {k: make_serializable(v) for k,v in obj.items()}
    if isinstance(obj, list):    return [make_serializable(i) for i in obj]
    return obj

json.dump(make_serializable(report), f)
```

### Bug 4 — Over-aggressive deduplication
**File:** `2_data_cleaning.py`

**What went wrong:**
```python
# Original dedup used only [customer_id, month_column]
# After bad date conversion, ALL month values were "1970-01-01 00:00:00.000000001"
# → every record appeared as a duplicate of the first
# Removed 193,063 rows when only 3,871 were real duplicates
```

**The fix:** Load directly from raw CSV (fresh integers), then the real-duplicate check works correctly.

---

## KEY CONCEPTS CHEAT SHEET

### Machine Learning Terms

**Supervised Learning**
> You give the model labeled examples (customer + did_they_churn).
> It learns patterns to predict unseen customers.

**Binary Classification**
> Output is one of two classes: churned (1) or not churned (0).
> The model outputs a probability 0–1, you pick a threshold (e.g. 0.5).

**Train/Test Split**
> Never evaluate a model on the same data it trained on.
> You'd be measuring memorization, not generalization.
> 80% train, 20% test is standard.

**Cross-Validation**
> More robust than a single split. Rotate the test window 5 times,
> average the results. Reduces luck in the split.

**Overfitting**
> Model learns the training data too well (including noise).
> Works great on training, fails on new data.
> Signs: train accuracy >> test accuracy.
> Fixes: regularization, early stopping, max_depth, min_samples_leaf.

**Gradient Boosting (XGBoost)**
> An ensemble of weak trees built sequentially.
> Each tree fixes the errors of the previous trees.
> "Gradient" = uses calculus (gradients) to decide which errors to fix.

**SMOTE (Synthetic Minority Oversampling)**
> For 84/16 class split, create fake churn examples by:
> Pick a real churner → find its K nearest churner neighbors →
> Interpolate a new point between them → add to training set.
> Goal: balance classes without losing information.

### Statistics Terms

**AUC-ROC (0.87)**
> Probability that a randomly chosen churner gets a higher churn score
> than a randomly chosen non-churner. 0.87 = excellent.

**Recall / Sensitivity (68.5%)**
> Of all actual churners, what % did we flag?
> Formula: True Positives / (True Positives + False Negatives)

**Precision (53.8%)**
> Of all our churn alerts, what % were real churners?
> Formula: True Positives / (True Positives + False Positives)

**F1 Score (60.3%)**
> Harmonic mean of precision and recall.
> Formula: 2 × (Precision × Recall) / (Precision + Recall)
> Use when you care about both equally.

**K-Means Clustering**
> Unsupervised algorithm. Groups data by minimizing within-cluster variance.
> No labels needed — finds natural groupings in the data.

**Silhouette Score**
> Measures cluster quality. For each point:
> (distance to nearest other cluster − distance to own cluster) / max of both
> Range -1 to 1. Higher = better separated clusters.

**Log Transform**
> Applied to CLV because raw CLV has a long right tail (a few very rich customers).
> log(x) compresses large values, makes distribution more normal.
> Helps linear models and prevents outliers from dominating.

### Business Terms

**CLV (Customer Lifetime Value)**
> Total revenue a customer has generated. Here: total flight invoice value in CAD.

**Churn**
> Customer stops being active. Two types:
> Hard churn = formally cancels. Soft/silent churn = just stops using the service.

**Retention Rate**
> % of customers who continue using the service year-over-year.
> Complement of churn rate. 100% − 16.3% = 83.7% retention.

**RFM Analysis**
> Classic marketing framework:
> R = Recency (when did they last buy?)
> F = Frequency (how often do they buy?)
> M = Monetary (how much do they spend?)
> High RFM = high-value, loyal customer.

**Revenue at Risk**
> Estimated revenue that will be lost if the customer churns.
> Formula: CLV × churn_probability × 0.82

**Intervention ROI**
> If offer costs $50 and saves $2,300 in CLV → 46× return.
> Compare this to CAC (customer acquisition cost) to justify the spend.

---

## SUBMISSION CHECKLIST

For the IIT Guwahati C&A Club submission:

- [x] Working Prototype → `dashboard.py` (Streamlit, 4 pages)
- [x] Technical Report (6-8 pages) → `outputs/reports/technical_report.txt`
- [x] Churn prediction model → `outputs/models/best_model.pkl` (XGBoost AUC 0.87)
- [x] Customer segmentation → 4 named segments with behavioral profiles
- [x] Smart retention → per-customer specific offers with channel + timing
- [x] Data quality documentation → bugs found and fixed, decisions justified
- [x] Temporal leakage prevention → all features from 2017, labels from 2018
- [x] Three business recommendations → in technical_report.txt Section 6
- [x] All visualizations → `outputs/figures/` (7 charts)
- [x] Retention actions CSV → `outputs/reports/retention_actions.csv`

---

*Stack: Python 3.12 · Pandas · NumPy · Scikit-learn · XGBoost · SMOTE · Streamlit · Plotly*
*Dataset: 16,737 Canadian airline loyalty members, 2017–2018*
