# ✈️ Airline Loyalty Behavioral Intelligence Platform
### AI-Powered Churn Prediction, Customer Segmentation & Retention Engine

---

## � What This Project Builds

A **production-ready AI system** that:
- Predicts customer churn **3+ months early** with AUC 0.80+
- Creates **actionable behavioral segments** (not just "Cluster 1, 2, 3")
- Recommends **specific retention actions** per customer
- Delivers an **executive dashboard** any manager can use

---

## �️ File Structure

```
airline_behavioral_intelligence/
│
├── � resume.py                    ← START HERE after any break
├── � run_pipeline.py              ← Run all stages automatically
│
├── � 00_project_setup.py          Stage 0: Setup
├── � 01_data_exploration.py       Stage 1: Explore data
├── � 02_data_cleaning.py          Stage 2: Clean data
├── � 03_churn_definition.py       Stage 3: Define churn ⭐
├── � 04_feature_engineering.py    Stage 4: Build 30-50 features ⭐
├── � 05_customer_segmentation.py  Stage 5: Create segments
├── � 06_baseline_models.py        Stage 6: Train ML models
├── � 07_retention_engine.py       Stage 7: Build retention engine ⭐
├── � 08_dashboard.py              Stage 8: Streamlit dashboard
│
├── � data/
│   ├── raw/                        ← Put your CSV files here
│   ├── processed/                  ← Auto-generated checkpoints
│   └── final/                      ← Final outputs
│
├── � outputs/
│   ├── models/                     ← Saved ML models (.pkl)
│   ├── figures/                    ← All visualizations (.png)
│   └── reports/                    ← Reports (.json, .csv)
│
├── � checkpoints/
│   └── progress.json               ← Tracks your progress
│
└── � logs/                        ← Error logs
```

---

## � Quick Start (First Time)

### Step 1: Setup
```bash
python 00_project_setup.py
```

### Step 2: Download Data
Place these files in `data/raw/`:
- `Customer_Loyalty_History.csv`
- `Customer_Flight_Activity.csv`
- `Calendar.csv`
- `Data_Dictionary.csv`

### Step 3: Run the Pipeline
```bash
# Option A: Run all stages automatically
python run_pipeline.py

# Option B: Run stage by stage (recommended for learning)
python 01_data_exploration.py
python 02_data_cleaning.py
python 03_churn_definition.py
python 04_feature_engineering.py
python 05_customer_segmentation.py
python 06_baseline_models.py
python 07_retention_engine.py
```

### Step 4: Launch Dashboard
```bash
streamlit run 08_dashboard.py
```

---

## � Resuming After a Break

**Every time you restart, run this first:**
```bash
python resume.py
```

This tells you exactly where you left off and what to run next.

---

## � Stage-by-Stage Guide

| Stage | Script | What It Does | Output |
|-------|--------|-------------|--------|
| 0 | `00_project_setup.py` | Creates directories, installs packages | Project structure |
| 1 | `01_data_exploration.py` | Data quality report, initial analysis | EDA visualizations |
| 2 | `02_data_cleaning.py` | Handle missing values, fix types | `loyalty_clean.csv`, `activity_clean.csv` |
| 3 | `03_churn_definition.py` | Define & validate churn (CRITICAL) | `churn_labels.csv` |
| 4 | `04_feature_engineering.py` | 30-50 behavioral features (CRITICAL) | `customer_features.csv` |
| 5 | `05_customer_segmentation.py` | 4-7 named segments + strategies | `customer_segments.csv` |
| 6 | `06_baseline_models.py` | LR + RF + XGBoost, AUC 0.80+ | `best_model.pkl`, predictions |
| 7 | `07_retention_engine.py` | Next-best-action per customer | `retention_actions.csv` |
| 8 | `08_dashboard.py` | 4-page interactive Streamlit app | Live dashboard |

---

## � Feature Engineering (Stage 4) — The Differentiator

This stage creates **30-50 behavioral features** across 7 categories:

| Category | Examples | Why Important |
|----------|----------|---------------|
| **RFM** | Recency days, avg flights/month, CLV | Foundation metrics |
| **Momentum** | Flight trend 3m vs 6m, engagement slope | Direction of change |
| **Volatility** | Consistency score, CV of flights | Predictability of behavior |
| **Temporal** | Seasonality score, holiday traveler | Travel motivation |
| **Psychology** | Redemption rate, earn-burn ratio | Loyalty engagement |
| **Trajectory** | YoY growth, activity trajectory | Long-term value trend |
| **Engagement** | Health score 0-100, active month ratio | Overall loyalty strength |

---

## � Customer Segments (Stage 5)

| Segment | Description | Strategy | Urgency |
|---------|-------------|---------|---------|
| **Premium Loyalists** | High CLV, consistent | VIP treatment | Low |
| **Silent Drifters** | Declining engagement | Urgent win-back | � High |
| **Miles Hoarders** | High balance, low redemption | Redemption incentives | Medium |
| **Seasonal Travelers** | Holiday/vacation flyers | Seasonal campaigns | Low-Med |
| **Rising Stars** | Growing engagement | Tier upgrade challenge | Low |
| **Budget Frequent Flyers** | High freq, low CLV | Partner benefits | Medium |
| **At-Risk VIPs** | High-value, declining | Maximum intervention | � Critical |

---

## � Expected Model Performance

| Model | AUC Target | Recall Target | Notes |
|-------|-----------|--------------|-------|
| Logistic Regression | 0.70-0.75 | 60%+ | Interpretable baseline |
| Random Forest | 0.75-0.80 | 65%+ | Feature importance |
| **XGBoost** | **0.80-0.85+** | **70%+** | **Best model** |

**Business Translation:** XGBoost at AUC 0.84 means:
- We identify **70% of churners** 3+ months before they leave
- **63% of churn alerts** are correct (acceptable false positive rate)
- Early enough for **marketing to intervene** successfully

---

## � Retention Engine Outputs (Stage 7)

For every at-risk customer, the system outputs:

```
Customer: #12345
Segment:  Silent Drifters
Risk:     HIGH (82% churn probability)
CLV:      $4,200

Why at risk:
  - No flights in 4 months
  - Zero redemptions in 6 months
  - Declining engagement trend

Recommended Action: Urgent reactivation
Offer: 8K bonus miles + waived change fees for 60 days
Channel: SMS + Email + App push
Timing: Within 48 hours
Expected Retention Lift: +28%
Revenue at Risk: $3,444
Potential Save: $964
```

---

## � Dashboard Pages (Stage 8)

| Page | What You See | Who Uses It |
|------|-------------|------------|
| � Executive Overview | KPIs, risk distribution, top at-risk list | CEO/CMO |
| � Churn Intelligence | Probability distributions, segment breakdown | Analytics team |
| � Segment Analysis | Behavioral profiles, risk maps | Marketing managers |
| � Retention Actions | Priority action list, downloadable CSV | Operations team |

---

## � Common Issues & Fixes

### "File not found" errors
```bash
python resume.py    # Check which stage is missing
python run_pipeline.py --from 3   # Re-run from Stage 3
```

### SMOTE import error
```bash
pip install imbalanced-learn
```

### XGBoost not found
```bash
pip install xgboost
```

### Streamlit won't start
```bash
pip install streamlit
streamlit run 08_dashboard.py
```

### Memory error on large data
- Edit `06_baseline_models.py` → add `n_jobs=1` to models
- Reduce sample: `df = df.sample(frac=0.5, random_state=42)`

---

## � Technical Report Outline (6-8 pages)

Your report should cover:

1. **Executive Summary** (1 page)
   - Problem: `$X.XM` revenue at risk from churn
   - Solution: Behavioral intelligence platform
   - Key finding: Catch 70% of churners 3 months early
   - Recommendation: Segment-specific interventions

2. **Churn Definition** (0.5 page)
   - Combined approach (cancellation OR 12-month inactivity)
   - Why this definition, temporal validation

3. **Methodology** (1.5 pages)
   - 30-50 behavioral features (momentum, volatility, trajectory)
   - XGBoost with temporal validation
   - Segment-specific analysis

4. **Segmentation Insights** (1 page)
   - 5-7 named segments with profiles
   - Revenue contribution per segment

5. **Model Results** (1 page)
   - AUC, Recall, Precision (business-translated)
   - Feature importance (top 10 behavioral drivers)

6. **Business Recommendations** (1.5 pages)
   - Segment-specific retention strategies
   - Priority actions with expected ROI
   - Implementation roadmap

7. **Limitations & Next Steps** (0.5 page)
   - Dataset ends 2018, model needs refresh
   - A/B test recommendations
   - Real-time scoring system roadmap

---

## � Presentation Strategy

**Lead with business impact, not ML metrics:**

❌ "Our model has AUC of 0.84"
✅ "We identify 70% of churners 3 months early, potentially saving $X.XM annually"

**Show the dashboard first** — visual impact
**Walk through 2-3 customer examples** with recommendations
**Quantify everything** — $ saved, % improvement, ROI

---

## � Getting Help

If you're stuck:

1. Run `python resume.py` — shows exactly where you are
2. Check `logs/` folder for error messages
3. Each script has detailed error messages built in
4. Re-run any stage independently — they all load from checkpoints

---

*Built with: Python, Pandas, Scikit-learn, XGBoost, Streamlit, Plotly*
*Dataset: ~16,700 Canadian loyalty members, 2012-2018*