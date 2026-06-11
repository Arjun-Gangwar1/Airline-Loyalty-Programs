# ✈️ Airline Loyalty Behavioral Intelligence Platform
### Churn Prediction · Customer Segmentation · Smart Retention
**IIT Guwahati — C&A Club Summer Projects '26**

---

## Quick Start (3 commands)

```bash
# 1. Install dependencies (first time only)
venv/bin/pip install pandas numpy scipy matplotlib seaborn plotly \
  scikit-learn xgboost imbalanced-learn streamlit joblib

# 2. Run the full pipeline (all 9 stages, ~90 seconds)
venv/bin/python complete_pipeline.py

# 3. Launch the dashboard
venv/bin/streamlit run dashboard.py
# Open browser → http://localhost:8501

# Optional: regenerate the technical report
venv/bin/python generate_technical_report.py
```

---

## Project Structure

```
airline_behavioral_intelligence/
│
├── data/
│   ├── raw/                          ← Original CSVs (never modify)
│   │   ├── Customer Loyalty History.csv      (16,737 members)
│   │   ├── Customer Flight Activity.csv      (392,936 monthly records)
│   │   ├── Calendar.csv                      (date dimension)
│   │   └── Airline Loyalty Data Dictionary.csv
│   ├── processed/                    ← Cleaning & churn label outputs
│   └── final/                        ← Model-ready feature matrices
│       ├── customer_features.csv     (16,737 × 54 features)
│       └── customer_segments.csv     (features + segment + churn prob + risk level)
│
├── outputs/
│   ├── models/                       ← Trained models (.pkl)
│   │   ├── best_model.pkl            (XGBoost — production model)
│   │   ├── xgboost.pkl
│   │   ├── random_forest.pkl
│   │   └── logistic_regression.pkl
│   ├── figures/                      ← All PNG charts
│   │   ├── exploration/              (geographic, cohort, demographic analysis)
│   │   ├── models/                   (ROC curves, confusion matrix, model comparison)
│   │   ├── segments/                 (behavioral heatmap, segment analysis)
│   │   └── churn/                    (definition comparison, probability distribution)
│   └── reports/
│       ├── retention_actions.csv     (4,800+ at-risk customers + specific actions)
│       ├── technical_report.txt      (formal 8-section report)
│       ├── model_comparison.csv
│       ├── feature_importance.csv
│       └── pipeline_summary.json     (all key numbers in one file)
│
├── complete_pipeline.py              ★ MAIN SCRIPT — runs all 9 stages
├── dashboard.py                      ★ STREAMLIT DASHBOARD (5 pages)
├── generate_technical_report.py      ← Generates formatted technical report
│
├── 1_data_exploration.py             ← Individual stage scripts (fixed)
├── 2_data_cleaning.py
├── 3_churn_definition.py
├── 4_feature_engineering.py
├── 5_customer_segmentation.py
├── 6_baseline_models.py
├── 7_retention_engine.py
│
├── src/                              ← Module source files
├── notebooks/                        ← Jupyter notebooks
├── checkpoints/progress.json         ← Pipeline progress tracking
├── requirements.txt
└── config.yaml
```

---

## Final Results

| Metric | Value |
|--------|-------|
| Total customers analyzed | 16,737 |
| Combined churn rate | 16.3% (2,728 customers) |
| Hard churn (formal cancellation) | 3.9% (645) |
| Activity churn (silent, no 2018 flights) | 12.7% (2,123) |
| High-risk customers | ~1,862 |
| Revenue at risk (CAD) | ~$29.4M |
| Potential recoverable revenue | ~$7.9M |
| Best model AUC | 0.874 (Random Forest) |
| Production model AUC | 0.871 (XGBoost) |
| Production model Recall | 67.2% |
| 5-fold CV AUC (XGBoost) | 0.877 |

### 4 Customer Segments

| Segment | Count | Churn Rate | Avg CLV | Priority |
|---------|-------|-----------|---------|----------|
| **Premium Dormant** | ~3,447 | **36.8%** | $9,134 | CRITICAL |
| **Seasonal Travelers** | ~2,843 | 35.4% | $6,742 | HIGH |
| Active Champions | ~5,653 | 4.6% | $8,976 | Monitor |
| Miles Hoarders | ~4,794 | 4.0% | $6,742 | Nurture |

### 3 Models Compared

| Model | AUC | Recall | Precision | F1 |
|-------|-----|--------|-----------|-----|
| Logistic Regression | 0.8435 | 0.7729 | 0.4058 | 0.5322 |
| **Random Forest** | **0.8740** | 0.5018 | 0.9073 | 0.6462 |
| XGBoost ★ (production) | 0.8711 | 0.6722 | 0.5288 | 0.5919 |

---

## Pipeline Stages

| Step | What Happens | Key Output |
|------|-------------|-----------|
| 1 | Load CSVs, fix salary, remove 3,871 duplicates, parse Calendar | Clean dataframes |
| 2 | Define churn: Hard (3.9%) + Activity (12.7%) = Combined (16.3%) | `churn_labels.csv` |
| 3 | Engineer 27 features — RFM, activity, engagement, geography, cohort, demographics | `customer_features.csv` |
| 4 | K-Means clustering (K=4 by silhouette score), name 4 segments | `segment_name` |
| 5 | Train LR + RF + XGBoost, SMOTE, 5-fold CV | `best_model.pkl` |
| 6 | XGBoost feature importance ranking | `feature_importance.csv` |
| 7 | Score all customers, risk level, per-customer retention action | `retention_actions.csv` |
| 8 | Generate 9 visualisation charts | PNG files in `outputs/figures/` |
| 9 | Write summary JSON + progress checkpoint | `pipeline_summary.json` |

---

## Dashboard Pages

| Page | Audience | Content |
|------|----------|---------|
| 🏠 Executive Overview | CEO / CMO | KPIs, revenue at risk, top 15 priority customers, churn definition table |
| 📊 Churn Intelligence | Analytics team | Model cards, CV scores, feature importance, CLV vs risk scatter |
| 👥 Segment Analysis | Marketing | Behavioral heatmap, risk-value matrix, segment drill-down |
| 🗺️ Geography & Demographics | Regional / HR | Province churn map, enrollment cohort analysis, education/gender/marital breakdown |
| 🎯 Retention Actions | Operations | Filtered action table, download CSV for CRM, segment playbooks |

---

## Key Concepts

| Term | What It Means |
|------|--------------|
| AUC-ROC | Probability the model ranks a real churner above a non-churner (0.5=random, 1.0=perfect) |
| Recall | % of actual churners the model catches — maximize this for churn prevention |
| Precision | % of churn alerts that are genuine churners |
| SMOTE | Creates synthetic churn examples to balance the 84%/16% class split |
| Data Leakage | Using future data in features — prevented by strict 2017 feature cutoff |
| Revenue at Risk | `CLV × churn_probability × 0.82` |
| Potential Save | `Revenue at Risk × 0.27` (27% industry retention lift benchmark) |
| Silhouette Score | Measures how well each point fits its cluster vs. the nearest other cluster |

---

## Submission Checklist

- [x] **Working Prototype** → `dashboard.py` (5-page Streamlit at `http://localhost:8501`)
- [x] **Technical Report** → `outputs/reports/technical_report.txt`
- [x] **Churn Prediction** → `outputs/models/best_model.pkl` (XGBoost AUC 0.871)
- [x] **Customer Segmentation** → 4 named behavioral segments
- [x] **Smart Retention** → `outputs/reports/retention_actions.csv` (who, what, when, channel)
- [x] **Data Leakage Prevention** → Features = 2017 only, Labels = 2018 only
- [x] **Geographic Analysis** → Province-level churn rates + revenue at risk
- [x] **Cohort Analysis** → Enrollment year 2012–2018 churn patterns
- [x] **Demographic Analysis** → Education, gender, marital status
- [x] **Cross-Validation** → 5-fold CV AUC alongside test AUC
- [x] **9 Visualizations** → `outputs/figures/`

---

*Python 3.12 · Pandas · NumPy · Scikit-learn · XGBoost · SMOTE · Streamlit · Plotly*
*16,737 Canadian airline loyalty members · 2017–2018 activity data*
