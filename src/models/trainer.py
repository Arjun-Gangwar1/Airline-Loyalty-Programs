"""
Churn prediction model trainer.

Trains Logistic Regression, Random Forest, and XGBoost models,
handles class imbalance with SMOTE, and saves the best model.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from src.evaluation import ModelEvaluator

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# Columns that are not features (IDs, labels, text)
NON_FEATURE_COLS = {
    'loyalty_number', 'loyalty_card', 'churn', 'country', 'province',
    'city', 'postal_code', 'gender', 'education', 'marital_status',
    'enrollment_type', 'enrollment_year', 'enrollment_month',
    'cancellation_year', 'cancellation_month', 'last_activity_date',
    'segment', 'segment_name',
}


class ModelTrainer:
    """Train and persist churn prediction models."""

    def __init__(self):
        self.final_path  = Path("data/final")
        self.models_path = Path("outputs/models")
        self.reports_path = Path("outputs/reports")
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)

        self.models    = {}
        self.evaluator = ModelEvaluator()
        self.scaler    = StandardScaler()

    # ── Data preparation ──────────────────────────────────────────────────────

    def load_data(self):
        fp = self.final_path / 'customer_features_segmented.csv'
        if not fp.exists():
            fp = self.final_path / 'customer_features.csv'
        self.df = pd.read_csv(fp)
        print(f"Loaded: {self.df.shape}")

    def prepare_features(self, test_size: float = 0.2, random_state: int = 42):
        feature_cols = [c for c in self.df.columns
                        if c not in NON_FEATURE_COLS and
                        self.df[c].dtype in [np.float64, np.int64, float, int]]

        X = self.df[feature_cols].fillna(0)
        y = self.df['churn']

        self.feature_names = feature_cols
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # SMOTE oversampling on training set
        smote = SMOTE(random_state=random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        # Scale
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test  = self.scaler.transform(self.X_test)

        joblib.dump(self.scaler, self.models_path / 'feature_scaler.pkl')
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}, "
              f"Churn rate: {y.mean()*100:.1f}%")

    # ── Model training ────────────────────────────────────────────────────────

    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        metrics = self.evaluator.evaluate('Logistic Regression', model, self.X_test, self.y_test)
        joblib.dump(model, self.models_path / 'logistic_regression.pkl')
        print(f"LR — {metrics['business_summary']}")
        return model

    def train_random_forest(self):
        model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                       class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        metrics = self.evaluator.evaluate('Random Forest', model, self.X_test, self.y_test)
        joblib.dump(model, self.models_path / 'random_forest.pkl')
        print(f"RF  — {metrics['business_summary']}")
        return model

    def train_xgboost(self):
        if not XGB_AVAILABLE:
            print("XGBoost not installed. Skipping.")
            return None
        scale_pos = (self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   scale_pos_weight=scale_pos, random_state=42,
                                   eval_metric='auc', verbosity=0)
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        metrics = self.evaluator.evaluate('XGBoost', model, self.X_test, self.y_test)
        joblib.dump(model, self.models_path / 'xgboost.pkl')
        print(f"XGB — {metrics['business_summary']}")
        return model

    # ── Best model & saving ──────────────────────────────────────────────────

    def select_best_model(self):
        comparison = self.evaluator.compare()
        best_name  = comparison.iloc[0]['model']
        best_model = self.models[best_name]
        joblib.dump(best_model, self.models_path / 'best_model.pkl')
        print(f"\nBest model: {best_name} (AUC {comparison.iloc[0]['roc_auc']:.4f})")
        return best_model, best_name

    def save_predictions(self):
        best_model, best_name = self.select_best_model()
        X_full   = self.scaler.transform(
            self.df[[c for c in self.feature_names]].fillna(0)
        )
        probs    = best_model.predict_proba(X_full)[:, 1]
        preds    = best_model.predict(X_full)

        out = self.df[[c for c in self.df.columns
                       if c in ['loyalty_number', 'churn', 'clv', 'segment', 'segment_name']]].copy()
        out['churn_probability'] = probs.round(4)
        out['predicted_churn']   = preds
        out['risk_level'] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                                   labels=['low', 'medium', 'high'])

        out.to_csv(self.final_path / 'all_customer_predictions.csv', index=False)
        at_risk = out[out['predicted_churn'] == 1]
        at_risk.to_csv(self.final_path / 'churn_predictions.csv', index=False)
        self.evaluator.compare().to_csv(self.reports_path / 'model_comparison.csv', index=False)
        print(f"Saved predictions — {preds.sum():,} at-risk customers identified.")

    # ── Visualisations ────────────────────────────────────────────────────────

    def plot_all(self):
        self.evaluator.plot_roc_curves(self.models, self.X_test, self.y_test)
        best_model = self.models.get('XGBoost') or self.models.get('Random Forest')
        if best_model:
            self.evaluator.plot_feature_importance(best_model, self.feature_names)
            y_pred = best_model.predict(self.X_test)
            self.evaluator.plot_confusion_matrix('Best Model', self.y_test, y_pred)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self):
        self.load_data()
        self.prepare_features()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.plot_all()
        self.save_predictions()
        return self.evaluator.compare()
