"""
STAGE 6: BASELINE CHURN PREDICTION MODELS
==========================================

Build interpretable baseline models FIRST.
Understand what drives churn before going advanced.

CHECKPOINT: After this stage, you'll have:
✅ Logistic Regression baseline
✅ Random Forest model
✅ XGBoost model
✅ Model comparison table
✅ Feature importance analysis
✅ Business-translated metrics

Expected Time: 30-45 minutes
Expected Outputs:
- baseline_models/ (saved .pkl models)
- model_comparison.csv
- feature_importance.png
- roc_curves.png

MODEL PROGRESSION:
1. Logistic Regression → interpretable baseline (AUC target: 0.70-0.75)
2. Random Forest       → handles non-linearity (AUC target: 0.75-0.80)
3. XGBoost             → industry standard (AUC target: 0.80-0.85+)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class BaselineModelTrainer:
    """
    Train and evaluate baseline churn prediction models.
    Translates all metrics into business language.
    """

    def __init__(self):
        self.final_path  = Path("data/final")
        self.models_path = Path("outputs/models")
        self.fig_path    = Path("outputs/figures/models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.fig_path.mkdir(parents=True, exist_ok=True)

        self.models       = {}
        self.results      = []
        self.X_train = self.X_test = self.y_train = self.y_test = None

    # ─────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────

    def load_data(self):
        print("\n" + "=" * 60)
        print("STAGE 6: BASELINE CHURN PREDICTION MODELS")
        print("=" * 60)
        print("\n� Loading feature matrix from Stage 4/5...")

        try:
            # Prefer segmented version (has segment info)
            fp = self.final_path / 'customer_features_segmented.csv'
            if not fp.exists():
                fp = self.final_path / 'customer_features.csv'

            self.df = pd.read_csv(fp)
            print(f"   ✓ Loaded: {self.df.shape}")

            # Accept 'churn' or 'churned' column
            if 'churn' not in self.df.columns and 'churned' in self.df.columns:
                self.df = self.df.rename(columns={'churned': 'churn'})
            if 'churn' not in self.df.columns:
                print("❌ 'churn' column missing. Run Stage 3 first.")
                return False

            churn_rate = self.df['churn'].mean() * 100
            print(f"   ✓ Overall churn rate: {churn_rate:.2f}%")
            print(f"   ✓ Churned:   {self.df['churn'].sum():,}")
            print(f"   ✓ Retained:  {(self.df['churn']==0).sum():,}")

            # Validate churn rate is reasonable before training
            n_classes = self.df['churn'].nunique()
            if n_classes < 2:
                print(f"\n❌ CRITICAL: churn column has only {n_classes} class(es).")
                print("   This means Stage 3 produced an extreme churn definition.")
                print("   Fix: re-run Stage 3 (3_churn_definition.py) — it will")
                print("   auto-detect a better prediction date from your data.")
                return False
            if churn_rate > 90:
                print(f"\n⚠️  WARNING: churn rate {churn_rate:.1f}% is very high.")
                print("   Models may be unreliable. Consider re-running Stage 3.")
            if churn_rate < 3:
                print(f"\n⚠️  WARNING: churn rate {churn_rate:.1f}% is very low.")
                print("   Models may struggle. Consider re-running Stage 3.")
            return True

        except FileNotFoundError as e:
            print(f"❌ {e}\n   Run Stage 4 first.")
            return False

    # ─────────────────────────────────────────────
    # 2. PREPARE FEATURES
    # ─────────────────────────────────────────────

    def prepare_features(self):
        print("\n� Preparing feature matrix...")

        # Drop non-feature columns
        drop_cols = ['churn', 'segment_id', 'segment_name']
        id_like = [c for c in self.df.columns
                   if 'loyalty' in c.lower() or 'customer' in c.lower() or 'id' == c.lower()]
        drop_cols += id_like

        self.feature_cols = [c for c in self.df.columns
                             if c not in drop_cols
                             and self.df[c].dtype != object]

        X = self.df[self.feature_cols].fillna(0)
        y = self.df['churn']

        print(f"   ✓ Features selected: {len(self.feature_cols)}")
        print(f"   Feature list (first 10): {self.feature_cols[:10]}")

        # ── Temporal train/test split ──────────────
        # Use 80% for training, 20% for testing
        # Stratify to preserve class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        print(f"\n   � Train/Test Split:")
        print(f"      Train: {len(self.X_train):,}  |  "
              f"Churn rate: {self.y_train.mean()*100:.2f}%")
        print(f"      Test:  {len(self.X_test):,}   |  "
              f"Churn rate: {self.y_test.mean()*100:.2f}%")

        # ── Handle class imbalance with SMOTE ──────
        print("\n⚖️  Handling class imbalance with SMOTE...")
        smote = SMOTE(random_state=42)
        self.X_train_bal, self.y_train_bal = smote.fit_resample(
            self.X_train, self.y_train
        )
        print(f"   ✓ After SMOTE → Train: {len(self.X_train_bal):,}  "
              f"|  Churn rate: {self.y_train_bal.mean()*100:.2f}%")

        # ── Scale features ─────────────────────────
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_bal)
        self.X_test_scaled  = self.scaler.transform(self.X_test)
        joblib.dump(self.scaler, self.models_path / 'feature_scaler.pkl')
        print("   ✓ Features scaled and scaler saved")

    # ─────────────────────────────────────────────
    # 3. EVALUATION HELPER
    # ─────────────────────────────────────────────

    def evaluate_model(self, model, model_name, X_test, y_test, X_train=None, y_train=None):
        """Compute metrics and translate to business language."""

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'Model':     model_name,
            'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
            'ROC_AUC':   round(roc_auc_score(y_test, y_proba), 4),
        }

        # Cross-validation AUC on training set
        if X_train is not None and y_train is not None:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                scoring='roc_auc'
            )
            metrics['CV_AUC_mean'] = round(cv_scores.mean(), 4)
            metrics['CV_AUC_std']  = round(cv_scores.std(), 4)

        print(f"\n   � {model_name} Results:")
        print(f"      ROC-AUC  : {metrics['ROC_AUC']:.4f}  ← Overall discrimination ability")
        print(f"      Recall   : {metrics['Recall']:.4f}  ← % of churners we catch")
        print(f"      Precision: {metrics['Precision']:.4f}  ← % of alerts that are correct")
        print(f"      F1-Score : {metrics['F1']:.4f}")

        # Business interpretation
        caught = int(metrics['Recall'] * y_test.sum())
        missed = int(y_test.sum() - caught)
        false_alarms = int(y_pred.sum() - caught) if y_pred.sum() > caught else 0

        print(f"\n   � Business Translation:")
        print(f"      We catch {caught:,} of {y_test.sum():,} actual churners ({metrics['Recall']*100:.1f}%)")
        print(f"      We miss  {missed:,} churners  (false negatives)")
        print(f"      False alerts: ~{false_alarms:,} non-churners flagged as at-risk")

        if 'CV_AUC_mean' in metrics:
            print(f"      Cross-Val AUC: {metrics['CV_AUC_mean']:.4f} ± {metrics['CV_AUC_std']:.4f}")

        return metrics, y_proba

    # ─────────────────────────────────────────────
    # 4. MODEL 1: LOGISTIC REGRESSION
    # ─────────────────────────────────────────────

    def train_logistic_regression(self):
        print("\n" + "=" * 60)
        print("MODEL 1: LOGISTIC REGRESSION (Baseline)")
        print("=" * 60)
        print("   Why: Interpretable, fast, good baseline")

        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        lr.fit(self.X_train_scaled, self.y_train_bal)

        metrics, y_proba = self.evaluate_model(
            lr, 'Logistic Regression',
            self.X_test_scaled, self.y_test,
            self.X_train_scaled, self.y_train_bal
        )

        self.models['Logistic Regression'] = {
            'model': lr, 'proba': y_proba, 'metrics': metrics
        }
        self.results.append(metrics)

        joblib.dump(lr, self.models_path / 'logistic_regression.pkl')
        print("\n   ✓ Model saved: logistic_regression.pkl")

    # ─────────────────────────────────────────────
    # 5. MODEL 2: RANDOM FOREST
    # ─────────────────────────────────────────────

    def train_random_forest(self):
        print("\n" + "=" * 60)
        print("MODEL 2: RANDOM FOREST")
        print("=" * 60)
        print("   Why: Handles non-linearity, built-in feature importance")

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train_bal, self.y_train_bal)

        metrics, y_proba = self.evaluate_model(
            rf, 'Random Forest',
            self.X_test, self.y_test,
            self.X_train_bal, self.y_train_bal
        )

        self.models['Random Forest'] = {
            'model': rf, 'proba': y_proba, 'metrics': metrics
        }
        self.results.append(metrics)

        joblib.dump(rf, self.models_path / 'random_forest.pkl')
        print("   ✓ Model saved: random_forest.pkl")

        # Feature importance
        self.rf_model = rf
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        self.feature_importance = importance_df
        print(f"\n   Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

    # ─────────────────────────────────────────────
    # 6. MODEL 3: XGBOOST
    # ─────────────────────────────────────────────

    def train_xgboost(self):
        print("\n" + "=" * 60)
        print("MODEL 3: XGBOOST (Industry Standard)")
        print("=" * 60)

        if not XGB_AVAILABLE:
            print("   ⚠️  XGBoost not available. Skipping.")
            return

        print("   Why: Best performance on tabular data, handles imbalance natively")

        # Auto-detect GPU
        import subprocess as _sp
        try:
            _sp.check_output(['nvidia-smi'], stderr=_sp.DEVNULL)
            device    = 'cuda'
            tree_method = 'hist'
            print("   🚀 GPU detected — training on CUDA")
        except Exception:
            device    = 'cpu'
            tree_method = 'hist'
            print("   💻 No GPU — training on CPU")

        # Scale pos weight to handle imbalance
        neg = (self.y_train == 0).sum()
        pos = (self.y_train == 1).sum()
        scale_pos = neg / max(pos, 1)

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            device=device,
            tree_method=tree_method,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )

        metrics, y_proba = self.evaluate_model(
            xgb_model, 'XGBoost',
            self.X_test, self.y_test,
            self.X_train, self.y_train
        )

        self.models['XGBoost'] = {
            'model': xgb_model, 'proba': y_proba, 'metrics': metrics
        }
        self.results.append(metrics)

        joblib.dump(xgb_model, self.models_path / 'xgboost.pkl')
        print("   ✓ Model saved: xgboost.pkl")

        # XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.xgb_importance = xgb_importance
        print(f"\n   Top 10 XGBoost Features:")
        print(xgb_importance.head(10).to_string(index=False))

    # ─────────────────────────────────────────────
    # 7. COMPARE MODELS
    # ─────────────────────────────────────────────

    def compare_models(self):
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)

        comparison_df = pd.DataFrame(self.results)
        comparison_df = comparison_df.sort_values('ROC_AUC', ascending=False)
        print("\n", comparison_df.to_string(index=False))

        # Save comparison
        comparison_df.to_csv(
            self.final_path / 'model_comparison.csv', index=False
        )
        print(f"\n✓ Saved: model_comparison.csv")

        # Identify best model
        best_row = comparison_df.iloc[0]
        self.best_model_name = best_row['Model']
        print(f"\n� Best Model: {self.best_model_name}")
        print(f"   AUC: {best_row['ROC_AUC']:.4f}")

        return comparison_df

    # ─────────────────────────────────────────────
    # 8. VISUALIZATIONS
    # ─────────────────────────────────────────────

    def visualize_results(self):
        print("\n� Generating model visualizations...")

        # ── ROC Curves ────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['steelblue', 'coral', 'green', 'purple']

        for i, (name, data) in enumerate(self.models.items()):
            fpr, tpr, _ = roc_curve(self.y_test, data['proba'])
            auc = data['metrics']['ROC_AUC']
            ax.plot(fpr, tpr, linewidth=2.5,
                    color=colors[i % len(colors)],
                    label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves — Churn Prediction Models',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        plt.tight_layout()
        plt.savefig(self.fig_path / 'roc_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: roc_curves.png")

        # ── Feature Importance ─────────────────────
        if hasattr(self, 'feature_importance'):
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # Random Forest importance
            top20 = self.feature_importance.head(20)
            axes[0].barh(top20['feature'][::-1],
                         top20['importance'][::-1],
                         color='steelblue', alpha=0.8)
            axes[0].set_title('Random Forest — Top 20 Features',
                               fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Feature Importance')
            axes[0].grid(True, alpha=0.3, axis='x')

            # XGBoost importance (if available)
            if hasattr(self, 'xgb_importance'):
                top20_xgb = self.xgb_importance.head(20)
                axes[1].barh(top20_xgb['feature'][::-1],
                             top20_xgb['importance'][::-1],
                             color='coral', alpha=0.8)
                axes[1].set_title('XGBoost — Top 20 Features',
                                   fontsize=12, fontweight='bold')
                axes[1].set_xlabel('Feature Importance')
                axes[1].grid(True, alpha=0.3, axis='x')
            else:
                axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(self.fig_path / 'feature_importance.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: feature_importance.png")

        # ── Confusion Matrix Grid ──────────────────
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, self.models.items()):
            y_pred = (data['proba'] >= 0.5).astype(int)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Retained', 'Churned'],
                        yticklabels=['Retained', 'Churned'])
            ax.set_title(f'{name}\nConfusion Matrix', fontsize=11, fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')

        plt.tight_layout()
        plt.savefig(self.fig_path / 'confusion_matrices.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: confusion_matrices.png")

        # ── Metric Comparison Bar Chart ────────────
        metrics_df = pd.DataFrame(self.results)[
            ['Model', 'ROC_AUC', 'Recall', 'Precision', 'F1']
        ]
        metrics_df = metrics_df.set_index('Model')

        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.plot(kind='bar', ax=ax, alpha=0.85,
                        color=['steelblue', 'coral', 'green', 'purple'])
        ax.set_title('Model Performance Comparison',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4, label='Target AUC=0.8')
        plt.tight_layout()
        plt.savefig(self.fig_path / 'model_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: model_comparison.png")

    # ─────────────────────────────────────────────
    # 9. SAVE BEST MODEL + PREDICTIONS
    # ─────────────────────────────────────────────

    def save_best_model_predictions(self):
        """Save best model and generate predictions for all customers."""

        print("\n� Saving best model predictions...")

        best_data = self.models[self.best_model_name]
        best_model = best_data['model']

        # Save as "best_model.pkl"
        joblib.dump(best_model, self.models_path / 'best_model.pkl')
        print(f"   ✓ Best model saved: best_model.pkl ({self.best_model_name})")

        # Generate predictions on TEST set
        predictions_df = self.X_test.copy()
        predictions_df['actual_churn'] = self.y_test.values
        predictions_df['churn_probability'] = best_data['proba']
        predictions_df['predicted_churn'] = (best_data['proba'] >= 0.5).astype(int)
        predictions_df['risk_level'] = pd.cut(
            best_data['proba'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )

        pred_file = self.final_path / 'churn_predictions.csv'
        predictions_df.to_csv(pred_file, index=False)
        print(f"   ✓ Predictions saved: {pred_file}")

        # Generate predictions for ALL customers
        fp = self.final_path / 'customer_features_segmented.csv'
        if not fp.exists():
            fp = self.final_path / 'customer_features.csv'

        all_customers = pd.read_csv(fp)
        id_col = all_customers.columns[0]
        drop_cols = ['churn', 'segment_id', 'segment_name', id_col]
        feat_cols = [c for c in all_customers.columns
                     if c not in drop_cols and all_customers[c].dtype != object]

        X_all = all_customers[feat_cols].fillna(0)

        # Align columns with training features
        for col in self.feature_cols:
            if col not in X_all.columns:
                X_all[col] = 0
        X_all = X_all[self.feature_cols]

        # Predict
        if self.best_model_name == 'Logistic Regression':
            X_all_input = self.scaler.transform(X_all)
        else:
            X_all_input = X_all

        all_proba = best_model.predict_proba(X_all_input)[:, 1]

        all_customers['churn_probability'] = all_proba
        all_customers['risk_level'] = pd.cut(
            all_proba,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )

        all_pred_file = self.final_path / 'all_customer_predictions.csv'
        all_customers.to_csv(all_pred_file, index=False)
        print(f"   ✓ All-customer predictions saved: {all_pred_file}")

        # Risk distribution summary
        risk_dist = all_customers['risk_level'].value_counts()
        print(f"\n   � Risk Distribution (All Customers):")
        for level, count in risk_dist.items():
            pct = count / len(all_customers) * 100
            print(f"      {level:<15}: {count:,} ({pct:.1f}%)")

        # Update progress
        summary = {
            'stage': 6,
            'completed_at': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'best_auc': float(self.models[self.best_model_name]['metrics']['ROC_AUC']),
            'model_comparison': self.results
        }
        with open(self.final_path / 'modeling_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        progress_file = Path("checkpoints/progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            progress['current_stage'] = 6
            if 6 not in progress['completed_stages']:
                progress['completed_stages'].append(6)
            progress['last_updated'] = datetime.now().isoformat()
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            print("   ✓ Progress updated")

    # ─────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────

    def run_baseline_modeling(self):
        if not self.load_data():
            return False

        self.prepare_features()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.compare_models()
        self.visualize_results()
        self.save_best_model_predictions()

        # Final summary
        best = self.models[self.best_model_name]['metrics']
        print("\n" + "=" * 60)
        print("✅ STAGE 6 COMPLETE — BASELINE MODELS")
        print("=" * 60)
        print(f"\n� Best Model : {self.best_model_name}")
        print(f"   ROC-AUC   : {best['ROC_AUC']:.4f}")
        print(f"   Recall    : {best['Recall']:.4f}  (catching {best['Recall']*100:.1f}% of churners)")
        print(f"   Precision : {best['Precision']:.4f}")
        print(f"\n� Models saved: outputs/models/")
        print(f"   - best_model.pkl")
        print(f"   - logistic_regression.pkl")
        print(f"   - random_forest.pkl")
        print(f"   - xgboost.pkl (if installed)")
        print(f"\n� Visualizations: outputs/figures/models/")
        print(f"\n� Next Step: Run 07_retention_engine.py")
        return True


def main():
    trainer = BaselineModelTrainer()
    success = trainer.run_baseline_modeling()
    if not success:
        print("\n❌ Stage 6 failed. Check errors above.")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())