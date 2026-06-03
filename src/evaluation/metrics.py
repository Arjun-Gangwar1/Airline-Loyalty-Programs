"""Model evaluation utilities — translates ML metrics into business language."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)


class ModelEvaluator:
    """Evaluate and compare classification models with business-translated metrics."""

    def __init__(self, output_dir: str = "outputs/figures/models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def evaluate(self, name: str, model, X_test, y_test) -> dict:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] \
                 if hasattr(model, 'predict_proba') else y_pred.astype(float)

        metrics = {
            'model':     name,
            'accuracy':  round(accuracy_score(y_test, y_pred),  4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall':    round(recall_score(y_test, y_pred,    zero_division=0), 4),
            'f1':        round(f1_score(y_test, y_pred,         zero_division=0), 4),
            'roc_auc':   round(roc_auc_score(y_test, y_prob),   4),
        }
        metrics['business_summary'] = self._business_translate(metrics)
        self.results.append(metrics)
        return metrics

    def _business_translate(self, m: dict) -> str:
        return (
            f"Catches {m['recall']*100:.0f}% of churners | "
            f"{m['precision']*100:.0f}% of churn alerts are correct | "
            f"AUC {m['roc_auc']:.2f}"
        )

    def compare(self) -> pd.DataFrame:
        return pd.DataFrame(self.results).sort_values('roc_auc', ascending=False)

    def plot_roc_curves(self, models: dict, X_test, y_test):
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — Churn Prediction Models')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, name: str, y_test, y_pred):
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Retained', 'Churned'],
                    yticklabels=['Retained', 'Churned'], ax=ax)
        ax.set_title(f'Confusion Matrix — {name}')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_{name.lower().replace(" ", "_")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, model, feature_names: list, top_n: int = 20):
        if not hasattr(model, 'feature_importances_'):
            return
        imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
        fig, ax = plt.subplots(figsize=(8, 6))
        imp.sort_values().plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        return imp


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """Return sklearn classification_report as a DataFrame."""
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()
