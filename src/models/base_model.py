"""
═══════════════════════════════════════════════════════════════
NetSentinel — Base Model Class
Abstract base class for all anomaly detection models
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import mlflow
import mlflow.sklearn
import time
from abc import ABC, abstractmethod


class BaseAnomalyDetector(ABC):
    """
    Abstract base class that all NetSentinel models inherit from.
    Provides common evaluation, logging, and visualization methods.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.training_time = 0
        self.results = {}

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model. Must be implemented by each subclass."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions. Must be implemented by each subclass."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Generate probability scores. Must be implemented by each subclass."""
        pass

    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation.
        Computes all metrics and generates visualizations.
        """
        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {self.model_name}")
        print(f"{'=' * 60}")

        # Predictions
        y_pred = self.predict(X_test)
        y_scores = self.predict_proba(X_test)

        # Core metrics
        self.results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_scores) if y_scores is not None else 0,
            'training_time': self.training_time
        }

        # Print results
        print(f"\n  {'Metric':<20} {'Score':>10}")
        print(f"  {'─' * 30}")
        for metric, value in self.results.items():
            if metric == 'training_time':
                print(f"  {metric:<20} {value:>10.2f}s")
            else:
                print(f"  {metric:<20} {value:>10.4f}")

        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Benign', 'Attack'], digits=4))

        return self.results

    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """
        Generate and display confusion matrix.

        The confusion matrix shows:
        - True Positives (TP): Correctly identified attacks
        - True Negatives (TN): Correctly identified benign traffic
        - False Positives (FP): Benign traffic flagged as attack (false alarm)
        - False Negatives (FN): Attacks missed (dangerous!)
        """
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt=',d',
            cmap='Blues',
            xticklabels=['Benign', 'Attack'],
            yticklabels=['Benign', 'Attack'],
            ax=ax
        )
        ax.set_title(f'{self.model_name} — Confusion Matrix',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        # Print interpretation
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  Interpretation:")
        print(f"    True Negatives  (correctly benign):  {tn:,}")
        print(f"    True Positives  (correctly attack):  {tp:,}")
        print(f"    False Positives (false alarms):      {fp:,}")
        print(f"    False Negatives (missed attacks):    {fn:,}")
        print(f"\n    False Alarm Rate: {fp/(fp+tn)*100:.2f}%")
        print(f"    Miss Rate:        {fn/(fn+tp)*100:.2f}%")

    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """
        Generate ROC curve.

        ROC curve shows the trade-off between:
        - True Positive Rate (catching attacks)
        - False Positive Rate (false alarms)

        AUC = Area Under Curve
        - AUC = 1.0: perfect model
        - AUC = 0.5: random guessing
        """
        y_scores = self.predict_proba(X_test)
        if y_scores is None:
            print("  ⚠️ Model doesn't support probability scores. Skipping ROC.")
            return

        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        auc = roc_auc_score(y_test, y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#0F3057', lw=2,
                label=f'{self.model_name} (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{self.model_name} — ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_precision_recall(self, X_test, y_test, save_path=None):
        """
        Generate Precision-Recall curve.

        More informative than ROC for imbalanced datasets.
        - Precision: Of predicted attacks, how many are real?
        - Recall: Of real attacks, how many did we catch?
        """
        y_scores = self.predict_proba(X_test)
        if y_scores is None:
            print("  ⚠️ Model doesn't support probability scores. Skipping PR curve.")
            return

        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='#E8683F', lw=2, label=self.model_name)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{self.model_name} — Precision-Recall Curve',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def log_to_mlflow(self, experiment_name: str = "network_anomaly_detection"):
        """
        Log model, parameters, and metrics to MLflow.
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=self.model_name):
            # Log metrics
            for metric, value in self.results.items():
                mlflow.log_metric(metric, value)

            # Log model name
            mlflow.log_param("model_name", self.model_name)

            print(f"\n  ✅ Logged to MLflow: {self.model_name}")