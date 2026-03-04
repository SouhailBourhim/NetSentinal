"""
═══════════════════════════════════════════════════════════════
NetSentinel — Random Forest Classifier
Supervised classification model

HOW IT WORKS:
1. Creates many decision trees (a "forest")
2. Each tree is trained on a random subset of data + features
3. Each tree votes on the prediction
4. Final prediction = majority vote

WHY FOR NETWORK ANOMALY DETECTION:
- Handles high-dimensional data well
- Provides feature importance ranking
- Robust to noise and outliers
- Fast training and inference
- Strong baseline for supervised detection
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseAnomalyDetector


class RandomForestDetector(BaseAnomalyDetector):

    def __init__(self, n_estimators=200, max_depth=20,
                 min_samples_split=5, min_samples_leaf=2,
                 random_state=42):
        super().__init__("Random Forest")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_importances = None

    def train(self, X_train, y_train):
        """
        Train Random Forest.

        This is SUPERVISED — it uses labeled data (y_train)
        to learn the mapping: network features → benign/attack.
        """
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"  Estimators:       {self.model.n_estimators}")
        print(f"  Max depth:        {self.model.max_depth}")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Features:         {X_train.shape[1]}")

        start = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start

        # Extract feature importances
        if hasattr(X_train, 'columns'):
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)

        print(f"  ✅ Training complete ({self.training_time:.2f}s)")

    def predict(self, X):
        """Predict binary labels (0=benign, 1=attack)."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get probability of being an attack.
        Returns P(attack) for each sample.
        """
        return self.model.predict_proba(X)[:, 1]

    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot the most important features.

        Feature importance in Random Forest =
        how much each feature reduces impurity (Gini)
        across all trees.

        This tells us WHICH network characteristics
        are most useful for detecting attacks.
        """
        if self.feature_importances is None:
            print("  ⚠️ No feature importances available. Train first.")
            return

        top_features = self.feature_importances.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features.plot(kind='barh', ax=ax, color='#0F3057')
        ax.set_title(f'{self.model_name} — Top {top_n} Feature Importances',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance (Gini)', fontsize=12)
        ax.invert_yaxis()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n  Top 10 Most Important Features:")
        for i, (feat, imp) in enumerate(top_features.head(10).items()):
            print(f"    {i+1:2d}. {feat:<35s} {imp:.4f}")

    def log_to_mlflow(self, experiment_name="network_anomaly_detection"):
        """Log Random Forest specific parameters."""
        import mlflow

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=self.model_name):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("model_type", "supervised")
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("max_depth", self.model.max_depth)

            for metric, value in self.results.items():
                mlflow.log_metric(metric, value)

            if self.feature_importances is not None:
                for feat, imp in self.feature_importances.head(10).items():
                    mlflow.log_metric(f"importance_{feat}", imp)

            mlflow.sklearn.log_model(self.model, "model")
            print(f"  ✅ Logged to MLflow: {self.model_name}")