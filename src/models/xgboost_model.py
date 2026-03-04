"""
═══════════════════════════════════════════════════════════════
NetSentinel — XGBoost Classifier
High-performance gradient boosting model

HOW IT WORKS:
1. Trains trees SEQUENTIALLY (not in parallel like Random Forest)
2. Each new tree focuses on the ERRORS of previous trees
3. Uses gradient descent to minimize prediction error
4. Combines all trees for final prediction

WHY FOR NETWORK ANOMALY DETECTION:
- State-of-the-art performance on tabular data
- Handles class imbalance well (scale_pos_weight)
- Built-in regularization prevents overfitting
- Feature importance analysis
- Very fast with GPU support
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from xgboost import XGBClassifier
from src.models.base_model import BaseAnomalyDetector


class XGBoostDetector(BaseAnomalyDetector):

    def __init__(self, n_estimators=300, max_depth=10,
                 learning_rate=0.1, subsample=0.8,
                 colsample_bytree=0.8, random_state=42):
        super().__init__("XGBoost")

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.feature_importances = None

    def train(self, X_train, y_train):
        """
        Train XGBoost.

        XGBoost trains trees sequentially:
        Tree 1 → makes errors → Tree 2 focuses on those errors →
        makes fewer errors → Tree 3 focuses on remaining errors → ...

        This "boosting" approach typically outperforms Random Forest
        on structured/tabular data.
        """
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"  Estimators:     {self.model.n_estimators}")
        print(f"  Max depth:      {self.model.max_depth}")
        print(f"  Learning rate:  {self.model.learning_rate}")
        print(f"  Subsample:      {self.model.subsample}")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Features:         {X_train.shape[1]}")

        start = time.time()
        self.model.fit(X_train, y_train, verbose=False)
        self.training_time = time.time() - start

        # Extract feature importances
        if hasattr(X_train, 'columns'):
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)

        print(f"  ✅ Training complete ({self.training_time:.2f}s)")

    def predict(self, X):
        """Predict binary labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get probability of being an attack."""
        return self.model.predict_proba(X)[:, 1]

    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot top feature importances."""
        if self.feature_importances is None:
            print("  ⚠️ Train model first.")
            return

        top_features = self.feature_importances.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features.plot(kind='barh', ax=ax, color='#E8683F')
        ax.set_title(f'{self.model_name} — Top {top_n} Feature Importances',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.invert_yaxis()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def log_to_mlflow(self, experiment_name="network_anomaly_detection"):
        """Log XGBoost specific parameters."""
        import mlflow

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=self.model_name):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("model_type", "supervised_boosting")
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("max_depth", self.model.max_depth)
            mlflow.log_param("learning_rate", self.model.learning_rate)

            for metric, value in self.results.items():
                mlflow.log_metric(metric, value)

            mlflow.sklearn.log_model(self.model, "model")
            print(f"  ✅ Logged to MLflow: {self.model_name}")