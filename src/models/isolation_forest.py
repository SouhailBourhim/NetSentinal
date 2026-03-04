"""
═══════════════════════════════════════════════════════════════
NetSentinel — Isolation Forest
Unsupervised anomaly detection model

HOW IT WORKS:
1. Randomly selects a feature
2. Randomly selects a split value between min and max
3. Repeats until each data point is isolated
4. Anomalies are isolated in FEWER splits (shorter path)
5. Normal points require MORE splits (longer path)

WHY FOR NETWORK ANOMALY DETECTION:
- No labels needed (can detect unknown attacks)
- Fast training on large datasets
- Good at finding outliers in high-dimensional data
- Attack traffic is inherently "different" from normal
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import time
from sklearn.ensemble import IsolationForest
from src.models.base_model import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):

    def __init__(self, contamination=0.1, n_estimators=200,
                 max_samples='auto', random_state=42):
        super().__init__("Isolation Forest")

        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X_train, y_train=None):
        """
        Train Isolation Forest.

        Note: Isolation Forest is UNSUPERVISED.
        It doesn't use y_train.
        It learns what "normal" looks like from the data distribution.
        Anything far from normal = anomaly.

        We pass X_train which contains both benign and attack samples.
        The model learns that attack patterns are "isolatable" —
        they differ from the majority (benign) traffic.
        """
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"  Contamination: {self.contamination}")
        print(f"  Estimators:    {self.model.n_estimators}")
        print(f"  Samples:       {X_train.shape[0]:,}")
        print(f"  Features:      {X_train.shape[1]}")

        start = time.time()
        self.model.fit(X_train)
        self.training_time = time.time() - start

        print(f"  ✅ Training complete ({self.training_time:.2f}s)")

    def predict(self, X):
        """
        Predict anomalies.

        Isolation Forest returns:
          1  = normal (inlier)
         -1  = anomaly (outlier)

        We convert to:
          0  = benign
          1  = attack

        To match our label convention.
        """
        raw_predictions = self.model.predict(X)
        # Convert: -1 (anomaly) → 1 (attack), 1 (normal) → 0 (benign)
        return np.where(raw_predictions == -1, 1, 0)

    def predict_proba(self, X):
        """
        Get anomaly scores.

        score_samples() returns the anomaly score:
        - More negative = more anomalous
        - Less negative = more normal

        We negate so that higher = more anomalous
        (to match convention for ROC curves).
        """
        scores = self.model.score_samples(X)
        return -scores  # Negate: higher = more anomalous

    def log_to_mlflow(self, experiment_name="network_anomaly_detection"):
        """Log Isolation Forest specific parameters."""
        import mlflow

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=self.model_name):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("model_type", "unsupervised")
            mlflow.log_param("contamination", self.contamination)
            mlflow.log_param("n_estimators", self.model.n_estimators)

            for metric, value in self.results.items():
                mlflow.log_metric(metric, value)

            mlflow.sklearn.log_model(self.model, "model")
            print(f"  ✅ Logged to MLflow: {self.model_name}")