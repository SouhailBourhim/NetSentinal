"""
═══════════════════════════════════════════════════════════════
NetSentinel — Prediction Service
Loads trained models and serves predictions
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path


class NetSentinelPredictor:
    """
    Loads the trained XGBoost + Isolation Forest models
    and serves predictions on new network flow data.

    Supports:
    - XGBoost-only prediction
    - Isolation Forest-only anomaly scoring
    - Hybrid (weighted combination)
    """

    def __init__(self, models_path: str = "saved_models"):
        self.models_path = Path(models_path)
        self.xgb_model = None
        self.iso_model = None
        self.scaler = None
        self.feature_names = None
        self.hybrid_weight_xgb = 0.7
        self.hybrid_weight_iso = 0.3

        self._load_models()

    def _load_models(self):
        """Load all saved models and artifacts."""
        print("Loading NetSentinel models...")

        # Load XGBoost
        xgb_path = self.models_path / "xgboost_tuned.pkl"
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)
            print(f"  ✅ XGBoost loaded from {xgb_path}")
        else:
            print(f"  ⚠️ XGBoost not found at {xgb_path}")

        # Load Isolation Forest
        iso_path = self.models_path / "isolation_forest.pkl"
        if iso_path.exists():
            self.iso_model = joblib.load(iso_path)
            print(f"  ✅ Isolation Forest loaded from {iso_path}")
        else:
            print(f"  ⚠️ Isolation Forest not found at {iso_path}")

        # Load Scaler
        scaler_path = self.models_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"  ✅ Scaler loaded from {scaler_path}")
        else:
            print(f"  ⚠️ Scaler not found at {scaler_path}")

        # Load Feature Names
        features_path = self.models_path / "feature_names.json"
        if features_path.exists():
            with open(features_path, "r") as f:
                self.feature_names = json.load(f)
            print(f"  ✅ Feature names loaded ({len(self.feature_names)} features)")
        else:
            print(f"  ⚠️ Feature names not found at {features_path}")

        print("  ✅ All models loaded successfully")

    def _validate_input(self, features: dict) -> np.ndarray:
        """
        Validate and prepare input features.

        Accepts a dictionary of feature_name: value pairs.
        Returns a scaled numpy array ready for prediction.
        """
        if self.feature_names is None:
            raise ValueError("Feature names not loaded. Cannot validate input.")

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([features])

        # Check for missing features
        missing = set(self.feature_names) - set(input_df.columns)
        if missing:
            # Fill missing features with 0
            for col in missing:
                input_df[col] = 0

        # Keep only expected features in correct order
        input_df = input_df[self.feature_names]

        # Replace inf/nan
        input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale
        if self.scaler is not None:
            scaled = self.scaler.transform(input_df)
        else:
            scaled = input_df.values

        return scaled

    def predict_single(self, features: dict) -> dict:
        """
        Predict on a single network flow.

        Args:
            features: dict of feature_name: value

        Returns:
            dict with prediction results:
            - label: "benign" or "attack"
            - confidence: float [0, 1]
            - xgb_score: XGBoost attack probability
            - iso_score: Isolation Forest anomaly score
            - hybrid_score: weighted combination
        """
        scaled = self._validate_input(features)

        result = {
            "label": "benign",
            "confidence": 0.0,
            "xgb_score": 0.0,
            "iso_score": 0.0,
            "hybrid_score": 0.0,
            "threshold": 0.5
        }

        # XGBoost prediction
        if self.xgb_model is not None:
            xgb_proba = self.xgb_model.predict_proba(scaled)[0, 1]
            result["xgb_score"] = float(xgb_proba)

        # Isolation Forest prediction
        if self.iso_model is not None:
            iso_raw = self.iso_model.score_samples(scaled)[0]
            # Normalize: more negative = more anomalous
            iso_score = -iso_raw
            # Rough normalization to [0, 1]
            iso_score = max(0, min(1, (iso_score + 0.5)))
            result["iso_score"] = float(iso_score)

        # Hybrid score
        hybrid = (self.hybrid_weight_xgb * result["xgb_score"] +
                  self.hybrid_weight_iso * result["iso_score"])
        result["hybrid_score"] = float(hybrid)

        # Final decision
        if hybrid >= 0.5:
            result["label"] = "attack"
            result["confidence"] = float(hybrid)
        else:
            result["label"] = "benign"
            result["confidence"] = float(1 - hybrid)

        return result

    def predict_batch(self, features_list: list) -> list:
        """
        Predict on multiple network flows.

        Args:
            features_list: list of dicts, each with feature_name: value

        Returns:
            list of prediction result dicts
        """
        results = []
        for features in features_list:
            result = self.predict_single(features)
            results.append(result)
        return results

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on a pandas DataFrame.

        Args:
            df: DataFrame where each row is a network flow

        Returns:
            DataFrame with original data + prediction columns
        """
        # Prepare features
        if self.feature_names is not None:
            missing = set(self.feature_names) - set(df.columns)
            for col in missing:
                df[col] = 0
            X = df[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            X = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        # XGBoost scores
        if self.xgb_model is not None:
            xgb_scores = self.xgb_model.predict_proba(X_scaled)[:, 1]
        else:
            xgb_scores = np.zeros(len(df))

        # Isolation Forest scores
        if self.iso_model is not None:
            iso_raw = self.iso_model.score_samples(X_scaled)
            iso_scores = np.clip(-iso_raw + 0.5, 0, 1)
        else:
            iso_scores = np.zeros(len(df))

        # Hybrid scores
        hybrid_scores = (self.hybrid_weight_xgb * xgb_scores +
                         self.hybrid_weight_iso * iso_scores)

        # Build result DataFrame
        result_df = df.copy()
        result_df['xgb_score'] = xgb_scores
        result_df['iso_score'] = iso_scores
        result_df['hybrid_score'] = hybrid_scores
        result_df['prediction'] = np.where(hybrid_scores >= 0.5, 'attack', 'benign')
        result_df['confidence'] = np.where(
            hybrid_scores >= 0.5, hybrid_scores, 1 - hybrid_scores
        )

        return result_df

    def get_model_info(self) -> dict:
        """Return information about loaded models."""
        return {
            "xgb_loaded": self.xgb_model is not None,
            "iso_loaded": self.iso_model is not None,
            "scaler_loaded": self.scaler is not None,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "hybrid_weights": {
                "xgb": self.hybrid_weight_xgb,
                "iso": self.hybrid_weight_iso
            }
        }