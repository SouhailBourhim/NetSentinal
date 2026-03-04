# tests/test_temporal_detection.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Make sure we can import from src/
import sys
sys.path.append(".")

from src.api.predictor import NetSentinelPredictor


def load_temporal_data(raw_dir: str = "data/raw"):
    """
    Load Thursday + Friday traffic from CIC-IDS2017.
    These days contain many attack types.
    """
    raw_path = Path(raw_dir)

    # Use all Thursday + Friday files
    files = [
        f for f in raw_path.glob("*.csv")
        if "Thursday" in f.name or "Friday" in f.name
    ]

    if not files:
        raise FileNotFoundError(
            f"No Thursday/Friday CSVs found in {raw_dir}. "
            "Make sure CIC-IDS2017 CSVs are extracted there."
        )

    dfs = []
    for f in files:
        print(f"Loading {f.name}...")
        temp = pd.read_csv(f, low_memory=False)
        temp.columns = (
            temp.columns
            .str.strip()
            .str.replace(' ', '_')
            .str.replace('/', '_')
            .str.lower()
        )
        dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} flows from {len(files)} files")

    return df


def prepare_features(df: pd.DataFrame, feature_names):
    """
    Select and clean feature columns to match those used by the predictor.
    """
    df_features = df.copy()

    # Ensure all expected features are present
    missing = set(feature_names) - set(df_features.columns)
    if missing:
        print(f"Warning: {len(missing)} missing features, filling with 0:")
        for col in missing:
            print(f"  → {col}")
            df_features[col] = 0

    # Keep only expected columns, in order
    df_features = df_features[feature_names]

    # Clean inf / NaN
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_features


def main():
    # 1) Load raw temporal data
    df = load_temporal_data("data/raw")

    # 2) Extract ground truth labels
    if "label" not in df.columns:
        raise KeyError("Column 'Label' (or 'label') not found in data.")

    # Normalize label column name/case
    if "Label" in df.columns and "label" not in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)

    # Binary labels: benign (0) vs attack (1)
    df["label_binary"] = (df["label"] != "BENIGN").astype(int)

    print("\nLabel distribution (temporal test set):")
    print(df["label"].value_counts())

    # 3) Instantiate predictor (loads models + scaler + feature_names)
    predictor = NetSentinelPredictor(models_path="saved_models")
    info = predictor.get_model_info()
    feature_names = predictor.feature_names

    if not info["xgb_loaded"]:
        raise RuntimeError("XGBoost model not loaded. Check saved_models/xgboost_tuned.pkl")

    if predictor.scaler is None:
        raise RuntimeError("Scaler not loaded. Check saved_models/scaler.pkl")

    print("\nModel info:", info)

    # 4) Prepare features
    X_df = prepare_features(df, feature_names)

    # 5) Predict using predictor.predict_dataframe (hybrid scores)
    print("\nRunning predictions on temporal test data...")
    results_df = predictor.predict_dataframe(X_df)

    # 6) Attach labels back
    results_df["label"] = df["label"].values
    results_df["label_binary"] = df["label_binary"].values

    # 7) Global metrics
    y_true = results_df["label_binary"].values
    y_pred = (results_df["hybrid_score"].values >= 0.5).astype(int)
    y_scores = results_df["hybrid_score"].values

    print("\n=== GLOBAL BINARY METRICS (Benign vs Attack) ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"], digits=4))

    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
    try:
        print(f"AUC-ROC:   {roc_auc_score(y_true, y_scores):.4f}")
    except Exception as e:
        print("AUC-ROC could not be computed:", e)

    # 8) Per-attack-type detection analysis
    print("\n=== DETECTION BY ATTACK TYPE (Temporal) ===")
    print(f"{'Attack Type':<35} {'Total':>8} {'Detected':>10} {'Rate':>8}")
    print("-" * 65)

    grouped = results_df.groupby("label")
    for label, g in grouped:
        total = len(g)
        if label == "BENIGN":
            detected_attacks = (g["hybrid_score"] >= 0.5).sum()
            rate = detected_attacks / total * 100 if total > 0 else 0
            print(f"{'BENIGN (FP rate)':<35} {total:>8} {detected_attacks:>10} {rate:>7.2f}%")
        else:
            detected = (g["hybrid_score"] >= 0.5).sum()
            rate = detected / total * 100 if total > 0 else 0
            print(f"{label:<35} {total:>8} {detected:>10} {rate:>7.2f}%")

    # 9) Save results (optional)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "temporal_predictions.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed predictions to {out_csv}")


if __name__ == "__main__":
    main()
