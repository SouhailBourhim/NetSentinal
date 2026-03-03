"""
═══════════════════════════════════════════════════════════════
NetSentinel — Data Splitter
Handles train/test splitting with stratification and balancing
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib


class DataSplitter:
    """
    Splits processed data into train/test sets.
    Handles:
    - Stratified splitting (preserves class ratios)
    - Feature scaling (StandardScaler)
    - Class balancing (SMOTE)
    """

    def __init__(self, processed_data_path: str):
        self.processed_data_path = Path(processed_data_path)
        self.scaler = StandardScaler()

    def load_processed_data(self) -> pd.DataFrame:
        """Load the processed dataset."""
        print("=" * 60)
        print("LOADING PROCESSED DATA")
        print("=" * 60)

        filepath = self.processed_data_path / "processed_traffic.csv"
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df

    def prepare_features_and_labels(self, df: pd.DataFrame, mode: str = 'binary'):
        """
        Separate features (X) from labels (y).

        mode:
        - 'binary'  → BENIGN (0) vs ATTACK (1)
        - 'multi'   → All 15 traffic categories
        """
        print(f"\n{'=' * 60}")
        print(f"PREPARING FEATURES ({mode} mode)")
        print("=" * 60)

        # Columns to exclude from features
        exclude_cols = ['label', 'label_binary', 'label_multi']
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        drop_cols = list(set(exclude_cols + non_numeric))
        drop_cols = [col for col in drop_cols if col in df.columns]

        X = df.drop(columns=drop_cols)
        y = df[f'label_{mode}'] if f'label_{mode}' in df.columns else df['label_binary']

        print(f"  Features (X): {X.shape}")
        print(f"  Labels (y):   {y.shape}")
        print(f"  Label distribution:")
        for val, count in y.value_counts().sort_index().items():
            print(f"    Class {val}: {count:,} ({count/len(y)*100:.1f}%)")

        return X, y

    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        """
        Stratified train/test split.

        Stratified = preserves the same class ratio in both
        train and test sets. This is critical for imbalanced data.
        """
        print(f"\n{'=' * 60}")
        print(f"SPLITTING DATA (test_size={test_size})")
        print("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"  Train: {X_train.shape[0]:,} samples")
        print(f"  Test:  {X_test.shape[0]:,} samples")
        print(f"\n  Train label distribution:")
        for val, count in y_train.value_counts().sort_index().items():
            print(f"    Class {val}: {count:,} ({count/len(y_train)*100:.1f}%)")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """
        Standardize features using StandardScaler.

        StandardScaler transforms each feature to have:
        - Mean = 0
        - Standard deviation = 1

        WHY: ML models (especially neural networks) perform better
        when all features are on the same scale.

        IMPORTANT: Fit scaler on TRAIN data only, then transform both.
        This prevents data leakage from test set.
        """
        print(f"\n{'=' * 60}")
        print("SCALING FEATURES")
        print("=" * 60)

        feature_names = X_train.columns.tolist()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

        print(f"  ✅ Features scaled (mean≈0, std≈1)")
        print(f"  Train sample means (first 5): "
              f"{X_train_scaled.iloc[:, :5].mean().values.round(3)}")

        return X_train_scaled, X_test_scaled

    def balance_classes(self, X_train, y_train, strategy: str = 'auto'):
        """
        Balance classes using SMOTE.

        SMOTE = Synthetic Minority Over-sampling Technique

        HOW IT WORKS:
        1. Takes a minority class sample
        2. Finds its k nearest neighbors (same class)
        3. Creates a NEW synthetic sample between them
        4. Repeats until classes are balanced

        WHY:
        Without balancing, model learns to predict "BENIGN" always
        (since 80% of data is benign) and gets 80% accuracy
        while catching ZERO attacks.

        IMPORTANT: Only apply SMOTE to TRAINING data, never test data.
        Test data must reflect real-world distribution.
        """
        print(f"\n{'=' * 60}")
        print("BALANCING CLASSES (SMOTE)")
        print("=" * 60)

        print(f"  Before SMOTE:")
        for val, count in y_train.value_counts().sort_index().items():
            print(f"    Class {val}: {count:,}")

        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"\n  After SMOTE:")
        for val, count in pd.Series(y_resampled).value_counts().sort_index().items():
            print(f"    Class {val}: {count:,}")

        print(f"\n  ✅ Training data balanced")
        print(f"     Before: {len(y_train):,} samples")
        print(f"     After:  {len(y_resampled):,} samples")

        return X_resampled, y_resampled

    def save_splits(self, X_train, X_test, y_train, y_test):
        """
        Save train/test splits and scaler for reproducibility.
        """
        print(f"\n{'=' * 60}")
        print("SAVING SPLITS")
        print("=" * 60)

        # Save as CSV
        X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
        X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
        y_train_s = pd.Series(y_train, name='label')
        y_test_s = pd.Series(y_test, name='label')

        X_train_df.to_csv(self.processed_data_path / "X_train.csv", index=False)
        X_test_df.to_csv(self.processed_data_path / "X_test.csv", index=False)
        y_train_s.to_csv(self.processed_data_path / "y_train.csv", index=False)
        y_test_s.to_csv(self.processed_data_path / "y_test.csv", index=False)

        # Save scaler
        joblib.dump(self.scaler, self.processed_data_path / "scaler.pkl")

        print(f"  ✅ X_train.csv  ({len(X_train_df):,} rows)")
        print(f"  ✅ X_test.csv   ({len(X_test_df):,} rows)")
        print(f"  ✅ y_train.csv  ({len(y_train_s):,} rows)")
        print(f"  ✅ y_test.csv   ({len(y_test_s):,} rows)")
        print(f"  ✅ scaler.pkl")

    def run_full_pipeline(self, mode: str = 'binary', balance: bool = True):
        """
        Execute the complete splitting pipeline.
        """
        print("\n" + "║" * 60)
        print("  NetSentinel — Data Splitting Pipeline")
        print("║" * 60 + "\n")

        df = self.load_processed_data()
        X, y = self.prepare_features_and_labels(df, mode=mode)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train, X_test = self.scale_features(X_train, X_test)

        if balance:
            X_train, y_train = self.balance_classes(X_train, y_train)

        self.save_splits(X_train, X_test, y_train, y_test)

        print("\n" + "=" * 60)
        print("✅ DATA SPLITTING COMPLETE")
        print("=" * 60)

        return X_train, X_test, y_train, y_test