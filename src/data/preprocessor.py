"""
═══════════════════════════════════════════════════════════════
NetSentinel — Data Preprocessor
Cleans, transforms, and prepares network traffic data for ML
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class NetworkDataPreprocessor:
    """
    Handles all data cleaning and preparation steps
    for the CIC-IDS2017 network traffic dataset.
    """

    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.cleaning_report = {}

    def load_and_merge(self) -> pd.DataFrame:
        """
        Load all CSV files and merge into one DataFrame.
        """
        print("=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)

        csv_files = list(self.raw_data_path.glob("*.csv"))
        dfs = []

        for f in csv_files:
            print(f"  Loading {f.name}...")
            temp = pd.read_csv(f, low_memory=False)
            temp.columns = temp.columns.str.strip()
            dfs.append(temp)
            print(f"    → {temp.shape[0]:,} rows")

        df = pd.concat(dfs, ignore_index=True)
        self.cleaning_report['initial_rows'] = len(df)
        self.cleaning_report['initial_columns'] = len(df.columns)

        print(f"\n  ✅ Total: {len(df):,} rows, {len(df.columns)} columns")
        return df

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names:
        - Strip whitespace
        - Replace spaces with underscores
        - Convert to lowercase
        """
        print("\n" + "=" * 60)
        print("STEP 2: CLEANING COLUMN NAMES")
        print("=" * 60)

        df.columns = (
            df.columns
            .str.strip()
            .str.replace(' ', '_')
            .str.replace('/', '_')
            .str.lower()
        )

        print(f"  ✅ Columns standardized")
        print(f"  Example: {list(df.columns[:5])}")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate rows.
        """
        print("\n" + "=" * 60)
        print("STEP 3: REMOVING DUPLICATES")
        print("=" * 60)

        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        removed = before - after

        self.cleaning_report['duplicates_removed'] = removed
        print(f"  Before: {before:,}")
        print(f"  After:  {after:,}")
        print(f"  ✅ Removed {removed:,} duplicate rows ({removed/before*100:.1f}%)")
        return df

    def handle_missing_and_infinite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace infinite values with NaN, then drop rows with NaN.
        """
        print("\n" + "=" * 60)
        print("STEP 4: HANDLING MISSING & INFINITE VALUES")
        print("=" * 60)

        # Count before
        before = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Count infinites
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        print(f"  Infinite values found: {inf_count:,}")

        # Replace inf with NaN
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Count NaN
        nan_count = df.isnull().sum().sum()
        print(f"  NaN values (including replaced Inf): {nan_count:,}")

        # Drop rows with NaN
        df = df.dropna()
        after = len(df)
        removed = before - after

        self.cleaning_report['inf_values'] = int(inf_count)
        self.cleaning_report['rows_dropped_nan_inf'] = removed

        print(f"  Before: {before:,}")
        print(f"  After:  {after:,}")
        print(f"  ✅ Removed {removed:,} rows with NaN/Inf ({removed/before*100:.2f}%)")
        return df

    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns where all values are the same.
        These provide zero information to the model.
        """
        print("\n" + "=" * 60)
        print("STEP 5: REMOVING CONSTANT COLUMNS")
        print("=" * 60)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]

        self.cleaning_report['constant_columns_removed'] = constant_cols

        if constant_cols:
            print(f"  Found {len(constant_cols)} constant columns:")
            for col in constant_cols:
                print(f"    → {col}")
            df = df.drop(columns=constant_cols)
        else:
            print("  ✅ No constant columns found")

        print(f"  ✅ Remaining columns: {len(df.columns)}")
        return df

    def remove_highly_correlated(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove one of each pair of features with correlation > threshold.
        Keeps the one more correlated with the label.
        """
        print("\n" + "=" * 60)
        print(f"STEP 6: REMOVING HIGHLY CORRELATED FEATURES (>{threshold})")
        print("=" * 60)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()

        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find columns with correlation > threshold
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        self.cleaning_report['highly_correlated_removed'] = to_drop

        if to_drop:
            print(f"  Found {len(to_drop)} highly correlated columns:")
            for col in to_drop[:10]:
                print(f"    → {col}")
            if len(to_drop) > 10:
                print(f"    ... and {len(to_drop) - 10} more")
            df = df.drop(columns=to_drop)
        else:
            print("  ✅ No highly correlated columns found")

        print(f"  ✅ Remaining columns: {len(df.columns)}")
        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary and multiclass label columns.
        """
        print("\n" + "=" * 60)
        print("STEP 7: CREATING LABEL COLUMNS")
        print("=" * 60)

        # Binary: 0 = Benign, 1 = Attack
        df['label_binary'] = (df['label'] != 'BENIGN').astype(int)

        # Multiclass: map each attack type to a number
        attack_types = df['label'].unique()
        label_map = {label: idx for idx, label in enumerate(sorted(attack_types))}
        df['label_multi'] = df['label'].map(label_map)

        self.cleaning_report['label_map'] = label_map

        print("  Label mapping:")
        for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
            count = len(df[df['label'] == label])
            print(f"    {idx:2d}. {label:30s} → {count:>10,} samples")

        print(f"\n  Binary distribution:")
        print(f"    Benign: {len(df[df['label_binary'] == 0]):,}")
        print(f"    Attack: {len(df[df['label_binary'] == 1]):,}")

        return df

    def save_processed(self, df: pd.DataFrame):
        """
        Save processed data and cleaning report.
        """
        print("\n" + "=" * 60)
        print("STEP 8: SAVING PROCESSED DATA")
        print("=" * 60)

        # Save full processed dataset
        output_file = self.processed_data_path / "processed_traffic.csv"
        df.to_csv(output_file, index=False)
        print(f"  ✅ Saved to {output_file}")
        print(f"     → {len(df):,} rows, {len(df.columns)} columns")

        # Save cleaning report
        report_file = self.processed_data_path / "cleaning_report.json"

        # Convert non-serializable types
        report = {}
        for key, value in self.cleaning_report.items():
            if isinstance(value, (np.integer, np.int64)):
                report[key] = int(value)
            elif isinstance(value, dict):
                report[key] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else v
                              for k, v in value.items()}
            else:
                report[key] = value

        report['final_rows'] = len(df)
        report['final_columns'] = len(df.columns)

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  ✅ Cleaning report saved to {report_file}")

    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        """
        print("\n" + "║" * 60)
        print("  NetSentinel — Data Preprocessing Pipeline")
        print("║" * 60 + "\n")

        df = self.load_and_merge()
        df = self.clean_column_names(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing_and_infinite(df)
        df = self.remove_constant_columns(df)
        df = self.remove_highly_correlated(df, threshold=0.95)
        df = self.create_labels(df)
        self.save_processed(df)

        print("\n" + "=" * 60)
        print("✅ PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Initial:  {self.cleaning_report['initial_rows']:,} rows, "
              f"{self.cleaning_report['initial_columns']} columns")
        print(f"  Final:    {len(df):,} rows, {len(df.columns)} columns")
        print(f"  Removed:  {self.cleaning_report['initial_rows'] - len(df):,} rows")

        return df