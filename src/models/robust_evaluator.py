"""
═══════════════════════════════════════════════════════════════
NetSentinel — Robust Model Evaluator
Provides realistic evaluation by addressing data leakage
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class RobustEvaluator:
    """
    Evaluates models using three strategies:
    1. Deduplicated random split
    2. Temporal split (Monday→Friday)
    3. Cross-validation on deduplicated data

    This gives honest, publishable results.
    """

    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.results = {}

    # ─────────────────────────────────────────────────
    # STRATEGY 1: Deduplicated Evaluation
    # ─────────────────────────────────────────────────

    def deduplicated_evaluation(self):
        """
        Remove near-duplicates BEFORE splitting.

        Method:
        1. Round all features to 2 decimal places
        2. Drop duplicates on rounded values
        3. This removes flows that are nearly identical
        4. Split the remaining unique flows
        5. Train and evaluate

        Result: Model can't memorize — must generalize.
        """
        print("\n" + "═" * 60)
        print("  STRATEGY 1: DEDUPLICATED EVALUATION")
        print("═" * 60)

        # Load original processed data
        df = pd.read_csv(
            self.processed_data_path / "processed_traffic.csv",
            low_memory=False
        )

        # Identify columns
        exclude_cols = ['label', 'label_binary', 'label_multi']
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        drop_cols = list(set(exclude_cols + non_numeric))
        drop_cols = [col for col in drop_cols if col in df.columns]

        feature_cols = [col for col in df.columns if col not in drop_cols]

        print(f"  Original samples: {len(df):,}")

        # Round features and remove near-duplicates
        df_rounded = df.copy()
        df_rounded[feature_cols] = df_rounded[feature_cols].round(2)
        df_dedup = df_rounded.drop_duplicates(subset=feature_cols)

        print(f"  After deduplication: {len(df_dedup):,}")
        print(f"  Removed: {len(df) - len(df_dedup):,} near-duplicates "
              f"({(len(df) - len(df_dedup))/len(df)*100:.1f}%)")

        # Prepare features and labels
        X = df_dedup[feature_cols].values
        y = df_dedup['label_binary'].values

        print(f"  Benign: {(y == 0).sum():,} | Attack: {(y == 1).sum():,}")

        # Stratified split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train XGBoost
        print(f"\n  Training XGBoost on deduplicated data...")
        xgb = XGBClassifier(
            n_estimators=200, max_depth=10,
            learning_rate=0.1, random_state=42,
            n_jobs=-1, eval_metric='logloss',
            use_label_encoder=False
        )
        xgb.fit(X_train, y_train)

        y_pred = xgb.predict(X_test)
        y_scores = xgb.predict_proba(X_test)[:, 1]

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_scores)
        }

        self.results['deduplicated'] = results

        print(f"\n  {'Metric':<15} {'Score':>10}")
        print(f"  {'─' * 25}")
        for metric, value in results.items():
            print(f"  {metric:<15} {value:>10.4f}")

        # Train Random Forest too
        print(f"\n  Training Random Forest on deduplicated data...")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=20,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_test)
        y_scores_rf = rf.predict_proba(X_test)[:, 1]

        results_rf = {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf),
            'recall': recall_score(y_test, y_pred_rf),
            'f1': f1_score(y_test, y_pred_rf),
            'roc_auc': roc_auc_score(y_test, y_scores_rf)
        }

        self.results['deduplicated_rf'] = results_rf

        print(f"\n  {'Metric':<15} {'Score':>10}")
        print(f"  {'─' * 25}")
        for metric, value in results_rf.items():
            print(f"  {metric:<15} {value:>10.4f}")

        return results, y_test, y_scores, y_scores_rf

    # ─────────────────────────────────────────────────
    # STRATEGY 2: Temporal Split
    # ─────────────────────────────────────────────────

    def temporal_evaluation(self):
        """
        Train on early days, test on later days.
        Completely eliminates session-based leakage.

        Train: Monday + Tuesday + Wednesday
        Test: Thursday + Friday

        This simulates real deployment:
        "Can the model detect attacks it has never seen before?"
        """
        print("\n" + "═" * 60)
        print("  STRATEGY 2: TEMPORAL SPLIT EVALUATION")
        print("═" * 60)

        # Define day groups
        train_keywords = ['Monday', 'Tuesday', 'Wednesday']
        test_keywords = ['Thursday', 'Friday']

        csv_files = list(self.raw_data_path.glob("*.csv"))

        # Load train days
        train_dfs = []
        for f in csv_files:
            if any(day in f.name for day in train_keywords):
                print(f"  [TRAIN] Loading {f.name}...")
                temp = pd.read_csv(f, low_memory=False)
                temp.columns = temp.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.lower()
                train_dfs.append(temp)

        # Load test days
        test_dfs = []
        for f in csv_files:
            if any(day in f.name for day in test_keywords):
                print(f"  [TEST]  Loading {f.name}...")
                temp = pd.read_csv(f, low_memory=False)
                temp.columns = temp.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.lower()
                test_dfs.append(temp)

        df_train = pd.concat(train_dfs, ignore_index=True)
        df_test = pd.concat(test_dfs, ignore_index=True)

        # Create binary labels
        df_train['label_binary'] = (df_train['label'] != 'BENIGN').astype(int)
        df_test['label_binary'] = (df_test['label'] != 'BENIGN').astype(int)

        # Common numeric features
        exclude = ['label', 'label_binary']
        train_numeric = df_train.select_dtypes(include=[np.number]).columns.tolist()
        test_numeric = df_test.select_dtypes(include=[np.number]).columns.tolist()
        common_cols = [c for c in train_numeric
                       if c in test_numeric and c not in exclude]

        # Clean
        df_train = df_train[common_cols + ['label_binary']].replace(
            [np.inf, -np.inf], np.nan).dropna()
        df_test = df_test[common_cols + ['label_binary']].replace(
            [np.inf, -np.inf], np.nan).dropna()

        X_train = df_train[common_cols].values
        y_train = df_train['label_binary'].values
        X_test = df_test[common_cols].values
        y_test = df_test['label_binary'].values

        print(f"\n  Train (Mon-Wed): {len(X_train):,} samples "
              f"({y_train.sum():,} attacks)")
        print(f"  Test (Thu-Fri):  {len(X_test):,} samples "
              f"({y_test.sum():,} attacks)")

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train XGBoost
        print(f"\n  Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200, max_depth=10,
            learning_rate=0.1, random_state=42,
            n_jobs=-1, eval_metric='logloss',
            use_label_encoder=False
        )
        xgb.fit(X_train, y_train)

        y_pred = xgb.predict(X_test)
        y_scores = xgb.predict_proba(X_test)[:, 1]

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_scores)
        }

        self.results['temporal'] = results

        print(f"\n  {'Metric':<15} {'Score':>10}")
        print(f"  {'─' * 25}")
        for metric, value in results.items():
            print(f"  {metric:<15} {value:>10.4f}")

        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Benign', 'Attack'], digits=4))

        return results, y_test, y_scores

    # ─────────────────────────────────────────────────
    # STRATEGY 3: Cross-Validation
    # ─────────────────────────────────────────────────

    def crossval_evaluation(self, n_folds=5, sample_size=100000):
        """
        Stratified K-Fold cross-validation on deduplicated data.
        Most statistically rigorous evaluation method.
        """
        print("\n" + "═" * 60)
        print(f"  STRATEGY 3: {n_folds}-FOLD CROSS-VALIDATION")
        print("═" * 60)

        # Load and deduplicate
        df = pd.read_csv(
            self.processed_data_path / "processed_traffic.csv",
            low_memory=False
        )

        exclude_cols = ['label', 'label_binary', 'label_multi']
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        drop_cols = list(set(exclude_cols + non_numeric))
        drop_cols = [col for col in drop_cols if col in df.columns]
        feature_cols = [col for col in df.columns if col not in drop_cols]

        # Deduplicate
        df_rounded = df.copy()
        df_rounded[feature_cols] = df_rounded[feature_cols].round(2)
        df_dedup = df_rounded.drop_duplicates(subset=feature_cols)

        # Sample for speed
        if len(df_dedup) > sample_size:
            df_sample = df_dedup.sample(sample_size, random_state=42)
        else:
            df_sample = df_dedup

        X = df_sample[feature_cols].values
        y = df_sample['label_binary'].values

        print(f"  Samples: {len(X):,}")
        print(f"  Benign: {(y == 0).sum():,} | Attack: {(y == 1).sum():,}")

        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            xgb = XGBClassifier(
                n_estimators=200, max_depth=10,
                learning_rate=0.1, random_state=42,
                n_jobs=-1, eval_metric='logloss',
                use_label_encoder=False
            )
            xgb.fit(X_train, y_train)

            y_pred = xgb.predict(X_test)
            y_scores = xgb.predict_proba(X_test)[:, 1]

            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_scores)
            }
            fold_results.append(fold_result)

            print(f"  Fold {fold+1}: AUC={fold_result['roc_auc']:.4f} "
                  f"F1={fold_result['f1']:.4f}")

        # Average results
        df_folds = pd.DataFrame(fold_results)
        avg_results = {
            'accuracy': df_folds['accuracy'].mean(),
            'precision': df_folds['precision'].mean(),
            'recall': df_folds['recall'].mean(),
            'f1': df_folds['f1'].mean(),
            'roc_auc': df_folds['roc_auc'].mean(),
            'f1_std': df_folds['f1'].std(),
            'auc_std': df_folds['roc_auc'].std()
        }

        self.results['crossval'] = avg_results

        print(f"\n  {'Metric':<15} {'Mean':>10} {'± Std':>10}")
        print(f"  {'─' * 35}")
        
        f1_mean = avg_results['f1']
        f1_std = avg_results['f1_std']
        auc_mean = avg_results['roc_auc']
        auc_std = avg_results['auc_std']
        
        print(f"  {'F1':<15} {f1_mean:>10.4f} ± {f1_std:>9.4f}")
        print(f"  {'AUC':<15} {auc_mean:>10.4f} ± {auc_std:>9.4f}")

        return avg_results, df_folds

    # ─────────────────────────────────────────────────
    # FINAL COMPARISON
    # ─────────────────────────────────────────────────

    def generate_final_report(self, save_path=None):
        """
        Compare all evaluation strategies side by side.
        """
        print("\n" + "═" * 70)
        print("  NetSentinel — ROBUST EVALUATION REPORT")
        print("═" * 70)

        print(f"\n  {'Strategy':<25} {'AUC':>8} {'F1':>8} "
              f"{'Precision':>10} {'Recall':>8}")
        print(f"  {'─' * 59}")

        strategy_names = {
            'deduplicated': 'Dedup Split (XGBoost)',
            'deduplicated_rf': 'Dedup Split (RF)',
            'temporal': 'Temporal Split (XGBoost)',
            'crossval': 'Cross-Val (XGBoost)'
        }

        for key, name in strategy_names.items():
            if key in self.results:
                r = self.results[key]
                print(f"  {name:<25} {r['roc_auc']:>8.4f} "
                      f"{r['f1']:>8.4f} {r['precision']:>10.4f} "
                      f"{r['recall']:>8.4f}")

        print(f"\n  {'─' * 59}")
        print(f"  Original (with leakage):  AUC=1.0000  F1=1.0000")
        print(f"  {'─' * 59}")

        # Recommendation
        if 'temporal' in self.results:
            temporal_auc = self.results['temporal']['roc_auc']
            print(f"\n  📊 RECOMMENDED REPORTABLE METRICS:")
            print(f"     Use temporal split results (AUC={temporal_auc:.4f})")
            print(f"     This is the most realistic estimate of real-world performance")

        # Save
        if save_path:
            df = pd.DataFrame(self.results).T
            df.to_csv(save_path)
            print(f"\n  ✅ Report saved to {save_path}")

        # Plot comparison
        self._plot_comparison()

    def _plot_comparison(self, save_path=None):
        """Visual comparison of evaluation strategies."""

        strategies = []
        auc_scores = []
        f1_scores = []

        labels = {
            'deduplicated': 'Dedup\n(XGBoost)',
            'deduplicated_rf': 'Dedup\n(RF)',
            'temporal': 'Temporal\n(XGBoost)',
            'crossval': 'Cross-Val\n(XGBoost)'
        }

        for key, name in labels.items():
            if key in self.results:
                strategies.append(name)
                auc_scores.append(self.results[key]['roc_auc'])
                f1_scores.append(self.results[key]['f1'])

        # Add original for reference
        strategies.append('Original\n(Leaked)')
        auc_scores.append(1.0)
        f1_scores.append(1.0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(len(strategies))
        colors = ['#2E86AB'] * (len(strategies) - 1) + ['#E8683F']

        # AUC comparison
        axes[0].bar(x, auc_scores, color=colors, width=0.6)
        axes[0].set_title('AUC-ROC by Evaluation Strategy',
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('AUC-ROC')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategies, fontsize=9)
        axes[0].set_ylim(0, 1.1)
        axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5,
                        label='Random baseline')
        axes[0].legend()

        for i, v in enumerate(auc_scores):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

        # F1 comparison
        axes[1].bar(x, f1_scores, color=colors, width=0.6)
        axes[1].set_title('F1-Score by Evaluation Strategy',
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(strategies, fontsize=9)
        axes[1].set_ylim(0, 1.1)

        for i, v in enumerate(f1_scores):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

        plt.suptitle('NetSentinel — Impact of Data Leakage on Model Evaluation',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()