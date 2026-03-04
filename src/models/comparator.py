"""
═══════════════════════════════════════════════════════════════
NetSentinel — Model Comparator
Compare all trained models side by side
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ModelComparator:
    """
    Collects results from all models and generates
    comparison visualizations and reports.
    """

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, model):
        """Register a trained and evaluated model."""
        self.models[model.model_name] = model
        self.results[model.model_name] = model.results

    def get_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison DataFrame."""
        df = pd.DataFrame(self.results).T
        df = df.sort_values('f1', ascending=False)
        return df

    def plot_comparison(self, save_path=None):
        """
        Bar chart comparing all models across key metrics.
        """
        df = self.get_comparison_table()
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        df_plot = df[metrics]

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(df_plot.index))
        width = 0.15
        colors = ['#0F3057', '#2E86AB', '#E8683F', '#F6AE2D', '#86BA90']

        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, df_plot[metric], width,
                   label=metric.upper(), color=colors[i])

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('NetSentinel — Model Comparison',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_plot.index, fontsize=11)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_training_time(self, save_path=None):
        """Compare training times."""
        df = self.get_comparison_table()

        fig, ax = plt.subplots(figsize=(10, 5))
        df['training_time'].plot(kind='barh', ax=ax, color='#0F3057')
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_roc_comparison(self, X_test, y_test, save_path=None):
        """Overlay ROC curves for all models."""
        from sklearn.metrics import roc_curve, roc_auc_score

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#0F3057', '#2E86AB', '#E8683F', '#F6AE2D']

        for i, (name, model) in enumerate(self.models.items()):
            y_scores = model.predict_proba(X_test)
            if y_scores is not None:
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                auc = roc_auc_score(y_test, y_scores)
                ax.plot(fpr, tpr, color=colors[i % len(colors)],
                        lw=2, label=f'{name} (AUC={auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('NetSentinel — ROC Curve Comparison',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary(self):
        """Print formatted comparison summary."""
        df = self.get_comparison_table()

        print("\n" + "═" * 80)
        print("  NetSentinel — MODEL COMPARISON SUMMARY")
        print("═" * 80)

        print(f"\n  {'Model':<25} {'F1':>8} {'AUC':>8} {'Precision':>10} "
              f"{'Recall':>8} {'Time':>8}")
        print(f"  {'─' * 67}")

        for name, row in df.iterrows():
            print(f"  {name:<25} {row['f1']:>8.4f} {row['roc_auc']:>8.4f} "
                  f"{row['precision']:>10.4f} {row['recall']:>8.4f} "
                  f"{row['training_time']:>7.1f}s")

        best = df['f1'].idxmax()
        print(f"\n  🏆 Best model (F1): {best} ({df.loc[best, 'f1']:.4f})")
        print("═" * 80)