"""
═══════════════════════════════════════════════════════════════
NetSentinel — Autoencoder Anomaly Detector
Deep learning unsupervised anomaly detection

HOW IT WORKS:
1. ENCODER compresses input features into a small bottleneck
2. DECODER reconstructs the original features from the bottleneck
3. Model is trained on NORMAL traffic only
4. Normal traffic → low reconstruction error (model learned it)
5. Attack traffic → HIGH reconstruction error (model never saw it)
6. Threshold: if error > threshold → anomaly

ARCHITECTURE:
  Input (48) → 64 → 32 → [16] → 32 → 64 → Output (48)
                           ↑
                      bottleneck

WHY FOR NETWORK ANOMALY DETECTION:
- Can detect UNKNOWN attacks (zero-day)
- Learns complex non-linear patterns
- Doesn't need attack labels for training
- Complementary to supervised approaches
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import time

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Autoencoder will be disabled.")

from src.models.base_model import BaseAnomalyDetector


class AutoencoderDetector(BaseAnomalyDetector):

    def __init__(self, input_dim, encoding_dim=16,
                 hidden_layers=[64, 32], epochs=50,
                 batch_size=256, learning_rate=0.001,
                 threshold_percentile=95):
        super().__init__("Autoencoder")
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for AutoencoderDetector but is not available. "
                            "Please install TensorFlow or use a different model.")

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.history = None

        self.model = self._build_model()

    def _build_model(self):
        """
        Build the autoencoder architecture.

        Input (48) → 64 → 32 → [16] → 32 → 64 → Output (48)

        The bottleneck (16 neurons) forces the model to learn
        a compressed representation of normal traffic.
        Attack traffic won't compress well → high reconstruction error.
        """
        # Encoder
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs

        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Bottleneck
        bottleneck = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(x)

        # Decoder (mirror of encoder)
        x = bottleneck
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Output (reconstruct input)
        outputs = layers.Dense(self.input_dim, activation='linear')(x)

        model = Model(inputs, outputs, name='autoencoder')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def train(self, X_train, y_train=None):
        """
        Train autoencoder on NORMAL traffic only.

        Key insight: We filter X_train to keep only benign samples.
        The model learns to reconstruct normal traffic patterns.
        When it sees attack traffic, it can't reconstruct it well
        → high reconstruction error → detected as anomaly.
        """
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'=' * 60}")

        # Filter to normal traffic only
        if y_train is not None:
            if hasattr(y_train, 'values'):
                y_values = y_train.values
            else:
                y_values = np.array(y_train)

            normal_mask = y_values == 0
            X_normal = X_train[normal_mask] if hasattr(X_train, 'iloc') \
                else X_train[normal_mask]

            if hasattr(X_normal, 'values'):
                X_normal = X_normal.values
        else:
            X_normal = X_train if not hasattr(X_train, 'values') else X_train.values

        print(f"  Architecture:     {self.input_dim} → "
              f"{' → '.join(map(str, self.hidden_layers))} → "
              f"[{self.encoding_dim}] → "
              f"{' → '.join(map(str, reversed(self.hidden_layers)))} → "
              f"{self.input_dim}")
        print(f"  Normal samples:   {X_normal.shape[0]:,}")
        print(f"  Epochs:           {self.epochs}")
        print(f"  Batch size:       {self.batch_size}")
        print(f"  Learning rate:    {self.learning_rate}")

        start = time.time()

        self.history = self.model.fit(
            X_normal, X_normal,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
        )

        self.training_time = time.time() - start

        # Set threshold based on normal traffic reconstruction error
        reconstructions = self.model.predict(X_normal, verbose=0)
        mse = np.mean(np.square(X_normal - reconstructions), axis=1)
        self.threshold = np.percentile(mse, self.threshold_percentile)

        print(f"\n  ✅ Training complete ({self.training_time:.2f}s)")
        print(f"  Threshold (p{self.threshold_percentile}): {self.threshold:.6f}")

    def _get_reconstruction_error(self, X):
        """Calculate mean squared error for each sample."""
        if hasattr(X, 'values'):
            X = X.values
        reconstructions = self.model.predict(X, verbose=0)
        return np.mean(np.square(X - reconstructions), axis=1)

    def predict(self, X):
        """
        Predict anomalies based on reconstruction error.

        If error > threshold → attack (1)
        If error <= threshold → benign (0)
        """
        mse = self._get_reconstruction_error(X)
        return (mse > self.threshold).astype(int)

    def predict_proba(self, X):
        """
        Return reconstruction error as anomaly score.
        Higher error = more likely to be an attack.
        """
        return self._get_reconstruction_error(X)

    def plot_training_history(self, save_path=None):
        """Plot training and validation loss curves."""
        if self.history is None:
            print("  ⚠️ Train model first.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.history.history['loss'], label='Training Loss', color='#0F3057')
        ax.plot(self.history.history['val_loss'], label='Validation Loss', color='#E8683F')
        ax.set_title(f'{self.model_name} — Training History',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_error_distribution(self, X_test, y_test, save_path=None):
        """
        Plot reconstruction error distribution for benign vs attack traffic.

        Good model → clear separation between benign (low error)
        and attack (high error) distributions.
        """
        mse = self._get_reconstruction_error(X_test)

        if hasattr(y_test, 'values'):
            y_values = y_test.values
        else:
            y_values = np.array(y_test)

        benign_errors = mse[y_values == 0]
        attack_errors = mse[y_values == 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(benign_errors, bins=100, alpha=0.5, label='Benign',
                color='#2E86AB', density=True)
        ax.hist(attack_errors, bins=100, alpha=0.5, label='Attack',
                color='#E8683F', density=True)
        ax.axvline(self.threshold, color='red', linestyle='--',
                   label=f'Threshold ({self.threshold:.4f})')
        ax.set_title(f'{self.model_name} — Reconstruction Error Distribution',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Reconstruction Error (MSE)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_xlim(0, np.percentile(mse, 99))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def log_to_mlflow(self, experiment_name="network_anomaly_detection"):
        """Log Autoencoder specific parameters."""
        import mlflow

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=self.model_name):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("model_type", "deep_learning_unsupervised")
            mlflow.log_param("encoding_dim", self.encoding_dim)
            mlflow.log_param("hidden_layers", str(self.hidden_layers))
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("threshold", self.threshold)

            for metric, value in self.results.items():
                mlflow.log_metric(metric, value)

            print(f"  ✅ Logged to MLflow: {self.model_name}")