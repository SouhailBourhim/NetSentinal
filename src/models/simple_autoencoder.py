"""
═══════════════════════════════════════════════════════════════
NetSentinel — Simple Autoencoder (Scikit-learn based)
Alternative autoencoder implementation using PCA + reconstruction error

This is a simplified version that doesn't require TensorFlow.
Uses PCA for dimensionality reduction and reconstruction error for anomaly detection.
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseAnomalyDetector


class SimpleAutoencoderDetector(BaseAnomalyDetector):
    """
    Simple autoencoder using PCA for dimensionality reduction.
    
    How it works:
    1. Use PCA to reduce dimensions (encoding)
    2. Reconstruct back to original dimensions (decoding)
    3. Calculate reconstruction error
    4. High error = anomaly
    """

    def __init__(self, n_components=16, threshold_percentile=95):
        super().__init__("Simple Autoencoder (PCA)")
        
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.threshold = None
        self.reconstruction_errors = None

    def train(self, X_train, y_train=None):
        """
        Train the simple autoencoder on normal data only.
        """
        print(f"\n🔧 Training {self.model_name}...")
        
        # Use only normal data for training (unsupervised)
        if y_train is not None:
            normal_mask = (y_train == 0)
            X_normal = X_train[normal_mask]
            print(f"   Using {len(X_normal):,} normal samples for training")
        else:
            X_normal = X_train
            print(f"   Using all {len(X_normal):,} samples for training")
        
        # Fit scaler and PCA on normal data
        X_scaled = self.scaler.fit_transform(X_normal)
        self.pca.fit(X_scaled)
        
        # Calculate reconstruction errors on training data
        X_encoded = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_encoded)
        
        # Calculate reconstruction error (MSE)
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Set threshold based on percentile of normal data errors
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        print(f"   ✅ Training complete")
        print(f"   PCA components: {self.n_components}")
        print(f"   Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        print(f"   Anomaly threshold: {self.threshold:.6f}")

    def predict(self, X):
        """
        Predict anomalies based on reconstruction error.
        """
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Encode and decode
        X_encoded = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_encoded)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Store for analysis
        self.reconstruction_errors = reconstruction_errors
        
        # Predict: 1 if error > threshold (anomaly), 0 otherwise
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        return predictions

    def get_anomaly_scores(self, X):
        """
        Get anomaly scores (reconstruction errors).
        """
        X_scaled = self.scaler.transform(X)
        X_encoded = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_encoded)
        
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        return reconstruction_errors

    def plot_reconstruction_errors(self, y_true=None):
        """
        Plot distribution of reconstruction errors.
        """
        if self.reconstruction_errors is None:
            print("No reconstruction errors available. Run predict() first.")
            return
        
        plt.figure(figsize=(12, 5))
        
        if y_true is not None:
            # Plot separate distributions for normal vs attack
            plt.subplot(1, 2, 1)
            normal_errors = self.reconstruction_errors[y_true == 0]
            attack_errors = self.reconstruction_errors[y_true == 1]
            
            plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
            plt.hist(attack_errors, bins=50, alpha=0.7, label='Attack', color='red', density=True)
            plt.axvline(self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold:.6f})')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Density')
            plt.title('Reconstruction Error Distribution')
            plt.legend()
            plt.yscale('log')
            
            plt.subplot(1, 2, 2)
        else:
            plt.subplot(1, 1, 1)
        
        # Plot all reconstruction errors
        plt.hist(self.reconstruction_errors, bins=100, alpha=0.7, color='green', density=True)
        plt.axvline(self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold:.6f})')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('All Reconstruction Errors')
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()

    def get_feature_importance(self):
        """
        Get feature importance based on PCA components.
        """
        if self.pca is None:
            return None
        
        # Use the magnitude of the first principal component as feature importance
        feature_importance = np.abs(self.pca.components_[0])
        return feature_importance / feature_importance.sum()