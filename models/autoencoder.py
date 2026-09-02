import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import joblib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import AUTOENCODER_PARAMS, TRAINED_MODELS_DIR

class AutoencoderModel:
    """
    Autoencoder Anomaly Detector using multi-layer neural reconstruction error.
    Trained to reconstruct normal feature vectors. High reconstruction loss (MSE) indicates an anomaly.
    """
    def __init__(self, params=None):
        self.params = params or AUTOENCODER_PARAMS
        # Bottleneck MLP Autoencoder: input -> 16 -> 8 -> 16 -> output
        self.model = MLPRegressor(
            hidden_layer_sizes=(16, 8, 16),
            activation="relu",
            solver="adam",
            max_iter=self.params.get("epochs", 50),
            random_state=self.params.get("random_state", 42),
            early_stopping=True,
            validation_fraction=0.1
        )
        self.is_fitted = False
        self.reconstruction_threshold = 0.5

    def fit(self, X: pd.DataFrame):
        print("[Autoencoder] Training Neural Autoencoder...")
        # Train autoencoder to reconstruct target X from input X
        self.model.fit(X, X)
        self.is_fitted = True
        
        # Calculate training reconstruction error threshold (95th percentile)
        X_pred = self.model.predict(X)
        mse = np.mean(np.square(X.values - X_pred), axis=1)
        self.reconstruction_threshold = float(np.percentile(mse, 92))
        print(f"[Autoencoder] Training complete. Anomaly threshold set to MSE: {self.reconstruction_threshold:.4f}")

    def get_reconstruction_mse(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing reconstruction loss.")
        X_pred = self.model.predict(X)
        mse = np.mean(np.square(X.values - X_pred), axis=1)
        return mse

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns continuous anomaly score between 0.0 and 1.0 based on normalized reconstruction error.
        """
        mse = self.get_reconstruction_mse(X)
        # Sigmoid or linear scaling relative to threshold
        scores = 1.0 / (1.0 + np.exp(-3.0 * (mse - self.reconstruction_threshold) / (self.reconstruction_threshold + 1e-5)))
        return np.clip(scores, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.predict_score(X)
        return np.where(scores >= 0.5, 1, 0)

    def save(self, filepath=TRAINED_MODELS_DIR / "autoencoder.pkl"):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Autoencoder] Saved model to {filepath}")

    @staticmethod
    def load(filepath=TRAINED_MODELS_DIR / "autoencoder.pkl"):
        model = joblib.load(filepath)
        print(f"[Autoencoder] Loaded model from {filepath}")
        return model
