import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import ISOLATION_FOREST_PARAMS, TRAINED_MODELS_DIR

class IsolationForestModel:
    def __init__(self, params=None):
        self.params = params or ISOLATION_FOREST_PARAMS
        self.model = IsolationForest(**self.params)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        print("[IsolationForest] Training Isolation Forest model...")
        self.model.fit(X)
        self.is_fitted = True
        print("[IsolationForest] Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns binary predictions: 1 for anomaly, 0 for normal.
        Note: Sklearn IsolationForest outputs -1 for anomaly, 1 for normal.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns continuous anomaly score between 0.0 (normal) and 1.0 (highly anomalous).
        Uses raw decision_function score conversion.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict_score.")
        raw_scores = self.model.decision_function(X) # lower score = more anomalous
        # Min-Max normalize decision function inverse
        # Decision function is roughly between -0.35 (anomaly) and 0.25 (normal)
        score_norm = 1.0 - (raw_scores - (-0.4)) / (0.35 - (-0.4))
        return np.clip(score_norm, 0.0, 1.0)

    def save(self, filepath=TRAINED_MODELS_DIR / "isolation_forest.pkl"):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[IsolationForest] Saved model to {filepath}")

    @staticmethod
    def load(filepath=TRAINED_MODELS_DIR / "isolation_forest.pkl"):
        model = joblib.load(filepath)
        print(f"[IsolationForest] Loaded model from {filepath}")
        return model
