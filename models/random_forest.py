import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import RANDOM_FOREST_PARAMS, TRAINED_MODELS_DIR

class RandomForestModel:
    def __init__(self, params=None):
        self.params = params or RANDOM_FOREST_PARAMS
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        print("[RandomForest] Training Random Forest Classifier...")
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        print("[RandomForest] Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")
        return self.model.predict(X)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns probability of anomaly (class 1).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict_score.")
        probas = self.model.predict_proba(X)
        if probas.shape[1] > 1:
            return probas[:, 1]
        return probas[:, 0]

    def get_feature_importances(self) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importances.")
        importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
        return importances.sort_values(ascending=False)

    def save(self, filepath=TRAINED_MODELS_DIR / "random_forest.pkl"):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[RandomForest] Saved model to {filepath}")

    @staticmethod
    def load(filepath=TRAINED_MODELS_DIR / "random_forest.pkl"):
        model = joblib.load(filepath)
        print(f"[RandomForest] Loaded model from {filepath}")
        return model
