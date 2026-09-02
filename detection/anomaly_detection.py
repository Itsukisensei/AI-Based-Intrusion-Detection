import numpy as np
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.isolation_forest import IsolationForestModel
from models.random_forest import RandomForestModel
from models.autoencoder import AutoencoderModel

class EnsembleAnomalyDetector:
    def __init__(self, if_model=None, rf_model=None, ae_model=None, weights=None):
        self.if_model = if_model
        self.rf_model = rf_model
        self.ae_model = ae_model
        # Weights: Random Forest 40%, Isolation Forest 35%, Autoencoder 25%
        self.weights = weights or {"rf": 0.40, "if": 0.35, "ae": 0.25}

    def predict_detailed(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates individual model anomaly scores and weighted ensemble anomaly score.
        """
        results = pd.DataFrame(index=X.index)

        if self.rf_model and self.rf_model.is_fitted:
            results["rf_score"] = self.rf_model.predict_score(X)
        else:
            results["rf_score"] = 0.0

        if self.if_model and self.if_model.is_fitted:
            results["if_score"] = self.if_model.predict_score(X)
        else:
            results["if_score"] = 0.0

        if self.ae_model and self.ae_model.is_fitted:
            results["ae_score"] = self.ae_model.predict_score(X)
        else:
            results["ae_score"] = 0.0

        # Weighted Ensemble Anomaly Score (0.0 to 1.0)
        results["ensemble_score"] = (
            self.weights["rf"] * results["rf_score"] +
            self.weights["if"] * results["if_score"] +
            self.weights["ae"] * results["ae_score"]
        )

        results["is_anomaly_pred"] = np.where(results["ensemble_score"] >= 0.45, 1, 0)
        return results
