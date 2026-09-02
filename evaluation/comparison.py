import numpy as np
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from evaluation.metrics import EvaluationMetrics
from detection.anomaly_detection import EnsembleAnomalyDetector

class ModelComparator:
    def __init__(self, if_model, rf_model, ae_model, ensemble_detector: EnsembleAnomalyDetector):
        self.if_model = if_model
        self.rf_model = rf_model
        self.ae_model = ae_model
        self.ensemble = ensemble_detector

    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Runs evaluation across all individual models and ensemble model.
        Returns side-by-side comparison DataFrame.
        """
        comparison_results = []

        # 1. Isolation Forest
        if_scores = self.if_model.predict_score(X_test)
        if_preds = self.if_model.predict(X_test)
        if_metrics = EvaluationMetrics.evaluate_predictions(y_test, if_preds, if_scores)
        if_metrics["Model"] = "Isolation Forest (Unsupervised)"
        comparison_results.append(if_metrics)

        # 2. Random Forest
        rf_scores = self.rf_model.predict_score(X_test)
        rf_preds = self.rf_model.predict(X_test)
        rf_metrics = EvaluationMetrics.evaluate_predictions(y_test, rf_preds, rf_scores)
        rf_metrics["Model"] = "Random Forest (Supervised)"
        comparison_results.append(rf_metrics)

        # 3. Autoencoder
        ae_scores = self.ae_model.predict_score(X_test)
        ae_preds = self.ae_model.predict(X_test)
        ae_metrics = EvaluationMetrics.evaluate_predictions(y_test, ae_preds, ae_scores)
        ae_metrics["Model"] = "Neural Autoencoder (Reconstruction Error)"
        comparison_results.append(ae_metrics)

        # 4. Weighted Ensemble
        ensemble_res = self.ensemble.predict_detailed(X_test)
        ens_preds = ensemble_res["is_anomaly_pred"].values
        ens_scores = ensemble_res["ensemble_score"].values
        ens_metrics = EvaluationMetrics.evaluate_predictions(y_test, ens_preds, ens_scores)
        ens_metrics["Model"] = "Weighted Ensemble (Proposed Model)"
        comparison_results.append(ens_metrics)

        df_comp = pd.DataFrame(comparison_results)
        # Reorder columns
        cols = ["Model", "Precision", "Recall (Detection Rate)", "F1-Score", "False Positive Rate (FPR)", "ROC-AUC", "PR-AUC"]
        df_comp = df_comp[cols]
        return df_comp
