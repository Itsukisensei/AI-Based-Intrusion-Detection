import numpy as np
import pandas as pd
import shap
import joblib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.random_forest import RandomForestModel
from config.config import TRAINED_MODELS_DIR

class SHAPExplainer:
    """
    Explainable AI engine leveraging SHAP (SHapley Additive exPlanations)
    to interpret tree ensemble prediction scores and attribute risk drivers.
    """
    def __init__(self, rf_model: RandomForestModel = None):
        self.rf_model = rf_model
        self.explainer = None
        if rf_model and rf_model.is_fitted:
            self._init_explainer()

    def _init_explainer(self):
        # Use TreeExplainer for Random Forest
        self.explainer = shap.TreeExplainer(self.rf_model.model)
        print("[SHAPExplainer] Initialized SHAP TreeExplainer.")

    def get_local_explanation(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates SHAP values for a single sample row or small batch.
        Returns DataFrame with feature names, sample feature values, and SHAP impact scores.
        """
        if self.explainer is None:
            if self.rf_model and self.rf_model.is_fitted:
                self._init_explainer()
            else:
                raise ValueError("Random Forest model must be fitted to compute SHAP explanations.")

        shap_vals = self.explainer.shap_values(X_sample)
        
        # Handle multi-class / binary array shapes
        if isinstance(shap_vals, list):
            # Select class 1 (anomaly)
            vals = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            vals = shap_vals[:, :, 1]
        else:
            vals = shap_vals

        if vals.ndim == 2:
            vals_single = vals[0]
        else:
            vals_single = vals

        feat_names = list(X_sample.columns)
        feat_values = X_sample.iloc[0].values

        df_explanation = pd.DataFrame({
            "feature": feat_names,
            "feature_value": feat_values,
            "shap_value": vals_single,
            "abs_shap": np.abs(vals_single)
        }).sort_values(by="abs_shap", ascending=False)

        return df_explanation

    def generate_natural_explanation(self, df_explanation: pd.DataFrame, top_k=3) -> str:
        """
        Translates top SHAP feature attribution scores into plain-English security incident rationale.
        """
        top_drivers = df_explanation[df_explanation["shap_value"] > 0].head(top_k)
        if top_drivers.empty:
            top_drivers = df_explanation.head(top_k)

        reasons = []
        for _, row in top_drivers.iterrows():
            feat = row["feature"]
            val = row["feature_value"]
            impact = row["shap_value"]
            
            clean_feat_name = feat.replace("_", " ").title()
            reasons.append(f"{clean_feat_name} (val: {val:.2f}, SHAP impact: +{impact:.3f})")

        explanation_str = "Primary risk drivers: " + "; ".join(reasons) + "."
        return explanation_str
