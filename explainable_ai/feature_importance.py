import numpy as np
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from explainable_ai.shap_explanation import SHAPExplainer
from models.random_forest import RandomForestModel

class FeatureImportanceAnalyzer:
    def __init__(self, rf_model: RandomForestModel, shap_explainer: SHAPExplainer):
        self.rf_model = rf_model
        self.shap_explainer = shap_explainer

    def get_global_rf_importance(self) -> pd.Series:
        return self.rf_model.get_feature_importances()

    def get_global_shap_importance(self, X_sample: pd.DataFrame) -> pd.Series:
        """
        Computes mean absolute SHAP value per feature across multiple samples.
        """
        if self.shap_explainer.explainer is None:
            self.shap_explainer._init_explainer()

        shap_vals = self.shap_explainer.explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            vals = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            vals = shap_vals[:, :, 1]
        else:
            vals = shap_vals

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        s_shap = pd.Series(mean_abs_shap, index=X_sample.columns).sort_values(ascending=False)
        return s_shap
