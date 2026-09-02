import numpy as np
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import RISK_LEVELS

class RiskScorer:
    def __init__(self):
        pass

    def compute_risk_score(self, anomaly_probability: float) -> float:
        """
        Maps 0.0-1.0 anomaly probability to 0-100 Risk Score using non-linear risk curve.
        """
        # Non-linear curve to reflect heightened security sensitivity at upper end
        risk_score = 100.0 * (1.0 / (1.0 + np.exp(-6.0 * (anomaly_probability - 0.45))))
        return round(float(np.clip(risk_score, 0.0, 100.0)), 2)

    def categorize_risk_level(self, risk_score: float) -> str:
        if risk_score >= 90.0:
            return "CRITICAL"
        elif risk_score >= 70.0:
            return "HIGH"
        elif risk_score >= 30.0:
            return "MEDIUM"
        else:
            return "LOW"

    def identify_threat_type(self, row: pd.Series) -> str:
        """
        Classifies the specific cloud threat scenario based on log attributes.
        """
        failed_logins = row.get("failed_logins", 0)
        data_mb = row.get("data_transferred_mb", 0)
        sensitive_files = row.get("sensitive_file_access_count", 0)
        hour = row.get("login_time_hour", 12)
        role = row.get("role", "")
        unusual_loc = row.get("unusual_location_flag", 0)

        if failed_logins >= 6:
            return "Brute Force Authentication Attack"
        elif data_mb >= 2000 or sensitive_files >= 15:
            return "Mass Data Exfiltration"
        elif (hour < 5 or hour > 22) and (role in ["Admin", "DevOps"] or unusual_loc == 1):
            return "Off-Hours Privilege Escalation"
        elif row.get("file_access_count", 0) >= 100:
            return "Insider Threat Reconnaissance"
        else:
            return "Anomalous Behavioral Deviation"

    def process_dataframe(self, raw_df: pd.DataFrame, detection_results: pd.DataFrame) -> pd.DataFrame:
        """
        Appends Risk Scores, Risk Levels, and Threat Categories to detection results.
        """
        df_out = raw_df.copy()
        df_out["ensemble_score"] = detection_results["ensemble_score"]
        df_out["is_anomaly_pred"] = detection_results["is_anomaly_pred"]
        df_out["rf_score"] = detection_results["rf_score"]
        df_out["if_score"] = detection_results["if_score"]
        df_out["ae_score"] = detection_results["ae_score"]

        df_out["risk_score"] = df_out["ensemble_score"].apply(self.compute_risk_score)
        df_out["risk_level"] = df_out["risk_score"].apply(self.categorize_risk_level)
        df_out["threat_type"] = df_out.apply(self.identify_threat_type, axis=1)

        return df_out
