import json
import pandas as pd
import datetime
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import ALERT_TRIGGER_SCORE, ALERTS_DIR
from explainable_ai.shap_explanation import SHAPExplainer

class AlertSystem:
    def __init__(self, shap_explainer: SHAPExplainer = None):
        self.shap_explainer = shap_explainer

    def generate_alerts(self, df_processed: pd.DataFrame, X_matrix: pd.DataFrame) -> list:
        """
        Scans processed risk-scored activity data and generates rich XAI security alerts for high risk incidents.
        """
        alerts = []
        high_risk_indices = df_processed[df_processed["risk_score"] >= ALERT_TRIGGER_SCORE].index

        for idx in high_risk_indices:
            row = df_processed.loc[idx]
            x_row = X_matrix.loc[[idx]]

            # Generate XAI SHAP Explanation
            if self.shap_explainer:
                try:
                    df_exp = self.shap_explainer.get_local_explanation(x_row)
                    xai_text = self.shap_explainer.generate_natural_explanation(df_exp, top_k=3)
                except Exception as e:
                    xai_text = f"Explanation generation fallback: High deviation in session/data activity metrics ({e})"
            else:
                xai_text = "Risk score exceeds safety baseline threshold."

            # Determine Security Remediation Action
            remediation = self._recommend_remediation(row["threat_type"], row["risk_level"])

            if isinstance(idx, (int, float)) and int(idx) > 0:
                alert_id = f"ALT-{int(idx):06d}"
            else:
                alert_id = f"ALT-{random.randint(100000, 999999)}"

            alert = {
                "alert_id": alert_id,
                "timestamp": str(row["timestamp"]),
                "user_id": str(row["user_id"]),
                "role": str(row["role"]),
                "ip_address": str(row["ip_address"]),
                "location": str(row["location"]),
                "risk_score": float(row["risk_score"]),
                "risk_level": str(row["risk_level"]),
                "threat_type": str(row["threat_type"]),
                "xai_explanation": xai_text,
                "recommended_remediation": remediation
            }
            alerts.append(alert)

        return alerts

    def _recommend_remediation(self, threat_type: str, risk_level: str) -> str:
        if risk_level == "CRITICAL":
            return "IMMEDIATE: Lock IAM User Account, Terminate Active Sessions, and Trigger Incident Response."
        elif "Brute Force" in threat_type:
            return "HIGH: Enforce Password Reset, Enable Step-Up MFA, and Add IP to Firewall Deny List."
        elif "Exfiltration" in threat_type:
            return "HIGH: Restrict S3/Cloud Storage Egress, Revoke API Access Tokens, and Review Audit Trail."
        elif "Privilege Escalation" in threat_type:
            return "HIGH: Temporarily Downgrade User Role Permissions & Notify Cloud Security Operations Center."
        else:
            return "MEDIUM: Flag account for 24-hour enhanced security surveillance."

    def save_alerts(self, alerts: list, filename="active_security_alerts.json"):
        save_path = ALERTS_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(alerts, f, indent=4)
        print(f"[AlertSystem] Generated and saved {len(alerts)} security alerts -> {save_path}")
        return save_path
