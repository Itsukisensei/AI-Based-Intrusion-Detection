import time
import random
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import TRAINED_MODELS_DIR, PROCESSED_DATA_DIR, ALERTS_DIR
from preprocessing.feature_engineering import FeatureEngineer
from models.isolation_forest import IsolationForestModel
from models.random_forest import RandomForestModel
from models.autoencoder import AutoencoderModel
from detection.anomaly_detection import EnsembleAnomalyDetector
from detection.risk_scoring import RiskScorer
from explainable_ai.shap_explanation import SHAPExplainer
from alerts.alert_system import AlertSystem

LIVE_STREAM_LOG_PATH = PROCESSED_DATA_DIR / "live_stream_log.csv"
LIVE_ALERTS_PATH = ALERTS_DIR / "live_alerts.json"

class RealtimeCloudStreamSimulator:
    """
    Simulates a live streaming cloud log environment.
    Continuously ingests user activity, computes feature transformations,
    evaluates multi-model ensemble anomaly detection, computes risk scores,
    and dispatches SHAP-explained security alerts in real-time.
    """
    def __init__(self):
        print("[RealtimeStream] Loading model artifacts...")
        self.fe = FeatureEngineer.load(TRAINED_MODELS_DIR / "feature_engineer.pkl")
        self.if_model = IsolationForestModel.load(TRAINED_MODELS_DIR / "isolation_forest.pkl")
        self.rf_model = RandomForestModel.load(TRAINED_MODELS_DIR / "random_forest.pkl")
        self.ae_model = AutoencoderModel.load(TRAINED_MODELS_DIR / "autoencoder.pkl")
        self.ensemble = EnsembleAnomalyDetector(
            if_model=self.if_model, rf_model=self.rf_model, ae_model=self.ae_model
        )
        self.risk_scorer = RiskScorer()
        self.shap_explainer = SHAPExplainer(rf_model=self.rf_model)
        self.alert_system = AlertSystem(shap_explainer=self.shap_explainer)
        
        self.users = [f"user_{i:03d}" for i in range(1, 101)]
        self.roles = ["Developer", "Analyst", "DevOps", "Admin", "HR"]
        self.locations = ["US-East", "US-West", "EU-Central", "AP-South"]
        self.unusual_locations = ["CN-Beijing", "RU-Moscow", "KP-Pyongyang"]

    def generate_live_event(self) -> dict:
        """
        Generates a single live cloud activity log event (with ~15% chance of security anomaly burst).
        """
        now = datetime.now()
        is_attack = random.random() < 0.15

        if is_attack:
            scenario = random.choice(["brute_force", "exfiltration", "off_hours", "recon"])
            if scenario == "brute_force":
                return {
                    "timestamp": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                    "user_id": random.choice(self.users),
                    "role": random.choice(self.roles),
                    "ip_address": f"185.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}",
                    "location": random.choice(self.unusual_locations),
                    "login_time_hour": now.hour,
                    "session_duration_min": round(random.uniform(1.0, 5.0), 2),
                    "access_frequency": random.randint(40, 90),
                    "failed_logins": random.randint(8, 22),
                    "file_access_count": random.randint(0, 5),
                    "sensitive_file_access_count": random.randint(0, 1),
                    "data_transferred_mb": round(random.uniform(0.1, 5.0), 2),
                    "unusual_location_flag": 1,
                    "unusual_device_flag": 1,
                    "is_anomaly": 1
                }
            elif scenario == "exfiltration":
                return {
                    "timestamp": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                    "user_id": random.choice(self.users),
                    "role": random.choice(["Developer", "Analyst", "DevOps"]),
                    "ip_address": f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
                    "location": random.choice(self.locations),
                    "login_time_hour": now.hour,
                    "session_duration_min": round(random.uniform(180.0, 500.0), 2),
                    "access_frequency": random.randint(60, 150),
                    "failed_logins": random.randint(0, 2),
                    "file_access_count": random.randint(100, 350),
                    "sensitive_file_access_count": random.randint(25, 80),
                    "data_transferred_mb": round(random.uniform(3500.0, 15000.0), 2),
                    "unusual_location_flag": random.choice([0, 1]),
                    "unusual_device_flag": random.choice([0, 1]),
                    "is_anomaly": 1
                }
            else:
                return {
                    "timestamp": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                    "user_id": random.choice(self.users),
                    "role": random.choice(["Admin", "DevOps"]),
                    "ip_address": f"91.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}",
                    "location": random.choice(self.unusual_locations),
                    "login_time_hour": 3,
                    "session_duration_min": round(random.uniform(120.0, 350.0), 2),
                    "access_frequency": random.randint(50, 110),
                    "failed_logins": random.randint(2, 6),
                    "file_access_count": random.randint(40, 150),
                    "sensitive_file_access_count": random.randint(12, 45),
                    "data_transferred_mb": round(random.uniform(1200.0, 4500.0), 2),
                    "unusual_location_flag": 1,
                    "unusual_device_flag": 1,
                    "is_anomaly": 1
                }
        else:
            # Normal log event
            return {
                "timestamp": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "user_id": random.choice(self.users),
                "role": random.choice(self.roles),
                "ip_address": f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                "location": random.choice(self.locations),
                "login_time_hour": now.hour,
                "session_duration_min": round(random.uniform(10.0, 120.0), 2),
                "access_frequency": random.randint(5, 25),
                "failed_logins": random.choice([0, 0, 0, 1]),
                "file_access_count": random.randint(5, 30),
                "sensitive_file_access_count": random.randint(0, 3),
                "data_transferred_mb": round(random.uniform(10.0, 250.0), 2),
                "unusual_location_flag": 0,
                "unusual_device_flag": 0,
                "is_anomaly": 0
            }

    def process_single_event(self, event_dict: dict) -> (dict, dict):
        """
        Evaluates a single live incoming event through the AI pipeline.
        """
        df_single = pd.DataFrame([event_dict])
        df_single["timestamp"] = pd.to_datetime(df_single["timestamp"])
        
        # Transform features
        X_mat, _ = self.fe.transform(df_single)
        
        # Predict ensemble score
        det_res = self.ensemble.predict_detailed(X_mat)
        
        # Calculate Risk Score & Threat Taxonomy
        df_scored = self.risk_scorer.process_dataframe(df_single, det_res)
        scored_dict = df_scored.iloc[0].to_dict()
        scored_dict["timestamp"] = str(scored_dict["timestamp"])

        # Check for Alert Trigger (Risk Score >= 70)
        alert_obj = None
        if float(scored_dict["risk_score"]) >= 70.0:
            alerts = self.alert_system.generate_alerts(df_scored, X_mat)
            if alerts:
                alert_obj = alerts[0]
                alert_obj["alert_id"] = f"ALT-{random.randint(100000, 999999)}"
                alert_obj["timestamp"] = scored_dict["timestamp"]

        return scored_dict, alert_obj

    def start_streaming(self, interval_sec=1.5, max_events=50000):
        print(f"[RealtimeStream] Starting real-time cloud security stream (Interval: {interval_sec}s)... Press Ctrl+C to stop.")
        
        # Initialize live storage files if needed
        if not LIVE_STREAM_LOG_PATH.exists():
            df_empty = pd.DataFrame()
            df_empty.to_csv(LIVE_STREAM_LOG_PATH, index=False)
            
        if not LIVE_ALERTS_PATH.exists():
            with open(LIVE_ALERTS_PATH, "w") as f:
                json.dump([], f)

        events_count = 0
        try:
            while events_count < max_events:
                event_raw = self.generate_live_event()
                processed_event, alert_obj = self.process_single_event(event_raw)
                
                # Append to live log CSV with header preservation and 200 row cap
                df_event = pd.DataFrame([processed_event])
                if LIVE_STREAM_LOG_PATH.exists() and LIVE_STREAM_LOG_PATH.stat().st_size > 0:
                    try:
                        df_existing = pd.read_csv(LIVE_STREAM_LOG_PATH)
                        df_combined = pd.concat([df_existing, df_event], ignore_index=True)
                        df_combined.tail(2000).to_csv(LIVE_STREAM_LOG_PATH, index=False)
                    except Exception:
                        df_event.to_csv(LIVE_STREAM_LOG_PATH, index=False)
                else:
                    df_event.to_csv(LIVE_STREAM_LOG_PATH, index=False)

                # Append alert if triggered
                if alert_obj:
                    live_alerts = []
                    if LIVE_ALERTS_PATH.exists() and LIVE_ALERTS_PATH.stat().st_size > 0:
                        with open(LIVE_ALERTS_PATH, "r") as f:
                            try:
                                live_alerts = json.load(f)
                            except Exception:
                                live_alerts = []
                    live_alerts.insert(0, alert_obj) # prepend newest
                    with open(LIVE_ALERTS_PATH, "w") as f:
                        json.dump(live_alerts[:50], f, indent=4) # keep top 50

                print(f"[{processed_event['timestamp']}] Event #{events_count+1:04d} | User: {processed_event['user_id']} | Risk: {processed_event['risk_score']:.1f} [{processed_event['risk_level']}] | Threat: {processed_event['threat_type']}")
                if alert_obj:
                    print(f"  [ALERT TRIGGERED]: {alert_obj['alert_id']} -> {alert_obj['xai_explanation']}")

                events_count += 1
                time.sleep(interval_sec)

        except KeyboardInterrupt:
            print("[RealtimeStream] Live stream stopped by user.")

if __name__ == "__main__":
    streamer = RealtimeCloudStreamSimulator()
    streamer.start_streaming(interval_sec=2.0, max_events=50000)
