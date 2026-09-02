import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import RAW_DATA_PATH, SAMPLE_DATA_PATH, RANDOM_SEED

def generate_synthetic_cloud_logs(num_records=6000, anomaly_ratio=0.08, random_state=RANDOM_SEED):
    np.random.seed(random_state)
    
    num_anomalies = int(num_records * anomaly_ratio)
    num_normals = num_records - num_anomalies
    
    users = [f"user_{i:03d}" for i in range(1, 101)]
    roles = ["Developer", "Analyst", "DevOps", "Admin", "HR", "Sales"]
    locations = ["US-East", "US-West", "EU-Central", "AP-South", "UK-London", "DE-Frankfurt"]
    unusual_locations = ["CN-Beijing", "RU-Moscow", "BR-SaoPaulo", "KP-Pyongyang"]
    
    # -----------------------
    # 1. NORMAL USER ACTIVITY
    # -----------------------
    start_date = datetime(2026, 8, 1, 8, 0, 0)
    timestamps_normal = [
        start_date + timedelta(
            days=int(np.random.randint(0, 30)),
            hours=int(np.random.normal(13, 3)) % 24, # centered around daytime 1 PM
            minutes=int(np.random.randint(0, 60)),
            seconds=int(np.random.randint(0, 60))
        ) for _ in range(num_normals)
    ]
    
    user_id_normal = np.random.choice(users, size=num_normals)
    role_normal = np.random.choice(roles, size=num_normals, p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
    ip_normal = [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(num_normals)]
    location_normal = np.random.choice(locations, size=num_normals)
    
    session_dur_normal = np.random.gamma(shape=3.0, scale=30.0, size=num_normals) # avg ~90 min
    access_freq_normal = np.random.poisson(lam=12, size=num_normals)
    failed_logins_normal = np.random.choice([0, 1, 2], size=num_normals, p=[0.88, 0.10, 0.02])
    file_access_normal = np.random.poisson(lam=18, size=num_normals)
    sensitive_file_normal = np.random.binomial(n=file_access_normal, p=0.08)
    data_transferred_normal = np.random.exponential(scale=120.0, size=num_normals) # avg ~120 MB
    unusual_loc_flag_normal = np.random.choice([0, 1], size=num_normals, p=[0.98, 0.02])
    unusual_dev_flag_normal = np.random.choice([0, 1], size=num_normals, p=[0.96, 0.04])
    
    df_normal = pd.DataFrame({
        "timestamp": timestamps_normal,
        "user_id": user_id_normal,
        "role": role_normal,
        "ip_address": ip_normal,
        "location": location_normal,
        "login_time_hour": [t.hour for t in timestamps_normal],
        "session_duration_min": np.round(np.clip(session_dur_normal, 5, 480), 2),
        "access_frequency": np.clip(access_freq_normal, 1, 60),
        "failed_logins": failed_logins_normal,
        "file_access_count": np.clip(file_access_normal, 0, 100),
        "sensitive_file_access_count": sensitive_file_normal,
        "data_transferred_mb": np.round(np.clip(data_transferred_normal, 1, 800), 2),
        "unusual_location_flag": unusual_loc_flag_normal,
        "unusual_device_flag": unusual_dev_flag_normal,
        "is_anomaly": 0
    })

    # ---------------------------
    # 2. ANOMALOUS THREAT ACTIVITY
    # ---------------------------
    # Split anomalies across 4 security threat scenarios
    n1 = int(num_anomalies * 0.30) # Brute force
    n2 = int(num_anomalies * 0.30) # Mass exfiltration
    n3 = int(num_anomalies * 0.20) # Off-hours privilege escalation
    n4 = num_anomalies - (n1 + n2 + n3) # Reconnaissance / Insider threat

    # Scenario 1: Brute Force Login
    timestamps_a1 = [start_date + timedelta(days=int(np.random.randint(0, 30)), hours=int(np.random.randint(0, 24)), minutes=int(np.random.randint(0, 60))) for _ in range(n1)]
    df_a1 = pd.DataFrame({
        "timestamp": timestamps_a1,
        "user_id": np.random.choice(users, size=n1),
        "role": np.random.choice(roles, size=n1),
        "ip_address": [f"185.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n1)],
        "location": np.random.choice(unusual_locations, size=n1),
        "login_time_hour": [t.hour for t in timestamps_a1],
        "session_duration_min": np.random.uniform(1.0, 8.0, size=n1),
        "access_frequency": np.random.randint(30, 90, size=n1),
        "failed_logins": np.random.randint(8, 25, size=n1),
        "file_access_count": np.random.randint(0, 5, size=n1),
        "sensitive_file_access_count": np.random.randint(0, 2, size=n1),
        "data_transferred_mb": np.random.uniform(0.1, 10.0, size=n1),
        "unusual_location_flag": 1,
        "unusual_device_flag": 1,
        "is_anomaly": 1
    })

    # Scenario 2: Data Exfiltration
    timestamps_a2 = [start_date + timedelta(days=int(np.random.randint(0, 30)), hours=int(np.random.randint(1, 5)), minutes=int(np.random.randint(0, 60))) for _ in range(n2)]
    file_access_a2 = np.random.randint(80, 300, size=n2)
    df_a2 = pd.DataFrame({
        "timestamp": timestamps_a2,
        "user_id": np.random.choice(users, size=n2),
        "role": np.random.choice(["Developer", "Analyst", "DevOps"], size=n2),
        "ip_address": [f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n2)],
        "location": np.random.choice(locations, size=n2),
        "login_time_hour": [t.hour for t in timestamps_a2],
        "session_duration_min": np.random.uniform(180, 600, size=n2),
        "access_frequency": np.random.randint(50, 150, size=n2),
        "failed_logins": np.random.randint(0, 3, size=n2),
        "file_access_count": file_access_a2,
        "sensitive_file_access_count": np.random.randint(20, 90, size=n2),
        "data_transferred_mb": np.round(np.random.uniform(2500.0, 18000.0, size=n2), 2),
        "unusual_location_flag": np.random.choice([0, 1], size=n2, p=[0.3, 0.7]),
        "unusual_device_flag": np.random.choice([0, 1], size=n2, p=[0.2, 0.8]),
        "is_anomaly": 1
    })

    # Scenario 3: Off-Hours Admin Access & Unusual Location
    timestamps_a3 = [start_date + timedelta(days=int(np.random.randint(0, 30)), hours=int(np.random.choice([1, 2, 3, 4])), minutes=int(np.random.randint(0, 60))) for _ in range(n3)]
    df_a3 = pd.DataFrame({
        "timestamp": timestamps_a3,
        "user_id": np.random.choice(users, size=n3),
        "role": np.random.choice(["Admin", "DevOps"], size=n3),
        "ip_address": [f"91.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n3)],
        "location": np.random.choice(unusual_locations, size=n3),
        "login_time_hour": [t.hour for t in timestamps_a3],
        "session_duration_min": np.random.uniform(120, 450, size=n3),
        "access_frequency": np.random.randint(40, 100, size=n3),
        "failed_logins": np.random.randint(1, 5, size=n3),
        "file_access_count": np.random.randint(30, 120, size=n3),
        "sensitive_file_access_count": np.random.randint(10, 40, size=n3),
        "data_transferred_mb": np.round(np.random.uniform(800.0, 4000.0, size=n3), 2),
        "unusual_location_flag": 1,
        "unusual_device_flag": 1,
        "is_anomaly": 1
    })

    # Scenario 4: Insider Threat Reconnaissance
    timestamps_a4 = [start_date + timedelta(days=int(np.random.randint(0, 30)), hours=int(np.random.randint(8, 20)), minutes=int(np.random.randint(0, 60))) for _ in range(n4)]
    file_access_a4 = np.random.randint(120, 400, size=n4)
    df_a4 = pd.DataFrame({
        "timestamp": timestamps_a4,
        "user_id": np.random.choice(users, size=n4),
        "role": np.random.choice(roles, size=n4),
        "ip_address": [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n4)],
        "location": np.random.choice(locations, size=n4),
        "login_time_hour": [t.hour for t in timestamps_a4],
        "session_duration_min": np.random.uniform(60, 300, size=n4),
        "access_frequency": np.random.randint(80, 250, size=n4),
        "failed_logins": np.random.randint(0, 2, size=n4),
        "file_access_count": file_access_a4,
        "sensitive_file_access_count": np.random.randint(15, 60, size=n4),
        "data_transferred_mb": np.round(np.random.uniform(400.0, 2000.0, size=n4), 2),
        "unusual_location_flag": 0,
        "unusual_device_flag": np.random.choice([0, 1], size=n4, p=[0.7, 0.3]),
        "is_anomaly": 1
    })

    # Concatenate & Shuffle
    df_all = pd.concat([df_normal, df_a1, df_a2, df_a3, df_a4], ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Save raw data
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(RAW_DATA_PATH, index=False)
    print(f"[+] Successfully generated synthetic cloud activity data: {len(df_all)} records -> {RAW_DATA_PATH}")
    print(f"    - Normal records: {sum(df_all['is_anomaly'] == 0)}")
    print(f"    - Anomalous records: {sum(df_all['is_anomaly'] == 1)}")

    # Save sample subset
    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df = df_all.head(200)
    sample_df.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"[+] Saved sample dataset: {len(sample_df)} records -> {SAMPLE_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_cloud_logs()
