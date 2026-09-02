import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TRAINED_MODELS_DIR,
    CLEANED_DATA_PATH
)
from preprocessing.data_cleaning import DataCleaner

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.user_baselines = {}
        self.fitted = False
        self.encoded_cat_cols = []

    def engineer_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives cyclic time features, security ratios, and baseline metrics.
        """
        df_fe = df.copy()

        # 1. Cyclic hour features
        if "login_time_hour" in df_fe.columns:
            hours = df_fe["login_time_hour"].astype(float)
            df_fe["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            df_fe["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        # 2. Security Ratios
        df_fe["sensitive_file_ratio"] = df_fe["sensitive_file_access_count"] / (df_fe["file_access_count"] + 1.0)
        df_fe["failure_rate"] = df_fe["failed_logins"] / (df_fe["access_frequency"] + 1.0)
        df_fe["mb_per_minute"] = df_fe["data_transferred_mb"] / (df_fe["session_duration_min"] + 0.1)

        # 3. User Baseline Z-Scores
        if self.fitted:
            # Use cached baseline means/stds
            df_fe["user_zscore_data_transfer"] = df_fe.apply(
                lambda row: self._calc_zscore(row["user_id"], "data_transferred_mb", row["data_transferred_mb"]), axis=1
            )
            df_fe["user_zscore_session_duration"] = df_fe.apply(
                lambda row: self._calc_zscore(row["user_id"], "session_duration_min", row["session_duration_min"]), axis=1
            )
        else:
            # Calculate baseline statistics per user
            grouped = df_fe.groupby("user_id").agg({
                "data_transferred_mb": ["mean", "std"],
                "session_duration_min": ["mean", "std"]
            })
            
            for user, stats in grouped.iterrows():
                dt_mean = stats[("data_transferred_mb", "mean")]
                dt_std = stats[("data_transferred_mb", "std")]
                sd_mean = stats[("session_duration_min", "mean")]
                sd_std = stats[("session_duration_min", "std")]
                
                self.user_baselines[user] = {
                    "dt_mean": dt_mean if not np.isnan(dt_mean) else 100.0,
                    "dt_std": dt_std if not np.isnan(dt_std) and dt_std > 0 else 50.0,
                    "sd_mean": sd_mean if not np.isnan(sd_mean) else 60.0,
                    "sd_std": sd_std if not np.isnan(sd_std) and sd_std > 0 else 30.0,
                }
                
            df_fe["user_zscore_data_transfer"] = df_fe.apply(
                lambda row: self._calc_zscore(row["user_id"], "data_transferred_mb", row["data_transferred_mb"]), axis=1
            )
            df_fe["user_zscore_session_duration"] = df_fe.apply(
                lambda row: self._calc_zscore(row["user_id"], "session_duration_min", row["session_duration_min"]), axis=1
            )

        return df_fe

    def _calc_zscore(self, user_id, feature, val):
        baseline = self.user_baselines.get(user_id, {"dt_mean": 100.0, "dt_std": 50.0, "sd_mean": 60.0, "sd_std": 30.0})
        if feature == "data_transferred_mb":
            mean, std = baseline["dt_mean"], baseline["dt_std"]
        else:
            mean, std = baseline["sd_mean"], baseline["sd_std"]
        return float((val - mean) / (std if std > 0 else 1.0))

    def fit_transform(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        """
        Fits encoder, scaler, and user baselines on training dataset and returns scaled features X, y.
        """
        df_fe = self.engineer_raw_features(df)
        
        # Extract target if exists
        y = df_fe[TARGET_COLUMN] if TARGET_COLUMN in df_fe.columns else None

        # Fit Categorical Encoder
        cat_df = df_fe[CATEGORICAL_FEATURES]
        cat_encoded = self.encoder.fit_transform(cat_df)
        self.encoded_cat_cols = list(self.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=self.encoded_cat_cols, index=df_fe.index)

        # Extract Numerical Features
        num_df = df_fe[NUMERICAL_FEATURES]
        num_scaled = self.scaler.fit_transform(num_df)
        num_scaled_df = pd.DataFrame(num_scaled, columns=NUMERICAL_FEATURES, index=df_fe.index)

        # Combine
        X_processed = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
        self.fitted = True

        return X_processed, y

    def transform(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        """
        Transforms input DataFrame using previously fitted scaler and encoder.
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted with fit_transform before calling transform!")
            
        df_fe = self.engineer_raw_features(df)
        y = df_fe[TARGET_COLUMN] if TARGET_COLUMN in df_fe.columns else None

        # Transform Categorical
        cat_df = df_fe[CATEGORICAL_FEATURES]
        cat_encoded = self.encoder.transform(cat_df)
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=self.encoded_cat_cols, index=df_fe.index)

        # Transform Numerical
        num_df = df_fe[NUMERICAL_FEATURES]
        num_scaled = self.scaler.transform(num_df)
        num_scaled_df = pd.DataFrame(num_scaled, columns=NUMERICAL_FEATURES, index=df_fe.index)

        X_processed = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
        return X_processed, y

    def save(self, filepath=TRAINED_MODELS_DIR / "feature_engineer.pkl"):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[FeatureEngineer] Saved pipeline to {filepath}")

    @staticmethod
    def load(filepath=TRAINED_MODELS_DIR / "feature_engineer.pkl"):
        fe = joblib.load(filepath)
        print(f"[FeatureEngineer] Loaded pipeline from {filepath}")
        return fe

if __name__ == "__main__":
    from preprocessing.data_cleaning import DataCleaner
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_raw_file(save_path=CLEANED_DATA_PATH)
    
    fe = FeatureEngineer()
    X, y = fe.fit_transform(cleaned_df)
    fe.save()
    print(f"[+] Feature Engineering completed. Final feature matrix shape: {X.shape}")
