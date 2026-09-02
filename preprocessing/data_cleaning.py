import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import RAW_DATA_PATH, CLEANED_DATA_PATH, RAW_FEATURE_COLUMNS

class DataCleaner:
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by handling missing values, validating timestamps,
        and ensuring correct column data types.
        """
        df_clean = df.copy()

        # 1. Deduplicate
        initial_len = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        dropped_dups = initial_len - len(df_clean)
        if dropped_dups > 0:
            print(f"[Cleaner] Dropped {dropped_dups} duplicate rows.")

        # 2. Timestamp conversion
        if "timestamp" in df_clean.columns:
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
            # Impute invalid timestamps with modern fallback
            df_clean["timestamp"] = df_clean["timestamp"].fillna(pd.Timestamp.now())

        # 3. Handle missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    print(f"[Cleaner] Imputed missing values in '{col}' with median: {median_val}")
                else:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    print(f"[Cleaner] Imputed missing values in '{col}' with mode: {mode_val}")

        # 4. Type Enforcements
        numeric_cols = [
            "login_time_hour", "session_duration_min", "access_frequency",
            "failed_logins", "file_access_count", "sensitive_file_access_count",
            "data_transferred_mb", "unusual_location_flag", "unusual_device_flag"
        ]
        for num_col in numeric_cols:
            if num_col in df_clean.columns:
                df_clean[num_col] = pd.to_numeric(df_clean[num_col], errors="coerce").fillna(0)

        # Ensure login_time_hour bounded 0-23
        if "login_time_hour" in df_clean.columns:
            df_clean["login_time_hour"] = df_clean["login_time_hour"].astype(int) % 24

        return df_clean

    def clean_raw_file(self, raw_path=RAW_DATA_PATH, save_path=None) -> pd.DataFrame:
        df_raw = pd.read_csv(raw_path)
        df_clean = self.clean_data(df_raw)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(save_path, index=False)
            print(f"[Cleaner] Saved cleaned data to {save_path}")
        return df_clean

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.clean_raw_file()
