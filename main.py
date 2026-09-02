import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config.config import RAW_DATA_PATH, CLEANED_DATA_PATH, RANDOM_SEED
from preprocessing.generate_data import generate_synthetic_cloud_logs
from preprocessing.data_cleaning import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from models.isolation_forest import IsolationForestModel
from models.random_forest import RandomForestModel
from models.autoencoder import AutoencoderModel
from detection.anomaly_detection import EnsembleAnomalyDetector
from detection.risk_scoring import RiskScorer
from explainable_ai.shap_explanation import SHAPExplainer
from explainable_ai.feature_importance import FeatureImportanceAnalyzer
from alerts.alert_system import AlertSystem
from evaluation.comparison import ModelComparator
from visualizations.graphs import GraphGenerator

def run_pipeline():
    print("=" * 80)
    print(" EXPLAINABLE AI-BASED USER BEHAVIOR ANALYTICS (UBA) FOR CLOUD SECURITY ")
    print("=" * 80)

    # Step 1: Ensure Raw Dataset
    if not RAW_DATA_PATH.exists():
        print("[Step 1] Raw dataset missing. Generating synthetic cloud activity data...")
        generate_synthetic_cloud_logs(num_records=6000, anomaly_ratio=0.08)
    else:
        print(f"[Step 1] Found raw activity logs: {RAW_DATA_PATH}")

    # Step 2: Data Cleaning
    print("\n[Step 2] Executing Data Cleaning...")
    cleaner = DataCleaner()
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_clean = cleaner.clean_data(df_raw)
    print(f"    - Cleaned records: {len(df_clean)}")

    # Step 3: Train / Test Split
    print("\n[Step 3] Splitting Data into Train (80%) and Test (20%)...")
    df_train, df_test = train_test_split(df_clean, test_size=0.20, random_state=RANDOM_SEED, stratify=df_clean["is_anomaly"])
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"    - Train set: {len(df_train)} samples | Test set: {len(df_test)} samples")

    # Step 4: Feature Engineering
    print("\n[Step 4] Fitting Preprocessing & Feature Engineering Pipeline...")
    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(df_train)
    X_test, y_test = fe.transform(df_test)
    fe.save()
    print(f"    - Engineered Feature Matrix shape: {X_train.shape}")

    # Step 5: Model Training
    print("\n[Step 5] Training AI/ML Detection Engine...")
    
    # 5a. Isolation Forest
    if_model = IsolationForestModel()
    if_model.fit(X_train)
    if_model.save()

    # 5b. Random Forest
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    rf_model.save()

    # 5c. Autoencoder
    ae_model = AutoencoderModel()
    ae_model.fit(X_train)
    ae_model.save()

    # Step 6: Ensemble Anomaly Detection & Risk Scoring
    print("\n[Step 6] Running Ensemble Anomaly Detection & Risk Scoring on Test Set...")
    ensemble_detector = EnsembleAnomalyDetector(if_model=if_model, rf_model=rf_model, ae_model=ae_model)
    detection_results = ensemble_detector.predict_detailed(X_test)
    
    risk_scorer = RiskScorer()
    df_test_scored = risk_scorer.process_dataframe(df_test, detection_results)
    
    # Save full processed dataset
    df_all_processed = risk_scorer.process_dataframe(df_clean, ensemble_detector.predict_detailed(fe.transform(df_clean)[0]))
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all_processed.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"    - Saved processed risk-scored data to {CLEANED_DATA_PATH}")

    # Step 7: Explainable AI (XAI) Engine
    print("\n[Step 7] Initializing Explainable AI (SHAP) Engine...")
    shap_explainer = SHAPExplainer(rf_model=rf_model)

    # Step 8: Security Alert Generation
    print("\n[Step 8] Generating Security Alerts with XAI Rationale...")
    alert_sys = AlertSystem(shap_explainer=shap_explainer)
    alerts = alert_sys.generate_alerts(df_test_scored, X_test)
    alert_sys.save_alerts(alerts)
    
    if len(alerts) > 0:
        print("\n--- SAMPLE GENERATED SECURITY ALERT ---")
        sample_alert = alerts[0]
        print(f"Alert ID:     {sample_alert['alert_id']}")
        print(f"User ID:      {sample_alert['user_id']} ({sample_alert['role']})")
        print(f"Risk Score:   {sample_alert['risk_score']} [{sample_alert['risk_level']}]")
        print(f"Threat Type:  {sample_alert['threat_type']}")
        print(f"XAI Rationale:{sample_alert['xai_explanation']}")
        print(f"Action:       {sample_alert['recommended_remediation']}")
        print("---------------------------------------")

    # Step 9: Model Performance & Comparative Benchmark
    print("\n[Step 9] Running Comparative Model Evaluation...")
    comparator = ModelComparator(if_model, rf_model, ae_model, ensemble_detector)
    df_comparison = comparator.compare_models(X_test, y_test)
    print("\n" + df_comparison.to_string(index=False))

    # Step 10: Visualizations & Plot Generation
    print("\n[Step 10] Generating Visualizations & Performance Plots...")
    graph_gen = GraphGenerator()
    graph_gen.plot_risk_distribution(df_test_scored)
    graph_gen.plot_confusion_matrix(y_test, df_test_scored["is_anomaly_pred"].values)
    
    models_dict = {
        "Isolation Forest": if_model,
        "Random Forest": rf_model,
        "Autoencoder": ae_model,
        "Weighted Ensemble": ensemble_detector
    }
    graph_gen.plot_roc_curves(models_dict, X_test, y_test)

    fi_analyzer = FeatureImportanceAnalyzer(rf_model, shap_explainer)
    s_imp = fi_analyzer.get_global_rf_importance()
    graph_gen.plot_feature_importances(s_imp)

    print("\n" + "=" * 80)
    print(" PIPELINE SUCCESSFULLY EXECUTED! All models, alerts & plots are ready.")
    print(" Launch the Security Dashboard using: streamlit run dashboard/app.py")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
