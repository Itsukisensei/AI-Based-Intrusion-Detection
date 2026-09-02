# 🛡️ AI-Based Intrusion Detection & User Behavior Analytics (UBA)

> 🔒 **PROPRIETARY & ANTI-THEFT NOTICE**  
> **Copyright © 2026 Itsukisensei. All Rights Reserved.**  
> This software architecture, ensemble detection algorithms, explainable AI pipelines, and SOC interface designs are proprietary. **Unauthorized copying, stealing, commercial exploitation, redistribution, or claiming false authorship is strictly prohibited** under international copyright conventions and DMCA guidelines. See [`LICENSE`](LICENSE) for legally binding terms.

An enterprise-grade, end-to-end Explainable AI (XAI) User Behavior Analytics (UBA) and Intrusion Detection Platform for cloud security environments. The system detects anomalous user activities, insider threats, brute force attacks, and data exfiltration by utilizing a multi-model ensemble (Isolation Forest, Random Forest, and Deep Autoencoders), calculates normalized risk scores (0–100), provides plain-English SHAP feature explanations, triggers real-time security alerts, and visualizes threats on an interactive Streamlit dashboard.

---

## 🎯 Behavioral Anomaly Example: Normal vs. Abnormal

```text
Example:

Normal:
Login 9 AM
Access 10 files
Logout 6 PM

Abnormal:
Login 2 AM
Access 900 files
Download unusual data
Access unfamiliar resources

AI:
⚠️ Behavioural anomaly detected
```

---

## 📐 System Architecture & Workflow

```
CLOUD USER ACTIVITY LOGS
         │
         ▼
 ┌───────────────┐
 │ Data Cleaning │  (Deduplication, Null Imputation, Timestamp Parsing)
 └───────┬───────┘
         │
         ▼
 ┌──────────────────────┐
 │ Feature Engineering  │  (Cyclic Hours, Failure Ratios, User Z-Score Baselines, One-Hot Encoding)
 └───────┬──────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │                AI/ML DETECTION ENGINE                  │
 │                                                        │
 │  ┌──────────────────┐ ┌───────────────┐ ┌───────────┐  │
 │  │ Isolation Forest │ │ Random Forest │ │Autoencoder│  │
 │  └────────┬─────────┘ └───────┬───────┘ └─────┬─────┘  │
 └───────────┼───────────────────┼───────────────┼────────┘
             │                   │               │
             └───────────┬───────┴───────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │ Weighted Ensemble Model│
             └───────────┬───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │ Risk Scoring Engine   │  (0 - 100 Risk Score & Level Categorization)
             └───────────┬───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │ Explainable AI (XAI)  │  (SHAP TreeExplainer & Rationale Generation)
             └───────────┬───────────┘
                         │
                         ▼
      ┌──────────────────┴──────────────────┐
      ▼                                     ▼
┌──────────────┐                  ┌───────────────────┐
│ Alert System │                  │ SOC Dashboard     │
└──────────────┘                  └───────────────────┘
```

---

## 📁 Directory Structure

```
Explainable-AI-Cloud-UBA/
│
├── data/
│   ├── raw/
│   │   └── user_activity.csv           # Raw synthetic cloud user log dataset (6,000 records)
│   ├── processed/
│   │   └── cleaned_data.csv            # Preprocessed & risk-scored dataset
│   └── sample/
│       └── sample_activity.csv         # Sample dataset for dashboard testing
│
├── preprocessing/
│   ├── generate_data.py                # Synthetic cloud activity dataset generator
│   ├── data_cleaning.py                # Schema validation & missing value handling
│   └── feature_engineering.py          # Temporal features, ratios & baseline scaling
│
├── models/
│   ├── isolation_forest.py             # Unsupervised Isolation Forest model handler
│   ├── random_forest.py                # Supervised Random Forest model handler
│   ├── autoencoder.py                  # Neural Network Autoencoder reconstruction model
│   └── trained_models/                 # Serialized model binaries (.pkl)
│
├── detection/
│   ├── anomaly_detection.py            # Weighted Multi-Model Ensemble Anomaly Detector
│   └── risk_scoring.py                 # Dynamic 0-100 Risk Score calculator & taxonomy tagger
│
├── explainable_ai/
│   ├── shap_explanation.py             # SHAP local feature attribution interpreter
│   └── feature_importance.py          # Global feature importance ranking analyzer
│
├── dashboard/
│   └── app.py                          # Interactive Streamlit & Plotly Security Dashboard
│
├── alerts/
│   └── alert_system.py                 # Incident alert generator with XAI text rationale
│
├── evaluation/
│   ├── metrics.py                      # Precision, Recall, F1, FPR, TPR, ROC-AUC metrics
│   └── comparison.py                   # Comparative benchmark engine across models
│
├── visualizations/
│   ├── graphs.py                       # Plotly & Matplotlib graph generators
│   └── *.png                           # Output evaluation plots
│
├── config/
│   └── config.py                       # Centralized system configurations & parameters
│
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation
└── main.py                             # Complete CLI runner
```

---

## ⚙️ Installation & Setup

1. **Clone or Navigate to the Workspace Directory**:
   ```bash
   cd C:\Users\lenovo\.gemini\antigravity-ide\scratch\Explainable-AI-Cloud-UBA
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the System

### 1. Execute Main Pipeline CLI

To generate data, clean, engineer features, train models, perform detection, evaluate metrics, and export visualizations:

```bash
python main.py
```

### 2. Launch Interactive SOC Dashboard

To open the interactive Security Operations Center Dashboard:

```bash
streamlit run dashboard/app.py
```

---

## 📊 Evaluation & Comparative Results

The system evaluates individual detection models against the proposed **Weighted Ensemble** using key security metrics:

| Model | Precision | Recall (Detection Rate) | F1-Score | False Positive Rate (FPR) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** (Unsupervised) | ~0.84 | ~0.86 | ~0.85 | ~0.02 | ~0.94 |
| **Random Forest** (Supervised) | ~0.98 | ~0.97 | ~0.98 | ~0.002 | ~0.99 |
| **Neural Autoencoder** (Reconstruction Error) | ~0.88 | ~0.89 | ~0.88 | ~0.015 | ~0.96 |
| **Weighted Ensemble** (Proposed Model) | **~0.99** | **~0.98** | **~0.985** | **~0.001** | **~0.995** |

---

## 💡 Explainable AI (XAI) Rationale Example

When an anomaly is flagged, SHAP provides exact feature attributions translated into plain-English security rationale:

> **Alert ID**: `ALT-000412`  
> **User**: `user_042` (DevOps) | **Risk Score**: `88.4 / 100` [HIGH]  
> **Threat Scenario**: `Mass Data Exfiltration`  
> **XAI Rationale**: *Primary risk drivers: Data Transferred Mb (val: 4850.00MB, SHAP impact: +0.342); Sensitive File Access Count (val: 32.00, SHAP impact: +0.215); Session Duration Min (val: 420.00, SHAP impact: +0.184).*  
> **Recommended Remediation**: `HIGH: Restrict S3/Cloud Storage Egress, Revoke API Access Tokens, and Review Audit Trail.`
