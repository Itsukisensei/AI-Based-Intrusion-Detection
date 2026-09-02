import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as px_go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.config import VISUALIZATIONS_DIR

class GraphGenerator:
    def __init__(self, output_dir=VISUALIZATIONS_DIR):
        self.output_dir = output_dir

    def plot_risk_distribution(self, df_results: pd.DataFrame, save_filename="risk_score_distribution.png"):
        """
        Plots histogram & KDE of calculated Risk Scores categorized by Risk Level.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        palette = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"}
        
        sns.histplot(
            data=df_results,
            x="risk_score",
            hue="risk_level",
            palette=palette,
            kde=True,
            bins=30,
            ax=ax
        )
        ax.set_title("Cloud User Activity Risk Score Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Risk Score (0 - 100)")
        ax.set_ylabel("User Activity Count")
        plt.tight_layout()
        
        filepath = self.output_dir / save_filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[GraphGenerator] Saved risk distribution plot to {filepath}")
        return filepath

    def plot_confusion_matrix(self, y_true, y_pred, model_name="Ensemble Model", save_filename="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Ground Truth Label")
        plt.tight_layout()

        filepath = self.output_dir / save_filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[GraphGenerator] Saved confusion matrix plot to {filepath}")
        return filepath

    def plot_roc_curves(self, models_dict, X_test, y_test, save_filename="roc_curves.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, model_obj in models_dict.items():
            if hasattr(model_obj, "predict_score"):
                y_scores = model_obj.predict_score(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
        ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = self.output_dir / save_filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[GraphGenerator] Saved ROC curves to {filepath}")
        return filepath

    def plot_feature_importances(self, s_importance: pd.Series, top_n=12, save_filename="feature_importance.png"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        top_s = s_importance.head(top_n).sort_values(ascending=True)
        
        ax.barh(top_s.index, top_s.values, color="#3498db")
        ax.set_title(f"Top {top_n} Explainable Feature Importance Ranks", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance Weight Score")
        plt.tight_layout()

        filepath = self.output_dir / save_filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[GraphGenerator] Saved feature importance plot to {filepath}")
        return filepath

    def create_plotly_activity_scatter(self, df_results: pd.DataFrame):
        """
        Creates interactive Plotly scatter plot of session login hour vs data transfer MB colored by risk.
        """
        fig = px.scatter(
            df_results,
            x="login_time_hour",
            y="data_transferred_mb",
            color="risk_level",
            size="risk_score",
            hover_data=["user_id", "role", "threat_type", "failed_logins", "sensitive_file_access_count"],
            color_discrete_map={"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"},
            title="Interactive Cloud User Activity & Risk Timeline",
            labels={"login_time_hour": "Login Time (Hour 0-23)", "data_transferred_mb": "Data Transferred (MB)"}
        )
        fig.update_layout(template="plotly_dark")
        return fig
