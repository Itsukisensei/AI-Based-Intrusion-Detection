import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, average_precision_score
)
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

class EvaluationMetrics:
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
        """
        Computes precision, recall, F1, detection rate (TPR), false positive rate (FPR), ROC-AUC, and PR-AUC.
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0) # Detection Rate
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            fpr = 0.0
            tpr = recall

        metrics = {
            "Precision": round(float(precision), 4),
            "Recall (Detection Rate)": round(float(recall), 4),
            "F1-Score": round(float(f1), 4),
            "False Positive Rate (FPR)": round(float(fpr), 4),
            "True Positive Rate (TPR)": round(float(tpr), 4),
        }

        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
                metrics["ROC-AUC"] = round(float(auc), 4)
                metrics["PR-AUC"] = round(float(pr_auc), 4)
            except Exception:
                metrics["ROC-AUC"] = 0.5
                metrics["PR-AUC"] = 0.5

        return metrics
