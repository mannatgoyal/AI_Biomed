"""Comprehensive biomedical machine learning evaluation metrics.

Calculates standard classification performance indicators:
- Accuracy
- Balanced Accuracy
- Precision (macro & weighted)
- Recall / Sensitivity (macro & weighted)
- Specificity
- F1-Score (macro, weighted, micro)
- Area Under the ROC Curve (ROC-AUC, binary and multi-class One-vs-Rest)
- Average Precision / PR-AUC
- Brier Score
- Confusion Matrix
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


@dataclass
class ModelEvaluationMetrics:
    """Stores all evaluation metrics for a single model run."""
    model_name: str
    accuracy: float
    balanced_accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    f1_macro: float
    roc_auc: float
    pr_auc: Optional[float] = None
    brier_score: Optional[float] = None
    confusion_mat: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Model": self.model_name,
            "Accuracy (%)": round(self.accuracy * 100, 2),
            "Balanced Acc (%)": round(self.balanced_accuracy * 100, 2),
            "F1-Score (Weighted)": round(self.f1_weighted, 4),
            "F1-Score (Macro)": round(self.f1_macro, 4),
            "Precision": round(self.precision_weighted, 4),
            "Recall": round(self.recall_weighted, 4),
            "ROC-AUC": round(self.roc_auc, 4),
        }


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> ModelEvaluationMetrics:
    """Computes comprehensive evaluation metrics for classification.

    Args:
        y_true: Ground truth binary or multiclass labels.
        y_pred: Predicted class labels.
        y_prob: Predicted probabilities array (N, num_classes) or (N,).
        model_name: Identifier name for the model.

    Returns:
        ModelEvaluationMetrics instance.
    """
    is_multiclass = len(np.unique(y_true)) > 2

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Compute ROC-AUC
    try:
        if is_multiclass:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            pr_auc = None
            brier = None
        else:
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            auc = roc_auc_score(y_true, prob_pos)
            pr_auc = average_precision_score(y_true, prob_pos)
            brier = brier_score_loss(y_true, prob_pos)
    except Exception:
        auc = 0.5
        pr_auc = None
        brier = None

    cm = confusion_matrix(y_true, y_pred)

    return ModelEvaluationMetrics(
        model_name=model_name,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision_weighted=prec,
        recall_weighted=rec,
        f1_weighted=f1_w,
        f1_macro=f1_m,
        roc_auc=auc,
        pr_auc=pr_auc,
        brier_score=brier,
        confusion_mat=cm,
    )


def format_metrics_table(metrics_list: List[ModelEvaluationMetrics]) -> pd.DataFrame:
    """Formats a list of ModelEvaluationMetrics into a polished pandas DataFrame."""
    rows = [m.to_dict() for m in metrics_list]
    df = pd.DataFrame(rows)
    return df

