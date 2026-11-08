"""Stratified K-Fold Cross Validation Runner.

Executes rigorous K-fold cross validation across all models to ensure statistical reliability
and guard against split-induced overfitting.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ai_biomed.evaluation.metrics import compute_evaluation_metrics, ModelEvaluationMetrics

logger = logging.getLogger(__name__)


def run_stratified_cv(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """Executes stratified K-fold cross validation on a collection of models.

    Args:
        models: Dictionary mapping model names to model instances with fit/predict/predict_proba.
        X: Feature matrix.
        y: Target label vector.
        n_splits: Number of CV folds (default: 5).
        random_state: Random state seed.

    Returns:
        pd.DataFrame containing mean and standard deviation of all metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {name: [] for name in models}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        for name, model_cls in models.items():
            # Clone or re-instantiate if needed
            model = model_cls
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)
            metrics = compute_evaluation_metrics(y_val, preds, probs, model_name=name)
            results[name].append(metrics)

    summary_rows = []
    for name, fold_metrics in results.items():
        accs = [m.accuracy * 100 for m in fold_metrics]
        f1s = [m.f1_weighted for m in fold_metrics]
        aucs = [m.roc_auc for m in fold_metrics]

        summary_rows.append({
            "Model": name,
            "CV Accuracy (Mean +/- Std)": f"{np.mean(accs):.2f}% +/- {np.std(accs):.2f}%",
            "CV F1-Score (Weighted)": f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
            "CV ROC-AUC": f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}",
        })

    return pd.DataFrame(summary_rows)

