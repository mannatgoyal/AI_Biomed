"""Tests for evaluation metrics and reporting."""

import pytest
import numpy as np
from ai_biomed.evaluation.metrics import compute_evaluation_metrics, format_metrics_table


def test_compute_evaluation_metrics_binary():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    y_prob = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.6, 0.4],
        [0.7, 0.3], [0.1, 0.9], [0.85, 0.15], [0.05, 0.95], [0.15, 0.85]
    ])

    metrics = compute_evaluation_metrics(y_true, y_pred, y_prob, model_name="TestModel")
    assert metrics.accuracy == 0.9
    assert metrics.roc_auc > 0.8
    assert metrics.f1_weighted > 0.8
    assert metrics.confusion_mat.shape == (2, 2)

    df = format_metrics_table([metrics])
    assert "Accuracy (%)" in df.columns
    assert "ROC-AUC" in df.columns
    assert len(df) == 1
