"""Tests for machine learning models and baselines."""

import pytest
import numpy as np
from ai_biomed.models.baselines import LogisticRegressionBaseline, RandomForestBaseline, SVMBaseline
from ai_biomed.models.xgboost_classifier import XGBoostBiomedClassifier


def test_models_fit_and_predict():
    np.random.seed(42)
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(25, 20)

    models = [
        LogisticRegressionBaseline(max_iter=100),
        RandomForestBaseline(n_estimators=10),
        SVMBaseline(),
        XGBoostBiomedClassifier(n_estimators=10, max_depth=3)
    ]

    for model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        assert len(preds) == 25
        assert probs.shape == (25, 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
