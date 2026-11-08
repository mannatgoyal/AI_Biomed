"""Baseline Machine Learning Classifiers.

Implements baseline classifiers for comparative research evaluation:
- Logistic Regression (with balanced class weighting)
- Random Forest Classifier
- Support Vector Machine (RBF kernel with Platt scaling)
- Gradient Boosting Classifier
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from ai_biomed.config import ModelConfig

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """Logistic Regression baseline classifier."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        cfg = config or ModelConfig()
        params = {
            "max_iter": cfg.lr_max_iter,
            "class_weight": cfg.lr_class_weight,
            "random_state": cfg.random_state,
        }
        params.update(kwargs)
        self.model = LogisticRegression(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestBaseline:
    """Random Forest baseline classifier."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        cfg = config or ModelConfig()
        params = {
            "n_estimators": cfg.rf_n_estimators,
            "max_depth": cfg.rf_max_depth,
            "class_weight": cfg.rf_class_weight,
            "random_state": cfg.random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)
        self.model = RandomForestClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


class SVMBaseline:
    """Support Vector Machine classifier with RBF kernel and probability calibration."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        cfg = config or ModelConfig()
        params = {
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced",
            "random_state": cfg.random_state,
        }
        params.update(kwargs)
        self.model = SVC(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class GradientBoostingBaseline:
    """Gradient Boosting classifier."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        cfg = config or ModelConfig()
        params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": cfg.random_state,
        }
        params.update(kwargs)
        self.model = GradientBoostingClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class BaselineModelSuite:
    """Instantiates and evaluates all standard baseline models in a single interface."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = {
            "Logistic Regression": LogisticRegressionBaseline(self.config),
            "Random Forest": RandomForestBaseline(self.config),
            "Support Vector Machine": SVMBaseline(self.config),
            "Gradient Boosting": GradientBoostingBaseline(self.config),
        }

    def fit_all(self, X_train: np.ndarray, y_train: np.ndarray):
        for name, model in self.models.items():
            logger.info(f"Training baseline: {name}...")
            model.fit(X_train, y_train)
        return self
