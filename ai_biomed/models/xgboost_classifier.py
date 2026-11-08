"""Champion XGBoost Classifier for Multimodal CKD Risk Stratification.

Implements the high-performance gradient boosting decision tree architecture with regularized
objective optimization, tree subsampling, and class-weight balancing.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import xgboost as xgb

from ai_biomed.config import ModelConfig

logger = logging.getLogger(__name__)


class XGBoostBiomedClassifier:
    """XGBoost Classifier for multimodal clinical and BioBERT feature fusion."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        is_multiclass: bool = False,
        num_classes: int = 3,
        **kwargs
    ):
        self.config = config or ModelConfig()
        self.is_multiclass = is_multiclass
        self.num_classes = num_classes

        params: Dict[str, Any] = {
            "n_estimators": self.config.xgb_n_estimators,
            "max_depth": self.config.xgb_max_depth,
            "learning_rate": self.config.xgb_learning_rate,
            "subsample": self.config.xgb_subsample,
            "colsample_bytree": self.config.xgb_colsample_bytree,
            "gamma": self.config.xgb_gamma,
            "random_state": self.config.random_state,
            "n_jobs": -1,
        }

        if is_multiclass:
            params["objective"] = "multi:softprob"
            params["num_class"] = num_classes
            params["eval_metric"] = "mlogloss"
        else:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"

        params.update(kwargs)
        self.params = params
        self.model = xgb.XGBClassifier(**self.params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False
    ):
        """Fit the XGBoost classifier on training data."""
        logger.info(f"Training XGBoostBiomedClassifier (n_estimators={self.params['n_estimators']}, max_depth={self.params['max_depth']})...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def get_booster(self):
        return self.model.get_booster()

