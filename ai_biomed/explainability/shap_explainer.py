"""Clinical Explainability and Feature Attribution using SHAP (SHapley Additive exPlanations).

Provides model interpretability for clinicians and researchers:
- Global feature attribution ranking
- SHAP summary plots
- Patient-level local explanation
- Identification of dominant clinical drivers (eGFR, Creatinine, BioBERT components)
- Clean compatibility patch for XGBoost 3.x base_score format
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import shap
import shap.explainers._tree as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clean compatibility patch for SHAP TreeExplainer with XGBoost 3.x
# In XGBoost 3.x, learner_model_param['base_score'] can be serialized as '[5E-1]'
# Wrapping decode_ubjson_buffer cleanly strips the brackets before float parsing.
# ---------------------------------------------------------------------------
_orig_decode = st.decode_ubjson_buffer


def _safe_decode_ubjson_buffer(*args: Any, **kwargs: Any) -> Any:
    result = _orig_decode(*args, **kwargs)
    try:
        if isinstance(result, dict) and "learner" in result:
            learner = result["learner"]
            if isinstance(learner, dict) and "learner_model_param" in learner:
                param = learner["learner_model_param"]
                if isinstance(param, dict) and "base_score" in param:
                    raw_val = param["base_score"]
                    if isinstance(raw_val, bytes):
                        param["base_score"] = raw_val.decode("utf-8").strip("[]")
                    elif isinstance(raw_val, str):
                        param["base_score"] = raw_val.strip("[]")
    except Exception as exc:
        logger.debug("SHAP buffer decode patch note: %s", exc)
    return result


setattr(st, "decode_ubjson_buffer", _safe_decode_ubjson_buffer)


class BiomedSHAPExplainer:
    """SHAP Explainer tailored for biomedical tree-based models (XGBoost)."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model: Any = model
        self.feature_names: List[str] = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[Union[np.ndarray, List[np.ndarray]]] = None
        self.sample_X: Optional[np.ndarray] = None

    def explain(self, X: np.ndarray, max_evals: int = 250) -> Union[np.ndarray, List[np.ndarray]]:
        """Computes SHAP values using TreeExplainer.

        Args:
            X: Input feature array (N, num_features).
            max_evals: Maximum samples for evaluation.

        Returns:
            np.ndarray or list of np.ndarray containing SHAP values.
        """
        logger.info("Computing SHAP values using TreeExplainer...")
        booster = getattr(self.model, "model", self.model)
        self.explainer = shap.TreeExplainer(booster)

        sample_X = X[:max_evals] if len(X) > max_evals else X
        shap_vals = self.explainer.shap_values(sample_X)
        self.shap_values = shap_vals
        self.sample_X = sample_X
        return shap_vals

    def get_feature_importance_df(self, top_n: int = 20) -> pd.DataFrame:
        """Returns a DataFrame ranking the top features by mean absolute SHAP value."""
        if self.shap_values is None:
            raise ValueError("Must call explain() before get_feature_importance_df()")

        sv = self.shap_values
        if isinstance(sv, list):
            mean_abs = np.mean([np.abs(arr).mean(axis=0) for arr in sv], axis=0)
        elif getattr(sv, "ndim", 2) == 3:
            mean_abs = np.abs(sv).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(sv).mean(axis=0)

        df = pd.DataFrame({
            "Feature": self.feature_names[:len(mean_abs)],
            "Mean_Absolute_SHAP": mean_abs
        }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

        return df.head(top_n)

    def plot_summary(
        self,
        output_path: Union[str, Path],
        max_display: int = 15,
        title: str = "SHAP Feature Importance (AI_Biomed Multimodal Fusion)"
    ) -> None:
        """Generates and saves a SHAP summary bar plot."""
        if self.shap_values is None or self.sample_X is None:
            raise ValueError("Must call explain() before plot_summary()")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        sv = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values

        shap.summary_plot(
            sv,
            self.sample_X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(title, fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary plot to: %s", out)
