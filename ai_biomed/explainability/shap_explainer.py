"""Clinical Explainability and Feature Attribution using SHAP (SHapley Additive exPlanations).

Provides model interpretability for clinicians and researchers:
- Global feature attribution ranking
- SHAP summary plots
- Patient-level local explanation
- Identification of dominant clinical drivers (eGFR, Creatinine, BioBERT components)
- Compatibility patch for XGBoost 3.x bracketed base_score string formatting
"""

import io
import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import scipy.special
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import shap
import shap.explainers._tree as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility patch for SHAP TreeExplainer with XGBoost 3.x
# In XGBoost 3.x, learner_model_param['base_score'] can be serialized as '[5E-1]'
# ---------------------------------------------------------------------------
_orig_xgb_loader_init = st.XGBTreeModelLoader.__init__

def _patched_xgb_loader_init(self, xgb_model) -> None:
    import xgboost as xgb
    from shap.explainers._tree import _check_xgboost_version, decode_ubjson_buffer

    _check_xgboost_version(xgb.__version__)
    model: xgb.Booster = xgb_model

    raw = xgb_model.save_raw(raw_format="ubj")
    with io.BytesIO(raw) as fd:
        jmodel = decode_ubjson_buffer(fd)

    learner = jmodel["learner"]
    learner_model_param = learner["learner_model_param"]
    objective = learner["objective"]
    booster = learner["gradient_booster"]

    n_classes = max(int(learner_model_param["num_class"]), 1)
    n_targets = max(int(learner_model_param["num_target"]), 1)
    n_targets = max(n_targets, n_classes)

    if "gbtree" in booster and "model" not in booster:
        booster = booster["gbtree"]

    if booster["model"].get("iteration_indptr", None) is not None:
        iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
        diff = np.diff(iteration_indptr)
    else:
        n_parallel_trees = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
        diff = np.repeat(n_targets * n_parallel_trees, model.num_boosted_rounds())

    self.n_trees_per_iter = int(diff[0])
    self.n_targets = n_targets

    # Strip brackets from base_score if present
    raw_base = str(learner_model_param["base_score"]).strip("[]")
    self.base_score = float(raw_base)
    self.name_obj = objective["name"]
    self.name_gbm = booster["name"]

    base_score = float(raw_base)
    if self.name_obj in ("binary:logistic", "reg:logistic"):
        self.base_score = scipy.special.logit(base_score)
    elif self.name_obj in ("reg:gamma", "reg:tweedie", "count:poisson", "survival:cox", "survival:aft"):
        self.base_score = np.log(self.base_score)
    else:
        self.base_score = base_score

    self.num_feature = int(learner_model_param["num_feature"])
    self.num_class = int(learner_model_param["num_class"])

    trees = booster["model"]["trees"]
    self.num_trees = len(trees)

    self.node_parents = []
    self.node_cleft = []
    self.node_cright = []
    self.node_sindex = []
    self.children_default = []
    self.sum_hess = []
    self.values = []
    self.thresholds = []
    self.threshold_types = []
    self.features = []
    self.split_types = []
    self.categories = []

    feature_types = model.feature_types
    if feature_types is not None:
        cat_feature_indices = np.where(np.asarray(feature_types) == "c")[0]
        self.cat_feature_indices = cat_feature_indices if len(cat_feature_indices) > 0 else None
    else:
        self.cat_feature_indices = None

    def to_integers(data: list[int]) -> np.ndarray:
        assert isinstance(data, list)
        return np.asanyarray(data, dtype=np.uint8)

    for i in range(self.num_trees):
        tree = trees[i]
        self.node_parents.append(np.asarray(tree["parents"]))
        self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
        self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
        self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

        base_weight = np.asarray(tree["base_weights"], dtype=np.float32)
        if base_weight.size != self.node_cleft[-1].size:
            raise ValueError("vector-leaf is not yet supported.")

        default_left = to_integers(tree["default_left"])
        default_child = np.where(default_left == 1, self.node_cleft[-1], self.node_cright[-1]).astype(np.int64)
        self.children_default.append(default_child)
        self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

        is_leaf = self.node_cleft[-1] == -1
        split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
        leaf_weight = np.where(is_leaf, split_cond, 0.0)
        thresholds = np.where(is_leaf, 0.0, split_cond)
        thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))
        threshold_types = np.zeros_like(thresholds, dtype=np.int32)

        self.values.append(leaf_weight.reshape(leaf_weight.size, 1))
        self.thresholds.append(thresholds)
        self.threshold_types.append(threshold_types)

        self.features.append(np.asarray(tree["split_indices"], dtype=np.int64))
        self.split_types.append(to_integers(tree["split_type"]))
        self.categories.append(self.parse_categories(tree["categories_nodes"], tree["categories_segments"], tree["categories_sizes"], tree["categories"], self.node_cleft[-1]))

st.XGBTreeModelLoader.__init__ = _patched_xgb_loader_init


class BiomedSHAPExplainer:
    """SHAP Explainer tailored for biomedical tree-based models (XGBoost)."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.sample_X = None

    def explain(self, X: np.ndarray, max_evals: int = 250) -> np.ndarray:
        """Computes SHAP values using TreeExplainer.

        Args:
            X: Input feature array (N, num_features).
            max_evals: Maximum samples for evaluation.

        Returns:
            np.ndarray of SHAP values.
        """
        logger.info("Computing SHAP values using TreeExplainer...")
        booster = getattr(self.model, "model", self.model)
        self.explainer = shap.TreeExplainer(booster)
        
        sample_X = X[:max_evals] if len(X) > max_evals else X
        self.shap_values = self.explainer.shap_values(sample_X)
        self.sample_X = sample_X
        return self.shap_values

    def get_feature_importance_df(self, top_n: int = 20) -> pd.DataFrame:
        """Returns a DataFrame ranking the top features by mean absolute SHAP value."""
        if self.shap_values is None:
            raise ValueError("Must call explain() before get_feature_importance_df()")

        # If multiclass or list of arrays
        if isinstance(self.shap_values, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        elif self.shap_values.ndim == 3:
            mean_abs = np.abs(self.shap_values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(self.shap_values).mean(axis=0)

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
    ):
        """Generates and saves a SHAP summary bar plot."""
        if self.shap_values is None:
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
        logger.info(f"Saved SHAP summary plot to: {out}")
