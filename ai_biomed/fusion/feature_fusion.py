"""Multimodal Feature Fusion Engine.

Integrates structured clinical features (37 variables) with unstructured biomedical text
representations (768-dimensional BioBERT embeddings) via early and intermediate feature fusion.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class FusedDataset:
    """Container holding aligned multimodal fused feature matrices and metadata."""
    X_fused: np.ndarray
    y: np.ndarray
    clinical_dim: int
    text_dim: int
    total_dim: int
    clinical_feature_names: List[str]
    fused_feature_names: List[str]


class MultimodalFusionEngine:
    """Concatenates and aligns structured clinical features with BioBERT contextual embeddings."""

    def __init__(self, use_pca_projection: bool = False, pca_components: int = 16):
        self.use_pca_projection = use_pca_projection
        self.pca_components = pca_components
        self.pca: Optional[PCA] = None

    def fuse(
        self,
        X_clinical: np.ndarray,
        X_biobert: np.ndarray,
        y: np.ndarray,
        clinical_feature_names: Optional[List[str]] = None,
    ) -> FusedDataset:
        """Fuses clinical feature vectors with BioBERT embeddings.

        Args:
            X_clinical: Structured feature matrix (N, num_clinical_features).
            X_biobert: Contextual embedding matrix (N, 768).
            y: Target label vector (N,).
            clinical_feature_names: Optional names of clinical features.

        Returns:
            FusedDataset instance containing the fused matrix and metadata.
        """
        assert X_clinical.shape[0] == X_biobert.shape[0] == y.shape[0], (
            f"Sample size mismatch: clinical={X_clinical.shape[0]}, biobert={X_biobert.shape[0]}, y={y.shape[0]}"
        )

        n_samples = X_clinical.shape[0]
        n_clin = X_clinical.shape[1]

        clin_names = clinical_feature_names or [f"clin_feat_{i}" for i in range(n_clin)]

        if self.use_pca_projection:
            logger.info(f"Projecting BioBERT embeddings from {X_biobert.shape[1]} to {self.pca_components} dims via PCA...")
            if self.pca is None:
                self.pca = PCA(n_components=self.pca_components, random_state=42)
                X_text_fused = self.pca.fit_transform(X_biobert)
            else:
                X_text_fused = self.pca.transform(X_biobert)
            text_names = [f"biobert_pca_{i}" for i in range(self.pca_components)]
        else:
            X_text_fused = X_biobert
            text_names = [f"biobert_emb_{i}" for i in range(X_biobert.shape[1])]

        n_text = X_text_fused.shape[1]
        X_fused = np.hstack([X_clinical, X_text_fused])
        total_dim = X_fused.shape[1]

        fused_names = clin_names + text_names

        logger.info(
            f"Multimodal Fusion complete: {n_samples} samples | "
            f"Clinical: {n_clin} dims + BioBERT: {n_text} dims = Total: {total_dim} dims"
        )

        return FusedDataset(
            X_fused=X_fused,
            y=y,
            clinical_dim=n_clin,
            text_dim=n_text,
            total_dim=total_dim,
            clinical_feature_names=clin_names,
            fused_feature_names=fused_names,
        )

