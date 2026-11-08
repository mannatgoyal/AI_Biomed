"""Tests for multimodal feature fusion engine."""

import pytest
import numpy as np
from ai_biomed.fusion.feature_fusion import MultimodalFusionEngine


def test_multimodal_fusion_raw():
    engine = MultimodalFusionEngine(use_pca_projection=False)
    
    n_samples = 50
    X_clin = np.random.randn(n_samples, 37)
    X_biobert = np.random.randn(n_samples, 768)
    y = np.random.randint(0, 2, n_samples)
    
    fused = engine.fuse(X_clin, X_biobert, y)
    assert fused.X_fused.shape == (n_samples, 37 + 768)
    assert fused.clinical_dim == 37
    assert fused.text_dim == 768
    assert fused.total_dim == 805


def test_multimodal_fusion_with_pca():
    engine = MultimodalFusionEngine(use_pca_projection=True, pca_components=16)
    
    n_samples = 50
    X_clin = np.random.randn(n_samples, 37)
    X_biobert = np.random.randn(n_samples, 768)
    y = np.random.randint(0, 2, n_samples)
    
    fused = engine.fuse(X_clin, X_biobert, y)
    assert fused.X_fused.shape == (n_samples, 37 + 16)
    assert fused.clinical_dim == 37
    assert fused.text_dim == 16
    assert fused.total_dim == 53
