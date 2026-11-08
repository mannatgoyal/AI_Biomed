"""Tests for clinical preprocessing, imputation, scaling, and feature selection."""

import pytest
import numpy as np
from ai_biomed.data.loader import load_ckd_general_dataset
from ai_biomed.data.preprocessor import ClinicalPreprocessor, NephrotoxicDataPreprocessor
from ai_biomed.config import PreprocessingConfig


def test_clinical_preprocessor_shape_and_scaling():
    df = load_ckd_general_dataset()
    config = PreprocessingConfig(num_clinical_features=37, knn_neighbors=5, test_size=0.2)
    preprocessor = ClinicalPreprocessor(config=config)
    
    processed = preprocessor.fit_transform(df, target_col="Diagnosis")
    
    # Check dimensions
    assert processed.X_train.shape[1] == 37
    assert processed.X_test.shape[1] == 37
    assert len(processed.feature_names) == 37
    
    # Check MinMax bounds [0, 1]
    assert np.all(processed.X_train >= -1e-7)
    assert np.all(processed.X_train <= 1.0 + 1e-7)
    
    # Check no NaN values remaining
    assert not np.isnan(processed.X_train).any()
    assert not np.isnan(processed.X_test).any()


def test_clinical_preprocessor_full_transform():
    df = load_ckd_general_dataset()
    preprocessor = ClinicalPreprocessor()
    X_scaled, y = preprocessor.transform_full(df, target_col="Diagnosis")
    
    assert X_scaled.shape == (1659, 37)
    assert len(y) == 1659
