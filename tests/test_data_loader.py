"""Tests for data loading and schema validation."""

import pandas as pd
from ai_biomed.data.loader import (
    load_ckd_general_dataset,
    load_ckd_nephrotoxic_dataset,
    load_drug_labels_dataset,
)


def test_load_ckd_general_dataset():
    df = load_ckd_general_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1659
    assert df.shape[1] == 54
    assert "Diagnosis" in df.columns
    assert "PatientID" in df.columns


def test_load_ckd_nephrotoxic_dataset():
    df = load_ckd_nephrotoxic_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1500
    assert df.shape[1] == 37
    assert "ckd_risk_label" in df.columns
    assert "nephrotoxic_label" in df.columns
    assert "drug_name" in df.columns


def test_load_drug_labels_dataset():
    df = load_drug_labels_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1436
    assert "drug_name" in df.columns
    assert "indications" in df.columns
    assert "side_effects" in df.columns
    assert "warnings" in df.columns
    assert "contraindications" in df.columns
