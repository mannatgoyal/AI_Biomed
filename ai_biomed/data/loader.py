"""Dataset loading, validation, and schema verification module for AI_Biomed.

Loads surviving cohorts:
1. General CKD Cohort (1,659 samples, 54 columns)
2. Drug-Induced Nephrotoxicity Cohort (1,500 samples, 37 columns)
3. FDA-style Drug Labels & Safety Dataset (1,436 samples, 15 columns)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple
import pandas as pd

from ai_biomed.config import PathConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Container holding loaded DataFrames and metadata."""
    df_ckd_general: pd.DataFrame
    df_ckd_nephro: pd.DataFrame
    df_drug_labels: pd.DataFrame


def _find_file_case_insensitive(directory: Path, target_names: list[str]) -> Path:
    """Finds a file matching any name in target_names, handling typos like Dsease/Disease."""
    for name in target_names:
        p = directory / name
        if p.exists():
            return p
    # Fallback search in directory
    for f in directory.iterdir():
        if f.is_file():
            for name in target_names:
                if f.name.lower() == name.lower():
                    return f
    raise FileNotFoundError(f"Could not locate any file matching {target_names} in {directory}")


def load_ckd_general_dataset(
    file_path: Optional[Union[str, Path]] = None,
    path_config: Optional[PathConfig] = None
) -> pd.DataFrame:
    """Load and validate the general Chronic Kidney Disease dataset (1,659 samples).

    Args:
        file_path: Optional explicit path to the CSV file.
        path_config: Optional PathConfig instance.

    Returns:
        pd.DataFrame containing the general CKD patient data.
    """
    cfg = path_config or PathConfig()
    if file_path is not None:
        p = Path(file_path)
    else:
        candidates = ["Chronic_Kidney_Dsease_data.csv", "Chronic_Kidney_Disease_data.csv"]
        p = _find_file_case_insensitive(cfg.data_dir, candidates)

    logger.info(f"Loading CKD General Dataset from: {p}")
    df = pd.read_csv(p)
    logger.info(f"Loaded {len(df)} samples with {df.shape[1]} columns.")
    
    # Validation checks
    assert len(df) == 1659, f"Expected 1,659 samples, found {len(df)}"
    assert "Diagnosis" in df.columns, "Diagnosis column missing from general CKD dataset"
    return df


def load_ckd_nephrotoxic_dataset(
    file_path: Optional[Union[str, Path]] = None,
    path_config: Optional[PathConfig] = None
) -> pd.DataFrame:
    """Load and validate the Drug-Induced Nephrotoxicity dataset (1,500 samples, 37 columns).

    Args:
        file_path: Optional explicit path to the CSV file.
        path_config: Optional PathConfig instance.

    Returns:
        pd.DataFrame containing the nephrotoxic drug exposure data.
    """
    cfg = path_config or PathConfig()
    p = Path(file_path) if file_path else cfg.ckd_nephro_path
    if not p.exists():
        candidates = ["CKD_NephrotoxicDrug_Dataset.csv"]
        p = _find_file_case_insensitive(cfg.data_dir, candidates)

    logger.info(f"Loading CKD Nephrotoxic Dataset from: {p}")
    df = pd.read_csv(p)
    logger.info(f"Loaded {len(df)} samples with {df.shape[1]} columns.")
    
    assert len(df) == 1500, f"Expected 1,500 samples, found {len(df)}"
    assert "ckd_risk_label" in df.columns, "ckd_risk_label missing from dataset"
    assert "nephrotoxic_label" in df.columns, "nephrotoxic_label missing from dataset"
    return df


def load_drug_labels_dataset(
    file_path: Optional[Union[str, Path]] = None,
    path_config: Optional[PathConfig] = None
) -> pd.DataFrame:
    """Load and validate the FDA-style drug labels and safety dataset (1,436 samples).

    Args:
        file_path: Optional explicit path to the CSV file.
        path_config: Optional PathConfig instance.

    Returns:
        pd.DataFrame containing drug labels, warnings, and adverse effect annotations.
    """
    cfg = path_config or PathConfig()
    p = Path(file_path) if file_path else cfg.drug_labels_path
    if not p.exists():
        candidates = ["realistic_drug_labels_side_effects.csv"]
        p = _find_file_case_insensitive(cfg.data_dir, candidates)

    logger.info(f"Loading Drug Labels Dataset from: {p}")
    df = pd.read_csv(p)
    logger.info(f"Loaded {len(df)} samples with {df.shape[1]} columns.")
    
    assert len(df) == 1436, f"Expected 1,436 samples, found {len(df)}"
    return df


def load_all_datasets(path_config: Optional[PathConfig] = None) -> DatasetBundle:
    """Load all three surviving project datasets into a DatasetBundle.

    Args:
        path_config: Optional PathConfig instance.

    Returns:
        DatasetBundle with df_ckd_general, df_ckd_nephro, and df_drug_labels.
    """
    cfg = path_config or PathConfig()
    return DatasetBundle(
        df_ckd_general=load_ckd_general_dataset(path_config=cfg),
        df_ckd_nephro=load_ckd_nephrotoxic_dataset(path_config=cfg),
        df_drug_labels=load_drug_labels_dataset(path_config=cfg),
    )
