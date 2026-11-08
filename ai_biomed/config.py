"""Configuration parameters and settings for the AI_Biomed research framework.

This module defines standardized dataclasses for file paths, preprocessing hyperparameters,
BioBERT embedding parameters, baseline and XGBoost configurations, and reproducibility seeds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PathConfig:
    """Directory and file path configurations."""
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    # Surviving dataset filenames
    ckd_general_file: str = "Chronic_Kidney_Dsease_data.csv"
    ckd_nephro_file: str = "CKD_NephrotoxicDrug_Dataset.csv"
    drug_labels_file: str = "realistic_drug_labels_side_effects.csv"
    
    # Output and artifacts directories
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "cache")
    figures_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results" / "figures")

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ckd_general_path(self) -> Path:
        return self.data_dir / self.ckd_general_file

    @property
    def ckd_nephro_path(self) -> Path:
        return self.data_dir / self.ckd_nephro_file

    @property
    def drug_labels_path(self) -> Path:
        return self.data_dir / self.drug_labels_file

    @property
    def biobert_ckd_cache_path(self) -> Path:
        return self.cache_dir / "biobert_ckd_embeddings_768.npy"

    @property
    def biobert_nephro_cache_path(self) -> Path:
        return self.cache_dir / "biobert_nephro_embeddings_768.npy"

    @property
    def biobert_labels_cache_path(self) -> Path:
        return self.cache_dir / "biobert_labels_embeddings_768.npy"


@dataclass
class PreprocessingConfig:
    """Hyperparameters for data cleaning, imputation, and feature scaling."""
    knn_neighbors: int = 5
    num_clinical_features: int = 37
    scaler_type: str = "minmax"  # Min-Max Normalization [0, 1]
    feature_selection_method: str = "f_classif"
    random_state: int = 42
    test_size: float = 0.20
    stratify: bool = True


@dataclass
class BioBERTConfig:
    """Hyperparameters for biomedical NLP and contextual embedding extraction."""
    model_name: str = "dmis-lab/biobert-v1.1"
    embedding_dim: int = 768
    max_seq_length: int = 128
    batch_size: int = 32
    pooling_strategy: str = "mean"  # "mean" or "cls"
    pca_reduction_dim: Optional[int] = 16  # Optional dimensionality projection
    device: str = "cpu"


@dataclass
class ModelConfig:
    """Baseline models and XGBoost hyperparameter configuration."""
    random_state: int = 42
    
    # Logistic Regression
    lr_max_iter: int = 1000
    lr_class_weight: str = "balanced"
    
    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 15
    rf_class_weight: str = "balanced"
    
    # XGBoost Classifier
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_gamma: float = 0.1
    xgb_eval_metric: str = "logloss"


PUBLISHED_ABSTRACT: str = (
    "Chronic Kidney Disease (CKD) impacts almost 10% of adults globally, and it is frequently "
    "developing silently in an individual to become End-Stage Renal Disease (ESRD), necessitating immediate "
    "risk assessment. Current models are based on data of structured clinical parameters such as GFR, serum "
    "creatinine, and blood pressure but neglect pharmacological events controlling disease progression. Drug-induced "
    "nephrotoxicity, particularly when polypharmacy occurs, is one of the most potent but least-exploited predictors, "
    "as related information is usually encoded in unstructured data sources, including drug labels. We present "
    "a multimodal learning framework for combining structured clinical data with semantic representations from "
    "drug-related text. Here we present a 1,659 patient sample dataset over 37 features pre-processed via imputation "
    "and normalization. BioBERT was used to encode drug label information, which generates embeddings capturing hidden "
    "nephrotoxic signals. These embeddings applied with clinical characteristics were included in data models such as "
    "Logistic Regression, Random Forest and XGBoost. The multimodal approach outperformed unimodal models with a total "
    "of 96.00% accuracy, 0.9498 F1-score, and 0.9982 AUC on XGBoost. Text-based embedding contributed approximately "
    "24% of total performance while demonstrating the significant importance of pharmaceutical data. The results "
    "indicate high sensitivity and small misclassification, which enhances identification of high risk patients. "
    "This framework helps optimize CKD risk stratification by integrating clinical and text data, with future work "
    "primarily directed towards validation and interpretability."
)


@dataclass
class HistoricalBenchmarkTargets:
    """Previously reported experimental values for verification and reproduction."""
    dataset_size: int = 1659
    clinical_features: int = 37
    biobert_embedding_dim: int = 768
    logistic_regression_acc: float = 75.30
    random_forest_acc: float = 93.20
    xgboost_acc: float = 96.00
    xgboost_f1: float = 0.9498
    xgboost_roc_auc: float = 0.9982
    text_contribution_pct: float = 24.0

