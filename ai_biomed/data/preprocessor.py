"""Data preprocessing, KNN imputation (k=5), Min-Max normalization, and feature selection.

Reconstructs the original clinical feature engineering pipeline:
1. Feature extraction and exclusion of administrative columns (PatientID, DoctorInCharge).
2. Selection of exactly 37 salient clinical/laboratory/demographic features.
3. Handling missing values via KNN Imputation (k=5).
4. Feature normalization via MinMaxScaler into the range [0, 1].
5. Stratified train/test splitting preserving class proportions.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from ai_biomed.config import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedData:
    """Container holding preprocessed clinical feature arrays and metadata."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: MinMaxScaler
    imputer: KNNImputer
    label_encoder: Optional[LabelEncoder] = None


class ClinicalPreprocessor:
    """Preprocessor for the general CKD cohort (1,659 samples, 37 clinical features)."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
        self.scaler = MinMaxScaler()
        self.selector = SelectKBest(score_func=f_classif, k=self.config.num_clinical_features)
        self.selected_features_: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "Diagnosis"
    ) -> ProcessedData:
        """Preprocesses the DataFrame, fits imputers/scalers, and returns a train/test split.

        Args:
            df: Raw DataFrame (1,659 rows).
            target_col: Column name of target label.

        Returns:
            ProcessedData containing train/test arrays and transformer objects.
        """
        logger.info("Starting clinical data preprocessing pipeline...")
        
        # 1. Drop non-predictive administrative columns
        cols_to_drop = [c for c in ["PatientID", "DoctorInCharge", target_col] if c in df.columns]
        X_raw = df.drop(columns=cols_to_drop).copy()
        y_raw = df[target_col].values

        # 2. Select 37 most informative clinical features using ANOVA F-value
        self.selector.fit(X_raw, y_raw)
        mask = self.selector.get_support()
        self.selected_features_ = X_raw.columns[mask].tolist()
        logger.info(f"Selected {len(self.selected_features_)} clinical features: {self.selected_features_[:5]} ...")
        assert len(self.selected_features_) == self.config.num_clinical_features, (
            f"Expected {self.config.num_clinical_features} features, got {len(self.selected_features_)}"
        )

        X_selected = X_raw[self.selected_features_].values

        # 3. Stratified train/test split before fitting transformers to prevent data leakage
        X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
            X_selected,
            y_raw,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_raw if self.config.stratify else None,
        )

        # 4. KNN Imputation (k=5)
        logger.info(f"Applying KNN Imputation (k={self.config.knn_neighbors})...")
        X_train_imp = self.imputer.fit_transform(X_tr_raw)
        X_test_imp = self.imputer.transform(X_te_raw)

        # 5. Min-Max Normalization [0, 1]
        logger.info("Applying Min-Max Normalization [0, 1]...")
        X_train_scaled = self.scaler.fit_transform(X_train_imp)
        X_test_scaled = self.scaler.transform(X_test_imp)

        logger.info(f"Preprocessing complete. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

        return ProcessedData(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.selected_features_,
            scaler=self.scaler,
            imputer=self.imputer,
        )

    def transform_full(self, df: pd.DataFrame, target_col: str = "Diagnosis") -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform the full dataset (e.g. for whole-dataset embedding fusion & CV)."""
        cols_to_drop = [c for c in ["PatientID", "DoctorInCharge", target_col] if c in df.columns]
        X_raw = df.drop(columns=cols_to_drop).copy()
        y_raw = df[target_col].values

        self.selector.fit(X_raw, y_raw)
        self.selected_features_ = X_raw.columns[self.selector.get_support()].tolist()
        X_selected = X_raw[self.selected_features_].values

        X_imp = self.imputer.fit_transform(X_selected)
        X_scaled = self.scaler.fit_transform(X_imp)
        return X_scaled, y_raw


class NephrotoxicDataPreprocessor:
    """Preprocessor for the Drug-Induced Nephrotoxicity dataset (1,500 samples, 37 columns)."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.scaler = MinMaxScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names_: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "ckd_risk_label"
    ) -> ProcessedData:
        """Preprocesses clinical, molecular, and pharmacokinetic features for nephrotoxicity modeling."""
        cols_to_drop = [c for c in ["ckd_risk_label", "nephrotoxic_label"] if c in df.columns]
        X = df.drop(columns=cols_to_drop).copy()
        y = df[target_col].values

        # Encode categorical columns
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        self.feature_names_ = list(X.columns)

        X_tr, X_te, y_train, y_test = train_test_split(
            X.values,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.stratify else None,
        )

        X_train_scaled = self.scaler.fit_transform(X_tr)
        X_test_scaled = self.scaler.transform(X_te)

        return ProcessedData(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.feature_names_,
            scaler=self.scaler,
            imputer=KNNImputer(n_neighbors=self.config.knn_neighbors),
        )
