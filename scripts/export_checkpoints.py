"""Exports trained champion models and fitted preprocessors for real-time inference.

Serializes:
1. General CKD Model & Preprocessor Bundle (37 clinical features + BioBERT fusion -> XGBoost)
2. Drug-Induced Nephrotoxicity & Progression Model Bundle (37 tabular features + BioBERT monograph -> XGBoost)
Saved to checkpoints/.
"""

import sys
import logging
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import xgboost as xgb

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from ai_biomed.config import PathConfig, PreprocessingConfig
from ai_biomed.data.loader import load_ckd_general_dataset, load_ckd_nephrotoxic_dataset
from ai_biomed.data.preprocessor import ClinicalPreprocessor
from ai_biomed.fusion.feature_fusion import MultimodalFusionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def export_models():
    path_cfg = PathConfig()
    ckpt_dir = path_cfg.root_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Train and export General CKD Model
    # -------------------------------------------------------------
    logger.info("Training and serializing General CKD Champion Model...")
    df_gen = load_ckd_general_dataset(path_config=path_cfg)
    clin_prep = ClinicalPreprocessor()
    X_clin, y_gen = clin_prep.transform_full(df_gen, target_col="Diagnosis")

    cache_path = path_cfg.biobert_ckd_cache_path
    X_biobert = np.load(cache_path)

    fusion = MultimodalFusionEngine(use_pca_projection=True, pca_components=16)
    fused_bundle = fusion.fuse(X_clin, X_biobert, y_gen, clinical_feature_names=clin_prep.selected_features_)
    X_fused = fused_bundle.X_fused

    X_tr, X_te, y_tr, y_te = train_test_split(X_fused, y_gen, test_size=0.2, random_state=42, stratify=y_gen)

    xgb_gen = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_gen.fit(X_tr, y_tr)

    gen_bundle = {
        "model": xgb_gen,
        "preprocessor": clin_prep,
        "pca": fusion.pca,
        "selected_features": clin_prep.selected_features_,
        "fused_feature_names": fused_bundle.fused_feature_names,
    }
    gen_path = ckpt_dir / "ckd_general_champion.joblib"
    joblib.dump(gen_bundle, gen_path)
    logger.info(f"Saved General CKD model bundle to: {gen_path}")

    # -------------------------------------------------------------
    # 2. Train and export Drug-Induced Nephrotoxicity & Risk Model
    # -------------------------------------------------------------
    logger.info("Training and serializing Drug Nephrotoxicity Champion Model...")
    df_nephro = load_ckd_nephrotoxic_dataset(path_config=path_cfg)
    
    cache_nephro = path_cfg.biobert_nephro_cache_path
    X_biobert_nephro = np.load(cache_nephro)

    pca_nephro = PCA(n_components=16, random_state=42)
    X_text_pca = pca_nephro.fit_transform(X_biobert_nephro)

    drop_cols = ["ckd_risk_label", "nephrotoxic_label"]
    X_tab = df_nephro.drop(columns=drop_cols).copy()
    label_encoders = {}
    for col in X_tab.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X_tab[col] = le.fit_transform(X_tab[col])
        label_encoders[col] = le

    scaler_nephro = StandardScaler()
    X_tab_scaled = scaler_nephro.fit_transform(X_tab)
    X_fused_nephro = np.hstack([X_tab_scaled, X_text_pca])

    y_risk = df_nephro["ckd_risk_label"].values
    X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(X_fused_nephro, y_risk, test_size=0.2, random_state=42, stratify=y_risk)

    xgb_risk = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_risk.fit(X_tr_n, y_tr_n)

    nephro_bundle = {
        "model": xgb_risk,
        "scaler": scaler_nephro,
        "pca": pca_nephro,
        "label_encoders": label_encoders,
        "feature_names": list(X_tab.columns),
    }
    nephro_path = ckpt_dir / "ckd_nephro_champion.joblib"
    joblib.dump(nephro_bundle, nephro_path)
    logger.info(f"Saved Drug Nephrotoxicity model bundle to: {nephro_path}")

    print("\n[OK] Model checkpoints exported successfully to:", ckpt_dir)


if __name__ == "__main__":
    export_models()
