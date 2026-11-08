"""Multimodal Drug-Induced Nephrotoxicity and CKD Risk Progression Pipeline.

Integrates:
1. Patient Demographics & Baseline Renal Biomarkers (Age, BP, Creatinine, BUN, Glucose, Diabetes, Hypertension)
2. Pharmacotherapy & Drug Regimen (Dosage, Exposure Days)
3. Cheminformatics & 3D Molecular Properties (MW, LogP, H-bonds, TPSA, 3D Inertia)
4. Pharmacokinetics & Cellular Damage Biomarkers (Clearance, Half-life, Viability, Mitochondrial, ROS)
5. BioBERT Pharmacology Monographs (768-dim contextual embeddings)

Evaluates:
- Target 1: Intrinsic Drug Nephrotoxicity Classification (Binary: 0 vs 1)
- Target 2: Patient-Specific CKD Risk Stratification (Multi-class: Low, Moderate, Severe)
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import xgboost as xgb

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from ai_biomed.config import PathConfig, BioBERTConfig, ModelConfig
from ai_biomed.data.loader import load_ckd_nephrotoxic_dataset
from ai_biomed.data.text_generator import DRUG_PHARMACOLOGY_MONOGRAPHS
from ai_biomed.nlp.biobert_embedder import BioBERTEmbedder
from ai_biomed.evaluation.metrics import compute_evaluation_metrics, format_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    path_cfg = PathConfig()
    bert_cfg = BioBERTConfig()
    model_cfg = ModelConfig()

    logger.info("=" * 70)
    logger.info("DRUG-INDUCED NEPHROTOXICITY & CKD RISK PROGRESSION PIPELINE")
    logger.info("=" * 70)

    # 1. Load Dataset (1,500 samples, 37 columns)
    df = load_ckd_nephrotoxic_dataset(path_config=path_cfg)

    # 2. Extract / Load BioBERT embeddings for drug monographs
    cache_path = path_cfg.biobert_nephro_cache_path
    embedder = BioBERTEmbedder(config=bert_cfg, path_config=path_cfg)

    if cache_path.exists():
        logger.info(f"Loading cached BioBERT drug embeddings from: {cache_path}")
        X_biobert = np.load(cache_path)
    else:
        logger.info("Extracting BioBERT embeddings for pharmacological monographs...")
        unique_drugs = sorted(df["drug_name"].unique())
        unique_texts = [DRUG_PHARMACOLOGY_MONOGRAPHS.get(d, f"Therapeutic drug {d}") for d in unique_drugs]
        unique_embs = embedder.encode_texts(unique_texts)
        drug_to_emb = {d: emb for d, emb in zip(unique_drugs, unique_embs)}
        
        # Map to full dataset
        X_biobert = np.array([drug_to_emb[d] for d in df["drug_name"]])
        np.save(cache_path, X_biobert)
        logger.info(f"Saved drug monograph embeddings to: {cache_path}")

    # Dimensionality projection of BioBERT embeddings
    pca = PCA(n_components=16, random_state=42)
    X_text_pca = pca.fit_transform(X_biobert)

    # 3. Tabular preprocessing
    drop_cols = ["ckd_risk_label", "nephrotoxic_label"]
    X_tab = df.drop(columns=drop_cols).copy()
    for col in X_tab.select_dtypes(include=["object"]).columns:
        X_tab[col] = LabelEncoder().fit_transform(X_tab[col])

    scaler = StandardScaler()
    X_tab_scaled = scaler.fit_transform(X_tab)

    # Multimodal feature fusion
    X_fused = np.hstack([X_tab_scaled, X_text_pca])
    logger.info(f"Fused feature matrix: {X_fused.shape} (Tabular: {X_tab_scaled.shape[1]} + BioBERT: {X_text_pca.shape[1]})")

    # =========================================================================
    # TASK 1: Patient-Specific CKD Risk Stratification (Multi-class: 0, 1, 2)
    # =========================================================================
    logger.info("\n--- TASK 1: Patient-Specific CKD Progression Risk Stratification (3-Class) ---")
    y_risk = df["ckd_risk_label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fused, y_risk, test_size=0.2, random_state=42, stratify=y_risk
    )

    clf_risk = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    clf_risk.fit(X_tr, y_tr)
    pred_risk = clf_risk.predict(X_te)
    prob_risk = clf_risk.predict_proba(X_te)

    metrics_risk = compute_evaluation_metrics(y_te, pred_risk, prob_risk, model_name="BioBERT-Fused XGBoost (CKD Risk)")
    print(format_metrics_table([metrics_risk]).to_string(index=False))

    # =========================================================================
    # TASK 2: Intrinsic Nephrotoxic Event Prediction (Binary: 0 vs 1)
    # =========================================================================
    logger.info("\n--- TASK 2: Intrinsic Nephrotoxicity Classification (Binary) ---")
    y_nephro = df["nephrotoxic_label"].values

    X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(
        X_fused, y_nephro, test_size=0.2, random_state=42, stratify=y_nephro
    )

    clf_nephro = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    clf_nephro.fit(X_tr_n, y_tr_n)
    pred_nephro = clf_nephro.predict(X_te_n)
    prob_nephro = clf_nephro.predict_proba(X_te_n)

    metrics_nephro = compute_evaluation_metrics(y_te_n, pred_nephro, prob_nephro, model_name="BioBERT-Fused XGBoost (Nephrotoxicity)")
    print(format_metrics_table([metrics_nephro]).to_string(index=False))

    # Save results
    results_dir = path_cfg.results_dir
    nephro_df = format_metrics_table([metrics_risk, metrics_nephro])
    nephro_df.to_csv(results_dir / "nephrotoxicity_pipeline_metrics.csv", index=False)
    print(f"\n[OK] Nephrotoxicity pipeline metrics saved to: {results_dir}")


if __name__ == "__main__":
    main()
