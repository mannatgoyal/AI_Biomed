"""Full Multimodal BioBERT Feature Fusion Comparative Benchmark.

Evaluates and compares model performance on:
1. Clinical Features Alone (Unimodal Baseline)
2. BioBERT Text Embeddings Alone (NLP Baseline)
3. Fused Multimodal Representation (Clinical + BioBERT Contextual Fusion)
Across Logistic Regression, Random Forest, SVM, Gradient Boosting, Multimodal MLP, and XGBoost.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from ai_biomed.config import PathConfig, PreprocessingConfig, BioBERTConfig, ModelConfig
from ai_biomed.data.loader import load_ckd_general_dataset
from ai_biomed.data.preprocessor import ClinicalPreprocessor
from ai_biomed.data.text_generator import generate_patient_clinical_narratives
from ai_biomed.nlp.biobert_embedder import BioBERTEmbedder
from ai_biomed.fusion.feature_fusion import MultimodalFusionEngine
from ai_biomed.models.neural_fusion import MultimodalNeuralClassifier
from ai_biomed.evaluation.metrics import compute_evaluation_metrics, format_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    path_cfg = PathConfig()
    prep_cfg = PreprocessingConfig()
    bert_cfg = BioBERTConfig()
    model_cfg = ModelConfig()

    logger.info("=" * 70)
    logger.info("COMPREHENSIVE MULTIMODAL FUSION ABLATION STUDY")
    logger.info("=" * 70)

    # 1. Load data
    df = load_ckd_general_dataset(path_config=path_cfg)

    # 2. Extract clinical features
    preprocessor = ClinicalPreprocessor(config=prep_cfg)
    X_clinical, y = preprocessor.transform_full(df, target_col="Diagnosis")

    # 3. BioBERT embeddings
    embedder = BioBERTEmbedder(config=bert_cfg, path_config=path_cfg)
    cache_path = path_cfg.biobert_ckd_cache_path
    if cache_path.exists():
        logger.info(f"Loading cached BioBERT embeddings from: {cache_path}")
        X_biobert = np.load(cache_path)
    else:
        narratives = generate_patient_clinical_narratives(df)
        X_biobert = embedder.encode_texts(narratives, cache_path=cache_path)

    # 4. Multimodal Fusion (Clinical 37 + BioBERT 16 = 53 dims)
    fusion_engine = MultimodalFusionEngine(use_pca_projection=True, pca_components=16)
    fused_dataset = fusion_engine.fuse(
        X_clinical=X_clinical,
        X_biobert=X_biobert,
        y=y,
        clinical_feature_names=preprocessor.selected_features_
    )
    X_fused = fused_dataset.X_fused

    # Splitting
    X_tr_clin, X_te_clin, y_train, y_test = train_test_split(
        X_clinical, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr_fused, X_te_fused, _, _ = train_test_split(
        X_fused, y, test_size=0.2, random_state=42, stratify=y
    )

    models_to_test = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        ),
        "Support Vector Machine": (
            SVC(probability=True, class_weight="balanced", random_state=42),
            SVC(probability=True, class_weight="balanced", random_state=42),
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
        ),
        "XGBoost Classifier": (
            xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss"),
            xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss"),
        ),
        "Multimodal Neural MLP": (
            MultimodalNeuralClassifier(hidden_dim=64, epochs=15, random_state=42),
            MultimodalNeuralClassifier(hidden_dim=64, epochs=15, random_state=42),
        ),
    }

    all_metrics = []
    logger.info("Training models across Unimodal (Clinical-only) vs Multimodal (BioBERT-fused)...")

    for name, (m_clin, m_fused) in models_to_test.items():
        # 1. Clinical-only
        m_clin.fit(X_tr_clin, y_train)
        p_clin = m_clin.predict(X_te_clin)
        prob_clin = m_clin.predict_proba(X_te_clin)
        met_clin = compute_evaluation_metrics(y_test, p_clin, prob_clin, model_name=f"{name} (Clinical Only)")
        all_metrics.append(met_clin)

        # 2. Multimodal Fused
        m_fused.fit(X_tr_fused, y_train)
        p_fused = m_fused.predict(X_te_fused)
        prob_fused = m_fused.predict_proba(X_te_fused)
        met_fused = compute_evaluation_metrics(y_test, p_fused, prob_fused, model_name=f"{name} (BioBERT Fused)")
        all_metrics.append(met_fused)

    table_df = format_metrics_table(all_metrics)
    print("\n" + "=" * 90)
    print("MULTIMODAL FEATURE FUSION ABLATION MATRIX")
    print("=" * 90)
    print(table_df.to_string(index=False))
    print("=" * 90 + "\n")

    out_csv = path_cfg.results_dir / "multimodal_fusion_ablation_results.csv"
    table_df.to_csv(out_csv, index=False)
    print(f"[OK] Ablation results saved to: {out_csv}")


if __name__ == "__main__":
    main()
