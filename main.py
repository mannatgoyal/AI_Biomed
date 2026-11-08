"""AI_Biomed: Chronic Kidney Disease Risk Stratification using Biomedical NLP and Machine Learning.

Reconstructed unified entry point providing modular CLI workflows:
1. Historical Benchmark Reproduction (--mode benchmark)
2. Multimodal BioBERT Feature Fusion Ablation (--mode fusion)
3. Drug-Induced Nephrotoxicity & Progression Modeling (--mode nephro)
4. Cross-Validation Suite (--mode cv)
5. Publication Figures & SHAP Explainability (--mode figures)
6. Full End-to-End Execution (--mode all)

Author: Mannat Goyal
Repository: https://github.com/mannatgoyal/AI_Biomed
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from ai_biomed.config import PathConfig, PreprocessingConfig, BioBERTConfig, ModelConfig
from ai_biomed.evaluation.benchmark import run_historical_benchmark
from scripts.run_biobert_fusion import main as run_fusion_pipeline
from scripts.run_nephrotoxicity_pipeline import main as run_nephro_pipeline
from scripts.generate_figures import generate_all_figures
from ai_biomed.evaluation.cross_validation import run_stratified_cv
from ai_biomed.data.loader import load_ckd_general_dataset
from ai_biomed.data.preprocessor import ClinicalPreprocessor
from ai_biomed.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from ai_biomed.models.xgboost_classifier import XGBoostBiomedClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AI_Biomed")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for AI_Biomed execution."""
    parser = argparse.ArgumentParser(
        description="AI_Biomed: CKD Risk Stratification using Biomedical NLP and Machine Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=["benchmark", "fusion", "nephro", "figures", "cv", "all"],
        help="Pipeline execution mode."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputing BioBERT embeddings instead of using disk cache."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save metric tables, audit logs, and figures."
    )
    return parser.parse_args()


def run_cv_workflow(path_cfg: PathConfig):
    """Executes 5-fold stratified cross validation across baseline and champion models."""
    logger.info("Executing 5-Fold Stratified Cross Validation...")
    df = load_ckd_general_dataset(path_config=path_cfg)
    preprocessor = ClinicalPreprocessor()
    X_scaled, y = preprocessor.transform_full(df, target_col="Diagnosis")

    models = {
        "Logistic Regression": LogisticRegressionBaseline(class_weight="balanced", max_iter=1000),
        "Random Forest": RandomForestBaseline(n_estimators=100, class_weight="balanced"),
        "XGBoost Classifier": XGBoostBiomedClassifier(n_estimators=150, max_depth=4, learning_rate=0.05),
    }

    cv_results = run_stratified_cv(models, X_scaled, y, n_splits=5, random_state=42)
    print("\n" + "=" * 80)
    print("5-FOLD STRATIFIED CROSS VALIDATION RESULTS")
    print("=" * 80)
    print(cv_results.to_string(index=False))
    print("=" * 80 + "\n")

    out_path = path_cfg.results_dir / "stratified_5fold_cv_results.csv"
    cv_results.to_csv(out_path, index=False)
    logger.info(f"Cross-validation results saved to: {out_path}")


def main():
    """Main CLI execution controller."""
    args = parse_arguments()
    path_cfg = PathConfig(results_dir=ROOT_DIR / args.output_dir)

    print("\n" + "#" * 80)
    print("#  AI_BIOMED: CHRONIC KIDNEY DISEASE RISK STRATIFICATION FRAMEWORK")
    print("#  Biomedical NLP (BioBERT) & Multimodal Feature Fusion Architecture")
    print("#" * 80 + "\n")
    logger.info(f"Selected Mode: '{args.mode}' | Output Directory: '{args.output_dir}'")

    if args.mode == "benchmark":
        metrics_df, comp_df = run_historical_benchmark(path_config=path_cfg, verbose=True)
        metrics_df.to_csv(path_cfg.results_dir / "historical_reproduction_metrics.csv", index=False)
        comp_df.to_csv(path_cfg.results_dir / "historical_verification_table.csv", index=False)

    elif args.mode == "fusion":
        run_fusion_pipeline()

    elif args.mode == "nephro":
        run_nephro_pipeline()

    elif args.mode == "figures":
        generate_all_figures()

    elif args.mode == "cv":
        run_cv_workflow(path_cfg)

    elif args.mode == "all":
        logger.info("Executing Complete End-to-End AI_Biomed Research Suite...")
        # 1. Historical Benchmark
        m_df, c_df = run_historical_benchmark(path_config=path_cfg, verbose=True)
        m_df.to_csv(path_cfg.results_dir / "historical_reproduction_metrics.csv", index=False)
        c_df.to_csv(path_cfg.results_dir / "historical_verification_table.csv", index=False)
        
        # 2. Multimodal Fusion Ablation
        run_fusion_pipeline()
        
        # 3. Drug-Induced Nephrotoxicity
        run_nephro_pipeline()
        
        # 4. Cross Validation
        run_cv_workflow(path_cfg)
        
        # 5. Scientific Figures & SHAP Explainability
        generate_all_figures()
        
        logger.info("[SUCCESS] Full research suite completed. All artifacts generated!")


if __name__ == "__main__":
    main()
