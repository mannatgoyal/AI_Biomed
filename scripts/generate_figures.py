"""Publication-Quality Scientific Figure Generation for AI_Biomed.

Generates:
1. Multi-model ROC Curves
2. Precision-Recall Curves
3. Confusion Matrix Visualizations
4. SHAP Feature Importance Summary Plot
Saved to results/figures/.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from ai_biomed.config import PathConfig, PreprocessingConfig, BioBERTConfig
from ai_biomed.data.loader import load_ckd_general_dataset
from ai_biomed.data.preprocessor import ClinicalPreprocessor
from ai_biomed.nlp.biobert_embedder import BioBERTEmbedder
from ai_biomed.fusion.feature_fusion import MultimodalFusionEngine
from ai_biomed.explainability.shap_explainer import BiomedSHAPExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_all_figures():
    path_cfg = PathConfig()
    fig_dir = path_cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Seaborn style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    # 1. Load data & precomputed embeddings
    df = load_ckd_general_dataset(path_config=path_cfg)
    preprocessor = ClinicalPreprocessor()
    X_clin, y = preprocessor.transform_full(df, target_col="Diagnosis")

    cache_path = path_cfg.biobert_ckd_cache_path
    X_biobert = np.load(cache_path)

    fusion = MultimodalFusionEngine(use_pca_projection=True, pca_components=16)
    fused_bundle = fusion.fuse(X_clin, X_biobert, y, clinical_feature_names=preprocessor.selected_features_)
    X_fused = fused_bundle.X_fused

    X_tr, X_te, y_tr, y_te = train_test_split(X_fused, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "XGBoost Classifier": xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, random_state=42, eval_metric="logloss")
    }

    fitted_models = {}
    probs = {}
    preds = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        fitted_models[name] = model
        probs[name] = model.predict_proba(X_te)[:, 1]
        preds[name] = model.predict(X_te)

    # -------------------------------------------------------------
    # Figure 1: Multi-Model ROC Curves
    # -------------------------------------------------------------
    logger.info("Generating ROC Curves figure...")
    plt.figure(figsize=(8, 6))
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]

    for (name, pr), col in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_te, pr)
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {score:.4f})", color=col)

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Chance Level (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curves\nAI_Biomed Multimodal Feature Fusion", fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = fig_dir / "fig1_roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {roc_path}")

    # -------------------------------------------------------------
    # Figure 2: Precision-Recall Curves
    # -------------------------------------------------------------
    logger.info("Generating PR Curves figure...")
    plt.figure(figsize=(8, 6))
    for (name, pr), col in zip(probs.items(), colors):
        precision, recall, _ = precision_recall_curve(y_te, pr)
        pr_score = auc(recall, precision)
        plt.plot(recall, precision, lw=2.2, label=f"{name} (PR-AUC = {pr_score:.4f})", color=col)

    plt.xlabel("Recall (Sensitivity)", fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    plt.title("Precision-Recall Curves (AI_Biomed Multimodal Fusion)", fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    pr_path = fig_dir / "fig2_pr_curves.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {pr_path}")

    # -------------------------------------------------------------
    # Figure 3: XGBoost Confusion Matrix
    # -------------------------------------------------------------
    logger.info("Generating Confusion Matrix figure...")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, preds["XGBoost Classifier"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Non-CKD (0)", "CKD (1)"],
        yticklabels=["Non-CKD (0)", "CKD (1)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.xlabel("Predicted Clinical Label", fontweight="bold")
    plt.ylabel("True Clinical Label", fontweight="bold")
    plt.title("XGBoost Confusion Matrix (Test Cohort)", fontweight="bold", pad=12)
    plt.tight_layout()
    cm_path = fig_dir / "fig3_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {cm_path}")

    # -------------------------------------------------------------
    # Figure 4: SHAP Feature Attribution Plot
    # -------------------------------------------------------------
    logger.info("Generating SHAP Feature Attribution figure...")
    explainer = BiomedSHAPExplainer(fitted_models["XGBoost Classifier"], fused_bundle.fused_feature_names)
    explainer.explain(X_te, max_evals=250)
    shap_path = fig_dir / "fig4_shap_summary.png"
    explainer.plot_summary(shap_path, max_display=15)
    
    # Export top SHAP table
    shap_df = explainer.get_feature_importance_df(top_n=25)
    shap_df.to_csv(path_cfg.results_dir / "top_shap_features.csv", index=False)
    logger.info(f"Exported top SHAP features to: {path_cfg.results_dir / 'top_shap_features.csv'}")

    print("\n[OK] All scientific research figures successfully generated in:", fig_dir)


if __name__ == "__main__":
    generate_all_figures()

