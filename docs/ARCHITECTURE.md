# AI_Biomed System Architecture

## 1. High-Level Architectural Overview

```
                               +------------------------------------------+
                               |         Surviving Data Assets            |
                               +------------------------------------------+
                               | - Chronic_Kidney_Dsease_data.csv (1659)  |
                               | - CKD_NephrotoxicDrug_Dataset.csv (1500) |
                               | - realistic_drug_labels_side_effects     |
                               +--------------------+---------------------+
                                                    |
                      +-----------------------------+-----------------------------+
                      |                                                           |
                      v                                                           v
         +--------------------------+                               +--------------------------+
         | Structured Feature Flow  |                               | Unstructured Text Flow   |
         +--------------------------+                               +--------------------------+
         | 1. Exclude Admin Columns |                               | 1. Clinical Vignettes    |
         | 2. Select 37 Features    |                               | 2. Drug Monograph Texts  |
         | 3. KNN Imputation (k=5)  |                               | 3. FDA Product Labels    |
         | 4. Min-Max Scaling [0,1] |                               +------------+-------------+
         +------------+-------------+                                            |
                      |                                                          v
                      |                                             +--------------------------+
                      |                                             |    BioBERT v1.1 NLP      |
                      |                                             +--------------------------+
                      |                                             | - WordPiece Tokenizer    |
                      |                                             | - 12-layer Transformer   |
                      |                                             | - Mean Pooling (768-dim) |
                      |                                             | - Optional PCA (16-dim)  |
                      |                                             +------------+-------------+
                      |                                                          |
                      +-----------------------------+----------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |   Multimodal Feature Fusion Engine |
                                  +------------------------------------+
                                  |  Early / Intermediate Fusion:      |
                                  |  Z = [ X_clinical  ||  E_biobert ] |
                                  |  Dimension: 37 + 16/768            |
                                  +-----------------+------------------+
                                                    |
                      +-----------------------------+-----------------------------+
                      |                                                           |
                      v                                                           v
         +--------------------------+                               +--------------------------+
         |     Baseline Models      |                               |   Champion XGBoost       |
         +--------------------------+                               +--------------------------+
         | - Logistic Regression    |                               | - Multi-depth Trees (5)  |
         | - Random Forest (100)    |                               | - Regularized Shrinkage  |
         | - Support Vector Machine |                               | - Early Stopping & Subspl|
         | - Deep Residual MLP      |                               | - Calibrated Probabilities
         +------------+-------------+                               +------------+-------------+
                      |                                                           |
                      +-----------------------------+-----------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Evaluation & Explainability Suite  |
                                  +------------------------------------+
                                  | 1. Accuracy, F1, ROC-AUC, PR-AUC   |
                                  | 2. Stratified 5-Fold CV            |
                                  | 3. SHAP TreeExplainer Attribution  |
                                  | 4. Scientific Figures (ROC/PR/CM)  |
                                  +------------------------------------+
```

---

## 2. Directory Structure and Module Responsibilities

```
AIbiomed/
│
├── main.py                             # Unified CLI entrypoint (--mode benchmark|fusion|nephro|figures|cv|all)
├── requirements.txt                    # Project runtime dependencies
├── pyproject.toml                      # Modern Python packaging configuration
├── .gitignore                          # Standard git rules for ML/PyTorch
│
├── Chronic_Kidney_Dsease_data.csv      # Surviving Cohort 1: General CKD (1,659 samples, 54 cols)
├── CKD_NephrotoxicDrug_Dataset.csv     # Surviving Cohort 2: Drug Nephrotoxicity (1,500 samples, 37 cols)
├── realistic_drug_labels_side_effects.csv # Surviving Cohort 3: Realistic FDA Drug Labels (1,436 samples)
│
├── ai_biomed/                          # Core Research Package
│   ├── config.py                       # Dataclasses for paths, preprocessing, BioBERT, and model parameters
│   ├── data/
│   │   ├── loader.py                   # Dataset loading and integrity assertions
│   │   ├── preprocessor.py             # KNN Imputer (k=5), MinMaxScaler, 37-feature selection
│   │   └── text_generator.py           # Synthesis of EHR narratives & drug monographs
│   ├── nlp/
│   │   └── biobert_embedder.py         # Offline BioBERT v1.1 loader, tokenization, mean-pooling (768-dim)
│   ├── fusion/
│   │   └── feature_fusion.py           # Multimodal concatenation fusion and PCA projection
│   ├── models/
│   │   ├── baselines.py                # Logistic Regression, Random Forest, SVM, Gradient Boosting
│   │   ├── xgboost_classifier.py       # Tuned XGBoost binary and multi-class classifier
│   │   └── neural_fusion.py            # Deep multimodal residual MLP (PyTorch)
│   ├── evaluation/
│   │   ├── metrics.py                  # Accuracy, Balanced Acc, F1, ROC-AUC, PR-AUC, Confusion Matrix
│   │   ├── cross_validation.py         # Stratified K-Fold CV engine
│   │   └── benchmark.py                # Historical benchmark verification runner
│   └── explainability/
│       └── shap_explainer.py           # SHAP TreeExplainer attribution & summary plots
│
├── scripts/
│   ├── run_historical_benchmark.py     # Reproduces reported metrics (LR 75.3%, RF 93.2%, XGB 96.0%)
│   ├── run_biobert_fusion.py           # Unimodal vs Multimodal ablation comparison
│   ├── run_nephrotoxicity_pipeline.py  # Drug-induced nephrotoxicity & multi-class progression modeling
│   └── generate_figures.py             # Generates scientific ROC, PR, CM, and SHAP figures
│
├── tests/                              # Comprehensive Pytest test suite (11 unit/integration tests)
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_biobert_embedder.py
│   ├── test_fusion.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── results/                            # Benchmark tables, metrics CSVs, and audit reports
│   └── figures/                        # High-resolution scientific figures (fig1-fig4)
└── cache/                              # Precomputed 768-dim BioBERT embeddings
```

---

## 3. Dataflow & Pipeline Execution

1. **Loader Step**: `ai_biomed.data.loader` verifies file presence, schema integrity, and sample counts.
2. **Preprocessing Step**: `ai_biomed.data.preprocessor` selects the top 37 clinical features via ANOVA F-score, imputes missing values using 5 nearest neighbors, and scales variables into $[0, 1]$.
3. **NLP Step**: `ai_biomed.nlp.biobert_embedder` loads `dmis-lab/biobert-v1.1` from disk, processes textual descriptions, and generates 768-dimensional contextual vectors via attention-weighted mean pooling.
4. **Fusion Step**: `ai_biomed.fusion.feature_fusion` concatenates structured clinical vectors with BioBERT embeddings (or projected $d=16$ components).
5. **Model Step**: `ai_biomed.models` trains baseline models alongside the champion XGBoost classifier.
6. **Interpretability Step**: `ai_biomed.explainability` computes SHAP interaction values, attributing individual risk scores to clinical parameters and language features.
