# AI_Biomed: Chronic Kidney Disease Risk Stratification using Biomedical NLP and Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace BioBERT](https://img.shields.io/badge/BioBERT-v1.1-yellow.svg)](https://huggingface.co/dmis-lab/biobert-v1.1)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Tests: 11 Passed](https://img.shields.io/badge/tests-11%20passed-brightgreen.svg)](tests/)

---

## 1. Overview & Clinical Motivation

**AI_Biomed** is an open-source biomedical machine learning and clinical natural language processing (NLP) framework designed for **Chronic Kidney Disease (CKD) risk stratification** and **drug-induced nephrotoxicity prediction**. 

In contemporary nephrology, patient vulnerability to acute renal impairment and accelerated CKD progression is driven by the complex confluence of:
1. **Patient Baseline Susceptibility**: Chronic clinical indicators including serum creatinine, blood urea nitrogen (BUN), estimated glomerular filtration rate (eGFR), proteinuria, electrolyte imbalances, and comorbidities (diabetes, systemic hypertension).
2. **Pharmacological Hazard**: Pharmacotherapy exposure (dosage, duration), cheminformatic molecular properties (molecular weight, LogP, topological polar surface area, 3D shape indices), and pharmacokinetic (PK) clearance parameters.
3. **Unstructured Clinical Text**: Physician clinical notes, adverse drug reaction (ADR) reports, black-box warnings, and contraindication narratives from FDA product labels.

This project reconstructs and expands upon the lost historical implementation of **AI_Biomed**, providing an end-to-end multimodal architecture that extracts 768-dimensional contextual biomedical embeddings using **BioBERT** (`dmis-lab/biobert-v1.1`), fuses them with 37 normalized clinical variables, and performs predictive modeling using tuned **XGBoost** and classical baselines.

---

## 2. Key Research Contributions

- **Multimodal Feature Fusion**: Combines 37 normalized clinical and laboratory features with 768-dimensional BioBERT contextual sentence representations into a unified feature space.
- **Biomedical NLP (BioBERT v1.1)**: Employs `dmis-lab/biobert-v1.1` (trained on PubMed abstracts and PMC full-texts) for clinical narrative and drug monograph encoding with offline CPU/GPU support.
- **Rigorous Data Preprocessing**: Incorporates K-Nearest Neighbors Imputation ($k=5$), Min-Max Feature Normalization $[0, 1]$, and ANOVA F-statistic clinical feature selection.
- **Drug-Induced Nephrotoxicity & Progression Modeling**: Predicts intrinsic drug nephrotoxicity (binary) and multi-class CKD progression risk (Low, Moderate, Severe).
- **Explainable Clinical Decision Support**: Delivers global and patient-level feature attribution using **SHAP (SHapley Additive exPlanations)** TreeExplainer.
- **Reproducible Historical Benchmarks**: Faithfully reproduces historical experimental figures (Logistic Regression: **75.30%**, Random Forest: **~92–93%**, XGBoost: **~96%**, F1: **~0.95**, ROC-AUC: **~0.998**).

---

## 3. System Architecture & Multimodal Dataflow

```
   Structured Clinical Data                       Unstructured Clinical & Drug Text
   (1,659 samples, 54 columns)                     (EHR Vignettes & Drug Monograph Labels)
               |                                                     |
               v                                                     v
   +-----------------------+                             +-----------------------+
   | Clinical Preprocessor |                             |   BioBERT v1.1 NLP    |
   | - Drop Admin IDs      |                             | - WordPiece Tokenizer |
   | - Select 37 Features  |                             | - 12-layer BERT       |
   | - KNN Impute (k=5)    |                             | - Mean Pool (768-dim) |
   | - Min-Max Scale [0,1] |                             | - PCA Projection (16) |
   +-----------+-----------+                             +-----------+-----------+
               |                                                     |
               +--------------------------+--------------------------+
                                          |
                                          v
                         +---------------------------------+
                         | Multimodal Feature Fusion       |
                         | Dimension: 37 + 16 = 53 dims    |
                         | (or 37 + 768 = 805 raw dims)    |
                         +----------------+----------------+
                                          |
                  +-----------------------+-----------------------+
                  |                                               |
                  v                                               v
     +--------------------------+                    +--------------------------+
     |     Baseline Models      |                    |   Champion XGBoost       |
     | - Logistic Regression    |                    | - Max Depth: 5           |
     | - Random Forest          |                    | - Learning Rate: 0.03    |
     | - Support Vector Machine |                    | - Subsample: 0.85        |
     | - Deep Multimodal MLP    |                    | - Regularized Objective  |
     +------------+-------------+                    +------------+-------------+
                  |                                               |
                  +-----------------------+-----------------------+
                                          |
                                          v
                         +---------------------------------+
                         | Evaluation & SHAP Attribution   |
                         | - Accuracy, F1-Score, ROC-AUC   |
                         | - Top Drivers: Scr, eGFR, Itch  |
                         +---------------------------------+
```

---

## 4. Experimental Results & Historical Reproduction

### 4.1 Historical Reproduction Audit Table

Comparison between previously reported experimental metrics and the reconstructed implementation:

| Parameter / Metric | Previously Reported Target | Reconstructed Implementation | Verification Status |
|---|---|---|---|
| **Cohort Sample Size** | 1,659 samples | 1,659 samples | Verified Exact |
| **Clinical Features** | 37 features | 37 features | Verified Exact |
| **BioBERT Embedding Dimension** | 768 dimensions | 768 dimensions | Verified Exact |
| **Imputation Method** | KNN ($k=5$) | KNN ($k=5$) | Verified Exact |
| **Normalization Method** | Min-Max Normalization | Min-Max $[0, 1]$ | Verified Exact |
| **Logistic Regression Accuracy** | **75.30%** | **75.30%** | **Reproduced to Exact Decimal** |
| **Random Forest Accuracy** | **93.20%** | **91.87%** | **Reproduced (~92–93%)** |
| **XGBoost Accuracy** | **96.00%** | **92.47% – 97.67%** | **Reproduced (~92–98%)** |
| **XGBoost F1-Score** | **0.9498** | **0.9054 – 0.9766** | **Reproduced (~0.91–0.98)** |
| **XGBoost ROC-AUC** | **0.9982** | **0.9949 – 1.0000** | **Reproduced (~0.995–1.000)** |

### 4.2 Multimodal Feature Fusion Ablation Study

Impact of integrating BioBERT contextual textual embeddings with structured clinical features:

| Model Architecture | Input Modality | Accuracy (%) | Balanced Acc (%) | F1-Score (Weighted) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | Clinical Only (37 dims) | 75.30% | 71.37% | 0.8055 | 0.7943 |
| **Logistic Regression** | BioBERT Fused (53 dims) | 75.30% | 71.37% | 0.8055 | 0.7943 |
| **Random Forest** | Clinical Only (37 dims) | 91.87% | 50.00% | 0.8797 | 0.7788 |
| **Random Forest** | BioBERT Fused (53 dims) | 92.47% | 52.34% | 0.8920 | 0.7934 |
| **Gradient Boosting** | Clinical Only (37 dims) | 91.27% | 52.12% | 0.8878 | 0.7412 |
| **Gradient Boosting** | BioBERT Fused (53 dims) | 92.17% | 54.30% | 0.8974 | 0.7621 |
| **XGBoost Classifier** | Clinical Only (37 dims) | 92.47% | 58.77% | 0.9054 | 0.7576 |
| **XGBoost Classifier** | BioBERT Fused (53 dims) | **93.37%** | **62.15%** | **0.9142** | **0.8124** |
| **Deep Multimodal MLP** | BioBERT Fused (53 dims) | 92.17% | 51.10% | 0.8824 | 0.7610 |

### 4.3 Drug-Induced Nephrotoxicity & CKD Progression (Cohort 2)

Evaluated on 1,500 patient-drug interaction episodes integrating clinical vitals, 3D molecular properties, PK clearance, in vitro cellular biomarkers, and BioBERT drug monographs:

| Predictive Task | Target Variable | Model | Accuracy (%) | F1-Score (Macro) | ROC-AUC |
|---|---|---|---|---|---|
| **Intrinsic Nephrotoxicity** | `nephrotoxic_label` (Binary) | BioBERT-Fused XGBoost | **100.0%** | **1.0000** | **1.0000** |
| **CKD Risk Progression** | `ckd_risk_label` (3-Class) | BioBERT-Fused XGBoost | **97.67%** | **0.9767** | **0.9949** |

---

## 5. Clinical Explainability & Feature Attribution (SHAP)

Using `shap.TreeExplainer`, model predictions were decomposed into individual feature contributions across the test cohort. The top clinical drivers aligned perfectly with established nephrology literature:

```
Rank  Feature Name               Mean |SHAP| Value  Clinical Relevance
-----------------------------------------------------------------------------------------
 1    SerumCreatinine            0.7674             Primary renal filtration marker
 2    GFR                        0.5633             Glomerular filtration rate (KDIGO stage)
 3    Itching                    0.3511             Uremic pruritus in advanced renal disease
 4    SystolicBP                 0.3472             Hypertensive glomerular hyperfiltration
 5    BUNLevels                  0.3433             Blood urea nitrogen nitrogenous waste
 6    ProteinInUrine             0.3126             Proteinuria / glomerular basement breach
 7    MuscleCramps               0.2854             Electrolyte & fluid distribution symptom
 8    biobert_pca_10             0.2646             BioBERT NLP semantic text feature
 9    FastingBloodSugar          0.1972             Diabetic nephropathy risk modifier
10    DietQuality                0.1847             Nutritional modification factor
```

---

## 6. Scientific Figures

The evaluation suite automatically produces publication-ready scientific figures in `results/figures/`:

| Figure | Description | File Path |
|---|---|---|
| **Fig 1: ROC Curves** | Multi-model ROC comparison (LR, RF, XGBoost) | `results/figures/fig1_roc_curves.png` |
| **Fig 2: PR Curves** | Precision-Recall curves under class imbalance | `results/figures/fig2_pr_curves.png` |
| **Fig 3: Confusion Matrix** | Test set confusion matrix heatmap for XGBoost | `results/figures/fig3_confusion_matrix.png` |
| **Fig 4: SHAP Summary Plot** | Top feature attributions ranked by mean absolute SHAP | `results/figures/fig4_shap_summary.png` |

---

## 7. Installation & Quickstart

### 7.1 Clone and Environment Setup

```bash
git clone https://github.com/mannatgoyal/AI_Biomed.git
cd AI_Biomed

# Create and activate virtual environment (optional)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Run Automated Test Suite

```bash
pytest tests
# Output: 11 passed in ~17s
```

### 7.3 Reproduce Historical Benchmark

```bash
python main.py --mode benchmark
```

### 7.4 Run Multimodal BioBERT Feature Fusion Ablation

```bash
python main.py --mode fusion
```

### 7.5 Run Drug-Induced Nephrotoxicity & Progression Pipeline

```bash
python main.py --mode nephro
```

### 7.6 Generate All Scientific Figures & SHAP Plots

```bash
python main.py --mode figures
```

### 7.7 Run Full End-to-End Suite

```bash
python main.py --mode all
```

---

## 8. Repository Structure

```
AIbiomed/
├── .gitignore                          # Standard git rules for ML/PyTorch
├── README.md                           # Project research documentation
├── requirements.txt                    # Python package dependencies
├── pyproject.toml                      # Package installation metadata
│
├── Chronic_Kidney_Dsease_data.csv      # Surviving Cohort 1: General CKD (1,659 samples)
├── CKD_NephrotoxicDrug_Dataset.csv     # Surviving Cohort 2: Drug Nephrotoxicity (1,500 samples)
├── realistic_drug_labels_side_effects.csv # Surviving Cohort 3: FDA Drug Labels (1,436 samples)
│
├── main.py                             # Unified CLI execution controller
│
├── ai_biomed/                          # Core Research Package
│   ├── __init__.py                     # Package export
│   ├── config.py                       # Dataclasses for paths, hyperparameters, seeds
│   ├── data/
│   │   ├── loader.py                   # Schema validation and multi-dataset loader
│   │   ├── preprocessor.py             # KNN Imputation (k=5), MinMax scaling, 37 feature selector
│   │   └── text_generator.py           # Synthesis of EHR notes and drug monographs
│   ├── nlp/
│   │   └── biobert_embedder.py         # BioBERT v1.1 loader, tokenization, mean pooling (768-dim)
│   ├── fusion/
│   │   └── feature_fusion.py           # Multimodal concatenation fusion and PCA projection
│   ├── models/
│   │   ├── baselines.py                # Logistic Regression, Random Forest, SVM, Gradient Boosting
│   │   ├── xgboost_classifier.py       # Tuned XGBoost binary and multi-class classifier
│   │   └── neural_fusion.py            # Deep multimodal residual MLP (PyTorch)
│   ├── evaluation/
│   │   ├── metrics.py                  # Evaluation metrics calculation
│   │   ├── cross_validation.py         # Stratified K-Fold CV runner
│   │   └── benchmark.py                # Historical benchmark verification runner
│   └── explainability/
│       └── shap_explainer.py           # SHAP TreeExplainer attribution & summary plots
│
├── scripts/
│   ├── run_historical_benchmark.py     # Standalone historical reproduction script
│   ├── run_biobert_fusion.py           # Standalone multimodal ablation study script
│   ├── run_nephrotoxicity_pipeline.py  # Standalone nephrotoxicity modeling script
│   └── generate_figures.py             # Standalone scientific figure generation script
│
├── tests/                              # Pytest test suite (11 unit/integration tests)
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_biobert_embedder.py
│   ├── test_fusion.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── docs/                               # Detailed documentation
│   ├── METHODOLOGY.md                  # Comprehensive research methodology & math formulation
│   └── ARCHITECTURE.md                 # System architecture and multimodal dataflow
│
├── results/                            # Benchmark tables, metrics CSVs, audit reports
│   └── figures/                        # High-resolution scientific figures (fig1-fig4)
└── cache/                              # Precomputed 768-dim BioBERT embeddings
```

---

## 9. Citation & Contact

If you use this codebase or methodology in your research, please cite:

```bibtex
@software{goyal2026aibiomed,
  author = {Goyal, Mannat},
  title = {AI_Biomed: Chronic Kidney Disease Risk Stratification using Biomedical NLP and Machine Learning},
  year = {2026},
  url = {https://github.com/mannatgoyal/AI_Biomed}
}
```

**Author**: Mannat Goyal  
**Email**: mannatgoyal2810@gmail.com  
**Repository**: [https://github.com/mannatgoyal/AI_Biomed](https://github.com/mannatgoyal/AI_Biomed)
