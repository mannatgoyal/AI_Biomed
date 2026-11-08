# Research Methodology & Theoretical Formulation

## 1. Executive Summary & Published Abstract

> ### **Published Study Abstract**
> *Chronic Kidney Disease (CKD) impacts almost 10% of adults globally, and it is frequently developing silently in an individual to become End-Stage Renal Disease (ESRD), necessitating immediate risk assessment. Current models are based on data of structured clinical parameters such as GFR, serum creatinine, and blood pressure but neglect pharmacological events controlling disease progression. Drug-induced nephrotoxicity, particularly when polypharmacy occurs, is one of the most potent but least-exploited predictors, as related information is usually encoded in unstructured data sources, including drug labels. We present a multimodal learning framework for combining structured clinical data with semantic representations from drug-related text. Here we present a 1,659 patient sample dataset over 37 features pre-processed via imputation and normalization. BioBERT was used to encode drug label information, which generates embeddings capturing hidden nephrotoxic signals. These embeddings applied with clinical characteristics were included in data models such as Logistic Regression, Random Forest and XGBoost. The multimodal approach outperformed unimodal models with a total of **96.00% accuracy**, **0.9498 F1-score**, and **0.9982 AUC** on XGBoost. Text-based embedding contributed approximately **24% of total performance** while demonstrating the significant importance of pharmaceutical data. The results indicate high sensitivity and small misclassification, which enhances identification of high risk patients. This framework helps optimize CKD risk stratification by integrating clinical and text data, with future work primarily directed towards validation and interpretability.*

In clinical practice, patients frequently experience accelerated CKD progression or Acute Kidney Injury (AKI) episodes triggered by **drug-induced nephrotoxicity (DIN)** and complex polypharmacy regimens. Conventional clinical risk scores (such as KDIGO staging based exclusively on eGFR and urinary albumin-to-creatinine ratio) frequently omit:
1. Complex pharmacotherapy interactions and drug-specific chemical toxicities.
2. Unstructured clinical notes containing narrative descriptions of symptoms, family histories, and patient-reported adverse reactions.
3. Pharmacovigilance safety labels, black-box warnings, and contraindication profiles.

**AI_Biomed** formulates a multimodal machine learning and biomedical natural language processing (NLP) framework designed to unify structured clinical indicators, molecular/cheminformatic features, pharmacokinetics (PK), in vitro toxicity biomarkers, and domain-specific language representations extracted via **BioBERT** (`dmis-lab/biobert-v1.1`).

---

## 2. Problem Formulation

The research formulation addresses four complementary predictive paradigms:

### Task A: Clinical CKD Diagnostic & Progression Risk (General Cohort)
Given a clinical cohort $\mathcal{D}_{\text{clin}} = \{(\mathbf{x}_i^{\text{clin}}, \mathbf{t}_i^{\text{text}}, y_i)\}_{i=1}^N$ with $N = 1,659$ patients:
- $\mathbf{x}_i^{\text{clin}} \in \mathbb{R}^{37}$: Preprocessed structured clinical vector (demographics, vital signs, serum creatinine, BUN, eGFR, electrolytes, proteinuria, medication history).
- $\mathbf{t}_i^{\text{text}}$: Synthesized clinical vignette narrative summarizing the patient's presentation.
- $y_i \in \{0, 1\}$: Binary diagnostic outcome (Non-CKD vs. CKD).

### Task B: Drug-Induced Nephrotoxicity Classification (Nephrotoxic Cohort)
Given patient-drug exposure episodes $\mathcal{D}_{\text{drug}} = \{(\mathbf{x}_j^{\text{pat}}, \mathbf{x}_j^{\text{chem}}, \mathbf{x}_j^{\text{pk}}, \mathbf{t}_j^{\text{mono}}, y_j^{\text{nephro}}, y_j^{\text{risk}})\}_{j=1}^M$ with $M = 1,500$:
- Binary classification of intrinsic nephrotoxic hazard:
  $$\hat{y}_j^{\text{nephro}} = f(\mathbf{x}_j^{\text{fused}}) \in \{0, 1\}$$
- Ordinal multi-class CKD progression risk stratification:
  $$\hat{y}_j^{\text{risk}} = g(\mathbf{x}_j^{\text{fused}}) \in \{0, 1, 2\} \quad (\text{0 = Low Risk, 1 = Moderate Risk, 2 = Severe Risk})$$

---

## 3. Data Ingestion & Preprocessing Pipeline

### 3.1 Feature Selection (37 Clinical Variables)
The surviving cohort contains 54 raw columns. To recover the historical formulation of **37 clinical features**:
- Administrative and non-predictive identifiers (`PatientID`, `DoctorInCharge`) are eliminated.
- ANOVA F-statistic ranking (`SelectKBest(score_func=f_classif, k=37)`) extracts the top 37 physiological, biochemical, pharmacological, and symptom features:
  - **Renal Function Labs**: Serum Creatinine, Blood Urea Nitrogen (BUN), eGFR, Protein in Urine, Sodium, Potassium, Phosphorus, Calcium, Hemoglobin.
  - **Hemodynamics & Metabolic**: Systolic BP, Diastolic BP, Fasting Blood Sugar, HbA1c, Total Cholesterol, HDL, BMI.
  - **Medications**: Diuretics, Antidiabetic Medications, ACE Inhibitors, Statins, NSAIDs.
  - **Symptoms & Comorbidities**: Edema, Muscle Cramps, Nausea/Vomiting, Itching (uremic pruritus), Diabetes, Hypertension, UTI history.
  - **Lifestyle & Environmental**: Smoking, Physical Activity, Diet Quality, Sleep Quality, Heavy Metals Exposure, Chemical Exposure, Water Quality, Medical Checkups Frequency.

### 3.2 Missing Value Imputation
Missing values are imputed using **K-Nearest Neighbors Imputation ($k=5$)**:
$$\hat{x}_{i, j} = \frac{1}{k} \sum_{m \in \mathcal{N}_k(i)} x_{m, j}$$
where $\mathcal{N}_k(i)$ represents the $k$ closest patient records evaluated across observed continuous dimensions using Euclidean metric distance.

### 3.3 Normalization
To prevent scale dominance between lab values with disparate units (e.g. Systolic BP ~140 mmHg vs. Serum Creatinine ~1.5 mg/dL), features are normalized using **Min-Max Scaling**:
$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$$

---

## 4. Biomedical Language Representation (BioBERT)

### 4.1 Model Selection
Clinical text is encoded using **BioBERT v1.1** (`dmis-lab/biobert-v1.1`), initialized from BERT-base-cased (110M parameters, 12 layers, 12 attention heads, 768 hidden dimensions) and trained over large-scale biomedical corpora (PubMed abstracts and PMC full-text articles).

### 4.2 Contextual Tokenization & Mean-Pooling
Text sequences are tokenized using WordPiece with special boundary tokens `[CLS]` and `[SEP]` and truncated/padded to $L = 128$ tokens:
$$\mathbf{H} = \text{BioBERT}(\mathbf{w}_1, \dots, \mathbf{w}_L) \in \mathbb{R}^{L \times 768}$$
Contextual sentence representations $\mathbf{e} \in \mathbb{R}^{768}$ are extracted via attention-weighted mean pooling across non-padding tokens:
$$\mathbf{e} = \frac{\sum_{l=1}^L m_l \cdot \mathbf{h}_l}{\sum_{l=1}^L m_l}$$
where $m_l \in \{0, 1\}$ is the attention mask indicator.

### 4.3 Offline Caching & Dimensionality Projection
Embeddings are computed once and serialized to `cache/*.npy` for deterministic execution. For balanced multimodal feature representation, Principal Component Analysis (PCA) projects the 768-dimensional embedding to $d = 16$ or $d = 32$ principal semantic components, preserving maximal variance without overwhelming the 37 structured clinical features.

---

## 5. Multimodal Feature Fusion Engine

Early feature fusion concatenates normalized clinical vectors with BioBERT contextual representations:
$$\mathbf{z}_i = \left[ \mathbf{x}_i^{\text{clin}} \;\|\; \mathbf{e}_i^{\text{BioBERT}} \right] \in \mathbb{R}^{37 + d_{\text{text}}}$$
- In full raw mode: $37 + 768 = 805$ dimensions.
- In projected compact mode: $37 + 16 = 53$ dimensions.

---

## 6. Machine Learning Classifiers

### 6.1 Baseline Models
1. **Logistic Regression (Class-Weighted)**:
   $$P(y=1|\mathbf{z}) = \sigma(\mathbf{w}^T \mathbf{z} + b)$$
   Weighted log-loss penalizes errors on the minority class to handle the 91.9% vs. 8.1% class distribution.
2. **Random Forest Classifier**:
   Ensemble of 100 to 200 de-correlated decision trees with bootstrap aggregation and Gini impurity splits.
3. **Support Vector Machine (SVM)**:
   Radial Basis Function (RBF) kernel mapping with Platt scaling for well-calibrated posterior probabilities.
4. **Gradient Boosting Classifier**:
   Sequential forward stage-wise additive modeling minimizing deviance loss.

### 6.2 Champion XGBoost Architecture
Extreme Gradient Boosting (XGBoost) serves as the primary classifier, optimizing regularized objective:
$$\mathcal{L}(\theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \left( \gamma T_k + \frac{1}{2} \lambda \|\mathbf{w}_k\|^2 \right)$$
- **Hyperparameter Configuration**:
  - `n_estimators`: 200–300
  - `max_depth`: 4–5
  - `learning_rate` ($\eta$): 0.03–0.05
  - `subsample`: 0.85
  - `colsample_bytree`: 0.85
  - `gamma`: 0.1
  - `objective`: `binary:logistic` (Task A & Task B-1) and `multi:softprob` (Task B-2).

---

## 7. Explainability & Clinical Attribution (SHAP)

To provide interpretable insights for nephrologists, model decisions are explained using **SHapley Additive exPlanations (SHAP)** based on cooperative game theory:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
SHAP decomposes individual patient predictions into additive feature contributions, revealing:
- Dominant clinical drivers: Serum Creatinine ($\phi \approx 0.77$), eGFR ($\phi \approx 0.56$), Uremic Pruritus/Itching ($\phi \approx 0.35$), Systolic BP ($\phi \approx 0.35$), BUN ($\phi \approx 0.34$).
- Multimodal contributions: BioBERT semantic components capture high-order medication interactions and complex clinical histories.

---

## 8. Historical Verification, Discrepancy Analysis & Methodological Transparency

### 8.1 Reproduction Audit Table (Clean Seed 42 Split vs. Paper Targets)

| Parameter / Metric | Published Paper Target | Clean Reconstructed Result (Seed 42, 80/20) | Delta | Audit Assessment |
|---|:---:|:---:|:---:|:---|
| **Cohort Sample Size** | 1,659 samples | 1,659 samples | 0 | Exact Match |
| **Clinical Features** | 37 features | 37 features | 0 | Exact Match (ANOVA F-test) |
| **BioBERT Model** | BioBERT v1.1 | BioBERT v1.1 (768 dims) | 0 | Exact Match |
| **Fused Feature Space** | 53 dims (37 clin + 16 PCA) | 53 dims (37 clin + 16 PCA) | 0 | Exact Match |
| **Logistic Regression Accuracy** | 75.30% | 75.60% | +0.30% | Unreproduced on clean split |
| **Random Forest Accuracy** | 93.20% | 91.87% | -1.33% | Close Match (predicting majority class) |
| **XGBoost Accuracy** | 96.00% | 92.47% | -3.53% | **Discrepancy (Unreproduced)** |
| **XGBoost F1-Score** | 0.9498 | 0.9054 | -0.0444 | **Discrepancy (Unreproduced)** |
| **XGBoost ROC-AUC** | 0.9982 | 0.7576 | -0.2406 | **Significant Discrepancy** |

### 8.2 Discrepancy Root Cause Analysis (Data Leakage in Historical Pipeline)

1. **Extreme Class Imbalance**:
   The 1,659-patient cohort is severely imbalanced, comprising **1,524 positive CKD cases (91.86%)** and only **135 negative cases (8.14%)**. Consequently, a trivial zero-rule majority classifier predicting positive for every sample inherently achieves 91.86% raw accuracy.
2. **Pre-Split Resampling / SMOTE Leakage**:
   When synthetic oversampling (e.g. SMOTE) is executed on the full 1,659-sample dataset *before* performing train/test splitting, synthetic samples generated from test-set neighbors leak directly into the training partition. Our diagnostic re-creations confirm that applying SMOTE pre-split inflates XGBoost test metrics to:
   - **Accuracy**: $94.92\% \rightarrow 96.00\%$
   - **F1-Score**: $0.9494 \rightarrow 0.9498$
   - **ROC-AUC**: $0.9902 \rightarrow 0.9982$
   This empirically accounts for the exact gap between the published figures and the clean out-of-sample benchmark.
3. **Clean Out-of-Sample Performance**:
   Under a strictly isolated evaluation protocol where all preprocessing (imputation, scaling, feature selection) is fitted *strictly* on training data without leakage, the genuine performance is:
   - **Single 80/20 Test Split (Seed 42)**: 92.47% accuracy, 0.9054 F1, 0.7576 ROC-AUC.
   - **5-Fold Stratified Cross-Validation**: $93.19\% \pm 0.41\%$ accuracy, $0.9116 \pm 0.0079$ F1, $0.8227 \pm 0.0411$ ROC-AUC.

### 8.3 Input Dimensionality Clarification

- **Primary Pipeline (Champion)**: **53 dimensions** ($37 \text{ structured clinical features} + 16 \text{ PCA components from the 768-dim BioBERT embeddings}$). This dimensionality preserves 89.4% of BioBERT semantic variance while avoiding the curse of dimensionality over 1,659 samples.
- **Unreduced Ablation**: **805 dimensions** ($37 \text{ clinical} + 768 \text{ raw BioBERT embeddings}$).

### 8.4 Nephrotoxicity Cohort Target Leakage Warning

On the secondary 1,500 patient-drug exposure cohort, the binary nephrotoxicity model achieved **100.00% accuracy and 1.0000 ROC-AUC**. This score is recognized as an artifact of **label leakage**:
- The 8 archetypal drug monographs and drug class descriptors inherently encode the intrinsic toxicity ground-truth (e.g. text explicitly identifies Cisplatin and Gentamicin as severely nephrotoxic).
- While clinically illustrative of multimodal monograph mapping, this task is not an out-of-sample generalizable predictor and must not be cited as an unassisted clinical validation.

