# Model Card: AI_Biomed Multimodal Clinical Decision Support Framework

## 1. Model Details

- **Model Name**: AI_Biomed Multimodal CKD & Nephrotoxicity Risk Stratifier
- **Version**: 1.0.0
- **Model Type**: Multimodal Hybrid Pipeline combining Contextual Biomedical Language Representations (BioBERT v1.1) and Gradient-Boosted Decision Trees (XGBoost)
- **Base Language Model**: `dmis-lab/biobert-v1.1` (PubMed 1M + PMC 270K pre-trained Transformer encoder)
- **Tabular Classifiers**:
  - `GeneralCKDClassifier`: XGBoost with TreeExplainer integration (`n_estimators=200`, `max_depth=6`, `learning_rate=0.05`)
  - `NephrotoxicityClassifier`: Multi-target gradient boosting for intrinsic nephrotoxicity (binary) and CKD progression risk (ordinal 3-stage)
  - `Baselines`: Class-balanced Logistic Regression, Random Forest, Support Vector Classifier, Residual MLP
- **License**: MIT License
- **Primary Developer**: Mannat Goyal (<mannatgoyal2810@gmail.com>)
- **Repository**: [https://github.com/mannatgoyal/AI_Biomed](https://github.com/mannatgoyal/AI_Biomed)

---

## 2. Intended Use

### Primary Intended Uses
- **Clinical Decision Support (CDSS)**: Providing nephrologists and hospital clinicians with second-opinion risk assessments for Chronic Kidney Disease progression.
- **Drug Safety & Nephrotoxicity Pharmacovigilance**: Screening medication regimens (e.g., aminoglycosides, NSAIDs, platinum chemotherapeutics) against patient baseline renal function to mitigate Drug-Induced Kidney Disease (DIKD).
- **Multimodal Biomedical Research**: Serving as an open, reproducible benchmark for fusing unstructured clinical text/drug monographs with structured electronic health records.

### Out-of-Scope & Disallowed Uses
- **Autonomous Clinical Diagnosis**: This model is strictly a supportive analytical tool. It is **not** cleared by the FDA/EMA for autonomous clinical diagnosis without licensed physician oversight.
- **Pediatric Nephrology**: The underlying cohorts reflect adult physiological distributions. Model predictions are uncalibrated for pediatric renal physiology.
- **Acute Kidney Injury (AKI) Emergency Resuscitation**: While serum creatinine spikes correlate with renal injury, the primary training endpoints focus on chronic staging and subacute nephrotoxic drug exposure.

---

## 3. Factors and Subgroups

- **Demographics**: Evaluated across age brackets (18–90 years), biological sex distributions, and common metabolic comorbidities (Type 2 Diabetes, Hypertension, Coronary Artery Disease).
- **Renal Stages**: Stratified across Stage 1–2 (Mild), Stage 3 (Moderate), and Stage 4–5 (Severe/End-Stage Renal Disease).
- **Confounding Variables**: Evaluated against medication classes with known tubular, glomerular, and interstitial toxicity profiles (e.g., Aminoglycosides, Amphotericin B, Platinum agents).

---

## 4. Quantitative Performance, Discrepancy Analysis & Benchmarks

### General CKD Cohort ($N = 1,659$, Clean Out-of-Sample Evaluation, Seed 42)

| Model Architecture | Input Modality | Clean Reconstructed Acc | Clean F1 | Clean ROC-AUC | Paper Reported Target | Methodological Assessment |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Logistic Regression** | Clinical Only (37D) | 75.60% | 0.8089 | 0.8080 | 75.30% | Close to baseline target |
| **Random Forest** | BioBERT Fused (53D) | 91.87% | 0.8797 | 0.7934 | 93.20% | Reflects majority class (91.86%) |
| **XGBoost (Champion)** | Clinical + BioBERT (53D) | **92.47%** | **0.9054** | **0.7576** | **96.00% / 0.9498 / 0.9982** | **Discrepancy (Unreproduced on clean split)** |
| **5-Fold CV XGBoost** | Clinical + BioBERT (53D) | 93.19% ± 0.41% | 0.9116 ± 0.008 | 0.8227 ± 0.041 | — | Out-of-fold generalization |

> [!NOTE]
> **Discrepancy Analysis**: The paper's reported 96.00% accuracy and 0.9982 AUC were reproduced when pre-split resampling (SMOTE) was applied across the cohort prior to splitting (yielding 94.92% accuracy, 0.9494 F1, and 0.9902 AUC). Under strictly clean out-of-sample testing with no leakage, true generalization is 92.47% accuracy and 0.7576–0.8227 AUC.

### Nephrotoxic Drug Exposure Cohort ($N = 1,500$, Secondary Demonstration)

| Task Endpoint | Architecture | Accuracy | Macro F1 | ROC-AUC | Methodological Caveat |
|:---|:---|:---:|:---:|:---:|:---|
| **Intrinsic Nephrotoxicity** | Clinical + Monograph BioBERT | 100.00% | 1.0000 | 1.0000 | **Target Leakage**: Monograph text directly encodes toxicity ground-truth. |
| **CKD Progression Staging** | Clinical + Monograph BioBERT | 97.67% | 0.9767 | 0.9949 | Correlated drug exposure proxy variables. |

---

## 5. Explainability & Interpretability

Feature attribution is computed via TreeExplainer SHAP ($\phi_i$):
- **Top Positive Drivers of Risk**: Serum Creatinine ($\phi=+0.767$), Systolic BP ($\phi=+0.347$), Blood Urea Nitrogen ($\phi=+0.343$), Proteinuria ($\phi=+0.298$).
- **Top Protective Drivers**: Glomerular Filtration Rate (eGFR, $\phi=-0.563$), Hemoglobin ($\phi=-0.214$), Serum Albumin ($\phi=-0.189$).
- **BioBERT Semantic Contributions**: Monograph semantic embeddings ($\text{BioBERT\_PCA}_{1-4}$) contribute $\approx 12.4\%$ of total attribution mass, elevating recall in atypical drug-drug interaction presentations.

---

## 6. Ethical Considerations & Limitations

1. **Synthetic & Semi-Synthetic Data Footprint**: Portions of the patient EHR monographs are synthesized using clinical templates to avoid HIPAA/GDPR PHI exposure while maintaining biomedical realism.
2. **Alert Fatigue Mitigation**: The Streamlit CDSS application provides explicit probability calibration thresholds to prevent false-alarm fatigue in high-throughput outpatient clinics.
3. **External Generalizability**: The model should undergo site-specific external validation prior to hospital deployment to account for variations in assay calibration (e.g., Jaffe vs. enzymatic creatinine assays).

