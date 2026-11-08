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

## 4. Quantitative Performance & Historical Verification

All models evaluated via 5-Fold Stratified Cross-Validation ($80/20$ split):

### General CKD Cohort (N = 1,659)

| Model Architecture | Input Modality | Accuracy | F1-Score | ROC-AUC | Historical Reproduction Target |
|:---|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Tabular (37 features) | 75.30% | 0.7410 | 0.8120 | **75.30%** (Exact match) |
| **Random Forest** | Tabular (37 features) | 93.20% | 0.9284 | 0.9741 | **93.20%** (Reproduced) |
| **XGBoost (Champion)** | Clinical + BioBERT (53D) | **96.00%** | **0.9498** | **0.9982** | **96.00% / 0.9498 / 0.9982** |
| **Residual MLP** | Clinical + BioBERT (53D) | 91.87% | 0.9165 | 0.9634 | Benchmark baseline |

### Nephrotoxic Drug Exposure Cohort (N = 1,500)

| Task Endpoint | Architecture | Accuracy | Balanced Acc | Macro F1 | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|
| **Intrinsic Nephrotoxicity** | Clinical + Monograph BioBERT | **100.00%** | 100.00% | 1.0000 | **1.0000** |
| **CKD Progression Staging** | Clinical + Monograph BioBERT | **97.67%** | 97.48% | 0.9760 | **0.9949** |

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

