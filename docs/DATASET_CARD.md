# Dataset Card: AI_Biomed Clinical & Pharmacological Cohorts

## 1. Dataset Summary

The **AI_Biomed** framework synthesizes and processes three complementary biomedical datasets representing patient demographics, laboratory biomarkers, medication exposures, and pharmacological label documentation:

| Dataset Identifier | File Path | Records | Dimensions | Task Role |
|:---|:---|:---:|:---:|:---|
| **General CKD Cohort** | `Chronic_Kidney_Dsease_data.csv` | 1,659 | 54 columns | General CKD risk stratification & historical benchmark reproduction |
| **Nephrotoxicity Cohort** | `CKD_NephrotoxicDrug_Dataset.csv` | 1,500 | 37 columns | Drug-induced kidney disease (DIKD) & multi-stage CKD progression |
| **Drug Label Corpus** | `realistic_drug_labels_side_effects.csv` | 1,436 | 15 columns | FDA/RxNorm drug monographs, contraindications, and adverse events |

---

## 2. Feature Schema & Column Definitions

### 2.1 General CKD Cohort (`Chronic_Kidney_Dsease_data.csv`)
- **Sample Count**: 1,659 patients
- **Raw Features**: 54 columns (demographics, vital signs, biochemical panels, urinalysis, lifestyle indicators)
- **Top 37 Selected Clinical Predictors**:
  - *Renal Function*: Serum Creatinine, Blood Urea Nitrogen (BUN), Glomerular Filtration Rate (eGFR), Proteinuria, Serum Albumin
  - *Electrolytes & Minerals*: Serum Sodium, Potassium, Calcium, Phosphate, Bicarbonate
  - *Hematology*: Hemoglobin, Hematocrit, Red Blood Cell Count, White Blood Cell Count, Platelet Count
  - *Metabolic & Glycemic*: Fasting Blood Glucose, HbA1c, Total Cholesterol, LDL, HDL, Triglycerides
  - *Hemodynamics*: Systolic Blood Pressure, Diastolic Blood Pressure, Mean Arterial Pressure, Pulse Rate
  - *Symptomatology*: Peripheral Edema, Uremic Pruritus (Itching), Fatigue, Nausea, Muscle Cramps
  - *Demographics & History*: Age, Gender, BMI, Smoking Status, Alcohol Intake, Family History of CKD, Hypertension Duration
- **Target Variable**: `Diagnosis` / `Class` (Binary: 0 = Low Risk / Non-CKD, 1 = Chronic Kidney Disease Positive)

### 2.2 Nephrotoxic Drug Exposure Cohort (`CKD_NephrotoxicDrug_Dataset.csv`)
- **Sample Count**: 1,500 patient-medication encounters
- **Drug Classes Represented**:
  - *High Intrinsic Nephrotoxicity*: Gentamicin, Amphotericin B, Cisplatin, Tobramycin, Vancomycin
  - *Low Intrinsic Nephrotoxicity / Analgesics*: Aspirin, Ibuprofen, Paracetamol
- **Key Measured Parameters**:
  - Baseline Creatinine, Peak Creatinine, Delta-Creatinine
  - Baseline GFR, Post-Exposure GFR, Percentage GFR Decline
  - Daily Dose (mg/day), Cumulative Treatment Duration (days)
  - Co-administration with other nephrotoxins (Binary index)
  - Urinalysis Biomarkers: Tubular epithelial casts, fractional excretion of sodium ($FE_{Na}$)
- **Target Endpoints**:
  - `Nephrotoxicity_Binary`: Intrinsic drug nephrotoxicity classification (0 vs 1)
  - `CKD_Progression_Stage`: Multi-class progression trajectory (0 = Mild/Stable, 1 = Moderate Risk, 2 = Severe Decline)

### 2.3 Pharmacological Monograph Corpus (`realistic_drug_labels_side_effects.csv`)
- **Record Count**: 1,436 drug product formulations
- **Key Textual Fields**:
  - `drug_name`, `active_ingredient`, `therapeutic_class`
  - `boxed_warning`: Black-box regulatory warnings for renal/hepatic toxicity
  - `indications_and_usage`: FDA-approved clinical indications
  - `contraindications`: Renal impairment thresholds and hypersensitivity alerts
  - `adverse_reactions`: Explicit frequencies of nephrotoxicity, acute tubular necrosis, interstitial nephritis, crystalluria
  - `drug_interactions`: Synergistic nephrotoxicity when combined with renin-angiotensin-aldosterone system (RAAS) inhibitors or loop diuretics

---

## 3. Data Cleaning, Preprocessing & Transformation

1. **Missing Data Imputation**:
   - Tabular missingness handled via K-Nearest Neighbors imputation ($k=5$, Euclidean distance weighting across normalized features).
2. **Min-Max Scaling**:
   - All continuous physiological biomarkers are bounded within $[0.0, 1.0]$:
     $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
3. **Feature Selection**:
   - Univariate ANOVA F-statistic ranking selects the top 37 most discriminatory features, aligning with historical benchmarks.
4. **NLP Vectorization**:
   - Clinical monographs and patient narrative vignettes tokenized with WordPiece (`dmis-lab/biobert-v1.1`), producing 768-dimensional contextual vector representations via mean-pooling across the last hidden state.

