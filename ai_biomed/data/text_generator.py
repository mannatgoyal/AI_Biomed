"""Clinical narrative, EHR vignette, and drug monograph generator for BioBERT embedding.

Synthesizes rich biomedical textual descriptions from structured records to enable
meaningful contextual representations from biomedical language models like BioBERT.
"""

from typing import List, Dict
import pandas as pd


# Comprehensive clinical pharmacology and nephrotoxicity monographs for the 8 archetypal drugs
DRUG_PHARMACOLOGY_MONOGRAPHS: Dict[str, str] = {
    "Gentamicin": (
        "Gentamicin is an aminoglycoside antibiotic indicated for severe Gram-negative bacillary infections. "
        "Nephrotoxicity is a major dose-limiting adverse effect, characterized by proximal tubular cell necrosis, "
        "lysosomal phospholipidosis, decreased glomerular filtration rate (GFR), and acute renal failure. "
        "Requires therapeutic drug monitoring and dosage adjustment in renal impairment."
    ),
    "Cisplatin": (
        "Cisplatin is a potent platinum-based chemotherapeutic agent for solid malignancies. "
        "Induces severe dose-dependent nephrotoxicity through crosslinking of nuclear and mitochondrial DNA, "
        "reactive oxygen species (ROS) generation, and proximal tubular apoptosis and necrosis. "
        "Clinical management requires aggressive pre- and post-hydration with renal function surveillance."
    ),
    "Ibuprofen": (
        "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) acting as a non-selective inhibitor "
        "of cyclooxygenase enzymes (COX-1 and COX-2). Renal adverse effects arise from inhibition of renal "
        "prostaglandin synthesis, impairing compensatory vasodilatation and precipitating acute interstitial nephritis, "
        "reduced renal clearance, edema, and hyperkalemia."
    ),
    "Aspirin": (
        "Aspirin (acetylsalicylic acid) is an analgesic, antipyretic, and antiplatelet agent that irreversibly "
        "inhibits COX-1 and COX-2. Renally excreted; chronic high-dose exposure or toxic accumulation can cause "
        "renal vasoconstriction, papillary necrosis, or mild tubular changes, though standard antiplatelet doses carry low nephrotoxic risk."
    ),
    "Paracetamol": (
        "Paracetamol (acetaminophen) is an analgesic and antipyretic with central analgesic properties. "
        "Primarily metabolized in hepatocytes via glucuronidation and sulfation. Nephrotoxicity is uncommon at therapeutic doses "
        "but can occur secondary to massive toxic overdose through accumulation of N-acetyl-p-benzoquinone imine (NAPQI) "
        "causing acute tubular necrosis."
    ),
    "Tobramycin": (
        "Tobramycin is an aminoglycoside antibiotic primarily prescribed for Pseudomonas aeruginosa and other serious "
        "aerobic Gram-negative infections. Actively accumulates within renal cortical cells causing disruption of membrane "
        "transport, mitochondrial damage, acute tubular injury, and progressive decrease in creatinine clearance."
    ),
    "Amphotericin-B": (
        "Amphotericin-B is a polyene antifungal agent indicated for progressive, life-threatening systemic fungal infections. "
        "Known for severe, high-incidence nephrotoxicity mediated by intense renal vasoconstriction, medullary ischemia, "
        "and direct membrane disruption in distal renal tubules leading to hypokalemia, hypomagnesemia, and renal tubular acidosis."
    ),
    "Vancomycin": (
        "Vancomycin is a tricyclic glycopeptide antibiotic for severe Gram-positive infections including MRSA. "
        "Induces oxidative proximal tubular necrosis and acute interstitial nephritis. Nephrotoxic potential increases with "
        "elevated trough concentrations, prolonged duration of therapy, and concurrent administration of other nephrotoxic agents."
    ),
}


def generate_patient_clinical_narratives(df: pd.DataFrame) -> List[str]:
    """Synthesizes structured clinical records into clinical vignette narratives for BioBERT.

    Args:
        df: DataFrame containing clinical CKD features (1,659 samples).

    Returns:
        List of formatted clinical text notes.
    """
    narratives = []
    for _, row in df.iterrows():
        age = int(row.get("Age", 50))
        gender = "Male" if row.get("Gender", 0) == 1 else "Female"
        sbp = int(row.get("SystolicBP", 120))
        dbp = int(row.get("DiastolicBP", 80))
        scr = float(row.get("SerumCreatinine", 1.0))
        gfr = float(row.get("GFR", 90.0))
        bun = float(row.get("BUNLevels", 15.0))
        protein = float(row.get("ProteinInUrine", 0.0))
        
        # Medical history
        history = []
        if row.get("FamilyHistoryDiabetes", 0) == 1:
            history.append("diabetes mellitus")
        if row.get("FamilyHistoryHypertension", 0) == 1:
            history.append("systemic hypertension")
        if row.get("PreviousAcuteKidneyInjury", 0) == 1:
            history.append("prior acute kidney injury (AKI)")
        if row.get("FamilyHistoryKidneyDisease", 0) == 1:
            history.append("familial chronic kidney disease")
        hist_str = ", ".join(history) if history else "unremarkable medical history"

        # Medications
        meds = []
        if row.get("ACEInhibitors", 0) == 1:
            meds.append("ACE inhibitor")
        if row.get("Diuretics", 0) == 1:
            meds.append("loop/thiazide diuretic")
        if row.get("NSAIDsUse", 0) > 1:
            meds.append("frequent NSAID therapy")
        if row.get("Statins", 0) == 1:
            meds.append("statin")
        if row.get("AntidiabeticMedications", 0) == 1:
            meds.append("antidiabetic medication")
        med_str = ", ".join(meds) if meds else "no chronic nephroactive medications"

        # Symptoms
        symptoms = []
        if row.get("Edema", 0) == 1:
            symptoms.append("peripheral edema")
        if row.get("FatigueLevels", 0) > 5:
            symptoms.append("clinically significant fatigue")
        if row.get("MuscleCramps", 0) > 5:
            symptoms.append("muscle cramps")
        symp_str = ", ".join(symptoms) if symptoms else "asymptomatic clinical presentation"

        text = (
            f"Patient {age}yo {gender}, BP {sbp}/{dbp} mmHg. "
            f"Laboratory findings: Serum creatinine {scr:.2f} mg/dL, estimated GFR {gfr:.1f} mL/min/1.73m2, "
            f"Blood urea nitrogen {bun:.1f} mg/dL, proteinuria {protein:.2f} g/24hr. "
            f"Relevant history includes {hist_str}. "
            f"Current pharmacotherapy regimen: {med_str}. "
            f"Clinical signs: {symp_str}. "
            f"Clinical evaluation for chronic kidney disease progression and nephrotoxic vulnerability."
        )
        narratives.append(text)
    return narratives


def generate_drug_monographs(df_nephro: pd.DataFrame) -> List[str]:
    """Generates drug-specific pharmacological monographs for patient exposure episodes.

    Args:
        df_nephro: DataFrame containing drug exposure records (1,500 samples).

    Returns:
        List of clinical pharmacological descriptions matching each row.
    """
    monographs = []
    for _, row in df_nephro.iterrows():
        drug_name = str(row.get("drug_name", "Unknown"))
        dosage = row.get("drug_dosage_mg", 0)
        days = row.get("exposure_days", 0)
        
        base_desc = DRUG_PHARMACOLOGY_MONOGRAPHS.get(
            drug_name,
            f"Pharmaceutical compound {drug_name} administered for clinical therapy."
        )
        exposure_ctx = f" Regimen: {dosage} mg daily for {days} exposure days."
        monographs.append(base_desc + exposure_ctx)
    return monographs


def generate_label_narratives(df_labels: pd.DataFrame) -> List[str]:
    """Generates FDA-style product label narrative texts from the drug labels dataset.

    Args:
        df_labels: DataFrame with realistic drug labels (1,436 samples).

    Returns:
        List of synthesized FDA label text narratives.
    """
    narratives = []
    for _, row in df_labels.iterrows():
        drug = row.get("drug_name", "Drug")
        d_class = row.get("drug_class", "Therapeutic Agent")
        indic = row.get("indications", "Clinical therapy")
        se = row.get("side_effects", "None reported")
        warn = row.get("warnings", "Standard precautions")
        contra = row.get("contraindications", "Hypersensitivity")
        
        narrative = (
            f"Drug: {drug}. Classification: {d_class}. "
            f"Indications: {indic}. "
            f"Adverse Drug Reactions and Side Effects: {se}. "
            f"Clinical Warnings and Precautions: {warn}. "
            f"Contraindications: {contra}."
        )
        narratives.append(narrative)
    return narratives

