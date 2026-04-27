"""
generate_panel.py
Generates a realistic synthetic Medicare Advantage patient panel
with chronic conditions, lab values, visit history, and SDOH factors.
Produces data that mirrors what Synthea would generate.
Run: python src/generate_panel.py
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

np.random.seed(42)

N_PATIENTS = 800  # Typical PCP panel size

# ── Clinical constants ────────────────────────────────────────────────────────

CONDITIONS = [
    "Type 2 Diabetes",
    "Hypertension",
    "Hyperlipidemia",
    "Coronary Artery Disease",
    "Heart Failure",
    "COPD",
    "CKD",
    "Depression",
    "Obesity",
    "Atrial Fibrillation",
]

CONDITION_PREVALENCE = [0.28, 0.52, 0.44, 0.12, 0.08, 0.10, 0.14, 0.18, 0.38, 0.06]

MEDICATIONS = {
    "Type 2 Diabetes": ["Metformin", "Semaglutide", "Empagliflozin", "Insulin Glargine"],
    "Hypertension": ["Lisinopril", "Amlodipine", "Metoprolol", "Losartan"],
    "Hyperlipidemia": ["Atorvastatin", "Rosuvastatin", "Ezetimibe"],
    "Coronary Artery Disease": ["Aspirin", "Clopidogrel", "Atorvastatin"],
    "Heart Failure": ["Carvedilol", "Furosemide", "Spironolactone", "Sacubitril/Valsartan"],
    "COPD": ["Tiotropium", "Fluticasone/Salmeterol", "Albuterol"],
    "CKD": ["Lisinopril", "Dapagliflozin", "Sodium Bicarbonate"],
    "Depression": ["Sertraline", "Escitalopram", "Bupropion"],
    "Obesity": ["Semaglutide", "Phentermine/Topiramate"],
    "Atrial Fibrillation": ["Apixaban", "Metoprolol", "Digoxin"],
}

QUALITY_MEASURES = [
    "HbA1c_controlled",
    "BP_controlled",
    "LDL_controlled",
    "Annual_wellness_visit",
    "Diabetic_eye_exam",
    "Diabetic_kidney_screening",
    "Colorectal_screening",
    "Mammogram",
    "Flu_vaccine",
    "Pneumococcal_vaccine",
]

INSURANCE_TYPES = ["Medicare Advantage", "Medicare FFS", "Dual Eligible"]
INSURANCE_WEIGHTS = [0.60, 0.28, 0.12]

ZIP_INCOME_QUINTILES = [1, 2, 3, 4, 5]
RACE_ETHNICITY = ["White", "Black", "Hispanic", "Asian", "Other"]
RACE_WEIGHTS = [0.63, 0.14, 0.13, 0.06, 0.04]


# ── Helper functions ──────────────────────────────────────────────────────────

def generate_age():
    """Medicare Advantage skews 65–85."""
    return int(np.clip(np.random.normal(72, 7), 65, 95))


def generate_conditions(age):
    """Assign chronic conditions — prevalence increases with age and multimorbidity."""
    age_multiplier = 1 + (age - 65) / 60
    conditions = []
    for condition, prev in zip(CONDITIONS, CONDITION_PREVALENCE):
        adjusted = min(prev * age_multiplier, 0.85)
        if np.random.random() < adjusted:
            conditions.append(condition)
    return conditions


def generate_lab_values(conditions, adherent=True):
    """Generate realistic lab values based on conditions and adherence."""
    labs = {}

    # HbA1c
    if "Type 2 Diabetes" in conditions:
        if adherent:
            labs["HbA1c"] = round(np.clip(np.random.normal(7.4, 0.8), 5.5, 12.0), 1)
        else:
            labs["HbA1c"] = round(np.clip(np.random.normal(9.1, 1.2), 7.0, 14.0), 1)
    else:
        labs["HbA1c"] = round(np.clip(np.random.normal(5.6, 0.4), 4.5, 6.4), 1)

    # Blood pressure
    if "Hypertension" in conditions:
        if adherent:
            labs["SBP"] = int(np.clip(np.random.normal(128, 10), 100, 160))
            labs["DBP"] = int(np.clip(np.random.normal(78, 8), 60, 100))
        else:
            labs["SBP"] = int(np.clip(np.random.normal(148, 14), 120, 200))
            labs["DBP"] = int(np.clip(np.random.normal(88, 10), 70, 120))
    else:
        labs["SBP"] = int(np.clip(np.random.normal(120, 8), 90, 140))
        labs["DBP"] = int(np.clip(np.random.normal(74, 6), 55, 90))

    # LDL
    if "Hyperlipidemia" in conditions or "Coronary Artery Disease" in conditions:
        if adherent:
            labs["LDL"] = int(np.clip(np.random.normal(85, 20), 40, 140))
        else:
            labs["LDL"] = int(np.clip(np.random.normal(130, 30), 80, 220))
    else:
        labs["LDL"] = int(np.clip(np.random.normal(105, 25), 60, 180))

    # eGFR
    if "CKD" in conditions:
        labs["eGFR"] = int(np.clip(np.random.normal(42, 12), 15, 59))
    else:
        labs["eGFR"] = int(np.clip(np.random.normal(78, 15), 45, 120))

    # BMI
    if "Obesity" in conditions:
        labs["BMI"] = round(np.clip(np.random.normal(36, 4), 30, 55), 1)
    else:
        labs["BMI"] = round(np.clip(np.random.normal(27, 4), 18, 35), 1)

    return labs


def generate_visit_history(age, conditions, adherent):
    """Generate visit and quality measure completion history."""
    today = datetime.today()
    n_conditions = len(conditions)

    # Days since last PCP visit
    if adherent:
        days_since_pcp = int(np.clip(np.random.exponential(90), 7, 365))
    else:
        days_since_pcp = int(np.clip(np.random.exponential(200), 30, 730))

    last_pcp_visit = (today - timedelta(days=days_since_pcp)).strftime("%Y-%m-%d")

    # ER visits last 12 months (higher with more conditions and non-adherence)
    er_base = 0.3 * n_conditions / len(CONDITIONS)
    if not adherent:
        er_base *= 2.5
    er_visits_12m = np.random.poisson(er_base)

    # Hospitalizations last 12 months
    hosp_base = 0.1 * n_conditions / len(CONDITIONS)
    if not adherent:
        hosp_base *= 3
    hospitalizations_12m = np.random.poisson(hosp_base)

    # Quality measure completion
    quality_gaps = []
    quality_completed = []

    for measure in QUALITY_MEASURES:
        # Only apply relevant measures
        if measure == "HbA1c_controlled" and "Type 2 Diabetes" not in conditions:
            continue
        if measure == "Diabetic_eye_exam" and "Type 2 Diabetes" not in conditions:
            continue
        if measure == "Diabetic_kidney_screening" and "Type 2 Diabetes" not in conditions:
            continue
        if measure == "BP_controlled" and "Hypertension" not in conditions:
            continue
        if measure == "LDL_controlled" and "Hyperlipidemia" not in conditions:
            continue

        completion_prob = 0.75 if adherent else 0.35
        if np.random.random() < completion_prob:
            quality_completed.append(measure)
        else:
            quality_gaps.append(measure)

    return {
        "last_pcp_visit": last_pcp_visit,
        "days_since_pcp_visit": days_since_pcp,
        "er_visits_12m": int(er_visits_12m),
        "hospitalizations_12m": int(hospitalizations_12m),
        "quality_gaps": quality_gaps,
        "quality_completed": quality_completed,
        "n_quality_gaps": len(quality_gaps),
    }


def compute_risk_score(patient):
    """
    Composite risk score (0–100) used as the model target proxy.
    Higher = more likely to benefit from proactive outreach.
    Combines: condition burden, lab control, visit recency, quality gaps, ER utilization.
    """
    score = 0

    # Condition burden (0–30)
    score += min(len(patient["conditions"]) * 4, 30)

    # Lab control (0–25)
    labs = patient["labs"]
    if "HbA1c" in labs and labs["HbA1c"] > 8.0:
        score += 12
    elif "HbA1c" in labs and labs["HbA1c"] > 7.0:
        score += 6
    if labs.get("SBP", 0) > 140:
        score += 8
    elif labs.get("SBP", 0) > 130:
        score += 4
    if labs.get("LDL", 0) > 130:
        score += 5

    # Visit recency (0–20)
    days = patient["visit_history"]["days_since_pcp_visit"]
    if days > 365:
        score += 20
    elif days > 180:
        score += 12
    elif days > 90:
        score += 5

    # Quality gaps (0–15)
    score += min(patient["visit_history"]["n_quality_gaps"] * 4, 15)

    # Utilization (0–10)
    score += min(patient["visit_history"]["er_visits_12m"] * 3, 8)
    score += min(patient["visit_history"]["hospitalizations_12m"] * 4, 10)

    # SDOH penalty (0–5)
    if patient["sdoh"]["zip_income_quintile"] <= 2:
        score += 3
    if patient["insurance"] == "Dual Eligible":
        score += 2

    return min(int(score), 100)


def generate_outreach_priority(risk_score, conditions, visit_history):
    """Classify into actionable priority tiers."""
    gaps = visit_history["n_quality_gaps"]
    days_since = visit_history["days_since_pcp_visit"]

    if risk_score >= 70 or visit_history["hospitalizations_12m"] >= 2:
        return "Critical"
    elif risk_score >= 50 or (gaps >= 3 and days_since > 180):
        return "High"
    elif risk_score >= 30 or gaps >= 2:
        return "Moderate"
    else:
        return "Routine"


def assign_medications(conditions):
    """Assign 1–3 medications per condition."""
    meds = set()
    for condition in conditions:
        if condition in MEDICATIONS:
            n = np.random.randint(1, min(3, len(MEDICATIONS[condition])) + 1)
            selected = np.random.choice(MEDICATIONS[condition], n, replace=False)
            meds.update(selected)
    return list(meds)


# ── Main generation ───────────────────────────────────────────────────────────

def generate_panel(n_patients=N_PATIENTS):
    patients = []

    for i in range(n_patients):
        patient_id = f"PT-{str(i+1).zfill(4)}"
        age = generate_age()
        gender = np.random.choice(["F", "M"], p=[0.54, 0.46])
        race = np.random.choice(RACE_ETHNICITY, p=RACE_WEIGHTS)
        insurance = np.random.choice(INSURANCE_TYPES, p=INSURANCE_WEIGHTS)
        income_quintile = np.random.choice(ZIP_INCOME_QUINTILES)
        conditions = generate_conditions(age)
        adherent = np.random.random() > 0.35  # 65% adherent
        labs = generate_lab_values(conditions, adherent)
        visit_history = generate_visit_history(age, conditions, adherent)
        medications = assign_medications(conditions)
        risk_score = compute_risk_score({
            "conditions": conditions,
            "labs": labs,
            "visit_history": visit_history,
            "sdoh": {"zip_income_quintile": income_quintile},
            "insurance": insurance,
        })
        priority = generate_outreach_priority(risk_score, conditions, visit_history)

        patients.append({
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "race_ethnicity": race,
            "insurance": insurance,
            "sdoh": {
                "zip_income_quintile": income_quintile,
                "food_insecurity_flag": int(income_quintile <= 2 and np.random.random() < 0.4),
                "housing_instability_flag": int(income_quintile == 1 and np.random.random() < 0.25),
                "transportation_barrier_flag": int(np.random.random() < 0.15),
            },
            "conditions": conditions,
            "medications": medications,
            "n_conditions": len(conditions),
            "labs": labs,
            "visit_history": visit_history,
            "adherent": adherent,
            "risk_score": risk_score,
            "outreach_priority": priority,
        })

    return patients


def flatten_for_dataframe(patients):
    """Flatten nested patient dicts into a flat DataFrame for modeling."""
    rows = []
    for p in patients:
        row = {
            "patient_id": p["patient_id"],
            "age": p["age"],
            "gender": p["gender"],
            "race_ethnicity": p["race_ethnicity"],
            "insurance": p["insurance"],
            "zip_income_quintile": p["sdoh"]["zip_income_quintile"],
            "food_insecurity": p["sdoh"]["food_insecurity_flag"],
            "housing_instability": p["sdoh"]["housing_instability_flag"],
            "transportation_barrier": p["sdoh"]["transportation_barrier_flag"],
            "n_conditions": p["n_conditions"],
            "n_medications": len(p["medications"]),
            "n_quality_gaps": p["visit_history"]["n_quality_gaps"],
            "days_since_pcp_visit": p["visit_history"]["days_since_pcp_visit"],
            "er_visits_12m": p["visit_history"]["er_visits_12m"],
            "hospitalizations_12m": p["visit_history"]["hospitalizations_12m"],
            "HbA1c": p["labs"].get("HbA1c", np.nan),
            "SBP": p["labs"].get("SBP", np.nan),
            "DBP": p["labs"].get("DBP", np.nan),
            "LDL": p["labs"].get("LDL", np.nan),
            "eGFR": p["labs"].get("eGFR", np.nan),
            "BMI": p["labs"].get("BMI", np.nan),
            "risk_score": p["risk_score"],
            "outreach_priority": p["outreach_priority"],
            "adherent": p["adherent"],
        }
        # One-hot conditions
        for condition in CONDITIONS:
            row[f"has_{condition.replace(' ', '_')}"] = int(condition in p["conditions"])

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating synthetic patient panel...")
    patients = generate_panel(N_PATIENTS)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)

    # Save full JSON
    with open(os.path.join(out_dir, "patient_panel.json"), "w") as f:
        json.dump(patients, f, indent=2)

    # Save flat CSV
    df = flatten_for_dataframe(patients)
    df.to_csv(os.path.join(out_dir, "patient_panel_flat.csv"), index=False)

    print(f"Generated {len(patients)} patients")
    print(f"Priority distribution:\n{df['outreach_priority'].value_counts().to_string()}")
    print(f"Saved to data/raw/")
