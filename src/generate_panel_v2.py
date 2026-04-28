"""
generate_panel_v2.py
Generates a CMS-calibrated 800-patient Medicare Advantage panel with:
- Condition prevalence anchored to CMS Chronic Conditions PUF rates
- 90-day daily wearable time series (steps, resting HR, HRV, sleep)
  mirroring CovIdentify/Fitbit schema
- Longitudinal quarterly snapshots (Q1-Q4) for each patient
- SDOH factors and equity analysis fields

Run: python src/generate_panel_v2.py
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

np.random.seed(42)

N_PATIENTS = 800
N_WEARABLE_DAYS = 90
STUDY_START = datetime(2024, 1, 1)

CONDITIONS = [
    "Type 2 Diabetes", "Hypertension", "Hyperlipidemia",
    "Coronary Artery Disease", "Heart Failure", "COPD",
    "CKD", "Depression", "Obesity", "Atrial Fibrillation",
]

RACE_ETHNICITY = ["White", "Black", "Hispanic", "Asian", "Other"]
RACE_WEIGHTS   = [0.63, 0.14, 0.13, 0.06, 0.04]
INSURANCE      = ["Medicare Advantage", "Medicare FFS", "Dual Eligible"]
INSURANCE_W    = [0.60, 0.28, 0.12]

DEVICE_OWNERSHIP_BY_QUINTILE = {1: 0.38, 2: 0.52, 3: 0.67, 4: 0.79, 5: 0.88}

QUALITY_MEASURES = [
    "HbA1c_controlled", "BP_controlled", "LDL_controlled",
    "Annual_wellness_visit", "Diabetic_eye_exam",
    "Diabetic_kidney_screening", "Colorectal_screening",
    "Mammogram", "Flu_vaccine", "Pneumococcal_vaccine",
]


def load_cms_puf(puf_path: str) -> pd.DataFrame:
    return pd.read_csv(puf_path)


def get_cms_prevalence(puf_df: pd.DataFrame, age: int, gender: str, dual: bool) -> Dict[str, float]:
    age_cat = ("65-69" if age < 70 else "70-74" if age < 75 else
               "75-79" if age < 80 else "80-84" if age < 85 else "85+")
    dual_flag = "Y" if dual else "N"
    row = puf_df[(puf_df["age_category"] == age_cat) &
                 (puf_df["gender"] == gender) &
                 (puf_df["dual_eligible"] == dual_flag)]
    if row.empty:
        row = puf_df[(puf_df["age_category"] == age_cat) & (puf_df["gender"] == gender)].head(1)
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "Type 2 Diabetes": r["diabetes"],
        "Hypertension":    r["hypertension"],
        "Hyperlipidemia":  r["hyperlipidemia"],
        "Heart Failure":   r["heart_failure"],
        "COPD":            r["copd"],
        "CKD":             r["ckd"],
        "Depression":      r["depression"],
        "Obesity":         r["obesity"],
        "Atrial Fibrillation": r["afib"],
        "Coronary Artery Disease": r["cad"],
        "avg_hcc_score":   float(r["avg_hcc_score"]),
        "avg_annual_cost": float(r["avg_annual_cost"]),
    }


def generate_wearable_timeseries(
    patient_id: str,
    conditions: List[str],
    adherent: bool,
    has_device: bool,
    income_quintile: int,
    n_days: int = N_WEARABLE_DAYS,
) -> pd.DataFrame:
    """
    Generate daily wearable signals mirroring CovIdentify/Fitbit schema:
    - steps: daily step count
    - resting_hr: resting heart rate
    - hrv: heart rate variability proxy
    - sleep_hours: total daily sleep
    - active_minutes: minutes fairly/very active
    Signals are correlated with conditions and adherence.
    Missing data rate reflects income-based device ownership patterns.
    """
    if not has_device:
        return pd.DataFrame()

    dates = [STUDY_START + timedelta(days=i) for i in range(n_days)]

    # Base physiological parameters
    base_steps   = 6200 if not adherent else 7800
    base_rhr     = 72   if "Heart Failure" not in conditions else 78
    base_hrv     = 42   if adherent else 31
    base_sleep   = 6.8  if not adherent else 7.2
    base_active  = 28   if not adherent else 42

    # Condition adjustments
    if "Type 2 Diabetes" in conditions:
        base_steps  -= 800
        base_hrv    -= 5
    if "Heart Failure" in conditions:
        base_steps  -= 1800
        base_rhr    += 8
        base_hrv    -= 10
    if "COPD" in conditions:
        base_steps  -= 1200
        base_active -= 10
    if "Depression" in conditions:
        base_sleep  -= 0.6
        base_steps  -= 600
        base_hrv    -= 4
    if "Obesity" in conditions:
        base_steps  -= 900
        base_rhr    += 5

    # Time series with trend + noise
    t = np.arange(n_days)
    trend_factor = -0.008 if not adherent else 0.003

    steps = (base_steps
             + trend_factor * base_steps * t
             + np.random.normal(0, 800, n_days)).clip(0, 20000).astype(int)

    rhr = (base_rhr
           + np.random.normal(0, 3, n_days)
           - trend_factor * 5 * t).clip(40, 120).round(1)

    hrv = (base_hrv
           + trend_factor * base_hrv * t
           + np.random.normal(0, 4, n_days)).clip(8, 100).round(1)

    sleep = (base_sleep
             + np.random.normal(0, 0.7, n_days)).clip(2, 12).round(2)

    active = (base_active
              + trend_factor * base_active * t
              + np.random.normal(0, 8, n_days)).clip(0, 120).astype(int)

    # Inject deterioration event for non-adherent high-risk patients
    if not adherent and len(conditions) >= 3:
        event_day = np.random.randint(55, 75)
        steps[event_day:event_day+7]  = (steps[event_day:event_day+7] * 0.45).astype(int)
        rhr[event_day:event_day+7]    = (rhr[event_day:event_day+7] * 1.15).round(1)
        hrv[event_day:event_day+7]    = (hrv[event_day:event_day+7] * 0.65).round(1)
        sleep[event_day:event_day+7]  = (sleep[event_day:event_day+7] * 0.70).round(2)
        active[event_day:event_day+7] = (active[event_day:event_day+7] * 0.30).astype(int)

    # Missing data: lower-income patients have more gaps even with a device
    missing_rate = max(0.02, 0.12 - income_quintile * 0.02)
    mask = np.random.random(n_days) < missing_rate

    df = pd.DataFrame({
        "patient_id":   patient_id,
        "date":         [d.strftime("%Y-%m-%d") for d in dates],
        "steps":        steps,
        "resting_hr":   rhr,
        "hrv":          hrv,
        "sleep_hours":  sleep,
        "active_minutes": active,
        "day_index":    t,
    })
    df.loc[mask, ["steps", "resting_hr", "hrv", "sleep_hours", "active_minutes"]] = np.nan
    df["missing"] = mask.astype(int)
    return df


def extract_wearable_features(ts_df: pd.DataFrame) -> Dict:
    """
    Extract clinically meaningful features from wearable time series.
    Produces rolling averages, trend slopes, anomaly flags, and
    deviation from personal baseline — matching CovIdentify feature patterns.
    """
    if ts_df.empty:
        return {
            "wear_steps_7d_avg":     np.nan, "wear_steps_trend":     np.nan,
            "wear_rhr_7d_avg":       np.nan, "wear_rhr_trend":       np.nan,
            "wear_hrv_7d_avg":       np.nan, "wear_hrv_trend":       np.nan,
            "wear_sleep_7d_avg":     np.nan, "wear_active_7d_avg":   np.nan,
            "wear_steps_baseline_dev": np.nan, "wear_rhr_baseline_dev": np.nan,
            "wear_anomaly_flag":     0,       "wear_completeness":    0.0,
            "wear_deterioration_day": np.nan, "has_wearable":         0,
        }

    df = ts_df.copy()
    n = len(df)
    complete = df["missing"].eq(0).mean()

    def safe_slope(series):
        valid = series.dropna()
        if len(valid) < 7:
            return np.nan
        x = np.arange(len(valid))
        try:
            return np.polyfit(x, valid.values, 1)[0]
        except Exception:
            return np.nan

    baseline_days = min(14, n // 3)
    baseline_steps = df["steps"].iloc[:baseline_days].mean()
    baseline_rhr   = df["resting_hr"].iloc[:baseline_days].mean()

    recent = df.iloc[-14:]
    steps_recent = recent["steps"].mean()
    rhr_recent   = recent["resting_hr"].mean()

    steps_dev = ((steps_recent - baseline_steps) / baseline_steps) if baseline_steps > 0 else np.nan
    rhr_dev   = ((rhr_recent   - baseline_rhr)   / baseline_rhr)   if baseline_rhr   > 0 else np.nan

    anomaly_flag = int(
        (pd.notna(steps_dev) and steps_dev < -0.25) or
        (pd.notna(rhr_dev)   and rhr_dev   >  0.12)
    )

    # Detect deterioration window (7-day rolling min steps drops sharply)
    rolling_min = df["steps"].rolling(7, min_periods=3).min()
    deter_day = None
    if rolling_min.notna().any():
        low_idx = rolling_min.idxmin()
        if pd.notna(low_idx) and rolling_min[low_idx] < baseline_steps * 0.5:
            deter_day = int(df.loc[low_idx, "day_index"]) if "day_index" in df.columns else int(low_idx)

    return {
        "wear_steps_7d_avg":      round(df["steps"].rolling(7).mean().iloc[-1], 1) if df["steps"].notna().any() else np.nan,
        "wear_steps_trend":       round(safe_slope(df["steps"]), 3),
        "wear_rhr_7d_avg":        round(df["resting_hr"].rolling(7).mean().iloc[-1], 1) if df["resting_hr"].notna().any() else np.nan,
        "wear_rhr_trend":         round(safe_slope(df["resting_hr"]), 4),
        "wear_hrv_7d_avg":        round(df["hrv"].rolling(7).mean().iloc[-1], 1) if df["hrv"].notna().any() else np.nan,
        "wear_hrv_trend":         round(safe_slope(df["hrv"]), 4),
        "wear_sleep_7d_avg":      round(df["sleep_hours"].rolling(7).mean().iloc[-1], 2) if df["sleep_hours"].notna().any() else np.nan,
        "wear_active_7d_avg":     round(df["active_minutes"].rolling(7).mean().iloc[-1], 1) if df["active_minutes"].notna().any() else np.nan,
        "wear_steps_baseline_dev": round(steps_dev, 3) if pd.notna(steps_dev) else np.nan,
        "wear_rhr_baseline_dev":   round(rhr_dev, 3)   if pd.notna(rhr_dev)   else np.nan,
        "wear_anomaly_flag":      anomaly_flag,
        "wear_completeness":      round(complete, 3),
        "wear_deterioration_day": deter_day,
        "has_wearable":           1,
    }


def generate_quarterly_snapshots(patient_base: Dict, n_quarters: int = 4) -> List[Dict]:
    """Generate longitudinal quarterly clinical snapshots for a patient."""
    snapshots = []
    conds = patient_base["conditions"]
    adherent = patient_base["adherent"]

    for q in range(1, n_quarters + 1):
        drift = (q - 1) * (0.03 if not adherent else -0.01)
        labs = dict(patient_base["labs"])

        if "Type 2 Diabetes" in conds:
            labs["HbA1c"] = round(np.clip(labs["HbA1c"] + drift * 2 + np.random.normal(0, 0.3), 5.5, 14.0), 1)
        if "Hypertension" in conds:
            labs["SBP"] = int(np.clip(labs["SBP"] + drift * 20 + np.random.normal(0, 4), 90, 210))
        if "Hyperlipidemia" in conds:
            labs["LDL"] = int(np.clip(labs["LDL"] + drift * 15 + np.random.normal(0, 6), 40, 230))

        er = patient_base["visit_history"]["er_visits_12m"]
        hosp = patient_base["visit_history"]["hospitalizations_12m"]
        er_q   = int(np.random.poisson(max(0, er / 4 * (1 + drift * 2))))
        hosp_q = int(np.random.poisson(max(0, hosp / 4 * (1 + drift * 2))))

        snapshot_date = STUDY_START + timedelta(days=(q - 1) * 91)
        snapshots.append({
            "patient_id":    patient_base["patient_id"],
            "quarter":       f"Q{q}",
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
            "labs":          labs,
            "er_visits_q":   er_q,
            "hosp_q":        hosp_q,
            "n_quality_gaps": max(0, patient_base["visit_history"]["n_quality_gaps"] + int(drift * 3)),
            "risk_score":    min(100, int(patient_base["risk_score"] + drift * 15 + np.random.normal(0, 3))),
            "adherent":      adherent,
        })
    return snapshots


def generate_patient(patient_id: str, puf_df: pd.DataFrame) -> Dict:
    age = int(np.clip(np.random.normal(72, 7), 65, 95))
    gender = np.random.choice(["F", "M"], p=[0.54, 0.46])
    race = np.random.choice(RACE_ETHNICITY, p=RACE_WEIGHTS)
    insurance = np.random.choice(INSURANCE, p=INSURANCE_W)
    dual = insurance == "Dual Eligible"
    income_q = np.random.choice([1, 2, 3, 4, 5])

    prev = get_cms_prevalence(puf_df, age, gender, dual)
    conditions = []
    for c in CONDITIONS:
        p = prev.get(c, 0.15)
        if np.random.random() < p:
            conditions.append(c)

    adherent = np.random.random() > 0.35

    has_device = np.random.random() < DEVICE_OWNERSHIP_BY_QUINTILE.get(income_q, 0.6)

    food_ins = int(income_q <= 2 and np.random.random() < 0.40)
    housing  = int(income_q == 1 and np.random.random() < 0.25)
    transport = int(np.random.random() < 0.15)

    labs = _generate_labs(conditions, adherent)
    visit_history = _generate_visit_history(conditions, adherent)
    risk_score = _compute_risk_score(conditions, labs, visit_history, income_q, insurance)
    priority = _outreach_priority(risk_score, visit_history)

    ts_df = generate_wearable_timeseries(
        patient_id, conditions, adherent, has_device, income_q)
    wear_features = extract_wearable_features(ts_df)

    quarterly = generate_quarterly_snapshots({
        "patient_id": patient_id,
        "conditions": conditions,
        "labs": labs,
        "visit_history": visit_history,
        "risk_score": risk_score,
        "adherent": adherent,
    })

    return {
        "patient_id":    patient_id,
        "age":           age,
        "gender":        gender,
        "race_ethnicity": race,
        "insurance":     insurance,
        "sdoh": {
            "zip_income_quintile":   income_q,
            "food_insecurity_flag":  food_ins,
            "housing_instability_flag": housing,
            "transportation_barrier_flag": transport,
        },
        "conditions":    conditions,
        "n_conditions":  len(conditions),
        "labs":          labs,
        "visit_history": visit_history,
        "adherent":      adherent,
        "has_wearable":  int(has_device),
        "risk_score":    risk_score,
        "outreach_priority": priority,
        "wearable_features": wear_features,
        "quarterly_snapshots": quarterly,
        "cms_calibration": {
            "avg_hcc_score":   prev.get("avg_hcc_score", np.nan),
            "avg_annual_cost": prev.get("avg_annual_cost", np.nan),
        },
    }


def _generate_labs(conditions, adherent):
    def n(mu, sd, lo, hi):
        return round(float(np.clip(np.random.normal(mu, sd), lo, hi)), 1)

    hba1c = n(9.1, 1.2, 7.0, 14.0) if "Type 2 Diabetes" in conditions and not adherent else \
            n(7.4, 0.8, 5.5, 12.0) if "Type 2 Diabetes" in conditions else \
            n(5.6, 0.4, 4.5, 6.4)

    sbp = int(np.clip(np.random.normal(148 if ("Hypertension" in conditions and not adherent) else
                                       128 if "Hypertension" in conditions else 120, 12), 90, 200))
    dbp = int(np.clip(np.random.normal(88 if ("Hypertension" in conditions and not adherent) else
                                       78 if "Hypertension" in conditions else 74, 8), 55, 120))

    ldl = int(np.clip(np.random.normal(130 if ("Hyperlipidemia" in conditions and not adherent) else
                                       85 if "Hyperlipidemia" in conditions else 105, 25), 40, 230))

    egfr = int(np.clip(np.random.normal(42 if "CKD" in conditions else 78, 12), 15, 120))
    bmi  = round(float(np.clip(np.random.normal(36 if "Obesity" in conditions else 27, 4), 18, 55)), 1)
    return {"HbA1c": hba1c, "SBP": sbp, "DBP": dbp, "LDL": ldl, "eGFR": egfr, "BMI": bmi}


def _generate_visit_history(conditions, adherent):
    days = int(np.clip(np.random.exponential(200 if not adherent else 90), 30, 730))
    er   = int(np.random.poisson(0.3 * len(conditions) / 10 * (2.5 if not adherent else 1)))
    hosp = int(np.random.poisson(0.1 * len(conditions) / 10 * (3 if not adherent else 1)))
    today = datetime.today()
    gaps, done = [], []
    for m in QUALITY_MEASURES:
        if np.random.random() < (0.35 if not adherent else 0.75):
            done.append(m)
        else:
            gaps.append(m)
    return {
        "last_pcp_visit":       (today - timedelta(days=days)).strftime("%Y-%m-%d"),
        "days_since_pcp_visit": days,
        "er_visits_12m":        er,
        "hospitalizations_12m": hosp,
        "quality_gaps":         gaps,
        "quality_completed":    done,
        "n_quality_gaps":       len(gaps),
    }


def _compute_risk_score(conditions, labs, visit_history, income_q, insurance):
    s = min(len(conditions) * 4, 30)
    if labs.get("HbA1c", 0) > 8.0:   s += 12
    elif labs.get("HbA1c", 0) > 7.0: s += 6
    if labs.get("SBP", 0) > 140:  s += 8
    elif labs.get("SBP", 0) > 130: s += 4
    if labs.get("LDL", 0) > 130:  s += 5
    d = visit_history["days_since_pcp_visit"]
    s += 20 if d > 365 else 12 if d > 180 else 5 if d > 90 else 0
    s += min(visit_history["n_quality_gaps"] * 4, 15)
    s += min(visit_history["er_visits_12m"] * 3, 8)
    s += min(visit_history["hospitalizations_12m"] * 4, 10)
    if income_q <= 2: s += 3
    if insurance == "Dual Eligible": s += 2
    return min(int(s), 100)


def _outreach_priority(risk_score, visit_history):
    if risk_score >= 70 or visit_history["hospitalizations_12m"] >= 2:
        return "Critical"
    elif risk_score >= 50 or (visit_history["n_quality_gaps"] >= 3 and
                               visit_history["days_since_pcp_visit"] > 180):
        return "High"
    elif risk_score >= 30 or visit_history["n_quality_gaps"] >= 2:
        return "Moderate"
    return "Routine"


def flatten_panel(patients: List[Dict]) -> pd.DataFrame:
    rows = []
    for p in patients:
        wf = p["wearable_features"]
        row = {
            "patient_id":             p["patient_id"],
            "age":                    p["age"],
            "gender":                 p["gender"],
            "race_ethnicity":         p["race_ethnicity"],
            "insurance":              p["insurance"],
            "zip_income_quintile":    p["sdoh"]["zip_income_quintile"],
            "food_insecurity":        p["sdoh"]["food_insecurity_flag"],
            "housing_instability":    p["sdoh"]["housing_instability_flag"],
            "transportation_barrier": p["sdoh"]["transportation_barrier_flag"],
            "n_conditions":           p["n_conditions"],
            "n_quality_gaps":         p["visit_history"]["n_quality_gaps"],
            "days_since_pcp_visit":   p["visit_history"]["days_since_pcp_visit"],
            "er_visits_12m":          p["visit_history"]["er_visits_12m"],
            "hospitalizations_12m":   p["visit_history"]["hospitalizations_12m"],
            "HbA1c":                  p["labs"].get("HbA1c"),
            "SBP":                    p["labs"].get("SBP"),
            "DBP":                    p["labs"].get("DBP"),
            "LDL":                    p["labs"].get("LDL"),
            "eGFR":                   p["labs"].get("eGFR"),
            "BMI":                    p["labs"].get("BMI"),
            "risk_score":             p["risk_score"],
            "outreach_priority":      p["outreach_priority"],
            "adherent":               p["adherent"],
            "has_wearable":           p["has_wearable"],
            "avg_hcc_score":          p["cms_calibration"].get("avg_hcc_score"),
            "avg_annual_cost":        p["cms_calibration"].get("avg_annual_cost"),
        }
        for c in ["Type_2_Diabetes","Hypertension","Hyperlipidemia","Coronary_Artery_Disease",
                  "Heart_Failure","COPD","CKD","Depression","Obesity","Atrial_Fibrillation"]:
            row[f"has_{c}"] = int(c.replace("_"," ") in p["conditions"])
        row.update(wf)
        rows.append(row)
    return pd.DataFrame(rows)


def flatten_wearables(patients: List[Dict], ts_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for p in patients:
        ts = ts_map.get(p["patient_id"])
        if ts is not None and not ts.empty:
            frames.append(ts)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flatten_longitudinal(patients: List[Dict]) -> pd.DataFrame:
    rows = []
    for p in patients:
        for q in p["quarterly_snapshots"]:
            row = {
                "patient_id":    q["patient_id"],
                "quarter":       q["quarter"],
                "snapshot_date": q["snapshot_date"],
                "HbA1c":         q["labs"].get("HbA1c"),
                "SBP":           q["labs"].get("SBP"),
                "LDL":           q["labs"].get("LDL"),
                "er_visits_q":   q["er_visits_q"],
                "hosp_q":        q["hosp_q"],
                "n_quality_gaps": q["n_quality_gaps"],
                "risk_score":    q["risk_score"],
                "adherent":      q["adherent"],
                "insurance":     p["insurance"],
                "zip_income_quintile": p["sdoh"]["zip_income_quintile"],
                "race_ethnicity": p["race_ethnicity"],
                "has_wearable":  p["has_wearable"],
                "n_conditions":  p["n_conditions"],
            }
            rows.append(row)
    return pd.DataFrame(rows)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj): return None
        return super().default(obj)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    puf_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cms_puf", "cms_chronic_conditions_puf.csv")
    puf_df = load_cms_puf(puf_path)
    print(f"CMS PUF loaded: {len(puf_df)} demographic cells")

    out_raw  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    out_proc = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(os.path.join(out_raw, "wearables"), exist_ok=True)
    os.makedirs(out_proc, exist_ok=True)
    os.makedirs(os.path.join(out_raw, "panel"), exist_ok=True)

    print(f"Generating {N_PATIENTS} patients...")
    patients, ts_map = [], {}
    for i in range(N_PATIENTS):
        pid = f"PT-{str(i+1).zfill(4)}"
        p = generate_patient(pid, puf_df)
        patients.append(p)
        has_dev = p["has_wearable"]
        if has_dev:
            ts = generate_wearable_timeseries(
                pid, p["conditions"], p["adherent"],
                True, p["sdoh"]["zip_income_quintile"])
            ts_map[pid] = ts
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_PATIENTS}")

    panel_df = flatten_panel(patients)
    panel_df.to_csv(os.path.join(out_raw, "panel", "patient_panel_flat.csv"), index=False)

    wear_df = flatten_wearables(patients, ts_map)
    if not wear_df.empty:
        wear_df.to_csv(os.path.join(out_raw, "wearables", "wearable_timeseries.csv"), index=False)

    long_df = flatten_longitudinal(patients)
    long_df.to_csv(os.path.join(out_proc, "longitudinal_snapshots.csv"), index=False)

    with open(os.path.join(out_raw, "panel", "patient_panel.json"), "w") as f:
        json.dump(patients[:50], f, cls=NumpyEncoder, indent=2)

    wearable_pct = panel_df["has_wearable"].mean() * 100
    print(f"\nDone.")
    print(f"  Panel: {len(panel_df)} patients")
    print(f"  Wearable coverage: {wearable_pct:.1f}%")
    print(f"  Wearable time series rows: {len(wear_df)}")
    print(f"  Longitudinal snapshots: {len(long_df)}")
    print(f"  Priority: {panel_df['outreach_priority'].value_counts().to_dict()}")
