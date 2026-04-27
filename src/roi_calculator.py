"""
roi_calculator.py
Business impact and ROI calculations for PCP panel optimization.
Translates model outputs into clinical and financial metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict


# ── Published benchmarks ──────────────────────────────────────────────────────
# Sources: CMS, AHRQ, NEJM Catalyst, Advisory Board

AVG_AVOIDABLE_HOSPITALIZATION_COST = 12_000   # USD, AHRQ 2023
AVG_ER_VISIT_COST = 2_200                      # USD, HCUP 2022
QUALITY_BONUS_PER_STAR_POINT = 850_000         # USD/yr for avg MA plan, CMS 2024
OUTREACH_COST_PER_PATIENT = 45                 # USD, care management labor estimate
AVG_HEDIS_GAP_CLOSURE_RATE = 0.35             # 35% of gaps closed with targeted outreach
AVOIDABLE_HOSP_REDUCTION_RATE = 0.18          # 18% reduction with proactive management (NEJM Catalyst)


def compute_panel_roi(
    panel_size: int = 800,
    n_high_risk: int = 152,
    n_outreached: int = 100,
    avg_quality_gaps_per_patient: float = 2.3,
    er_visits_baseline: int = 85,
    hospitalizations_baseline: int = 28,
) -> Dict:
    """
    Compute annual ROI from proactive panel management.

    Parameters
    ----------
    panel_size : total patients in panel
    n_high_risk : patients identified as High/Critical priority
    n_outreached : patients successfully reached for intervention
    avg_quality_gaps_per_patient : average HEDIS gaps per high-risk patient
    er_visits_baseline : current annual ER visits from panel
    hospitalizations_baseline : current annual hospitalizations from panel
    """

    # Quality gap closure
    gaps_total = n_outreached * avg_quality_gaps_per_patient
    gaps_closed = gaps_total * AVG_HEDIS_GAP_CLOSURE_RATE
    star_points_gained = gaps_closed / 50  # ~50 gap closures per Star point improvement
    quality_bonus_value = star_points_gained * QUALITY_BONUS_PER_STAR_POINT

    # Avoidable utilization reduction
    hosp_reduced = hospitalizations_baseline * AVOIDABLE_HOSP_REDUCTION_RATE
    hosp_cost_avoided = hosp_reduced * AVG_AVOIDABLE_HOSPITALIZATION_COST
    er_reduced = er_visits_baseline * AVOIDABLE_HOSP_REDUCTION_RATE * 0.6
    er_cost_avoided = er_reduced * AVG_ER_VISIT_COST

    # Program cost
    outreach_cost = n_outreached * OUTREACH_COST_PER_PATIENT * 12  # annual

    # Net impact
    gross_benefit = quality_bonus_value + hosp_cost_avoided + er_cost_avoided
    net_benefit = gross_benefit - outreach_cost
    roi_pct = (net_benefit / outreach_cost) * 100 if outreach_cost > 0 else 0

    return {
        "panel_size": panel_size,
        "n_high_risk_identified": n_high_risk,
        "n_outreached": n_outreached,
        "gaps_closed_annually": round(gaps_closed),
        "star_points_gained": round(star_points_gained, 2),
        "quality_bonus_value": round(quality_bonus_value),
        "hospitalizations_avoided": round(hosp_reduced, 1),
        "hosp_cost_avoided": round(hosp_cost_avoided),
        "er_visits_avoided": round(er_reduced, 1),
        "er_cost_avoided": round(er_cost_avoided),
        "outreach_program_cost": round(outreach_cost),
        "gross_annual_benefit": round(gross_benefit),
        "net_annual_benefit": round(net_benefit),
        "roi_pct": round(roi_pct, 1),
        "break_even_months": round((outreach_cost / (gross_benefit / 12)), 1) if gross_benefit > 0 else None,
    }


def quality_gap_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate financial value of closing each quality gap type.
    Returns a ranked table of gap closure opportunities.
    """
    gap_values = {
        "HbA1c_controlled": {
            "clinical_benefit": "Reduces microvascular complications",
            "cost_avoided_per_closure": 1_850,
            "star_rating_impact": "Diabetes Care composite",
        },
        "BP_controlled": {
            "clinical_benefit": "Reduces stroke and MI risk",
            "cost_avoided_per_closure": 2_100,
            "star_rating_impact": "Controlling BP measure",
        },
        "LDL_controlled": {
            "clinical_benefit": "Reduces cardiovascular events",
            "cost_avoided_per_closure": 1_400,
            "star_rating_impact": "Statin therapy measure",
        },
        "Annual_wellness_visit": {
            "clinical_benefit": "Early detection, care planning",
            "cost_avoided_per_closure": 800,
            "star_rating_impact": "Care Coordination composite",
        },
        "Diabetic_eye_exam": {
            "clinical_benefit": "Prevents diabetic retinopathy progression",
            "cost_avoided_per_closure": 950,
            "star_rating_impact": "Diabetes Care composite",
        },
        "Diabetic_kidney_screening": {
            "clinical_benefit": "Early CKD detection",
            "cost_avoided_per_closure": 1_200,
            "star_rating_impact": "Diabetes Care composite",
        },
        "Colorectal_screening": {
            "clinical_benefit": "Early cancer detection",
            "cost_avoided_per_closure": 3_200,
            "star_rating_impact": "Colorectal Cancer Screening",
        },
        "Mammogram": {
            "clinical_benefit": "Early breast cancer detection",
            "cost_avoided_per_closure": 2_800,
            "star_rating_impact": "Breast Cancer Screening",
        },
        "Flu_vaccine": {
            "clinical_benefit": "Prevents flu-related hospitalization",
            "cost_avoided_per_closure": 420,
            "star_rating_impact": "Flu Vaccine measure",
        },
        "Pneumococcal_vaccine": {
            "clinical_benefit": "Prevents pneumonia hospitalization",
            "cost_avoided_per_closure": 680,
            "star_rating_impact": "Pneumococcal Vaccine measure",
        },
    }

    rows = []
    for gap, info in gap_values.items():
        rows.append({
            "quality_measure": gap.replace("_", " "),
            "clinical_benefit": info["clinical_benefit"],
            "est_cost_avoided_per_closure": f"${info['cost_avoided_per_closure']:,}",
            "star_rating_impact": info["star_rating_impact"],
        })

    return pd.DataFrame(rows).sort_values(
        "est_cost_avoided_per_closure", ascending=False
    ).reset_index(drop=True)


def sdoh_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize SDOH burden by outreach priority tier."""
    return df.groupby("outreach_priority").agg(
        n_patients=("patient_id", "count"),
        pct_food_insecure=("food_insecurity", "mean"),
        pct_housing_unstable=("housing_instability", "mean"),
        pct_transport_barrier=("transportation_barrier", "mean"),
        avg_income_quintile=("zip_income_quintile", "mean"),
    ).round(3).reset_index()


def format_roi_report(roi: Dict) -> str:
    """Format ROI dict into a readable text summary."""
    return f"""
PCP Panel Optimization — Annual Business Impact Summary
========================================================

Panel size:                    {roi['panel_size']:,} patients
High/Critical patients:        {roi['n_high_risk_identified']:,} identified
Patients outreached:           {roi['n_outreached']:,}

QUALITY IMPROVEMENT
  Quality gaps closed:         {roi['gaps_closed_annually']:,}
  Star rating points gained:   {roi['star_points_gained']}
  Quality bonus value:         ${roi['quality_bonus_value']:,}

UTILIZATION REDUCTION
  Hospitalizations avoided:    {roi['hospitalizations_avoided']}
  Hospital cost avoided:       ${roi['hosp_cost_avoided']:,}
  ER visits avoided:           {roi['er_visits_avoided']}
  ER cost avoided:             ${roi['er_cost_avoided']:,}

PROGRAM ECONOMICS
  Outreach program cost:       ${roi['outreach_program_cost']:,}
  Gross annual benefit:        ${roi['gross_annual_benefit']:,}
  Net annual benefit:          ${roi['net_annual_benefit']:,}
  ROI:                         {roi['roi_pct']}%
  Break-even:                  {roi['break_even_months']} months

Sources: CMS 2024, AHRQ HCUP 2022, NEJM Catalyst, Advisory Board
"""
