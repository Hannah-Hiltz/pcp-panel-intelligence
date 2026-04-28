"""
longitudinal_tracker.py
Quarterly trend analysis and deterioration detection.
Tracks each patient's risk trajectory across Q1-Q4.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def compute_risk_trajectory(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-patient risk score trajectory across quarters.
    Flags patients whose risk is trending upward (deteriorating).
    """
    traj = []
    for pid, group in long_df.groupby("patient_id"):
        group = group.sort_values("quarter")
        scores = group["risk_score"].values
        if len(scores) < 2:
            continue
        slope = float(np.polyfit(range(len(scores)), scores, 1)[0]) if len(scores) >= 2 else 0.0
        traj.append({
            "patient_id":       pid,
            "q1_risk":          scores[0] if len(scores) > 0 else np.nan,
            "q2_risk":          scores[1] if len(scores) > 1 else np.nan,
            "q3_risk":          scores[2] if len(scores) > 2 else np.nan,
            "q4_risk":          scores[3] if len(scores) > 3 else np.nan,
            "risk_slope":       round(slope, 3),
            "risk_change_q1_q4": int(scores[-1] - scores[0]) if len(scores) >= 4 else np.nan,
            "deteriorating":    int(slope > 2.0),
            "improving":        int(slope < -2.0),
            "stable":           int(abs(slope) <= 2.0),
            "n_quarters":       len(scores),
        })
    return pd.DataFrame(traj)


def compute_lab_trends(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-patient HbA1c, SBP, LDL trends across quarters."""
    trends = []
    for pid, group in long_df.groupby("patient_id"):
        group = group.sort_values("quarter")

        def slope_if_enough(col):
            vals = group[col].dropna()
            if len(vals) < 2:
                return np.nan
            return round(float(np.polyfit(range(len(vals)), vals.values, 1)[0]), 3)

        trends.append({
            "patient_id":    pid,
            "hba1c_trend":   slope_if_enough("HbA1c"),
            "sbp_trend":     slope_if_enough("SBP"),
            "ldl_trend":     slope_if_enough("LDL"),
            "hba1c_q4":      group["HbA1c"].iloc[-1] if "HbA1c" in group.columns else np.nan,
            "sbp_q4":        group["SBP"].iloc[-1]   if "SBP" in group.columns else np.nan,
            "ldl_q4":        group["LDL"].iloc[-1]   if "LDL" in group.columns else np.nan,
            "total_er_visits": group["er_visits_q"].sum(),
            "total_hosp":      group["hosp_q"].sum(),
        })
    return pd.DataFrame(trends)


def flag_high_velocity_deterioration(
    traj_df: pd.DataFrame,
    risk_slope_threshold: float = 5.0,
    risk_change_threshold: int = 15,
) -> pd.DataFrame:
    """
    Flag patients with rapid deterioration — high slope AND large Q1→Q4 change.
    These are the highest-value intervention targets.
    """
    df = traj_df.copy()
    df["high_velocity"] = (
        (df["risk_slope"] >= risk_slope_threshold) |
        (df["risk_change_q1_q4"] >= risk_change_threshold)
    ).astype(int)
    return df


def population_trend_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Panel-level trend summary by quarter — used in dashboard."""
    return long_df.groupby("quarter").agg(
        avg_risk_score=("risk_score", "mean"),
        avg_hba1c=("HbA1c", "mean"),
        avg_sbp=("SBP", "mean"),
        avg_ldl=("LDL", "mean"),
        total_er=("er_visits_q", "sum"),
        total_hosp=("hosp_q", "sum"),
        avg_quality_gaps=("n_quality_gaps", "mean"),
    ).round(2).reset_index()


def sdoh_trajectory_analysis(long_df: pd.DataFrame, traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-reference deterioration trajectory with SDOH factors.
    Sociology lens: do lower-income patients deteriorate faster?
    """
    merged = long_df[long_df["quarter"] == "Q1"][
        ["patient_id", "zip_income_quintile", "race_ethnicity", "has_wearable", "insurance"]
    ].merge(traj_df, on="patient_id", how="left")

    return merged.groupby("zip_income_quintile").agg(
        n=("patient_id", "count"),
        avg_risk_slope=("risk_slope", "mean"),
        pct_deteriorating=("deteriorating", "mean"),
        avg_q4_risk=("q4_risk", "mean"),
        pct_high_velocity=("high_velocity", "mean"),
    ).round(3).reset_index()
