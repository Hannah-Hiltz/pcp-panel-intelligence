"""
wearable_features.py
Time series feature engineering for wearable data.
Converts raw daily signals into model-ready features.
Handles missing data with equity-aware imputation.
"""

import numpy as np
import pandas as pd
from typing import Optional


WEARABLE_COLS = ["steps", "resting_hr", "hrv", "sleep_hours", "active_minutes"]


def compute_rolling_features(ts: pd.DataFrame, windows: list = [7, 14, 30]) -> pd.DataFrame:
    """Compute rolling mean, std, and min for each signal at multiple windows."""
    ts = ts.copy().sort_values("date")
    features = {}
    for col in WEARABLE_COLS:
        if col not in ts.columns:
            continue
        series = ts[col]
        for w in windows:
            features[f"{col}_mean_{w}d"] = series.rolling(w, min_periods=max(3, w//2)).mean().iloc[-1]
            features[f"{col}_std_{w}d"]  = series.rolling(w, min_periods=max(3, w//2)).std().iloc[-1]
        features[f"{col}_trend"] = _slope(series)
    return features


def compute_baseline_deviation(ts: pd.DataFrame, baseline_days: int = 14) -> pd.DataFrame:
    """Compute deviation of recent signal from personal baseline (first N days)."""
    ts = ts.copy().sort_values("date")
    features = {}
    for col in WEARABLE_COLS:
        if col not in ts.columns:
            continue
        baseline = ts[col].iloc[:baseline_days].mean()
        recent   = ts[col].iloc[-14:].mean()
        if pd.notna(baseline) and baseline != 0:
            features[f"{col}_baseline_dev"] = (recent - baseline) / baseline
        else:
            features[f"{col}_baseline_dev"] = np.nan
    return features


def detect_anomaly_windows(ts: pd.DataFrame, sigma: float = 2.0) -> dict:
    """
    Flag anomaly windows where signal deviates >sigma SDs from rolling mean.
    Returns count of anomaly days and index of first anomaly.
    """
    ts = ts.copy().sort_values("date")
    results = {}
    for col in ["steps", "resting_hr"]:
        if col not in ts.columns:
            continue
        series = ts[col].dropna()
        if len(series) < 14:
            results[f"{col}_anomaly_days"] = 0
            results[f"{col}_first_anomaly"] = np.nan
            continue
        mu  = series.rolling(14, min_periods=7).mean()
        std = series.rolling(14, min_periods=7).std().replace(0, np.nan)
        z   = (series - mu) / std
        anomaly = z.abs() > sigma
        results[f"{col}_anomaly_days"] = int(anomaly.sum())
        first = anomaly[anomaly].index.min() if anomaly.any() else np.nan
        results[f"{col}_first_anomaly"] = int(first) if pd.notna(first) else np.nan
    return results


def equity_aware_imputation(
    panel_df: pd.DataFrame,
    wear_df: pd.DataFrame,
    strategy: str = "conservative_upward",
) -> pd.DataFrame:
    """
    Handle missing wearable data with equity-aware imputation.

    Patients without wearables are disproportionately lower-income.
    Two strategies:
    - 'conservative_upward': patients without wearables get a risk bump
      rather than being treated as low-risk (default, equity-preserving)
    - 'median_fill': fill with panel median (can mask inequity)

    This is the equity-critical design decision documented in the README.
    """
    panel = panel_df.copy()
    wear_cols = [c for c in panel.columns if c.startswith("wear_")]

    if strategy == "conservative_upward":
        # No wearable → set anomaly flag = 1 (conservative), trends = NaN
        no_device = panel["has_wearable"] == 0
        panel.loc[no_device, "wear_anomaly_flag"] = 1
        panel.loc[no_device, "wear_completeness"] = 0.0
        for col in wear_cols:
            if "anomaly" not in col and "completeness" not in col:
                panel.loc[no_device & panel[col].isna(), col] = np.nan
    elif strategy == "median_fill":
        for col in wear_cols:
            if panel[col].isna().any():
                panel[col] = panel[col].fillna(panel[col].median())

    return panel


def wearable_equity_report(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize wearable ownership and data completeness by income quintile and race.
    This is the sociological equity analysis — surfaces structural access disparities.
    """
    eq = panel_df.groupby("zip_income_quintile").agg(
        n_patients=("patient_id", "count"),
        pct_with_device=("has_wearable", "mean"),
        avg_completeness=("wear_completeness", "mean"),
        pct_anomaly_flag=("wear_anomaly_flag", "mean"),
    ).round(3).reset_index()
    eq["pct_with_device"] *= 100
    eq["avg_completeness"] *= 100
    eq["pct_anomaly_flag"] *= 100
    return eq


def _slope(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 7:
        return np.nan
    x = np.arange(len(valid))
    try:
        return float(np.polyfit(x, valid.values, 1)[0])
    except Exception:
        return np.nan
