"""
risk_model.py
XGBoost risk stratification model for PCP panel prioritization.
Predicts high-priority outreach candidates from patient features.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Tuple, Dict, List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier


NUMERIC_FEATURES = [
    "age", "n_conditions", "n_medications", "n_quality_gaps",
    "days_since_pcp_visit", "er_visits_12m", "hospitalizations_12m",
    "HbA1c", "SBP", "DBP", "LDL", "eGFR", "BMI",
    "zip_income_quintile", "food_insecurity",
    "housing_instability", "transportation_barrier",
    "has_Type_2_Diabetes", "has_Hypertension", "has_Hyperlipidemia",
    "has_Coronary_Artery_Disease", "has_Heart_Failure", "has_COPD",
    "has_CKD", "has_Depression", "has_Obesity", "has_Atrial_Fibrillation",
]

CATEGORICAL_FEATURES = ["gender", "race_ethnicity", "insurance"]


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def make_binary_target(df: pd.DataFrame) -> pd.Series:
    """
    Binary target: 1 = Critical or High priority (needs proactive outreach).
    0 = Moderate or Routine.
    """
    return df["outreach_priority"].isin(["Critical", "High"]).astype(int)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")


def build_model() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,  # Adjust for class imbalance
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def cross_validate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    ap = cross_val_score(model, X, y, cv=cv, scoring="average_precision").mean()
    print(f"CV AUC: {auc:.3f} | CV AP: {ap:.3f}")
    return {"cv_auc": auc, "cv_ap": ap}


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"\nTest AUC: {auc:.3f} | AP: {ap:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Routine/Moderate", "High/Critical"]))
    return {"auc": auc, "ap": ap, "y_prob": y_prob, "y_pred": y_pred}


def get_feature_names(model: Pipeline) -> List[str]:
    pre = model.named_steps["preprocessor"]
    cat_names = pre.named_transformers_["cat"]["encoder"].get_feature_names_out(
        CATEGORICAL_FEATURES
    ).tolist()
    return NUMERIC_FEATURES + cat_names


def save_model(model: Pipeline, path: str = "models/risk_model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str = "models/risk_model.joblib") -> Pipeline:
    return joblib.load(path)


def score_panel(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score the full patient panel and return a prioritized worklist.
    Returns DataFrame sorted by risk probability descending.
    """
    available = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[available]
    probs = model.predict_proba(X)[:, 1]

    result = df[["patient_id", "age", "gender", "insurance",
                 "n_conditions", "n_quality_gaps", "days_since_pcp_visit",
                 "er_visits_12m", "hospitalizations_12m",
                 "outreach_priority"]].copy()
    result["risk_probability"] = probs.round(3)
    result["risk_tier"] = pd.cut(
        probs,
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=["Low", "Moderate", "High", "Critical"],
    )
    return result.sort_values("risk_probability", ascending=False).reset_index(drop=True)


def generate_weekly_worklist(
    scored_panel: pd.DataFrame,
    n_patients: int = 15,
    include_tiers: List[str] = ["Critical", "High"],
) -> pd.DataFrame:
    """
    Return the top N patients for this week's proactive outreach.
    Prioritizes Critical then High, fills remaining slots from Moderate.
    """
    priority = scored_panel[scored_panel["risk_tier"].isin(include_tiers)].head(n_patients)
    if len(priority) < n_patients:
        remaining = n_patients - len(priority)
        moderate = scored_panel[scored_panel["risk_tier"] == "Moderate"].head(remaining)
        priority = pd.concat([priority, moderate]).reset_index(drop=True)
    priority.index = range(1, len(priority) + 1)
    priority.index.name = "rank"
    return priority
