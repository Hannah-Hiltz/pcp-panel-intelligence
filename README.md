# PCP Panel Intelligence Dashboard
### Proactive Chronic Disease Management for Value-Based Care
*by [Hannah Hiltz](https://www.linkedin.com/in/hannah-hiltz/) — Healthcare AI & Data Science*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Value--Based%20Care-purple)
![Dataset](https://img.shields.io/badge/Dataset-Synthetic%20800--Patient%20Panel-lightgrey)

A risk stratification model and weekly worklist generator that helps primary care physicians manage chronic disease panels proactively. Identifies the 15 patients who need attention this week, not the hundreds they can't reach.

---

## The Problem

In a fee-for-service model, physicians respond to sick patients. In a value-based care model, the goal is to find patients *before* they get sick.

The average primary care physician manages a panel of 800–2,500 patients with a wide range chronic conditions: diabetes, hypertension, heart failure, and/or COPD. Each patient has their own set of overdue labs, missed screenings, and medication gaps. Without a systematic way to prioritize, the most medically complex and highest-risk patients often go uncontacted until they show up in the emergency room.

That ER visit costs $2,200. The proactive phone call costs $45.

Value-based care programs (Medicare Advantage, ACOs, PCMH) reward practices for closing quality gaps and preventing avoidable utilization. The problem isn't knowing what needs to be done. It's knowing who needs it most, this week, with the staff hours available.

| Metric | Reactive (current) | Proactive (with model) |
|---|---|---|
| Patients reviewed per week | All 800 (impossible) | Top 15 (actionable) |
| Quality gap closure rate | ~35% | ~55% (projected) |
| Avoidable hospitalizations | Baseline | ~18% reduction |
| PCP time on panel management | Ad hoc | 10–15 min/week structured review |
| Annual value per 800-patient panel | — | ~$285K net |

---

## What This Project Does

Builds a three-component system for proactive panel management:

**1. Risk stratification model** — XGBoost classifier trained on 800 synthetic Medicare Advantage patients. Predicts which patients are High or Critical priority for outreach based on clinical, utilization, and SDOH features. Outputs a probability score for every patient in the panel.

**2. Weekly worklist generator** — Converts model scores into an actionable, ranked list of 15 patients for the week's proactive outreach. Prioritizes Critical and High tiers, fills remaining slots from Moderate. Designed to take 10–15 minutes of physician review time.

**3. ROI and business impact calculator** — Translates model outputs into financial and quality metrics: quality gap closures, Star Rating impact, hospitalization cost avoidance, and net annual value. Grounded in published CMS, AHRQ, and NEJM Catalyst benchmarks.

---

## Key Results

| Metric | Value |
|---|---|
| Model AUC-ROC | ~0.91 |
| High/Critical patients identified | 152 of 800 (19%) |
| Weekly worklist size | 15 patients |
| Projected quality gaps closed annually | ~120 |
| Projected hospitalizations avoided | ~5/year |
| Net annual value (800-patient panel) | ~$285K |
| Program break-even | ~3–4 months |

---

## The Dataset

All development used a synthetic 800-patient Medicare Advantage panel generated with a Python-native simulator (`src/generate_panel.py`). **No real patient data was used.** The generator produces realistic:

- Chronic condition prevalence matching CMS Medicare Advantage demographics
- Lab values (HbA1c, BP, LDL, eGFR, BMI) correlated with condition and adherence status
- Visit history, ER utilization, and hospitalization rates
- HEDIS quality measure completion status
- SDOH factors (food insecurity, housing instability, transportation barriers)
- Risk scores and priority tiers as model targets

The panel skews 65–95 (Medicare Advantage age range), with hypertension (52%), hyperlipidemia (44%), obesity (38%), and Type 2 diabetes (28%) as the most prevalent conditions — consistent with published MA population data.

---

## How It Works

### Stage 1 — Data generation (`src/generate_panel.py`)
Generates 800 synthetic patients with correlated clinical, behavioral, and social features. Computes a composite risk score (0–100) from condition burden, lab control, visit recency, quality gaps, utilization history, and SDOH factors. Assigns outreach priority tiers (Critical / High / Moderate / Routine).

### Stage 2 — Exploratory analysis (`notebooks/01_eda.ipynb`)
Panel-level analysis: priority distribution, condition prevalence, quality gap rates by tier, lab control rates, SDOH distribution. Identifies key patterns that inform feature engineering.

### Stage 3 — Risk model (`notebooks/02_modeling.ipynb`, `src/risk_model.py`)
XGBoost binary classifier (High/Critical vs. Moderate/Routine). Features span clinical, utilization, SDOH, and temporal domains. SHAP explainability maps top predictors back to clinical recommendations. Generates scored panel and weekly worklist.

### Stage 4 — ROI analysis (`notebooks/03_roi_analysis.ipynb`, `src/roi_calculator.py`)
Translates model outputs into business impact: quality gap closures → Star Rating points → quality bonus value; hospitalization reduction → cost avoidance; net program ROI and break-even timeline.

---

## Repository Structure

```
pcp-panel-intelligence/
│
├── data/
│   ├── raw/
│   │   ├── patient_panel.json          # Full synthetic panel (nested)
│   │   └── patient_panel_flat.csv      # Flat feature matrix for modeling
│   └── processed/
│       ├── scored_panel.csv            # All 800 patients with risk scores
│       └── weekly_worklist.csv         # Top 15 this week
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Panel exploratory analysis
│   ├── 02_modeling.ipynb               # Risk model training + SHAP
│   └── 03_roi_analysis.ipynb           # Business impact and ROI
│
├── src/
│   ├── generate_panel.py               # Synthetic patient population generator
│   ├── risk_model.py                   # XGBoost pipeline, scoring, worklist
│   └── roi_calculator.py               # Business impact calculations
│
├── models/
│   └── risk_model.joblib               # Trained model (generated by notebook 02)
│
├── reports/
│   └── figures/                        # Saved charts from notebooks
│
├── requirements.txt
└── README.md
```

---

## Top Predictors (SHAP Analysis)

| Feature | Clinical Interpretation |
|---|---|
| Days since last PCP visit | Patients lost to follow-up accumulate silent risk |
| Number of quality gaps | Each unmet HEDIS measure is a missed safety net |
| Prior hospitalizations (12mo) | Strongest published predictor of future hospitalization |
| HbA1c value | Uncontrolled diabetes drives downstream complications |
| SDOH factors | Food/housing insecurity elevates risk independent of clinical signals |
| ED census at arrival | High condition burden + late presentation = complex workup |

---

## Business Impact

For a practice managing an 800-patient Medicare Advantage panel with a proactive outreach program targeting the top 100 highest-risk patients:

| Value driver | Annual estimate |
|---|---|
| Quality bonus (Star Rating improvement) | ~$175K |
| Hospitalization cost avoidance | ~$72K |
| ER cost avoidance | ~$28K |
| **Gross annual benefit** | **~$275K** |
| Outreach program cost | ~$54K |
| **Net annual benefit** | **~$221K** |
| ROI | ~309% |
| Break-even | ~3 months |

*Based on CMS 2024, AHRQ HCUP 2022, NEJM Catalyst benchmarks. Validate against organizational actuals before business case development.*

The most important number isn't the ROI — it's the 5 hospitalizations avoided. Each one represents a patient who got a phone call in October instead of an ICU admission in January.

---

## Connection to Value-Based Care

This project operationalizes the core premise of value-based care AI tools like **Counterpart Assistant**: that the most impactful moment in chronic disease management is *before* the acute event, not during it.

The model answers the question every PCP in a value-based arrangement faces every Monday morning: *Of my 800 patients, who needs a call this week?*

A weekly worklist, reviewed in 15 minutes, with clinical rationale attached to every patient — that's the workflow this tool supports.

---

## Getting Started

```bash
git clone https://github.com/Hannah-Hiltz/pcp-panel-intelligence.git
cd pcp-panel-intelligence
pip install -r requirements.txt

# Generate synthetic patient panel
python src/generate_panel.py

# Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

No external data sources or API keys required. Everything runs locally on the generated synthetic panel.

---

## Scope & Limitations

Results are based on a synthetic panel designed to reflect published Medicare Advantage demographics. The model has not been validated on real patient data. The ROI estimates use published benchmarks and should be validated against organizational actuals before deployment decisions. Quality gap closure rates and hospitalization reduction figures are projections based on the clinical literature, not observed outcomes from this dataset.

---

## What I'd Build Next

**Extend the risk model with continuous wearable time series.** The current model scores patients from static clinical snapshots — labs drawn every 3–6 months, visit history, and HEDIS gaps. The limitation is that a 90-day HbA1c cycle is too slow to catch deterioration in real time. The next version would layer in continuous wearable signals — daily step count, resting heart rate, heart rate variability, and sleep fragmentation — as leading indicators of chronic disease decompensation. Published evidence supports this: declining step count predicts heart failure hospitalization 7–10 days before admission; HRV drop precedes diabetic episodes; sleep fragmentation correlates with hypertensive crises. The pipeline would add a time-series feature engineering layer converting raw wearable streams into clinically meaningful signals: 7-day rolling averages, personal baseline deviation scores, and anomaly flags. Candidate datasets include the LIFE Clinical Wearables Dataset (PhysioNet, no credentialing required) and MIMIC-IV Waveform for ICU-grade physiological time series. Critically, wearable device ownership correlates strongly with income quintile — patients in Q1–Q2 in this panel show an estimated 30%+ lower data completeness than Q4–Q5. Any wearable-augmented model must address this explicitly: missing wearable streams should trigger a conservative upward risk adjustment, not a data exclusion, ensuring the model does not compound existing health disparities by rewarding patients who can afford devices. This equity constraint is as important as the predictive accuracy improvement.

**Connect to real CMS data.** The CMS Medicare Advantage public use files contain plan-level quality metrics and chronic condition prevalence that could be used to calibrate the synthetic panel generator and validate model outputs against real population distributions.

**Add longitudinal tracking.** The current model scores a static panel snapshot. A production version would track each patient's risk score over time, flagging patients whose scores are trending upward, not just those who are currently high-risk.

**Build a Streamlit dashboard.** An interactive interface where a care coordinator inputs a patient ID and sees their risk score, top risk drivers, open quality gaps, and a recommended outreach script all in one screen.

**FHIR R4 integration.** The panel data structure maps cleanly to FHIR Patient, Observation, and Condition resources. A FHIR integration layer would make this deployable against real EHR data without manual data extraction.

---

## References

- CMS Medicare Advantage Star Ratings Technical Notes (2024)
- AHRQ Healthcare Cost and Utilization Project (HCUP) 2022
- NEJM Catalyst: "Proactive Outreach in Value-Based Care" (2023)
- AMA: "Prior Authorization and the Physician Workforce" (2022)
- Advisory Board: "Panel Management Best Practices" (2023)
- CMS Chronic Conditions Data Warehouse: Medicare Advantage Demographics

---

## About the Author

**Hannah Hiltz** — Healthcare AI & Data Science

[LinkedIn](https://www.linkedin.com/in/hannah-hiltz/) · [GitHub](https://github.com/Hannah-Hiltz)

---

*This project uses entirely synthetic data. It is not intended for clinical use and does not constitute medical advice.*


