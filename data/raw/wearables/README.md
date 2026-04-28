# Wearable Data — Real Dataset Instructions

## Currently included: synthetic data (schema-compatible)

`wearable_timeseries.csv` ships with this repo as a synthetic dataset that mirrors the exact schema of the CovIdentify dataset (PhysioNet). All pipeline code runs against this file without modification.

## Swapping in real CovIdentify data

**Dataset:** CovIdentify — Fitbit, Garmin, and Apple Watch data from 2,887 participants  
**Access:** Free with PhysioNet credentialing (physionet.org account + data use agreement)  
**Credentialing time:** ~10 minutes  
**URL:** https://physionet.org/content/covidentify/1.0.0/

### Steps

1. Create a free PhysioNet account at physionet.org
2. Navigate to the CovIdentify dataset and sign the data use agreement
3. Download the Fitbit summary files (daily-level step count, resting HR, sleep)
4. Rename and place in this folder as `wearable_timeseries.csv`
5. Map columns to the pipeline schema below

### Schema mapping

| CovIdentify column | Pipeline column | Notes |
|---|---|---|
| participant_id | patient_id | |
| date | date | YYYY-MM-DD |
| steps | steps | Daily step count |
| resting_heart_rate | resting_hr | Resting HR (bpm) |
| heart_rate_variability | hrv | Max daily HRV |
| sleep_duration | sleep_hours | Total sleep (hours) |
| fairly_active_minutes + very_active_minutes | active_minutes | Sum both columns |

### Note on cohort

CovIdentify participants were recruited via social media (2020–2021) and skew younger and more tech-literate than a Medicare Advantage population. For production validation, MIMIC-IV Waveform or a health-system-specific wearable integration would be more appropriate. The synthetic data in this repo is calibrated to Medicare Advantage demographics.

## CMS Chronic Conditions PUF

**File:** `data/raw/cms_puf/cms_chronic_conditions_puf.csv`  
**Source:** CMS.gov — no credentialing required  
**URL:** https://www.cms.gov/data-research/statistics-trends-and-reports/basic-stand-alone-medicare-claims-public-use-files/chronic-conditions-puf  
**Use:** Calibrates synthetic panel condition prevalence to real CMS Medicare Advantage rates by age, gender, and dual-eligibility status.

The included file contains the key prevalence rates extracted from the 2022 CMS Chronic Conditions PUF. To update with the latest release, download the PUF and re-run `src/generate_panel_v2.py`.
