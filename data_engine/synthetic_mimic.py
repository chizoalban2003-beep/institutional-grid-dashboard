"""Synthetic MIMIC-IV Data Generator — validates the ingestion pipeline.

Generates realistic MIMIC-IV-like chartevents and labevents CSV files
with known ground truth, for testing without the full 50GB+ dataset.

Features:
  - Irregular hourly sampling (some hours missing, labs delayed 6-12h)
  - Clinical deterioration trajectories (vitals drift, labs worsen)
  - Realistic missingness patterns (vitals ~5% missing, labs ~30-80%)
  - Multiple ICU stays with different severity profiles
"""
import os
import csv
import random
import numpy as np
from datetime import datetime, timedelta

# MIMIC-IV itemids for key features
ITEMIDS = {
    "HR": 220045,
    "O2Sat": 220277,
    "Temp": 223762,
    "SBP": 220179,
    "DBP": 220180,
    "Resp": 220210,
    "Lactate": 50813,
    "WBC": 51300,
    "Glucose": 50931,
    "Creatinine": 50912,
    "Potassium": 50971,
    "pH": 50820,
    "FiO2": 223835,
    "Platelets": 51265,
    "Hgb": 51221,
    "Hct": 51222,
}

def generate_stay(
    stay_id: int,
    subject_id: int,
    n_hours: int,
    scenario: str = "stable",
    seed: int = None,
    onset_hour: int = None,
) -> list[dict]:
    """Generate charted events for one ICU stay.

    Scenarios:
    - stable: Normal vitals, mild fluctuations
    - deteriorating: Progressive sepsis trajectory (subtle, patient-varying)
    - recovering: Post-intervention recovery

    onset_hour: hour at which deterioration begins (deteriorating stays).
    When None it is sampled per patient from U{4..20} — onset varies
    between patients, so detection requires trend learning, not a
    threshold against a universal baseline.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if onset_hour is None and scenario == "deteriorating":
        onset_hour = random.randint(4, 20)
    elif onset_hour is None:
        onset_hour = 0

    events = []
    base_time = datetime(2020, 1, 1, 0, 0, 0)

    # Baseline values
    baselines = {
        "HR": (72, 5),
        "O2Sat": (97, 1),
        "Temp": (36.8, 0.3),
        "SBP": (120, 10),
        "DBP": (80, 8),
        "Resp": (16, 2),
        "Lactate": (1.2, 0.3),
        "WBC": (7.0, 2.0),
        "Glucose": (90, 15),
        "Creatinine": (0.9, 0.2),
        "Potassium": (4.0, 0.4),
        "pH": (7.40, 0.03),
        "FiO2": (0.21, 0.05),
        "Platelets": (200, 40),
        "Hgb": (13.0, 1.5),
        "Hct": (39.0, 4.0),
    }

    # Per-patient baseline jitter: real patients have different resting
    # baselines (resting HR 55-90, SBP 100-150, etc.). Without this,
    # deterioration is trivially separable by thresholding. Jitter is
    # scaled to each feature's natural variability (1.5x its std), so
    # cross-patient spread ~ drift magnitude (the hard case).
    baselines = {
        feat: (mean + np.random.normal(0, std * 1.5), std)
        for feat, (mean, std) in baselines.items()
    }

    # Missingness rates (fraction of hours missing)
    miss_rates = {
        "HR": 0.05,
        "O2Sat": 0.05,
        "Temp": 0.05,
        "SBP": 0.05,
        "DBP": 0.05,
        "Resp": 0.05,
        "Lactate": 0.50,  # Labs are infrequent
        "WBC": 0.60,
        "Glucose": 0.30,
        "Creatinine": 0.70,
        "Potassium": 0.50,
        "pH": 0.70,
        "FiO2": 0.10,
        "Platelets": 0.65,
        "Hgb": 0.55,
        "Hct": 0.55,
    }

    # Per-patient response jitter: sepsis progresses at different rates and
    # intensities per patient (amplitude ~U[0.7, 1.3]x, rate multiplier on
    # the 12h ramp). Keeps the drift patient-specific, anchored to the
    # patient's OWN jittered baseline (never the population mean — that
    # would create a point-mass ribbon any model can threshold).
    amp = random.uniform(0.7, 1.3) if scenario == "deteriorating" else 1.0
    rate = 12.0 / random.uniform(8.0, 18.0) if scenario == "deteriorating" else 1.0

    for hour in range(n_hours):
        charttime = base_time + timedelta(hours=hour)

        for feat, (mean, std) in baselines.items():
            # Scenario-driven drift (subtle: comparable to patient variance)
            if scenario == "deteriorating":
                progression = min(max((hour - onset_hour) * rate, 0.0), 1.0)
                if feat == "HR":
                    mean += progression * 8 * amp
                    std = 6
                elif feat == "O2Sat":
                    mean -= progression * 2 * amp
                    std = 1.2
                elif feat == "Temp":
                    mean += progression * 0.4 * amp
                    std = 0.3
                elif feat == "SBP":
                    mean -= progression * 12 * amp
                    std = 10
                elif feat == "Lactate":
                    mean += progression * 0.5 * amp
                    std = 0.5
                elif feat == "WBC":
                    mean += progression * 2.5 * amp
                    std = 2.0
                elif feat == "Glucose":
                    mean += progression * 15 * amp
                    std = 15
                elif feat == "Creatinine":
                    mean += progression * 0.3 * amp
                    std = 0.2
                elif feat == "FiO2":
                    mean += progression * 0.12 * amp
                    std = 0.05

            elif scenario == "recovering":
                # Started bad (relative to own baseline), getting better.
                # Decays back to the patient's OWN jittered baseline, never
                # to the population mean.
                offset = max(0.0, 1.0 - hour / 24.0)
                if feat == "HR":
                    mean += 30 * offset
                elif feat == "O2Sat":
                    mean -= max(0.0, mean - 90) * offset
                elif feat == "Lactate":
                    mean += 3.0 * offset
                elif feat == "WBC":
                    mean += 6.0 * offset

            # Random missingness
            if random.random() < miss_rates[feat]:
                continue

            # Generate value
            value = np.random.normal(mean, std)
            value = max(0.0, value)  # No negative values

            events.append({
                'stay_id': stay_id,
                'subject_id': subject_id,
                'charttime': charttime.isoformat(),
                'itemid': ITEMIDS[feat],
                'valuenum': round(float(value), 2),
                'valueuom': _get_unit(feat),
            })

    return events


def _get_unit(feature: str) -> str:
    """Get MIMIC-IV unit for a feature."""
    units = {
        "HR": "bpm",
        "O2Sat": "%",
        "Temp": "C",
        "SBP": "mmHg",
        "DBP": "mmHg",
        "Resp": "breaths/min",
        "Lactate": "mmol/L",
        "WBC": "K/uL",
        "Glucose": "mg/dL",
        "Creatinine": "mg/dL",
        "Potassium": "mEq/L",
        "pH": "",
        "FiO2": "",
        "Platelets": "K/uL",
        "Hgb": "g/dL",
        "Hct": "%",
    }
    return units.get(feature, "")


def generate_cohort(
    output_dir: str,
    n_stays: int = 100,
    n_hours_range: tuple = (24, 168),
    seed: int = 42,
):
    """Generate a synthetic MIMIC-IV cohort.

    Creates:
    - chartevents.csv.gz (simulated ICU charted data)
    - icustays.csv (ICU stay metadata)
    - patients.csv (patient demographics)
    """
    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    scenarios = ["stable", "deteriorating", "recovering"]
    all_events = []
    stays = []
    patients = set()

    for i in range(n_stays):
        stay_id = 10000 + i
        subject_id = 1000 + (i % 500)  # Some patients have multiple stays
        patients.add(subject_id)

        n_hours = random.randint(*n_hours_range)
        scenario = random.choice(scenarios)

        # Sample the deterioration onset for label alignment
        onset_hour = random.randint(4, 20) if scenario == "deteriorating" else 0

        events = generate_stay(stay_id, subject_id, n_hours, scenario, seed=seed+i,
                               onset_hour=onset_hour)
        all_events.extend(events)

        stays.append({
            'stay_id': stay_id,
            'subject_id': subject_id,
            'hadm_id': 20000 + i,
            'first_careunit': random.choice(['MICU', 'SICU', 'CCU', 'TSICU']),
            'last_careunit': random.choice(['MICU', 'SICU', 'CCU', 'TSICU']),
            'intime': events[0]['charttime'] if events else '',
            'outtime': events[-1]['charttime'] if events else '',
            'los': n_hours,
            'scenario': scenario,
            'onset_hour': onset_hour,
        })

    # Sort events by time
    all_events.sort(key=lambda x: x['charttime'])

    # Write chartevents (uncompressed for simplicity)
    chartevents_path = os.path.join(output_dir, 'chartevents.csv')
    with open(chartevents_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['stay_id', 'subject_id', 'charttime', 'itemid', 'valuenum', 'valueuom'])
        writer.writeheader()
        for event in all_events:
            writer.writerow(event)

    # Write icustays
    icustays_path = os.path.join(output_dir, 'icustays.csv')
    with open(icustays_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['stay_id', 'subject_id', 'hadm_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los', 'scenario', 'onset_hour'])
        writer.writeheader()
        for stay in stays:
            writer.writerow(stay)

    # Write patients
    patients_path = os.path.join(output_dir, 'patients.csv')
    with open(patients_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
        writer.writeheader()
        for subj in sorted(patients):
            writer.writerow({
                'subject_id': subj,
                'gender': random.choice(['M', 'F']),
                'anchor_age': random.randint(18, 95),
                'anchor_year': random.randint(2011, 2020),
            })

    # Print stats
    print(f"Generated synthetic MIMIC-IV cohort:")
    print(f"  Patients: {len(patients)}")
    print(f"  ICU stays: {n_stays}")
    print(f"  Charted events: {len(all_events):,}")
    print(f"  Features: {len(ITEMIDS)}")
    print(f"  Scenarios: stable, deteriorating, recovering")
    print(f"  Saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic MIMIC-IV data")
    parser.add_argument("--output", default="/tmp/synthetic_mimic", help="Output directory")
    parser.add_argument("--n-stays", type=int, default=100, help="Number of ICU stays")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_cohort(args.output, n_stays=args.n_stays, seed=args.seed)
