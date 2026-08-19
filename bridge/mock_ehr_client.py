"""Mock EHR Client — simulates FHIR observation streams for testing.

Sends standard FHIR Observation resources to the Institutional Grid
backend to demonstrate real-time integration.

Usage:
    python mock_ehr_client.py [--patient-id P001] [--interval 5] [--n-hours 24]
"""
import argparse
import json
import random
import time
import sys
import requests
from datetime import datetime, timezone, timedelta

BASE_URL = "http://localhost:8000"

# ─── FHIR Observation Templates ──────────────────────────────────────────────

def make_observation(
    loinc_code: str,
    value: float,
    unit: str,
    timestamp: str,
    patient_id: str = "P001",
    status: str = "final",
) -> dict:
    """Build a FHIR Observation resource."""
    return {
        "resourceType": "Observation",
        "id": f"obs-{loinc_code}-{int(time.time()*1000)}",
        "status": status,
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": timestamp,
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": loinc_code,
                "display": loinc_code
            }]
        },
        "valueQuantity": {
            "value": value,
            "unit": unit,
        }
    }


def make_missing_observation(
    loinc_code: str,
    timestamp: str,
    patient_id: str = "P001",
    reason: str = "pending",
) -> dict:
    """Build a FHIR Observation with missing value (pending lab)."""
    return {
        "resourceType": "Observation",
        "id": f"obs-{loinc_code}-{int(time.time()*1000)}",
        "status": reason,  # "pending", "cancelled", "entered-in-error"
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": timestamp,
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": loinc_code,
            }]
        },
        # No valueQuantity — signals missingness
    }


def make_condition(
    icd_code: str,
    display: str,
    timestamp: str,
    patient_id: str = "P001",
) -> dict:
    """Build a FHIR Condition resource."""
    return {
        "resourceType": "Condition",
        "id": f"cond-{icd_code}-{int(time.time()*1000)}",
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "onsetDateTime": timestamp,
        "code": {
            "coding": [{
                "system": "http://hl7.org/fhir/sid/icd-10",
                "code": icd_code,
                "display": display,
            }]
        },
    }


# ─── Clinical Scenarios ──────────────────────────────────────────────────────

def scenario_healthy(hour: int) -> dict:
    """Normal vitals, stable patient."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=24-hour)).isoformat()
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": make_observation("8867-4", 72 + random.uniform(-3, 3), "bpm", ts)},
            {"resource": make_observation("2708-6", 97 + random.uniform(-1, 1), "%", ts)},
            {"resource": make_observation("8310-5", 36.8 + random.uniform(-0.3, 0.3), "C", ts)},
            {"resource": make_observation("8480-6", 120 + random.uniform(-10, 10), "mmHg", ts)},
            {"resource": make_observation("8462-4", 80 + random.uniform(-8, 8), "mmHg", ts)},
            {"resource": make_observation("9279-1", 16 + random.uniform(-2, 2), "breaths/min", ts)},
            {"resource": make_observation("2345-7", 90 + random.uniform(-10, 10), "mg/dL", ts)},
            {"resource": make_observation("2601-3", 2.0 + random.uniform(-0.2, 0.2), "mg/dL", ts)},
            {"resource": make_observation("2823-3", 4.0 + random.uniform(-0.3, 0.3), "mEq/L", ts)},
            {"resource": make_observation("718-7", 14.0 + random.uniform(-1, 1), "g/dL", ts)},
        ]
    }


def scenario_deteriorating(hour: int) -> dict:
    """Progressive clinical deterioration — sepsis trajectory.

    Over 12 hours: HR rises, O2Sat drops, temp spikes, labs worsen.
    """
    progression = min(hour / 12.0, 1.0)  # 0 -> 1 over 12 hours

    hr = 72 + progression * 40 + random.uniform(-2, 2)
    o2sat = 97 - progression * 8 + random.uniform(-1, 1)
    temp = 36.8 + progression * 2.0 + random.uniform(-0.2, 0.2)
    sbp = 120 - progression * 25 + random.uniform(-5, 5)
    dbp = 80 - progression * 15 + random.uniform(-4, 4)
    resp = 16 + progression * 12 + random.uniform(-1, 1)
    lactate = 1.2 + progression * 4.0 + random.uniform(-0.2, 0.2)
    wbc = 7.0 + progression * 10 + random.uniform(-0.5, 0.5)

    ts = (datetime.now(timezone.utc) - timedelta(hours=24-hour)).isoformat()

    # Some labs become missing as deterioration progresses (delayed labs)
    entries = [
        {"resource": make_observation("8867-4", hr, "bpm", ts)},
        {"resource": make_observation("2708-6", o2sat, "%", ts)},
        {"resource": make_observation("8310-5", temp, "C", ts)},
        {"resource": make_observation("8480-6", sbp, "mmHg", ts)},
        {"resource": make_observation("8462-4", dbp, "mmHg", ts)},
        {"resource": make_observation("9279-1", resp, "breaths/min", ts)},
        {"resource": make_observation("2524-7", lactate, "mmol/L", ts)},
        {"resource": make_observation("6690-2", wbc, "K/uL", ts)},
    ]

    # Delayed labs (some missing early, arrive later)
    if hour < 6:
        entries.append({"resource": make_missing_observation("2019-8", ts, reason="pending")})
        entries.append({"resource": make_missing_observation("2160-0", ts, reason="pending")})
    else:
        entries.append({"resource": make_observation("2019-8", 45 + progression * 15, "mmHg", ts)})
        entries.append({"resource": make_observation("2160-0", 1.0 + progression * 1.5, "mg/dL", ts)})

    # Add sepsis condition at hour 8+
    if hour >= 8:
        entries.append({
            "resource": make_condition("A41.9", "Sepsis, unspecified organism", ts)
        })

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }


# ─── Client ──────────────────────────────────────────────────────────────────

def run_simulation(patient_id: str, scenario: str, n_hours: int, interval: float):
    """Run a multi-hour FHIR observation stream simulation."""
    print(f"Starting FHIR simulation: patient={patient_id}, scenario={scenario}")
    print(f"  Hours: {n_hours}, Interval: {interval}s")
    print(f"  Endpoint: {BASE_URL}")
    print()

    # Reset patient state
    requests.post(f"{BASE_URL}/fhir/patient/{patient_id}/reset")

    scenario_fn = {
        "healthy": scenario_healthy,
        "deteriorating": scenario_deteriorating,
    }.get(scenario, scenario_healthy)

    for hour in range(n_hours):
        bundle = scenario_fn(hour)
        payload = {
            "patient_id": patient_id,
            "bundle": bundle,
            "governance": {"cost_scale": 1.0, "risk_scale": 1.5, "bounty": 1.0},
        }

        try:
            # Ingest
            resp = requests.post(f"{BASE_URL}/fhir/ingest", json=payload, timeout=5)
            ingest_result = resp.json()

            # Predict
            pred_resp = requests.post(
                f"{BASE_URL}/fhir/predict",
                json={"patient_id": patient_id, "governance": payload["governance"]},
                timeout=5,
            )
            pred_result = pred_resp.json()

            alert = "🚨 ALERT" if pred_result.get("alert") else "✅ CLEAR"
            pred = pred_result.get("prediction", 0)
            n_obs = pred_result.get("n_features_observed", 0)
            completeness = pred_result.get("completeness", 0)

            print(f"Hour {hour:2d} | pred={pred:.4f} | {alert} | obs={n_obs}")

        except requests.ConnectionError:
            print(f"ERROR: Cannot connect to {BASE_URL}")
            print("Start the backend: cd backend && uvicorn main:app --reload")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR hour {hour}: {e}")

        time.sleep(interval)

    print("\nSimulation complete.")
    print(f"\nAudit ledger: {BASE_URL}/fhir/audit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock EHR Client — FHIR observation stream")
    parser.add_argument("--patient-id", default="P001", help="Patient ID")
    parser.add_argument("--scenario", default="deteriorating", choices=["healthy", "deteriorating"])
    parser.add_argument("--n-hours", type=int, default=24, help="Number of hours to simulate")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between hours")
    args = parser.parse_args()

    run_simulation(args.patient_id, args.scenario, args.n_hours, args.interval)
