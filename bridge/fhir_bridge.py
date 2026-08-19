"""FHIR Bridge — HL7/FHIR to Institutional Grid Tensor Translator.

Accepts standard FHIR Observation/Condition resources from EHR systems
(Epic, Cerner, PCS) and translates them into the (N, K×3) tensor format
expected by the Grid Engine.

Produces FHIR DiagnosticReport resources for audit ledger output.

Mapping:
  FHIR Observation.value → Value (z-scored)
  FHIR Observation absent/pending → Mask (0 = missing)
  Time since last observation → Delta (capped at 48h)
"""
import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field

# ─── Feature Schema: FHIR → Grid Mapping ─────────────────────────────────────

# Maps FHIR LOINC/SNOMED codes to our 39 clinical features
FHIR_TO_FEATURE = {
    # Vitals (LOINC codes)
    "8867-4": "HR",              # Heart rate
    "2708-6": "O2Sat",           # Oxygen saturation
    "8310-5": "Temp",            # Body temperature
    "8480-6": "SBP",             # Systolic BP
    "8462-4": "DBP",             # Diastolic BP
    "8478-0": "MAP",             # Mean arterial pressure (computed if absent)
    "9279-1": "Resp",            # Respiratory rate
    "71942-4": "EtCO2",          # End tidal CO2

    # Labs (LOINC codes)
    "34552-6": "BaseExcess",     # Base excess
    "1963-8": "HCO3",            # Bicarbonate
    "3150-0": "FiO2",            # FiO2
    "2744-1": "pH",              # pH
    "2019-8": "PaCO2",           # PaCO2
    "2703-7": "SaO2",            # SaO2
    "1920-8": "AST",             # AST
    "12966-8": "BUN",            # BUN
    "6768-6": "Alkalinephos",    # Alkaline phosphatase
    "17861-6": "Calcium",        # Calcium
    "2075-0": "Chloride",        # Chloride
    "2160-0": "Creatinine",      # Creatinine
    "1968-7": "Bilirubin_direct",# Direct bilirubin
    "2345-7": "Glucose",         # Glucose
    "2524-7": "Lactate",         # Lactate
    "2601-3": "Magnesium",       # Magnesium
    "2777-1": "Phosphate",       # Phosphate
    "2823-3": "Potassium",       # Potassium
    "1974-5": "Bilirubin_total", # Total bilirubin
    "10839-9": "TroponinI",      # Troponin I
    "20570-8": "Hct",            # Hematocrit
    "718-7": "Hgb",              # Hemoglobin
    "5902-2": "PTT",             # PTT
    "6690-2": "WBC",             # WBC
    "32558-5": "Fibrinogen",     # Fibrinogen
    "777-3": "Platelets",        # Platelets

    # Demographics / Context
    "AGE": "Age",
    "GENDER": "Gender",
    "UNIT1": "Unit1",
    "UNIT2": "Unit2",
    "HOSP_ADM_TIME": "HospAdmTime",
}

# Reverse mapping: feature name → index
FEATURE_NAMES = [
    'HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3',
    'FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride',
    'Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate',
    'Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC',
    'Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime'
]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
K = len(FEATURE_NAMES)  # 39
K3 = K * 3  # 117

# ─── Patient State Tracker ───────────────────────────────────────────────────

@dataclass
class FeatureState:
    """Tracks a single feature's temporal state for one patient."""
    last_value: Optional[float] = None
    last_observed_at: Optional[float] = None  # Unix timestamp
    n_missing: int = 0
    total_observed: int = 0

    @property
    def time_since_observation(self) -> float:
        """Hours since last observation."""
        if self.last_observed_at is None:
            return float('inf')
        return (time.time() - self.last_observed_at) / 3600.0


@dataclass
class PatientState:
    """Tracks all feature states for one patient."""
    patient_id: str
    features: dict = field(default_factory=dict)
    window_size: int = 14
    hour: int = 0

    def __post_init__(self):
        if not self.features:
            self.features = {name: FeatureState() for name in FEATURE_NAMES}

    def ingest_observation(self, fhir_obs: dict):
        """Process a FHIR Observation resource.

        Example FHIR Observation:
        {
            "resourceType": "Observation",
            "id": "obs-123",
            "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
            "valueQuantity": {"value": 72, "unit": "bpm"},
            "effectiveDateTime": "2024-01-15T10:30:00Z",
            "status": "final"
        }
        """
        # Extract LOINC code
        code = None
        coding = fhir_obs.get("code", {}).get("coding", [])
        if coding:
            code = coding[0].get("code")

        if code not in FHIR_TO_FEATURE:
            return  # Unknown feature, skip

        feature_name = FHIR_TO_FEATURE[code]
        fs = self.features[feature_name]

        # Extract value
        value = None
        if "valueQuantity" in fhir_obs:
            value = fhir_obs["valueQuantity"].get("value")
        elif "valueString" in fhir_obs:
            val_str = fhir_obs["valueString"]
            try:
                value = float(val_str)
            except (ValueError, TypeError):
                value = None

        # Parse timestamp
        ts = None
        if "effectiveDateTime" in fhir_obs:
            try:
                dt = datetime.fromisoformat(fhir_obs["effectiveDateTime"].replace("Z", "+00:00"))
                ts = dt.timestamp()
            except:
                ts = time.time()
        else:
            ts = time.time()

        if value is not None:
            fs.last_value = value
            fs.last_observed_at = ts
            fs.n_missing = 0
            fs.total_observed += 1
        else:
            # Missing observation (pending, cancelled, etc.)
            status = fhir_obs.get("status", "unknown")
            if status in ("cancelled", "entered-in-error"):
                pass  # Ignore
            else:
                fs.n_missing += 1

    def get_current_vector(self) -> tuple:
        """Get current [value, mask, delta] vector (3K dims)."""
        now = time.time()
        values = np.zeros(K, dtype=np.float32)
        mask = np.zeros(K, dtype=np.float32)
        delta = np.zeros(K, dtype=np.float32)

        for i, name in enumerate(FEATURE_NAMES):
            fs = self.features[name]
            if fs.last_value is not None:
                values[i] = fs.last_value
                mask[i] = 1.0
                delta[i] = min((now - fs.last_observed_at) / 3600.0, 48.0)
            else:
                delta[i] = 48.0  # Max cap

        return values, mask, delta

    def get_window_tensor(self, z_mu: np.ndarray, z_sd: np.ndarray) -> np.ndarray:
        """Build (W, K×3) window tensor from current state.

        Uses the current values as the final hour, with history from
        the patient's tracked states.
        """
        W = self.window_size
        tensor = np.zeros((W, K3), dtype=np.float32)

        values, mask, delta = self.get_current_vector()

        # Z-score
        z = np.zeros(K, dtype=np.float32)
        for c in range(K):
            if z_sd[c] > 1e-6:
                z[c] = (values[c] - z_mu[c]) / z_sd[c]
            else:
                z[c] = values[c] - z_mu[c]

        # For a live stream, we only have the current hour.
        # Fill the window with current values (simulates "steady state" history)
        # In production, this would be populated from historical observations.
        content = np.concatenate([z, mask, delta], axis=-1)
        tensor[-1] = content  # Last row = current hour

        # Fill earlier rows with slightly decayed values (simulates history)
        # In production, these would come from actual past observations
        for t in range(W - 1):
            decay = 1.0 - 0.05 * (W - 1 - t)  # 5% decay per step back
            tensor[t] = content * decay

        return tensor

    def get_sepsis_label(self) -> int:
        """Determine sepsis label from FHIR Condition resources.

        In production, this would check for active sepsis conditions.
        For the bridge demo, returns 0 (stable) — labels come from the EHR.
        """
        return 0


# ─── FHIR DiagnosticReport Output ────────────────────────────────────────────

def build_diagnostic_report(
    patient_id: str,
    prediction: float,
    governance: dict,
    jurisdiction_pies: dict,
    top_cells: list,
    alert_triggered: bool,
) -> dict:
    """Build FHIR DiagnosticReport for audit ledger.

    This is what gets written back to the hospital's EHR system.
    """
    report = {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "CG",
                "display": "Cytogenetics"  # Placeholder — use custom code system
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://institutional-grid.ai/code-system",
                "code": "SEPSIS_RISK_ASSESSMENT",
                "display": "Institutional Grid Sepsis Risk Assessment"
            }]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
        "issued": datetime.now(timezone.utc).isoformat(),
        "result": [
            {
                "resourceType": "Observation",
                "code": {
                    "coding": [{
                        "system": "http://institutional-grid.ai/code-system",
                        "code": "SEPSIS_PREDICTION",
                        "display": "Sepsis Risk Prediction"
                    }]
                },
                "valueQuantity": {
                    "value": round(prediction, 4),
                    "unit": "probability"
                },
                "interpretation": [{
                    "coding": [{
                        "code": "H" if prediction > 0.5 else "N",
                        "display": "High Risk" if prediction > 0.5 else "Normal"
                    }]
                }]
            },
            {
                "resourceType": "Observation",
                "code": {
                    "coding": [{
                        "system": "http://institutional-grid.ai/code-system",
                        "code": "GOVERNANCE_POLICY",
                        "display": "Applied Governance Policy"
                    }]
                },
                "valueString": json.dumps(governance)
            },
            {
                "resourceType": "Observation",
                "code": {
                    "coding": [{
                        "system": "http://institutional-grid.ai/code-system",
                        "code": "CELL_ECONOMY",
                        "display": "Cell Economic Posture"
                    }]
                },
                "valueString": json.dumps({
                    "jurisdiction_pies": jurisdiction_pies,
                    "top_cells": top_cells,
                    "alert_triggered": alert_triggered
                })
            }
        ],
        "conclusion": "SEPSIS_ALERT" if alert_triggered else "NO_ALERT",
        "conclusionCode": [{
            "coding": [{
                "system": "http://institutional-grid.ai/code-system",
                "code": "ALERT" if alert_triggered else "CLEAR",
            }]
        }]
    }
    return report


# ─── FHIR Bundle Ingestion ───────────────────────────────────────────────────

def ingest_fhir_bundle(
    patient_state: PatientState,
    fhir_bundle: dict
) -> dict:
    """Process a FHIR Bundle of observations for one patient.

    Example Bundle:
    {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {...Observation...}},
            {"resource": {...Observation...}}
        ]
    }

    Returns:
        {
            "n_processed": int,
            "n_skipped": int,
            "current_hour": int,
            "features_updated": [str],
        }
    """
    n_processed = 0
    n_skipped = 0
    features_updated = []

    entries = fhir_bundle.get("entry", [])
    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")

        if rtype == "Observation":
            prev_state = patient_state.features.get(
                FHIR_TO_FEATURE.get(
                    resource.get("code", {}).get("coding", [{}])[0].get("code", "")
                )
            )
            patient_state.ingest_observation(resource)
            n_processed += 1

            # Track which features were updated
            code = resource.get("code", {}).get("coding", [{}])[0].get("code", "")
            fname = FHIR_TO_FEATURE.get(code)
            if fname:
                features_updated.append(fname)

        elif rtype == "Condition":
            # Track sepsis diagnosis
            codes = resource.get("code", {}).get("coding", [])
            for coding in codes:
                if coding.get("code") in ("A41.9", "R65.20", "R65.21"):  # ICD-10 sepsis codes
                    pass  # Mark patient as septic in production
            n_processed += 1

        else:
            n_skipped += 1

    patient_state.hour += 1

    return {
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "current_hour": patient_state.hour,
        "features_updated": features_updated,
    }
