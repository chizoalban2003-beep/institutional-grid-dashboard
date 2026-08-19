# FHIR Bridge — HL7/FHIR Integration Layer

Translates standard **FHIR R4** resources from EHR systems (Epic, Cerner, PCS) into the `(N, K×3)` tensor format required by the Institutional Grid Engine.

## Architecture

```
┌─────────────────┐     FHIR JSON       ┌──────────────────┐     (W,K×3)      ┌──────────────┐
│   EHR System    │ ──────────────────► │   FHIR Bridge    │ ───────────────► │ Grid Engine  │
│  (Epic/Cerner)  │   Observation       │  fhir_bridge.py  │   Tensor         │ (PyTorch)    │
│                 │   Condition         │                  │                  │              │
└─────────────────┘                     └──────────────────┘                  └──────────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │  Audit Ledger    │
                                    │ DiagnosticReport │
                                    │ (writes back to  │
                                    │  hospital EHR)   │
                                    └──────────────────┘
```

## Quick Start

### 1. Start the Backend
```bash
cd backend && uvicorn main:app --reload --port 8000
```

### 2. Run the FHIR Demo
```bash
# Deteriorating patient (sepsis trajectory)
python3 bridge/mock_ehr_client.py --scenario deteriorating --n-hours 24

# Stable patient
python3 bridge/mock_ehr_client.py --scenario healthy --n-hours 24
```

### 3. Check the Audit Ledger
```bash
curl http://localhost:8000/fhir/audit | jq
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fhir/ingest` | POST | Ingest FHIR Bundle of observations |
| `/fhir/predict` | POST | Run inference on current patient state |
| `/fhir/patient/{id}/state` | GET | View current patient state |
| `/fhir/audit` | GET | Get recent DiagnosticReports |
| `/fhir/patient/{id}/reset` | POST | Reset patient state |

## FHIR → Tensor Mapping

The bridge maps **39 clinical features** via LOINC/SNOMED codes:

| FHIR Code | Feature | Type |
|-----------|---------|------|
| `8867-4` | HR | Vital |
| `2708-6` | O2Sat | Vital |
| `8310-5` | Temp | Vital |
| `2744-1` | pH | Lab |
| `2524-7` | Lactate | Lab |
| `A41.9` | Sepsis (ICD-10) | Condition |

## Tensor Format

Each patient-hour becomes a `(W, K×3)` tensor:
- **W** = 14 (window size, hours)
- **K** = 39 (clinical features)
- **K×3** = `[z_value || mask || delta]` per feature

Where:
- `z_value` = z-scored observation value
- `mask` = 1 if observed, 0 if missing
- `delta` = hours since last observation (capped at 48h)

## Integration with Hospital Systems

### Epic EHR
```python
# Epic sends FHIR via SMART-on-FHIR or HL7 v2 → FHIR bridge
import requests

response = requests.post("http://grid-engine:8000/fhir/ingest", json={
    "patient_id": "MRN-12345",
    "bundle": {  # Epic FHIR Observation bundle
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [...]
    },
    "governance": {"cost_scale": 1.0, "risk_scale": 1.5, "bounty": 1.0}
})
```

### Cerner EHR
```python
# Cerner sends via FHIR Subscriptions
# Subscribe to observation updates:
subscription = {
    "resourceType": "Subscription",
    "status": "requested",
    "criteria": "Observation?patient=Patient/MRN-12345",
    "channel": {
        "type": "rest-hook",
        "endpoint": "http://grid-engine:8000/fhir/ingest"
    }
}
```

## Output: FHIR DiagnosticReport

When the Grid makes a prediction, it generates a FHIR `DiagnosticReport`:

```json
{
  "resourceType": "DiagnosticReport",
  "status": "final",
  "code": {
    "coding": [{
      "system": "http://institutional-grid.ai/code-system",
      "code": "SEPSIS_RISK_ASSESSMENT"
    }]
  },
  "subject": {"reference": "Patient/MRN-12345"},
  "result": [
    {"valueQuantity": {"value": 0.8765, "unit": "probability"}},
    {"valueString": "{\"cost_scale\": 1.0, \"risk_scale\": 1.5}"},
    {"valueString": "{\"jurisdiction_pies\": {...}, \"top_cells\": [...]}"}
  ],
  "conclusion": "SEPSIS_ALERT"
}
```

This is written back to the hospital's EHR for:
- **CQC compliance** (UK Care Quality Commission)
- **FDA transparency** (US regulatory audit trail)
- **Clinical governance** (local policy justification)

## Tests

```bash
make test-fhir    # 17 tests
```
