# Institutional Grid — Policy Dashboard

Cross-institutional governance evaluation demo. Shows how the same patient
triggers different alerts under different institutional risk postures.

**Now with HL7/FHIR integration** — connect directly to Epic, Cerner, or any FHIR-capable EHR.

## Quick Start

### Option A: Docker (Recommended)

```bash
cd grid-dashboard
make download-ckpt  # Fetch trained PhysioNet weights from Kaggle
make build
make run
```

Dashboard: **http://localhost:8501**
API docs: **http://localhost:8000/docs**

### Option B: Direct Python

```bash
cd grid-dashboard
pip install -r backend/requirements.txt -r frontend/requirements.txt
# Backend
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 &
# Frontend (new terminal)
cd frontend && BACKEND_URL=http://localhost:8000 streamlit run app.py
```

## FHIR Integration

The system now accepts **standard FHIR R4** resources from EHR systems:

```bash
# Start backend
cd backend && uvicorn main:app --reload --port 8000

# Run the FHIR demo (simulates a deteriorating patient)
python3 bridge/mock_ehr_client.py --scenario deteriorating --n-hours 24

# Check the audit ledger (FHIR DiagnosticReports)
curl http://localhost:8000/fhir/audit
```

See `bridge/README.md` for full integration docs.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Docker Container (edge-deployed, GDPR/HIPAA by design)              │
│                                                                       │
│  ┌─────────────────┐     FHIR JSON       ┌──────────────────┐        │
│  │   EHR System    │ ──────────────────► │   FHIR Bridge    │        │
│  │  (Epic/Cerner)  │   Observation       │  fhir_bridge.py  │        │
│  │                 │   Condition         │                  │        │
│  └─────────────────┘                     └────────┬─────────┘        │
│                                                   │ (W,K×3)          │
│                                          ┌────────▼─────────┐        │
│  FastAPI ──→ PyTorch Inference           │  Grid Engine     │        │
│  POST /predict  — single-hour prediction │  (PyTorch ckpt)  │        │
│  POST /stream   — full patient trajectory│                  │        │
│  POST /fhir/ingest — FHIR bundle ingest  └────────┬─────────┘        │
│  GET  /fhir/audit  — audit ledger                 │                   │
│                                                   ▼                   │
│  Streamlit Dashboard (port 8501)           ┌──────────────┐          │
│  Panel A: Aggressive ICU (low risk fine)   │ Audit Ledger │          │
│  Panel B: Conservative Care Home (high)    │ FHIR Report  │          │
│  Same patient, same data, different outcome│ (writes back │          │
│  └─────────────────────────────────────────┘ to EHR)      │          │
│                                               └──────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
```

## What the Demo Shows

1. **Select a patient** — 5 real PhysioNet ICU patients with sepsis onset
2. **Adjust governance sliders** — cost scale, risk fine, TP bounty
3. **Run simulation** — streams patient data hour-by-hour through both profiles
4. **Compare outcomes** — when each profile triggers its first alert
5. **Inspect economics** — jurisdiction pie charts, top bidding cells, audit ledger

## API Endpoints

### Core Engine

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/patients` | GET | List available demo patients |
| `/predict` | POST | Single-hour prediction with governance |
| `/stream` | POST | Full patient trajectory replay |

### FHIR Bridge

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fhir/ingest` | POST | Ingest FHIR Bundle of observations |
| `/fhir/predict` | POST | Run inference on FHIR-tracked patient |
| `/fhir/patient/{id}/state` | GET | View current patient state |
| `/fhir/audit` | GET | Get recent DiagnosticReports |
| `/fhir/patient/{id}/reset` | POST | Reset patient state |

### Example Request

```json
POST /predict
{
  "patient_id": "p000011",
  "hour": 6,
  "governance": {
    "cost_scale": 0.3,
    "risk_scale": 0.2,
    "bounty": 3.0
  }
}
```

## Model

- **Architecture**: TrainableGrid (GRU encoder + 100-cell MoE + differentiable router)
- **Checkpoint**: PhysioNet 2019 Sepsis, seed 0 (val AUC 0.8195)
- **Features**: 39 clinical variables → [value || mask || delta] tensor (117-dim)
- **Window**: 14 hours causal history, right-aligned with zero-padding

## FHIR → Tensor Mapping

The bridge maps **39 clinical features** via LOINC/SNOMED codes from FHIR Observations:

| FHIR Code | Feature | Type |
|-----------|---------|------|
| `8867-4` | HR | Vital |
| `2708-6` | O2Sat | Vital |
| `8310-5` | Temp | Vital |
| `2744-1` | pH | Lab |
| `2524-7` | Lactate | Lab |
| `A41.9` | Sepsis (ICD-10) | Condition |

Full mapping in `bridge/fhir_bridge.py`.

## Tests

```bash
make test       # All tests (17 passing)
make test-fhir  # FHIR bridge tests
```
