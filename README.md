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

## MIMIC-IV Data Engine

The `data_engine/` module translates MIMIC-IV's relational EHR tables into the Grid's tensor format:

```bash
# Generate synthetic MIMIC data (100 ICU stays)
make mimic-generate

# Run MIMIC ingestion tests
make test-mimic
```

### Architecture

```
MIMIC-IV CSV/Parquet ──→ MIMICStayAssembler ──→ (T, K×3) tensor
  chartevents.csv         - Forward fill         - Value (z-scored)
  labevents.csv           - Delta computation     - Mask (0/1)
  icustays.csv            - Window extraction     - Delta (hours since obs)
```

### Features

- **100+ MIMIC itemids** mapped to 34 clinical features (plus 5 metadata)
- **Irregular sampling** → hourly bins with forward-fill
- **12h+ lab delays** handled via delta tracking (capped at 48h)
- **Memmap output** for memory-efficient tensor storage
- **Synthetic data generator** for pipeline validation without full dataset

### Tensor Format

Each patient-hour becomes a `(W, K×3)` tensor:
- **W** = 14 (window size, hours)
- **K** = 39 (clinical features)
- **K×3** = `[z_value || mask || delta]` per feature

### Tests

```bash
make test       # All tests (34 passing: 17 FHIR + 17 MIMIC)
make test-fhir  # FHIR bridge tests
make test-mimic # MIMIC ingestion tests
```
