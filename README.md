# Institutional Grid — Policy Dashboard

Cross-institutional governance evaluation demo. Shows how the same patient
triggers different alerts under different institutional risk postures.

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

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Docker Container (edge-deployed V1)             │
│                                                  │
│  FastAPI ──→ PyTorch Inference (PhysioNet ckpt) │
│  POST /predict  — single-hour prediction         │
│  POST /stream   — full patient trajectory        │
│  GET  /patients — list demo patients             │
│                                                  │
│  Streamlit Dashboard (port 8501)                 │
│  Panel A: Aggressive ICU (low risk fine)         │
│  Panel B: Conservative Care Home (high risk)     │
│  Same patient, same data, different outcomes     │
└─────────────────────────────────────────────────┘
```

## What the Demo Shows

1. **Select a patient** — 5 real PhysioNet ICU patients with sepsis onset
2. **Adjust governance sliders** — cost scale, risk fine, TP bounty
3. **Run simulation** — streams patient data hour-by-hour through both profiles
4. **Compare outcomes** — when each profile triggers its first alert
5. **Inspect economics** — jurisdiction pie charts, top bidding cells, audit ledger

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/patients` | GET | List available demo patients |
| `/predict` | POST | Single-hour prediction with governance |
| `/stream` | POST | Full patient trajectory replay |

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
