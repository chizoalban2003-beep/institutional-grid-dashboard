"""Institutional Grid Inference Engine — FastAPI Backend.

Loads trained PhysioNet checkpoint and serves real-time predictions
with configurable governance parameters (cost/risk/bounty weights).

Endpoints:
  POST /predict  — single-hour prediction with governance params
  POST /stream   — replay patient trajectory hour-by-hour
  GET  /patients — list available demo patients
  GET  /health   — health check
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os, json, glob
from typing import Optional

# Add bridge to path so it can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bridge.fhir_bridge import PatientState, ingest_fhir_bundle, build_diagnostic_report
from backend.mimic.loader import get_mimic_loader, load_mimic_into_backend

# ─── FHIR Patient State Store ────────────────────────────────────────────────

fhir_patients: dict[str, PatientState] = {}  # patient_id -> PatientState
fhir_audit_log: list[dict] = []  # DiagnosticReports for audit ledger

app = FastAPI(title="Institutional Grid Inference Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Model Architecture (matches train_grid.py) ─────────────────────────────

K = 39
K3 = K * 3
WINDOW_SIZE = 14
N_CELLS = 100
K_ACTIVE = 3

FEATURE_COLS = [
    'HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3',
    'FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride',
    'Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate',
    'Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC',
    'Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime'
]

class DifferentiableRouter(nn.Module):
    def __init__(self, n_cells=100, k=3, tau=1.0):
        super().__init__()
        self.n_cells = n_cells
        self.k = k
        self.tau = tau
        self.gate_logits = nn.Parameter(torch.zeros(n_cells))

    def forward(self, bids, training=False):
        if training and self.tau > 0.1:
            gumbel = -torch.log(-torch.log(torch.rand_like(bids) + 1e-10) + 1e-10)
            logits = bids + self.gate_logits + gumbel
            weights = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
            selected_idx = torch.topk(bids, self.k, dim=-1).indices
        else:
            weights = torch.zeros_like(bids)
            _, topk_idx = torch.topk(bids, self.k, dim=-1)
            weights.scatter_(-1, topk_idx, 1.0)
            selected_idx = topk_idx
        return weights, selected_idx


class TrainableCell(nn.Module):
    def __init__(self, input_dim, jurisdiction="GENERAL"):
        super().__init__()
        self.input_dim = input_dim
        self.jurisdiction = jurisdiction
        self.W = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
        self.pie_logits = nn.Parameter(torch.zeros(3))
        self.budget = 100.0

    def forward(self, z):
        confidence = torch.sigmoid(z @ self.W + self.b)
        pie_weights = F.softmax(self.pie_logits, dim=-1).unsqueeze(0).expand(z.shape[0], -1)
        return confidence, pie_weights

    def compute_bid(self, z, uncertainty=None, resource_cost=None, governance=None):
        """Compute bid with governance-adjusted penalties.

        governance = {'cost_scale': float, 'risk_scale': float, 'bounty': float}
        """
        confidence, pie = self.forward(z)
        cost_scale = governance.get('cost_scale', 1.0) if governance else 1.0
        risk_scale = governance.get('risk_scale', 1.0) if governance else 1.0
        penalty = torch.ones_like(confidence)
        if uncertainty is not None:
            penalty = penalty - pie[:, 1] * uncertainty * risk_scale
        if resource_cost is not None:
            penalty = penalty - pie[:, 0] * resource_cost * cost_scale
        return confidence * penalty.clamp(min=0.0)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                         dropout=0.1 if n_layers > 1 else 0.0)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]


class TrainableGrid(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_cells=100, k=3, tau=1.0):
        super().__init__()
        self.n_cells = n_cells
        self.k = k
        self.encoder = TemporalEncoder(input_dim, hidden_dim)
        jurisdictions = ["VITALS", "HYDRATION", "MOBILITY", "COGNITIVE", "OPERATIONAL"]
        self.cells = nn.ModuleList([
            TrainableCell(hidden_dim, jur)
            for jur in (jurisdictions * (n_cells // len(jurisdictions) + 1))[:n_cells]
        ])
        self.router = DifferentiableRouter(n_cells, k, tau)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * k, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x, governance=None, training=False):
        h = self.encoder(x)
        confidences, pie_weights, bids = [], [], []
        for cell in self.cells:
            conf, pie = cell(h)
            bid = cell.compute_bid(h, governance=governance)
            confidences.append(conf)
            pie_weights.append(pie)
            bids.append(bid)
        confidences = torch.stack(confidences, dim=-1)
        pie_weights = torch.stack(pie_weights, dim=-1)
        bids = torch.stack(bids, dim=-1)
        route_weights, selected_idx = self.router(bids, training=training)
        selected_confidences = torch.gather(confidences, 1, selected_idx)
        predictions = selected_confidences.mean(dim=-1)
        return {
            'predictions': predictions,
            'confidences': confidences,
            'bids': bids,
            'route_weights': route_weights,
            'selected_idx': selected_idx,
            'pie_weights': pie_weights,
        }


# ─── Global State ────────────────────────────────────────────────────────────

model = None
z_mu = None
z_sd = None
patient_data = {}  # pid -> {'hours': [...], 'labels': [...], 'windows': [...]}

CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_model():
    """Load trained checkpoint and patient data."""
    global model, z_mu, z_sd, patient_data

    # Find checkpoint
    ckpt_files = glob.glob(os.path.join(CKPT_DIR, 'grid_physionet_s*.pt'))
    if not ckpt_files:
        # Fallback: try /tmp
        ckpt_files = glob.glob('/tmp/kaggle_grid_out/grid_physionet_s*.pt')
    if not ckpt_files:
        raise RuntimeError("No checkpoint found. Run 'make download-ckpt' first.")

    ckpt_path = sorted(ckpt_files)[0]  # seed 0
    print(f"Loading checkpoint: {ckpt_path}", flush=True)

    model = TrainableGrid(input_dim=K3, hidden_dim=64, n_cells=N_CELLS, k=K_ACTIVE, tau=1.0)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    # Load z-score stats from summary
    summary_files = glob.glob(os.path.join(CKPT_DIR, 'grid_physionet_summary.json'))
    if not summary_files:
        summary_files = glob.glob('/tmp/kaggle_grid_out/grid_physionet_summary.json')
    if summary_files:
        with open(summary_files[0]) as f:
            summary = json.load(f)
        # Stats are computed during training — we need to recompute from patient data
        # For now, use reasonable defaults (will be overridden when patient data loads)

    # Load patient data
    load_patient_data()

    print(f"Model loaded: {len(ckpt_files)} checkpoints, {len(patient_data)} demo patients", flush=True)


def load_patient_data():
    """Load demo patient trajectories from local PhysioNet files."""
    global patient_data, z_mu, z_sd

    data_dir = DATA_DIR
    if not os.path.exists(os.path.join(data_dir, 'patients.json')):
        # Generate from raw .psv files
        generate_demo_patients()
        return

    with open(os.path.join(data_dir, 'patients.json')) as f:
        patient_data = json.load(f)

    # Load z-score stats
    stats_path = os.path.join(data_dir, 'z_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        z_mu = np.array(stats['mu'], dtype=np.float32)
        z_sd = np.array(stats['sd'], dtype=np.float32)


def generate_demo_patients():
    """Extract real patient trajectories from PhysioNet .psv files."""
    import pandas as pd

    data_a = os.path.expanduser('~/l3_sepsis/data/training_setA/training')
    files = sorted(glob.glob(os.path.join(data_a, '*.psv')))

    # Find sepsis patients with 15-50 hours of data
    candidates = []
    for f in files:
        df = pd.read_csv(f, sep='|')
        n_pos = int(df['SepsisLabel'].sum())
        n_hours = len(df)
        if n_pos > 0 and 15 <= n_hours <= 60:
            candidates.append((os.path.basename(f), n_pos, n_hours, df))

    candidates.sort(key=lambda x: -x[1])

    # Take top 5
    global patient_data, z_mu, z_sd
    patient_data = {}
    all_vals = []
    all_mask = []

    for pid, n_pos, n_hours, df in candidates[:5]:
        vals = df[FEATURE_COLS].values.astype(np.float32)
        mask = np.isfinite(vals).astype(np.float32)

        # Forward fill
        ff = vals.copy()
        for c in range(K):
            last = np.nan
            for t in range(len(vals)):
                if np.isfinite(vals[t, c]):
                    last = vals[t, c]
                elif not np.isnan(last):
                    ff[t, c] = last
                else:
                    ff[t, c] = 0.0

        # Delta
        delta = np.zeros((len(vals), K), dtype=np.float32)
        for c in range(K):
            d = 0.0
            for t in range(len(vals)):
                if mask[t, c]:
                    d = 0.0
                else:
                    d += 1.0
                delta[t, c] = min(d, 48.0)

        labels = df['SepsisLabel'].values.astype(int).tolist()

        # Accumulate stats
        for c in range(K):
            obs = ff[:, c][mask[:, c] > 0]
            if len(obs) > 0:
                all_vals.append(obs)

        patient_data[pid] = {
            'n_hours': n_hours,
            'n_positive': n_pos,
            'labels': labels,
            'raw_values': vals.tolist(),
            'mask': mask.tolist(),
            'delta': delta.tolist(),
            'ffilled': ff.tolist(),
        }

    # Compute z-score stats
    all_vals = np.concatenate(all_vals, axis=0)
    z_mu = all_vals.mean(axis=0).astype(np.float32)
    z_sd = all_vals.std(axis=0).astype(np.float32)
    z_sd[z_sd < 1e-6] = 1.0

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    # Save patient data without raw arrays (too large for JSON)
    save_data = {}
    for pid, pdata in patient_data.items():
        save_data[pid] = {k: v for k, v in pdata.items() if k not in ('raw_values', 'ffilled', 'mask', 'delta')}
    with open(os.path.join(DATA_DIR, 'patients.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    with open(os.path.join(DATA_DIR, 'z_stats.json'), 'w') as f:
        json.dump({'mu': z_mu.tolist(), 'sd': z_sd.tolist()}, f)


def zscore_window(window_raw, window_mask, window_delta):
    """Build (W, K3) tensor from raw values."""
    W = WINDOW_SIZE
    content = np.zeros((W, K3), dtype=np.float32)
    n = len(window_raw)
    pad_len = W - n

    for t in range(n):
        for c in range(K):
            v = window_raw[t][c]
            if z_sd[c] > 1e-6:
                content[t + pad_len, c] = (v - z_mu[c]) / z_sd[c]
            else:
                content[t + pad_len, c] = v - z_mu[c]
            content[t + pad_len, K + c] = window_mask[t][c]
            content[t + pad_len, 2*K + c] = window_delta[t][c]

    return content


@app.on_event("startup")
def startup():
    load_model()


class MIMCLoadRequest(BaseModel):
    n_stays: int = 50
    seed: int = 42


@app.post("/mimic/load")
def load_mimic(req: MIMCLoadRequest):
    """Load synthetic MIMIC-IV data into the backend.

    Merges MIMIC ICU stays with existing PhysioNet patients,
    making them available through /predict and /stream endpoints.
    """
    global patient_data, z_mu, z_sd

    summary = load_mimic_into_backend(globals(), n_stays=req.n_stays)

    return {
        "status": "loaded",
        "summary": summary,
        "total_patients": len(patient_data),
    }


@app.get("/mimic/summary")
def mimic_summary():
    """Get summary of loaded MIMIC data."""
    loader = get_mimic_loader()
    return loader.summary()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "patients": len(patient_data)}


@app.get("/patients")
def list_patients():
    """List available demo patients."""
    result = {}
    for pid, pdata in patient_data.items():
        result[pid] = {
            'n_hours': pdata.get('n_hours', 0),
            'n_positive': pdata.get('n_positive', 0),
            'labels': pdata.get('labels', []),
        }
    return result


class PredictRequest(BaseModel):
    patient_id: str
    hour: int  # current hour (0-based)
    governance: dict = {"cost_scale": 1.0, "risk_scale": 1.0, "bounty": 1.0}


class StreamRequest(BaseModel):
    patient_id: str
    governance: dict = {"cost_scale": 1.0, "risk_scale": 1.0, "bounty": 1.0}


@app.post("/predict")
def predict(req: PredictRequest):
    """Single-hour prediction with governance params."""
    if model is None:
        raise HTTPException(500, "Model not loaded")
    if req.patient_id not in patient_data:
        raise HTTPException(404, f"Patient {req.patient_id} not found")

    pdata = patient_data[req.patient_id]
    if req.hour >= pdata['n_hours']:
        raise HTTPException(400, f"Hour {req.hour} exceeds patient data ({pdata['n_hours']} hours)")

    # Build window: look back WINDOW_SIZE hours (or less if near start)
    start = max(0, req.hour - WINDOW_SIZE + 1)
    raw = pdata['ffilled'][start:req.hour + 1]
    mask = pdata['mask'][start:req.hour + 1]
    delta = pdata['delta'][start:req.hour + 1]

    tensor = zscore_window(raw, mask, delta)
    x = torch.from_numpy(tensor).unsqueeze(0)  # (1, W, K3)

    governance = req.governance

    with torch.no_grad():
        out = model(x, governance=governance, training=False)

    pred = float(out['predictions'][0])
    bids = out['bids'][0].numpy().tolist()
    pie = out['pie_weights'][0].numpy().tolist()  # (3, N_CELLS)
    selected = out['selected_idx'][0].numpy().tolist()
    label = pdata['labels'][req.hour]

    # Aggregate pie by jurisdiction
    jurisdictions = ["VITALS", "HYDRATION", "MOBILITY", "COGNITIVE", "OPERATIONAL"]
    jur_pies = {}
    cells_per_jur = N_CELLS // len(jurisdictions)
    for i, jur in enumerate(jurisdictions):
        start_c = i * cells_per_jur
        end_c = start_c + cells_per_jur
        jur_pie = np.array(pie)[:, start_c:end_c].mean(axis=1)
        jur_pies[jur] = {
            'cost': float(jur_pie[0]),
            'risk': float(jur_pie[1]),
            'neutrality': float(jur_pie[2]),
        }

    # Top bidding cells
    top_bids = sorted(range(len(bids)), key=lambda i: -bids[i])[:5]
    top_cell_info = []
    for ci in top_bids:
        jur_idx = ci // cells_per_jur
        jur = jurisdictions[min(jur_idx, len(jurisdictions)-1)]
        top_cell_info.append({
            'cell_id': ci,
            'jurisdiction': jur,
            'bid': bids[ci],
            'pie': {'cost': pie[0][ci], 'risk': pie[1][ci], 'neutrality': pie[2][ci]},
        })

    return {
        'patient_id': req.patient_id,
        'hour': req.hour,
        'prediction': round(pred, 4),
        'true_label': label,
        'governance': governance,
        'jurisdiction_pies': jur_pies,
        'top_cells': top_cell_info,
        'selected_indices': selected,
    }


@app.post("/stream")
def stream(req: StreamRequest):
    """Replay full patient trajectory with governance params."""
    if model is None:
        raise HTTPException(500, "Model not loaded")
    if req.patient_id not in patient_data:
        raise HTTPException(404, f"Patient {req.patient_id} not found")

    pdata = patient_data[req.patient_id]
    trajectory = []

    for hour in range(pdata['n_hours']):
        start = max(0, hour - WINDOW_SIZE + 1)
        raw = pdata['ffilled'][start:hour + 1]
        mask = pdata['mask'][start:hour + 1]
        delta = pdata['delta'][start:hour + 1]
        tensor = zscore_window(raw, mask, delta)
        x = torch.from_numpy(tensor).unsqueeze(0)

        with torch.no_grad():
            out = model(x, governance=req.governance, training=False)

        pred = float(out['predictions'][0])
        bids = out['bids'][0].numpy().tolist()
        top_bid = max(bids)
        pie = out['pie_weights'][0].numpy()
        mean_pie = pie.mean(axis=1)

        trajectory.append({
            'hour': hour,
            'prediction': round(pred, 4),
            'true_label': pdata['labels'][hour],
            'top_bid': round(top_bid, 4),
            'avg_cost': round(float(mean_pie[0]), 4),
            'avg_risk': round(float(mean_pie[1]), 4),
            'avg_neutral': round(float(mean_pie[2]), 4),
        })

    return {
        'patient_id': req.patient_id,
        'n_hours': len(trajectory),
        'governance': req.governance,
        'trajectory': trajectory,
    }


# ─── FHIR Bridge Endpoints ───────────────────────────────────────────────────

class FHIRBundleRequest(BaseModel):
    patient_id: str
    bundle: dict
    governance: dict = {"cost_scale": 1.0, "risk_scale": 1.0, "bounty": 1.0}


class FHIRPredictRequest(BaseModel):
    patient_id: str
    governance: dict = {"cost_scale": 1.0, "risk_scale": 1.0, "bounty": 1.0}


@app.post("/fhir/ingest")
def fhir_ingest(req: FHIRBundleRequest):
    """Ingest a FHIR Bundle of observations for a patient.

    Accepts standard FHIR Observation/Condition resources and updates
    the patient's temporal state. Returns ingestion statistics.
    """
    global fhir_patients

    if req.patient_id not in fhir_patients:
        fhir_patients[req.patient_id] = PatientState(patient_id=req.patient_id)

    ps = fhir_patients[req.patient_id]
    result = ingest_fhir_bundle(ps, req.bundle)

    return {
        "patient_id": req.patient_id,
        "ingestion": result,
        "current_hour": ps.hour,
    }


@app.post("/fhir/predict")
def fhir_predict(req: FHIRPredictRequest):
    """Run inference on current patient state from FHIR observations.

    Uses the patient's tracked feature values to build a tensor,
    runs the Grid model, and returns prediction + cell economy.
    """
    if model is None:
        raise HTTPException(500, "Model not loaded")
    if req.patient_id not in fhir_patients:
        raise HTTPException(404, f"Patient {req.patient_id} not found. Send observations first.")

    ps = fhir_patients[req.patient_id]

    # Build tensor from current state
    tensor = ps.get_window_tensor(z_mu, z_sd)  # (W, K*3)
    x = torch.from_numpy(tensor).unsqueeze(0)  # (1, W, K*3)

    with torch.no_grad():
        out = model(x, governance=req.governance, training=False)

    pred = float(out['predictions'][0])
    bids = out['bids'][0].numpy().tolist()
    pie = out['pie_weights'][0].numpy().tolist()
    selected = out['selected_idx'][0].numpy().tolist()

    # Jurisdiction aggregation
    jurisdictions = ["VITALS", "HYDRATION", "MOBILITY", "COGNITIVE", "OPERATIONAL"]
    jur_pies = {}
    cells_per_jur = N_CELLS // len(jurisdictions)
    for i, jur in enumerate(jurisdictions):
        start_c = i * cells_per_jur
        end_c = start_c + cells_per_jur
        jur_pie = np.array(pie)[:, start_c:end_c].mean(axis=1)
        jur_pies[jur] = {
            'cost': float(jur_pie[0]),
            'risk': float(jur_pie[1]),
            'neutrality': float(jur_pie[2]),
        }

    # Top bidding cells
    top_bids = sorted(range(len(bids)), key=lambda i: -bids[i])[:5]
    top_cell_info = []
    for ci in top_bids:
        jur_idx = ci // cells_per_jur
        jur = jurisdictions[min(jur_idx, len(jurisdictions)-1)]
        top_cell_info.append({
            'cell_id': ci,
            'jurisdiction': jur,
            'bid': bids[ci],
            'pie': {'cost': pie[0][ci], 'risk': pie[1][ci], 'neutrality': pie[2][ci]},
        })

    alert_triggered = pred > 0.5

    # Generate FHIR DiagnosticReport for audit ledger
    report = build_diagnostic_report(
        patient_id=req.patient_id,
        prediction=pred,
        governance=req.governance,
        jurisdiction_pies=jur_pies,
        top_cells=top_cell_info,
        alert_triggered=alert_triggered,
    )
    fhir_audit_log.append(report)

    return {
        "patient_id": req.patient_id,
        "hour": ps.hour,
        "prediction": round(pred, 4),
        "alert": alert_triggered,
        "governance": req.governance,
        "jurisdiction_pies": jur_pies,
        "top_cells": top_cell_info,
        "selected_indices": selected,
        "n_features_observed": sum(1 for fs in ps.features.values() if fs.last_value is not None),
    }


@app.get("/fhir/patient/{patient_id}/state")
def fhir_patient_state(patient_id: str):
    """Get current patient state from tracked FHIR observations."""
    if patient_id not in fhir_patients:
        raise HTTPException(404, f"Patient {patient_id} not found")

    ps = fhir_patients[patient_id]
    values, mask, delta = ps.get_current_vector()

    feature_summary = {}
    for name, fs in ps.features.items():
        if fs.last_value is not None:
            feature_summary[name] = {
                "value": fs.last_value,
                "observed": True,
                "hours_since_obs": round(fs.time_since_observation, 2),
            }
        else:
            feature_summary[name] = {
                "value": None,
                "observed": False,
                "n_missing": fs.n_missing,
            }

    return {
        "patient_id": patient_id,
        "hour": ps.hour,
        "n_features_observed": sum(1 for fs in ps.features.values() if fs.last_value is not None),
        "n_total_features": len(ps.features),
        "completeness": round(sum(mask) / len(mask) * 100, 1),
        "features": feature_summary,
    }


@app.get("/fhir/audit")
def fhir_audit(limit: int = 10):
    """Get recent FHIR DiagnosticReports from the audit ledger."""
    return {
        "total_reports": len(fhir_audit_log),
        "recent": fhir_audit_log[-limit:],
    }


@app.post("/fhir/patient/{patient_id}/reset")
def fhir_reset_patient(patient_id: str):
    """Reset patient state (for testing / new admission)."""
    global fhir_patients
    if patient_id in fhir_patients:
        del fhir_patients[patient_id]
    return {"patient_id": patient_id, "status": "reset"}
