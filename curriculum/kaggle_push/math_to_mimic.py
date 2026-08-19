"""Phase 2 — Math-to-MIMIC transfer (stage-1 scaffold).

Physics-priors warm start into the clinical stem, then low-LR clinical
learning with a math no-forgetting exam.

  Stage 1 (this kernel):
    init = certified Phase-1 checkpoint (math_school_s42.pt, dataset
    albanchigozirim/math-school-phase-1-checkpoint) — copy only
    shape-compatible priors (recurrent hh dynamics, routing manifold:
    scorer + cell_block, decode recurrence); re-init input projections
    and the 39 per-feature heads.
    loss  = risk-weighted fidelity on dropped clinical slots
            + LAMBDA_MATH * masked MSE on the math exam windows
            + LAMBDA_COST * n_active/100 (bandwidth; default 0.0)
    two Adam groups: transfer lr=1e-5, new lr=1e-4.
    exam  = masked R2 per math kind EVERY EXAM_EVERY epochs — never
            grades the gate, only the readout; verdict needs every kind
            >= 0.80 (Phase-1 certificates: sine/cos 0.985, decay 0.988,
            step 0.965, sigmoid 0.995, lorenz 0.895).

  Data: synthetic MIMIC-contract windows (see vendored mimic_contract
  block BELOW — byte-identical to curriculum/mimic_contract.py via
  curriculum/sync_vendored.py; real MIMIC-IV stays slot in unchanged).
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

# ---------------------------- constants

SEED = 42
W = 14
HIDDEN = 192
N_CELLS = 100
K_ACTIVE = 3
N_STAYS = 2048
N_TEST_STAYS = 256
N_EPOCHS = 40
BATCH = 512
LR_TRANSFER = 1e-5
LR_NEW = 1e-4
LAMBDA_MATH = 0.01
LAMBDA_COST = 0.0
EXAM_EVERY = 10
EXAM_FLOOR = 0.80
RISK_DROP_WEIGHT = 3.0

EXAM_KINDS = ["sine", "cosine", "decay", "step", "sigmoid", "lorenz"]

# ---------------------------- model (vendored Phase-1 architecture)

class MathSchoolGrid(nn.Module):
    """Certified Phase-1 grid; clamp_decay_channel=None for clinical geometry."""

    def __init__(self, d_in, hidden, n_cells, k, k_subjects,
                 clamp_decay_channel=None):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.scorer = nn.Linear(hidden, n_cells)
        self.cell_block = nn.Linear(hidden, n_cells * 64)
        self.decode_cell = nn.GRUCell(d_in + 64 + k_subjects, hidden)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(k_subjects)])
        self.k = k
        self.n_cells = n_cells
        self.k_subjects = k_subjects
        self.clamp_decay_channel = clamp_decay_channel

    def forward(self, x, return_routing=False):
        B, Wn, D = x.shape
        h, _ = self.gru(x)
        h_last = h[:, -1]
        scores = self.scorer(h_last)
        topk = torch.topk(scores, self.k, dim=1)
        votes = torch.zeros(B, self.n_cells, device=x.device)
        votes.scatter_(1, topk.indices, 1.0)
        cells = torch.relu(self.cell_block(h_last))
        cells = cells.view(B, self.n_cells, 64)
        selected = torch.gather(cells, 1,
                                topk.indices.unsqueeze(-1).expand(-1, -1, 64))
        pooled = selected.mean(dim=1)                     # (B, 64)
        value = x[:, :, 0::3]
        m = x[:, :, 1::3]
        state = h_last.contiguous()
        prev = torch.zeros(B, self.k_subjects, device=x.device)
        outs = []
        for t in range(Wn):
            ctx = torch.cat([x[:, t], pooled, prev], dim=1)
            state = self.decode_cell(ctx, state)
            y_est = torch.cat([head(state) for head in self.heads], dim=1)
            if self.clamp_decay_channel is not None:
                c = self.clamp_decay_channel
                y_est = torch.cat(
                    [y_est[:, :c], y_est[:, c:c + 1].clamp(min=-0.05, max=2.20),
                     y_est[:, c + 1:]], dim=1)
            y = m[:, t] * value[:, t] + (1.0 - m[:, t]) * y_est
            outs.append(y)
            prev = y
        out = torch.stack(outs, dim=1)
        if return_routing:
            return out, votes
        return out

# ---------------------------- vendored data contracts (see sync_vendored)

# >>> VENDOR (mimic_contract) — do not edit outside the reference module

W = 14
DELTA_CAP = 24.0

K_MATH = 6
K_CLINICAL = 33
K_SUBJECTS = K_MATH + K_CLINICAL            # 39 heads
D_IN = K_SUBJECTS * 3                       # 117 input channels
HEAD_OFFSET_CLINICAL = K_MATH               # clinical heads start at 6
CLINICAL_SLOT = HEAD_OFFSET_CLINICAL * 3    # clinical triplets start at 18

# The 6 most-missing labs on real MIMIC train (>= 98.4% missing) — cut so
# the math sub-grid gets heads 0..5 without growing the model.
DROP_6_LABS = {
    "AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total",
    "TroponinI", "Fibrinogen",
}

FEATURE_NAMES_ALL = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime",
]
FEATURE_NAMES = [n for n in FEATURE_NAMES_ALL if n not in DROP_6_LABS]
assert len(FEATURE_NAMES) == K_CLINICAL

VITALS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
          "FiO2", "pH", "SaO2", "Age", "Gender", "Unit1", "Unit2"}
DEMOGRAPHICS_FROM = FEATURE_NAMES.index("Age")  # 28 — always observed


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def clinical_stay(g: np.random.Generator) -> np.ndarray:
    """One 168-step stay of the 33 clinical features (vitals rhythmic,
    labs slow). Values are z-scored on the full cohort (train stats) in
    the real pipeline; here they are synthesized already standardized.
    """
    T = 168
    V = np.zeros((T, K_CLINICAL))
    t = np.arange(T)
    for i, name in enumerate(FEATURE_NAMES):
        if name in VITALS:
            base = g.normal(0, 1)
            amp = g.uniform(0.15, 0.5)
            freq = g.uniform(0.02, 0.10)
            trend = g.normal(0, 0.6) * np.linspace(0, 1, T)
            V[:, i] = base + amp * np.sin(freq * t + g.uniform(0, 6.28)) \
                + trend
        else:
            level = g.normal(0, 1)
            drift = g.uniform(0.005, 0.02)
            V[:, i] = level + g.normal(0, 0.15) * np.sin(drift * t
                                                         + g.uniform(0, 6.28))
            V[:, i] = np.clip(V[:, i], -4, 4)
    return V


def clinical_windows(n_stays: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, M): X (B, W, 117) sub-grid triplets, Y (B, W, 39) true
    values, M (B, W, 39) observation flags (1 = observed).

    Math channels 0:18 are DORMANT (value 0, mask 1, delta 0) so the
    clinical loss grades heads 6-38 only by construction.
    """
    g = _rng(seed)
    X, Y, M = [], [], []
    for stay in range(n_stays):
        V = clinical_stay(_rng(int(g.integers(0, 2 ** 31))))
        T = V.shape[0]
        n_win = min(24, max(1, (T - W) // 2))
        for _ in range(n_win):
            start = int(g.integers(0, T - W))
            win = V[start: start + W]
            drop = g.uniform(0.15, 0.75)
            mask = (g.random((W, K_CLINICAL)) > drop).astype(np.float32)
            mask[:, DEMOGRAPHICS_FROM:] = 1.0
            ff = win.copy()
            for k in range(K_CLINICAL):
                last = None
                for tt in range(W):
                    if mask[tt, k] > 0.5:
                        last = win[tt, k]
                    elif last is None:
                        last = win[0, k]
                    else:
                        ff[tt, k] = last
            delta = np.zeros_like(win)
            for k in range(K_CLINICAL):
                last_obs = -1
                for tt in range(W):
                    if mask[tt, k] > 0.5:
                        last_obs = tt
                    elif last_obs >= 0:
                        delta[tt, k] = min(tt - last_obs, DELTA_CAP)
                    else:
                        delta[tt, k] = DELTA_CAP
            # sub-grid assembly: math slots dormant, clinical at 18:117
            x = np.zeros((W, D_IN), dtype=np.float32)
            x[:, 0::3] = 0.0            # math value
            x[:, 1::3] = 1.0            # math mask (dormant = observed)
            x[:, 2::3] = 0.0            # math delta
            x[:, CLINICAL_SLOT + 0::3] = ff
            x[:, CLINICAL_SLOT + 1::3] = mask
            x[:, CLINICAL_SLOT + 2::3] = delta
            y = np.zeros((W, K_SUBJECTS), dtype=np.float32)
            y[:, HEAD_OFFSET_CLINICAL:] = win
            m = np.zeros((W, K_SUBJECTS), dtype=np.float32)
            m[:, HEAD_OFFSET_CLINICAL:] = mask
            m[:, :HEAD_OFFSET_CLINICAL] = 1.0
            X.append(x)
            Y.append(y)
            M.append(m)
    return np.stack(X), np.stack(Y), np.stack(M)


def embed_math_exam(X3: np.ndarray, kinds: np.ndarray,
                    total_dim: int = D_IN) -> np.ndarray:
    """(B, W, 3) [value, mask, delta] math windows -> (B, W, 117) grid.

    Dormant-slot protocol: every triplet starts dormant (value=0, mask=1,
    delta=0); each row's OWN kind triplet (kinds[b] in 0..5) is then
    overwritten with that window's [value, mask, delta] — the other 5 math
    kinds and all 33 clinical triplets stay hard-masked, so the router
    allocates them zero bandwidth and the hard copy pins their heads to 0.
    kinds[b] selects the triplet (value, mask, delta) at columns
    (3*kind, 3*kind+1, 3*kind+2).
    """
    B, Wn, _ = X3.shape
    kinds = np.asarray(kinds, dtype=np.int64)
    if kinds.ndim != 1 or len(kinds) != B:
        raise ValueError(f"kinds must be 1-D length {B}, got {kinds.shape}")
    if kinds.min() < 0 or kinds.max() >= K_MATH:
        raise ValueError(f"kinds out of math range 0..{K_MATH - 1}: "
                         f"{kinds.min()}..{kinds.max()}")
    grid = np.zeros((B, Wn, total_dim), dtype=np.float32)
    grid[:, :, 1::3] = 1.0                      # dormant: mask=1, value=0, delta=0
    rows = np.arange(B)
    k3 = kinds * 3
    grid[rows, :, k3] = X3[:, :, 0]             # value
    grid[rows, :, k3 + 1] = X3[:, :, 1]         # mask
    grid[rows, :, k3 + 2] = X3[:, :, 2]         # delta
    return grid


# <<< VENDOR (mimic_contract) — do not edit outside the reference module
# ---------------------------- math exam generators (exam machinery)

LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_DT = 0.02
LORENZ_STEPS = 2000


def lorenz_x(g: np.random.Generator, n: int = LORENZ_STEPS,
             dt: float = LORENZ_DT) -> np.ndarray:
    s, r, b = LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA
    x, y, z = 1.0 + g.normal(0, 0.05, 3)
    xs = np.empty(n)
    for i in range(n):
        xs[i] = x

        def f(xv, yv, zv):
            return (s * (yv - xv), xv * (r - zv) - yv, xv * yv - b * zv)

        k1 = f(x, y, z)
        k2 = f(x + dt * k1[0] / 2, y + dt * k1[1] / 2, z + dt * k1[2] / 2)
        k3 = f(x + dt * k2[0] / 2, y + dt * k2[1] / 2, z + dt * k2[2] / 2)
        k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        x += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        y += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        z += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
    return xs


def continuum(g: np.random.Generator) -> np.ndarray:
    """One 256-step stay of the 6 math channels (sine..lorenz)."""
    V = np.zeros((256, 6))
    t = np.arange(256)
    freqs = g.uniform(0.02, 0.08, 2)
    phases = g.uniform(0, 2 * np.pi, 2)
    V[:, 0] = np.sin(freqs[0] * t + phases[0])
    V[:, 1] = np.cos(freqs[1] * t + phases[1])
    V[:, 2] = 2.0 * np.exp(-t / g.uniform(20, 90)) + g.uniform(0, 0.02)
    for _ in range(int(g.integers(1, 4))):
        a = int(g.integers(1, 251))
        V[a:, 3] = g.uniform(-0.9, 0.9)
    mid = int(g.integers(1, 251))
    V[:, 4] = g.uniform(-1.0, 1.0) / (1.0 + np.exp(-(t - mid) / 12.0))
    idx = int(g.integers(0, LORENZ_STEPS - 256))
    V[:, 5] = lorenz_x(g)[idx: idx + 256]
    return V


def exam_windows(n_windows: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """(B, W, 3) [value, mask, delta] windows, one KIND per row.

    Rows cycle through the 6 kinds deterministically so every kind gets
    ~n_windows/6 windows. Mask column = per-position observation flag
    drawn in (DROP_LO, DROP_HI) independently; delta = 0 (single channel,
    no feature structure to track). embed_math_exam ports these into the
    (B, W, 117) grid at the row's own kind triplet.
    """
    g = _rng(seed)
    X = np.zeros((n_windows, W, 3), dtype=np.float32)
    kinds = []
    for i in range(n_windows):
        kind = i % 6
        kinds.append(kind)
        V = continuum(_rng(int(g.integers(0, 2 ** 31))))
        start = int(g.integers(0, 256 - W))
        X[i, :, 0] = V[start: start + W, kind]
        X[i, :, 1] = g.uniform(0.30, 0.70, W) > 0.5
    return X, kinds


# ---------------------------- checkpoint discovery

def discover_input(name):
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if name in (root, os.path.basename(root)):
            return root
    raise FileNotFoundError(f"dataset {name} not found under {base}")

# full-tensor copies: recurrent/routing priors + PERMANENT math heads 0..5
TRANSFER_KEYS = {
    "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "decode_cell.weight_hh", "decode_cell.bias_ih", "decode_cell.bias_hh",
    "scorer.weight", "scorer.bias",
    "cell_block.weight", "cell_block.bias",
}
TRANSFER_KEYS = TRANSFER_KEYS | {
    f"heads.{i}.{s}" for i in range(K_MATH) for s in ("weight", "bias")
}

# column-block partial copies: (tgt_lo, tgt_hi, src_lo, src_hi) — the math
# window channels + pooled/prev blocks keep their Phase-1 columns so the
# transferred machinery stays in-distribution on the math exam.
COLUMN_COPY_SPEC = {
    "gru.weight_ih_l0": [(0, 18, 0, 18)],
    "decode_cell.weight_ih": [
        (0, 18, 0, 18),
        (117, 181, 18, 82),
        (181, 187, 82, 88),
    ],
}


def load_transfer(ckpt_path, model):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd_model = model.state_dict()
    copied, partial_copied, reinit = [], [], []
    for k, v in sd.items():
        if k in COLUMN_COPY_SPEC and k in sd_model:
            with torch.no_grad():
                for tlo, thi, slo, shi in COLUMN_COPY_SPEC[k]:
                    sd_model[k][:, tlo:thi] = v[:, slo:shi]
            partial_copied.append(k)
        elif k in TRANSFER_KEYS and k in sd_model and \
                tuple(sd_model[k].shape) == tuple(v.shape):
            with torch.no_grad():
                sd_model[k].copy_(v)
            copied.append(k)
        else:
            reinit.append(k)
    model.load_state_dict(sd_model)
    print(f"[transfer] copied {len(copied)} keys: {sorted(copied)}")
    print(f"[transfer] column-copied {len(partial_copied)} keys: "
          f"{sorted(partial_copied)}")
    print(f"[transfer] re-init {len(reinit)} keys: {sorted(reinit)}")
    return copied, partial_copied, reinit


def param_groups(model):
    """Two Adams: transfer priors (incl. column-copied projections) at
    1e-5, everything new at 1e-4."""
    slow = TRANSFER_KEYS | set(COLUMN_COPY_SPEC)
    transfer, fresh = [], []
    for k, p in model.named_parameters():
        (transfer if k in slow else fresh).append(p)
    return [
        {"params": transfer, "lr": LR_TRANSFER},
        {"params": fresh, "lr": LR_NEW},
    ]


# ---------------------------- risk-driven fidelity (pie Risk slice)

def risk_weights(values, mask, drop_weight=RISK_DROP_WEIGHT):
    var = []
    for k in range(values.shape[-1]):
        v = values[:, :, k]
        m = mask[:, :, k] > 0.5
        var.append(float(v[m].var()) if m.sum() > 2 else 1.0)
    var = np.asarray(var, dtype=np.float32)
    med = max(float(np.median(var)), 1e-6)
    return torch.tensor(1.0 + np.clip(var / med - 1.0, 0.0, drop_weight),
                        dtype=torch.float32)


def fidelity_loss(pred, target, drop_mask, values, mask):
    risk = risk_weights(values, mask).to(pred.device)
    sq = (pred - target) ** 2 * drop_mask
    return (sq * risk.view(1, 1, -1)).sum() / max(drop_mask.sum(), 1)


# ---------------------------- math exam helpers

def exam_inputs():
    """(B, W, 117) exam grid via the dormant-slot protocol + kinds."""
    X3, kinds = exam_windows(192, SEED + 2)
    X = embed_math_exam(X3, np.asarray(kinds))
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(kinds, dtype=torch.long))


def exam_loss(model, xb, kinds):
    """Masked MSE pairing each window with ITS OWN kind head (gather).

    xb is the (B, W, 117) grid; window b's active triplet lives at
    columns (3*kinds[b] .. 3*kinds[b]+2) = [value, mask, delta]. The
    model output has 39 HEAD columns (math heads 0..5 = kind index), so
    head and triplet indices differ.
    """
    with torch.no_grad():
        l = model(xb)
    head_idx = kinds.view(-1, 1, 1).expand(xb.shape[0], W, 1)
    k3 = (kinds * 3).view(-1, 1, 1)
    pred_k = torch.gather(l, 2, head_idx)
    target = torch.gather(xb, 2, k3)
    dm = 1.0 - torch.gather(xb, 2, k3 + 1)
    return ((pred_k - target) ** 2 * dm).sum() / max(dm.sum(), 1)


def exam_r2(model, xb, kinds):
    with torch.no_grad():
        l = model(xb)
    r2 = {}
    for ki, kind in enumerate(EXAM_KINDS):
        sel = (kinds == ki)
        if sel.sum() == 0:
            r2[kind] = float("nan")
            continue
        # every row in sel has kind ki -> its active triplet is at 3*ki
        p = l[sel][:, :, ki:ki + 1]
        t = xb[sel][:, :, 3 * ki:3 * ki + 1]
        dm = (1.0 - xb[sel][:, :, 3 * ki + 1:3 * ki + 2])
        num = ((p - t) ** 2 * dm).sum()
        den = ((t - t.mean(dim=(0, 1), keepdim=True)) ** 2 * dm).sum()
        r2[kind] = float(1.0 - num / max(den, 1e-9))
    return r2


def masked_r2_nd(pred, target, drop_mask):
    num = ((pred - target) ** 2 * drop_mask).sum()
    den = ((target - target.mean(dim=1, keepdim=True)) ** 2 * drop_mask).sum()
    return float(1.0 - num / max(den, 1e-9))


# ---------------------------- main

def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[1/5] clinical data...", flush=True)
    Xtr, Ytr, Mtr = clinical_windows(N_STAYS, SEED)
    Xte, Yte, Mte = clinical_windows(N_TEST_STAYS, SEED + 1)
    Xtr = torch.tensor(Xtr); Ytr = torch.tensor(Ytr); Mtr = torch.tensor(Mtr)
    Xte = torch.tensor(Xte); Yte = torch.tensor(Yte); Mte = torch.tensor(Mte)
    print(f"  train {Xtr.shape[0]} windows, test {Xte.shape[0]}", flush=True)

    print("[2/5] transfer init...", flush=True)
    model = MathSchoolGrid(D_IN, HIDDEN, N_CELLS, K_ACTIVE, K_SUBJECTS)
    ckpt_dir = discover_input("math-school-phase-1-checkpoint")
    copied, partial_copied, skipped = load_transfer(
        os.path.join(ckpt_dir, "math_school_s42.pt"), model)
    if len(copied) != len(TRANSFER_KEYS) or \
            len(partial_copied) != len(COLUMN_COPY_SPEC):
        raise SystemExit(
            f"FATAL: expected {len(TRANSFER_KEYS)} full + "
            f"{len(COLUMN_COPY_SPEC)} column transfers, got "
            f"{len(copied)} + {len(partial_copied)}")
    opt = torch.optim.Adam(param_groups(model))
    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for g in opt.param_groups for p in g["params"]
               if g["lr"] == LR_TRANSFER)
    print(f"  {n_par:,} params | {n_tr:,} priors @ {LR_TRANSFER} "
          f"| {n_par - n_tr:,} new @ {LR_NEW}", flush=True)

    print("[3/5] math exam baseline...", flush=True)
    Xex, kinds_ex = exam_inputs()
    r2_base = exam_r2(model, Xex, kinds_ex)
    print("  " + " ".join(f"{k} {r2_base[k]:.3f}" for k in EXAM_KINDS), flush=True)

    print("[4/5] training...", flush=True)
    n = Xtr.shape[0]
    n_batches = (n + BATCH - 1) // BATCH
    for ep in range(N_EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(n_batches):
            idx = perm[i * BATCH: (i + 1) * BATCH]
            xb, yb, mb = Xtr[idx], Ytr[idx], Mtr[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = fidelity_loss(pred, yb, 1.0 - mb, yb, mb)
            if LAMBDA_MATH > 0:
                loss = loss + LAMBDA_MATH * exam_loss(model, Xex, kinds_ex)
            if LAMBDA_COST > 0:
                _, votes = model(xb, return_routing=True)
                loss = loss + LAMBDA_COST * votes.mean() / N_CELLS
            if not torch.isfinite(loss):
                print(f"  WARN ep {ep} non-finite loss — skipping step")
                continue
            loss.backward()
            opt.step()
            tot += float(loss)
        note = ""
        if (ep + 1) % EXAM_EVERY == 0 or ep == N_EPOCHS - 1:
            r2 = exam_r2(model, Xex, kinds_ex)
            worst = min(v for v in r2.values() if v == v)
            note = " EXAM " + " ".join(f"{k} {v:.3f}" for k, v in r2.items()) \
                + f" [worst {worst:.3f}]"
        print(f"  ep {ep:3d} loss {tot / n_batches:9.3f}{note}", flush=True)

    print("[5/5] eval...", flush=True)
    with torch.no_grad():
        pred = model(Xte)
    r2_all = masked_r2_nd(pred, Yte, 1.0 - Mte)
    r2_v, r2_l = [], []
    for i in range(K_CLINICAL):
        j = i + HEAD_OFFSET_CLINICAL        # clinical head index
        dm = 1.0 - Mte[:, :, j]
        num = ((pred[:, :, j] - Yte[:, :, j]) ** 2 * dm).sum()
        den = ((Yte[:, :, j] - Yte[:, :, j].mean()) ** 2 * dm).sum()
        r2v = float(1.0 - num / max(den, 1e-9))
        (r2_v if FEATURE_NAMES[i] in VITALS else r2_l).append(r2v)
    r2_final = exam_r2(model, Xex, kinds_ex)
    ok = r2_all >= 0.90 and all(v >= EXAM_FLOOR for v in r2_final.values())
    print(f"  clinical masked R2 {r2_all:.4f} | vitals {np.mean(r2_v):.3f} "
          f"labs {np.mean(r2_l):.3f}", flush=True)
    print("  exam " + " ".join(f"{k} {v:.3f}" for k, v in r2_final.items()), flush=True)
    print("  VERDICT:", "PASS" if ok else "FAIL", flush=True)

    torch.save(model.state_dict(), "/kaggle/working/math2clinic.pt")
    report = {
        "clinical_masked_r2": r2_all,
        "vitals_masked_r2": float(np.mean(r2_v)),
        "labs_masked_r2": float(np.mean(r2_l)),
        "exam_r2_final": r2_final,
        "exam_r2_base": r2_base,
        "transfer_copied": copied,
        "transfer_column_copied": partial_copied,
        "transfer_reinit": skipped,
        "verdict": "PASS" if ok else "FAIL",
        "seconds": round(time.time() - t0, 1),
    }
    with open("/kaggle/working/phase2_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {report['seconds']}s | verdict {report['verdict']}", flush=True)


if __name__ == "__main__":
    main()