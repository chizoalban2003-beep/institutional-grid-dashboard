"""Fisher v2 — FULL-SURFACE diagonal Fisher on the Phase-2 model (kernel).

The v1 Fisher (fisher_math.npz) was computed in Phase-1 geometry and
zero-padded on clinical columns. The mechanism autopsy REFUTED that
design: math function migrates into the clinical columns (iso made the
exam WORSE), so "the knowledge lives everywhere" — the Cost slice of
the pie chart must be measured everywhere.

v2 protocol:
  - build the PHASE-2 model (117-dim) exactly as the transfer does
    (column copies + ZERO_COLUMNS_SPEC dormant zeroing)
  - feed CERTIFIED math exam windows through embed_math_block
    (all 6 kinds active, clinical dormant)
  - loss = masked MSE on math heads vs TRUE values Y (Phase-1 protocol)
  - accumulate squared gradients on ALL named parameters over
    N_BATCHES batches — no key filtering, no zeroing. Clinical input
    columns receive gradient through the dormant mask=1 constant
    (their learned weights affect the exam), so F captures the true
    exam-loading of every synapse.

Outputs /kaggle/working/fisher_v2.npz + fisher_v2_report.json.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

SEED = 42
BATCH = 128
N_BATCHES = 16          # 2048 exam windows — stable full-surface F

# ---------------------------- constants (certified Phase-1 recipe)

K_MATH_SUBJECTS = 6       # math-generator subjects (NOT the vendored K_SUBJECTS=39)
EXAM_KINDS = ["sine", "cosine", "decay", "step", "sigmoid", "lorenz"]
W = 14
T_STAY = 256
SLIDE = 2
CAP_WINDOWS = 24
DROP_LO, DROP_HI = 0.4, 0.8
DROP_RANGES = {"lorenz": (0.20, 0.50), "sine": (0.30, 0.70),
               "cosine": (0.30, 0.70)}
DELTA_CAP = 24.0
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_DT = 0.02
LORENZ_STEPS = 2000

K_MATH = 6
K_CLINICAL = 33
K_SUBJECTS_P2 = K_MATH + K_CLINICAL            # 39 heads
D_IN = K_SUBJECTS_P2 * 3                       # 117
CLINICAL_SLOT = K_MATH * 3                     # 18
HIDDEN = 192
N_CELLS = 100
K_ACTIVE = 3

TRANSFER_KEYS = {
    "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "decode_cell.weight_hh", "decode_cell.bias_ih", "decode_cell.bias_hh",
    "scorer.weight", "scorer.bias",
    "cell_block.weight", "cell_block.bias",
}
TRANSFER_KEYS = TRANSFER_KEYS | {
    f"heads.{i}.{s}" for i in range(K_MATH) for s in ("weight", "bias")
}
COLUMN_COPY_SPEC = {
    "gru.weight_ih_l0": [(0, 18, 0, 18)],
    "decode_cell.weight_ih": [
        (0, 18, 0, 18),
        (117, 181, 18, 82),
        (181, 187, 82, 88),
    ],
}
ZERO_COLUMNS_SPEC = {
    "gru.weight_ih_l0": [(18, D_IN)],
    "decode_cell.weight_ih": [(18, D_IN), (D_IN + 64 + K_MATH,
                                           D_IN + 64 + K_SUBJECTS_P2)],
}

# ---------------------------- certified generator (Phase-1 byte-identical)

class Rng:
    def __init__(self, seed):
        self.g = np.random.default_rng(seed)

    def uniform(self, lo, hi, size=None):
        return self.g.uniform(lo, hi, size)

    def normal(self, loc, scale, size=None):
        return self.g.normal(loc, scale, size)

    def random(self, size=None):
        return self.g.random(size)

    def integers(self, lo, hi):
        return int(self.g.integers(lo, hi))


def lorenz_x(stay_rng, n=LORENZ_STEPS, dt=LORENZ_DT):
    s, r, b = LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA
    x, y, z = 1.0 + stay_rng.normal(0, 0.05, 3)
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


def continuum(stay_rng):
    T = T_STAY
    t = np.arange(T, dtype=float)
    V = np.empty((T, K_MATH_SUBJECTS))
    A = stay_rng.uniform(0.5, 1.5)
    f1 = stay_rng.uniform(0.02, 0.12)
    phi = stay_rng.uniform(0, 2 * np.pi)
    V[:, 0] = A * np.sin(2 * np.pi * f1 * t + phi)
    V[:, 1] = A * np.cos(2 * np.pi * f1 * t + phi)
    A2 = stay_rng.uniform(1.0, 2.0)
    tau = stay_rng.uniform(8.0, 45.0)
    V[:, 2] = A2 * np.exp(-t / tau)
    lo = stay_rng.uniform(-1.0, -0.2)
    hi = stay_rng.uniform(0.2, 1.0)
    n_steps = stay_rng.integers(1, 3)
    level = np.ones(T) * lo
    for k in range(n_steps):
        t0 = int(stay_rng.uniform(0.2 * T, 0.9 * T))
        level[t0:] = hi if k % 2 == 0 else lo
    V[:, 3] = level
    t0 = stay_rng.uniform(0.3 * T, 0.7 * T)
    width = stay_rng.uniform(2.0, 12.0)
    V[:, 4] = lo + (hi - lo) / (1.0 + np.exp(-(t - t0) / width))
    xraw = lorenz_x(stay_rng)
    idx = np.linspace(0, len(xraw) - 1, T).astype(int)
    xs = xraw[idx]
    V[:, 5] = 2.0 * xs / max(np.abs(xs).max(), 1e-9)
    return V


def damage(stay_rng, V):
    T, K = V.shape
    value = np.zeros_like(V)
    mask = np.zeros_like(V)
    delta = np.zeros_like(V)
    for k, kind in enumerate(EXAM_KINDS):
        lo_, hi_ = DROP_RANGES.get(kind, (DROP_LO, DROP_HI))
        p = stay_rng.uniform(lo_, hi_)
        obs = stay_rng.random(T) >= p
        last = -1
        for i in range(T):
            if obs[i]:
                value[i, k] = V[i, k]
                mask[i, k] = 1.0
                delta[i, k] = 0.0
                last = i
            else:
                delta[i, k] = min(i - last, DELTA_CAP) if last >= 0 else DELTA_CAP
                value[i, k] = value[last, k] if last >= 0 else 0.0
    return value, mask, delta


def build_dataset(n_stays, seed):
    rng = Rng(seed)
    all_x, all_y, all_m = [], [], []
    for i in range(n_stays):
        sr = Rng(int(rng.integers(0, 2 ** 31)))
        V = continuum(sr)
        value, mask, delta = damage(sr, V)
        T, K = V.shape
        n_win = (T - W) // SLIDE + 1
        idx = np.arange(n_win)
        if n_win > CAP_WINDOWS:
            g2 = np.random.default_rng(1000 + i)
            idx = np.sort(g2.choice(idx, size=CAP_WINDOWS, replace=False))
        for s in idx:
            a, b = s * SLIDE, s * SLIDE + W
            x = np.empty((W, K * 3))
            x[:, 0::3] = value[a:b]
            x[:, 1::3] = mask[a:b]
            x[:, 2::3] = delta[a:b]
            all_x.append(x)
            all_y.append(V[a:b])
            all_m.append(mask[a:b])
    return (np.stack(all_x), np.stack(all_y), np.stack(all_m))


def exam_windows(n_windows, seed):
    n_stays = max(1, n_windows // CAP_WINDOWS)
    X, Y, M = build_dataset(n_stays, seed)
    return X[:n_windows], Y[:n_windows], M[:n_windows]


# ---------------------------- Phase-2 grid + transfer (exact v8 init)

class MathSchoolGrid(nn.Module):
    def __init__(self, d_in, hidden, n_cells, k, k_subjects):
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
        pooled = selected.mean(dim=1)
        value = x[:, :, 0::3]
        m = x[:, :, 1::3]
        state = h_last.contiguous()
        prev = torch.zeros(B, self.k_subjects, device=x.device)
        outs = []
        for t in range(Wn):
            ctx = torch.cat([x[:, t], pooled, prev], dim=1)
            state = self.decode_cell(ctx, state)
            y_est = torch.cat(
                [head(state) for head in self.heads], dim=1)
            y_est = torch.cat(
                [y_est[:, :2], y_est[:, 2:3].clamp(min=-0.05, max=2.20),
                 y_est[:, 3:]], dim=1)
            y = m[:, t] * value[:, t] + (1.0 - m[:, t]) * y_est
            outs.append(y)
            prev = y
        out = torch.stack(outs, dim=1)
        if return_routing:
            return out, votes
        return out


def embed_math_block(X18, total_dim=D_IN):
    B, Wn, _ = X18.shape
    grid = np.zeros((B, Wn, total_dim), dtype=np.float32)
    grid[:, :, 1::3] = 1.0
    grid[:, :, :K_MATH * 3] = X18
    return grid


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
    for k, blocks in ZERO_COLUMNS_SPEC.items():
        if k in sd_model:
            with torch.no_grad():
                for tlo, thi in blocks:
                    sd_model[k][:, tlo:thi] = 0.0
    model.load_state_dict(sd_model)
    return copied, partial_copied, reinit


def discover_input(name):
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if name in (root, os.path.basename(root)):
            return root
    raise FileNotFoundError(f"dataset {name} not found under {base}")


def clinical_loss(pred, yb, mb):
    """Masked MSE over clinical heads 6..38 (the Phase-2 fidelity)."""
    dm = 1.0 - mb[:, :, K_MATH:]
    return ((pred[:, :, K_MATH:] - yb[:, :, K_MATH:]) ** 2 * dm).sum() \
        / max(dm.sum(), 1)


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


def embed_math_block(X18: np.ndarray,
                     total_dim: int = D_IN) -> np.ndarray:
    """(B, W, 18) Phase-1 full math windows -> (B, W, 117) grid.

    Phase-1 parity exam protocol: the exam grades ALL 6 kinds per window,
    exactly like the Phase-1 certificate (masked R2 per kind on full
    multi-kind test windows). X18 carries all kinds' [value, mask, delta]
    triplets (columns 0..17); they land in the math sub-grid 0:18 and the
    33 clinical triplets stay dormant (value=0, mask=1, delta=0) so the
    hard copy pins clinical heads to 0 (zero bandwidth, zero gradient).
    """
    B, Wn, _ = X18.shape
    if X18.shape[2] != K_MATH * 3:
        raise ValueError(f"math windows must be (B, W, {K_MATH * 3}), "
                         f"got {X18.shape}")
    grid = np.zeros((B, Wn, total_dim), dtype=np.float32)
    grid[:, :, 1::3] = 1.0                      # dormant: mask=1, value=0, delta=0
    grid[:, :, :K_MATH * 3] = X18               # all kinds active (Phase-1 parity)
    return grid


# <<< VENDOR (mimic_contract) — do not edit outside the reference module

N_STAYS = 2048      # kernel-side (not part of the vendored contract)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[1/4] CROWNED checkpoint (v2 lam10 fast)...", flush=True)
    model = MathSchoolGrid(D_IN, HIDDEN, N_CELLS, K_ACTIVE, K_SUBJECTS_P2)
    ckpt_dir = discover_input("crowned-ckpt-fast")
    sd = torch.load(os.path.join(ckpt_dir, "math2clinic_fast.pt"),
                    map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model.eval()
    print("  crowned ckpt strict load OK", flush=True)

    print("[2/4] windows: math exam + clinical...", flush=True)
    X18, YM, MM = exam_windows(N_BATCHES * BATCH, SEED + 2)
    XC, YC, MC = clinical_windows(N_STAYS, SEED + 3)
    print(f"  math {X18.shape[0]} windows | clinical {XC.shape[0]} windows",
          flush=True)

    print("[3/4] F_total = F_math + F_clinical accumulation...", flush=True)
    F = {k: torch.zeros_like(p) for k, p in model.named_parameters()}

    def accumulate(xb, yb, mb, loss_fn, tag):
        for k in F:
            pass
        for bi in range(N_BATCHES):
            sl = slice(bi * BATCH, (bi + 1) * BATCH)
            x = xb[sl] if xb.shape[0] >= (bi + 1) * BATCH else xb[bi * BATCH:]
            y = yb[sl] if yb.shape[0] >= (bi + 1) * BATCH else yb[bi * BATCH:]
            m = mb[sl] if mb.shape[0] >= (bi + 1) * BATCH else mb[bi * BATCH:]
            pred = model(x)
            loss = loss_fn(pred, y, m)
            loss.backward()
            for k, p in model.named_parameters():
                if p.grad is not None:
                    F[k] += p.grad.detach() ** 2
            model.zero_grad(set_to_none=True)
        for k in F:
            F[k] /= N_BATCHES
        print(f"  [{tag}] accumulated over {N_BATCHES} batches", flush=True)

    # surface 1: math (certified exam windows, clinical dormant)
    G_m = torch.tensor(embed_math_block(X18), dtype=torch.float32)
    accumulate(G_m, torch.tensor(YM), torch.tensor(MM),
               lambda p, y, m: ((p[:, :, :K_MATH] - y) ** 2
                                * (1.0 - m)).sum() / max((1.0 - m).sum(), 1),
               "math")

    # surface 2: clinical (mimic_contract windows, math dormant)
    F_math = {k: v.clone() for k, v in F.items()}
    accumulate(torch.tensor(XC), torch.tensor(YC), torch.tensor(MC),
               clinical_loss, "clinical")
    F_clin = {k: v.clone() for k, v in F.items()}
    F = {k: F_math[k] + F_clin[k] for k in F}

    for k, v in F.items():
        assert torch.isfinite(v).all(), f"non-finite F on {k}"
    # Structural zeros are HONEST (scorer hard-topk, clinical heads via
    # hard-copy m=1); keys that MUST carry loading:
    must_carry = ["gru.weight_ih_l0", "gru.weight_hh_l0",
                  "decode_cell.weight_ih", "decode_cell.weight_hh",
                  "cell_block.weight", "cell_block.bias",
                  "heads.0.weight", "heads.5.weight",
                  "heads.6.weight", "heads.38.weight"]
    for k in must_carry:
        assert F[k].sum() > 0.0, f"zero F on {k} (expected gradient path)"
    zero_keys = [k for k, v in F.items() if v.sum() == 0.0]
    print(f"  structural-zero F keys (allowed): {sorted(zero_keys)}",
          flush=True)

    print("[4/4] save...", flush=True)
    np.savez("/kaggle/working/fisher_v3.npz",
             **{k: v.numpy() for k, v in F.items()})
    report = {
        "keys": sorted(F),
        "shapes": {k: list(F[k].shape) for k in F},
        "mean_F": {k: float(F[k].mean()) for k in F},
        "max_F": {k: float(F[k].max()) for k in F},
        "clinical_cols_covered": {
            "gru.weight_ih_l0_clinical": float(
                F["gru.weight_ih_l0"][:, 18:].mean()),
            "decode_cell.weight_ih_clinical": float(
                F["decode_cell.weight_ih"][:, 18:117].mean()),
            "decode_cell.weight_ih_prev_clinical": float(
                F["decode_cell.weight_ih"][:, 187:].mean()),
        },
        "recurrent_F": {
            "gru.weight_hh_l0": float(F["gru.weight_hh_l0"].mean()),
            "decode_cell.weight_hh": float(F["decode_cell.weight_hh"].mean()),
            "scorer.weight": float(F["scorer.weight"].mean()),
            "cell_block.weight": float(F["cell_block.weight"].mean()),
        },
        "f_math_mean": {k: float(F_math[k].mean()) for k in F_math},
        "f_clinical_mean": {k: float(F_clin[k].mean()) for k in F_clin},
        "n_batches": N_BATCHES,
        "batch": BATCH,
        "seconds": round(time.time() - t0, 1),
    }
    with open("/kaggle/working/fisher_v3_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  clinical-col mean F: "
          + " ".join(f"{k} {v:.3e}" for k, v
                     in report["clinical_cols_covered"].items()), flush=True)
    print("  recurrent mean F: "
          + " ".join(f"{k} {v:.3e}" for k, v
                     in report["recurrent_F"].items()), flush=True)
    print(f"[done] {report['seconds']}s — fisher_v3.npz + "
          f"fisher_v3_report.json", flush=True)


if __name__ == "__main__":
    main()
