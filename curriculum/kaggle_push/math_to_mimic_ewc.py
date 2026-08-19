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

# EWC elasticity arms: lambda sweep as sequential runs (plasticity economy)
LAM_SWEEP = [1.0, 10.0, 100.0]
FISHER_DATASET = "math-school-fisher-f"
FISHER_FILE = "fisher_math.npz"

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
# ---------------------------- math exam generators (exam machinery)

# CERTIFIED Phase-1 generator, byte-identical to math_school_train.py
# (lines 55-220, the frozen v9/v10 artifact). The v4/v5 exam used a
# re-implemented continuum with different constants (A=1 fixed, tau~
# U(20,90), raw lorenz, uniform drops) — distribution mismatch; the
# epoch-0 exam failed even with perfect weights (v5: 0.36-0.75 vs certs
# ~0.98). The exam MUST test the model on windows drawn exactly as the
# Phase-1 certificate did.

T_STAY = 256
SLIDE = 2
CAP_WINDOWS = 24
DROP_LO, DROP_HI = 0.4, 0.8
DROP_RANGES = {"lorenz": (0.20, 0.50), "sine": (0.30, 0.70),
               "cosine": (0.30, 0.70)}

LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_DT = 0.02
LORENZ_STEPS = 2000

BOUNDS = {
    "sine": (-1.75, 1.75), "cosine": (-1.75, 1.75),
    "decay": (-0.05, 2.20), "step": (-1.20, 1.20),
    "sigmoid": (-1.20, 1.20), "lorenz": (-4.00, 4.00),
}


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
    V = np.empty((T, K_MATH))
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


def exam_windows(n_windows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(B, W, 18) exam windows drawn by the CERTIFIED Phase-1 generator.

    Byte-identical to the Phase-1 certificate protocol: build_dataset
    (same continuum amplitudes/frequencies/taus, same per-kind drop
    ranges, same SLIDE/cap windowing, same RNG seeding) so the exam tests
    the transferred model on the EXACT distribution it was certified on.
    """
    n_stays = max(1, n_windows // CAP_WINDOWS)
    X, Y, M = build_dataset(n_stays, seed)
    return X[:n_windows], Y[:n_windows], M[:n_windows]


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
# DORMANT-INPUT ZEROING (v5 fix): randomly re-initialized clinical input
# columns multiply the dormant mask=1.0 constant -> fixed gate bias at
# exam time -> epoch-0 exam collapses. Zero-init makes dormant slots
# exactly silent; clinical columns still learn in clinical training.
ZERO_COLUMNS_SPEC = {
    "gru.weight_ih_l0": [(18, D_IN)],
    "decode_cell.weight_ih": [(18, D_IN), (D_IN + 64 + K_MATH,
                                           D_IN + 64 + K_SUBJECTS)],
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
    for k, blocks in ZERO_COLUMNS_SPEC.items():
        if k in sd_model:
            with torch.no_grad():
                for tlo, thi in blocks:
                    sd_model[k][:, tlo:thi] = 0.0
    model.load_state_dict(sd_model)
    print(f"[transfer] copied {len(copied)} keys: {sorted(copied)}")
    print(f"[transfer] column-copied {len(partial_copied)} keys: "
          f"{sorted(partial_copied)}")
    print(f"[transfer] re-init {len(reinit)} keys: {sorted(reinit)}")
    print(f"[transfer] dormant-input zeroed: {sorted(ZERO_COLUMNS_SPEC)}")
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


# ---------------------------- EWC plasticity economy

def load_fisher_normalized():
    """Load fisher_math.npz, map into Phase-2 shapes, normalize mean-1.

    The empirical Fisher from the Phase-1 optimum is tiny (1e-8..1e-5);
    mean-1 normalization over ALL math parameters together decouples the
    relative importance structure (F) from the absolute loss scale, so
    the lab lambda sweep transfers directly. Clinical columns stay F=0
    (free to learn biology).
    """
    ckpt_dir = discover_input(FISHER_DATASET)
    f_p1 = np.load(os.path.join(ckpt_dir, FISHER_FILE))
    f_p1 = {k: f_p1[k] for k in f_p1.files}

    def col_blocks(blocks, tgt_shape):
        out = np.zeros(tgt_shape, dtype=np.float64)
        for tlo, thi, slo, shi in blocks:
            out[:, tlo:thi] = f_p1["decode_cell.weight_ih"][:, slo:shi]
        return out

    f2 = {
        "gru.weight_ih_l0": np.pad(
            f_p1["gru.weight_ih_l0"], ((0, 0), (0, D_IN - 18))),
        "decode_cell.weight_ih": col_blocks(
            COLUMN_COPY_SPEC["decode_cell.weight_ih"],
            (3 * HIDDEN, D_IN + 64 + K_SUBJECTS)),
    }
    for i in range(K_MATH):
        f2[f"heads.{i}.weight"] = f_p1[f"heads.{i}.weight"]
        f2[f"heads.{i}.bias"] = f_p1[f"heads.{i}.bias"]
    total = sum(float(v.sum()) for v in f2.values())
    n = sum(v.size for v in f2.values())
    mean = total / n
    assert np.isfinite(mean) and mean > 0, f"bad Fisher mean {mean}"
    f2 = {k: torch.tensor(v / mean, dtype=torch.float32)
          for k, v in f2.items()}
    print(f"[ewc] fisher loaded {len(f2)} keys, global mean normalized to 1 "
          f"(raw mean {mean:.2e})", flush=True)
    return f2


def ewc_penalty(model, f2, lam):
    """L_EWC = (lam/2) * sum F * (theta - theta*)^2 over the math sub-grid.

    F is zero-padded on clinical columns, so only the math columns accrue
    cost — clinical learning is uncharged. theta* is the post-transfer
    snapshot (detached copy of the Phase-1 math weights).
    """
    total = torch.zeros((), dtype=torch.float32)
    for k, F in f2.items():
        p = dict(model.named_parameters())[k]
        th_star = getattr(_THETA_STAR, k)
        total = total + (F * (p - th_star) ** 2).sum()
    return 0.5 * lam * total


class _ThetaStar:
    pass


def snapshot_theta_star(model, f2):
    """Freeze the transferred math weights as theta* (detached copies)."""
    st = _ThetaStar()
    for k in f2:
        setattr(st, k, dict(model.named_parameters())[k].detach().clone())
    return st


_THETA_STAR = _ThetaStar()


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
    """(Xb, Y, M) exam triple: embedded (B, W, 117) grid (all 6 math kinds
    active via embed_math_block, clinical dormant), TRUE values Y (B, W, 6)
    and observation masks M (B, W, 6) from the CERTIFIED generator.

    The target MUST be Y, not the grid's value channel: build_dataset's
    value column is FFILLED (carried forward), and at dropped slots the
    ffill baseline is a stale copy of the last observation — grading
    against it penalized fast channels (sine/cosine/lorenz) while
    flattering slow ones (decay/step/sigmoid). Phase-1's masked_r2
    grades against Yte (true values).
    """
    X18, Y, M = exam_windows(192, SEED + 2)
    X = embed_math_block(X18)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
            torch.tensor(M, dtype=torch.float32))


def exam_loss(model, xb, yb, mb):
    """Masked MSE over math heads 0..5, graded on TRUE values Y at
    dropped slots only (Phase-1 certificate protocol). Clinical heads
    are pinned to 0 by the dormant protocol (no gradient, no surgery).
    """
    l = model(xb)   # v8: gradients MUST flow (dead-anchor bug, see v7)
    pred_k = l[:, :, :K_MATH]                       # (B, W, 6)
    dm = 1.0 - mb                                    # dropped slots
    return ((pred_k - yb) ** 2 * dm).sum() / max(dm.sum(), 1)


def exam_r2(model, xb, yb, mb):
    with torch.no_grad():
        l = model(xb)
    r2 = {}
    for ki, kind in enumerate(EXAM_KINDS):
        p = l[:, :, ki:ki + 1]
        t = yb[:, :, ki:ki + 1]
        dm = 1.0 - mb[:, :, ki:ki + 1]
        num = ((p - t) ** 2 * dm).sum()
        den = ((t - t.mean(dim=(0, 1), keepdim=True)) ** 2 * dm).sum()
        r2[kind] = float(1.0 - num / max(den, 1e-9))
    return r2


def masked_r2_nd(pred, target, drop_mask):
    """Masked R2 with the GLOBAL mean denominator (Phase-1 semantics).

    The per-window de-mean (dim=1) is REFUTED: near-constant lab channels
    inside a 14-step window collapse the denominator and explode the
    pooled R2 (-60.7 in the v3 run while per-feature vitals/labs were
    0.94/0.91) — a metric artifact, not a model signal.
    """
    num = ((pred - target) ** 2 * drop_mask).sum()
    den = ((target - target.mean(dim=(0, 1), keepdim=True)) ** 2 * drop_mask).sum()
    return float(1.0 - num / max(den, 1e-9))


# ---------------------------- main

def run_arm(lam, Xtr, Ytr, Mtr, Xte, Yte, Mte, f2, t0):
    """One lambda arm: transfer init + train + eval (independent model)."""
    tag = f"lam{lam:g}"
    print(f"\n===== ARM {tag} ===== (0 = v4 control)", flush=True)
    torch.manual_seed(SEED)

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
    if lam > 0:
        global _THETA_STAR
        _THETA_STAR = snapshot_theta_star(model, f2)
    opt = torch.optim.Adam(param_groups(model))
    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for g in opt.param_groups for p in g["params"]
               if g["lr"] == LR_TRANSFER)
    print(f"  {n_par:,} params | {n_tr:,} priors @ {LR_TRANSFER} "
          f"| {n_par - n_tr:,} new @ {LR_NEW} | lam {lam:g}", flush=True)

    Xex, Yex, Mex = exam_inputs()
    r2_base = exam_r2(model, Xex, Yex, Mex)
    print("  baseline " + " ".join(f"{k} {r2_base[k]:.3f}" for k in EXAM_KINDS),
          flush=True)

    n = Xtr.shape[0]
    n_batches = (n + BATCH - 1) // BATCH
    ewc_terms = []
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
                loss = loss + LAMBDA_MATH * exam_loss(model, Xex, Yex, Mex)
            if LAMBDA_COST > 0:
                _, votes = model(xb, return_routing=True)
                loss = loss + LAMBDA_COST * votes.mean() / N_CELLS
            if lam > 0:
                ewc = ewc_penalty(model, f2, lam)
                loss = loss + ewc
                ewc_terms.append(float(ewc))
            if not torch.isfinite(loss):
                print(f"  WARN ep {ep} non-finite loss — skipping step")
                continue
            loss.backward()
            opt.step()
            tot += float(loss)
        note = ""
        if (ep + 1) % EXAM_EVERY == 0 or ep == N_EPOCHS - 1:
            r2 = exam_r2(model, Xex, Yex, Mex)
            worst = min(v for v in r2.values() if v == v)
            note = " EXAM " + " ".join(f"{k} {v:.3f}" for k, v in r2.items()) \
                + f" [worst {worst:.3f}]"
        print(f"  ep {ep:3d} loss {tot / n_batches:9.3f}{note}", flush=True)

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
    r2_final = exam_r2(model, Xex, Yex, Mex)
    ok = r2_all >= 0.90 and all(v >= EXAM_FLOOR for v in r2_final.values())
    print(f"  clinical masked R2 {r2_all:.4f} | vitals {np.mean(r2_v):.3f} "
          f"labs {np.mean(r2_l):.3f}", flush=True)
    print("  exam " + " ".join(f"{k} {v:.3f}" for k, v in r2_final.items()),
          flush=True)
    print("  VERDICT:", "PASS" if ok else "FAIL", flush=True)

    torch.save(model.state_dict(), f"/kaggle/working/math2clinic_{tag}.pt")
    report = {
        "arm": tag,
        "lambda": lam,
        "clinical_masked_r2": r2_all,
        "vitals_masked_r2": float(np.mean(r2_v)),
        "labs_masked_r2": float(np.mean(r2_l)),
        "exam_r2_final": r2_final,
        "exam_r2_base": r2_base,
        "ewc_term_mean": float(np.mean(ewc_terms)) if ewc_terms else 0.0,
        "transfer_copied": copied,
        "transfer_column_copied": partial_copied,
        "transfer_reinit": skipped,
        "verdict": "PASS" if ok else "FAIL",
        "seconds": round(time.time() - t0, 1),
    }
    with open(f"/kaggle/working/phase2_report_{tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[arm {tag}] {report['seconds']}s | verdict {report['verdict']}",
          flush=True)
    return report


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

    print("[2/5] fisher...", flush=True)
    f2 = load_fisher_normalized()
    mean_ewc = {k: float(f2[k].mean()) for k in f2}
    print("  normalized mean F per key: "
          + " ".join(f"{k} {v:.3f}" for k, v in mean_ewc.items()), flush=True)

    reports = []
    for lam in LAM_SWEEP:
        reports.append(run_arm(lam, Xtr, Ytr, Mtr, Xte, Yte, Mte, f2, t0))

    summary = {r["arm"]: {"clinical_r2": r["clinical_masked_r2"],
                          "vitals": r["vitals_masked_r2"],
                          "labs": r["labs_masked_r2"],
                          "worst_exam": min(r["exam_r2_final"].values()),
                          "verdict": r["verdict"]} for r in reports}
    print("\n=== FLEET SUMMARY ===", flush=True)
    for arm, s in summary.items():
        print(f"  {arm}: clinical {s['clinical_r2']:.4f} | vitals {s['vitals']:.3f} "
              f"| labs {s['labs']:.3f} | worst exam {s['worst_exam']:.3f} "
              f"| {s['verdict']}", flush=True)
    with open("/kaggle/working/ewc_fleet.json", "w") as f:
        json.dump({"summary": summary, "reports": reports}, f, indent=2)
    print(f"[done] {round(time.time() - t0, 1)}s total", flush=True)


if __name__ == "__main__":
    main()