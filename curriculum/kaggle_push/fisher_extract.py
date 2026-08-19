"""Fisher diagonal extraction — Phase-1 math sub-grid (kernel).

Computes the diagonal Fisher information for the Phase-1 "Primary School"
checkpoint over the math columns that Phase-2 will protect elastically:

    L_EWC = (lam/2) * sum F[i,j] * (theta[i,j] - theta*[i,j])^2

F is the mean squared gradient of the CERTIFIED Phase-1 loss (fidelity +
obs + conservation, exactly the v9/v10 recipe) at the checkpoint optimum,
averaged over N_BATCHES random math windows. The empirical Fisher over
the training distribution is the standard EWC diagonal approximation.

Keys captured (Phase-1 geometry — 6 math subjects):
    gru.weight_ih_l0       (576, 18)  input projection = the math block
    decode_cell.weight_ih  (576, 88)  decoder ctx: [0:18] window,
                                      [18:82] pooled, [82:88] prev heads
    heads.0..5.weight/bias (1, 192)   per-subject readouts

Outputs /kaggle/working/fisher_math.npz (F in Phase-1 shapes) +
fisher_report.json. Phase-2 maps F through the same COLUMN_COPY_SPEC
(see curriculum/phase2_transfer.py map_fisher_to_phase2).

CPU kernel (no GPU needed — 1.5M-param grid, a few minutes).
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

# ---------------------------- constants (certified Phase-1 recipe)

K_SUBJECTS = 6
SUBJECT_KINDS = ["sine", "cosine", "decay", "step", "sigmoid", "lorenz"]
W = 14
T_STAY = 256
SLIDE = 2
CAP_WINDOWS = 24
DROP_LO, DROP_HI = 0.40, 0.80
DELTA_CAP = 24.0
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_DT = 0.02
LORENZ_STEPS = 2000
BOUNDS = {
    "sine": (-1.60, 1.60), "cosine": (-1.60, 1.60),
    "decay": (-0.05, 2.20), "step": (-1.20, 1.20),
    "sigmoid": (-1.20, 1.20), "lorenz": (-3.50, 3.50),
}
DROP_RANGES = {"lorenz": (0.20, 0.50), "sine": (0.30, 0.70),
               "cosine": (0.30, 0.70), "decay": (0.30, 0.70),
               "step": (0.30, 0.70), "sigmoid": (0.30, 0.70)}

BATCH = 128
N_BATCHES = 8            # 1024 windows — cheap, stable F
HIDDEN = 192
N_CELLS = 100
K_ACTIVE = 3
LAMBDA_CONS = 0.5
LAMBDA_OBS = 0.25
SEED = 42

FISHER_KEYS = {"gru.weight_ih_l0", "decode_cell.weight_ih"}
FISHER_KEYS = FISHER_KEYS | {
    f"heads.{i}.{s}" for i in range(K_SUBJECTS) for s in ("weight", "bias")
}

# ---------------------------- generator (byte-identical to Phase-1 kernel)

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
    t = np.arange(T_STAY, dtype=float)
    V = np.empty((T_STAY, K_SUBJECTS))
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
    level = np.ones(T_STAY) * lo
    for k in range(n_steps):
        t0 = int(stay_rng.uniform(0.2 * T_STAY, 0.9 * T_STAY))
        level[t0:] = hi if k % 2 == 0 else lo
    V[:, 3] = level
    t0 = stay_rng.uniform(0.3 * T_STAY, 0.7 * T_STAY)
    width = stay_rng.uniform(2.0, 12.0)
    V[:, 4] = lo + (hi - lo) / (1.0 + np.exp(-(t - t0) / width))
    xraw = lorenz_x(stay_rng)
    idx = np.linspace(0, len(xraw) - 1, T_STAY).astype(int)
    xs = xraw[idx]
    V[:, 5] = 2.0 * xs / max(np.abs(xs).max(), 1e-9)
    return V


def damage(stay_rng, V):
    T, K = V.shape
    value = np.zeros_like(V)
    mask = np.zeros_like(V)
    delta = np.zeros_like(V)
    for k, kind in enumerate(SUBJECT_KINDS):
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
        sr = Rng(int(rng.integers(0, 2**31)))
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


# ---------------------------- model (byte-identical to Phase-1 kernel)

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


# ---------------------------- Fisher accumulation

def phase1_loss(pred, yb, mb):
    msk = mb > 0.5
    dropped = ~msk
    fid = torch.square(pred[dropped] - yb[dropped]).mean()
    obs = torch.square(pred[msk] - yb[msk]).mean()
    bounds = torch.tensor([BOUNDS[k][0] for k in SUBJECT_KINDS],
                          dtype=torch.float32)
    bhi = torch.tensor([BOUNDS[k][1] for k in SUBJECT_KINDS],
                       dtype=torch.float32)
    cons = torch.relu(bounds - pred).mean() + torch.relu(pred - bhi).mean()
    decay_neg = torch.relu(-0.05 - pred[..., 2]).mean()
    return fid + LAMBDA_OBS * obs + LAMBDA_CONS * cons + LAMBDA_CONS * decay_neg


def main():
    t0 = time.time()
    torch.manual_seed(SEED)

    print("[1/4] data (certified Phase-1 generator, seed 42)...", flush=True)
    Xtr, Ytr, Mtr = build_dataset(64, SEED)
    n = Xtr.shape[0]
    print(f"  {n} windows", flush=True)

    print("[2/4] checkpoint...", flush=True)
    base = "/kaggle/input"
    ckpt_dir = None
    for root, dirs, files in os.walk(base):
        if "math_school_s42.pt" in files:
            ckpt_dir = root
            break
    if ckpt_dir is None:
        raise SystemExit("FATAL: math_school_s42.pt not found")
    model = MathSchoolGrid(K_SUBJECTS * 3, HIDDEN, N_CELLS, K_ACTIVE,
                           K_SUBJECTS)
    sd = torch.load(os.path.join(ckpt_dir, "math_school_s42.pt"),
                    map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    print("  strict load OK", flush=True)

    print(f"[3/4] Fisher accumulation ({N_BATCHES} batches x {BATCH})...",
          flush=True)
    F = {k: torch.zeros_like(p) for k, p in model.state_dict().items()
         if k in FISHER_KEYS}
    for bi in range(N_BATCHES):
        idx = torch.randperm(n)[:BATCH]
        xb = torch.tensor(Xtr[idx], dtype=torch.float32)
        yb = torch.tensor(Ytr[idx], dtype=torch.float32)
        mb = torch.tensor(Mtr[idx], dtype=torch.float32)
        pred = model(xb)
        loss = phase1_loss(pred, yb, mb)
        loss.backward()
        for k in F:
            g = model.get_parameter(k).grad
            if g is not None:
                F[k] += g.detach() ** 2
        model.zero_grad(set_to_none=True)
    for k in F:
        F[k] /= N_BATCHES
    print("  mean F: "
          + " ".join(f"{k} {F[k].mean().item():.3e}" for k in F), flush=True)
    for k, v in F.items():
        assert torch.isfinite(v).all(), f"non-finite F on {k}"
        assert v.sum() > 0.0, f"zero F on {k} — gradients did not flow"

    print("[4/4] save...", flush=True)
    np.savez("/kaggle/working/fisher_math.npz",
             **{k: v.numpy() for k, v in F.items()})
    report = {
        "keys": sorted(F),
        "shapes": {k: list(F[k].shape) for k in F},
        "mean_F": {k: float(F[k].mean()) for k in F},
        "std_F": {k: float(F[k].std()) for k in F},
        "max_F": {k: float(F[k].max()) for k in F},
        "n_batches": N_BATCHES,
        "batch": BATCH,
        "loss_recipe": "fid + 0.25*obs + 0.5*cons + 0.5*decay_neg (v9/v10)",
        "seconds": round(time.time() - t0, 1),
    }
    with open("/kaggle/working/fisher_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {report['seconds']}s — fisher_math.npz + "
          f"fisher_report.json", flush=True)


if __name__ == "__main__":
    main()
