"""Phase 1 "Primary School" — Math Curriculum Graduation Exam (kernel).

The Institutional Grid goes to school: it must learn to reconstruct
continuous mathematical functions (sine, cosine, exponential decay, step,
sigmoid, Lorenz chaos) from EHR-style damaged streams — 40-80% of points
dropped per channel, forward-carried values, mask + time-delta channels.

Grading rubric (v5 — structural physics, not parameter tuning):
  Fidelity    = masked imputation R2 on DROPPED positions (primary exam)
                graded on a banded scale (subject-class-specific gates)
  Conservation= hinge penalties on physically illegal predictions, weighted
                LAMBDA_CONS = 0.5; exponential decay is STRUCTURALLY clamped
                to [-0.05, 2.20] in forward() = negatives impossible by
                construction (the soft hinge stays as a tripwire)
  Efficiency  = routing footprint: how many of the 100 cells solve each
                subject, plus top-3 sparsity (hard-capped by architecture)

Banded graduation gates (user rubric):
  smooth deterministic (sine/cosine/sigmoid)  R2 >= 0.97
  physical law (decay)                        R2 >= 0.97 AND <= DECAY_NEG_TOL
                                               negative predictions
  discontinuous (step)                        R2 >= 0.95
  chaotic (lorenz)                            R2 >= 0.70

v5 structural rewrites (user-mandated, no more lambda whack-a-mole):
  (1) MASK-CONDITIONED decoder: per-position GRUCell sees the full context
      vector x[t] = [value_ffill, mask, delta] + pooled grid latent + the
      previous blended prediction; at OBSERVED slots the output is the TRUE
      value by construction (hard copy y = m*value + (1-m)*y_est) — the
      model anchors on local observed points instead of decoding all 14
      positions blind.
  (2) HARD decay clamp: y_est[:, 2] clamped to [-0.05, 2.20] — the PINN
      gold standard (soft hinge was drowned by MSE on the ~1.5% tail).
  (3) ENV TWEAKS: Lorenz sampled denser (kept 0.5-0.8 vs 0.2-0.6) AND the
      sine/cosine harmonics sampled at 0.30-0.70 — extreme 80% masking left
      2-3 scattered points in a 14-step window so phase/frequency were
      under-determined; these are data-availability fixes, not model ones.

Model: GRU encoder (hidden 192) -> shared 100-cell grid (top-3) ->
      mask-conditioned autoregressive GRUCell decoder with per-subject
      heads.

CPU-only kernel (unsloth/bnb not used; grid is a small GRU+MoE).
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

# ------------------------------------------------ constants

K_SUBJECTS = 6
SUBJECT_KINDS = ["sine", "cosine", "decay", "step", "sigmoid", "lorenz"]
W = 14
T_STAY = 256
SLIDE = 2
CAP_WINDOWS = 24
DROP_LO, DROP_HI = 0.4, 0.8
DROP_RANGES = {"lorenz": (0.20, 0.50), "sine": (0.30, 0.70),
               "cosine": (0.30, 0.70)}   # denser: chaos + harmonics need
                                         # observed state to stay anchored
DELTA_CAP = 24.0
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_DT = 0.02
LORENZ_STEPS = 2000

BOUNDS = {
    "sine": (-1.75, 1.75), "cosine": (-1.75, 1.75),
    "decay": (-0.05, 2.20), "step": (-1.20, 1.20),
    "sigmoid": (-1.20, 1.20), "lorenz": (-4.00, 4.00),
}
R2_GATES = {"sine": 0.97, "cosine": 0.97, "decay": 0.97,
            "step": 0.95, "sigmoid": 0.97, "lorenz": 0.70}
# Hard physical constraint: exponential decay may not go negative on test
# (decay lower bound is -0.05). Allow at most DECAY_NEG_TOL violations
# (per test prediction count) before the exam is failed on that subject.
DECAY_NEG_TOL = 5

SEED = 42
N_TRAIN_STAYS = 800
N_TEST_STAYS = 200
N_EPOCHS = 100
LR = 1e-3
BATCH = 512
HIDDEN = 192
N_CELLS = 100
K_ACTIVE = 3
LAMBDA_CONS = 0.5    # conservation hinge weight (v4: physics-enforcing)
LAMBDA_OBS = 0.25    # observed-slot fidelity weight (imputation is primary)

# ------------------------------------------------ generator (numpy)

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
    V = np.empty((T, K_SUBJECTS))
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


def masked_r2(pred, target, mask):
    m = mask > 0.5
    if not m.any():
        return 0.0
    y, f = target[~m], pred[~m]
    denom = np.sum((y - y.mean()) ** 2)
    if denom <= 1e-12:
        return 1.0 if np.max(np.abs(f - y)) < 1e-12 else 0.0
    return float(1.0 - np.sum((f - y) ** 2) / denom)


# ------------------------------------------------ model

class MathSchoolGrid(nn.Module):
    """GRU encoder + shared 100-cell grid + MASK-CONDITIONED decoder (v5).

    The decoder is a per-position GRUCell that sees the FULL context vector
    x[t] = [value_ffill, mask, delta] at every step, plus the pooled routing
    latent and the blended prediction of the previous step. At OBSERVED slots
    the output is the TRUE value by construction (hard copy:
    y = mask*value + (1-mask)*y_est) — the model can anchor on local observed
    points instead of decoding all 14 positions blind. The exponential-decay
    channel (index 2) is STRUCTURALLY clamped to [-0.05, 2.20] so negative
    predictions are impossible (PINN-style hard constraint, no soft hinge).
    """

    def __init__(self, d_in, hidden, n_cells, k, k_subjects):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.scorer = nn.Linear(hidden, n_cells)
        self.cell_block = nn.Linear(hidden, n_cells * 64)
        # mask-conditioned decoder: ctx = x[t] (18) + pooled (64) + prev (K)
        self.decode_cell = nn.GRUCell(d_in + 64 + k_subjects, hidden)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(k_subjects)])
        self.k = k
        self.n_cells = n_cells
        self.k_subjects = k_subjects

    def forward(self, x, return_routing=False):
        B, W, D = x.shape
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
        value = x[:, :, 0::3]                             # (B, W, K) ffill'd
        m = x[:, :, 1::3]                                 # (B, W, K) 1=obs
        state = h_last.contiguous()                       # decode init state
        prev = torch.zeros(B, self.k_subjects, device=x.device)
        outs = []
        for t in range(W):
            ctx = torch.cat([x[:, t], pooled, prev], dim=1)   # (B, 88)
            state = self.decode_cell(ctx, state)               # (B, hidden)
            y_est = torch.cat(
                [head(state) for head in self.heads], dim=1)   # (B, K)
            # structural clamp, FUNCTIONAL ONLY: both clone() and plain
            # in-place splicing bump the recorded view's version counter
            # and autograd raises "modified by an inplace operation".
            # Splice the clamped column through cat() instead — no tensor
            # in the graph is ever written to.
            y_est = torch.cat(
                [y_est[:, :2], y_est[:, 2:3].clamp(min=-0.05, max=2.20),
                 y_est[:, 3:]], dim=1)
            y = m[:, t] * value[:, t] + (1.0 - m[:, t]) * y_est
            outs.append(y)
            prev = y
        out = torch.stack(outs, dim=1)                    # (B, W, K)
        if return_routing:
            return out, votes
        return out


# ------------------------------------------------ train

def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[1/4] Generating school cohort...", flush=True)
    Xtr, Ytr, Mtr = build_dataset(N_TRAIN_STAYS, seed=SEED)
    Xte, Yte, Mte = build_dataset(N_TEST_STAYS, seed=SEED + 1)
    print(f"  train {Xtr.shape[0]} windows, test {Xte.shape[0]} windows", flush=True)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    Mtr_t = torch.tensor(Mtr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    Yte_t = torch.tensor(Yte, dtype=torch.float32)
    Mte_t = torch.tensor(Mte, dtype=torch.float32)

    n = Xtr_t.shape[0]

    print("[2/4] Training...", flush=True)
    model = MathSchoolGrid(K_SUBJECTS * 3, HIDDEN, N_CELLS, K_ACTIVE,
                           K_SUBJECTS)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} params | hidden {HIDDEN} | cells {N_CELLS} "
          f"top-{K_ACTIVE}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_batches = (n + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_EPOCHS * n_batches)

    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot_fid = tot_cons = tot_obs = 0.0
        n_step = 0
        for i in range(n_batches):
            bidx = perm[i * BATCH:(i + 1) * BATCH]
            xb = Xtr_t[bidx]
            yb = Ytr_t[bidx]
            mb = Mtr_t[bidx]
            pred = model(xb)
            msk = mb > 0.5           # OBSERVED slots (exposed to the model)
            dropped = ~msk           # DROPPED slots = the imputation exam
            # Fidelity v5: the decoder copies TRUE values at observed slots
            # (hard anchor) so `obs` is ~0 by construction; the exam is the
            # DROPPED positions where y = y_est is the grid's own extrapolation.
            fid = torch.square(pred[dropped] - yb[dropped]).mean()
            obs = torch.square(pred[msk] - yb[msk]).mean()
            # conservation: hinge on bounds per subject
            bounds = torch.tensor([BOUNDS[k][0] for k in SUBJECT_KINDS],
                                  dtype=torch.float32)
            bhi = torch.tensor([BOUNDS[k][1] for k in SUBJECT_KINDS],
                               dtype=torch.float32)
            cons = torch.relu(bounds - pred).mean() + torch.relu(pred - bhi).mean()
            # hard physical law: exponential decay (channel 2) is STRUCTURALLY
            # clamped to [-0.05, 2.20] in forward() -> this term is 0 by
            # construction; kept as a tripwire in case the clamp is moved.
            decay_neg = torch.relu(-0.05 - pred[..., 2]).mean()
            loss = (fid + LAMBDA_OBS * obs
                    + LAMBDA_CONS * cons + LAMBDA_CONS * decay_neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot_fid += fid.item()
            tot_obs += obs.item()
            tot_cons += cons.item()
            n_step += 1
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"  ep {epoch + 1}/{N_EPOCHS} fid {tot_fid / n_step:.4f} "
                  f"obs {tot_obs / n_step:.4f} cons {tot_cons / n_step:.4f}",
                  flush=True)

    print("[3/4] Grading exam...", flush=True)
    model.eval()
    with torch.no_grad():
        pred_te = model(Xte_t).numpy()
    r2 = {}
    for k, kind in enumerate(SUBJECT_KINDS):
        r2[kind] = masked_r2(pred_te[..., k].ravel(), Yte[..., k].ravel(),
                             Mte[..., k].ravel())
    # conservation violation counts on test predictions
    viol = {}
    decay_neg = 0
    for k, kind in enumerate(SUBJECT_KINDS):
        lo_, hi_ = BOUNDS[kind]
        v = pred_te[..., k].ravel()
        n_below = int((v < lo_ - 1e-9).sum())
        n_above = int((v > hi_ + 1e-9).sum())
        viol[kind] = {"below": n_below, "above": n_above}
        if kind == "decay":
            decay_neg = n_below
    # routing footprint: which cells fire per subject (test windows)
    with torch.no_grad():
        _, votes = model(Xte_t[:4000] if len(Xte_t) > 4000 else Xte_t,
                         return_routing=True)
    votes = votes.numpy()
    active_all = set(np.flatnonzero(votes.sum(0) > 0).tolist())
    footprint = {"cells_used": len(active_all), "n_cells": N_CELLS,
                 "k_active": K_ACTIVE}

    # Banded rubric: R2 gate per subject class + hard decay non-negativity
    r2_pass = {k: r2[k] >= R2_GATES[k] for k in SUBJECT_KINDS}
    decay_physical = decay_neg <= DECAY_NEG_TOL
    passed = {k: r2_pass[k] for k in SUBJECT_KINDS}
    passed["decay"] = r2_pass["decay"] and decay_physical
    graded = all(passed.values())
    report = {
        "phase": "1_math_school",
        "seed": SEED,
        "train_windows": int(n),
        "test_windows": int(Xte_t.shape[0]),
        "n_epochs": N_EPOCHS,
        "hidden": HIDDEN,
        "lambda_cons": LAMBDA_CONS,
        "drop_ranges": {k: DROP_RANGES.get(k, (DROP_LO, DROP_HI))
                        for k in SUBJECT_KINDS},
        "r2": r2,
        "r2_gates": R2_GATES,
        "decay_neg_predictions": decay_neg,
        "decay_neg_tol": DECAY_NEG_TOL,
        "passed": passed,
        "graded": graded,
        "conservation_violations": viol,
        "routing_footprint": footprint,
        "params": params,
        "train_seconds": round(time.time() - t0, 1),
    }
    # certified checkpoint: the Phase-2 transfer init source (v10+)
    torch.save(model.state_dict(), "/kaggle/working/math_school_s42.pt")
    report["checkpoint"] = "math_school_s42.pt"
    with open("/kaggle/working/phase1_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("/kaggle/working/phase1_summary.txt", "w") as f:
        f.write(f"PHASE 1 MATH SCHOOL — seed {SEED} | hidden {HIDDEN} "
                f"| cons {LAMBDA_CONS}\n")
        f.write(f"train {n} windows | test {Xte_t.shape[0]} | "
                f"{params:,} params | {N_EPOCHS} epochs\n")
        for k in SUBJECT_KINDS:
            mark = "PASS" if passed[k] else "FAIL"
            extra = ""
            if k == "decay":
                extra = f" | neg {decay_neg} (tol {DECAY_NEG_TOL})"
            f.write(f"  {k:8s} masked R2 {r2[k]:.4f} (gate {R2_GATES[k]:.2f}) "
                    f"{mark}{extra}\n")
        f.write(f"cells used: {footprint['cells_used']}/{N_CELLS}"
                f"(top-{K_ACTIVE} hard cap)\n")
        f.write(f"GRADUATED: {graded}\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"\n[train_seconds] {round(time.time() - t0, 1)}", flush=True)


if __name__ == "__main__":
    main()