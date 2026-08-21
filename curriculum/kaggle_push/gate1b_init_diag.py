"""Gate-1b init diagnostic: crowned 117-dim vs 120-dim extension on the
SAME math exam. Isolates whether the LanguageGrid copy corrupts math.

The crowned checkpoint (v2 lam10 fast) scored sine 0.976 / decay 0.975 /
lorenz 0.857 on the certified math exam. Gate 1b's base showed sine 0.35
/ decay -0.23 — if the 117-dim repro scores 0.976 here, the extension
copy is the bug; if it scores 0.35 too, the exam path is the bug.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

SEED = 42
K_MATH = 6
K_CLINICAL = 33
K_SUBJECTS = K_MATH + K_CLINICAL
D_IN_117 = 117
D_IN_LANG = 120
HIDDEN = 192
N_CELLS = 100
K_ACTIVE = 3
V_LANG = 4
W = 14


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
        value = x[:, :, 0::3][:, :, :self.k_subjects]
        m = x[:, :, 1::3][:, :, :self.k_subjects]
        state = h_last.contiguous()
        prev = torch.zeros(B, self.k_subjects, device=x.device)
        outs = []
        for t in range(Wn):
            ctx = torch.cat([x[:, t], pooled, prev], dim=1)
            state = self.decode_cell(ctx, state)
            y_est = torch.cat(
                [head(state) for head in self.heads], dim=1)
            y = m[:, t] * value[:, t] + (1.0 - m[:, t]) * y_est
            outs.append(y)
            prev = y
        out = torch.stack(outs, dim=1)
        if return_routing:
            return out, votes
        return out


class LanguageGrid(MathSchoolGrid):
    def __init__(self, d_in, hidden, n_cells, k, k_subjects, vocab):
        super().__init__(d_in, hidden, n_cells, k, k_subjects)
        self.decode_cell = nn.GRUCell(d_in + 64 + k_subjects + vocab,
                                      hidden)
        self.vocab_head = nn.Linear(hidden, vocab)
        self.vocab = vocab

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
        value = x[:, :, 0::3][:, :, :self.k_subjects]
        m = x[:, :, 1::3][:, :, :self.k_subjects]
        state = h_last.contiguous()
        prev = torch.zeros(B, self.k_subjects, device=x.device)
        prev_tok = torch.zeros(B, self.vocab, device=x.device)
        outs = []
        for t in range(Wn):
            ctx = torch.cat([x[:, t], pooled, prev, prev_tok], dim=1)
            state = self.decode_cell(ctx, state)
            y_est = torch.cat(
                [head(state) for head in self.heads], dim=1)
            y = m[:, t] * value[:, t] + (1.0 - m[:, t]) * y_est
            outs.append(y)
            prev = y
            lg = self.vocab_head(state)
            prev_tok = torch.softmax(lg, dim=1)
        out = torch.stack(outs, dim=1)
        if return_routing:
            return out, votes
        return out


def discover_input(name):
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if name in (root, os.path.basename(root)):
            return root
    raise FileNotFoundError(f"dataset {name} not found under {base}")


def main():
    torch.manual_seed(SEED)
    ckpt_dir = discover_input("crowned-ckpt-fast")
    crowned = torch.load(os.path.join(ckpt_dir, "math2clinic_fast.pt"),
                         map_location="cpu", weights_only=True)

    # --- build the certified math exam (18-dim windows) ---
    X18, YM, MM = _exam_windows(192, SEED + 2)

    # --- model A: 117-dim exact repro ---
    model117 = MathSchoolGrid(D_IN_117, HIDDEN, N_CELLS, K_ACTIVE,
                              K_SUBJECTS)
    model117.load_state_dict(crowned, strict=True)
    model117.eval()
    X117 = torch.tensor(_embed_math(X18), dtype=torch.float32)
    with torch.no_grad():
        p117 = model117(X117)
    r2_117 = _math_r2(p117, YM, MM)

    # --- model B: 120-dim LanguageGrid extension ---
    model120 = LanguageGrid(D_IN_LANG, HIDDEN, N_CELLS, K_ACTIVE,
                            K_SUBJECTS, V_LANG)
    DECODE_BLOCKS = [(0, 117, 0, 117), (120, 184, 117, 181),
                     (184, 223, 181, 220)]
    with torch.no_grad():
        for k, v in crowned.items():
            p = dict(model120.named_parameters())[k]
            if tuple(p.shape) == tuple(v.shape):
                p.copy_(v)
            elif k == "decode_cell.weight_ih":
                for tlo, thi, slo, shi in DECODE_BLOCKS:
                    p[:, tlo:thi].copy_(v[:, slo:shi])
            elif p.ndim == 2 and v.ndim == 2 and p.shape[0] == v.shape[0] \
                    and p.shape[1] > v.shape[1]:
                p[:, :v.shape[1]].copy_(v)
                p[:, v.shape[1]:].zero_()
            elif p.ndim == 1 and v.ndim == 1 and p.shape[0] == v.shape[0]:
                p.copy_(v)
            else:
                raise SystemExit(f"shape mismatch on {k}: {p.shape} vs "
                                 f"{v.shape}")
    model120.eval()
    X120 = torch.tensor(_pad120(_embed_math(X18)), dtype=torch.float32)
    with torch.no_grad():
        p120, _ = model120(X120)
    r2_120 = _math_r2(p120, YM, MM)

    print("117-dim repro : " + " ".join(f"{k} {v:.3f}"
                                        for k, v in r2_117.items()),
          flush=True)
    print("120-dim ext   : " + " ".join(f"{k} {v:.3f}"
                                        for k, v in r2_120.items()),
          flush=True)
    report = {"r2_117": r2_117, "r2_120": r2_120}
    with open("/kaggle/working/init_diag.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[done]", flush=True)


# ---------------------------- inline certified math exam machinery

LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_DT = 0.02
LORENZ_STEPS = 2000
T_STAY = 256
SLIDE = 2
CAP_WINDOWS = 24
DROP_LO, DROP_HI = 0.4, 0.8
DROP_RANGES = {"lorenz": (0.20, 0.50), "sine": (0.30, 0.70),
               "cosine": (0.30, 0.70)}
DELTA_CAP = 24.0
EXAM_KINDS = ["sine", "cosine", "decay", "step", "sigmoid", "lorenz"]


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


def _exam_windows(n_windows, seed):
    n_stays = max(1, n_windows // CAP_WINDOWS)
    X, Y, M = build_dataset(n_stays, seed)
    return X[:n_windows], Y[:n_windows], M[:n_windows]


def _embed_math(X18):
    B, Wn, _ = X18.shape
    grid = np.zeros((B, Wn, D_IN_117), dtype=np.float32)
    grid[:, :, 1::3] = 1.0
    grid[:, :, :K_MATH * 3] = X18
    return grid


def _pad120(x117):
    B, Wn, _ = x117.shape
    x = np.zeros((B, Wn, D_IN_LANG), dtype=np.float32)
    x[:, :, :117] = x117
    x[:, :, 118] = 1.0
    return x


def _math_r2(p, y, m):
    out = {}
    for i, k in enumerate(EXAM_KINDS):
        dm = 1.0 - m[:, :, i:i + 1]
        num = ((p[:, :, i:i + 1] - y[:, :, i:i + 1]) ** 2 * dm).sum()
        den = ((y[:, :, i:i + 1]
                - y[:, :, i:i + 1].mean(dim=(0, 1), keepdim=True)) ** 2
               * dm).sum()
        out[k] = float(1.0 - num / max(den, 1e-9))
    return out


if __name__ == "__main__":
    main()
