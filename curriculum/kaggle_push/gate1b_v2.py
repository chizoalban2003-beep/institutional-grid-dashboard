"""Gate 1b v2 — rebuilt on the diag-PROVEN foundation.

The diag kernel (gate1b_init_diag) scored the crowned checkpoint at
0.976 (117-dim) / 0.973 (120-dim extension) on identical windows. The
v1 1b kernel scored 0.352 with nominally identical code — an invisible
runtime divergence from vendored-block shadowing. This kernel uses the
diag foundation VERBATIM (no vendored blocks) plus the language stem:
dyck generator, vocab training, F_total EWC lam=10, triple exam.
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

N_WORDS = 4096
N_TEST_WORDS = 256
N_EPOCHS = 100
BATCH = 512
LR = 1e-4
ACC_EVERY = 5
ACC_FLOOR = 0.90
MATH_FLOOR = 0.80
CLIN_FLOOR = 0.90
LAM_EWC = 10.0
ALIGNED = True
FISHER_DATASET = "fisher-v3-total"
FISHER_FILE = "fisher_v3.npz"


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


# ---------------------------- certified math exam machinery (diag-proven)

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


def _clin_r2(p, y, m):
    dm = 1.0 - m[:, :, K_MATH:]
    num = ((p[:, :, K_MATH:] - y[:, :, K_MATH:]) ** 2 * dm).sum()
    den = ((y[:, :, K_MATH:]
            - y[:, :, K_MATH:].mean(dim=(0, 1), keepdim=True)) ** 2
           * dm).sum()
    return float(1.0 - num / max(den, 1e-9))


# ---------------------------- dyck language stem (vendored reference)

V = 4
TOKENS = ["(", ")", "[", "]"]
DEPTH_MAX = 4
T_MIN, T_MAX = 32, 64
DROP_LO_D, DROP_HI_D = 0.10, 0.30


def _rng(seed):
    return np.random.default_rng(seed)


def generate_word(rng, t_min=T_MIN, t_max=T_MAX, depth_max=DEPTH_MAX):
    n = int(rng.integers(t_min, t_max + 1))
    ids = np.zeros(n, dtype=np.int64)
    stack = []
    depth = 0
    for i in range(n):
        can_close = depth > 0
        can_open = depth < depth_max
        if can_close and (not can_open or rng.random() < 0.45):
            open_type = stack.pop()
            ids[i] = 2 + open_type
            depth -= 1
        else:
            open_type = int(rng.integers(0, 2))
            stack.append(open_type)
            ids[i] = open_type
            depth += 1
    while stack:
        open_type = stack.pop()
        ids = np.concatenate([ids, np.array([2 + open_type])])
    return ids


def build_dyck_dataset(n_words, seed, aligned=True):
    rng = _rng(seed)
    X, Y, M = [], [], []
    for i in range(n_words):
        ids = generate_word(_rng(int(rng.integers(0, 2 ** 31))))
        T = len(ids)
        mask = (rng.random(T) >= rng.uniform(DROP_LO_D, DROP_HI_D)).astype(
            np.float32)
        value = ids.astype(np.float32).copy()
        last = -1
        for t in range(T):
            if mask[t] > 0.5:
                last = t
            elif last >= 0:
                value[t] = value[last]
            else:
                value[t] = 0.0
        delta = np.zeros(T, dtype=np.float32)
        last_obs = -1
        for t in range(T):
            if mask[t] > 0.5:
                last_obs = t
                delta[t] = 0.0
            elif last_obs >= 0:
                delta[t] = min(t - last_obs, DELTA_CAP)
            else:
                delta[t] = DELTA_CAP
        if aligned:
            starts = range(0, T - W + 1, W)
        else:
            n_win = (T - W) // SLIDE + 1
            starts = range(0, n_win * SLIDE, SLIDE)
            if n_win > CAP_WINDOWS:
                starts = sorted(np.random.default_rng(1000 + i).choice(
                    list(starts), size=CAP_WINDOWS, replace=False))
        for s in starts:
            a, b = int(s), int(s) + W
            x = np.stack([value[a:b], mask[a:b], delta[a:b]], axis=-1)
            X.append(x.astype(np.float32))
            Y.append(ids[a:b])
            M.append(mask[a:b])
    return (np.stack(X), np.stack(Y), np.stack(M))


def token_accuracy(pred_logits, y, mask):
    pred = np.argmax(pred_logits, axis=-1)
    dropped = ~(mask > 0.5)
    if not dropped.any():
        return 0.0
    return float((pred[dropped] == y[dropped]).mean())


def stack_consistency(pred_logits, y, mask, depth_max=DEPTH_MAX):
    pred = np.argmax(pred_logits, axis=-1)
    B, Wn = y.shape
    hits, tot = 0, 0
    for b in range(B):
        stack = []
        for t in range(Wn):
            if mask[b, t] > 0.5:
                tok = int(y[b, t])
                if tok < 2:
                    stack.append(tok)
                elif stack and tok == 2 + stack[-1]:
                    stack.pop()
                continue
            p = int(pred[b, t])
            tot += 1
            if p < 2:
                if len(stack) < depth_max:
                    hits += 1
            else:
                if stack and p == 2 + stack[-1]:
                    hits += 1
            tok = int(y[b, t])
            if tok < 2:
                stack.append(tok)
            elif stack and tok == 2 + stack[-1]:
                stack.pop()
    if tot == 0:
        return 0.0
    return float(hits / tot)


def embed_language_block(X3, total_dim=D_IN_LANG):
    B, Wn, _ = X3.shape
    grid = np.zeros((B, Wn, total_dim), dtype=np.float32)
    grid[:, :, 1::3] = 1.0
    grid[:, :, total_dim - 3:total_dim] = X3
    return grid


# ---------------------------- EWC machinery

class _ThetaStar:
    pass


_THETA_STAR = _ThetaStar()


def load_fisher_normalized():
    ckpt_dir = discover_input(FISHER_DATASET)
    f_raw = np.load(os.path.join(ckpt_dir, FISHER_FILE))
    f2 = {k: np.asarray(f_raw[k], dtype=np.float64) for k in f_raw.files}
    f2p = {}
    for k, v in f2.items():
        if k in ("gru.weight_ih_l0", "decode_cell.weight_ih") and \
                v.shape[1] == 117:
            v = np.concatenate([v, np.zeros((v.shape[0], 3),
                                            dtype=np.float64)], axis=1)
        if k == "decode_cell.weight_ih" and v.shape[1] == 220:
            v = np.concatenate([v, np.zeros((v.shape[0], 7),
                                            dtype=np.float64)], axis=1)
        f2p[k] = v
    total = sum(float(v.sum()) for v in f2p.values())
    n = sum(v.size for v in f2p.values())
    mean = total / n
    assert np.isfinite(mean) and mean > 0, f"bad Fisher mean {mean}"
    return {k: torch.tensor(v / mean, dtype=torch.float32)
            for k, v in f2p.items()}


def snapshot_theta_star(model, f2):
    st = _ThetaStar()
    for k in f2:
        setattr(st, k, dict(model.named_parameters())[k].detach().clone())
    return st


def ewc_penalty(model, f2, lam):
    total = torch.zeros((), dtype=torch.float32)
    for k, F in f2.items():
        p = dict(model.named_parameters())[k]
        th_star = getattr(_THETA_STAR, k)
        total = total + (F * (p - th_star) ** 2).sum()
    return 0.5 * lam * total


# ---------------------------- main

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = __import__("time").time()

    print("[1/7] Dyck-2 data (aligned windows)...", flush=True)
    Xtr, Ytr, Mtr = build_dyck_dataset(N_WORDS, SEED, aligned=ALIGNED)
    Xte, Yte, Mte = build_dyck_dataset(N_TEST_WORDS, SEED + 1,
                                       aligned=ALIGNED)
    Xtr = torch.tensor(embed_language_block(Xtr), dtype=torch.float32)
    Ytr = torch.tensor(Ytr, dtype=torch.long)
    Mtr = torch.tensor(Mtr, dtype=torch.float32)
    Xte = torch.tensor(embed_language_block(Xte), dtype=torch.float32)
    Yte = torch.tensor(Yte, dtype=torch.long)
    Mte = torch.tensor(Mte, dtype=torch.float32)
    print(f"  train {Xtr.shape[0]} windows, test {Xte.shape[0]}",
          flush=True)

    print("[2/7] CROWNED init (diag-proven copy 117 -> 120)...", flush=True)
    model = LanguageGrid(D_IN_LANG, HIDDEN, N_CELLS, K_ACTIVE, K_SUBJECTS,
                         V_LANG)
    ckpt_dir = discover_input("crowned-ckpt-fast")
    crowned = torch.load(os.path.join(ckpt_dir, "math2clinic_fast.pt"),
                         map_location="cpu", weights_only=True)
    DECODE_BLOCKS = [(0, 117, 0, 117), (120, 184, 117, 181),
                     (184, 223, 181, 220)]
    with torch.no_grad():
        for k, v in crowned.items():
            if k in ("vocab_head.weight", "vocab_head.bias"):
                continue
            p = dict(model.named_parameters())[k]
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
    print("  crowned weights copied (block-shift decode)", flush=True)

    print("[3/7] F_total EWC ledger...", flush=True)
    f2 = load_fisher_normalized()
    global _THETA_STAR
    _THETA_STAR = snapshot_theta_star(model, f2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {n_par:,} params | uniform lr {LR} | lam {LAM_EWC:g}",
          flush=True)

    print("[4/7] legacy exams (diag-proven path)...", flush=True)
    X18, YM, MM = _exam_windows(192, SEED + 2)
    XC, YC, MC = build_clinical_windows(64, SEED + 3)
    XM = torch.tensor(_pad120(_embed_math(X18)), dtype=torch.float32)
    YM = torch.tensor(YM, dtype=torch.float32)
    MM = torch.tensor(MM, dtype=torch.float32)
    XC = torch.tensor(_pad120(XC), dtype=torch.float32)
    YC = torch.tensor(YC, dtype=torch.float32)
    MC = torch.tensor(MC, dtype=torch.float32)
    with torch.no_grad():
        p_m = model(XM)
        if isinstance(p_m, tuple):
            p_m = p_m[0]
        p_c = model(XC)
        if isinstance(p_c, tuple):
            p_c = p_c[0]
    r2_math = _math_r2(p_m, YM, MM)
    r2_clin = _clin_r2(p_c, YC, MC)
    print("  math base: " + " ".join(f"{k} {v:.3f}"
                                     for k, v in r2_math.items()),
          flush=True)
    print(f"  clinical base: {r2_clin:.4f}", flush=True)

    print("[5/7] training (Dyck-2 CE + F_total EWC lam=10)...", flush=True)
    n = Xtr.shape[0]
    n_batches = (n + BATCH - 1) // BATCH
    curve = []
    for ep in range(N_EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        ewc_tot = 0.0
        for i in range(n_batches):
            idx = perm[i * BATCH: (i + 1) * BATCH]
            xb, yb, mb = Xtr[idx], Ytr[idx], Mtr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            vlog = out[1] if isinstance(out, tuple) else out
            lg = vlog.reshape(-1, V_LANG)
            tg = yb.reshape(-1)
            loss = nn.functional.cross_entropy(lg, tg)
            ewc = ewc_penalty(model, f2, LAM_EWC)
            loss = loss + ewc
            loss.backward()
            opt.step()
            tot += float(loss - ewc)
            ewc_tot += float(ewc)
        note = ""
        if (ep + 1) % ACC_EVERY == 0 or ep == N_EPOCHS - 1:
            with torch.no_grad():
                out = model(Xte)
                vlog = out[1] if isinstance(out, tuple) else out
                p_m = model(XM)
                p_m = p_m[0] if isinstance(p_m, tuple) else p_m
                p_c = model(XC)
                p_c = p_c[0] if isinstance(p_c, tuple) else p_c
            cons = stack_consistency(vlog.numpy(), Yte.numpy(),
                                     Mte.numpy())
            acc = token_accuracy(vlog.numpy(), Yte.numpy(), Mte.numpy())
            worst_m = min(_math_r2(p_m, YM, MM).values())
            r2c = _clin_r2(p_c, YC, MC)
            curve.append({"epoch": ep + 1, "cons": cons, "acc": acc,
                          "math_worst": worst_m, "clin": r2c})
            note = f" CONS {cons:.4f} | math {worst_m:.3f} clin {r2c:.3f}"
        print(f"  ep {ep:3d} loss {tot / n_batches:9.4f} "
              f"ewc {ewc_tot / n_batches:9.4f}{note}", flush=True)

    print("[6/7] final triple verdict...", flush=True)
    with torch.no_grad():
        out = model(Xte)
        vlog = out[1] if isinstance(out, tuple) else out
        p_m = model(XM)
        p_m = p_m[0] if isinstance(p_m, tuple) else p_m
        p_c = model(XC)
        p_c = p_c[0] if isinstance(p_c, tuple) else p_c
    cons = stack_consistency(vlog.numpy(), Yte.numpy(), Mte.numpy())
    acc = token_accuracy(vlog.numpy(), Yte.numpy(), Mte.numpy())
    r2_math_f = _math_r2(p_m, YM, MM)
    r2_clin_f = _clin_r2(p_c, YC, MC)
    worst_m = min(r2_math_f.values())
    epochs_to_90 = next((c["epoch"] for c in curve if c["cons"] >= 0.90),
                        None)
    ok = (cons >= ACC_FLOOR and worst_m >= MATH_FLOOR
          and r2_clin_f >= CLIN_FLOOR)
    print(f"  language consistency {cons:.4f} (floor {ACC_FLOOR}) | "
          f"exact {acc:.4f}", flush=True)
    print("  math: " + " ".join(f"{k} {v:.3f}"
                                for k, v in r2_math_f.items()), flush=True)
    print(f"  clinical: {r2_clin_f:.4f} (floor {CLIN_FLOOR})", flush=True)
    print("  VERDICT:", "PASS" if ok else "FAIL", flush=True)

    torch.save(model.state_dict(), "/kaggle/working/lang_1b.pt")
    report = {
        "gate": "1b-v2",
        "final_consistency": cons,
        "final_accuracy": acc,
        "epochs_to_0.90": epochs_to_90,
        "math_r2_final": r2_math_f,
        "clinical_r2_final": r2_clin_f,
        "math_r2_base": r2_math,
        "clinical_r2_base": r2_clin,
        "curve": curve,
        "params": n_par,
        "verdict": "PASS" if ok else "FAIL",
        "seconds": round(t0 and __import__("time").time() - t0, 1),
    }
    with open("/kaggle/working/gate1b_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {report['seconds']}s | verdict {report['verdict']}",
          flush=True)


def build_clinical_windows(n_stays, seed):
    """Minimal clinical-window generator (mimic_contract semantics,
    inlined to avoid ALL vendored shadowing — the diag-proven pattern)."""
    rng = _rng(seed)
    X, Y, M = [], [], []
    for stay in range(n_stays):
        g = _rng(int(rng.integers(0, 2 ** 31)))
        T = 168
        V = np.zeros((T, K_CLINICAL))
        t = np.arange(T)
        for i in range(K_CLINICAL):
            if i < 11:      # vitals rhythmic + trend
                base = g.normal(0, 1)
                amp = g.uniform(0.15, 0.5)
                freq = g.uniform(0.02, 0.10)
                trend = g.normal(0, 0.6) * np.linspace(0, 1, T)
                V[:, i] = base + amp * np.sin(freq * t + g.uniform(0, 6.28)) \
                    + trend
            else:           # labs slow
                level = g.normal(0, 1)
                drift = g.uniform(0.005, 0.02)
                V[:, i] = level + g.normal(0, 0.15) * np.sin(
                    drift * t + g.uniform(0, 6.28))
                V[:, i] = np.clip(V[:, i], -4, 4)
        n_win = min(24, max(1, (T - W) // 2))
        for _ in range(n_win):
            start = int(g.integers(0, T - W))
            win = V[start: start + W]
            drop = g.uniform(0.15, 0.75)
            mask = (g.random((W, K_CLINICAL)) > drop).astype(np.float32)
            mask[:, 28:] = 1.0         # demographics always observed
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
            x = np.zeros((W, D_IN_117), dtype=np.float32)
            x[:, 0::3] = 0.0
            x[:, 1::3] = 1.0
            x[:, 2::3] = 0.0
            x[:, 18 + 0::3] = ff
            x[:, 18 + 1::3] = mask
            x[:, 18 + 2::3] = delta
            y = np.zeros((W, K_SUBJECTS), dtype=np.float32)
            y[:, K_MATH:] = win
            m = np.zeros((W, K_SUBJECTS), dtype=np.float32)
            m[:, K_MATH:] = mask
            m[:, :K_MATH] = 1.0
            X.append(x)
            Y.append(y)
            M.append(m)
    return (np.stack(X), np.stack(Y), np.stack(M))


if __name__ == "__main__":
    main()
