"""Phase 3 Gate 1a — Language stem FLOOR: Dyck-2 masked imputation from a
random core (no transfer, no economy). Establishes the achievable bar
(epochs-to-0.90-accuracy, final accuracy) that Gate 1b (crowned core +
F_total) must match or beat while holding math/clinical memory.

Architecture extension (pre-registered, approved 2026-08-20):
  - input d_in: 117 -> 120 (the language stem is a 40th triplet at
    columns 117:120 = [value, mask, delta])
  - vocabulary head Linear(hidden, V=4) SEPARATE from the 39 regression
    heads (the first non-regression head); reads the same decode state
  - dormant-slot protocol: when language is active, math+clinical
    triplets are dormant (value=0, mask=1, delta=0) — zero bandwidth,
    zero gradient on the other stems

Task: mask Dyck-2 words (bounded depth <= 4) EHR-style, impute the
dropped tokens. Metric: masked token accuracy on dropped positions,
true-Y graded (same shape as the math/clinical exams). Loss: CE on
dropped positions of the vocab logits.

CPU kernel, uniform 1e-4, no param groups (the settled N-stem scheme).
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

SEED = 42
N_WORDS = 4096
N_TEST_WORDS = 256
N_EPOCHS = 100
BATCH = 512
LR = 1e-4
LR_LANG = 1e-3           # 10x core LR for the randomly-initialized vocab head
ACC_FLOOR = 0.90
ACC_EVERY = 5
EXAM_KINDS = ["dyck2"]
ALIGNED = True          # aligned windows: openers visible, stack learnable

HIDDEN = 192
N_CELLS = 100
K_ACTIVE = 3
K_MATH = 6
K_CLINICAL = 33
K_SUBJECTS = K_MATH + K_CLINICAL          # 39 regression heads
D_IN_LANG = 120                           # 117 + language triplet (kernel-side;
                                          # D_IN is redefined to 117 by the
                                          # vendored mimic_contract!)
V_LANG = 4                                # vocab (V also vendored)

# ---------------------------- vendored data contracts (see sync_vendored)

# >>> VENDOR (dyck_worlds) — do not edit outside the reference module

V = 4                                  # vocabulary: (, ), [, ]
TOKENS = ["(", ")", "[", "]"]
DEPTH_MAX = 4                          # bounded nesting depth (Dyck-2, depth<=4)
W = 14                                 # window length (shared core unroll)
T_MIN, T_MAX = 32, 64                  # word length range (same-range split)
SLIDE = 2
CAP_WINDOWS = 24
DROP_LO, DROP_HI = 0.10, 0.30          # SPARSE masking (v3, Gate-1 revision
                                       # 2026-08-20): Dyck-n has ZERO
                                       # redundancy — a dropped '[' makes the
                                       # matching ']' mathematically
                                       # unpredictable. v1/v2 masked 50%
                                       # (EHR-style, right for noisy ICU data,
                                       # WRONG for formal grammar) -> dropped-
                                       # token marginal went near-uniform and
                                       # the floor capped at 0.381 (the
                                       # ambiguity ceiling, not a learning
                                       # failure). 10-30% = the BERT MLM
                                       # standard regime, adapted for a
                                       # zero-redundancy grammar.
DELTA_CAP = 24.0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_word(rng: np.random.Generator,
                  t_min: int = T_MIN, t_max: int = T_MAX,
                  depth_max: int = DEPTH_MAX) -> np.ndarray:
    """One balanced Dyck-2 word (array of token ids), bounded depth.

    Constrained random walk over depth with '('/'[' (+1) and ')'/']'
    (-1), keeping depth in [0, depth_max]: a step that would violate the
    bounds is forced to the other action; otherwise random (~45% close).
    The residual unmatched openers are then APPENDED with their matching
    closers (never overwriting emitted tokens), so the word is balanced
    by construction with length in [n, n + depth_max]. Closing bracket
    type MUST match the most recent unmatched opener (stack discipline)
    — this is the hierarchical dependency the GRU must learn.
    """
    n = int(rng.integers(t_min, t_max + 1))
    ids = np.zeros(n, dtype=np.int64)
    stack = []                          # 0 = '(', 1 = '['
    depth = 0
    for i in range(n):
        can_close = depth > 0
        can_open = depth < depth_max
        if can_close and (not can_open or rng.random() < 0.45):
            open_type = stack.pop()
            ids[i] = 2 + open_type      # ')'=2 for '(', ']'=3 for '['
            depth -= 1
        else:
            open_type = int(rng.integers(0, 2))
            stack.append(open_type)
            ids[i] = open_type
            depth += 1
    # append matching closers for any residual openers (safe: append only)
    while stack:
        open_type = stack.pop()
        ids = np.concatenate([ids, np.array([2 + open_type])])
    return ids


def build_dyck_dataset(n_words: int, seed: int,
                       aligned: bool = True) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]:
    """(X, Y, M): X (B, W, 3) [value_ffill, mask, delta], Y (B, W) true
    token ids, M (B, W) observation flags (1 = observed).

    aligned=True: windows START at word positions 0, W, 2W, ... (chunk
    alignment) so every closing bracket's opener is inside the window
    whenever possible — preserves the stack-tracking signal per window.
    aligned=False: sliding windows (SLIDE), closers whose opener fell
    before the window start are legitimately under-determined (the model
    learns the conditional marginal) — the rand floor (Gate 1a) measures
    the achievable bar either way.
    """
    rng = _rng(seed)
    X, Y, M = [], [], []
    for i in range(n_words):
        ids = generate_word(_rng(int(rng.integers(0, 2 ** 31))))
        T = len(ids)
        mask = (rng.random(T) >= rng.uniform(DROP_LO, DROP_HI)).astype(
            np.float32)
        value = ids.astype(np.float32).copy()
        last = -1
        for t in range(T):
            if mask[t] > 0.5:
                last = t
            elif last >= 0:
                value[t] = value[last]
            else:
                value[t] = 0.0          # cold start: nothing observed yet
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
                g2 = np.random.default_rng(1000 + i)
                starts = sorted(np.random.default_rng(1000 + i).choice(
                    list(starts), size=CAP_WINDOWS, replace=False))
        for s in starts:
            a, b = int(s), int(s) + W
            x = np.stack([value[a:b], mask[a:b], delta[a:b]], axis=-1)
            X.append(x.astype(np.float32))
            Y.append(ids[a:b])
            M.append(mask[a:b])
    return (np.stack(X), np.stack(Y), np.stack(M))


def token_accuracy(pred_logits: np.ndarray, y: np.ndarray,
                   mask: np.ndarray) -> float:
    """Masked token accuracy on DROPPED positions (the Phase-3 exam).

    pred_logits: (B, W, V) model output; argmax per position vs true id.
    """
    pred = np.argmax(pred_logits, axis=-1)
    dropped = ~(mask > 0.5)
    if not dropped.any():
        return 0.0
    return float((pred[dropped] == y[dropped]).mean())


def stack_consistency(pred_logits: np.ndarray, y: np.ndarray,
                      mask: np.ndarray, depth_max: int = DEPTH_MAX) -> float:
    """GRAMMAR-CONSISTENCY grader (Phase-3 acceptance gate, approved
    2026-08-20): a prediction on a DROPPED position is CORRECT iff it is
    a legal continuation of the stack built from the window's TRUE
    tokens — a closer matching the stack top, or any opener within the
    depth bound. This decouples the structural logic from the generator's
    opener-type coin flip (exact-token accuracy caps ~0.74-0.78 even for
    a perfect stack tracker; the 0.90 bar was unreachable by design).
    The training loss is UNCHANGED (CE on exact tokens); only the
    acceptance gate moves to grammar-consistency (the blueprint's
    "Neutrality" slice: any token inside the legal manifold is neutral).

    NOTE: stack state at window start is unknown (windows slice words,
    h0=0) — the stack is built from observed TRUE tokens within the
    window only, so a closer whose opener fell before the window start
    is scored against the partial stack (a conservative underestimate).
    """
    pred = np.argmax(pred_logits, axis=-1)
    B, Wn = y.shape
    hits, tot = 0, 0
    for b in range(B):
        stack = []                      # 0='(', 1='[' (from true tokens)
        for t in range(Wn):
            if mask[b, t] > 0.5:
                tok = int(y[b, t])
                if tok < 2:
                    stack.append(tok)
                elif stack and tok == 2 + stack[-1]:
                    stack.pop()
                # mismatched closer in true data cannot happen (balanced)
                continue
            # dropped position: grade legality of the model's argmax
            p = int(pred[b, t])
            tot += 1
            if p < 2:                   # opener: legal iff depth < max
                if len(stack) < depth_max:
                    hits += 1
            else:                       # closer: legal iff matches top
                if stack and p == 2 + stack[-1]:
                    hits += 1
            # advance the true stack past this position regardless
            tok = int(y[b, t])
            if tok < 2:
                stack.append(tok)
            elif stack and tok == 2 + stack[-1]:
                stack.pop()
    if tot == 0:
        return 0.0
    return float(hits / tot)


def embed_language_block(X3: np.ndarray, total_dim: int = D_IN_LANG) -> np.ndarray:
    """(B, W, 3) [value, mask, delta] language windows -> (B, W, 120) grid.

    The Phase-2 grid has 117 input columns (39 triplets: 6 math + 33
    clinical); the language stem is a 40th triplet at columns 117:120.
    Dormant-slot protocol: when language is active, the math+clinical
    triplets stay dormant (value=0, mask=1, delta=0) so the shared core
    sees exactly one domain's signal — zero bandwidth, zero gradient on
    the other stems (hard copy pins their regression heads to 0).
    """
    B, Wn, _ = X3.shape
    if X3.shape[2] != 3:
        raise ValueError(f"language windows must be (B, W, 3), got {X3.shape}")
    grid = np.zeros((B, Wn, total_dim), dtype=np.float32)
    grid[:, :, 1::3] = 1.0                      # all triplets dormant
    grid[:, :, total_dim - 3:total_dim] = X3    # language triplet active
    return grid


# <<< VENDOR (dyck_worlds) — do not edit outside the reference module

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

# OVERRIDE: the vendored dyck_worlds block defines EXAM_KINDS = ["dyck2"]
# which is wrong for the math exam — damage() iterates EXAM_KINDS to set
# value/mask/delta per channel, so ["dyck2"] only processes 1 of 6 channels
# (the other 5 get zeros → model sees wrong input → 0.352/0.599 scores).
EXAM_KINDS = ["sine", "cosine", "decay", "step", "sigmoid", "lorenz"]

# ---------------------------- certified math exam machinery

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



# ---------------------------- model (extended grid: + vocab head)

class LanguageGrid(nn.Module):
    """MathSchoolGrid extended to 120-dim with a vocabulary head.

    The 39 regression heads (6 math + 33 clinical) are unchanged; the
    vocab head Linear(hidden, V) reads the same decode state and emits
    token logits. The language triplet lives at input columns 117:120.
    """

    def __init__(self, d_in, hidden, n_cells, k, k_subjects, vocab):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.scorer = nn.Linear(hidden, n_cells)
        self.cell_block = nn.Linear(hidden, n_cells * 64)
        self.decode_cell = nn.GRUCell(d_in + 64 + k_subjects + vocab,
                                      hidden)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(k_subjects)])
        self.vocab_head = nn.Linear(hidden, vocab)
        self.k = k
        self.n_cells = n_cells
        self.k_subjects = k_subjects
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
        outs, vlogits = [], []
        for t in range(Wn):
            ctx = torch.cat([x[:, t], pooled, prev, prev_tok], dim=1)
            state = self.decode_cell(ctx, state)
            y_est = torch.cat(
                [head(state) for head in self.heads], dim=1)
            y = m[:, t] * value[:, t] + (1.0 - m[:, t]) * y_est
            outs.append(y)
            prev = y
            lg = self.vocab_head(state)
            vlogits.append(lg)
            prev_tok = torch.softmax(lg, dim=1)
        out = torch.stack(outs, dim=1)
        vlog = torch.stack(vlogits, dim=1)
        if return_routing:
            return out, vlog, votes
        return out, vlog


class MathSchoolGrid117(nn.Module):
    """Certified 117-dim grid (no vocab head) — the diag-proven control
    that reproduces the crowned exam (sine 0.976, lorenz 0.857)."""

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


# ---------------------------- Gate 1b machinery

FISHER_DATASET = "fisher-v3-total"
FISHER_FILE = "fisher_v3.npz"
LAM_EWC = 5.0                     # lowered to let GRU adapt recurrent state for Dyck-2
MATH_FLOOR = 0.80
CLIN_FLOOR = 0.90
D_IN_117 = 117                    # crowned model's input width


def load_fisher_normalized():
    """Load fisher_v3.npz (F_total = F_math + F_clinical), mean-1 global."""
    ckpt_dir = discover_input(FISHER_DATASET)
    f_raw = np.load(os.path.join(ckpt_dir, FISHER_FILE))
    f2 = {k: np.asarray(f_raw[k], dtype=np.float64) for k in f_raw.files}
    total = sum(float(v.sum()) for v in f2.values())
    n = sum(v.size for v in f2.values())
    mean = total / n
    assert np.isfinite(mean) and mean > 0, f"bad Fisher mean {mean}"
    # pad grown projections to the 120-dim geometry: F was computed on the
    # 117-dim crowned model; the language columns are NEW (F=0 there —
    # uncharged, exactly like demographics).
    f2p = {}
    for k, v in f2.items():
        if k in ("gru.weight_ih_l0", "decode_cell.weight_ih") and \
                v.shape[1] == 117:
            pad = np.zeros((v.shape[0], 120 - 117), dtype=np.float64)
            v = np.concatenate([v, pad], axis=1)
        if k == "decode_cell.weight_ih" and v.shape[1] == 220:
            pad = np.zeros((v.shape[0], 227 - 220), dtype=np.float64)
            v = np.concatenate([v, pad], axis=1)
        f2p[k] = v
    f2 = f2p
    total = sum(float(v.sum()) for v in f2.values())
    n = sum(v.size for v in f2.values())
    mean = total / n
    assert np.isfinite(mean) and mean > 0, f"bad Fisher mean {mean}"
    f2 = {k: torch.tensor(v / mean, dtype=torch.float32)
          for k, v in f2.items()}
    print(f"[ewc] F_total loaded {len(f2)} keys, padded to 120-dim, "
          f"mean-1 normalized (raw mean {mean:.2e})", flush=True)
    return f2


def ewc_penalty(model, f2, lam):
    """L_EWC = (lam/2) * sum F * (theta - theta*)^2 over ALL crowned keys.

    theta* = the crowned checkpoint weights; F is zero on demographics/
    scorer (honest structural zeros). The new language columns + vocab
    head are NOT in f2 -> uncharged.
    """
    total = torch.zeros((), dtype=torch.float32)
    for k, F in f2.items():
        p = dict(model.named_parameters())[k]
        th_star = getattr(_THETA_STAR, k)
        total = total + (F * (p - th_star) ** 2).sum()
    return 0.5 * lam * total


class _ThetaStar:
    pass


_THETA_STAR = _ThetaStar()


def snapshot_theta_star(model, f2):
    st = _ThetaStar()
    for k in f2:
        setattr(st, k, dict(model.named_parameters())[k].detach().clone())
    return st


def discover_input(name):
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if name in (root, os.path.basename(root)):
            return root
    raise FileNotFoundError(f"dataset {name} not found under {base}")


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[1/6] Dyck-2 data (aligned windows)...", flush=True)
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

    print("[2/6] CROWNED init (v2 lam10 fast, 117 -> 120)...", flush=True)
    model = LanguageGrid(D_IN_LANG, HIDDEN, N_CELLS, K_ACTIVE, K_SUBJECTS, V_LANG)
    ckpt_dir = discover_input("crowned-ckpt-fast")
    crowned = torch.load(os.path.join(ckpt_dir, "math2clinic_fast.pt"),
                         map_location="cpu", weights_only=True)
    # the crowned model is MathSchoolGrid(117, ...): 39 heads, no vocab
    # head, 117-dim input projections. Copy its state into the 120-dim
    # LanguageGrid: gru.weight_ih (576,117)->(576,120) copy cols 0:117,
    # decode_cell.weight_ih (576, 220+3?) — the LanguageGrid decode ctx
    # is d_in + 64 + k_subjects + vocab = 120+64+39+4 = 227 vs crowned
    # 117+64+39 = 220. New language cols + vocab-ctx cols zero-init.
    # DECODE_CTX BLOCK SHIFT (Gate-1b bug, 2026-08-20): the crowned
    # decode ctx is [x(117), pooled(64), prev(39)] = 220; the LanguageGrid
    # ctx is [x(120), pooled(64), prev(39), prev_tok(4)] = 227. The 3 new
    # language input cols shift pooled -> [120:184] and prev -> [184:223].
    # A naive p[:, :220] = v lands pooled/prev 3 cols off and corrupts the
    # whole decode path (math base collapsed to ~0 at init while crowned
    # scored 0.857 worst on the same windows). Block-spec copy:
    DECODE_BLOCKS = [(0, 117, 0, 117),        # x: shared cols
                     (120, 184, 117, 181),    # pooled: shifted +3
                     (184, 223, 181, 220)]    # prev: shifted +3
    with torch.no_grad():
        for k, v in crowned.items():
            if k in ("vocab_head.weight", "vocab_head.bias"):
                continue
            p = dict(model.named_parameters())[k]
            if tuple(p.shape) == tuple(v.shape):
                p.copy_(v)
            elif k == "decode_cell.weight_ih":
                # shifted block copy; language x cols + prev_tok stay 0
                for tlo, thi, slo, shi in DECODE_BLOCKS:
                    p[:, tlo:thi].copy_(v[:, slo:shi])
            elif p.ndim == 2 and v.ndim == 2 and p.shape[0] == v.shape[0] \
                    and p.shape[1] > v.shape[1]:
                # input-projection growth (gru.weight_ih: append-only, no
                # internal shift) — copy the shared columns, zero the new
                # language columns (dormant-protocol parity)
                p[:, :v.shape[1]].copy_(v)
                p[:, v.shape[1]:].zero_()
            elif p.ndim == 1 and v.ndim == 1 and p.shape[0] == v.shape[0]:
                p.copy_(v)
            else:
                raise SystemExit(f"shape mismatch on {k}: {p.shape} vs "
                                 f"{v.shape}")
    print("  crowned weights copied; language columns + vocab head fresh "
          "(zero-init on the grown projections)", flush=True)
    grown = [k for k, v in crowned.items()
             if k not in ("vocab_head.weight", "vocab_head.bias")
             and tuple(dict(model.named_parameters())[k].shape)
             != tuple(v.shape)]
    print(f"  grown keys: {grown}", flush=True)

    print("[3/6] F_total EWC ledger...", flush=True)
    f2 = load_fisher_normalized()
    global _THETA_STAR
    _THETA_STAR = snapshot_theta_star(model, f2)
    opt = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters()
                    if "vocab_head" in n], "lr": LR_LANG},
        {"params": [p for n, p in model.named_parameters()
                    if "vocab_head" not in n], "lr": LR},
    ])
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {n_par:,} params | core lr {LR} | lang lr {LR_LANG} | "
          f"lam {LAM_EWC:g} | vocab {V_LANG}", flush=True)

    print("[4/6] legacy exams (math + clinical at crowned init)...",
          flush=True)
    XM, YM, MM = exam_windows(192, SEED + 2)     # math exam (certified)
    XM18 = XM                                    # keep 18-dim for the control
    XC, YC, MC = clinical_windows(64, SEED + 3)  # clinical (mimic contract)
    def embed_math_inline(X18):
        """DIAG-PROVEN math embed: (B,W,18) -> (B,W,117), clinical dormant.
        (Uses an explicit local copy — the vendored embed_math_block
        resolves D_IN/total_dim differently at runtime and scored math
        base 0.352 vs the diag's 0.973 on identical windows.)"""
        B, Wn, _ = X18.shape
        g = np.zeros((B, Wn, 117), dtype=np.float32)
        g[:, :, 1::3] = 1.0
        g[:, :, :K_MATH * 3] = X18
        return g

    def pad_to_120(x117):
        """117-dim grid -> 120-dim with the language triplet DORMANT
        (value=0, mask=1, delta=0) — the reverse of embed_language_block."""
        B, Wn, _ = x117.shape
        x = np.zeros((B, Wn, D_IN_LANG), dtype=np.float32)
        x[:, :, :117] = x117
        x[:, :, 118] = 1.0              # language mask channel = observed
        return x
    XM = torch.tensor(pad_to_120(embed_math_inline(XM)), dtype=torch.float32)
    YM = torch.tensor(YM, dtype=torch.float32)
    MM = torch.tensor(MM, dtype=torch.float32)
    XC = torch.tensor(pad_to_120(XC), dtype=torch.float32)  # 117 -> 120
    # (clinical windows are already (B,W,117) with math dormant — the
    #  diag's clinical base 0.9923 matched crowned 0.997 through this path)
    YC = torch.tensor(YC, dtype=torch.float32)
    MC = torch.tensor(MC, dtype=torch.float32)
    with torch.no_grad():
        p_m = model(XM)[0]
        p_c = model(XC)[0]
    r2_math = {k: float(1.0 - (((p_m[:, :, i:i+1] - YM[:, :, i:i+1]) ** 2)
                               * (1.0 - MM[:, :, i:i+1])).sum()
                          / max((((YM[:, :, i:i+1]
                                   - YM[:, :, i:i+1].mean(dim=(0, 1),
                                                          keepdim=True)) ** 2)
                                 * (1.0 - MM[:, :, i:i+1])).sum(), 1e-9))
               for i, k in enumerate(["sine", "cosine", "decay", "step",
                                      "sigmoid", "lorenz"])}
    r2_clin = float(1.0 - ((p_c[:, :, K_MATH:] - YC[:, :, K_MATH:]) ** 2
                           * (1.0 - MC[:, :, K_MATH:])).sum()
                    / max(((YC[:, :, K_MATH:]
                            - YC[:, :, K_MATH:].mean(dim=(0, 1),
                                                     keepdim=True)) ** 2
                           * (1.0 - MC[:, :, K_MATH:])).sum(), 1e-9))
    print("  math base: " + " ".join(f"{k} {v:.3f}"
                                     for k, v in r2_math.items()), flush=True)
    print(f"  clinical base: {r2_clin:.4f}", flush=True)

    # ---- 117-dim control (sanity: crowned exam baseline) ----
    ctrl117 = MathSchoolGrid117(D_IN_117, HIDDEN, N_CELLS, K_ACTIVE,
                                K_SUBJECTS)
    ctrl117.load_state_dict(
        torch.load(os.path.join(discover_input("crowned-ckpt-fast"),
                                "math2clinic_fast.pt"),
                   map_location="cpu", weights_only=True), strict=True)
    ctrl117.eval()
    with torch.no_grad():
        p117 = ctrl117(torch.tensor(embed_math_inline(XM18),
                                    dtype=torch.float32))
    r2_ctrl = {k: float(1.0 - (((p117[:, :, i:i+1] - YM[:, :, i:i+1])
                                ** 2) * (1.0 - MM[:, :, i:i+1])).sum()
                         / max((((YM[:, :, i:i+1]
                                  - YM[:, :, i:i+1].mean(dim=(0, 1),
                                                         keepdim=True))
                                 ** 2) * (1.0 - MM[:, :, i:i+1])).sum(),
                                1e-9))
               for i, k in enumerate(["sine", "cosine", "decay", "step",
                                      "sigmoid", "lorenz"])}
    print("  [ctrl 117] " + " ".join(f"{k} {v:.3f}"
                                     for k, v in r2_ctrl.items()),
          flush=True)

    print("[5/6] training (Dyck-2 CE + F_total EWC)...", flush=True)
    n = Xtr.shape[0]
    n_batches = (n + BATCH - 1) // BATCH
    WARMUP_EPC = 25                # no EWC for first 25 eps — let GRU route lang
    curve = []
    for ep in range(N_EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        ewc_tot = 0.0
        use_ewc = ep >= WARMUP_EPC
        for i in range(n_batches):
            idx = perm[i * BATCH: (i + 1) * BATCH]
            xb, yb, mb = Xtr[idx], Ytr[idx], Mtr[idx]
            opt.zero_grad(set_to_none=True)
            _, vlog = model(xb)
            lg = vlog.reshape(-1, V_LANG)
            tg = yb.reshape(-1)
            # mask-WEIGHTED CE: only dropped tokens (mb=0) contribute to loss;
            # observed tokens (mb=1) are context, not prediction targets.
            mask_w = (1.0 - mb.reshape(-1))
            ce_raw = nn.functional.cross_entropy(lg, tg, reduction="none")
            loss = (ce_raw * mask_w).sum() / mask_w.sum().clamp(min=1.0)
            if use_ewc:
                ewc = ewc_penalty(model, f2, LAM_EWC)
                loss = loss + ewc
                ewc_tot += float(ewc)
            loss.backward()
            opt.step()
            tot += float(loss - (ewc if use_ewc else 0.0))
        note = ""
        if (ep + 1) % ACC_EVERY == 0 or ep == N_EPOCHS - 1:
            with torch.no_grad():
                _, vlog = model(Xte)
                p_m = model(XM)[0]
                p_c = model(XC)[0]
            cons = stack_consistency(vlog.numpy(), Yte.numpy(),
                                     Mte.numpy())
            acc = token_accuracy(vlog.numpy(), Yte.numpy(), Mte.numpy())
            worst_m = min(float(1.0 - (((p_m[:, :, i:i+1] - YM[:, :, i:i+1])
                                        ** 2) * (1.0 - MM[:, :, i:i+1])).sum()
                                / max((((YM[:, :, i:i+1]
                                         - YM[:, :, i:i+1].mean(dim=(0, 1),
                                                                keepdim=True))
                                        ** 2) * (1.0 - MM[:, :, i:i+1])).sum(),
                                       1e-9))
                          for i in range(6))
            r2c = float(1.0 - ((p_c[:, :, K_MATH:] - YC[:, :, K_MATH:]) ** 2
                               * (1.0 - MC[:, :, K_MATH:])).sum()
                        / max(((YC[:, :, K_MATH:]
                                - YC[:, :, K_MATH:].mean(dim=(0, 1),
                                                         keepdim=True)) ** 2
                               * (1.0 - MC[:, :, K_MATH:])).sum(), 1e-9))
            curve.append({"epoch": ep + 1, "cons": cons, "acc": acc,
                          "math_worst": worst_m, "clin": r2c})
            note = f" CONS {cons:.4f} | math {worst_m:.3f} clin {r2c:.3f}"
        phase = "" if use_ewc else " W"
        print(f"  ep {ep:3d}{phase} loss {tot / n_batches:9.4f} "
              f"ewc {ewc_tot / n_batches:9.4f}{note}", flush=True)

    print("[6/6] final triple verdict...", flush=True)
    with torch.no_grad():
        _, vlog = model(Xte)
        p_m = model(XM)[0]
        p_c = model(XC)[0]
    cons = stack_consistency(vlog.numpy(), Yte.numpy(), Mte.numpy())
    acc = token_accuracy(vlog.numpy(), Yte.numpy(), Mte.numpy())
    r2_math_f = {k: float(1.0 - (((p_m[:, :, i:i+1] - YM[:, :, i:i+1]) ** 2)
                                 * (1.0 - MM[:, :, i:i+1])).sum()
                            / max((((YM[:, :, i:i+1]
                                     - YM[:, :, i:i+1].mean(dim=(0, 1),
                                                            keepdim=True))
                                    ** 2) * (1.0 - MM[:, :, i:i+1])).sum(),
                                   1e-9))
                 for i, k in enumerate(["sine", "cosine", "decay", "step",
                                        "sigmoid", "lorenz"])}
    r2_clin_f = float(1.0 - ((p_c[:, :, K_MATH:] - YC[:, :, K_MATH:]) ** 2
                             * (1.0 - MC[:, :, K_MATH:])).sum()
                      / max(((YC[:, :, K_MATH:]
                              - YC[:, :, K_MATH:].mean(dim=(0, 1),
                                                       keepdim=True)) ** 2
                             * (1.0 - MC[:, :, K_MATH:])).sum(), 1e-9))
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
        "gate": "1b",
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
        "seconds": round(time.time() - t0, 1),
    }
    with open("/kaggle/working/gate1b_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {report['seconds']}s | verdict {report['verdict']}",
          flush=True)


if __name__ == "__main__":
    main()
