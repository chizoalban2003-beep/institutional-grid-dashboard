"""MIMIC-contract reference generators (numpy-only, torch-free, testable).

Single source of truth for the Phase-2 (math-to-MIMIC) kernel's clinical
data contract. The kernel vendors a byte-identical copy via
curriculum/sync_vendored.py (marked block below); the parity harness
diff-verifies the two copies after ANY edit.

Sub-grid geometry (user-locked 2026-08-19): the Phase-2 model is a single
(M=6 math subjects, C=33 clinical features) grid:

    heads  0..5   = Phase-1 math kinds (sine cosine decay step sigmoid
                    lorenz) — permanent, transfer-initialized
    heads  6..38  = MIMIC clinical targets (FEATURE_NAMES_CLINICAL)
    input  (B, W, 117) = [math triplets 0:18] + [clinical triplets 18:117]

Dormant-slot protocol: a domain's channels carry value=0, mask=1, delta=0
when the OTHER domain is active (hard copy pins the head output to 0,
prev-feedback stays 0, zero gradient — no loss surgery needed). The
clinical generator below emits the full (B,W,117) tensor with math slots
dormant; the kernel pads math windows to 117 the same way for the exam.

The 6 dropped labs are the most-missing on real MIMIC train (measured
2026-08-19): Bilirubin_direct 0.998, Fibrinogen 0.993, TroponinI 0.991,
Bilirubin_total 0.985, Alkalinephos 0.984, AST 0.984 — they lack the
chronological density that autoregressive modeling requires.

Clinical semantics (mirrors data_engine/mimic_ingest.py): vitals rhythmic
+ trend with low missingness, labs slow high-missingness, per-feature z
via TRAIN stats, ffill (causal, no backward fill), mask column, delta =
hours since last observation (backward-looking, CAPPED at DELTA_CAP, 0 on
observed slots). Demographics (Age..HospAdmTime) always observed.
"""

from __future__ import annotations

import numpy as np

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