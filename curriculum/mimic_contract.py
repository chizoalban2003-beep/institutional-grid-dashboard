"""MIMIC-contract reference generators (numpy-only, torch-free, testable).

Single source of truth for the Phase-2 (math-to-MIMIC) kernel's data
contracts. The kernel vendors a byte-identical copy via
curriculum/sync_vendored.py (marked block below); the parity harness
diff-verifies the two copies after ANY edit.

Two contracts:

1. math exam — per-kind single-channel windows (sine/cosine/decay/step/
   sigmoid/lorenz) used for the no-forgetting readout exam and the
   LAMBDA_MATH anchor loss. Since Phase-2 heads are per-feature, each
   exam window carries ONE kind; the model's head[kind] must reproduce
   it (readout + priors retention).

2. clinical windows — (B, W, K*3) [value_ffill, mask, delta] triplets
   over the 39 FEATURE_NAMES, mirroring data_engine/mimic_ingest.py's
   semantics: vitals rhythmic+trend with low missingness, labs slow
   high-missingness, per-feature z scores via TRAIN stats, ffill
   (causal, no backward fill), mask column, delta = hours since last
   observation (backward-looking, CAPPED at DELTA_CAP, 0 on observed
   slots). Demographics (Age..HospAdmTime, indices 34..38) always
   observed.
"""

from __future__ import annotations

import numpy as np

# >>> VENDOR (mimic_contract) — do not edit outside the reference module

W = 14
DELTA_CAP = 24.0
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_DT = 0.02
LORENZ_STEPS = 2000

K = 39
FEATURE_NAMES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime",
]
VITALS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
          "FiO2", "pH", "SaO2", "Age", "Gender", "Unit1", "Unit2"}
DEMOGRAPHICS_FROM = 34  # Age .. HospAdmTime are static, always observed


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


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
    """(B, W, 3) value/mask/delta windows, one KIND per window + kind list.

    Each window is a single math channel (channel = kind index); the B
    rows cycle through the 6 kinds deterministically so every kind gets
    ~n_windows/6 windows. Mask column = per-position observation flag
    drawn in (DROP_LO, DROP_HI) independently; delta = 0 (single channel,
    no feature structure to track).
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


def clinical_stay(g: np.random.Generator) -> np.ndarray:
    """One 168-step stay of 39 features (vitals rhythmic, labs slow)."""
    T = 168
    V = np.zeros((T, K))
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
    """(X, Y, M): X (B, W, K*3) triplets, Y (B, W, K) true values,
    M (B, W, K) observation flags (1 = observed).
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
            mask = (g.random((W, K)) > drop).astype(np.float32)
            mask[:, DEMOGRAPHICS_FROM:] = 1.0
            ff = win.copy()
            for k in range(K):
                last = None
                for tt in range(W):
                    if mask[tt, k] > 0.5:
                        last = win[tt, k]
                    elif last is None:
                        last = win[0, k]
                    else:
                        ff[tt, k] = last
            delta = np.zeros_like(win)
            for k in range(K):
                last_obs = -1
                for tt in range(W):
                    if mask[tt, k] > 0.5:
                        last_obs = tt
                    elif last_obs >= 0:
                        delta[tt, k] = min(tt - last_obs, DELTA_CAP)
                    else:
                        delta[tt, k] = DELTA_CAP
            X.append(np.stack([ff, mask, delta], axis=-1)
                     .reshape(W, K * 3).astype(np.float32))
            Y.append(win.astype(np.float32))
            M.append(mask)
    return np.stack(X), np.stack(Y), np.stack(M)


def masked_r2(pred: np.ndarray, target: np.ndarray,
              drop_mask: np.ndarray) -> float:
    """R2 over dropped slots only (drop_mask = 1 - mask)."""
    num = ((pred - target) ** 2 * drop_mask).sum()
    den = ((target - target.mean(axis=1, keepdims=True)) ** 2 * drop_mask).sum()
    return float(1.0 - num / max(den, 1e-9))

# <<< VENDOR (mimic_contract) — do not edit outside the reference module