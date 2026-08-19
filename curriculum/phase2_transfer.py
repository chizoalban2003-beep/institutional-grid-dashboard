"""Phase-2 transfer utilities — Physics priors -> clinical tensors (numpy).

Bridges the certified Phase-1 math grid into the MIMIC-IV clinical stem.

Sub-grid geometry (user-locked 2026-08-19): the Phase-2 model is ONE grid
with 39 heads = 6 math (heads 0-5) + 33 clinical (heads 6-38). The math
heads are PERMANENT — their readouts transfer wholesale.

Transfer surface (three kinds):

  FULL TENSOR (shapes identical):
    gru.weight_hh_l0 / biases            recurrent dynamics
    decode_cell.weight_hh / biases        mask-decoder recurrence
    scorer.{weight,bias}                  100-cell routing manifold
    cell_block.{weight,bias}              cell feature library
    heads.0..5.{weight,bias}              PERMANENT math readouts

  COLUMN BLOCKS (partial copies — REQUIRED for exam baseline parity):
    gru.weight_ih_l0        [0:18] <- p1 [0:18]   math window channels
    decode_cell.weight_ih   [0:18]  <- p1 [0:18]   math window channels
                            [117:181] <- p1 [18:82] pooled latent block
                            [181:187] <- p1 [82:88] prev math-head block
    Without these the random clinical input projections would drive the
    transferred recurrent/scorer machinery out of distribution on math
    windows and the no-forgetting exam baseline would collapse.

  RE-INIT (fresh, shapes differ / new keys):
    gru.weight_ih_l0        [18:117]   clinical window columns
    decode_cell.weight_ih   [18:117], [187:220]   clinical window + prev
    heads.6..38.{weight,bias}           new clinical readouts

Numpy-only here (this box has no torch): the mapping logic is tested
against shaped mock tensors; the kernel applies it to real state_dicts.
"""

from __future__ import annotations

import numpy as np

# keys whose shapes are identical between Phase 1 (d_in=18, K=6) and
# Phase 2 (d_in=117, K=39) at hidden=192 / cells=100 / top-3
TRANSFER_KEYS = {
    "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "decode_cell.weight_hh", "decode_cell.bias_ih", "decode_cell.bias_hh",
    "scorer.weight", "scorer.bias",
    "cell_block.weight", "cell_block.bias",
}
MATH_HEAD_KEYS = {
    f"heads.{i}.weight" for i in range(6)
} | {f"heads.{i}.bias" for i in range(6)}
TRANSFER_KEYS = TRANSFER_KEYS | MATH_HEAD_KEYS

# (target_key -> [(tgt_lo, tgt_hi, src_lo, src_hi)]); src slices index the
# Phase-1 tensor (columns), tgt slices index the Phase-2 tensor (columns).
COLUMN_COPY_SPEC = {
    "gru.weight_ih_l0": [(0, 18, 0, 18)],
    "decode_cell.weight_ih": [
        (0, 18, 0, 18),        # math window channels (same positions)
        (117, 181, 18, 82),    # pooled latent block (positionally shifted)
        (181, 187, 82, 88),    # prev math-head block (first 6 heads)
    ],
}

K_MATH = 6
K_CLINICAL = 33
D_IN = 117
CLINICAL_SLOT = K_MATH * 3  # 18
HIDDEN = 192
N_CELLS = 100


def split_transfer_keys(state_dict: dict) -> tuple[list[str], list[str]]:
    """Partition a checkpoint's keys into (transferable-named, otherwise)."""
    transfer, reinit = [], []
    for k in sorted(state_dict):
        if k in TRANSFER_KEYS:
            transfer.append(k)
        else:
            reinit.append(k)
    return transfer, reinit


def dims_for_phase2(k_subjects: int = K_MATH + K_CLINICAL,
                    d_in: int = D_IN) -> dict[str, tuple]:
    """Phase-2 tensor shapes for the MathSchoolGrid architecture."""
    ctx = d_in + 64 + k_subjects
    ctx_p1 = 18 + 64 + 6
    dims = {
        "gru.weight_ih_l0": (3 * HIDDEN, d_in),
        "gru.weight_hh_l0": (3 * HIDDEN, HIDDEN),
        "gru.bias_ih_l0": (3 * HIDDEN,),
        "gru.bias_hh_l0": (3 * HIDDEN,),
        "decode_cell.weight_ih": (3 * HIDDEN, ctx),
        "decode_cell.weight_hh": (3 * HIDDEN, HIDDEN),
        "decode_cell.bias_ih": (3 * HIDDEN,),
        "decode_cell.bias_hh": (3 * HIDDEN,),
        "scorer.weight": (N_CELLS, HIDDEN),
        "scorer.bias": (N_CELLS,),
        "cell_block.weight": (N_CELLS * 64, HIDDEN),
        "cell_block.bias": (N_CELLS * 64,),
    }
    for i in range(min(k_subjects, K_MATH)):   # permanent math readouts
        dims[f"heads.{i}.weight"] = (1, HIDDEN)
        dims[f"heads.{i}.bias"] = (1,)
    return dims


def transferable(state_key: str, state_shape: tuple,
                 target_shapes: dict) -> bool:
    """True iff state_key can be copied whole into the Phase-2 model."""
    if state_key not in TRANSFER_KEYS:
        return False
    shape = target_shapes.get(state_key)
    if shape is None:
        return False
    return tuple(state_shape) == tuple(shape)


def partial_key(state_key: str) -> bool:
    """True iff state_key has a column-block copy spec (partial transfer)."""
    return state_key in COLUMN_COPY_SPEC


def plan_transfer(math_state: dict, k_subjects: int = K_MATH + K_CLINICAL,
                  d_in: int = D_IN) -> tuple[list[str], list[str], list[str]]:
    """Plan a transfer: (full_copy, partial_copy, reinit) key lists.

    Keys present in the Phase-1 checkpoint but shaped for the Phase-1
    geometry are partial or re-init candidates even if TRANSFER_KEYS-
    named (e.g. gru.weight_ih_l0); keys that appear only in the target
    (heads.6..38) can never be copied and are simply absent from both
    lists.
    """
    target = dims_for_phase2(k_subjects=k_subjects, d_in=d_in)
    full, partial, reinit = [], [], []
    for k, v in sorted(math_state.items()):
        if partial_key(k):
            partial.append(k)
        elif transferable(k, v.shape, target):
            full.append(k)
        else:
            reinit.append(k)
    return full, partial, reinit


def map_fisher_to_phase2(f_p1: dict, k_subjects: int = K_MATH + K_CLINICAL,
                         d_in: int = D_IN) -> dict:
    """Map Phase-1 diagonal Fisher arrays into Phase-2 tensor shapes.

    The Fisher is computed on the Phase-1 checkpoint (its own geometry:
    gru.weight_ih (576, 18), decode_cell.weight_ih (576, 88), heads.0..5).
    EWC in Phase-2 needs F in Phase-2 shapes with ZERO weight on the
    free clinical columns (only the math sub-grid is protected):

      gru.weight_ih_l0     (576, 117): F[:, 0:18] = p1 F (whole matrix)
      decode_cell.weight_ih (576, 220): blocks [0:18] <- p1 [0:18],
                             [117:181] <- p1 [18:82] (pooled),
                             [181:187] <- p1 [82:88] (prev math heads)
      heads.0..5           (1, 192)/(1,) identical shapes

    Non-math columns are left at zero — clinical learning is uncharged.
    """
    out = {}
    f_gru = np.asarray(f_p1["gru.weight_ih_l0"], dtype=np.float64)
    f_dec = np.asarray(f_p1["decode_cell.weight_ih"], dtype=np.float64)
    gru = np.zeros((3 * HIDDEN, d_in), dtype=np.float64)
    gru[:, 0:f_gru.shape[1]] = f_gru
    out["gru.weight_ih_l0"] = gru
    ctx = d_in + 64 + k_subjects
    dec = np.zeros((3 * HIDDEN, ctx), dtype=np.float64)
    for tlo, thi, slo, shi in COLUMN_COPY_SPEC["decode_cell.weight_ih"]:
        dec[:, tlo:thi] = f_dec[:, slo:shi]
    out["decode_cell.weight_ih"] = dec
    for i in range(K_MATH):
        out[f"heads.{i}.weight"] = np.asarray(
            f_p1[f"heads.{i}.weight"], dtype=np.float64)
        out[f"heads.{i}.bias"] = np.asarray(
            f_p1[f"heads.{i}.bias"], dtype=np.float64)
    return out


def normalize_fisher_global(f2: dict) -> dict:
    """Mean-1 normalization GLOBALLY across all math parameters.

    Splits the absolute loss-scale out of F (which is ~1e-8..1e-5 at the
    Phase-1 optimum) so the lambda exchange rate is scale-free and the
    lab sweep (lambda in {1, 10, 100}) transfers directly. The mean is
    computed over ALL math tensors together (not per-tensor) so the
    relative structural importance between gru.weight_ih, the decode
    blocks and the math heads is preserved.
    """
    total = float(sum(np.asarray(v).sum() for v in f2.values()))
    n = float(sum(np.asarray(v).size for v in f2.values()))
    mean = total / max(n, 1.0)
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"bad global Fisher mean: {mean}")
    return {k: np.asarray(v, dtype=np.float64) / mean for k, v in f2.items()}


def risk_weights(values, mask, drop_weight: float = 3.0):
    """Per-feature risk from observed-slot volatility (pie-chart Risk slice).

    risk_k = 1 + clip(var_k / median_var - 1, 0, drop_weight)

    A feature whose observed slots are volatile (high variance vs the
    window median, e.g. a Lorenz-like divergent vital) demands higher
    fidelity: its dropped-slot MSE is weighted by risk_k. Quiescent
    features (sine-like stable rhythm) get risk ~1 (Neutral slice).
    """
    var = []
    for k in range(values.shape[-1]):
        v = values[:, :, k]
        m = mask[:, :, k] > 0.5
        var.append(float(v[m].var()) if m.sum() > 2 else 1.0)
    var = np.asarray(var)
    med = max(float(np.median(var)), 1e-6)
    risk = 1.0 + np.clip(var / med - 1.0, 0.0, drop_weight)
    return risk


def risk_weighted_fidelity(pred, target, drop_mask, values, mask,
                           drop_weight: float = 3.0):
    """MSE over dropped slots, weighted per feature by risk_weights."""
    risk = risk_weights(values, mask, drop_weight)          # (K,)
    sq = (pred - target) ** 2                                # (B, W, K)
    w = np.broadcast_to(risk, sq.shape)
    w = np.where(drop_mask, w, 0.0)
    return float(np.sum(w * sq) / max(np.sum(drop_mask), 1))