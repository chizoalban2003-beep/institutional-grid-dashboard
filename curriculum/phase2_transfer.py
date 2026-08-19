"""Phase-2 transfer utilities — Physics priors -> clinical tensors (numpy).

Bridges the certified Phase-1 math grid into the MIMIC-IV clinical stem:

  Stage-1 surface (transferable, same shapes):
    gru.weight_hh_l0 / bias_ih_l0 / bias_hh_l0   recurrent dynamics
    decode_cell.weight_hh / bias_ih / bias_hh    mask-decoder recurrence
    scorer.{weight,bias}                         100-cell routing manifold
    cell_block.{weight,bias}                     cell feature library

  Stage-2-only surface (re-initialized, shapes differ):
    gru.weight_ih_l0      d_in 18 -> 117 (feature semantics differ)
    decode_cell.weight_ih ctx 88 -> 220 (K 6 -> 39 heads with the great
    heads.*               per-feature readout, one per FEATURE_NAMES)

The recurrent hidden-to-hidden matrices and the whole grid (router +
cell library) carry the math priors; input projections and per-feature
readouts must re-learn their semantics. This is the honest low-LR warm
start: transfer_group gets LR_TRANSFER=1e-5, new_group LR_NEW=1e-4.

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


def split_transfer_keys(state_dict: dict) -> tuple[list[str], list[str]]:
    """Partition a checkpoint's keys into (transferable, re-initialized).

    A key is transferable iff it is in TRANSFER_KEYS AND its shape matches
    the counterpart in the target model dict (target_keys -> shapes).
    """
    transfer, reinit = [], []
    for k in sorted(state_dict):
        if k in TRANSFER_KEYS:
            transfer.append(k)
        else:
            reinit.append(k)
    return transfer, reinit


def dims_for_phase2(k_subjects: int = 39, d_in: int = 117,
                    hidden: int = 192, n_cells: int = 100) -> dict[str, tuple]:
    """Phase-2 tensor shapes for the MathSchoolGrid architecture."""
    return {
        "gru.weight_ih_l0": (3 * hidden, d_in),
        "gru.weight_hh_l0": (3 * hidden, hidden),
        "gru.bias_ih_l0": (3 * hidden,),
        "gru.bias_hh_l0": (3 * hidden,),
        "decode_cell.weight_ih": (3 * hidden, d_in + 64 + k_subjects),
        "decode_cell.weight_hh": (3 * hidden, hidden),
        "decode_cell.bias_ih": (3 * hidden,),
        "decode_cell.bias_hh": (3 * hidden,),
        "scorer.weight": (n_cells, hidden),
        "scorer.bias": (n_cells,),
        "cell_block.weight": (n_cells * 64, hidden),
        "cell_block.bias": (n_cells * 64,),
        "heads.0.weight": (1, hidden),
        "heads.0.bias": (1,),
    }


def transferable(state_key: str, state_shape: tuple,
                 target_shapes: dict, k_subjects: int) -> bool:
    """True iff state_key can be copied into the Phase-2 model."""
    if state_key not in TRANSFER_KEYS:
        return False
    shape = target_shapes.get(state_key)
    if shape is None:
        return False
    return tuple(state_shape) == tuple(shape)


def plan_transfer(math_state: dict, k_subjects: int = 39,
                  d_in: int = 117) -> tuple[list[str], list[str]]:
    """Plan a transfer: ([copy_keys], [reinit_keys]) for a Phase-2 model.

    Keys present in the Phase-1 checkpoint but shaped for the Phase-1
    geometry are re-init candidates even if TRANSFER_KEYS-named (e.g.
    gru.weight_ih_l0); keys that appear only in the target (heads.*) can
    never be copied and are simply absent from both lists.
    """
    target = dims_for_phase2(k_subjects=k_subjects, d_in=d_in)
    copy_keys, reinit_keys = [], []
    for k, v in sorted(math_state.items()):
        if transferable(k, v.shape, target, k_subjects):
            copy_keys.append(k)
        else:
            reinit_keys.append(k)
    return copy_keys, reinit_keys


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