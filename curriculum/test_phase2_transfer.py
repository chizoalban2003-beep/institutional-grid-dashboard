"""Tests for the Phase-2 transfer utilities (numpy, no torch needed)."""

import numpy as np
import pytest

from curriculum.phase2_transfer import (
    TRANSFER_KEYS, dims_for_phase2, plan_transfer, risk_weights,
    risk_weighted_fidelity, split_transfer_keys, transferable,
)


def _fake_state(shape_map):
    out = {}
    for k in TRANSFER_KEYS:
        shape = shape_map.get(k)
        if shape is not None:
            out[k] = np.zeros(shape)
    out["gru.weight_ih_l0"] = np.zeros(shape_map["gru.weight_ih_l0"])
    out["decode_cell.weight_ih"] = np.zeros(shape_map["decode_cell.weight_ih"])
    out["heads.0.weight"] = np.zeros((1, 192))
    out["heads.0.bias"] = np.zeros((1,))
    out["heads.1.weight"] = np.zeros((1, 192))
    out["heads.1.bias"] = np.zeros((1,))
    out["heads.2.weight"] = np.zeros((1, 192))
    out["heads.2.bias"] = np.zeros((1,))
    out["heads.3.weight"] = np.zeros((1, 192))
    out["heads.3.bias"] = np.zeros((1,))
    out["heads.4.weight"] = np.zeros((1, 192))
    out["heads.4.bias"] = np.zeros((1,))
    out["heads.5.weight"] = np.zeros((1, 192))
    out["heads.5.bias"] = np.zeros((1,))
    return out


class _A:
    shape = (0,)

    def __init__(self, shape):
        self.shape = tuple(shape)


def test_transfer_keys_partition_phase1_geometry():
    # Phase-1 state dict: d_in=18, K=6 heads
    p1 = dims_for_phase2(k_subjects=6, d_in=18)
    p1["heads.0.weight"] = (1, 192)
    p1["heads.0.bias"] = (1,)
    state = _fake_state(p1)
    copy_, reinit = plan_transfer(state, k_subjects=39, d_in=117)
    assert set(copy_) == (TRANSFER_KEYS - {"gru.weight_ih_l0",
                                           "decode_cell.weight_ih"})
    # input projections + all 6 phase-1 heads must re-init
    assert "gru.weight_ih_l0" in reinit
    assert "decode_cell.weight_ih" in reinit
    assert all(k.startswith("heads.") for k in reinit
               if k.startswith("heads."))
    assert len(reinit) == 2 + 12  # 2 input proj + 6 heads x {weight,bias}


def test_transferable_shape_gate():
    tgt = dims_for_phase2()
    assert transferable("scorer.weight", (100, 192), tgt, 39)
    assert not transferable("gru.weight_ih_l0", (576, 18), tgt, 39)
    assert not transferable("heads.0.weight", (1, 192), tgt, 39)
    assert not transferable("unknown.key", (1,), tgt, 39)


def test_split_transfer_keys_names_only():
    sd = {"scorer.weight": None, "gru.weight_ih_l0": None}
    t, r = split_transfer_keys(sd)
    assert t == ["scorer.weight"]
    assert r == ["gru.weight_ih_l0"]


def test_risk_weights_flags_volatile_feature():
    rng = np.random.default_rng(0)
    values = np.zeros((20, 14, 6))
    mask = np.ones((20, 14, 6))
    # feature 3 is volatile (Lorenz-like) -> risk must jump
    values[:, :, 3] = rng.normal(0, 5.0, (20, 14))
    risk = risk_weights(values, mask)
    assert risk[3] > risk[0]
    assert np.all(risk >= 1.0)
    assert risk[3] <= 4.0  # capped at 1 + drop_weight


def test_risk_weighted_fidelity_honors_mask_and_weight():
    rng = np.random.default_rng(1)
    pred = rng.normal(0, 1, (12, 14, 4))
    tgt = rng.normal(0, 1, (12, 14, 4))
    values = rng.normal(0, 1.0, (12, 14, 4))
    values[:, :, 0] = rng.normal(0, 9.0, (12, 14))  # volatile feature 0
    mask = np.ones((12, 14, 4))
    # isolate the weighting: only feature 0 contributes
    dm = np.zeros((12, 14, 4), dtype=bool)
    dm[:, :, 0] = True
    fid = risk_weighted_fidelity(pred, tgt, dm, values, mask)
    plain = float(((pred - tgt) ** 2)[dm].mean())
    assert fid >= 3.9 * plain  # risk_0 saturated at 1 + drop_weight = 4
    # without the volatile feature the risk would be ~1 -> fid ~ plain
    values2 = values.copy()
    values2[:, :, 0] = rng.normal(0, 0.1, (12, 14))
    fid2 = risk_weighted_fidelity(pred, tgt, dm, values2, mask)
    assert abs(fid2 / plain - 1.0) < 0.02