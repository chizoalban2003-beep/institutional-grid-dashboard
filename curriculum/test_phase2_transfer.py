"""Tests for the Phase-2 transfer utilities (numpy, no torch needed)."""

import numpy as np
import pytest

from curriculum.phase2_transfer import (
    COLUMN_COPY_SPEC, TRANSFER_KEYS, dims_for_phase2, partial_key,
    plan_transfer, risk_weights, risk_weighted_fidelity, split_transfer_keys,
    transferable,
)


def _fake_state(shape_map):
    out = {}
    for k in TRANSFER_KEYS:
        shape = shape_map.get(k)
        if shape is not None:
            out[k] = np.zeros(shape)
    out["gru.weight_ih_l0"] = np.zeros(shape_map["gru.weight_ih_l0"])
    out["decode_cell.weight_ih"] = np.zeros(shape_map["decode_cell.weight_ih"])
    return out


def test_transfer_keys_partition_phase1_geometry():
    # Phase-1 state dict: d_in=18, K=6 heads
    p1 = dims_for_phase2(k_subjects=6, d_in=18)
    p1["gru.weight_ih_l0"] = (576, 18)
    p1["decode_cell.weight_ih"] = (576, 88)
    state = _fake_state(p1)
    full, partial, reinit = plan_transfer(state, k_subjects=39, d_in=117)
    # every recurrent/routing key + the 6 math heads copy whole
    assert set(full) == (TRANSFER_KEYS - {"gru.weight_ih_l0",
                                          "decode_cell.weight_ih"})
    # the two input projections are partial-column copies, NOT reinit
    assert set(partial) == {"gru.weight_ih_l0", "decode_cell.weight_ih"}
    # nothing leaks into reinit: the Phase-1 dict has no other keys and
    # heads.6..38 do not exist in it at all (math heads are PERMANENT)
    assert reinit == []


def test_transferable_shape_gate():
    tgt = dims_for_phase2()
    assert transferable("scorer.weight", (100, 192), tgt)
    assert not transferable("gru.weight_ih_l0", (576, 18), tgt)
    assert transferable("heads.0.weight", (1, 192), tgt)   # permanent math head
    assert not transferable("heads.6.weight", (1, 192), tgt)  # clinical: new
    assert not transferable("unknown.key", (1,), tgt)


def test_split_transfer_keys_names_only():
    sd = {"scorer.weight": None, "gru.weight_ih_l0": None,
          "heads.5.bias": None}
    t, r = split_transfer_keys(sd)
    assert t == ["heads.5.bias", "scorer.weight"]
    assert r == ["gru.weight_ih_l0"]


def test_column_copy_spec_is_well_formed():
    # every (tgt_lo, tgt_hi, src_lo, src_hi) must be ordered and sane
    for key, blocks in COLUMN_COPY_SPEC.items():
        assert isinstance(blocks, list) and blocks
        for lo, hi, slo, shi in blocks:
            assert 0 <= lo < hi
            assert 0 <= slo < shi
    # the pooled block must land on the Phase-2 pooled position
    blocks = COLUMN_COPY_SPEC["decode_cell.weight_ih"]
    assert (117, 181, 18, 82) in blocks
    assert (181, 187, 82, 88) in blocks


def test_partial_key_detects_spec():
    assert partial_key("gru.weight_ih_l0")
    assert partial_key("decode_cell.weight_ih")
    assert not partial_key("scorer.weight")
    assert not partial_key("heads.0.weight")


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