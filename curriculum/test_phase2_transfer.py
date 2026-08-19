"""Tests for the Phase-2 transfer utilities (numpy, no torch needed)."""

import numpy as np
import pytest

from curriculum.phase2_transfer import (
    COLUMN_COPY_SPEC, TRANSFER_KEYS, ZERO_COLUMNS_SPEC, dims_for_phase2,
    map_fisher_to_phase2, normalize_fisher_global, partial_key,
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


def _fake_fisher_p1():
    return {
        "gru.weight_ih_l0": np.zeros((576, 18)) + 0.5,
        "decode_cell.weight_ih": np.zeros((576, 88)) + 0.25,
        **{f"heads.{i}.weight": np.zeros((1, 192)) + 0.1
           for i in range(6)},
        **{f"heads.{i}.bias": np.zeros((1,)) + 0.1 for i in range(6)},
    }


def test_fisher_map_places_math_columns_only():
    f2 = map_fisher_to_phase2(_fake_fisher_p1())
    # gru: math columns carry F, clinical columns FREE (zero)
    assert f2["gru.weight_ih_l0"].shape == (576, 117)
    assert np.all(f2["gru.weight_ih_l0"][:, :18] == 0.5)
    assert np.all(f2["gru.weight_ih_l0"][:, 18:] == 0.0)
    # decode: only the three column-copy blocks carry F
    dec = f2["decode_cell.weight_ih"]
    assert dec.shape == (576, 220)
    assert np.all(dec[:, 0:18] == 0.25)
    assert np.all(dec[:, 117:181] == 0.25)
    assert np.all(dec[:, 181:187] == 0.25)
    assert np.all(dec[:, 18:117] == 0.0)
    assert np.all(dec[:, 187:] == 0.0)
    # heads: full tensors
    for i in range(6):
        assert f2[f"heads.{i}.weight"].shape == (1, 192)
        assert f2[f"heads.{i}.bias"].shape == (1,)


def test_fisher_normalize_global_mean_one():
    f2 = map_fisher_to_phase2(_fake_fisher_p1())
    f2["heads.0.weight"] = f2["heads.0.weight"] * 100.0  # structural contrast
    fn = normalize_fisher_global(f2)
    vals = np.concatenate([np.asarray(v).ravel() for v in fn.values()])
    assert np.isclose(vals.mean(), 1.0)
    # relative importance survives: heads.0 (amplified 100x) still dominates
    h = fn["heads.0.weight"]
    g = fn["gru.weight_ih_l0"]
    assert h.mean() > 10.0 * g.mean()
    # clinical columns stay exactly zero (free to learn)
    assert np.all(fn["gru.weight_ih_l0"][:, 18:] == 0.0)


def test_fisher_normalize_rejects_degenerate():
    import pytest
    with pytest.raises(ValueError):
        normalize_fisher_global({"a": np.zeros((2, 2))})


def test_fisher_map_blocks_match_column_spec():
    f2 = map_fisher_to_phase2(_fake_fisher_p1())
    for tlo, thi, slo, shi in COLUMN_COPY_SPEC["decode_cell.weight_ih"]:
        assert np.all(f2["decode_cell.weight_ih"][:, tlo:thi] == 0.25)


def test_zero_columns_spec_covers_all_reinit_input_columns():
    # the columns zeroed must be EXACTLY the non-copied input columns:
    # gru 18:117 (all clinical), decode 18:117 (clinical window) and
    # 187:220 (prev clinical heads) — nothing else.
    gru = ZERO_COLUMNS_SPEC["gru.weight_ih_l0"]
    assert gru == [(18, 117)]
    dec = ZERO_COLUMNS_SPEC["decode_cell.weight_ih"]
    assert dec == [(18, 117), (187, 220)]
    # no overlap with the copied column blocks
    for tlo, thi, _, _ in COLUMN_COPY_SPEC["gru.weight_ih_l0"]:
        assert all(tlo >= hi or thi <= lo for lo, hi in gru)
    for tlo, thi, _, _ in COLUMN_COPY_SPEC["decode_cell.weight_ih"]:
        assert all(tlo >= hi or thi <= lo for lo, hi in dec)
    # union covers the full width (nothing silently left random-active)
    all_gru = sorted(x for b in gru for x in b)
    assert all_gru == [18, 117]
    all_dec = sorted(x for b in dec for x in b)
    assert all_dec == [18, 117, 187, 220]


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