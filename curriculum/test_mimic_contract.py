"""Numpy tests for the MIMIC-contract reference generators."""

import numpy as np
import pytest

from curriculum.mimic_contract import (
    CLINICAL_SLOT, D_IN, DROP_6_LABS, FEATURE_NAMES, K_CLINICAL,
    K_MATH, K_SUBJECTS, DEMOGRAPHICS_FROM, VITALS, W, clinical_windows,
    embed_math_exam,
)


def _win_X(X):
    """Slice the clinical triplet block out of a (B, W, D_IN) tensor."""
    return X[:, :, CLINICAL_SLOT:]


def test_drop_6_labs_leave_33_clinical_features():
    assert len(FEATURE_NAMES) == K_CLINICAL == 33
    assert K_SUBJECTS == K_MATH + K_CLINICAL == 39
    assert D_IN == K_SUBJECTS * 3 == 117
    assert DROP_6_LABS == {"AST", "Alkalinephos", "Bilirubin_direct",
                           "Bilirubin_total", "TroponinI", "Fibrinogen"}


def test_clinical_stay_feature_span():
    X, Y, M = clinical_windows(8, 3)
    assert np.all(np.isfinite(X))
    assert np.abs(Y).max() <= 4.5
    # (168-14)//2 = 77 windows minus cap 24 -> exactly 8*24 = 192
    assert X.shape == (8 * 24, W, D_IN)
    assert Y.shape == (8 * 24, W, K_SUBJECTS)
    assert M.shape == (8 * 24, W, K_SUBJECTS)


def test_math_slots_dormant_by_construction():
    X, Y, M = clinical_windows(4, 7)
    # math triplet block (0:CLINICAL_SLOT): value 0, mask 1 (observed),
    # delta 0 — only the clinical block carries signal
    assert np.all(X[:, :, :CLINICAL_SLOT][:, :, 0::3] == 0.0)
    assert np.all(X[:, :, :CLINICAL_SLOT][:, :, 1::3] == 1.0)
    assert np.all(X[:, :, :CLINICAL_SLOT][:, :, 2::3] == 0.0)
    # math heads: target 0, observation flag 1 -> hard copy pins to 0
    assert np.all(Y[:, :, :K_MATH] == 0.0)
    assert np.all(M[:, :, :K_MATH] == 1.0)


def test_clinical_contract_triplet_semantics():
    X, Y, M = clinical_windows(4, 7)
    v = _win_X(X)[:, :, 0::3]   # ffill'd value
    m = _win_X(X)[:, :, 1::3]   # observed flag
    d = _win_X(X)[:, :, 2::3]   # hours since last observation
    Yc = Y[:, :, K_MATH:]
    Mc = M[:, :, K_MATH:]
    assert np.allclose(np.round(m), Mc)
    # ffilled value must equal the true value at observed slots
    assert np.allclose(np.where(Mc > 0.5, v, 0.0),
                       np.where(Mc > 0.5, Yc, 0.0))
    # delta cap respected
    assert d.max() <= 24.0
    # delta == 0 exactly at observed slots
    assert np.allclose(np.where(Mc > 0.5, d, 0.0),
                       np.where(Mc > 0.5, np.zeros_like(d), 0.0))
    # demographics always observed
    assert np.all(M[:, :, K_MATH + DEMOGRAPHICS_FROM:] == 1.0)
    # dfill and mask sensible ranges
    assert 0.15 <= 1 - m.mean() <= 0.75 + 1e-6


def test_clinical_causal_ffill_no_lookahead():
    X, Y, M = clinical_windows(2, 11)
    v, m = _win_X(X)[:, :, 0::3], _win_X(X)[:, :, 1::3]
    Yc = Y[:, :, K_MATH:]
    for b in range(X.shape[0]):
        for k in range(K_CLINICAL):
            last_val = None  # value of the most recently observed slot
            for t in range(W):
                if m[b, t, k] > 0.5:
                    assert abs(v[b, t, k] - Yc[b, t, k]) < 1e-6
                    last_val = Yc[b, t, k]
                elif last_val is None:
                    assert abs(v[b, t, k] - Yc[b, 0, k]) < 1e-6  # cold start
                else:
                    assert abs(v[b, t, k] - last_val) < 1e-6  # causal ffill


def _fake_exam(n, seed=0):
    """(n, W, 3) exam windows with per-row kind cycling 0..5."""
    g = np.random.default_rng(seed)
    X3 = np.zeros((n, W, 3), dtype=np.float32)
    kinds = np.arange(n) % K_MATH
    X3[:, :, 0] = g.normal(size=(n, W)).astype(np.float32)  # value
    X3[:, :, 1] = (g.random((n, W)) > 0.5).astype(np.float32)  # mask
    return X3, kinds


def test_embed_math_exam_dormant_elsewhere():
    X3, kinds = _fake_exam(24)
    grid = embed_math_exam(X3, kinds)
    assert grid.shape == (24, W, D_IN)
    for b in range(24):
        k = int(kinds[b])
        lo = 3 * k
        # active triplet copied verbatim
        assert np.allclose(grid[b, :, lo:lo + 3], X3[b])
        # every other triplet stays dormant: value 0, mask 1, delta 0
        for j in range(K_SUBJECTS):
            if j == k:
                continue
            assert np.all(grid[b, :, 3 * j] == 0.0)
            assert np.all(grid[b, :, 3 * j + 1] == 1.0)
            assert np.all(grid[b, :, 3 * j + 2] == 0.0)


def test_embed_math_exam_vectorized_per_row_kind():
    X3, kinds = _fake_exam(12, seed=3)
    grid = embed_math_exam(X3, kinds)
    # brute-force reference
    ref = np.zeros_like(grid)
    ref[:, :, 1::3] = 1.0
    for b in range(12):
        lo = 3 * int(kinds[b])
        ref[b, :, lo:lo + 3] = X3[b]
    assert np.array_equal(grid, ref)


def test_embed_math_exam_validates_kinds():
    X3, kinds = _fake_exam(6)
    with pytest.raises(ValueError):
        embed_math_exam(X3, np.arange(5))  # wrong length
    with pytest.raises(ValueError):
        embed_math_exam(X3, np.array([6, 0, 0, 0, 0, 0]))  # kind out of range
    with pytest.raises(ValueError):
        embed_math_exam(X3, kinds.reshape(6, 1))  # not 1-D
