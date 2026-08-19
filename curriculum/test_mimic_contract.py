"""Numpy tests for the MIMIC-contract reference generators."""

import numpy as np

from curriculum.mimic_contract import (
    DEMOGRAPHICS_FROM, K, VITALS, W, clinical_windows, continuum,
    exam_windows, lorenz_x, masked_r2,
)


def test_exam_windows_shapes_and_kinds():
    X, kinds = exam_windows(192, 42)
    assert X.shape == (192, W, 3)
    assert len(kinds) == 192
    assert sorted(set(kinds)) == [0, 1, 2, 3, 4, 5]  # all 6 kinds present


def test_exam_windows_single_channel_consistency():
    # channel 0 of each triplet is the kind's own series; observed flags
    # should leave a real signal after dropping
    X, kinds = exam_windows(192, 42)
    for i in range(0, 192, 6):
        assert X[i, :, 2].std() == 0.0  # delta column unused
        assert X[i, :, 0].std() > 0.01  # value column non-degenerate


def test_lorenz_stays_bounded():
    g = np.random.default_rng(0)
    xs = lorenz_x(g)
    assert xs.shape == (2000,)
    assert np.all(np.isfinite(xs))


def test_continuum_bounded_and_deterministic():
    g1 = np.random.default_rng(1)
    g2 = np.random.default_rng(1)
    a, b = continuum(g1), continuum(g2)
    assert np.allclose(a, b)
    assert a.shape == (256, 6)
    assert np.all(np.isfinite(a))


def test_clinical_stay_feature_span():
    g = np.random.default_rng(3)
    X, Y, M = clinical_windows(8, 3)
    # vitals roughly unit z; labs clipped to [-4, 4]
    assert np.all(np.isfinite(X))
    assert np.abs(Y).max() <= 4.5
    # (168-14)//2 = 77 windows minus cap 24 -> exactly 8*24 = 192
    assert X.shape == (8 * 24, W, K * 3)


def test_clinical_contract_triplet_semantics():
    X, Y, M = clinical_windows(4, 7)
    v = X[:, :, 0::3]   # ffill'd value
    m = X[:, :, 1::3]   # observed flag
    d = X[:, :, 2::3]   # hours since last observation
    assert np.allclose(np.round(m), M)
    # ffilled value must equal the true value at observed slots
    assert np.allclose(np.where(M > 0.5, v, 0.0),
                       np.where(M > 0.5, Y, 0.0))
    # delta cap respected
    assert d.max() <= 24.0
    # delta == 0 exactly at observed slots
    assert np.allclose(np.where(M > 0.5, d, 0.0),
                       np.where(M > 0.5, np.zeros_like(d), 0.0))
    # demographics always observed
    assert np.all(M[:, :, DEMOGRAPHICS_FROM:] == 1.0)
    # dfill and mask sensible ranges
    assert 0.15 <= 1 - m.mean() <= 0.75 + 1e-6


def test_clinical_causal_ffill_no_lookahead():
    X, Y, M = clinical_windows(2, 11)
    v, m = X[:, :, 0::3], X[:, :, 1::3]
    for b in range(X.shape[0]):
        for k in range(K):
            last_val = None  # value of the most recently observed slot
            for t in range(W):
                if m[b, t, k] > 0.5:
                    assert abs(v[b, t, k] - Y[b, t, k]) < 1e-6
                    last_val = Y[b, t, k]
                elif last_val is None:
                    assert abs(v[b, t, k] - Y[b, 0, k]) < 1e-6  # cold start
                else:
                    assert abs(v[b, t, k] - last_val) < 1e-6  # causal ffill


def test_masked_r2_sanity():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(10, W))
    r2 = masked_r2(y, y, np.ones_like(y))
    assert r2 == 1.0  # perfect reconstruction
    r2b = masked_r2(np.zeros_like(y), y, np.ones_like(y))
    # zero predictor has R2 <= 0 vs the mean-irreducible denominator
    assert r2b <= 0.0