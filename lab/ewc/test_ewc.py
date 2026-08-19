"""Numpy tests for the EWC elasticity lab (plasticity economy).

Pins the analytic fixed point, the gradient identity, and the three
economic regimes (forgotten / elastic / hard-lock-like) so a change in
the EWC math can never silently break the Phase-2 upgrade path.
"""

import numpy as np

from lab.ewc.ewc_lab import (
    clinical_fit, exam_r2, fixed_point, make_clinical_target,
    make_fisher, make_phase1_weights,
)


def _setup(seed_fisher: int = 1, seed_clin: int = 2, drift: float = 0.2,
           frac: float = 0.5):
    ts = make_phase1_weights()
    f = make_fisher(seed_fisher)
    tc = make_clinical_target(ts, seed=seed_clin, frac=frac, drift=drift)
    return ts, f, tc


def test_fixed_point_is_analytic_blend():
    ts, f, tc = _setup()
    lam = 3.0
    th = fixed_point(ts, tc, f, lam)
    lamf = lam * f
    expected = (tc + lamf * ts) / (1.0 + lamf)
    assert np.allclose(th, expected)


def test_zero_lambda_equals_clinical_target():
    ts, f, tc = _setup()
    assert np.allclose(fixed_point(ts, tc, f, 0.0), tc)


def test_infinite_lambda_equals_phase1():
    ts, f, tc = _setup()
    assert np.allclose(fixed_point(ts, tc, f, 1e12), ts)


def test_elastic_regime_holds_exam_with_progress():
    # moderate drift: lam=10 must hold the 0.80 exam floor with > 50%
    # clinical progress (the hard lock gives 0% — this is why EWC earns
    # its keep only in the partial-drift regime)
    ts, f, tc = _setup(drift=0.2, frac=0.5)
    th = fixed_point(ts, tc, f, 10.0)
    assert exam_r2(th, ts) >= 0.80
    assert 1.0 - clinical_fit(th, ts, tc) > 0.50


def test_heavy_drift_degenerates_to_lock():
    # heavy rewrite: the floor needs lam=1000 and progress collapses
    ts, f, tc = _setup(drift=0.6, frac=0.8)
    lo = fixed_point(ts, tc, f, 10.0)
    hi = fixed_point(ts, tc, f, 1000.0)
    assert exam_r2(lo, ts) < 0.80
    assert exam_r2(hi, ts) >= 0.80
    assert 1.0 - clinical_fit(hi, ts, tc) < 0.20


def test_high_fisher_params_move_less():
    ts, f, tc = _setup()
    th = fixed_point(ts, tc, f, 10.0)
    lamf = 10.0 * f
    drift = np.abs(th - ts)
    hi = lamf >= np.quantile(lamf, 0.9)
    lo = lamf <= np.quantile(lamf, 0.1)
    assert drift[hi].mean() < drift[lo].mean()


def test_exam_perfect_on_phase1():
    ts, f, tc = _setup()
    assert exam_r2(ts, ts) == 1.0
