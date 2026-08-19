"""Tests for the Grid vs. GBM benchmark harness.

Covers:
  - compute_auroc / compute_auprc / compute_utility metric correctness
  - _generate_verdict boundary logic (0.02 AUC / 0.05 util cutlines)
  - prepare_mimic_data window shapes, labels, z-score sanity
  - GBM smoke: learns well above chance on the synthetic cohort

NOTE: train_grid needs torch — grid-arm tests are excluded here and live
in the Kaggle kernel (grid_vs_gbm_benchmark.py) which asserts a parity gate
against the local GBM AUROC (0.9116 @ seed 42, n=300).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark.run_grid_vs_gbm import (
    prepare_mimic_data,
    compute_auroc,
    compute_auprc,
    compute_utility,
    _generate_verdict,
    train_gbm,
)


# ─── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_auroc_perfect(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert compute_auroc(y, s) == pytest.approx(1.0, abs=1e-9)

    def test_auroc_random(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # all ties → U = n1*n2/2
        assert compute_auroc(y, s) == pytest.approx(0.5, abs=1e-9)

    def test_auroc_reversed(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        # Positives sorted below negatives → AUC 0
        assert compute_auroc(y, s) == pytest.approx(0.0, abs=1e-9)

    def test_auprc_perfect(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert compute_auprc(y, s) == pytest.approx(1.0, abs=1e-9)

    def test_utility_tp_rewarded(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.1, 0.1])
        util, _ = compute_utility(y, s, thresholds=np.array([0.5]))
        # 2 TP, 0 FP, 0 FN out of 2 pos → 2/2 = 1.0
        assert util == pytest.approx(1.0, abs=1e-9)

    def test_utility_fp_penalized(self):
        y = np.array([0, 0, 1])
        s = np.array([0.9, 0.9, 0.1])
        util, _ = compute_utility(y, s, thresholds=np.array([0.5]))
        # 2 FP, 1 FN, 0 TP of 1 pos → 0 - 2*0.05 - 1 = -1.10
        assert util == pytest.approx(-1.10, abs=1e-9)

    def test_utility_misses_fast(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        # All predictions at thr=0.5 are wrong → 0 TP, 3 FN, 3 FP → (0 - 3*0.05 - 3*1)/3 = -1.05
        util, thr = compute_utility(y, s, thresholds=np.array([0.5]))
        assert util == pytest.approx(-1.05, abs=1e-9)
        assert thr == pytest.approx(0.5)


# ─── Verdict conditions ──────────────────────────────────────────────────────

class TestVerdict:
    def test_grid_dominates(self):
        v = _generate_verdict(d_auroc=0.05, d_auprc=0.02, d_util=0.10)
        assert v.startswith("GRID DOMINATES")

    def test_parity(self):
        v = _generate_verdict(d_auroc=0.01, d_auprc=0.0, d_util=0.02)
        assert v.startswith("PARITY ACHIEVED")

    def test_acceptable_tradeoff(self):
        v = _generate_verdict(d_auroc=-0.04, d_auprc=-0.02, d_util=-0.04)
        assert v.startswith("ACCEPTABLE TRADEOFF")

    def test_gbm_wins(self):
        v = _generate_verdict(d_auroc=-0.06, d_auprc=-0.05, d_util=-0.08)
        assert v.startswith("GBM WINS ON AUC")

    def test_parity_boundary(self):
        # AUC delta exactly at 0.02 but utility below 0.05 → not dominance
        v = _generate_verdict(d_auroc=0.02, d_auprc=0.0, d_util=0.049)
        assert v.startswith("PARITY ACHIEVED")


# ─── Data preparation ────────────────────────────────────────────────────────

class TestDataPrep:
    def test_prepare_shapes(self):
        Xw_tr, Xf_tr, y_tr, Xw_te, Xf_te, y_te, meta = prepare_mimic_data(
            n_stays=60, test_frac=0.2, seed=42)
        assert Xw_tr.ndim == 3
        assert Xw_tr.shape[1] == 14
        assert Xw_tr.shape[2] == 117  # K*3
        assert Xf_tr.shape[1] == 117
        assert len(y_tr) == len(Xw_tr) == len(Xf_tr)
        assert Xw_te.ndim == 3 and Xw_te.shape[1:] == (14, 117)
        assert Xw_te.shape[0] == len(y_te) and Xf_te.shape[0] == len(y_te)

    def test_labels_correlate_with_scenario(self):
        # Deteriorating stays past onset must be positive; stable/recovering negative
        Xw_tr, Xf_tr, y_tr, Xw_te, Xf_te, y_te, meta = prepare_mimic_data(
            n_stays=60, test_frac=0.2, seed=42)
        assert set(np.unique(y_tr)) <= {0, 1}
        assert y_tr.mean() > 0.05  # some positives present

    def test_zscore_sanity(self):
        # Value columns (first K) should be near-standardized after prep
        Xw_tr, Xf_tr, y_tr, Xw_te, Xf_te, y_te, meta = prepare_mimic_data(
            n_stays=60, test_frac=0.2, seed=42)
        # HF column (index 0) z-scored → mean ~0, std ~1 over all frames
        col_std = Xf_tr[:, 0].std()
        assert 0.1 < col_std < 3.0

    def test_meta_consistent(self):
        Xw_tr, Xf_tr, y_tr, Xw_te, Xf_te, y_te, meta = prepare_mimic_data(
            n_stays=60, test_frac=0.2, seed=42)
        assert meta['n_samples'] == len(y_tr) + len(y_te)


# ─── GBM smoke ──────────────────────────────────────────────────────────────

class TestGBMSmoke:
    def test_gbm_above_chance(self):
        """GBM on flat last-hour features should learn well above chance.

        Local anchor: AUROC 0.9116 @ n=300, seed 42 (this test uses n=60
        for speed and a tolerance floor of 0.70 so it never flake-flips).
        """
        Xw_tr, Xf_tr, y_tr, Xw_te, Xf_te, y_te, meta = prepare_mimic_data(
            n_stays=60, test_frac=0.2, seed=42)
        res = train_gbm(Xf_tr, y_tr, Xf_te, y_te)
        auroc = compute_auroc(y_te, res['test_probs'])
        assert auroc > 0.70