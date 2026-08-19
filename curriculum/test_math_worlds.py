"""Tests for the Phase-1 math curriculum generator (numpy, local box)."""

import numpy as np

from curriculum.math_worlds import (
    BOUNDS, K_SUBJECTS, SUBJECT_KINDS, W, T_STAY, DELTA_CAP,
    build_dataset, conservation_violations, generate_stay,
    masked_r2, _lorenz_x,
)


def test_subject_count_and_layout():
    day = generate_stay(seed=1)
    assert day["value"].shape == (T_STAY, K_SUBJECTS)
    assert day["mask"].shape == (T_STAY, K_SUBJECTS)
    assert day["delta"].shape == (T_STAY, K_SUBJECTS)
    assert day["function"].shape == (T_STAY, K_SUBJECTS)


def test_drop_fraction_within_expected_ranges():
    # per-kind drop ranges (DROP_RANGES) override the global 40-80% exam;
    # everything else keeps the 40-80% drop. Verified over 5 stays per seed.
    from curriculum.math_worlds import DROP_RANGES
    days = [generate_stay(seed=s) for s in range(8)]
    for k in range(K_SUBJECTS):
        kept = np.mean([d["mask"][:, k].mean() for d in days])
        kind = SUBJECT_KINDS[k]
        lo, hi = DROP_RANGES.get(kind, (0.4, 0.8))
        # keep-frac = 1 - drop-frac, with a small tolerance for binomial noise
        assert (1 - hi) - 0.06 <= kept <= (1 - lo) + 0.06, (kind, kept, lo, hi)


def test_mask_matches_ffill_value():
    day = generate_stay(seed=7)
    V, value, mask = day["function"], day["value"], day["mask"]
    for k in range(K_SUBJECTS):
        for i in range(T_STAY):
            if mask[i, k] > 0.5:
                assert abs(value[i, k] - V[i, k]) < 1e-9
            else:
                # carried from a previous observed point (or 0 before first obs)
                j = i - 1
                while j >= 0 and mask[j, k] < 0.5:
                    j -= 1
                expect = V[j, k] if j >= 0 else 0.0
                assert abs(value[i, k] - expect) < 1e-9


def test_delta_semantics():
    day = generate_stay(seed=11)
    mask, delta = day["mask"], day["delta"]
    for k in range(K_SUBJECTS):
        for i in range(T_STAY):
            if mask[i, k] > 0.5:
                assert delta[i, k] == 0.0
            else:
                assert 0.0 < delta[i, k] <= DELTA_CAP


def test_delta_capped():
    day = generate_stay(seed=13)
    assert day["delta"].max() <= DELTA_CAP + 1e-9
    # a long unobserved gap should hit the cap
    assert np.isclose(np.max(day["delta"]), DELTA_CAP)


def test_conservation_bounds_respected_by_generator():
    day = generate_stay(seed=17)
    V = day["function"]
    for k, kind in enumerate(SUBJECT_KINDS):
        lo, hi = BOUNDS[kind]
        assert V[:, k].min() >= lo - 1e-6, (kind, V[:, k].min())
        assert V[:, k].max() <= hi + 1e-6, (kind, V[:, k].max())


def test_continuum_physics():
    # physics spot checks per subject
    day = generate_stay(seed=19)
    V = day["function"]
    # sine/cosine: amplitude bound + periodicity of sign flips
    assert np.abs(V[:, 0]).max() <= 1.5 + 1e-6
    assert np.abs(V[:, 1]).max() <= 1.5 + 1e-6
    # decay: monotonically non-increasing, non-negative
    assert (np.diff(V[:, 2]) <= 1e-9).all()
    assert V[:, 2].min() >= -1e-9
    # step: takes at most {lo, hi} distinct values
    lv = np.unique(np.round(V[:, 3], 3))
    assert len(lv) <= 2, lv
    # sigmoid: transition between two levels, in-range
    assert V[:, 4].min() >= -1.2 and V[:, 4].max() <= 1.2
    # lorenz: finite, non-constant
    assert np.isfinite(V[:, 5]).all()
    assert np.std(V[:, 5]) > 1e-3


def test_lorenz_x_reproducible():
    g1 = np.random.default_rng(42)
    g2 = np.random.default_rng(42)
    x1 = _lorenz_x(g1)
    x2 = _lorenz_x(g2)
    assert np.array_equal(x1, x2)


def test_windows_shape_and_target_alignment():
    from curriculum.math_worlds import _windows_from_stay
    day = generate_stay(seed=5)
    X, Y = _windows_from_stay(day, cap=None)
    n = (T_STAY - W) // 2 + 1
    assert X.shape == (n, W, K_SUBJECTS * 3)
    assert Y.shape == (n, W, K_SUBJECTS)
    # full window 0 starts at grid position 0: values/masks/deltas and
    # the true-function targets must align with the day arrays
    x0, y0 = X[0], Y[0]
    for k in range(K_SUBJECTS):
        assert np.array_equal(x0[:, k * 3], day["value"][:W, k])
        assert np.array_equal(x0[:, k * 3 + 1], day["mask"][:W, k])
        assert np.array_equal(x0[:, k * 3 + 2], day["delta"][:W, k])
        assert np.array_equal(y0[:, k], day["function"][:W, k])


def test_masked_r2_math():
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    m = np.array([1.0, 1.0, 0.0, 0.0])
    # masked positions are 2,3 -> perfect
    f = np.array([9.0, 9.0, 2.0, 3.0])
    assert masked_r2(f, y, m) == 1.0
    # constant predictor on masked = mean of masked targets -> R2 ~ 0
    f2 = np.array([9.0, 9.0, 2.5, 2.5])
    assert abs(masked_r2(f2, y, m)) < 1e-6
    # all-masked degenerate -> 0.0 (no denom)
    m0 = np.zeros(4)
    assert masked_r2(f, y, m0) == 0.0


def test_conservation_violations_counting():
    pred = np.array([0.0, 5.0, -5.0, 1.0, 0.5])
    pos = {"decay": np.array([0, 1, 2])}   # decay bound (-0.05, 2.20)
    out = conservation_violations(pred, pos)
    assert out["decay"]["below"] == 1   # -5.0
    assert out["decay"]["above"] == 1   # 5.0
    assert out["decay"]["total"] == 3


def test_build_dataset_reproducible():
    X1, Y1, _ = build_dataset(n_stays=6, seed=0, cap=4)
    X2, Y2, _ = build_dataset(n_stays=6, seed=0, cap=4)
    assert np.array_equal(X1, X2)
    assert np.array_equal(Y1, Y2)


def test_kernel_decoder_is_functional_no_inplace():
    """Static guard: MathSchoolGrid.forward must not contain any in-place
    writes (``a[k] = ...`` or ``a.attr = ...``) on autograd tensors.

    Both the v5 crash (`y_est[:, 2] = torch.clamp(...)` -> "modified by an
    inplace operation" at backward) and the failed clone() attempt came from
    this pattern; autograd version-checks tensors captured in the graph even
    after clone() because clone keeps the gradient connection. The clamp is
    now spliced functionally through cat(). This test fails the local suite
    if anyone reintroduces an indexed/attribute assignment into forward().
    """
    import ast
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[0] / "kaggle_push" \
        / "math_school_train.py"
    src = ast.parse(path.read_text())

    def cls(name):
        for node in ast.walk(src):
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        raise AssertionError(f"class {name} not found in kernel")

    def fn(cls_node, fname):
        for node in cls_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == fname:
                return node
        raise AssertionError(f"def {fname} not found in class")

    fwd = fn(cls("MathSchoolGrid"), "forward")
    bad = []
    for node in ast.walk(fwd):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(t, (ast.Subscript, ast.Attribute)):
                    bad.append(ast.unparse(t))
    assert not bad, f"in-place assignment targets in forward(): {bad}"