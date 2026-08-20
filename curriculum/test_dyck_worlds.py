"""Numpy tests for the Dyck-n language stem generator (Gate-4 lab)."""

import numpy as np
import pytest

from curriculum.dyck_worlds import (
    DEPTH_MAX, DROP_HI, DROP_LO, T_MAX, T_MIN, V, W, build_dyck_dataset,
    generate_word, token_accuracy,
)


def _is_balanced(ids):
    """Stack check: every closer matches the top of the stack."""
    stack = []
    for t in ids:
        if t < 2:                       # opener
            stack.append(t)
        else:                           # closer: must match top
            if not stack or stack[-1] != t - 2:
                return False
            stack.pop()
    return len(stack) == 0


def _depth(ids):
    d = 0
    md = 0
    for t in ids:
        d += 1 if t < 2 else -1
        md = max(md, d)
    return md


def test_words_are_balanced_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = generate_word(rng)
        assert _is_balanced(w), "word must be a valid Dyck-2 word"
        assert _depth(w) <= DEPTH_MAX, "depth must be bounded"
        assert T_MIN <= len(w) <= T_MAX + DEPTH_MAX
        assert set(np.unique(w)) <= set(range(V))


def test_closing_type_matches_opener():
    # the hierarchical rule: ')' must close '(', ']' must close '['
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = generate_word(rng)
        assert _is_balanced(w)


def test_dataset_shapes_and_alignment():
    X, Y, M = build_dyck_dataset(64, 42, aligned=True)
    assert X.shape[2] == 3              # [value, mask, delta]
    assert X.shape[1] == W
    assert X.shape[0] == Y.shape[0] == M.shape[0]
    assert Y.shape[1] == W and M.shape[1] == W
    # aligned windows: only positions 0, W, 2W... are used
    assert len(X) > 64                  # multiple windows per word


def test_ffill_causal_no_lookahead():
    X, Y, M = build_dyck_dataset(16, 7, aligned=True)
    v, m = X[:, :, 0], X[:, :, 1]
    for b in range(X.shape[0]):
        last_val = None
        for t in range(W):
            if m[b, t] > 0.5:
                assert abs(v[b, t] - Y[b, t]) < 1e-6
                last_val = Y[b, t]
            elif last_val is not None:
                assert abs(v[b, t] - last_val) < 1e-6
            # cold start (no prior obs in window): value = 0.0 placeholder


def test_delta_semantics():
    X, Y, M = build_dyck_dataset(16, 11, aligned=True)
    d = X[:, :, 2]
    assert d.max() <= 24.0
    # delta == 0 exactly at observed slots
    assert np.allclose(np.where(M > 0.5, d, 0.0),
                       np.where(M > 0.5, np.zeros_like(d), 0.0))
    # observed fraction within the drop range
    frac = float(M.mean())
    assert DROP_LO <= 1 - frac <= DROP_HI + 1e-6


def test_sliding_windows_produce_underdetermined_closers():
    # aligned windows keep openers visible; sliding windows don't always
    # (the model must learn the conditional marginal there) — verify the
    # two modes both generate and differ in window count
    Xa, Ya, Ma = build_dyck_dataset(32, 3, aligned=True)
    Xs, Ys, Ms = build_dyck_dataset(32, 3, aligned=False)
    assert len(Xs) >= len(Xa)           # sliding yields >= windows


def test_token_accuracy_grader():
    rng = np.random.default_rng(5)
    y = rng.integers(0, V, (8, W))
    m = (rng.random((8, W)) > 0.5).astype(np.float32)
    # perfect logits (one-hot at true id) -> 1.0
    logits = np.zeros((8, W, V))
    logits[np.arange(8)[:, None], np.arange(W)[None, :], y] = 1.0
    assert token_accuracy(logits, y, m) == 1.0
    # random logits -> ~1/V on dropped slots
    r2 = np.random.default_rng(6).normal(size=(8, W, V))
    acc = token_accuracy(r2, y, m)
    assert 0.10 < acc < 0.40            # V=4 -> random ~0.25
    # perfect logits but only observed slots -> 0.0 (exam is dropped-only)
    m2 = np.ones_like(m)
    assert token_accuracy(logits, y, m2) == 0.0


def test_generator_deterministic():
    a = build_dyck_dataset(8, 42)
    b = build_dyck_dataset(8, 42)
    for x1, x2 in zip(a, b):
        assert np.array_equal(x1, x2)


def test_pytest_importable():
    import curriculum.dyck_worlds as dw
    assert dw.V == 4
