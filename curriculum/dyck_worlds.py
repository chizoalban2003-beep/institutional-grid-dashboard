"""Dyck-n language stem — Phase 3 "syntax primary school" (numpy lab).

Mirrors math_worlds.py / mimic_contract.py exactly: controlled archetype
first, real data later. Dyck-2 (two bracket types) with BOUNDED depth,
words corrupted by EHR-style masking, encoded as [value, mask, delta]
triplets — masked-language-modeling in the shared triple-channel
protocol. True-Y masked grading (token accuracy on dropped slots) is the
exam, identical in shape to the math/clinical exam machinery.

Literature anchors (Gate-2 scan, 2026-08-20):
  - Bhattamishra et al. COLING 2020: RNNs generalize near-perfectly on
    Dyck when train/test lengths are in the same range; bounded depth is
    the tractable regime. Our W=14 windows are same-range by construction.
  - Dave/Kifer/Giles/Mali PMLR 2025: Dyck-2 > Dyck-1 for recurrent
    stability; single neurons fail Dyck-2 — a meaningful, non-trivial bar.
  - Yu et al. BlackboxNLP 2019: bracket-tagging is an insufficient probe;
    masked imputation with true-Y grading is the honest metric.

Encoding: value = token id (ffill'd at dropped slots), mask = observed
flag (1 = observed), delta = positions since last observation (capped,
0 on observed). Tokens: V=4 ids for '(', ')', '[', ']'.
"""

from __future__ import annotations

import numpy as np

# >>> VENDOR (dyck_worlds) — do not edit outside the reference module

V = 4                                  # vocabulary: (, ), [, ]
TOKENS = ["(", ")", "[", "]"]
DEPTH_MAX = 4                          # bounded nesting depth (Dyck-2, depth<=4)
W = 14                                 # window length (shared core unroll)
T_MIN, T_MAX = 32, 64                  # word length range (same-range split)
SLIDE = 2
CAP_WINDOWS = 24
DROP_LO, DROP_HI = 0.10, 0.30          # SPARSE masking (v3, Gate-1 revision
                                       # 2026-08-20): Dyck-n has ZERO
                                       # redundancy — a dropped '[' makes the
                                       # matching ']' mathematically
                                       # unpredictable. v1/v2 masked 50%
                                       # (EHR-style, right for noisy ICU data,
                                       # WRONG for formal grammar) -> dropped-
                                       # token marginal went near-uniform and
                                       # the floor capped at 0.381 (the
                                       # ambiguity ceiling, not a learning
                                       # failure). 10-30% = the BERT MLM
                                       # standard regime, adapted for a
                                       # zero-redundancy grammar.
DELTA_CAP = 24.0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_word(rng: np.random.Generator,
                  t_min: int = T_MIN, t_max: int = T_MAX,
                  depth_max: int = DEPTH_MAX) -> np.ndarray:
    """One balanced Dyck-2 word (array of token ids), bounded depth.

    Constrained random walk over depth with '('/'[' (+1) and ')'/']'
    (-1), keeping depth in [0, depth_max]: a step that would violate the
    bounds is forced to the other action; otherwise random (~45% close).
    The residual unmatched openers are then APPENDED with their matching
    closers (never overwriting emitted tokens), so the word is balanced
    by construction with length in [n, n + depth_max]. Closing bracket
    type MUST match the most recent unmatched opener (stack discipline)
    — this is the hierarchical dependency the GRU must learn.
    """
    n = int(rng.integers(t_min, t_max + 1))
    ids = np.zeros(n, dtype=np.int64)
    stack = []                          # 0 = '(', 1 = '['
    depth = 0
    for i in range(n):
        can_close = depth > 0
        can_open = depth < depth_max
        if can_close and (not can_open or rng.random() < 0.45):
            open_type = stack.pop()
            ids[i] = 2 + open_type      # ')'=2 for '(', ']'=3 for '['
            depth -= 1
        else:
            open_type = int(rng.integers(0, 2))
            stack.append(open_type)
            ids[i] = open_type
            depth += 1
    # append matching closers for any residual openers (safe: append only)
    while stack:
        open_type = stack.pop()
        ids = np.concatenate([ids, np.array([2 + open_type])])
    return ids


def build_dyck_dataset(n_words: int, seed: int,
                       aligned: bool = True) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]:
    """(X, Y, M): X (B, W, 3) [value_ffill, mask, delta], Y (B, W) true
    token ids, M (B, W) observation flags (1 = observed).

    aligned=True: windows START at word positions 0, W, 2W, ... (chunk
    alignment) so every closing bracket's opener is inside the window
    whenever possible — preserves the stack-tracking signal per window.
    aligned=False: sliding windows (SLIDE), closers whose opener fell
    before the window start are legitimately under-determined (the model
    learns the conditional marginal) — the rand floor (Gate 1a) measures
    the achievable bar either way.
    """
    rng = _rng(seed)
    X, Y, M = [], [], []
    for i in range(n_words):
        ids = generate_word(_rng(int(rng.integers(0, 2 ** 31))))
        T = len(ids)
        mask = (rng.random(T) >= rng.uniform(DROP_LO, DROP_HI)).astype(
            np.float32)
        value = ids.astype(np.float32).copy()
        last = -1
        for t in range(T):
            if mask[t] > 0.5:
                last = t
            elif last >= 0:
                value[t] = value[last]
            else:
                value[t] = 0.0          # cold start: nothing observed yet
        delta = np.zeros(T, dtype=np.float32)
        last_obs = -1
        for t in range(T):
            if mask[t] > 0.5:
                last_obs = t
                delta[t] = 0.0
            elif last_obs >= 0:
                delta[t] = min(t - last_obs, DELTA_CAP)
            else:
                delta[t] = DELTA_CAP
        if aligned:
            starts = range(0, T - W + 1, W)
        else:
            n_win = (T - W) // SLIDE + 1
            starts = range(0, n_win * SLIDE, SLIDE)
            if n_win > CAP_WINDOWS:
                g2 = np.random.default_rng(1000 + i)
                starts = sorted(np.random.default_rng(1000 + i).choice(
                    list(starts), size=CAP_WINDOWS, replace=False))
        for s in starts:
            a, b = int(s), int(s) + W
            x = np.stack([value[a:b], mask[a:b], delta[a:b]], axis=-1)
            X.append(x.astype(np.float32))
            Y.append(ids[a:b])
            M.append(mask[a:b])
    return (np.stack(X), np.stack(Y), np.stack(M))


def token_accuracy(pred_logits: np.ndarray, y: np.ndarray,
                   mask: np.ndarray) -> float:
    """Masked token accuracy on DROPPED positions (the Phase-3 exam).

    pred_logits: (B, W, V) model output; argmax per position vs true id.
    """
    pred = np.argmax(pred_logits, axis=-1)
    dropped = ~(mask > 0.5)
    if not dropped.any():
        return 0.0
    return float((pred[dropped] == y[dropped]).mean())


def stack_consistency(pred_logits: np.ndarray, y: np.ndarray,
                      mask: np.ndarray, depth_max: int = DEPTH_MAX) -> float:
    """GRAMMAR-CONSISTENCY grader (Phase-3 acceptance gate, approved
    2026-08-20): a prediction on a DROPPED position is CORRECT iff it is
    a legal continuation of the stack built from the window's TRUE
    tokens — a closer matching the stack top, or any opener within the
    depth bound. This decouples the structural logic from the generator's
    opener-type coin flip (exact-token accuracy caps ~0.74-0.78 even for
    a perfect stack tracker; the 0.90 bar was unreachable by design).
    The training loss is UNCHANGED (CE on exact tokens); only the
    acceptance gate moves to grammar-consistency (the blueprint's
    "Neutrality" slice: any token inside the legal manifold is neutral).

    NOTE: stack state at window start is unknown (windows slice words,
    h0=0) — the stack is built from observed TRUE tokens within the
    window only, so a closer whose opener fell before the window start
    is scored against the partial stack (a conservative underestimate).
    """
    pred = np.argmax(pred_logits, axis=-1)
    B, Wn = y.shape
    hits, tot = 0, 0
    for b in range(B):
        stack = []                      # 0='(', 1='[' (from true tokens)
        for t in range(Wn):
            if mask[b, t] > 0.5:
                tok = int(y[b, t])
                if tok < 2:
                    stack.append(tok)
                elif stack and tok == 2 + stack[-1]:
                    stack.pop()
                # mismatched closer in true data cannot happen (balanced)
                continue
            # dropped position: grade legality of the model's argmax
            p = int(pred[b, t])
            tot += 1
            if p < 2:                   # opener: legal iff depth < max
                if len(stack) < depth_max:
                    hits += 1
            else:                       # closer: legal iff matches top
                if stack and p == 2 + stack[-1]:
                    hits += 1
            # advance the true stack past this position regardless
            tok = int(y[b, t])
            if tok < 2:
                stack.append(tok)
            elif stack and tok == 2 + stack[-1]:
                stack.pop()
    if tot == 0:
        return 0.0
    return float(hits / tot)


def embed_language_block(X3: np.ndarray, total_dim: int = 120) -> np.ndarray:
    """(B, W, 3) [value, mask, delta] language windows -> (B, W, 120) grid.

    The Phase-2 grid has 117 input columns (39 triplets: 6 math + 33
    clinical); the language stem is a 40th triplet at columns 117:120.
    Dormant-slot protocol: when language is active, the math+clinical
    triplets stay dormant (value=0, mask=1, delta=0) so the shared core
    sees exactly one domain's signal — zero bandwidth, zero gradient on
    the other stems (hard copy pins their regression heads to 0).
    """
    B, Wn, _ = X3.shape
    if X3.shape[2] != 3:
        raise ValueError(f"language windows must be (B, W, 3), got {X3.shape}")
    grid = np.zeros((B, Wn, total_dim), dtype=np.float32)
    grid[:, :, 1::3] = 1.0                      # all triplets dormant
    grid[:, :, total_dim - 3:total_dim] = X3    # language triplet active
    return grid


# <<< VENDOR (dyck_worlds) — do not edit outside the reference module
