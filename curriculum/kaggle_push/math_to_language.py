"""Phase 3 Gate 1a — Language stem FLOOR: Dyck-2 masked imputation from a
random core (no transfer, no economy). Establishes the achievable bar
(epochs-to-0.90-accuracy, final accuracy) that Gate 1b (crowned core +
F_total) must match or beat while holding math/clinical memory.

Architecture extension (pre-registered, approved 2026-08-20):
  - input d_in: 117 -> 120 (the language stem is a 40th triplet at
    columns 117:120 = [value, mask, delta])
  - vocabulary head Linear(hidden, V=4) SEPARATE from the 39 regression
    heads (the first non-regression head); reads the same decode state
  - dormant-slot protocol: when language is active, math+clinical
    triplets are dormant (value=0, mask=1, delta=0) — zero bandwidth,
    zero gradient on the other stems

Task: mask Dyck-2 words (bounded depth <= 4) EHR-style, impute the
dropped tokens. Metric: masked token accuracy on dropped positions,
true-Y graded (same shape as the math/clinical exams). Loss: CE on
dropped positions of the vocab logits.

CPU kernel, uniform 1e-4, no param groups (the settled N-stem scheme).
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

SEED = 42
N_WORDS = 4096
N_TEST_WORDS = 256
N_EPOCHS = 100
BATCH = 512
LR = 1e-4
ACC_FLOOR = 0.90
ACC_EVERY = 5
EXAM_KINDS = ["dyck2"]
ALIGNED = True          # aligned windows: openers visible, stack learnable

HIDDEN = 192
N_CELLS = 100
K_ACTIVE = 3
K_MATH = 6
K_CLINICAL = 33
K_SUBJECTS = K_MATH + K_CLINICAL          # 39 regression heads
D_IN = 120                                # 117 + language triplet
V = 4

# ---------------------------- vendored data contracts (see sync_vendored)

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

# ---------------------------- model (extended grid: + vocab head)

class LanguageGrid(nn.Module):
    """MathSchoolGrid extended to 120-dim with a vocabulary head.

    The 39 regression heads (6 math + 33 clinical) are unchanged; the
    vocab head Linear(hidden, V) reads the same decode state and emits
    token logits. The language triplet lives at input columns 117:120.
    """

    def __init__(self, d_in, hidden, n_cells, k, k_subjects, vocab):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.scorer = nn.Linear(hidden, n_cells)
        self.cell_block = nn.Linear(hidden, n_cells * 64)
        self.decode_cell = nn.GRUCell(d_in + 64 + k_subjects + vocab,
                                      hidden)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(k_subjects)])
        self.vocab_head = nn.Linear(hidden, vocab)
        self.k = k
        self.n_cells = n_cells
        self.k_subjects = k_subjects
        self.vocab = vocab

    def forward(self, x, return_routing=False):
        B, Wn, D = x.shape
        h, _ = self.gru(x)
        h_last = h[:, -1]
        scores = self.scorer(h_last)
        topk = torch.topk(scores, self.k, dim=1)
        votes = torch.zeros(B, self.n_cells, device=x.device)
        votes.scatter_(1, topk.indices, 1.0)
        cells = torch.relu(self.cell_block(h_last))
        cells = cells.view(B, self.n_cells, 64)
        selected = torch.gather(cells, 1,
                                topk.indices.unsqueeze(-1).expand(-1, -1, 64))
        pooled = selected.mean(dim=1)
        value = x[:, :, 0::3][:, :, :self.k_subjects]
        m = x[:, :, 1::3][:, :, :self.k_subjects]
        state = h_last.contiguous()
        prev = torch.zeros(B, self.k_subjects, device=x.device)
        prev_tok = torch.zeros(B, self.vocab, device=x.device)
        outs, vlogits = [], []
        for t in range(Wn):
            ctx = torch.cat([x[:, t], pooled, prev, prev_tok], dim=1)
            state = self.decode_cell(ctx, state)
            y_est = torch.cat(
                [head(state) for head in self.heads], dim=1)
            y = m[:, t] * value[:, t] + (1.0 - m[:, t]) * y_est
            outs.append(y)
            prev = y
            lg = self.vocab_head(state)
            vlogits.append(lg)
            prev_tok = torch.softmax(lg, dim=1)
        out = torch.stack(outs, dim=1)
        vlog = torch.stack(vlogits, dim=1)
        if return_routing:
            return out, vlog, votes
        return out, vlog


# ---------------------------- main

def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[1/4] Dyck-2 data (aligned windows)...", flush=True)
    Xtr, Ytr, Mtr = build_dyck_dataset(N_WORDS, SEED, aligned=ALIGNED)
    Xte, Yte, Mte = build_dyck_dataset(N_TEST_WORDS, SEED + 1,
                                       aligned=ALIGNED)
    Xtr = torch.tensor(embed_language_block(Xtr), dtype=torch.float32)
    Ytr = torch.tensor(Ytr, dtype=torch.long)
    Mtr = torch.tensor(Mtr, dtype=torch.float32)
    Xte = torch.tensor(embed_language_block(Xte), dtype=torch.float32)
    Yte = torch.tensor(Yte, dtype=torch.long)
    Mte = torch.tensor(Mte, dtype=torch.float32)
    print(f"  train {Xtr.shape[0]} windows, test {Xte.shape[0]}",
          flush=True)

    print("[2/4] random-core LanguageGrid (no transfer, no economy)...",
          flush=True)
    model = LanguageGrid(D_IN, HIDDEN, N_CELLS, K_ACTIVE, K_SUBJECTS, V)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {n_par:,} params | uniform lr {LR} | vocab {V}", flush=True)

    print("[3/4] training (masked CE on dropped tokens)...", flush=True)
    n = Xtr.shape[0]
    n_batches = (n + BATCH - 1) // BATCH
    curve = []
    for ep in range(N_EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(n_batches):
            idx = perm[i * BATCH: (i + 1) * BATCH]
            xb, yb, mb = Xtr[idx], Ytr[idx], Mtr[idx]
            opt.zero_grad(set_to_none=True)
            _, vlog = model(xb)
            # v2: CE on ALL positions (Phase-1 parity — train on observed +
            # dropped, grade dropped-only). Masked-only CE starved the
            # vocab head: ~50% of tokens contributed zero gradient.
            lg = vlog.reshape(-1, V)
            tg = yb.reshape(-1)
            loss = nn.functional.cross_entropy(lg, tg)
            loss.backward()
            opt.step()
            tot += float(loss)
        note = ""
        if (ep + 1) % ACC_EVERY == 0 or ep == N_EPOCHS - 1:
            with torch.no_grad():
                _, vlog = model(Xte)
            cons = stack_consistency(vlog.numpy(), Yte.numpy(),
                                     Mte.numpy())
            acc = token_accuracy(vlog.numpy(), Yte.numpy(), Mte.numpy())
            curve.append({"epoch": ep + 1, "cons": cons, "acc": acc})
            note = f" CONS {cons:.4f} ACC {acc:.4f}"
        print(f"  ep {ep:3d} loss {tot / n_batches:9.4f}{note}", flush=True)

    print("[4/4] eval...", flush=True)
    with torch.no_grad():
        _, vlog = model(Xte)
    cons = stack_consistency(vlog.numpy(), Yte.numpy(), Mte.numpy())
    acc = token_accuracy(vlog.numpy(), Yte.numpy(), Mte.numpy())
    epochs_to_90 = next((c["epoch"] for c in curve if c["cons"] >= 0.90),
                        None)
    ok = cons >= ACC_FLOOR
    print(f"  stack consistency {cons:.4f} (floor {ACC_FLOOR}) | "
          f"exact-token {acc:.4f} (reference)", flush=True)
    print("  VERDICT:", "PASS" if ok else "FAIL", flush=True)

    torch.save(model.state_dict(), "/kaggle/working/lang_rand.pt")
    report = {
        "gate": "1a",
        "aligned": ALIGNED,
        "final_consistency": cons,
        "final_accuracy": acc,
        "epochs_to_0.90": epochs_to_90,
        "acc_curve": curve,
        "params": n_par,
        "verdict": "PASS" if ok else "FAIL",
        "seconds": round(time.time() - t0, 1),
    }
    with open("/kaggle/working/gate1a_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {report['seconds']}s | verdict {report['verdict']}",
          flush=True)


if __name__ == "__main__":
    main()
