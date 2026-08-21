#!/usr/bin/env python3
"""Phase 3 Gate-1a: Dedicated language stem on Dyck-2.

Proves a small GRU (~80k params) can learn Dyck-2 masked language modeling
at grammar-consistency >= 0.95 WITHOUT using the shared grid.

Architecture:
  tokens (B, W) → Embedding(4, 32) → GRU(128, 1) → proj(128, 64) → head(64, 4)
  z_lang = proj output (64-dim, ready for future injection into shared grid)

Training:
  Masked CE on dropped tokens only (10-30% dropout)
  Adam lr=1e-3, 100 epochs, batch 256

Evaluation:
  Grammar-consistency on 256 test words (any legal token = correct)
  Target: >= 0.95
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
V = 4                                  # vocab: (, ), [, ]
EMBED_DIM = 32
HIDDEN = 128
Z_DIM = 64                             # context vector for future injection
W = 14                                 # window size
BATCH = 256
LR = 1e-3
N_EPOCHS = 100
DROP_LO, DROP_HI = 0.10, 0.30
DEPTH_MAX = 4
N_TRAIN = 2048
N_VAL = 256
N_TEST = 256
ACC_EVERY = 5

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Dyck-2 generator (inlined from dyck_worlds.py)
# ---------------------------------------------------------------------------

def generate_word(rng, t_min=32, t_max=64, depth_max=DEPTH_MAX):
    n = int(rng.integers(t_min, t_max + 1))
    ids = np.zeros(n, dtype=np.int64)
    stack = []
    depth = 0
    for i in range(n):
        can_close = depth > 0
        can_open = depth < depth_max
        if can_close and (not can_open or rng.random() < 0.45):
            open_type = stack.pop()
            ids[i] = 2 + open_type
            depth -= 1
        else:
            open_type = int(rng.integers(0, 2))
            stack.append(open_type)
            ids[i] = open_type
            depth += 1
    while stack:
        open_type = stack.pop()
        ids = np.concatenate([ids, np.array([2 + open_type])])
    return ids


def build_dataset_aligned(n_words, seed, w=W):
    """Aligned windows: every closing bracket's opener is inside the window."""
    rng = np.random.default_rng(seed)
    Y, M = [], []
    for _ in range(n_words):
        ids = generate_word(np.random.default_rng(int(rng.integers(0, 2**31))))
        T = len(ids)
        mask = (rng.random(T) >= rng.uniform(DROP_LO, DROP_HI)).astype(np.float32)
        starts = range(0, T - w + 1, w)
        for s in starts:
            a, b = int(s), int(s) + w
            Y.append(ids[a:b])
            M.append(mask[a:b])
    return np.stack(Y), np.stack(M)


# ---------------------------------------------------------------------------
# Grammar-consistency grader (from dyck_worlds.py)
# ---------------------------------------------------------------------------

def stack_consistency(pred_logits, y, mask, depth_max=DEPTH_MAX):
    pred = np.argmax(pred_logits, axis=-1)
    B, Wn = y.shape
    hits, tot = 0, 0
    for b in range(B):
        stack = []
        for t in range(Wn):
            if mask[b, t] > 0.5:
                tok = int(y[b, t])
                if tok < 2:
                    stack.append(tok)
                elif stack and tok == 2 + stack[-1]:
                    stack.pop()
                continue
            p = int(pred[b, t])
            tot += 1
            if p < 2:
                if len(stack) < depth_max:
                    hits += 1
            else:
                if stack and p == 2 + stack[-1]:
                    hits += 1
            tok = int(y[b, t])
            if tok < 2:
                stack.append(tok)
            elif stack and tok == 2 + stack[-1]:
                stack.pop()
    return float(hits / tot) if tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Model: dedicated language stem
# ---------------------------------------------------------------------------

class LanguageStem(nn.Module):
    """Dedicated GRU encoder for Dyck-2.

    Produces z_lang (64-dim) at each timestep — the context vector that will
    be concatenated with math/clinical features for the shared grid in
    Gate-1b. For Gate-1a, we evaluate the stem's language modeling directly.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, EMBED_DIM)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN, batch_first=True)
        self.proj = nn.Linear(HIDDEN, Z_DIM)
        self.head = nn.Linear(Z_DIM, V)

    def forward(self, x):
        """x: (B, W) token indices → logits (B, W, V), z_lang (B, W, Z_DIM)."""
        emb = self.embed(x)
        h, _ = self.gru(emb)
        z = self.proj(h)
        logits = self.head(z)
        return logits, z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 3 Gate-1a: Dedicated language stem on Dyck-2")
    print(f"  seed={SEED} | stem ~{sum(p.numel() for p in LanguageStem().parameters()):,} params")
    print(f"  z_dim={Z_DIM} (context vector for future grid injection)")
    print("=" * 60, flush=True)

    print("[1/4] Building datasets (aligned windows)...", flush=True)
    Ytr, Mtr = build_dataset_aligned(N_TRAIN, SEED)
    Yva, Mva = build_dataset_aligned(N_VAL, SEED + 1)
    Yte, Mte = build_dataset_aligned(N_TEST, SEED + 2)
    print(f"  train {Ytr.shape} val {Yva.shape} test {Yte.shape}", flush=True)
    n_dropped = int((Mtr < 0.5).sum())
    print(f"  train dropped tokens: {n_dropped} / {Mtr.size} "
          f"({n_dropped / Mtr.size:.1%})", flush=True)

    print("[2/4] Building model...", flush=True)
    model = LanguageStem()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {n_par:,} params (embed {V*EMBED_DIM} + gru "
          f"{3*(EMBED_DIM+HIDDEN)*HIDDEN} + proj {HIDDEN*Z_DIM+Z_DIM} "
          f"+ head {Z_DIM*V+V})", flush=True)

    print("[3/4] Training (masked CE, adam lr=1e-3)...", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Ytr_t = torch.tensor(Ytr, dtype=torch.long)
    Mtr_t = torch.tensor(Mtr, dtype=torch.float32)
    Yva_t = torch.tensor(Yva, dtype=torch.long)
    Mva_t = torch.tensor(Mva, dtype=torch.float32)
    Yte_t = torch.tensor(Yte, dtype=torch.long)
    Mte_t = torch.tensor(Mte, dtype=torch.float32)

    n = Ytr.shape[0]
    n_batches = (n + BATCH - 1) // BATCH
    t0 = time.time()
    best_cons = 0.0

    for ep in range(N_EPOCHS):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * BATCH: (i + 1) * BATCH]
            yb = Ytr_t[idx]
            mb = Mtr_t[idx]
            # build input: observed tokens pass through, dropped get ffill
            xb = yb.clone()
            # simple left-to-right ffill for dropped slots
            for t in range(1, W):
                fill = mb[:, t] < 0.5
                xb[fill, t] = xb[fill, t - 1]
            # cold start: if first token dropped, use 0
            cold = mb[:, 0] < 0.5
            xb[cold, 0] = 0

            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            mask_w = (1.0 - mb.reshape(-1))
            ce_raw = nn.functional.cross_entropy(
                logits.reshape(-1, V), yb.reshape(-1), reduction="none")
            loss = (ce_raw * mask_w).sum() / mask_w.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            ep_loss += float(loss)

        note = ""
        if (ep + 1) % ACC_EVERY == 0 or ep == N_EPOCHS - 1:
            with torch.no_grad():
                # eval input: same ffill protocol
                xb_va = Yva_t.clone()
                for t in range(1, W):
                    fill = Mva_t[:, t] < 0.5
                    xb_va[fill, t] = xb_va[fill, t - 1]
                cold = Mva_t[:, 0] < 0.5
                xb_va[cold, 0] = 0
                logits_va, _ = model(xb_va)

                xb_te = Yte_t.clone()
                for t in range(1, W):
                    fill = Mte_t[:, t] < 0.5
                    xb_te[fill, t] = xb_te[fill, t - 1]
                cold = Mte_t[:, 0] < 0.5
                xb_te[cold, 0] = 0
                logits_te, _ = model(xb_te)

            cons_va = stack_consistency(logits_va.numpy(), Yva, Mva)
            cons_te = stack_consistency(logits_te.numpy(), Yte, Mte)
            best_cons = max(best_cons, cons_te)
            note = (f" | CONS val {cons_va:.4f} test {cons_te:.4f} "
                    f"best {best_cons:.4f}")
        print(f"  ep {ep:3d} loss {ep_loss / n_batches:9.4f}{note}", flush=True)

    # Final evaluation
    elapsed = time.time() - t0
    print("\n[4/4] Final evaluation...", flush=True)
    with torch.no_grad():
        xb_te = Yte_t.clone()
        for t in range(1, W):
            fill = Mte_t[:, t] < 0.5
            xb_te[fill, t] = xb_te[fill, t - 1]
        cold = Mte_t[:, 0] < 0.5
        xb_te[cold, 0] = 0
        logits_te, z_te = model(xb_te)

    cons_te = stack_consistency(logits_te.numpy(), Yte, Mte)
    pred_te = np.argmax(logits_te.numpy(), axis=-1)
    dropped = ~(Mte > 0.5)
    exact_te = float((pred_te[dropped] == Yte[dropped]).mean()) if dropped.any() else 0.0

    print(f"  test CONS {cons_te:.4f} | exact {exact_te:.4f}", flush=True)
    print(f"  best CONS {best_cons:.4f}", flush=True)
    print(f"  z_lang shape: {list(z_te.shape)} (should be [{N_TEST}, {W}, {Z_DIM}])", flush=True)

    gate = "PASS" if cons_te >= 0.95 else "FAIL"
    print(f"\n  VERDICT: {gate} (target >= 0.95)", flush=True)
    print(f"  [done] {elapsed:.1f}s | verdict {gate}", flush=True)

    report = {
        "seed": SEED, "n_params": n_par,
        "test_cons": cons_te, "best_cons": best_cons,
        "test_exact": exact_te,
        "z_dim": Z_DIM, "verdict": gate,
    }
    with open("stem_gate1a_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [saved] stem_gate1a_report.json", flush=True)


if __name__ == "__main__":
    main()
