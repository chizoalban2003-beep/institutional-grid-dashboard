#!/usr/bin/env python3
"""Phase 3 Gate-1b: Crowned grid + language stem injection (Dyck-2).

Architecture:
  1. Language Stem (trained inline, ~143k params):
     tokens -> Embedding(4,32) -> GRU(192,1) -> proj(192,64) -> z_lang (64-dim)
     Trained first (Phase A, 30 eps, grammar-aware CE).
     FROZEN before grid training begins.

  2. Crowned LanguageGrid (extended decode context):
     120-dim input GRU grid -> scorer -> cell_block -> decode_cell
     decode_cell context: [x(120), pooled(64), prev(39), prev_tok(4), z_lang(64)]
     = 291 dims (was 227 without stem).

  3. Training:
     Phase A (ep 0-29): Train language stem only
     Phase B (ep 30-99): Train grid with EWC, stem frozen

Metrics: CONS >= 0.95, math worst R2 >= 0.80, clinical R2 >= 0.90
"""

import json, os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
HIDDEN = 192
N_CELLS = 100
K_ACTIVE = 3
K_MATH = 6
K_CLINICAL = 33
K_SUBJECTS = K_MATH + K_CLINICAL          # 39 regression heads
D_IN_LANG = 120                           # 117 + language triplet
V_LANG = 4

N_WORDS = 4096
N_TEST_WORDS = 256
N_EPOCHS = 100
BATCH = 512
LR = 1e-4
LR_LANG = 1e-3
ACC_FLOOR = 0.95
MATH_FLOOR = 0.80
CLIN_FLOOR = 0.90
LAM_EWC = 2.0
WARMUP_EPC = 30

STEM_EPOCHS = 30
STEM_LR = 1e-4
STEM_BATCH = 128
STEM_HIDDEN = 192
STEM_EMBED = 32
STEM_Z = 64
LABEL_SMOOTH = 0.1
WEIGHT_DECAY = 1e-4
STEM_PATIENCE = 40

FISHER_DATASET = "fisher-v3-total"
FISHER_FILE = "fisher_v3.npz"
DEPTH_MAX = 4
DROP_LO, DROP_HI = 0.10, 0.30
W_WINDOW = 14

# ---------------------------------------------------------------------------
# Dyck-2 generator (inlined from dyck_worlds.py)
# ---------------------------------------------------------------------------

def _rng(seed):
    return np.random.default_rng(seed)


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


def build_dyck_dataset(n_words, seed, aligned=True, w=W_WINDOW):
    rng = _rng(seed)
    X, Y, M = [], [], []
    for _ in range(n_words):
        ids = generate_word(_rng(int(rng.integers(0, 2 ** 31))))
        T = len(ids)
        mask = (rng.random(T) >= rng.uniform(DROP_LO, DROP_HI)).astype(np.float32)
        value = ids.astype(np.float32).copy()
        last = -1
        for t in range(T):
            if mask[t] > 0.5:
                last = t
            elif last >= 0:
                value[t] = value[last]
            else:
                value[t] = 0.0
        delta = np.zeros(T, dtype=np.float32)
        last_obs = -1
        for t in range(T):
            if mask[t] > 0.5:
                last_obs = t
                delta[t] = 0.0
            elif last_obs >= 0:
                delta[t] = min(t - last_obs, 24.0)
            else:
                delta[t] = 24.0
        if aligned:
            starts = range(0, T - w + 1, w)
        else:
            starts = range(0, T - w + 1, 2)
        for s in starts:
            a, b = int(s), int(s) + w
            x = np.stack([value[a:b], mask[a:b], delta[a:b]], axis=-1)
            X.append(x.astype(np.float32))
            Y.append(ids[a:b])
            M.append(mask[a:b])
    return np.stack(X), np.stack(Y), np.stack(M)


# ---------------------------------------------------------------------------
# Language block embedding (120-dim grid input)
# ---------------------------------------------------------------------------

def embed_language_block(X3, total_dim=D_IN_LANG):
    B, Wn, _ = X3.shape
    grid = np.zeros((B, Wn, total_dim), dtype=np.float32)
    grid[:, :, 1::3] = 1.0
    grid[:, :, total_dim - 3:total_dim] = X3
    return grid


# ---------------------------------------------------------------------------
# Grammar-consistency grader
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


def token_accuracy(pred_logits, y, mask):
    pred = np.argmax(pred_logits, axis=-1)
    dropped = ~(mask > 0.5)
    if not dropped.any():
        return 0.0
    return float((pred[dropped] == y[dropped]).mean())


# ---------------------------------------------------------------------------
# Grammar-aware CE loss
# ---------------------------------------------------------------------------

def build_legal_mask(y_np, depth_max=DEPTH_MAX):
    B, Wn = y_np.shape
    legal = np.zeros((B, Wn, V_LANG), dtype=np.float32)
    for b in range(B):
        stack = []
        for t in range(Wn):
            tok = int(y_np[b, t])
            if len(stack) < depth_max:
                legal[b, t, 0] = 1.0
                legal[b, t, 1] = 1.0
            if stack:
                legal[b, t, 2 + stack[-1]] = 1.0
            if tok < 2:
                stack.append(tok)
            elif stack and tok == 2 + stack[-1]:
                stack.pop()
    return legal


def grammar_aware_ce(logits, y, mask, legal_mask):
    dropped = (mask < 0.5).unsqueeze(-1)
    lse_all = torch.logsumexp(logits, dim=-1)
    lse_legal = torch.logsumexp(
        logits + (1.0 - legal_mask).clamp(min=0) * (-1e6), dim=-1)
    per_tok = lse_all - lse_legal
    loss = (per_tok.unsqueeze(-1) * dropped.float()).sum()
    n_dropped = dropped.float().sum().clamp(min=1.0)
    return loss / n_dropped


# ---------------------------------------------------------------------------
# Math exam (vendored from math_to_language_1b.py)
# ---------------------------------------------------------------------------

def _exam_rng(seed):
    return np.random.RandomState(seed)


def continuum(rng):
    n_math = 6
    W = W_WINDOW
    X = np.zeros((W, n_math * 3), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, W, dtype=np.float32)
    kinds = ["sine", "cosine", "decay", "step", "sigmoid", "lorenz"]
    for k, kind in enumerate(kinds):
        off = k * 3
        if kind == "sine":
            amp = rng.uniform(0.3, 0.7)
            freq = rng.uniform(0.5, 2.0)
            X[:, off] = amp * np.sin(freq * t)
        elif kind == "cosine":
            amp = rng.uniform(0.3, 0.7)
            freq = rng.uniform(0.5, 2.0)
            X[:, off] = amp * np.cos(freq * t)
        elif kind == "decay":
            tau = rng.uniform(2.0, 6.0)
            X[:, off] = np.exp(-t / tau)
        elif kind == "step":
            loc = rng.randint(3, W - 3)
            X[:, off] = (t >= t[loc]).astype(np.float32)
        elif kind == "sigmoid":
            loc = rng.uniform(t[2], t[-3])
            scale = rng.uniform(0.5, 2.0)
            X[:, off] = 1.0 / (1.0 + np.exp(-(t - loc) / scale))
        elif kind == "lorenz":
            dt = 0.05
            x, y, z = 1.0, 1.0, 1.0
            xs = []
            for _ in range(W):
                dx = 10 * (y - x) * dt
                dy = (x * (28 - z) - y) * dt
                dz = (x * y - 8 / 3 * z) * dt
                x += dx; y += dy; z += dz
                xs.append(x)
            X[:, off] = np.array(xs, dtype=np.float32)
            X[:, off] = (X[:, off] - X[:, off].mean()) / (X[:, off].std() + 1e-8)
        X[:, off + 1] = 1.0
        X[:, off + 2] = 0.0
    return X, kinds


def damage(rng, X):
    B, Wn, C = X.shape
    Xd = X.copy()
    n_math = 6
    for b in range(B):
        kind_idx = rng.randint(0, n_math)
        off = kind_idx * 3
        n_drop = max(1, int(Wn * rng.uniform(0.1, 0.3)))
        drops = rng.choice(Wn, size=n_drop, replace=False)
        Xd[b, drops, off] = 0.0
        Xd[b, drops, off + 1] = 0.0
        Xd[b, drops, off + 2] = 1.0
    return Xd


def build_math_dataset(n_stays, seed):
    rng = _exam_rng(seed)
    Xs = []
    for _ in range(n_stays):
        x, _ = continuum(rng)
        Xs.append(x)
    X = np.stack(Xs)
    Y = X.copy()
    X = damage(rng, X)
    return X, Y


def embed_math_block(X18, total_dim=117):
    B, Wn, _ = X18.shape
    g = np.zeros((B, Wn, total_dim), dtype=np.float32)
    g[:, :, 1::3] = 1.0
    g[:, :, :X18.shape[2]] = X18
    return g


def exam_windows(n_stays, seed):
    X, Y = build_math_dataset(n_stays, seed)
    M = np.ones_like(Y[:, :, 0])
    return X, Y, M


# ---------------------------------------------------------------------------
# Clinical windows (vendored from math_to_language_1b.py)
# ---------------------------------------------------------------------------

def clinical_windows(n_stays, seed):
    rng = _exam_rng(seed)
    n_feat = K_CLINICAL
    W = W_WINDOW
    Xs, Ys, Ms = [], [], []
    for _ in range(n_stays):
        stay_len = rng.randint(W, W * 2)
        x = np.zeros((stay_len, n_feat * 3), dtype=np.float32)
        for f in range(n_feat):
            base = rng.uniform(-1, 1)
            trend = rng.uniform(-0.02, 0.02)
            for t in range(stay_len):
                x[t, f * 3] = base + trend * t + rng.normal(0, 0.1)
                x[t, f * 3 + 1] = 1.0
                x[t, f * 3 + 2] = 0.0
        n_drop = rng.randint(1, max(2, n_feat // 3))
        drop_cols = rng.choice(n_feat, size=n_drop, replace=False)
        for c in drop_cols:
            n_cd = rng.randint(2, max(3, stay_len // 3))
            cd_starts = rng.choice(stay_len - n_cd, size=min(n_cd, stay_len - n_cd), replace=False)
            for s in cd_starts:
                x[s:s+n_cd, c*3] = 0.0
                x[s:s+n_cd, c*3+1] = 0.0
                x[s:s+n_cd, c*3+2] = 1.0
        starts = list(range(0, stay_len - W + 1, W))
        for s in starts:
            Xs.append(x[s:s+W])
            Ys.append(x[s:s+W, :, ] if False else np.stack([x[s:s+W, f*3] for f in range(n_feat)], axis=-1))
            Ms.append(np.stack([x[s:s+W, f*3+1] for f in range(n_feat)], axis=-1))
    return np.stack(Xs), np.stack(Ys), np.stack(Ms)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LanguageStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V_LANG, STEM_EMBED)
        self.gru = nn.GRU(STEM_EMBED, STEM_HIDDEN, batch_first=True)
        self.proj = nn.Linear(STEM_HIDDEN, STEM_Z)
        self.head = nn.Linear(STEM_Z, V_LANG)

    def forward(self, x):
        emb = self.embed(x)
        h, _ = self.gru(emb)
        z = self.proj(h)
        logits = self.head(z)
        return logits, z


class LanguageGridStem(nn.Module):
    """LanguageGrid with stem-injected decode context (291 dims).

    decode ctx = [x(120), pooled(64), prev(39), prev_tok(4), z_lang(64)]
    """

    def __init__(self, d_in, hidden, n_cells, k, k_subjects, vocab, z_dim=STEM_Z):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.scorer = nn.Linear(hidden, n_cells)
        self.cell_block = nn.Linear(hidden, n_cells * 64)
        decode_in = d_in + 64 + k_subjects + vocab + z_dim  # 291
        self.decode_cell = nn.GRUCell(decode_in, hidden)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(k_subjects)])
        self.vocab_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, vocab),
        )
        self.k = k
        self.n_cells = n_cells
        self.k_subjects = k_subjects
        self.vocab = vocab
        self.z_dim = z_dim

    def forward(self, x, z_lang=None, return_routing=False):
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
        if z_lang is None:
            z_lang = torch.zeros(B, Wn, self.z_dim, device=x.device)
        outs, vlogits = [], []
        for t in range(Wn):
            ctx = torch.cat([x[:, t], pooled, prev, prev_tok, z_lang[:, t]], dim=1)
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


# ---------------------------------------------------------------------------
# EWC machinery
# ---------------------------------------------------------------------------

def discover_input(name):
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if name in (root, os.path.basename(root)):
            return root
    raise FileNotFoundError(f"dataset {name} not found under {base}")


def load_fisher_normalized():
    ckpt_dir = discover_input(FISHER_DATASET)
    f_raw = np.load(os.path.join(ckpt_dir, FISHER_FILE))
    f2 = {k: np.asarray(f_raw[k], dtype=np.float64) for k in f_raw.files}
    total = sum(float(v.sum()) for v in f2.values())
    n = sum(v.size for v in f2.values())
    mean = total / n
    assert np.isfinite(mean) and mean > 0, f"bad Fisher mean {mean}"
    f2p = {}
    for k, v in f2.items():
        if k in ("gru.weight_ih_l0", "decode_cell.weight_ih") and \
                v.shape[1] == 117:
            pad = np.zeros((v.shape[0], D_IN_LANG - 117), dtype=np.float64)
            v = np.concatenate([v, pad], axis=1)
        if k == "decode_cell.weight_ih":
            cur_cols = v.shape[1]
            target = D_IN_LANG + 64 + K_SUBJECTS + V_LANG + STEM_Z  # 291
            if cur_cols < target:
                pad = np.zeros((v.shape[0], target - cur_cols), dtype=np.float64)
                v = np.concatenate([v, pad], axis=1)
        f2p[k] = v
    f2 = f2p
    total = sum(float(v.sum()) for v in f2.values())
    n = sum(v.size for v in f2.values())
    mean = total / n
    assert np.isfinite(mean) and mean > 0
    f2 = {k: torch.tensor(v / mean, dtype=torch.float32)
          for k, v in f2.items()}
    print(f"[ewc] F_total loaded {len(f2)} keys, mean-1 normalised", flush=True)
    return f2


class _ThetaStar:
    pass

_THETA_STAR = _ThetaStar()


def snapshot_theta_star(model, f2):
    st = _ThetaStar()
    for k in f2:
        st_attr = k.replace(".", "_")
        setattr(st, st_attr, dict(model.named_parameters())[k].detach().clone())
    return st


def ewc_penalty(model, f2, lam):
    total = torch.zeros((), dtype=torch.float32)
    for k, F in f2.items():
        st_attr = k.replace(".", "_")
        p = dict(model.named_parameters())[k]
        th_star = getattr(_THETA_STAR, st_attr)
        total = total + (F * (p - th_star) ** 2).sum()
    return 0.5 * lam * total


# ---------------------------------------------------------------------------
# pad helpers
# ---------------------------------------------------------------------------

def embed_math_inline(X18):
    B, Wn, _ = X18.shape
    g = np.zeros((B, Wn, 117), dtype=np.float32)
    g[:, :, 1::3] = 1.0
    g[:, :, :K_MATH * 3] = X18
    return g


def pad_to_120(x117):
    B, Wn, _ = x117.shape
    x = np.zeros((B, Wn, D_IN_LANG), dtype=np.float32)
    x[:, :, :117] = x117
    x[:, :, 118] = 1.0
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- Phase A: Train language stem ----
    print("=" * 60)
    print("Phase 3 Gate-1b: Crowned grid + stem injection")
    print("=" * 60, flush=True)

    print("[Phase A] Training language stem (Dyck-2)...", flush=True)
    Xtr_lang, Ytr_lang, Mtr_lang = build_dyck_dataset(N_WORDS, SEED, aligned=True)
    Xva_lang, Yva_lang, Mva_lang = build_dyck_dataset(N_TEST_WORDS, SEED + 1, aligned=True)
    Xte_lang, Yte_lang, Mte_lang = build_dyck_dataset(N_TEST_WORDS, SEED + 2, aligned=True)

    Ytr_l = torch.tensor(Ytr_lang, dtype=torch.long)
    Mtr_l = torch.tensor(Mtr_lang, dtype=torch.float32)
    Yva_l = torch.tensor(Yva_lang, dtype=torch.long)
    Mva_l = torch.tensor(Mva_lang, dtype=torch.float32)
    Yte_l = torch.tensor(Yte_lang, dtype=torch.long)
    Mte_l = torch.tensor(Mte_lang, dtype=torch.float32)

    legal_tr = torch.tensor(build_legal_mask(Ytr_lang), dtype=torch.float32)
    legal_va = torch.tensor(build_legal_mask(Yva_lang), dtype=torch.float32)

    stem = LanguageStem()
    n_stem = sum(p.numel() for p in stem.parameters())
    print(f"  stem {n_stem:,} params | train {Ytr_l.shape[0]} windows", flush=True)

    opt_stem = torch.optim.AdamW(stem.parameters(), lr=STEM_LR, weight_decay=WEIGHT_DECAY)
    n_tr = Ytr_l.shape[0]
    n_batches_s = (n_tr + STEM_BATCH - 1) // STEM_BATCH
    best_val_cons = 0.0
    best_test_cons = 0.0
    best_stem_state = None
    wait = 0

    for ep in range(STEM_EPOCHS):
        perm = torch.randperm(n_tr)
        ep_loss = 0.0
        for i in range(n_batches_s):
            idx = perm[i * STEM_BATCH: (i + 1) * STEM_BATCH]
            yb = Ytr_l[idx]
            mb = Mtr_l[idx]
            xb = yb.clone()
            for t in range(1, W_WINDOW):
                fill = mb[:, t] < 0.5
                xb[fill, t] = xb[fill, t - 1]
            cold = mb[:, 0] < 0.5
            xb[cold, 0] = 0

            opt_stem.zero_grad(set_to_none=True)
            logits, _ = stem(xb)
            gce = grammar_aware_ce(logits, yb, mb, legal_tr[idx])
            cce = nn.functional.cross_entropy(
                logits.reshape(-1, V_LANG), yb.reshape(-1), reduction="none",
                label_smoothing=LABEL_SMOOTH)
            mask_w = (1.0 - mb.reshape(-1))
            cce_masked = (cce * mask_w).sum() / mask_w.sum().clamp(min=1.0)
            loss = 0.5 * gce + 0.5 * cce_masked
            loss.backward()
            opt_stem.step()
            ep_loss += float(loss)

        with torch.no_grad():
            xb_va = Yva_l.clone()
            for t in range(1, W_WINDOW):
                fill = Mva_l[:, t] < 0.5
                xb_va[fill, t] = xb_va[fill, t - 1]
            cold = Mva_l[:, 0] < 0.5
            xb_va[cold, 0] = 0
            logits_va, _ = stem(xb_va)
            xb_te = Yte_l.clone()
            for t in range(1, W_WINDOW):
                fill = Mte_l[:, t] < 0.5
                xb_te[fill, t] = xb_te[fill, t - 1]
            cold = Mte_l[:, 0] < 0.5
            xb_te[cold, 0] = 0
            logits_te, _ = stem(xb_te)

        cons_va = stack_consistency(logits_va.numpy(), Yva_lang, Mva_lang)
        cons_te = stack_consistency(logits_te.numpy(), Yte_lang, Mte_lang)
        note = ""
        if cons_va > best_val_cons:
            best_val_cons = cons_va
            best_test_cons = cons_te
            best_stem_state = {k: v.clone() for k, v in stem.state_dict().items()}
            note = " *BEST*"
            wait = 0
        else:
            wait += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  stem ep {ep:3d} loss {ep_loss / n_batches_s:9.4f} "
                  f"CONS val {cons_va:.4f} te {cons_te:.4f} best {best_val_cons:.4f}{note}",
                  flush=True)
        if wait >= STEM_PATIENCE:
            print(f"  stem early stop at ep {ep}", flush=True)
            break

    if best_stem_state is not None:
        stem.load_state_dict(best_stem_state)
    print(f"  stem best val CONS {best_val_cons:.4f} test {best_test_cons:.4f}", flush=True)

    for p in stem.parameters():
        p.requires_grad = False
    stem.eval()

    # Precompute z_lang for all datasets
    print("  precomputing z_lang vectors...", flush=True)
    def get_z_lang(Y_t, M_t):
        with torch.no_grad():
            xb = Y_t.clone()
            for t in range(1, W_WINDOW):
                fill = M_t[:, t] < 0.5
                xb[fill, t] = xb[fill, t - 1]
            cold = M_t[:, 0] < 0.5
            xb[cold, 0] = 0
            _, z = stem(xb)
            return z  # (N, W, 64)

    z_tr = get_z_lang(Ytr_l, Mtr_l)
    z_te = get_z_lang(Yte_l, Mte_l)
    print(f"  z_lang shapes: train {list(z_tr.shape)} test {list(z_te.shape)}", flush=True)

    # ---- Phase B: Train grid with EWC ----
    print("\n[Phase B] Crowned grid + stem injection...", flush=True)

    Xtr, Ytr, Mtr = build_dyck_dataset(N_WORDS, SEED, aligned=True)
    Xte, Yte, Mte = build_dyck_dataset(N_TEST_WORDS, SEED + 1, aligned=True)
    Xtr_g = torch.tensor(embed_language_block(Xtr), dtype=torch.float32)
    Ytr_g = torch.tensor(Ytr, dtype=torch.long)
    Mtr_g = torch.tensor(Mtr, dtype=torch.float32)
    Xte_g = torch.tensor(embed_language_block(Xte), dtype=torch.float32)
    Yte_g = torch.tensor(Yte, dtype=torch.long)
    Mte_g = torch.tensor(Mte, dtype=torch.float32)

    model = LanguageGridStem(D_IN_LANG, HIDDEN, N_CELLS, K_ACTIVE,
                             K_SUBJECTS, V_LANG, STEM_Z)

    ckpt_dir = discover_input("crowned-ckpt-fast")
    crowned = torch.load(os.path.join(ckpt_dir, "math2clinic_fast.pt"),
                         map_location="cpu", weights_only=True)

    DECODE_BLOCKS = [(0, 117, 0, 117),
                     (120, 184, 117, 181),
                     (184, 223, 181, 220)]
    with torch.no_grad():
        for k, v in crowned.items():
            if k.startswith("vocab_head"):
                continue
            if k not in dict(model.named_parameters()):
                continue
            p = dict(model.named_parameters())[k]
            if tuple(p.shape) == tuple(v.shape):
                p.copy_(v)
            elif k == "decode_cell.weight_ih":
                for tlo, thi, slo, shi in DECODE_BLOCKS:
                    p[:, tlo:thi].copy_(v[:, slo:shi])
            elif p.ndim == 2 and v.ndim == 2 and p.shape[0] == v.shape[0] \
                    and p.shape[1] > v.shape[1]:
                p[:, :v.shape[1]].copy_(v)
                p[:, v.shape[1]:].zero_()
            elif p.ndim == 1 and v.ndim == 1 and p.shape[0] == v.shape[0]:
                p.copy_(v)
    print("  crowned weights copied; z_lang columns + vocab head fresh", flush=True)

    print("[Phase B] F_total EWC ledger...", flush=True)
    f2 = load_fisher_normalized()
    global _THETA_STAR
    _THETA_STAR = snapshot_theta_star(model, f2)

    opt = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters()
                    if "vocab_head" in n], "lr": LR_LANG},
        {"params": [p for n, p in model.named_parameters()
                    if "vocab_head" not in n], "lr": LR},
    ])
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {n_par:,} params | core lr {LR} | lang lr {LR_LANG} | "
          f"lam {LAM_EWC:g}", flush=True)

    # math/clinical exams
    XM18, YM, MM = exam_windows(192, SEED + 2)
    XC, YC, MC = clinical_windows(64, SEED + 3)
    XM = torch.tensor(pad_to_120(embed_math_inline(XM18)), dtype=torch.float32)
    YM = torch.tensor(YM, dtype=torch.float32)
    MM = torch.tensor(MM, dtype=torch.float32)
    XC = torch.tensor(pad_to_120(XC), dtype=torch.float32)
    YC = torch.tensor(YC, dtype=torch.float32)
    MC = torch.tensor(MC, dtype=torch.float32)

    z_xm = torch.zeros(XM.shape[0], W_WINDOW, STEM_Z)
    z_xc = torch.zeros(XC.shape[0], W_WINDOW, STEM_Z)

    with torch.no_grad():
        p_m = model(XM, z_lang=z_xm)[0]
        p_c = model(XC, z_lang=z_xc)[0]
    r2_math = {k: float(1.0 - (((p_m[:, :, i:i+1] - YM[:, :, i:i+1]) ** 2)
                                * (1.0 - MM[:, :, i:i+1])).sum()
                           / max((((YM[:, :, i:i+1]
                                    - YM[:, :, i:i+1].mean(dim=(0, 1),
                                                           keepdim=True)) ** 2)
                                  * (1.0 - MM[:, :, i:i+1])).sum(), 1e-9))
               for i, k in enumerate(["sine", "cosine", "decay", "step",
                                       "sigmoid", "lorenz"])}
    r2_clin = float(1.0 - ((p_c[:, :, K_MATH:] - YC[:, :, K_MATH:]) ** 2
                            * (1.0 - MC[:, :, K_MATH:])).sum()
                     / max(((YC[:, :, K_MATH:]
                             - YC[:, :, K_MATH:].mean(dim=(0, 1),
                                                      keepdim=True)) ** 2
                            * (1.0 - MC[:, :, K_MATH:])).sum(), 1e-9))
    print("  math base: " + " ".join(f"{k} {v:.3f}" for k, v in r2_math.items()), flush=True)
    print(f"  clinical base: {r2_clin:.4f}", flush=True)

    # ---- Training loop ----
    print("\n[Phase B] Training (Dyck-2 CE + F_total EWC)...", flush=True)
    n = Xtr_g.shape[0]
    n_batches = (n + BATCH - 1) // BATCH
    curve = []

    for ep in range(N_EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        ewc_tot = 0.0
        use_ewc = ep >= WARMUP_EPC
        for i in range(n_batches):
            idx = perm[i * BATCH: (i + 1) * BATCH]
            xb = Xtr_g[idx]
            yb = Ytr_g[idx]
            mb = Mtr_g[idx]
            zb = z_tr[idx]
            opt.zero_grad(set_to_none=True)
            _, vlog = model(xb, z_lang=zb)
            lg = vlog.reshape(-1, V_LANG)
            tg = yb.reshape(-1)
            mask_w = (1.0 - mb.reshape(-1))
            ce_raw = nn.functional.cross_entropy(lg, tg, reduction="none")
            loss = (ce_raw * mask_w).sum() / mask_w.sum().clamp(min=1.0)
            if use_ewc:
                ewc = ewc_penalty(model, f2, LAM_EWC)
                loss = loss + ewc
                ewc_tot += float(ewc)
            loss.backward()
            opt.step()
            tot += float(loss - (ewc if use_ewc else 0.0))

        note = ""
        if (ep + 1) % 5 == 0 or ep == N_EPOCHS - 1:
            with torch.no_grad():
                _, vlog = model(Xte_g, z_lang=z_te)
                p_m = model(XM, z_lang=z_xm)[0]
                p_c = model(XC, z_lang=z_xc)[0]
            cons = stack_consistency(vlog.numpy(), Yte_lang, Mte_lang)
            acc = token_accuracy(vlog.numpy(), Yte_lang, Mte_lang)
            worst_m = min(float(1.0 - (((p_m[:, :, i:i+1] - YM[:, :, i:i+1])
                                        ** 2) * (1.0 - MM[:, :, i:i+1])).sum()
                               / max((((YM[:, :, i:i+1]
                                        - YM[:, :, i:i+1].mean(dim=(0, 1),
                                                               keepdim=True))
                                       ** 2) * (1.0 - MM[:, :, i:i+1])).sum(),
                                      1e-9))
                          for i in range(6))
            r2c = float(1.0 - ((p_c[:, :, K_MATH:] - YC[:, :, K_MATH:]) ** 2
                               * (1.0 - MC[:, :, K_MATH:])).sum()
                        / max(((YC[:, :, K_MATH:]
                                - YC[:, :, K_MATH:].mean(dim=(0, 1),
                                                         keepdim=True)) ** 2
                               * (1.0 - MC[:, :, K_MATH:])).sum(), 1e-9))
            curve.append({"epoch": ep + 1, "cons": cons, "acc": acc,
                          "math_worst": worst_m, "clin": r2c})
            note = f" CONS {cons:.4f} | math {worst_m:.3f} clin {r2c:.3f}"
        phase = "" if use_ewc else " W"
        print(f"  ep {ep:3d}{phase} loss {tot / n_batches:9.4f} "
              f"ewc {ewc_tot / n_batches:9.4f}{note}", flush=True)

    # ---- Final verdict ----
    print("\n[Final] Triple verdict...", flush=True)
    with torch.no_grad():
        _, vlog = model(Xte_g, z_lang=z_te)
        p_m = model(XM, z_lang=z_xm)[0]
        p_c = model(XC, z_lang=z_xc)[0]
    cons = stack_consistency(vlog.numpy(), Yte_lang, Mte_lang)
    acc = token_accuracy(vlog.numpy(), Yte_lang, Mte_lang)
    r2_math_f = {k: float(1.0 - (((p_m[:, :, i:i+1] - YM[:, :, i:i+1]) ** 2)
                                 * (1.0 - MM[:, :, i:i+1])).sum()
                            / max((((YM[:, :, i:i+1]
                                     - YM[:, :, i:i+1].mean(dim=(0, 1),
                                                            keepdim=True))
                                    ** 2) * (1.0 - MM[:, :, i:i+1])).sum(),
                                   1e-9))
                 for i, k in enumerate(["sine", "cosine", "decay", "step",
                                        "sigmoid", "lorenz"])}
    r2_clin_f = float(1.0 - ((p_c[:, :, K_MATH:] - YC[:, :, K_MATH:]) ** 2
                              * (1.0 - MC[:, :, K_MATH:])).sum()
                       / max(((YC[:, :, K_MATH:]
                               - YC[:, :, K_MATH:].mean(dim=(0, 1),
                                                        keepdim=True)) ** 2
                              * (1.0 - MC[:, :, K_MATH:])).sum(), 1e-9))
    worst_m = min(r2_math_f.values())

    ok = (cons >= ACC_FLOOR and worst_m >= MATH_FLOOR
          and r2_clin_f >= CLIN_FLOOR)
    print(f"  language CONS {cons:.4f} (floor {ACC_FLOOR}) | exact {acc:.4f}", flush=True)
    print("  math: " + " ".join(f"{k} {v:.3f}" for k, v in r2_math_f.items()), flush=True)
    print(f"  clinical: {r2_clin_f:.4f} (floor {CLIN_FLOOR})", flush=True)
    print("  VERDICT:", "PASS" if ok else "FAIL", flush=True)

    elapsed = time.time() - t0
    torch.save(model.state_dict(), "/kaggle/working/gate1b_stem.pt")
    report = {
        "gate": "1b_stem",
        "stem_cons": best_val_cons,
        "stem_test_cons": best_test_cons,
        "grid_cons": cons,
        "grid_acc": acc,
        "math_r2_final": r2_math_f,
        "clinical_r2_final": r2_clin_f,
        "math_r2_base": r2_math,
        "clinical_r2_base": r2_clin,
        "curve": curve,
        "params": n_par,
        "verdict": "PASS" if ok else "FAIL",
        "seconds": round(elapsed, 1),
    }
    with open("/kaggle/working/gate1b_stem_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {elapsed:.1f}s | verdict {report['verdict']}", flush=True)


if __name__ == "__main__":
    main()
