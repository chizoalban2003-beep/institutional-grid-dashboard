"""MathSchoolGrid — shared architecture (Phase-1 certified v9/v10).

Byte-identical copy of the class in curriculum/kaggle_push/math_school_train.py
(the frozen, certified artifact). Phase-2 (math-to-MIMIC transfer) imports
this class with K_SUBJECTS=39, d_in=117; the transfer plan in
curriculum/phase2_transfer.py copies only shape-compatible weights.

GRU encoder + shared 100-cell grid + MASK-CONDITIONED decoder (v5).

The decoder is a per-position GRUCell that sees the FULL context vector
x[t] = [value_ffill, mask, delta] at every step, plus the pooled routing
latent and the blended prediction of the previous step. At OBSERVED slots
the output is the TRUE value by construction (hard copy:
y = mask*value + (1-mask)*y_est) — the model can anchor on local observed
points instead of decoding all W positions blind.

Structural clamp on channel 2 is Phase-1-specific (decay pedagogy); it is a
geometry detail of the 6-kind exam, NOT part of the transferable prior — the
MIMIC stem does not inherit it.
"""

import torch
import torch.nn as nn


class MathSchoolGrid(nn.Module):
    def __init__(self, d_in, hidden, n_cells, k, k_subjects,
                 clamp_decay_channel: int = 2):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.scorer = nn.Linear(hidden, n_cells)
        self.cell_block = nn.Linear(hidden, n_cells * 64)
        self.decode_cell = nn.GRUCell(d_in + 64 + k_subjects, hidden)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(k_subjects)])
        self.k = k
        self.n_cells = n_cells
        self.k_subjects = k_subjects
        self.clamp_decay_channel = clamp_decay_channel

    def forward(self, x, return_routing=False):
        B, W, D = x.shape
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
        pooled = selected.mean(dim=1)                     # (B, 64)
        value = x[:, :, 0::3]                             # (B, W, K) ffill'd
        m = x[:, :, 1::3]                                 # (B, W, K) 1=obs
        state = h_last.contiguous()
        prev = torch.zeros(B, self.k_subjects, device=x.device)
        outs = []
        for t in range(W):
            ctx = torch.cat([x[:, t], pooled, prev], dim=1)
            state = self.decode_cell(ctx, state)
            y_est = torch.cat(
                [head(state) for head in self.heads], dim=1)   # (B, K)
            if self.clamp_decay_channel is not None:
                c = self.clamp_decay_channel
                # functional splice — no in-place writes in the autograd graph
                y_est = torch.cat(
                    [y_est[:, :c], y_est[:, c:c + 1].clamp(min=-0.05,
                                                            max=2.20),
                     y_est[:, c + 1:]], dim=1)
            y = m[:, t] * value[:, t] + (1.0 - m[:, t]) * y_est
            outs.append(y)
            prev = y
        out = torch.stack(outs, dim=1)                    # (B, W, K)
        if return_routing:
            return out, votes
        return out