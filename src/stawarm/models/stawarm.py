"""STAWarm model: shared LSTM encoder + lag-aware sparse attention + lead-shifted head.

Shapes used throughout:
  B  batch of (target function, decision slot) pairs
  M  = K + 1 candidate functions per target; index 0 is the target itself
  S  history length in slots (must be >= L + 1)
  F  per-slot input features
  L  largest attention lag
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, ICML 2016): Euclidean projection onto the simplex."""
    z = z - z.max(dim=dim, keepdim=True).values
    zs, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    k = k.view(view)
    cs = zs.cumsum(dim)
    support = (1 + k * zs) > cs
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cs.gather(dim, k_z - 1) - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class STAWarm(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, key_dim: int = 32, max_lag: int = 30,
                 static_dim: int = 0, head_hidden: int = 64, use_sparsemax: bool = True,
                 use_lag_embedding: bool = True, n_nodes: int = 0, node_dim: int = 16):
        super().__init__()
        self.L = max_lag
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.lag_emb = nn.Embedding(max_lag + 1, 16) if use_lag_embedding else None
        # Node (function) embeddings let attention learn *which* candidate matters, not only
        # what its recent state looks like. Without them, content-only keys cannot tell a
        # predictive upstream function from an unrelated one with similar activity.
        self.node_emb = nn.Embedding(n_nodes, node_dim) if n_nodes > 0 else None
        if self.node_emb is not None:
            nn.init.normal_(self.node_emb.weight, std=0.02)
        nd = node_dim if n_nodes > 0 else 0
        kin = hidden + (16 if use_lag_embedding else 0) + nd
        self.Wq = nn.Linear(hidden + nd, key_dim, bias=False)
        self.Wk = nn.Linear(kin, key_dim, bias=False)
        self.Wv = nn.Linear(hidden, hidden, bias=False)
        self.use_sparsemax = use_sparsemax
        # Train with softmax first, then switch to sparsemax (set_warmup(False)). Softmax passes
        # gradient to every candidate, so the model can find the useful ones before sparsemax
        # prunes the rest; starting with sparsemax sometimes locks onto a wrong candidate.
        self.warmup = use_sparsemax
        self.head = nn.Sequential(nn.Linear(2 * hidden + static_dim, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, 1))
        self.key_dim = key_dim
        # Start from near-uniform attention: sparsemax gives zero gradient to entries outside
        # its support, so large initial scores can lock attention onto an arbitrary candidate.
        nn.init.normal_(self.Wq.weight, std=0.01)

    def set_warmup(self, on: bool) -> None:
        self.warmup = bool(on) and self.use_sparsemax

    def forward(self, x: torch.Tensor, mask: torch.Tensor, static: torch.Tensor | None = None,
                ids: torch.Tensor | None = None):
        """x: (B, M, S, F); mask: (B, M) bool, True where the candidate exists (mask[:, 0] is True);
        ids: (B, M) function indices, required when the model was built with n_nodes > 0.

        Returns (logit (B,), attention (B, M, L+1)).
        """
        B, M, S, F = x.shape
        L = self.L
        if S < L + 1:
            raise ValueError("history length S must be at least max_lag + 1")
        h, _ = self.encoder(x.reshape(B * M, S, F))
        h = h.reshape(B, M, S, -1)
        h_i = h[:, 0, -1]                                          # (B, d)
        lagged = h[:, :, S - 1 - torch.arange(L + 1, device=x.device)]  # (B, M, L+1, d); index l = lag l
        if self.lag_emb is not None:
            p = self.lag_emb.weight.view(1, 1, L + 1, -1).expand(B, M, L + 1, -1)
            kin = torch.cat([lagged, p], dim=-1)
        else:
            kin = lagged
        qin = h_i
        if self.node_emb is not None:
            e = self.node_emb(ids)                                  # (B, M, nd)
            kin = torch.cat([kin, e[:, :, None, :].expand(B, M, L + 1, -1)], dim=-1)
            qin = torch.cat([h_i, e[:, 0]], dim=-1)
        q = self.Wq(qin)                                           # (B, k)
        k = self.Wk(kin)                                           # (B, M, L+1, k)
        v = self.Wv(lagged)                                        # (B, M, L+1, d)
        s = torch.einsum("bk,bmlk->bml", q, k) / math.sqrt(self.key_dim)
        s = s.masked_fill(~mask[:, :, None], -1e4)
        flat = s.reshape(B, -1)
        a = sparsemax(flat) if (self.use_sparsemax and not self.warmup) else torch.softmax(flat, dim=-1)
        a = a.reshape(B, M, L + 1)
        c = torch.einsum("bml,bmld->bd", a, v)
        parts = [h_i, c] + ([static] if static is not None else [])
        logit = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        return logit, a


def focal_loss(logit: torch.Tensor, y: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(logit)
    ce = Fnn.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = torch.where(y > 0.5, p, 1 - p)
    return ((1 - pt) ** gamma * ce).mean()


# --------------------------------------------------------------------------- batching

def slot_inputs(counts: np.ndarray, slot_seconds: float, t0_slot_of_day: int = 0) -> np.ndarray:
    """Per-slot input features (N, T, 4): log1p(count), active flag, time-of-day sin/cos."""
    n, T = counts.shape
    per_day = int(round(86400 / slot_seconds))
    tod = ((np.arange(T) + t0_slot_of_day) % per_day) / per_day * 2 * np.pi
    z = np.empty((n, T, 4), np.float32)
    z[..., 0] = np.log1p(counts)
    z[..., 1] = counts > 0
    z[..., 2] = np.sin(tod)[None, :]
    z[..., 3] = np.cos(tod)[None, :]
    return z


def make_batch(z: np.ndarray, neighbours: list[list[tuple[int, float]]], pairs: np.ndarray, S: int, K: int):
    """Assemble (x, mask, ids) for (function, slot) pairs. ``pairs`` is (B, 2) with t >= S - 1."""
    B = len(pairs)
    F = z.shape[-1]
    x = np.zeros((B, K + 1, S, F), np.float32)
    mask = np.zeros((B, K + 1), bool)
    ids = np.zeros((B, K + 1), np.int64)
    for b, (i, t) in enumerate(pairs):
        cand = [i] + [j for j, _ in neighbours[i][:K]]
        for m, j in enumerate(cand):
            x[b, m] = z[j, t - S + 1: t + 1]
            mask[b, m] = True
            ids[b, m] = j
    return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(ids)


def fit_temperature(logits: np.ndarray, y: np.ndarray, iters: int = 200) -> float:
    """Temperature scaling on validation data (Guo et al., ICML 2017)."""
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = Fnn.binary_cross_entropy_with_logits(lt / log_T.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp())
