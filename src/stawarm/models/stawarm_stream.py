"""Streaming STAWarm for one-second decisions: a shared LSTM over each function's inputs (temporal part)
plus sparse attention over the hidden states of functions in the same application (spatial part).

The original STAWarm (``stawarm.py``) re-encodes a history window for every (function, slot) sample,
which is fine at minute resolution but far too slow for hundreds of millions of per-second decisions.
Here the LSTM runs once along time for all functions together and keeps its state, so a prediction
costs one LSTM step plus one small attention over at most K neighbours.

Two trainers share the model:
  * ``train_and_predict_lstm``      - a CPU-sized trainer on random 15-min windows with the function's own inputs.
  * ``train_and_predict_lstm_full`` - full-trace truncated back-propagation through time over every
    second of the training period, the same inputs as the tree models, a held-out validation day with
    early stopping, any model size, CPU or GPU.

Shapes: N functions, T seconds in a block, F input features, d hidden size, K neighbours per function.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .stawarm import sparsemax
from ..seconds import FeatureStream, chunk_targets

OWN_ONLY = ("own_log_since", "own_10s", "own_60s", "own_600s", "own_3600s", "tod_sin", "tod_cos",
            "log_rate", "log_lead")


def neighbour_table(app_idx: np.ndarray, rate: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each function, up to k other functions of the same application, busiest first (training
    call rate). Returns (nb (N, k) int64, valid (N, k) bool)."""
    n = len(app_idx)
    nb = np.zeros((n, k), np.int64)
    valid = np.zeros((n, k), bool)
    for a in np.unique(app_idx[app_idx >= 0]):
        rows = np.flatnonzero(app_idx == a)
        order = rows[np.argsort(-rate[rows], kind="stable")]
        for i in rows:
            others = order[order != i][:k]
            nb[i, :len(others)] = others
            valid[i, :len(others)] = True
    return nb, valid


class STAWarmStream(nn.Module):
    def __init__(self, n_features: int, n_nodes: int, hidden: int = 32, key_dim: int = 16,
                 node_dim: int = 8, attention: bool = True, layers: int = 1, dropout: float = 0.0,
                 value_dim: int | None = None, head_hidden: int = 32):
        super().__init__()
        self.attention = attention
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.node_emb = nn.Embedding(n_nodes, node_dim)
        nn.init.normal_(self.node_emb.weight, std=0.02)
        vd = value_dim or hidden
        self.Wq = nn.Linear(hidden + node_dim, key_dim, bias=False)
        self.Wk = nn.Linear(hidden + node_dim, key_dim, bias=False)
        self.Wv = nn.Linear(hidden, vd, bias=False)
        nn.init.normal_(self.Wq.weight, std=0.01)        # near-uniform attention at the start
        self.key_dim = key_dim
        self.warmup = True                                # softmax first, then sparsemax
        self.head = nn.Sequential(nn.Linear(hidden + (vd if attention else 0), head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor, state=None, nb: torch.Tensor | None = None,
                valid: torch.Tensor | None = None, return_attention: bool = False):
        """x: (N, T, F) for ALL functions (attention needs the neighbours' states at the same second).
        nb, valid: (N, K) neighbour table. Returns (logits (N, T), new state[, attention (N, T, K+1)])."""
        h, state = self.lstm(x, state)                                    # (N, T, d)
        if not self.attention:
            out = self.head(h).squeeze(-1)
            return (out, state, None) if return_attention else (out, state)
        N, T, _ = h.shape
        dev = h.device
        e = self.node_emb.weight                                          # (N, nd)
        he = torch.cat([h, e[:, None, :].expand(-1, T, -1)], dim=-1)      # (N, T, d + nd)
        q = self.Wq(he)                                                   # (N, T, k)
        k_all = self.Wk(he)                                               # keys and values are computed
        v_all = self.Wv(h)                                                # once per function, then gathered
        cand = torch.cat([torch.arange(N, device=dev)[:, None], nb.to(dev)], dim=1)   # (N, K+1): self first
        cmask = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=dev), valid.to(dev)], dim=1)
        s = torch.einsum("ntk,nmtk->ntm", q, k_all[cand]) / math.sqrt(self.key_dim)  # (N, T, K+1)
        s = s.masked_fill(~cmask[:, None, :], -1e4)
        a = torch.softmax(s, dim=-1) if self.warmup else sparsemax(s, dim=-1)
        c = torch.einsum("ntm,nmtd->ntd", a, v_all[cand])                 # (N, T, dv)
        out = self.head(torch.cat([h, c], dim=-1)).squeeze(-1)
        return (out, state, a) if return_attention else (out, state)


def _select(ch_x: np.ndarray, names: list, keep: tuple | None) -> tuple[np.ndarray, list]:
    if keep is None:
        return ch_x, list(names)
    idx = [i for i, n in enumerate(names) if n in keep]
    return ch_x[..., idx], [names[i] for i in idx]


def _predict(model, active, app_idx, rate, lead_feature, use_app, keep, pred_start, nb_t, valid_t,
             block, device):
    """One causal pass from six hours before pred_start, carrying the state; returns P (N, T - pred_start)."""
    N, T = active.shape
    model.eval()
    P = np.zeros((N, T - pred_start), np.float16)
    state = None
    warm_from = max(0, pred_start - 21600)
    with torch.no_grad():
        for ch in FeatureStream(active, app_idx, rate, use_app=use_app, lead=lead_feature).chunks(start=warm_from + 1):
            X, _ = _select(ch.X, ch.names, keep)
            for b in range(0, ch.hi - ch.lo, block):
                lo = ch.lo + b
                hi = min(lo + block, ch.hi)
                x = torch.from_numpy(np.ascontiguousarray(X[:, b:b + (hi - lo)], dtype=np.float32)).to(device)
                logit, state = model(x, state, nb_t, valid_t)
                if hi > pred_start:
                    s = max(lo, pred_start)
                    P[:, s - pred_start: hi - pred_start] = torch.sigmoid(logit[:, s - lo:].float()).cpu().numpy()
    return P


def train_and_predict_lstm(active, app_idx, rate, lead_target, H, train_end, pred_start, attention,
                           seed=0, log=print, lead_feature=None, k_neighbours=8, window=900, burn=300,
                           windows_per_chunk=2, epochs=3, hidden=32, block=600):
    """CPU-sized trainer on randomly sampled windows (not used for the paper's final results).

    Training: from every 6-hour chunk of the training period, ``windows_per_chunk`` random windows of
    ``window`` seconds for all functions; the first ``burn`` seconds of a window only warm up the LSTM
    state. Loss = weighted cross-entropy so positives make about a third of the effective weight, as in
    the tree models' sampling. Softmax attention for the first half of training, sparsemax after.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N, T = active.shape
    nb, valid = neighbour_table(app_idx, rate, k_neighbours)
    nb_t, valid_t = torch.from_numpy(nb), torch.from_numpy(valid)

    n_chunks = math.ceil(train_end / 21600)
    windows_per_chunk = max(windows_per_chunk, math.ceil(60 / n_chunks))   # at least ~60 windows overall
    Xw, Yw, names = [], [], None
    for ch in FeatureStream(active, app_idx, rate, use_app=False, lead=lead_feature).chunks(stop=train_end):
        X, names = _select(ch.X, ch.names, OWN_ONLY)
        L = ch.hi - ch.lo
        if L < window:
            continue
        y = chunk_targets(active, ch.lo, ch.hi, lead_target, H)
        for s in rng.integers(0, L - window + 1, windows_per_chunk):
            Xw.append(X[:, s:s + window].astype(np.float16))
            Yw.append(y[:, s:s + window])
    Y = np.stack(Yw)[:, :, burn:]
    pos = max(int(Y.sum()), 1)
    pos_weight = torch.tensor(max((Y.size - pos) / (2.0 * pos), 1.0))
    log(f"    LSTM training windows: {len(Xw)} x {N} functions x {window} s, positives {pos / Y.size:.1%}, "
        f"features {len(names)}, attention {'on' if attention else 'off'}")

    model = STAWarmStream(len(names), N, hidden=hidden, attention=attention)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    steps = epochs * len(Xw)
    for step in range(steps):
        w = rng.integers(len(Xw))
        model.warmup = step < steps // 2
        x = torch.from_numpy(Xw[w].astype(np.float32))
        y = torch.from_numpy(Yw[w][:, burn:].astype(np.float32))
        logit, _ = model(x, None, nb_t, valid_t)
        loss = Fnn.binary_cross_entropy_with_logits(logit[:, burn:], y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, steps // 6) == 0:
            log(f"    step {step + 1}/{steps}: loss {loss.item():.4f}")
    del Xw, Yw, Y
    P = _predict(model, active, app_idx, rate, lead_feature, False, OWN_ONLY, pred_start, nb_t, valid_t,
                 block, torch.device("cpu"))
    return P, model, names


def train_and_predict_lstm_full(active, app_idx, rate, lead_target, H, train_end, pred_start, attention,
                                seed=0, log=print, lead_feature=None, union=False, k_neighbours=8,
                                hidden=128, layers=2, dropout=0.1, key_dim=32, node_dim=16, value_dim=32,
                                head_hidden=64, bptt=900, lr=2e-3, max_epochs=20, patience=3,
                                val_seconds=None, feature_warmup=3600, block=900, device=None):
    """Full-trace training. Same contract as seconds.train_and_predict.

    * Inputs: every feature the tree models get (own + application trailing activity + time of day +
      call rate [+ cold-start lead]). The LSTM therefore sees at least the information the trees see.
    * Data: every second of [feature_warmup, train_end - val_seconds) for all functions, every epoch,
      in consecutive blocks of ``bptt`` seconds with the LSTM state carried (and detached) between
      blocks. A random offset per epoch changes the block boundaries.
    * Validation: the last ``val_seconds`` of the training period (default: one day, or one sixth of the
      training period if that is shorter), never used for gradients. Early stopping on validation loss
      with ``patience`` epochs; the best epoch's weights are kept.
    * Loss: cross-entropy with positives weighted to about one third of the effective weight (as for the
      trees), so the threshold grid means the same thing for every predictor.
    * Attention: softmax for the first epoch, sparsemax afterwards.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    N, T = active.shape
    val_seconds = val_seconds or min(86400, train_end // 6)
    val_start = train_end - val_seconds
    nb, valid = neighbour_table(app_idx, rate, k_neighbours)
    nb_t, valid_t = torch.from_numpy(nb).to(device), torch.from_numpy(valid).to(device)

    # features and targets for the whole training period, held in memory as float16 / bool
    t0 = time.time()
    X_all, names = None, None
    for ch in FeatureStream(active, app_idx, rate, use_app=True, lead=lead_feature).chunks(stop=train_end):
        if X_all is None:
            names = ch.names
            X_all = np.zeros((N, train_end, len(names)), np.float16)
        X_all[:, ch.lo:ch.hi] = ch.X
    Y_all = chunk_targets(active, 0, train_end, lead_target, H, union=union)
    tr = Y_all[:, feature_warmup:val_start]
    pos = max(int(tr.sum()), 1)
    pos_weight = torch.tensor(max((tr.size - pos) / (2.0 * pos), 1.0), device=device)
    log(f"    LSTM-full: {N} functions, train {feature_warmup}-{val_start} s, validate {val_start}-{train_end} s, "
        f"features {len(names)}, positives {pos / tr.size:.1%}, device {device}, built in {time.time() - t0:.0f} s")

    model = STAWarmStream(len(names), N, hidden=hidden, key_dim=key_dim, node_dim=node_dim, attention=attention,
                          layers=layers, dropout=dropout, value_dim=value_dim, head_hidden=head_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    def run(lo, hi, train: bool, state=None):
        """Loss over [lo, hi) in blocks of bptt seconds; returns (mean loss, state)."""
        total, count = 0.0, 0
        for b in range(lo, hi, bptt):
            e = min(b + bptt, hi)
            x = torch.from_numpy(X_all[:, b:e].astype(np.float32)).to(device, non_blocking=True)
            y = torch.from_numpy(Y_all[:, b:e].astype(np.float32)).to(device, non_blocking=True)
            if state is not None:
                state = tuple(s.detach() for s in state)
            if train:
                logit, state = model(x, state, nb_t, valid_t)
                loss = Fnn.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                with torch.no_grad():
                    logit, state = model(x, state, nb_t, valid_t)
                    loss = Fnn.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            total += loss.item() * (e - b)
            count += e - b
        return total / max(count, 1), state

    best, best_loss, bad = None, np.inf, 0
    for epoch in range(max_epochs):
        t1 = time.time()
        model.train()
        model.warmup = epoch == 0
        start = feature_warmup + int(rng.integers(0, bptt))
        train_loss, state = run(start, val_start, True)
        model.eval()
        model.warmup = False                 # validate (and later predict) with sparsemax attention
        # validation continues from the training state at val_start (no gap, no leakage of labels)
        _, state_v = run(max(feature_warmup, val_start - 3600), val_start, False)
        val_loss, _ = run(val_start, train_end, False, state_v)
        sched.step()
        improved = val_loss < best_loss - 1e-4
        if improved:
            best, best_loss, bad = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            bad += 1
        log(f"    epoch {epoch + 1}: train {train_loss:.4f}, validation {val_loss:.4f}"
            f"{' (best)' if improved else ''}, {time.time() - t1:.0f} s")
        if bad >= patience:
            break
    model.load_state_dict(best)
    model.warmup = False
    del X_all, Y_all
    P = _predict(model, active, app_idx, rate, lead_feature, True, None, pred_start, nb_t, valid_t, block, device)
    return P, model, names
