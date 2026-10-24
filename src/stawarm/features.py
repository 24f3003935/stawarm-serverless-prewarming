"""Targets, candidate neighbours and tabular features.

The lead-shifted target (paper, Problem Formulation):
    y[i, t] = 1  if function i is invoked in slots t + lead_i ... t + lead_i + H - 1.

Candidate neighbours (used by the minute-level model): for each function, the K functions with the largest maximum
lagged cross-correlation of activity on training data, restricted to the same owner/tenant.
Restricting to the owner keeps the search linear in practice and matches how multi-function
applications are deployed; functions of different tenants rarely call each other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lead_slots(cold_start_s: np.ndarray, slot_seconds: float) -> np.ndarray:
    """kappa_i = ceil(d_i / Delta), at least one slot (the lead of function i)."""
    return np.maximum(1, np.ceil(np.asarray(cold_start_s, float) / slot_seconds)).astype(np.int64)


def future_window_any(active: np.ndarray, lead: np.ndarray, H: int) -> np.ndarray:
    """y[i, t] = any(active[i, t+lead_i : t+lead_i+H]). Slots whose window runs past the end are 0."""
    n, T = active.shape
    a = active.astype(np.int32)
    csum = np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(a, axis=1)], axis=1)
    y = np.zeros((n, T), dtype=bool)
    t = np.arange(T)
    for L in np.unique(lead):
        rows = np.flatnonzero(lead == L)
        lo = np.minimum(t + L, T)
        hi = np.minimum(t + L + H, T)
        y[rows] = (csum[np.ix_(rows, hi)] - csum[np.ix_(rows, lo)]) > 0
    return y


def future_range_any(active: np.ndarray, start: np.ndarray, length: np.ndarray) -> np.ndarray:
    """y[i, t] = any(active[i, t+start_i : t+start_i+length_i]) with per-function start and length."""
    n, T = active.shape
    a = active.astype(np.int32)
    csum = np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(a, axis=1)], axis=1)
    y = np.zeros((n, T), dtype=bool)
    t = np.arange(T)
    pairs = np.stack([np.asarray(start, np.int64), np.asarray(length, np.int64)], axis=1)
    for s, L in np.unique(pairs, axis=0):
        rows = np.flatnonzero((pairs[:, 0] == s) & (pairs[:, 1] == L))
        lo = np.minimum(t + s, T)
        hi = np.minimum(t + s + L, T)
        y[rows] = (csum[np.ix_(rows, hi)] - csum[np.ix_(rows, lo)]) > 0
    return y


def trailing_sum(active: np.ndarray, w: int) -> np.ndarray:
    """Number of active slots in the last w slots, including t."""
    a = active.astype(np.int32)
    c = np.cumsum(a, axis=1)
    out = c.copy()
    out[:, w:] = c[:, w:] - c[:, :-w]
    return out


def slots_since_last(active: np.ndarray, cap: int) -> np.ndarray:
    """Slots since the most recent active slot at or before t (0 if active at t), capped."""
    n, T = active.shape
    idx = np.where(active, np.arange(T)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    since = np.where(last >= 0, np.arange(T)[None, :] - last, cap)
    return np.minimum(since, cap)


def lagged_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """max over lags l in [0, max_lag] of corr(a[t - l], b[t]); a leads b."""
    best = 0.0
    for l in range(0, max_lag + 1):
        x = a[: len(a) - l] if l else a
        y = b[l:]
        if x.std() == 0 or y.std() == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        best = max(best, c)
    return best


def candidate_neighbours(active: np.ndarray, owner: np.ndarray, K: int, max_lag: int,
                         min_corr: float = 0.05) -> list[list[tuple[int, float]]]:
    """For each function, up to K (neighbour, score) pairs from the same owner (training data only)."""
    n = active.shape[0]
    out: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    af = active.astype(np.float32)
    groups = pd.Series(np.arange(n)).groupby(pd.Series(owner)).apply(list)
    for members in groups:
        if len(members) < 2:
            continue
        for i in members:
            scores = []
            for j in members:
                if j == i:
                    continue
                s = lagged_xcorr(af[j], af[i], max_lag)
                if s >= min_corr:
                    scores.append((j, s))
            scores.sort(key=lambda z: -z[1])
            out[i] = scores[:K]
    return out


def build_features(active: np.ndarray, slot_seconds: float, neighbours=None, K: int = 3,
                   windows=(1, 5, 15, 60), cap: int = 1440, t0_slot_of_day: int = 0) -> tuple[np.ndarray, list[str]]:
    """Tabular features per (function, slot) for gradient-boosting and logistic baselines.

    Returns X with shape (N, T, F) as float32 and feature names. Self features: slots since last
    invocation (log), trailing activity over several windows, time of day. If ``neighbours`` is
    given, adds for each of the top-K neighbours: slots since its last invocation (log) and its
    trailing activity over the shortest two windows, plus the neighbour's score.
    """
    n, T = active.shape
    per_day = int(round(86400 / slot_seconds))
    feats, names = [], []
    since = slots_since_last(active, cap).astype(np.float32)
    feats.append(np.log1p(since)); names.append("self_log_since")
    for w in windows:
        feats.append(trailing_sum(active, w).astype(np.float32) / w); names.append(f"self_rate_{w}")
    tod = ((np.arange(T) + t0_slot_of_day) % per_day) / per_day * 2 * np.pi
    feats.append(np.broadcast_to(np.sin(tod), (n, T)).astype(np.float32)); names.append("tod_sin")
    feats.append(np.broadcast_to(np.cos(tod), (n, T)).astype(np.float32)); names.append("tod_cos")
    if neighbours is not None:
        short = windows[:2]
        nb_since = np.log1p(since)
        nb_rates = {w: trailing_sum(active, w).astype(np.float32) / w for w in short}
        for k in range(K):
            idx = np.array([nb[k][0] if len(nb) > k else -1 for nb in neighbours])
            score = np.array([nb[k][1] if len(nb) > k else 0.0 for nb in neighbours], np.float32)
            has = idx >= 0
            safe = np.where(has, idx, 0)
            f_since = np.where(has[:, None], nb_since[safe], np.log1p(cap)).astype(np.float32)
            feats.append(f_since); names.append(f"nb{k}_log_since")
            for w in short:
                feats.append(np.where(has[:, None], nb_rates[w][safe], 0).astype(np.float32)); names.append(f"nb{k}_rate_{w}")
            feats.append(np.broadcast_to(score[:, None], (n, T)).astype(np.float32)); names.append(f"nb{k}_score")
    X = np.stack(feats, axis=-1)
    return X, names
