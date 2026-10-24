"""One-second pipeline: dense activity, cold-start profiles, chunked features, predictors.

Everything is processed in time chunks so that 14 days x hundreds of functions at one-second
resolution fits in 16 GB of memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import future_range_any, future_window_any

WINDOWS_OWN = (10, 60, 600, 3600)
WINDOWS_APP = (10, 60, 600)
SINCE_CAP = 86400


def events_to_active(ev: pd.DataFrame, n_seconds: int | None = None) -> np.ndarray:
    """(N, T) boolean: function i was called during second t. Uses integer fid codes 0..N-1."""
    n = int(ev["fid"].max()) + 1
    sec = np.floor(ev["start"].to_numpy()).astype(np.int64)
    T = int(sec.max()) + 1 if n_seconds is None else n_seconds
    keep = sec < T
    a = np.zeros((n, T), dtype=bool)
    a[ev["fid"].to_numpy()[keep], sec[keep]] = True
    return a


def app_index(ev: pd.DataFrame, n: int) -> np.ndarray:
    m = ev.drop_duplicates("fid").set_index("fid")["app"]
    return m.reindex(range(n)).fillna(-1).astype(np.int64).to_numpy()


def assign_profiles(n: int, stats: pd.DataFrame, seed: int = 0, scale: float = 1.0) -> pd.DataFrame:
    """Give each function a cold-start profile sampled from Huawei's measured functions.

    Azure traces have no cold-start durations or memory sizes, so (d50, d90, memory) triples are
    drawn jointly from Huawei Region 2 functions with at least 5 observed cold starts. ``scale``
    multiplies the durations for the sensitivity analysis (0.5, 1, 2).
    """
    pool = stats[(stats["n_cold"] >= 5) & stats["d90_s"].notna()].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pick = pool.iloc[rng.integers(0, len(pool), n)].reset_index(drop=True)
    mem = pick["mem_mb"].fillna(pool["mem_mb"].median()).to_numpy(float)
    d90 = np.minimum(pick["d90_s"].to_numpy(float) * scale, 60.0)
    d50 = np.minimum(pick["d50_s"].to_numpy(float) * scale, d90)
    return pd.DataFrame({"d50_s": d50, "d90_s": d90, "mem_mb": mem,
                         "lead_s": np.maximum(1, np.ceil(d90)).astype(np.int64)})


def _trailing(c: np.ndarray, w: int, first: int) -> np.ndarray:
    """From a cumulative sum c over [s, hi), trailing w-window sums for columns first..end."""
    idx = np.arange(first, c.shape[1])
    lo = idx - w
    val = c[:, idx] - np.where(lo[None, :] >= 0, c[:, np.maximum(lo, 0)], 0)
    return val


@dataclass
class ChunkFeatures:
    lo: int
    hi: int
    X: np.ndarray        # (N, hi-lo, F) float32
    names: list


class FeatureStream:
    """Sequential pass over time producing per-(function, second) features in chunks.

    Feature set 'self': own recency and trailing call counts, time of day, training call rate.
    Feature set 'app' adds the same for the other functions of the same application (or user).
    """

    def __init__(self, active: np.ndarray, app_idx: np.ndarray, rate: np.ndarray, use_app: bool,
                 chunk: int = 21600, hist: int = 3600, lead: np.ndarray | None = None):
        """``lead`` (N,) seconds: if given, log(1 + cold-start lead) is added as a feature, so a
        model trained on a lead-dependent target can tell fast functions from slow ones."""
        self.log_lead = None if lead is None else np.log1p(np.asarray(lead, float)).astype(np.float32)
        self.a = active
        self.N, self.T = active.shape
        self.app = app_idx
        self.rate = np.log1p(rate).astype(np.float32)
        self.use_app = use_app
        self.chunk, self.hist = chunk, hist
        if use_app:
            n_apps = int(app_idx.max()) + 1
            self.app_count = np.zeros((n_apps, self.T), dtype=np.int16)
            for k in range(n_apps):
                rows = np.flatnonzero(app_idx == k)
                if len(rows):
                    self.app_count[k] = active[rows].sum(axis=0, dtype=np.int16)

    @staticmethod
    def _since(block: np.ndarray, carry: np.ndarray, offset: int):
        """Seconds since the last True at or before each column; carry = last global index or -1."""
        n, L = block.shape
        idx = np.where(block, np.arange(offset, offset + L)[None, :], -1)
        idx = np.maximum.accumulate(np.concatenate([carry[:, None], idx], axis=1), axis=1)[:, 1:]
        since = np.where(idx >= 0, np.arange(offset, offset + L)[None, :] - idx, SINCE_CAP)
        return np.minimum(since, SINCE_CAP), idx[:, -1]

    def chunks(self, start: int = 0, stop: int | None = None):
        stop = self.T if stop is None else stop
        carry_own = np.full(self.N, -1, np.int64)
        carry_app = np.full(self.N, -1, np.int64)
        for lo in range(0, stop, self.chunk):
            hi = min(lo + self.chunk, stop)
            s = max(0, lo - self.hist)
            if hi <= start:                       # before the requested range: only carry state
                _, carry_own = self._since(self.a[:, lo:hi], carry_own, lo)
                if self.use_app:
                    oth = self.app_count[np.maximum(self.app, 0), lo:hi].astype(np.int32) - self.a[:, lo:hi]
                    oth = np.where(self.app[:, None] >= 0, oth, 0)
                    _, carry_app = self._since(oth > 0, carry_app, lo)
                continue
            blk = self.a[:, s:hi].astype(np.int32)
            first = lo - s
            feats, names = [], []
            since, carry_own = self._since(self.a[:, lo:hi], carry_own, lo)
            feats.append(np.log1p(since).astype(np.float32)); names.append("own_log_since")
            c = np.cumsum(blk, axis=1)
            for w in WINDOWS_OWN:
                feats.append(np.log1p(_trailing(c, w, first)).astype(np.float32)); names.append(f"own_{w}s")
            if self.use_app:
                oth = np.where(self.app[:, None] >= 0,
                               self.app_count[np.maximum(self.app, 0), s:hi].astype(np.int32) - blk, 0)
                osince, carry_app = self._since(oth[:, first:] > 0, carry_app, lo)
                feats.append(np.log1p(osince).astype(np.float32)); names.append("app_log_since")
                co = np.cumsum(oth, axis=1)
                for w in WINDOWS_APP:
                    feats.append(np.log1p(_trailing(co, w, first)).astype(np.float32)); names.append(f"app_{w}s")
            tod = (np.arange(lo, hi) % 86400) / 86400 * 2 * np.pi
            feats.append(np.broadcast_to(np.sin(tod).astype(np.float32), (self.N, hi - lo))); names.append("tod_sin")
            feats.append(np.broadcast_to(np.cos(tod).astype(np.float32), (self.N, hi - lo))); names.append("tod_cos")
            feats.append(np.broadcast_to(self.rate[:, None], (self.N, hi - lo))); names.append("log_rate")
            if self.log_lead is not None:
                feats.append(np.broadcast_to(self.log_lead[:, None], (self.N, hi - lo))); names.append("log_lead")
            yield ChunkFeatures(lo, hi, np.stack(feats, axis=-1), names)


def chunk_targets(active: np.ndarray, lo: int, hi: int, lead: np.ndarray, H: int,
                  union: bool = False) -> np.ndarray:
    """y[i, t] for t in [lo, hi).

    Default ("lead-shifted"): function i is called in [t + lead_i, t + lead_i + H - 1].
    union=True: called anywhere in [t + 1, t + lead_i + H - 1], i.e. from now until the end of the
    hold. It covers both the near term (relevant to a warm instance) and the part after the lead
    (relevant to a cold one).
    """
    end = min(active.shape[1], hi + int(lead.max()) + H)
    if union:
        lead = np.asarray(lead, np.int64)
        y = future_range_any(active[:, lo:end], np.ones_like(lead), lead + H - 1)
    else:
        y = future_window_any(active[:, lo:end], lead, H)
    return y[:, : hi - lo]


def train_and_predict(active, app_idx, rate, lead_target, H, train_end, pred_start, use_app,
                      pos_per_chunk=40_000, neg_per_chunk=80_000, seed=0, log=print, union=False,
                      lead_feature=None):
    """One sequential pass: collect training rows before train_end, then predict from pred_start.

    Returns (P float16 array (N, T - pred_start), fitted model, feature names).
    """
    rng = np.random.default_rng(seed)
    stream = FeatureStream(active, app_idx, rate, use_app, lead=lead_feature)
    Xs, ys = [], []
    names = None
    for ch in stream.chunks(stop=train_end):
        names = ch.names
        lo = max(ch.lo, 3600)
        if lo >= ch.hi:
            continue
        y = chunk_targets(active, ch.lo, ch.hi, lead_target, H, union=union)
        off = lo - ch.lo
        yy = y[:, off:]
        pi, pt = np.nonzero(yy)
        if len(pi) > pos_per_chunk:
            k = rng.choice(len(pi), pos_per_chunk, replace=False)
            pi, pt = pi[k], pt[k]
        ni = rng.integers(0, yy.shape[0], neg_per_chunk)
        nt = rng.integers(0, yy.shape[1], neg_per_chunk)
        keep = ~yy[ni, nt]
        ii = np.concatenate([pi, ni[keep]]); tt = np.concatenate([pt, nt[keep]]) + off
        Xs.append(ch.X[ii, tt]); ys.append(y[ii, tt])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    log(f"    training rows: {len(y):,} ({y.mean():.1%} positive), features: {len(names)}")
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                         early_stopping=True, random_state=seed)
    clf.fit(X, y)
    del X, y
    T = active.shape[1]
    P = np.zeros((active.shape[0], T - pred_start), dtype=np.float16)
    stream = FeatureStream(active, app_idx, rate, use_app, lead=lead_feature)
    for ch in stream.chunks(start=pred_start):
        lo = max(ch.lo, pred_start)
        Xc = ch.X[:, lo - ch.lo:]
        p = clf.predict_proba(Xc.reshape(-1, Xc.shape[-1]))[:, 1]
        P[:, lo - pred_start: ch.hi - pred_start] = p.reshape(Xc.shape[0], Xc.shape[1])
    return P, clf, names
