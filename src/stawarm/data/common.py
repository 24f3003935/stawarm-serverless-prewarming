"""Common in-memory representation of a multi-function invocation trace."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FunctionTrace:
    """Invocation counts for N functions over T equal time slots.

    Attributes
    ----------
    meta : DataFrame with one row per function (index 0..N-1). Required column: ``fid``.
        Optional columns used downstream: ``owner`` (user/tenant), ``app``, ``trigger``,
        ``runtime``, ``mem_mb``, ``cold_start_s`` (a high quantile of cold-start duration).
    counts : int32 array of shape (N, T); invocations of function i in slot t.
    slot_seconds : length of one slot in seconds.
    t0 : label of the first slot (for example "d01 00:00"), informational only.
    """

    meta: pd.DataFrame
    counts: np.ndarray
    slot_seconds: float
    t0: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.counts.ndim != 2:
            raise ValueError("counts must be 2-D (functions x slots)")
        if len(self.meta) != self.counts.shape[0]:
            raise ValueError("meta rows must match number of functions")
        self.meta = self.meta.reset_index(drop=True)

    @property
    def n_functions(self) -> int:
        return self.counts.shape[0]

    @property
    def n_slots(self) -> int:
        return self.counts.shape[1]

    def slice_slots(self, start: int, stop: int) -> "FunctionTrace":
        return FunctionTrace(self.meta.copy(), self.counts[:, start:stop], self.slot_seconds,
                             t0=f"{self.t0}+{start}", extra=dict(self.extra))

    def subset(self, idx) -> "FunctionTrace":
        idx = np.asarray(idx)
        return FunctionTrace(self.meta.iloc[idx].copy(), self.counts[idx], self.slot_seconds,
                             t0=self.t0, extra=dict(self.extra))

    def active_slots_per_day(self) -> np.ndarray:
        """Mean number of slots per day in which each function was invoked."""
        days = self.n_slots * self.slot_seconds / 86400.0
        return (self.counts > 0).sum(axis=1) / max(days, 1e-9)

    def save(self, path: str) -> None:
        np.savez_compressed(path, counts=self.counts, slot_seconds=self.slot_seconds, t0=self.t0)
        self.meta.to_parquet(str(path) + ".meta.parquet")

    @staticmethod
    def load(path: str) -> "FunctionTrace":
        z = np.load(path if str(path).endswith(".npz") else str(path) + ".npz")
        base = str(path)[:-4] if str(path).endswith(".npz") else str(path)
        meta = pd.read_parquet(base + ".meta.parquet")
        return FunctionTrace(meta, z["counts"], float(z["slot_seconds"]), str(z["t0"]))


def frequency_bucket(active_per_day: np.ndarray) -> np.ndarray:
    """Buckets used throughout the paper: rare (<10/day), medium (10-1000/day), frequent (>1000/day).

    ``active_per_day`` counts invoked slots per day, which is the quantity that drives cold starts.
    """
    out = np.full(active_per_day.shape, "medium", dtype=object)
    out[active_per_day < 10] = "rare"
    out[active_per_day > 1000] = "frequent"
    return out
