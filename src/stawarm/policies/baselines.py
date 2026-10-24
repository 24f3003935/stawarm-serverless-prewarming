"""Baseline policies: fixed keep-alive and the hybrid histogram policy of Shahrad et al. (ATC 2020)."""
from __future__ import annotations

import numpy as np

from ..sim.simulator import SlotPolicy


class FixedKeepAlive(SlotPolicy):
    """Keep an instance warm for ``k`` slots after each invocation (10 or 20 minutes in practice)."""

    def __init__(self, k: int):
        self.k = int(k)
        self.name = f"keepalive-{k}"

    def reset(self, n: int, n_slots: int) -> None:
        self.n = n

    def keepalive(self) -> np.ndarray:
        return np.full(self.n, self.k)

    def windows(self, i: int, ev: np.ndarray):
        """Same policy expressed for the gap simulator (used to cross-check the two simulators)."""
        return np.zeros(len(ev), np.int64), np.full(len(ev), self.k, np.int64)


class HybridHistogram:
    """Hybrid histogram policy (Shahrad et al., USENIX ATC 2020), expressed on idle times.

    For each function an idle-time (IT) histogram with 1-slot bins over ``range_slots`` is updated
    online. When the histogram is representative (enough samples and coefficient of variation of
    bin counts above ``cv_threshold``), the pre-warm window is the ``head`` percentile of ITs minus a
    ``margin`` and the keep-alive window reaches the ``tail`` percentile plus the margin. Otherwise
    the policy falls back to keeping the instance alive for the whole range.

    Simplification (documented in the paper): the ARIMA fallback for functions whose idle times
    exceed the histogram range is replaced by the same conservative keep-alive.
    """

    def __init__(self, range_slots: int = 240, head: float = 5, tail: float = 99, margin: float = 0.10,
                 cv_threshold: float = 2.0, min_samples: int = 10, bin_slots: int = 1):
        self.range_slots = range_slots
        self.bin = int(bin_slots)       # e.g. 60 when slots are seconds: histogram bins stay one minute
        self.head, self.tail, self.margin = head, tail, margin
        self.cv_threshold, self.min_samples = cv_threshold, min_samples
        self.name = "hybrid-histogram"

    def windows(self, i: int, ev: np.ndarray):
        R = self.range_slots
        hist = np.zeros(R + 1, np.int64)       # index R collects out-of-bounds ITs
        pw = np.zeros(len(ev), np.int64)
        ka = np.full(len(ev), R, np.int64)
        gaps = np.ceil((np.diff(ev) - 1) / self.bin).astype(np.int64)   # idle time in bins
        for k in range(len(ev)):
            if k > 0:
                g = gaps[k - 1]
                if g > 0:                      # back-to-back slots are not idle times
                    hist[min(g, R)] += 1
            inb = hist[:R]
            nsamp = inb.sum()
            if nsamp >= self.min_samples:
                mean = inb.mean()
                cv = inb.std() / mean if mean > 0 else 0.0
                if cv > self.cv_threshold:
                    cdf = np.cumsum(inb) / nsamp
                    h = int(np.searchsorted(cdf, self.head / 100.0))
                    t = int(np.searchsorted(cdf, self.tail / 100.0)) + 1
                    p = int(np.floor(h * (1 - self.margin)))
                    ka[k] = max(1, int(np.ceil(t * (1 + self.margin))) - p)
                    pw[k] = p
        return pw * self.bin, ka * self.bin
