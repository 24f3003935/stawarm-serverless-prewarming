"""Trace-driven simulation of the zero-to-one cold-start problem.

Two simulators with identical accounting:

* ``simulate_slots`` steps through time and is vectorised across functions. It supports any
  policy expressed through the ``SlotPolicy`` interface: keep-alive after an invocation,
  probability-driven pre-warming with a per-function lead, and online threshold updates.
* ``simulate_gaps`` walks through the gaps between invoked slots of one function. It is used for
  policies that are naturally defined on idle times, such as the hybrid histogram policy.

Accounting, per function i and slot t (only slots t >= warmup are counted):
  invoked   : x[i, t] > 0
  cold      : invoked and no instance was warm at the start of slot t
  idle_warm : an instance was warm during slot t but the function was not invoked
Wasted memory-time = idle_warm slots x slot length x memory of the function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    invoked: np.ndarray      # (N,) invoked slots
    cold: np.ndarray         # (N,) cold slots
    idle_warm: np.ndarray    # (N,) idle warm slots
    mem_mb: np.ndarray       # (N,)
    slot_seconds: float
    name: str = ""
    invoked_all: np.ndarray | None = None   # (N,) including the warm-up period (for online-bound checks)
    cold_all: np.ndarray | None = None

    @property
    def cold_rate(self) -> float:
        return float(self.cold.sum() / max(self.invoked.sum(), 1))

    @property
    def wasted_gb_s(self) -> float:
        return float((self.idle_warm * self.mem_mb).sum() * self.slot_seconds / 1024.0)

    def per_function(self) -> pd.DataFrame:
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(self.invoked > 0, 100.0 * self.cold / self.invoked, np.nan)
        return pd.DataFrame({"invoked": self.invoked, "cold": self.cold, "idle_warm": self.idle_warm,
                             "cold_pct": pct})

    def summary(self, reference: "SimResult | None" = None) -> dict:
        pf = self.per_function()
        out = {"policy": self.name, "cold_rate_pooled": self.cold_rate,
               "cold_pct_p50_fn": float(np.nanpercentile(pf["cold_pct"], 50)),
               "cold_pct_p75_fn": float(np.nanpercentile(pf["cold_pct"], 75)),
               "wasted_gb_s": self.wasted_gb_s}
        if reference is not None and reference.wasted_gb_s > 0:
            out["wasted_vs_ref"] = self.wasted_gb_s / reference.wasted_gb_s
        return out


class SlotPolicy:
    """Interface for slot-level policies. All methods are vectorised over functions."""

    name = "policy"

    def reset(self, n: int, n_slots: int) -> None:  # noqa: D401
        """Called once before simulation."""

    def keepalive(self) -> np.ndarray:
        """Slots an instance stays warm after an invocation (N,) int."""
        raise NotImplementedError

    def want_warm(self, t: int) -> np.ndarray | None:
        """Boolean (N,) mask: at slot t, the policy wants an instance ready from slot t + lead.
        Return None if the policy never pre-warms."""
        return None

    def lead(self) -> np.ndarray:
        """Slots needed for a pre-warm to complete (N,) int >= 1."""
        return None  # type: ignore[return-value]

    def hold(self) -> np.ndarray:
        """Slots a pre-warmed instance is kept if unused (N,) int >= 1."""
        return None  # type: ignore[return-value]

    def observe(self, t: int, invoked: np.ndarray, cold: np.ndarray) -> None:
        """Feedback after slot t (used by the online risk controller)."""


def simulate_slots(counts: np.ndarray, policy: SlotPolicy, mem_mb: np.ndarray, slot_seconds: float,
                   warmup: int = 0, delay: int = 0, charge_loading: bool = False,
                   start_duration=None) -> SimResult:
    """Realism options (all off by default, which reproduces the original accounting):
      delay          : slots between a decision and its effect. A start is ready after delay + lead; an
                       instance is only extended if it is still alive when the command lands.
      charge_loading : count the slots an instance spends loading (start to ready) as idle memory.
      start_duration : callable(rows) -> int array; the actual load time of each start, replacing the
                       planned lead (e.g. random draws from the function's cold-start distribution).
    The policy can read ``policy.active_now`` (instance warm or loading at this slot) in want_warm."""
    n, T = counts.shape
    policy.reset(n, T)
    ka = np.asarray(policy.keepalive(), dtype=np.int64)
    warm_from = np.full(n, np.iinfo(np.int64).max // 4, dtype=np.int64)
    warm_until = np.full(n, -1, dtype=np.int64)
    invoked_n = np.zeros(n, np.int64)
    cold_n = np.zeros(n, np.int64)
    idle_n = np.zeros(n, np.int64)
    inv_all = np.zeros(n, np.int64)
    cold_all = np.zeros(n, np.int64)
    lead = policy.lead()
    hold = policy.hold()
    for t in range(T):
        inv = counts[:, t] > 0
        warm = (warm_from <= t) & (t <= warm_until)
        cold = inv & ~warm
        inv_all += inv
        cold_all += cold
        if t >= warmup:
            invoked_n += inv
            cold_n += cold
            idle_n += warm & ~inv
        # an invocation leaves a warm instance behind for the keep-alive period
        if inv.any():
            warm_from = np.where(inv, np.minimum(warm_from, t), warm_from)
            warm_until = np.where(inv, np.maximum(warm_until, t + ka), warm_until)
        policy.observe(t, inv, cold)
        policy.active_now = warm_until >= t + delay
        want = policy.want_warm(t)
        if want is not None:
            want_until = t + delay + lead + hold - 1
            # an instance that is warm or already loading (and still alive when the command lands) is
            # simply kept longer; a function with no instance starts loading and is ready after its load time
            active = policy.active_now
            start = want & ~active
            extend = want & active & (want_until > warm_until)
            ready_at = t + delay + lead
            if start_duration is not None and start.any():
                rows = np.flatnonzero(start)
                ready_at = ready_at.copy()
                ready_at[rows] = t + delay + np.asarray(start_duration(rows), np.int64)
            if charge_loading and t >= warmup and start.any():
                idle_n += np.where(start, ready_at - t, 0)
            warm_from = np.where(start, ready_at, warm_from)
            warm_until = np.where(start | extend, np.maximum(np.where(start, -1, warm_until), want_until), warm_until)
    return SimResult(invoked_n, cold_n, idle_n, np.asarray(mem_mb, float), slot_seconds, policy.name,
                     inv_all, cold_all)


# --------------------------------------------------------------------------- gap-based

def _window_overlap(lo: np.ndarray, hi: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Length of [lo, hi] ∩ [a, b] for integer intervals (element-wise)."""
    return np.maximum(0, np.minimum(hi, b) - np.maximum(lo, a) + 1)


def simulate_gaps(counts: np.ndarray, window_fn, mem_mb: np.ndarray, slot_seconds: float,
                  warmup: int = 0, name: str = "gap-policy", prewarm_cost=None) -> SimResult:
    """Simulate a policy defined by (pre-warm, keep-alive) windows after each invocation.

    ``window_fn(i, event_times) -> (pw, ka)`` returns two int arrays aligned with the events of
    function i: after the event at e_k the instance is warm on slots [e_k + max(pw_k, 1), e_k + pw_k + ka_k].
    ``window_fn`` must only use information available at e_k (it receives all event times and
    is responsible for being causal).
    """
    n, T = counts.shape
    invoked_n = np.zeros(n, np.int64)
    cold_n = np.zeros(n, np.int64)
    idle_n = np.zeros(n, np.int64)
    for i in range(n):
        ev = np.flatnonzero(counts[i] > 0)
        if len(ev) == 0:
            continue
        pw, ka = window_fn(i, ev)
        pw = np.asarray(pw, np.int64)
        ka = np.asarray(ka, np.int64)
        lo = ev + np.maximum(pw, 1)
        hi = ev + pw + ka
        nxt = np.append(ev[1:], T)            # next event (or end of trace)
        # cold: first event, and any later event not covered by the previous window
        warm_next = (nxt[:-1] >= lo[:-1]) & (nxt[:-1] <= hi[:-1])
        is_cold = np.concatenate([[True], ~warm_next])
        counted = ev >= warmup
        invoked_n[i] = counted.sum()
        cold_n[i] = (is_cold & counted).sum()
        # idle warm slots strictly between events, clipped to [warmup, T-1]
        a = np.maximum(ev + 1, warmup)
        b = np.minimum(nxt - 1, T - 1)
        idle_n[i] = _window_overlap(lo, hi, a, b).sum()
        if prewarm_cost is not None:            # loading time of each scheduled pre-warm (pw > 0), if charged
            idle_n[i] += int(prewarm_cost[i]) * int(((pw > 0) & counted).sum())
    return SimResult(invoked_n, cold_n, idle_n, np.asarray(mem_mb, float), slot_seconds, name)
