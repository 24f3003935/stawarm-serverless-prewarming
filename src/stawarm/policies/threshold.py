"""Decision layer of STAWarm: cost-sensitive thresholds plus the online risk controller.

Given a probability matrix P (N x T) where P[i, t] is the predicted probability that function i
is invoked in its decision window W_i(t) = {t + lead_i, ..., t + lead_i + H - 1}, the policy keeps
(or makes) function i warm for that window when

    P[i, t] >= min(theta_star_i, theta_i(t)),

where theta_star_i = C_idle / (C_idle + C_cold) (the cost threshold of the paper) and theta_i(t) is updated after
every invoked slot by   theta <- min(1, theta - eta * (miss - alpha))   (the budget controller).
"""
from __future__ import annotations

import numpy as np

from ..sim.simulator import SlotPolicy


def cost_threshold(mem_mb: np.ndarray, cold_start_s: np.ndarray, hold_s: float,
                   price_per_gb_s: float, penalty_per_s: np.ndarray | float) -> np.ndarray:
    """theta*_i = C_idle / (C_idle + C_cold), with C_idle = price * mem * (d + H) and C_cold = beta * d."""
    c_idle = price_per_gb_s * (np.asarray(mem_mb, float) / 1024.0) * (np.asarray(cold_start_s, float) + hold_s)
    c_cold = np.asarray(penalty_per_s, float) * np.asarray(cold_start_s, float)
    return c_idle / (c_idle + c_cold)


class ThresholdPolicy(SlotPolicy):
    def __init__(self, probs: np.ndarray, theta_star: np.ndarray | float, lead: np.ndarray | int = 1,
                 hold: int = 1, keepalive_min: int = 1, alpha: float | None = None, eta: float = 0.01,
                 theta0: float | np.ndarray | None = None, groups: np.ndarray | None = None,
                 name: str = "threshold"):
        """``groups`` (N,) int: functions with the same group id share one online threshold, so a
        rare function learns from its group's misses. None = one threshold per function.
        ``theta0`` may be per function (e.g. a low, cautious start for rare functions); a group
        starts at the lowest theta0 of its members."""
        self.P = probs
        self.theta_star_in = theta_star
        self.lead_in, self.hold_in, self.ka_min = lead, int(hold), int(keepalive_min)
        self.alpha, self.eta, self.theta0 = alpha, eta, theta0
        self.groups_in = groups
        self.name = name

    def reset(self, n: int, n_slots: int) -> None:
        if self.P.shape != (n, n_slots):
            raise ValueError(f"probability matrix {self.P.shape} does not match trace {(n, n_slots)}")
        self.theta_star = np.broadcast_to(np.asarray(self.theta_star_in, float), (n,)).copy()
        self._lead = np.broadcast_to(np.asarray(self.lead_in, np.int64), (n,)).copy()
        start = self.theta_star if self.theta0 is None else np.broadcast_to(np.asarray(self.theta0, float), (n,))
        if self.groups_in is None:
            self.g = np.arange(n)
        else:
            _, self.g = np.unique(np.asarray(self.groups_in), return_inverse=True)
        self.n_groups = int(self.g.max()) + 1
        g0 = np.full(self.n_groups, np.inf)
        np.minimum.at(g0, self.g, start)
        self.theta = g0
        self.max_step = 0              # most same-second updates one group received (for the bound)
        self.n = n
        self.theta_trace = []          # sampled for plotting risk control

    def keepalive(self) -> np.ndarray:
        return np.full(self.n, self.ka_min)

    def lead(self) -> np.ndarray:
        return self._lead

    def hold(self) -> np.ndarray:
        return np.full(self.n, self.hold_in)

    def effective_threshold(self) -> np.ndarray:
        if self.alpha is None:
            return self.theta_star
        return np.minimum(self.theta_star, self.theta[self.g])

    def want_warm(self, t: int) -> np.ndarray:
        return self.P[:, t] >= self.effective_threshold()

    def observe(self, t: int, invoked: np.ndarray, cold: np.ndarray) -> None:
        if self.alpha is None or not invoked.any():
            return
        idx = np.flatnonzero(invoked)
        gi = self.g[idx]
        err = np.bincount(gi, weights=cold[idx].astype(float) - self.alpha, minlength=self.n_groups)
        hits = np.bincount(gi, minlength=self.n_groups)
        self.max_step = max(self.max_step, int(hits.max()))
        touched = hits > 0
        self.theta = np.where(touched, np.minimum(1.0, self.theta - self.eta * err), self.theta)
        if t % 60 == 0:
            self.theta_trace.append((t, float(np.median(self.theta))))


class StateAwarePolicy(ThresholdPolicy):
    """Uses a different prediction depending on the instance's state (paper: state-aware targets).

    * Instance warm or loading -> ``P_warm``: will the function be called in the next hold period?
      (the near-term question: is it worth keeping the instance?)
    * No instance -> ``P_cold``: will it be called after its cold-start lead? (only a call that arrives
      once loading has finished can be served by a start made now)

    The simulator sets ``active_now`` before every call of ``want_warm``.
    """

    def __init__(self, P_warm: np.ndarray, P_cold: np.ndarray, theta_star, theta_cold=None, **kw):
        super().__init__(P_warm, theta_star, **kw)
        if P_cold.shape != P_warm.shape:
            raise ValueError("P_warm and P_cold must have the same shape")
        self.P_cold = P_cold
        self.theta_cold = theta_star if theta_cold is None else theta_cold

    def want_warm(self, t: int) -> np.ndarray:
        act = getattr(self, "active_now", None)
        if act is None:
            return super().want_warm(t)
        th = self.effective_threshold()
        return np.where(act, self.P[:, t] >= th, self.P_cold[:, t] >= np.minimum(self.theta_cold, th))
