"""Event-level analyses that do not need a dense functions x seconds matrix.

The main one is the *early-signal test* (paper, RQ3):

    For a call of function f at time t that would be a cold start (its previous call was more than
    ``gap_s`` seconds earlier), is some *other* function of the same application invoked in the
    window [t - window_s, t - lead_s]? Compare that probability with the same probability at random
    times. The ratio is the "lift". A lift near 1 means the other functions carry no early warning;
    a lift of 2 or more means they do.

Everything is computed with sorted arrays and binary search, so two weeks of per-second data fit in
16 GB of memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cold_prone_calls(starts: np.ndarray, gap_s: float) -> np.ndarray:
    """Calls whose previous call of the same function was more than gap_s earlier (first call included)."""
    starts = np.sort(starts)
    prev = np.concatenate([[-np.inf], starts[:-1]])
    return starts[(starts - prev) > gap_s]


def _count_in(sorted_times: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Number of times in [lo, hi] for each (lo, hi)."""
    return np.searchsorted(sorted_times, hi, side="right") - np.searchsorted(sorted_times, lo, side="left")


def early_signal_table(events: pd.DataFrame, bucket: pd.Series, t_from: float, t_to: float,
                       leads=(1, 2, 5, 10), windows=(10, 30, 60, 300), gap_s: float = 600.0,
                       n_random: int = 200, seed: int = 0) -> pd.DataFrame:
    """Lift of 'another function of the same app was invoked shortly before' at cold-prone calls.

    events : DataFrame with columns fid, app, start (seconds).
    bucket : Series fid -> frequency bucket ("rare", "medium", "frequent").
    Only calls with t_from <= t < t_to are evaluated; earlier events still count as history.
    Returns one row per (bucket, lead, window) with counts and the lift.
    """
    rng = np.random.default_rng(seed)
    rows = []
    acc = {}
    for app, g in events.groupby("app", sort=False):
        fids = g["fid"].unique()
        if len(fids) < 2:
            continue                                   # no neighbours inside this app
        all_t = np.sort(g["start"].to_numpy())
        for fid, gf in g.groupby("fid", sort=False):
            b = bucket.get(fid)
            if b is None:
                continue
            own = np.sort(gf["start"].to_numpy())
            q = cold_prone_calls(own, gap_s)
            q = q[(q >= t_from) & (q < t_to)]
            r = rng.uniform(t_from, t_to, n_random)
            for L in leads:
                for W in windows:
                    if W <= L:
                        continue
                    for kind, times in (("call", q), ("random", r)):
                        if len(times) == 0:
                            continue
                        lo, hi = times - W, times - L
                        others = _count_in(all_t, lo, hi) - _count_in(own, lo, hi)
                        key = (b, L, W, kind)
                        hit, n = acc.get(key, (0, 0))
                        acc[key] = (hit + int((others > 0).sum()), n + len(times))
    for (b, L, W, kind), (hit, n) in acc.items():
        if kind != "call":
            continue
        rh, rn = acc.get((b, L, W, "random"), (0, 0))
        p_call = hit / n if n else np.nan
        p_rand = rh / rn if rn else np.nan
        rows.append({"bucket": b, "lead_s": L, "window_s": W, "n_calls": n, "p_signal_at_call": p_call,
                     "p_signal_random": p_rand, "lift": p_call / p_rand if p_rand else np.nan})
    return pd.DataFrame(rows).sort_values(["bucket", "lead_s", "window_s"]).reset_index(drop=True)


def frequency_buckets_from_events(events: pd.DataFrame, t_from: float, t_to: float) -> pd.Series:
    """Rare (<10), medium (10-1000), frequent (>1000) active minutes per day, measured in [t_from, t_to)."""
    e = events[(events["start"] >= t_from) & (events["start"] < t_to)]
    days = (t_to - t_from) / 86400.0
    active = e.assign(minute=(e["start"] // 60).astype(np.int64)).groupby("fid")["minute"].nunique() / days
    out = pd.Series("medium", index=active.index, dtype=object)
    out[active < 10] = "rare"
    out[active > 1000] = "frequent"
    return out


# --------------------------------------------------------------------------- controls for the early-signal test

def _baseline_times(kind: str, rng, t_from: float, t_to: float, calls: np.ndarray, app_times: np.ndarray,
                    n: int) -> np.ndarray:
    """Random comparison times.

    uniform    : anywhere in the test period
    same_hour  : same time of day as a real call, on a random test day (removes daily rhythm)
    app_active : within 10 minutes after some activity of the same app (removes 'the app is busy')
    """
    if kind == "uniform":
        return rng.uniform(t_from, t_to, n)
    if kind == "same_hour":
        if len(calls) == 0:
            return np.empty(0)
        tod = rng.choice(calls % 86400.0, n)
        days = rng.integers(int(t_from // 86400), int(np.ceil(t_to / 86400)), n)
        t = days * 86400.0 + tod
        return t[(t >= t_from) & (t < t_to)]
    if kind == "app_active":
        base = app_times[(app_times >= t_from - 600) & (app_times < t_to)]
        if len(base) == 0:
            return np.empty(0)
        t = rng.choice(base, n) + rng.uniform(0, 600, n)
        return t[(t >= t_from) & (t < t_to)]
    raise ValueError(kind)


def early_signal_controls(events: pd.DataFrame, bucket: pd.Series, t_from: float, t_to: float,
                          lead_s: float = 2, window_s: float = 60, gap_s: float = 600,
                          n_random: int = 400, seed: int = 0) -> pd.DataFrame:
    """Controls for one (lead, window) setting, per bucket:

    p_before       : share of cold-prone calls with other same-app activity in [t - W, t - L]
    p_after        : same, but in [t + L, t + W]  (direction control: co-activity vs early warning)
    p_uniform, p_same_hour, p_app_active : share of random baseline times with activity in [t - W, t - L]
    lift_*         : p_before / p_baseline
    precision      : share of other-function events in the test period that are followed, within
                     [e + L, e + W], by a target call that would otherwise have been cold ('hints that pay off')
    precision_any  : same, counting any target call (including ones a keep-alive would already serve)
    hints_per_hit  : 1 / precision, i.e. pre-warms per cold start avoided if every hint triggered one
    """
    rng = np.random.default_rng(seed)
    acc: dict = {}

    def add(b, key, hit, n):
        h, m = acc.setdefault(b, {}).get(key, (0, 0))
        acc[b][key] = (h + hit, m + n)

    for app, g in events.groupby("app", sort=False):
        if g["fid"].nunique() < 2:
            continue
        all_t = np.sort(g["start"].to_numpy())
        for fid, gf in g.groupby("fid", sort=False):
            b = bucket.get(fid)
            if b is None:
                continue
            own = np.sort(gf["start"].to_numpy())
            q = cold_prone_calls(own, gap_s)
            q = q[(q >= t_from) & (q < t_to)]

            def others_in(lo, hi):
                return (_count_in(all_t, lo, hi) - _count_in(own, lo, hi)) > 0

            if len(q):
                add(b, "before", int(others_in(q - window_s, q - lead_s).sum()), len(q))
                add(b, "after", int(others_in(q + lead_s, q + window_s).sum()), len(q))
            for kind in ("uniform", "same_hour", "app_active"):
                r = _baseline_times(kind, rng, t_from, t_to, q, all_t, n_random)
                if len(r):
                    add(b, kind, int(others_in(r - window_s, r - lead_s).sum()), len(r))
            # precision of the rule "pre-warm the target whenever another function of the app fires"
            other = np.setdiff1d(all_t[(all_t >= t_from) & (all_t < t_to)], own, assume_unique=False)
            if len(other):
                followed = _count_in(q, other + lead_s, other + window_s) > 0
                add(b, "precision", int(followed.sum()), len(other))
                in_test = own[(own >= t_from) & (own < t_to)]
                followed_any = _count_in(in_test, other + lead_s, other + window_s) > 0
                add(b, "precision_any", int(followed_any.sum()), len(other))
    rows = []
    for b, d in acc.items():
        def p(k):
            h, n = d.get(k, (0, 0))
            return h / n if n else np.nan
        row = {"bucket": b, "lead_s": lead_s, "window_s": window_s, "n_calls": d.get("before", (0, 0))[1],
               "p_before": p("before"), "p_after": p("after"), "p_uniform": p("uniform"),
               "p_same_hour": p("same_hour"), "p_app_active": p("app_active"), "precision": p("precision"),
               "precision_any": p("precision_any")}
        for k in ("uniform", "same_hour", "app_active"):
            row[f"lift_{k}"] = row["p_before"] / row[f"p_{k}"] if row[f"p_{k}"] else np.nan
        row["hints_per_hit"] = 1 / row["precision"] if row["precision"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
