"""Loaders for the Huawei Public Cloud Trace 2025 (Joosen et al., EuroSys 2025).

Files are on Google Drive; links are in
https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md
Download them with scripts/data/fetch_traces.py, or in a browser if Google Drive refuses scripted downloads.

Tables and how this project uses them (schemas follow the README above):
  * cold start events, 31 days, all 5 regions:
      day, time, clusterName, funcName, userID, requestID, totalCost_cold_start,
      podAllocationCost, deployCodeCost, deployDependencyCost, schedulingCost, podID
    -> per-function cold-start duration quantiles d_i (use totalCost_cold_start).
  * per-minute time series, 31 days, all regions, keyed by funcID = funcName---userID---poolName
    -> 31-day minute-level invocation counts (metric ``requests``).
  * request tables, day 30 only, per-request timestamps (time_worker, time_frontend)
    -> second-level slots for one day.
  * runtimes and trigger types, Region 2 only (funcID, cpu_request, runtime, triggerType-invocationType)
    -> synchronous/asynchronous split for the latency penalty beta_i.

The exact file layout inside the downloaded archives may differ from these column names;
check with ``peek`` first and adapt ``COLUMN_ALIASES`` if needed.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import FunctionTrace

COLUMN_ALIASES = {"totalCost_cold_start": ["totalCost_cold_start", "totalCost"],
                  "time": ["time", "timestamp"]}
POOL_RE = re.compile(r"pool\d+-(\d+)-(\d+)")


def peek(path: str, n: int = 5) -> pd.DataFrame:
    """Show the first rows of any CSV/Parquet file to confirm the schema."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p).head(n)
    return pd.read_csv(p, nrows=n)


def _col(df: pd.DataFrame, name: str) -> str:
    for c in COLUMN_ALIASES.get(name, [name]):
        if c in df.columns:
            return c
    raise KeyError(f"none of {COLUMN_ALIASES.get(name, [name])} in {list(df.columns)}")


def pool_resources(pod_or_pool: str) -> tuple[Optional[int], Optional[int]]:
    """'pool22-300-128-0000618412' -> (300 millicores, 128 MB)."""
    m = POOL_RE.search(str(pod_or_pool))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def function_id(func_name, user_id, pod_or_pool) -> str:
    m = POOL_RE.search(str(pod_or_pool))
    pool = m.group(0) if m else "pool?"
    return f"{func_name}---{user_id}---{pool}"


def cold_start_quantiles(cold: pd.DataFrame, q: float = 0.9, train_days: Optional[range] = None) -> pd.DataFrame:
    """Per-function cold-start duration statistics, computed on training days only.

    Returns columns: fid, cold_start_s (q-quantile), cold_start_median_s, n_cold, mem_mb, cpu_milli.
    """
    df = cold
    if train_days is not None:
        df = df[df["day"].isin(list(train_days))]
    tc = _col(df, "totalCost_cold_start")
    fid = [function_id(f, u, p) for f, u, p in zip(df["funcName"], df["userID"], df["podID"])]
    df = df.assign(fid=fid)
    g = df.groupby("fid")[tc]
    out = pd.DataFrame({"cold_start_s": g.quantile(q), "cold_start_median_s": g.median(), "n_cold": g.size()})
    res = [pool_resources(f.split("---")[-1]) for f in out.index]
    out["cpu_milli"] = [r[0] for r in res]
    out["mem_mb"] = [r[1] for r in res]
    return out.reset_index()


def requests_to_trace(req: pd.DataFrame, slot_seconds: float = 1.0, time_col: str = "time_frontend") -> FunctionTrace:
    """Day-30 request table -> FunctionTrace with fine-grained slots."""
    fid = [function_id(f, u, p) for f, u, p in zip(req["funcName"], req["userID"], req["podID"])]
    t = req[time_col].to_numpy(dtype=float)
    t0 = np.floor(t.min())
    codes, uniques = pd.factorize(pd.Series(fid), sort=True)
    slot = ((t - t0) // slot_seconds).astype(np.int64)
    counts = np.zeros((len(uniques), int(slot.max()) + 1), dtype=np.int32)
    np.add.at(counts, (codes, slot), 1)
    parts = [u.split("---") for u in uniques]
    meta = pd.DataFrame({"fid": list(uniques), "func": [p[0] for p in parts], "owner": [p[1] for p in parts],
                         "mem_mb": [pool_resources(p[2])[1] for p in parts]})
    return FunctionTrace(meta, counts, slot_seconds, t0=f"{t0:.0f}s")


def timeseries_to_trace(ts: pd.DataFrame, metric: str = "requests", fid_col: str = "funcID",
                        time_col: str = "time") -> FunctionTrace:
    """Per-minute time-series table (long format: funcID, time, metric...) -> one-minute FunctionTrace.

    If the downloaded file is already wide (one column per minute), pivot it before calling this.
    """
    wide = ts.pivot_table(index=fid_col, columns=time_col, values=metric, aggfunc="sum", fill_value=0)
    counts = wide.to_numpy(dtype=np.int64).clip(0, np.iinfo(np.int32).max).astype(np.int32)
    fids = wide.index.astype(str)
    parts = [f.split("---") for f in fids]
    meta = pd.DataFrame({"fid": fids, "owner": [p[1] if len(p) > 1 else "" for p in parts],
                         "mem_mb": [pool_resources(p[2])[1] if len(p) > 2 else None for p in parts]})
    return FunctionTrace(meta, counts, 60.0)
