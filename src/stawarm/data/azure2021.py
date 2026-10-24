"""Loader for the Azure Functions invocation trace 2021 (Zhang et al., SOSP 2021).

Download (about 150 MB .rar, stored with Git LFS; scripts/data/fetch_traces.py tries it, otherwise download it in a browser):
https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar

Schema: app, func, end_timestamp (s), duration (s). Two weeks starting 2021-01-31.
Unlike the 2019 trace, this one has per-invocation timestamps, so it supports second-level slots,
which is what the lead-time experiments need. The invocation start is end_timestamp - duration.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .common import FunctionTrace


def read_invocations(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read the extracted CSV/TXT file (comma separated, with a header row)."""
    df = pd.read_csv(path, nrows=nrows, dtype={"app": str, "func": str,
                                               "end_timestamp": np.float64, "duration": np.float64})
    df["start"] = df["end_timestamp"] - df["duration"]
    return df


def to_trace(df: pd.DataFrame, slot_seconds: float = 1.0, t_start: Optional[float] = None,
             t_end: Optional[float] = None, min_invocations: int = 1) -> FunctionTrace:
    """Bin invocation start times into slots. Functions are keyed by (app, func)."""
    t_start = df["start"].min() if t_start is None else t_start
    t_end = df["start"].max() + slot_seconds if t_end is None else t_end
    d = df[(df["start"] >= t_start) & (df["start"] < t_end)]
    key = d["app"] + "|" + d["func"]
    vc = key.value_counts()
    keep = vc[vc >= min_invocations].index
    d = d[key.isin(keep)]
    key = key[key.isin(keep)]
    codes, uniques = pd.factorize(key, sort=True)
    slot = ((d["start"].to_numpy() - t_start) // slot_seconds).astype(np.int64)
    n_slots = int(np.ceil((t_end - t_start) / slot_seconds))
    counts = np.zeros((len(uniques), n_slots), dtype=np.int32)
    np.add.at(counts, (codes, slot), 1)
    app = [u.split("|")[0] for u in uniques]
    meta = pd.DataFrame({"fid": list(uniques), "app": app, "owner": app})
    return FunctionTrace(meta, counts, slot_seconds, t0=f"{t_start:.0f}s")
