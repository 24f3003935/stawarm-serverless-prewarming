"""Loader for the Azure Functions 2019 trace (Shahrad et al., USENIX ATC 2020).

Download (about 140 MB compressed):
https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz

Files used: ``invocations_per_function_md.anon.dNN.csv`` (per-minute counts, one file per day),
``app_memory_percentiles.anon.dNN.csv`` (average allocated memory per app, days 1-12 only).
Slots are one minute long. A function is identified by (HashOwner, HashApp, HashFunction).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .common import FunctionTrace

MINUTES = 1440
KEYS = ["HashOwner", "HashApp", "HashFunction"]


def _read_day(root: Path, day: int) -> pd.DataFrame:
    f = root / f"invocations_per_function_md.anon.d{day:02d}.csv"
    cols = {str(m): np.uint32 for m in range(1, MINUTES + 1)}
    df = pd.read_csv(f, dtype={**cols, "HashOwner": str, "HashApp": str, "HashFunction": str, "Trigger": str})
    # A few functions appear on several rows (one per trigger); sum them.
    minute_cols = [str(m) for m in range(1, MINUTES + 1)]
    trig = df.groupby(KEYS)["Trigger"].first()
    agg = df.groupby(KEYS)[minute_cols].sum()
    return pd.concat([agg, trig], axis=1)       # one concat avoids pandas' fragmentation warning


def _read_app_memory(root: Path, days: Iterable[int]) -> pd.Series:
    vals = []
    for d in days:
        f = root / f"app_memory_percentiles.anon.d{d:02d}.csv"
        if f.exists():
            m = pd.read_csv(f, usecols=["HashOwner", "HashApp", "AverageAllocatedMb"])
            vals.append(m)
    if not vals:
        return pd.Series(dtype=float)
    m = pd.concat(vals).groupby(["HashOwner", "HashApp"])["AverageAllocatedMb"].median()
    return m


def load_azure2019(root: str, days: Iterable[int], select: Optional[pd.Index] = None,
                   default_mem_mb: float = 170.0) -> FunctionTrace:
    """Load consecutive days into a FunctionTrace with one-minute slots.

    Parameters
    ----------
    root : folder with the extracted CSV files.
    days : day numbers (1..14), in order.
    select : optional MultiIndex of (HashOwner, HashApp, HashFunction) to keep; reduces memory.
    default_mem_mb : memory used when an app has no memory record (the trace's median app uses
        roughly this order of magnitude; the value only scales wasted memory-time).
    """
    root = Path(root)
    days = list(days)
    frames = []
    for d in days:
        day = _read_day(root, d)
        if select is not None:
            day = day[day.index.isin(select)]
        frames.append(day)
    index = frames[0].index
    for fr in frames[1:]:
        index = index.union(fr.index)
    if select is not None:
        index = index.intersection(select)
    minute_cols = [str(m) for m in range(1, MINUTES + 1)]
    counts = np.zeros((len(index), MINUTES * len(days)), dtype=np.int32)
    trigger = pd.Series(index=index, dtype=object)
    for k, fr in enumerate(frames):
        fr = fr.reindex(index)
        block = fr[minute_cols].to_numpy(dtype=np.float64, na_value=0.0)
        counts[:, k * MINUTES:(k + 1) * MINUTES] = np.minimum(block, np.iinfo(np.int32).max).astype(np.int32)
        trigger = trigger.fillna(fr["Trigger"])
    meta = index.to_frame(index=False)
    meta.columns = ["owner", "app", "func"]
    meta["fid"] = meta["owner"].str[:8] + "/" + meta["app"].str[:8] + "/" + meta["func"].str[:8]
    meta["trigger"] = trigger.to_numpy()
    mem = _read_app_memory(root, days)
    if len(mem):
        key = pd.MultiIndex.from_frame(meta[["owner", "app"]])
        meta["mem_mb"] = mem.reindex(key).to_numpy()
    else:
        meta["mem_mb"] = np.nan
    meta["mem_mb"] = meta["mem_mb"].fillna(default_mem_mb)
    return FunctionTrace(meta, counts, 60.0, t0=f"d{days[0]:02d}")


def function_index(root: str, day: int) -> pd.DataFrame:
    """Per-function totals for one day, used to choose a sample before loading many days."""
    day_df = _read_day(Path(root), day)
    minute_cols = [str(m) for m in range(1, MINUTES + 1)]
    x = day_df[minute_cols].to_numpy()
    out = pd.DataFrame({"active_minutes": (x > 0).sum(1), "invocations": x.sum(1),
                        "trigger": day_df["Trigger"].to_numpy()}, index=day_df.index)
    return out
