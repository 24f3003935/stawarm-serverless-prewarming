"""Convert the extracted Azure Functions 2021 trace into a compact event table.

Reads the largest CSV/TXT file under data/raw/azure2021/ and writes data/processed/azure2021_events.parquet
with columns fid, app, start, duration (start = end timestamp - duration, in seconds from the start of the
trace). Prints a short summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

src_dir = Path("data/raw/azure2021")
files = sorted((p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".csv", ".txt")),
               key=lambda p: p.stat().st_size, reverse=True)
if not files:
    raise SystemExit("No CSV/TXT file found under data/raw/azure2021 - run: python scripts/data/fetch_traces.py azure2021")
src = files[0]
print("reading", src)
df = pd.read_csv(src, dtype={"app": "string", "func": "string"})
missing = {"app", "func", "end_timestamp", "duration"} - set(df.columns)
if missing:
    raise SystemExit(f"unexpected columns {list(df.columns)}; missing {missing}.")
df["start"] = df["end_timestamp"] - df["duration"]
df["start"] -= df["start"].min()
app_codes, _ = pd.factorize(df["app"], sort=True)
fid_codes, _ = pd.factorize(df["app"] + "|" + df["func"], sort=True)
ev = pd.DataFrame({"fid": fid_codes.astype(np.int32), "app": app_codes.astype(np.int32),
                   "start": df["start"].to_numpy(np.float64), "duration": df["duration"].to_numpy(np.float32)})
ev = ev.sort_values("start").reset_index(drop=True)
out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
ev.to_parquet(out / "azure2021_events.parquet")

days = ev["start"].max() / 86400
per_app = ev.groupby("app")["fid"].nunique()
print(f"\ninvocations: {len(ev):,}")
print(f"applications: {ev['app'].nunique():,}   functions: {ev['fid'].nunique():,}")
print(f"time span: {days:.2f} days")
print(f"apps with 2+ functions: {(per_app >= 2).sum():,} of {len(per_app):,}")
print("functions per app (quantiles 50/90/max):", per_app.quantile([.5, .9]).tolist(), per_app.max())
print("wrote", out / "azure2021_events.parquet")
