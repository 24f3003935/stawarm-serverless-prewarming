"""Per-function cold-start durations and memory sizes from Huawei Cloud Region 2 (days 1-21).

These profiles set every function's lead time and memory size in the experiments. Reads
data/raw/huawei/cold_starts_region2/ and, if present, the runtime/trigger table. Writes
data/processed/huawei_r2_cold_start_stats.parquet and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from stawarm.data.huawei import function_id, pool_resources

RAW = Path("data/raw/huawei")
NEED = ["day", "funcName", "userID", "podID", "totalCost_cold_start"]
PARTS = ["podAllocationCost", "deployCodeCost", "deployDependencyCost", "schedulingCost"]


def table_files(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and
                  (p.suffix in (".csv", ".parquet") or p.name.endswith(".csv.gz")))


def read_any(p: Path, cols):
    if p.suffix == ".parquet":
        return pd.read_parquet(p, columns=cols)
    head = pd.read_csv(p, nrows=0).columns
    missing = set(NEED) - set(head)
    if missing:
        raise SystemExit(f"{p.name}: columns are {list(head)}; missing {missing}.")
    return pd.read_csv(p, usecols=[c for c in cols if c in head])


files = table_files(RAW / "cold_starts_region2")
if not files:
    raise SystemExit("No cold-start files found. Run: python scripts/data/fetch_traces.py huawei")
cols = NEED + PARTS
cs = pd.concat([read_any(p, cols) for p in files], ignore_index=True)
print(f"cold-start events: {len(cs):,} over days {int(cs['day'].min())}-{int(cs['day'].max())}")
train = cs[cs["day"] <= 20]                     # days 0-20 = the first 21 days
pool = train["podID"].astype(str).str.extract(r"(pool\d+-\d+-\d+)", expand=False).fillna("pool?")
train = train.assign(fid=train["funcName"].astype(str) + "---" + train["userID"].astype(str) + "---" + pool)
g = train.groupby("fid")["totalCost_cold_start"]
stats = pd.DataFrame({"n_cold": g.size(), "d50_s": g.median(), "d90_s": g.quantile(0.9), "d99_s": g.quantile(0.99)})
for c in PARTS:
    if c in train.columns:
        stats[f"{c}_median_s"] = train.groupby("fid")[c].median()
res = [pool_resources(f.split("---")[-1]) for f in stats.index]
stats["cpu_milli"] = [r[0] for r in res]
stats["mem_mb"] = [r[1] for r in res]

rt_files = table_files(RAW / "runtimes_triggers_region2")
if rt_files:
    rt = pd.concat([pd.read_csv(p) if p.suffix != ".parquet" else pd.read_parquet(p) for p in rt_files])
    rt = rt.drop_duplicates("funcID").set_index("funcID")
    stats = stats.join(rt, how="left")
    trig = stats.get("triggerType-invocationType")
    if trig is not None:
        stats["synchronous"] = trig.astype(str).str.contains("-S")

out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
stats.reset_index(names="fid").to_parquet(out / "huawei_r2_cold_start_stats.parquet")

pd.set_option("display.width", 160)
print(f"functions with cold starts in days 1-21: {len(stats):,}")
print("\ncold-start time per function, 90th percentile (seconds), across functions:")
print(stats["d90_s"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
for s in (0.5, 1, 2, 5):
    print(f"  functions whose d90 exceeds {s:>3} s: {(stats['d90_s'] > s).mean():.1%}")
if "runtime" in stats.columns:
    print("\nmedian of d90 by runtime:")
    print(stats.groupby("runtime")["d90_s"].agg(["count", "median"]).sort_values("count", ascending=False)
          .head(10).round(3).to_string())
if "synchronous" in stats.columns:
    print("\nshare of functions with a synchronous trigger:", f"{stats['synchronous'].mean():.1%}")
print("\nwrote", out / "huawei_r2_cold_start_stats.parquet")
