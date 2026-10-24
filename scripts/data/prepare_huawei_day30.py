"""Turn Huawei Region 2 day-30 request tables into the same event format as Azure 2021.

Reads every CSV/Parquet file under data/raw/huawei/requests_region2_part*/ in chunks (the tables can
hold hundreds of millions of requests), and writes data/processed/huawei_r2_day30_events.parquet with
columns fid, app, start, duration.

Memory: the early-signal analysis only needs to know *whether* a function was called in a given
second, not how many times. Each chunk is therefore reduced to unique (function, second) pairs before
it is kept, which shrinks busy functions from millions of rows to at most 86,400 per day.

Huawei has no 'application' field. The user (tenant) ID plays that role: functions of the same user
are the candidate neighbours. Timestamps are the original per-request times.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/huawei")
TIME_COLS = ["time_frontend", "time_worker"]
KEY_COLS = ["funcName", "userID", "podID"]
CHUNK = 5_000_000


def reduce_chunk(df: pd.DataFrame) -> pd.DataFrame:
    tcol = next(c for c in TIME_COLS if c in df.columns)
    pool = df["podID"].astype(str).str.extract(r"(pool\d+-\d+-\d+)", expand=False).fillna("pool?")
    fid = df["funcName"].astype(str) + "---" + df["userID"].astype(str) + "---" + pool
    sec = np.floor(df[tcol].to_numpy(dtype=np.float64))
    out = pd.DataFrame({"fid": fid.to_numpy(), "user": df["userID"].astype(str).to_numpy(), "second": sec})
    out = out[np.isfinite(out["second"])]
    return out.drop_duplicates(["fid", "second"])


def iter_chunks(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = [c for c in TIME_COLS + KEY_COLS if c in pf.schema.names]
        for batch in pf.iter_batches(batch_size=CHUNK, columns=cols):
            yield batch.to_pandas()
    else:
        head = pd.read_csv(path, nrows=0).columns
        cols = [c for c in TIME_COLS + KEY_COLS if c in head]
        missing = set(KEY_COLS) - set(cols)
        if missing or not any(c in cols for c in TIME_COLS):
            raise SystemExit(f"{path.name}: unexpected columns {list(head)}.")
        yield from pd.read_csv(path, usecols=cols, chunksize=CHUNK)


def main() -> None:
    files = sorted(p for p in RAW.glob("requests_region2_part*/**/*")
                   if p.is_file() and (p.suffix in (".csv", ".parquet") or p.name.endswith(".csv.gz")))
    if not files:
        raise SystemExit("No request files found. Run: python scripts/data/fetch_traces.py huawei --requests")
    t0 = time.time()
    kept, n_rows = [], 0
    for k, p in enumerate(files, 1):
        for chunk in iter_chunks(p):
            n_rows += len(chunk)
            kept.append(reduce_chunk(chunk))
        print(f"[{k}/{len(files)}] {p.name}: {n_rows:,} requests read so far ({time.time() - t0:.0f} s)")
    ev = pd.concat(kept, ignore_index=True).drop_duplicates(["fid", "second"])
    ev["start"] = ev["second"] - ev["second"].min()
    out = pd.DataFrame({"fid": pd.factorize(ev["fid"], sort=True)[0].astype(np.int32),
                        "app": pd.factorize(ev["user"], sort=True)[0].astype(np.int32),
                        "start": ev["start"].to_numpy(np.float64),
                        "duration": np.float32(np.nan)}).sort_values("start").reset_index(drop=True)
    dst = Path("data/processed"); dst.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst / "huawei_r2_day30_events.parquet")
    per_app = out.groupby("app")["fid"].nunique()
    print(f"\nrequests read: {n_rows:,}   kept as (function, second) events: {len(out):,}")
    print(f"users: {out['app'].nunique():,}   functions: {out['fid'].nunique():,}")
    print(f"time span: {out['start'].max() / 3600:.1f} hours")
    print(f"users with 2+ functions: {(per_app >= 2).sum():,} of {len(per_app):,}")
    print(f"took {time.time() - t0:.0f} s; wrote {dst / 'huawei_r2_day30_events.parquet'}")


if __name__ == "__main__":
    main()
