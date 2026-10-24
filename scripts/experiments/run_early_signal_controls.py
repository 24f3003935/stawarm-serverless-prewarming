"""Controls for the early-signal test (paper, RQ3): is the signal real, directional and useful?

Run after run_early_signal_test.py. Writes results/early_signal/azure2021_controls.csv and prints a
summary.

What the columns mean (per bucket, for each warning time 'lead' and window):
  p_before      share of would-be-cold calls with other same-app activity just BEFORE the call
  p_after       same, just AFTER the call. If p_after is about as high as p_before, the functions
                fire together; the hint is co-activity, not an early warning.
  lift_uniform  p_before vs random moments (as in run_early_signal_test.py)
  lift_same_hour  vs random moments at the same time of day (removes daily rhythm)
  lift_app_active vs random moments when the app was active in the last 10 minutes (removes 'busy app')
  precision     share of other-function calls that are followed by a would-be-cold call of the target:
                how often a pre-warm triggered by the hint would actually be used
  hints_per_hit pre-warms needed per cold start avoided, if every hint triggered a pre-warm
"""
import argparse
from pathlib import Path

import pandas as pd

from stawarm.analysis import early_signal_controls, frequency_buckets_from_events

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["azure2021", "huawei_day30"], default="azure2021")
args = ap.parse_args()
if args.dataset == "azure2021":
    ev = pd.read_parquet("data/processed/azure2021_events.parquet")
    split = 9 * 86400.0                      # buckets from days 1-9, test on days 10-14
else:
    ev = pd.read_parquet("data/processed/huawei_r2_day30_events.parquet")
    split = 16 * 3600.0                      # one day: buckets from the first 16 h, test on the last 8 h
bucket = frequency_buckets_from_events(ev, 0, split)
t_end = ev["start"].max() + 1
parts = []
for lead, window in [(1, 10), (2, 60), (5, 60), (10, 300)]:
    parts.append(early_signal_controls(ev, bucket, split, t_end, lead_s=lead, window_s=window, gap_s=600))
tab = pd.concat(parts, ignore_index=True)
out = Path("results/early_signal"); out.mkdir(parents=True, exist_ok=True)
tab.to_csv(out / f"{args.dataset}_controls.csv", index=False)
print("dataset:", args.dataset, "| functions per bucket:", bucket.value_counts().to_dict())
pd.set_option("display.width", 200)
cols = ["bucket", "lead_s", "window_s", "n_calls", "p_before", "p_after", "lift_uniform", "lift_same_hour",
        "lift_app_active", "precision", "hints_per_hit"]
print(tab[cols].round(3).to_string(index=False))

r = tab[(tab.bucket == "rare") & (tab.lead_s == 2) & (tab.window_s == 60)]
if not r.empty:
    r = r.iloc[0]
    print("\nRare functions, 2 s warning, 60 s window:")
    print(f"  direction  : {r.p_before:.1%} of calls have activity before vs {r.p_after:.1%} after")
    print(f"  busy-app control lift: {r.lift_app_active:.2f}  (>= 2 means the signal survives the control)")
    print(f"  usefulness : {r.hints_per_hit:.1f} pre-warms per cold start avoided if every hint is used")
