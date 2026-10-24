"""Early-signal test (paper, RQ3): do other functions of the same application warn of a cold-prone call?

Uses data/processed/azure2021_events.parquet. Frequency classes are measured on days 1-9; calls are
evaluated on days 10-14. Writes results/early_signal/azure2021.csv and prints a summary.
"""
from pathlib import Path

import pandas as pd

from stawarm.analysis import early_signal_table, frequency_buckets_from_events

DAY = 86400.0
ev = pd.read_parquet("data/processed/azure2021_events.parquet")
bucket = frequency_buckets_from_events(ev, 0, 9 * DAY)
print("functions per bucket (days 1-9):", bucket.value_counts().to_dict())
tab = early_signal_table(ev, bucket, 9 * DAY, ev["start"].max() + 1,
                         leads=(1, 2, 5, 10), windows=(10, 30, 60, 300), gap_s=600)
out = Path("results/early_signal"); out.mkdir(parents=True, exist_ok=True)
tab.to_csv(out / "azure2021.csv", index=False)
pd.set_option("display.width", 160)
print(tab.round(3).to_string(index=False))

# Decision rule written in the plan before seeing the data: lift >= 2 for rare functions
# with at least 2 seconds of warning inside a 60-second window.
row = tab[(tab.bucket == "rare") & (tab.lead_s == 2) & (tab.window_s == 60)]
if row.empty or row.n_calls.iloc[0] < 50:
    print("\nRESULT: not enough rare-function calls in multi-function apps to decide.")
elif row.lift.iloc[0] >= 2:
    print(f"\nRESULT: early signal for rare functions (lift {row.lift.iloc[0]:.2f}); run early_signal_controls.py to test it against the controls.")
else:
    print(f"\nRESULT: no useful early signal for rare functions (lift {row.lift.iloc[0]:.2f}).")
