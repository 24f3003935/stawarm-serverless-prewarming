"""Overhead of the streaming STAWarm model as the number of functions grows (paper, RQ8).

For each N, times (a) inference for one hour of per-second decisions and (b) one training step on a
300-second window, with and without attention. Inputs are random; only the cost is measured.
Applications have 4 functions each (K = 8 neighbours at most, so attention cost is what it would be).

Run : python scripts/experiments/run_overhead_benchmark.py            (a few minutes)
Out : results/benchmark_overhead.csv
"""
from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stawarm.models.stawarm_stream import STAWarmStream, neighbour_table

F = 13            # inputs of the final model: 8 own, 4 application, 1 cold-start time
SIZE = dict(hidden=64, layers=1, key_dim=32, node_dim=16, value_dim=32, head_hidden=64)   # validation-selected size


def main() -> None:
    rows = []
    for N in [100, 424, 1000, 1612]:
        app = np.arange(N) // 4
        nb, valid = neighbour_table(app, np.ones(N), 8)
        nb, valid = torch.from_numpy(nb), torch.from_numpy(valid)
        for att in [False, True]:
            torch.manual_seed(0)
            m = STAWarmStream(F, N, attention=att, **SIZE)
            opt = torch.optim.Adam(m.parameters())
            x = torch.randn(N, 300, F)
            m(x[:, :50], None, nb, valid)                     # warm-up
            t0 = time.perf_counter()
            for _ in range(2):
                logit, _ = m(x, None, nb, valid)
                logit.mean().backward(); opt.step(); opt.zero_grad()
            train_step = (time.perf_counter() - t0) / 2
            m.eval()
            xb = torch.randn(N, 300, F)
            t0 = time.perf_counter()
            with torch.no_grad():
                state = None
                for _ in range(12):                            # one hour in 300-s blocks
                    _, state = m(xb, state, nb, valid)
            hour = time.perf_counter() - t0
            rows.append({"functions": N, "attention": att, "train_step_s": train_step,
                         "inference_s_per_hour": hour, "us_per_decision": 1e6 * hour / (N * 3600)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["cpu"] = f"{platform.processor() or platform.machine()} ({os.cpu_count()} logical cores, {torch.get_num_threads()} threads)"
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/benchmark_overhead.csv", index=False)
    print(df.drop(columns="cpu").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
