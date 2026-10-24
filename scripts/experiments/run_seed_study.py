"""Five-seed study of the tree policy, with and without the cold-start-time input (paper, RQ6).

Runs run_policies.py for each seed and setting, reads each policy's curve at exactly 0.8x and 1.0x the
memory of a 10-minute keep-alive (linear interpolation between thresholds, no extrapolation), and reports
mean, spread and the paired difference per seed. A seed changes both the model's randomness and the
cold-start profile assigned to each function. Finished runs are skipped, so the script can be restarted.

Run : python scripts/experiments/run_seed_study.py              (normal cold-start times)
      python scripts/experiments/run_seed_study.py --scale 2    (doubled cold-start times)
Out : results/seedstudy_<dataset>[_x<scale>]/seeds.csv, summary.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THETAS = "0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
FAMILY = "learned: own + app, no lead"
SETTINGS = {"plain": [], "coldtime": ["--lead-feature"]}
MEMS = [0.8, 1.0]


def run_dir(dataset: str, scale: float, tag: str) -> Path:
    return Path(f"results/policies_{dataset}" + ("" if scale == 1.0 else f"_x{scale:g}") + f"_{tag}")


def at_memory(d: pd.DataFrame, col: str, m: float) -> float:
    """Value of ``col`` on the policy's curve at memory m (linear between neighbouring thresholds)."""
    d = d.sort_values("memory_vs_keepalive10")
    x, y = d["memory_vs_keepalive10"].to_numpy(), d[col].to_numpy(float)
    if m < x.min() or m > x.max():
        return np.nan                     # outside the curve: do not extrapolate
    return float(np.interp(m, x, y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="azure2021")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(f"results/seedstudy_{args.dataset}" + ("" if args.scale == 1.0 else f"_x{args.scale:g}"))
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))
    import run_policies

    rows = []
    for seed in seeds:
        for name, extra in SETTINGS.items():
            tag = f"s{seed}_{name}"
            folder = run_dir(args.dataset, args.scale, tag)
            if not (folder / "policies.csv").exists():
                print(time.strftime("%H:%M:%S"), f"seed {seed}, {name}: running", flush=True)
                sys.argv = ["run_policies.py", "--dataset", args.dataset, "--scale", str(args.scale),
                            "--seed", str(seed), "--variants", "app_nolead", "--risk-on", "none",
                            "--thetas", THETAS, "--tag", tag] + extra
                run_policies.main()
            else:
                print(time.strftime("%H:%M:%S"), f"seed {seed}, {name}: already done", flush=True)
            df = pd.read_csv(folder / "policies.csv")
            d = df[df.family == FAMILY]
            ka = df[(df.family == "fixed keep-alive") & (df.param == "10 min")].iloc[0]
            row = {"seed": seed, "setting": name, "keepalive10_cold": ka["cold_rate"],
                   "keepalive10_cold_slow": ka.get("cold_lead >5s", np.nan)}
            for m in MEMS:
                row[f"cold@{m}"] = at_memory(d, "cold_rate", m)
                row[f"cold_slow@{m}"] = at_memory(d, "cold_lead >5s", m)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "seeds.csv", index=False)
    pct = lambda v: f"{100 * v:.2f}%"
    text = [f"Five-seed study ({args.dataset}, scale {args.scale:g}, seeds {seeds}): learned no-lead policy, "
            f"cold-start rate read at matched memory (x = memory relative to a 10-minute keep-alive).", ""]
    metrics = [f"cold@{m}" for m in MEMS] + [f"cold_slow@{m}" for m in MEMS]
    for name in SETTINGS:
        s = df[df.setting == name]
        text.append(f"{name:9s}: " + "; ".join(
            f"{k} {pct(s[k].mean())} ± {pct(s[k].std())}" for k in metrics))
    k = df[df.setting == "plain"]
    text += [f"keep-alive 10 min (1.0x): cold {pct(k.keepalive10_cold.mean())} ± {pct(k.keepalive10_cold.std())}; "
             f"slow functions {pct(k.keepalive10_cold_slow.mean())}", "",
             "Paired difference per seed (coldtime - plain; negative = the cold-start-time input helps):"]
    p = df.pivot(index="seed", columns="setting")
    for key in metrics:
        diff = (p[(key, "coldtime")] - p[(key, "plain")]).dropna()
        if len(diff):
            text.append(f"  {key:15s} mean {100 * diff.mean():+.3f} pts, sd {100 * diff.std():.3f}, "
                        f"helps in {(diff < 0).sum()} of {len(diff)} seeds")
    gain = (k["keepalive10_cold"] - k["cold@1.0"]) / k["keepalive10_cold"]
    text += ["", f"Learned (plain) vs keep-alive at equal memory: {100 * gain.mean():.1f}% fewer cold starts "
             f"(range {100 * gain.min():.1f}% to {100 * gain.max():.1f}% over seeds)"]
    (out / "summary.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
