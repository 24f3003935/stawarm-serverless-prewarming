"""Summarise several runs of run_policies.py (seeds) at matched memory.

For every run folder matching the pattern, each learned policy's curve is read at exactly 0.8x and 1.0x
the memory of a 10-minute keep-alive. Results are averaged over runs, with the spread, the number of runs
better than keep-alive, paired t-tests against keep-alive (Holm-corrected) and between policies, and the
mean AUPRC from prediction.csv.

Example:
  python scripts/analysis/summarize_runs.py "results/policies_azure2021_predictors_s*" --out results/summary_predictors_azure.txt
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MEMS = (0.8, 1.0)


def at_memory(d: pd.DataFrame, col: str, m: float) -> float:
    d = d.sort_values("memory_vs_keepalive10")
    x, y = d["memory_vs_keepalive10"].to_numpy(), d[col].to_numpy(float)
    if len(x) == 0 or m < x.min() or m > x.max():
        return np.nan
    return float(np.interp(m, x, y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help='glob for run folders, e.g. "results/policies_azure2021_lstm_s*"')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    folders = sorted(Path(p) for p in glob.glob(args.pattern) if (Path(p) / "policies.csv").exists())
    if not folders:
        raise SystemExit(f"no finished runs match {args.pattern}")

    rows, pred = [], []
    for f in folders:
        df = pd.read_csv(f / "policies.csv")
        ka = df[(df.family == "fixed keep-alive") & (df.param == "10 min")].iloc[0]
        hist = df[df.family == "hybrid histogram"]
        for fam in [x for x in df.family.unique() if x.startswith("learned")]:
            d = df[df.family == fam]
            r = {"run": f.name, "family": fam.replace("learned: ", ""), "keepalive_cold": ka.cold_rate}
            for m in MEMS:
                r[f"cold@{m}"] = at_memory(d, "cold_rate", m)
                if "cold_lead >5s" in d:
                    r[f"slow@{m}"] = at_memory(d, "cold_lead >5s", m)
            r["histogram_cold"] = float(hist.cold_rate.iloc[0]) if len(hist) else np.nan
            r["histogram_mem"] = float(hist.memory_vs_keepalive10.iloc[0]) if len(hist) else np.nan
            rows.append(r)
        if (f / "prediction.csv").exists():
            p = pd.read_csv(f / "prediction.csv")
            p["run"] = f.name
            pred.append(p)
    df = pd.DataFrame(rows)

    pct = lambda v: "n/a" if pd.isna(v) else f"{100 * v:.2f}%"
    lines = [f"Runs: {len(folders)} ({', '.join(x.name for x in folders)})",
             f"10-minute keep-alive: {pct(df.keepalive_cold.mean())} cold at memory 1.00", ""]
    if df.histogram_cold.notna().any():
        lines.append(f"Hybrid histogram: {pct(df.histogram_cold.mean())} cold at memory "
                     f"{df.histogram_mem.mean():.2f} (mean over runs)\n")
    lines.append("Cold-start rate at matched memory (mean +/- sd over runs; 'better' = runs below keep-alive;"
                 " p = paired t-test vs keep-alive at 1.0x, p_holm = Holm-corrected over all policies listed):")
    pvals = {}
    for fam, g in df.groupby("family", sort=False):
        v = g["cold@1.0"].dropna()
        k = g.loc[v.index, "keepalive_cold"]
        pvals[fam] = stats.ttest_rel(v, k).pvalue if len(v) > 1 and (v - k).std() > 0 else np.nan
    order = sorted([f for f in pvals if not np.isnan(pvals[f])], key=lambda f: pvals[f])
    holm, running = {}, 0.0
    for r, f in enumerate(order):
        running = max(running, min(1.0, (len(order) - r) * pvals[f]))
        holm[f] = running
    for fam, g in df.groupby("family", sort=False):
        parts = []
        for m in MEMS:
            v = g[f"cold@{m}"].dropna()
            parts.append(f"@{m}x {pct(v.mean())} +/- {pct(v.std()) if len(v) > 1 else 'n/a'}")
        v = g["cold@1.0"].dropna()
        k = g.loc[v.index, "keepalive_cold"]
        better = int((v < k).sum())
        p = pvals[fam]
        gain = 100 * (1 - v.mean() / k.mean()) if len(v) else np.nan
        slow = g["slow@1.0"].dropna() if "slow@1.0" in g else pd.Series(dtype=float)
        lines.append(f"  {fam:35s} " + "; ".join(parts) +
                     f"; {gain:.1f}% fewer than keep-alive; better in {better}/{len(v)}; p={p:.3f}, p_holm={holm.get(fam, np.nan):.3f}" +
                     (f"; slow fns @1.0x {pct(slow.mean())}" if len(slow) else ""))
    fams = list(df.family.unique())
    if len(fams) > 1:
        lines += ["", f"Paired differences at 1.0x vs '{fams[0]}' (negative = better):"]
        base = df[df.family == fams[0]].set_index("run")["cold@1.0"]
        for fam in fams[1:]:
            other = df[df.family == fam].set_index("run")["cold@1.0"]
            d = (other - base).dropna()
            if len(d):
                p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else np.nan
                lines.append(f"  {fam:35s} mean {100 * d.mean():+.3f} pts, better in {(d < 0).sum()}/{len(d)}, p={p:.3f}")
    if pred:
        p = pd.concat(pred)
        keys = ["model", "target", "bucket"] if "target" in p else ["model", "bucket"]
        t = p.groupby(keys)["auprc"].agg(["mean", "std", "count"]).round(4)
        lines += ["", "Prediction quality (AUPRC vs the true target, mean/sd over runs):", t.to_string()]
    text = "\n".join(lines)
    print(text)
    out = Path(args.out) if args.out else folders[0].parent / "summary_runs.txt"
    out.write_text(text, encoding="utf-8")
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
