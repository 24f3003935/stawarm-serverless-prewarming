"""Compare a reproduction against the result files of the paper.

For every run folder present in both directories, each policy's cold-start rate is read at exactly 0.8x and
1.0x the memory of a 10-minute keep-alive (as in the paper) and the difference is reported in percentage
points. The early-signal tables, the budget-controller tables and the five-seed study are compared value by
value. Tree-based runs are deterministic for a given seed and library version, so their differences should be
zero in the same environment; runs of the neural predictors differ slightly between executions on a GPU.

Run : python scripts/analysis/compare_results.py <new results dir> <paper results dir> [--out report.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MEMS = (0.8, 1.0)


def at_memory(d: pd.DataFrame, col: str, m: float) -> float:
    d = d.sort_values("memory_vs_keepalive10")
    x, y = d["memory_vs_keepalive10"].to_numpy(), d[col].to_numpy(float)
    return np.nan if (len(x) == 0 or m < x.min() or m > x.max()) else float(np.interp(m, x, y))


def compare_policies(new: Path, ref: Path) -> pd.DataFrame:
    a, b = pd.read_csv(new / "policies.csv"), pd.read_csv(ref / "policies.csv")
    rows = []
    for fam in sorted(set(a.family) & set(b.family)):
        da, db = a[a.family == fam], b[b.family == fam]
        if fam.startswith("learned"):
            for m in MEMS:
                va, vb = at_memory(da, "cold_rate", m), at_memory(db, "cold_rate", m)
                rows.append({"family": fam, "point": f"{m}x memory", "new": va, "paper": vb})
        else:
            j = da.merge(db, on="param", suffixes=("_new", "_paper"))
            for _, r in j.iterrows():
                rows.append({"family": fam, "point": str(r.param), "new": r.cold_rate_new, "paper": r.cold_rate_paper})
    out = pd.DataFrame(rows)
    out["diff_pts"] = 100 * (out["new"] - out["paper"])
    return out


def compare_table(new: Path, ref: Path, keys: list[str]) -> pd.DataFrame:
    a, b = pd.read_csv(new), pd.read_csv(ref)
    keys = [k for k in keys if k in a and k in b]
    num = [c for c in a.columns if c in b.columns and c not in keys and pd.api.types.is_numeric_dtype(a[c])
           and pd.api.types.is_numeric_dtype(b[c])]
    j = a.merge(b, on=keys, suffixes=("_new", "_paper"))
    rows = [{"column": c, "max_abs_diff": float(np.nanmax(np.abs(j[c + "_new"] - j[c + "_paper"]))) if len(j) else np.nan}
            for c in num]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("paper")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    new, ref = Path(args.new), Path(args.paper)
    lines = [f"Comparison of {new} with {ref}", ""]

    runs = sorted(p.name for p in new.iterdir() if (p / "policies.csv").exists() and (ref / p.name / "policies.csv").exists())
    worst = 0.0
    for name in runs:
        d = compare_policies(new / name, ref / name)
        m = float(np.nanmax(np.abs(d.diff_pts))) if len(d) else np.nan
        worst = max(worst, m) if np.isfinite(m) else worst
        lines.append(f"{name}: largest difference {m:.4f} percentage points")
        learned = d[d.family.str.startswith("learned")]
        if len(learned):
            show = learned.copy()
            for c in ("new", "paper"):
                show[c] = (100 * show[c]).round(3)
            show["diff_pts"] = show["diff_pts"].round(4)
            lines += ["    " + l for l in show.to_string(index=False).splitlines()]
    if runs:
        lines += ["", f"Policy runs compared: {len(runs)}; largest difference overall: {worst:.4f} percentage points"]
    else:
        lines.append("No policy runs in common.")

    for sub, keys in (("early_signal", ["bucket", "lead_s", "window_s"]),):
        for f in sorted((new / sub).glob("*.csv")) if (new / sub).exists() else []:
            if (ref / sub / f.name).exists():
                t = compare_table(f, ref / sub / f.name, keys)
                lines += ["", f"{sub}/{f.name}: largest absolute difference per column",
                          "    " + t.round(6).to_string(index=False).replace("\n", "\n    ")]
    for p in sorted(new.glob("budget_*")):
        if (p / "risk.csv").exists() and (ref / p.name / "risk.csv").exists():
            t = compare_table(p / "risk.csv", ref / p.name / "risk.csv", ["sharing", "eta", "start", "alpha"])
            lines += ["", f"{p.name}/risk.csv: largest absolute difference per column",
                      "    " + t.round(6).to_string(index=False).replace("\n", "\n    ")]
    for p in sorted(new.glob("seedstudy_*")):
        if (p / "seeds.csv").exists() and (ref / p.name / "seeds.csv").exists():
            t = compare_table(p / "seeds.csv", ref / p.name / "seeds.csv", ["seed", "setting"])
            lines += ["", f"{p.name}/seeds.csv: largest absolute difference per column",
                      "    " + t.round(6).to_string(index=False).replace("\n", "\n    ")]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
