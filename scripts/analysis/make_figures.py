"""Draw every figure of the paper from the result files (PDF for LaTeX, PNG for a quick look).

Each figure is drawn only if its inputs exist; otherwise it is skipped with a note, so the script can
be run at any stage. Colours: categorical slots in fixed order; every series also has its own marker or
line style, so the figures read in black and white.

Run : python scripts/analysis/make_figures.py
Out : results/figures/fig_*.pdf and .png
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("results/figures")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # categorical slots 1-5, fixed order
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e6e5e0", "#b9b8b2"
MARK = ["o", "s", "^", "D", "v"]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42, "lines.linewidth": 2})


def style(ax, title=None):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, loc="left", color=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    if not getattr(fig, "laid_out", False):            # a figure may lay itself out (shared legend)
        fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def ecdf(v):
    v = np.sort(np.asarray(v, float))
    return v, np.arange(1, len(v) + 1) / len(v)


def at_memory(d, col, m):
    d = d.sort_values("memory_vs_keepalive10")
    x, y = d["memory_vs_keepalive10"].to_numpy(), d[col].to_numpy(float)
    return np.interp(m, x, y) if len(x) and x.min() <= m <= x.max() else np.nan


def runs(pattern):
    return [Path(p) for p in sorted(glob.glob(pattern)) if (Path(p) / "policies.csv").exists()]


def mean_curve(curves):
    """Mean over runs, only where every run has a value (so the set of runs never changes along the curve)."""
    return np.array(curves, float).mean(axis=0)


def mean_by_param(fs, family):
    """One family's points averaged over runs, matched by parameter (e.g. the keep-alive length)."""
    d = pd.concat(pd.read_csv(f / "policies.csv") for f in fs)
    d = d[d.family == family]
    d = d.groupby("param", sort=False)[["memory_vs_keepalive10", "cold_rate"]].mean().reset_index()
    return d.sort_values("memory_vs_keepalive10")


# ----------------------------------------------------------------------------------------------- figures

def fig_coldstart_cdf():
    p = Path("data/processed/huawei_r2_cold_start_stats.parquet")
    s = pd.read_parquet(p)
    s = s[s.n_cold >= 5]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x, y = ecdf(s.d90_s)
    ax.plot(x, y, color=INK, lw=2, label=f"all functions ({len(s):,})")
    if "runtime" in s:
        top = s.runtime.value_counts().head(3).index
        for k, rt in enumerate(top):
            x, y = ecdf(s[s.runtime == rt].d90_s)
            ax.plot(x, y, color=C[k], lw=1.6, ls=["-", "--", ":"][k], label=f"{rt} ({(s.runtime == rt).sum():,})")
    for v in (1, 5):
        ax.axvline(v, color=AXIS, lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("90th-percentile cold-start time (s)")
    ax.set_ylabel("Share of functions")
    style(ax)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_coldstart_cdf")


def fig_gaps():
    ev = pd.read_parquet("data/processed/azure2021_events.parquet", columns=["fid", "start"])
    ev["s"] = np.floor(ev.start)
    ev = ev.drop_duplicates(["fid", "s"]).sort_values(["fid", "s"])
    gap = ev.groupby("fid").s.diff()
    gap = gap[(ev.s >= 10 * 86400) & gap.notna()]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x, y = ecdf(gap)
    ax.plot(x, y, color=C[0])
    for v, lab in ((60, "60 s"), (600, "10 min")):
        share = (gap <= v).mean()
        ax.axvline(v, color=AXIS, lw=0.8, ls="--", zorder=1)
        ax.annotate(f"{lab}: {share:.1%}", (v, share), xytext=(4, -12), textcoords="offset points", color=INK2,
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Time since the function's previous call (s)")
    ax.set_ylabel("Share of test call-seconds")
    style(ax)
    save(fig, "fig_gaps")


def fig_signal():
    files = {"Azure 2021": Path("results/early_signal/azure2021_controls.csv"),
             "Huawei day 30": Path("results/early_signal/huawei_day30_controls.csv")}
    files = {k: v for k, v in files.items() if v.exists()}
    if not files:
        raise FileNotFoundError("results/early_signal/*_controls.csv")
    fig, axes = plt.subplots(1, len(files), figsize=(3.4 * len(files), 2.6), sharey=True, squeeze=False)
    kinds = [("lift_uniform", "vs random times"), ("lift_same_hour", "vs same hour"),
             ("lift_app_active", "vs busy application")]
    for ax, (name, f) in zip(axes[0], files.items()):
        d = pd.read_csv(f)
        if {"lead_s", "window_s"} <= set(d.columns) and ((d.lead_s == 2) & (d.window_s == 60)).any():
            d = d[(d.lead_s == 2) & (d.window_s == 60)]
        d = d[d.bucket.isin(["rare", "medium"])].set_index("bucket").reindex(["rare", "medium"])
        x = np.arange(len(d))
        w = 0.26
        for k, (col, lab) in enumerate(kinds):
            # bars start at lift 1 (no signal), so a lift below 1 hangs down instead of looking like a short bar
            ax.bar(x + (k - 1) * w, d[col] - 1, w * 0.92, bottom=1, color=C[k], label=lab, zorder=3,
                   hatch=[None, "//", ".."][k], edgecolor="white", linewidth=0.5)
            for xi, v in zip(x, d[col]):
                ax.annotate(f"{v:.1f}", (xi + (k - 1) * w, v), xytext=(0, 2 if v >= 1 else -2),
                            textcoords="offset points", ha="center", va="bottom" if v >= 1 else "top",
                            fontsize=7, color=INK2)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.axhline(2, color=AXIS, lw=0.8, ls="--")
        ax.set_xticks(x, [f"{b}\n({int(n):,} calls)" for b, n in zip(d.index, d.n_calls)])
        ax.set_yscale("log")
        style(ax, name)
    axes[0][0].set_ylabel("Lift (2 s lead, 60 s window)")
    axes[0][-1].legend(frameon=False, loc="upper right")
    save(fig, "fig_signal")


def fig_pareto():
    fs = runs("results/policies_azure2021_s*_coldtime") or runs("results/policies_azure2021_s*_plain")
    if not fs:
        raise FileNotFoundError("results/policies_azure2021_s*_coldtime")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    grid = np.linspace(0.6, 3.3, 60)
    curves = []
    for f in fs:
        d = pd.read_csv(f / "policies.csv")
        l = d[d.family == "learned: own + app, no lead"].sort_values("memory_vs_keepalive10")
        ax.plot(l.memory_vs_keepalive10, 100 * l.cold_rate, color=C[0], lw=0.8, alpha=0.35, zorder=2)
        curves.append([at_memory(l, "cold_rate", m) for m in grid])
    mean = mean_curve(curves)
    ax.plot(grid, 100 * mean, color=C[0], lw=2, label=f"learned, mean of {len(fs)} seeds", zorder=4)
    ka = mean_by_param(fs, "fixed keep-alive")
    ax.plot(ka.memory_vs_keepalive10, 100 * ka.cold_rate, color=C[1], marker="s", ms=5, ls="--", lw=1.4,
            label="fixed keep-alive (1-60 min)", zorder=3)
    for _, r in ka.iterrows():
        if r.param in ("10 min", "60 min"):
            ax.annotate(r.param, (r.memory_vs_keepalive10, 100 * r.cold_rate), xytext=(6, 5),
                        textcoords="offset points", fontsize=7, color=INK2)
    h = mean_by_param(fs, "hybrid histogram")
    ax.plot(h.memory_vs_keepalive10, 100 * h.cold_rate, color=C[2], marker="^", ms=7, ls="none",
            label="hybrid histogram", zorder=5)
    ax.axvline(1.0, color=AXIS, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 3.5)
    ax.set_xlabel("Idle memory relative to keep-alive 10 min")
    ax.set_ylabel("Cold-start rate (%)")
    style(ax)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_pareto")


def fig_seeds():
    files = {"normal cold starts": Path("results/seedstudy_azure2021/seeds.csv"),
             "cold starts x2": Path("results/seedstudy_azure2021_x2/seeds.csv")}
    files = {k: v for k, v in files.items() if v.exists()}
    if not files:
        raise FileNotFoundError("results/seedstudy_*/seeds.csv")
    fig, axes = plt.subplots(1, len(files), figsize=(3.4 * len(files), 2.5), sharey=True, squeeze=False)
    for ax, (name, f) in zip(axes[0], files.items()):
        d = pd.read_csv(f)
        for k, (setting, lab) in enumerate((("plain", "learned"), ("coldtime", "learned + cold-start time"))):
            s = d[d.setting == setting].sort_values("seed")
            ax.plot(s.seed + (k - 0.5) * 0.15, 100 * s["cold@1.0"], marker=MARK[k], ms=7, ls="none",
                    color=C[k], label=lab, zorder=3)
            m = s["cold@1.0"].mean()
            ax.hlines(100 * m, -0.4, 4.4, color=C[k], lw=1, ls=["-", "--"][k], zorder=2)
        ax.axhline(100 * d.keepalive10_cold.mean(), color=INK2, lw=1.2, ls=":", zorder=2)
        ax.annotate("keep-alive 10 min", (4.4, 100 * d.keepalive10_cold.mean()), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=7, color=INK2)
        ax.set_xticks(range(5), [f"seed {i}" for i in range(5)])
        style(ax, name)
    axes[0][0].set_ylabel("Cold starts at 1.0x memory (%)")
    axes[0][0].legend(frameon=False, loc="upper left")
    save(fig, "fig_seeds")


def fig_groups():
    fs = runs("results/policies_azure2021_s*_coldtime")
    if not fs:
        raise FileNotFoundError("results/policies_azure2021_s*_coldtime")
    groups = [("cold_rare", "rare"), ("cold_medium", "medium"), ("cold_lead 1s", "lead\n1 s"),
              ("cold_lead 2-5s", "lead\n2-5 s"), ("cold_lead >5s", "lead\n> 5 s")]
    ka, le = [], []
    for f in fs:
        d = pd.read_csv(f / "policies.csv")
        k = d[(d.family == "fixed keep-alive") & (d.param == "10 min")].iloc[0]
        l = d[d.family == "learned: own + app, no lead"]
        ka.append([k[c] for c, _ in groups])
        le.append([at_memory(l, c, 1.0) for c, _ in groups])
    ka, le = 100 * np.nanmean(ka, 0), 100 * np.nanmean(le, 0)
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    x = np.arange(len(groups))
    for k, (vals, lab) in enumerate(((ka, "keep-alive 10 min"), (le, "learned, 1.0x memory"))):
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=C[k], label=lab, zorder=3, hatch=[None, "//"][k],
               edgecolor="white", linewidth=0.5)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.1f}", (xi + (k - 0.5) * 0.38, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [g for _, g in groups])
    ax.set_ylabel("Cold-start rate (%)")
    ax.set_ylim(0, 1.15 * np.nanmax(np.r_[ka, le]))           # a group with no functions is NaN
    style(ax, f"Mean of {len(fs)} seeds")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_groups")


def fig_risk():
    files = sorted(glob.glob("results/budget_azure2021_s*/risk.csv"))   # every seed of the budget study
    if not files:
        raise FileNotFoundError("results/budget_azure2021_s*/risk.csv")
    d = pd.concat(pd.read_csv(f) for f in files)
    d = d[d.sharing != "none (fixed 0.5)"]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    labels = {"per function": "function", "per app": "application", "per bucket": "frequency class"}
    for k, sh in enumerate(labels):
        s = d[d.sharing == sh]
        ax.scatter(s.memory, 100 * s.fns_within, s=24, marker=MARK[k], color=C[k], label=labels[sh], zorder=3,
                   edgecolors="white", linewidths=0.6, alpha=0.9)
    ax.axhline(90, color=AXIS, lw=0.8, ls="--")
    ax.axvline(1.0, color=AXIS, lw=0.8, ls=":")
    ax.set_xscale("log")
    lo, hi = d.memory.min() / 1.15, d.memory.max() * 1.15
    ax.set_xlim(lo, hi)
    ticks = [t for t in (0.5, 1, 2, 4, 8, 16) if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_ylim(top=96)
    ax.annotate("target: 90% of functions", (lo, 90), xytext=(3, 3), textcoords="offset points",
                fontsize=7, color=INK2)
    ax.set_xlabel("Idle memory relative to keep-alive 10 min")
    ax.set_ylabel("Busy functions within budget (%)")
    style(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0,
              handletextpad=0.2, title="Threshold shared per", title_fontsize=8)
    save(fig, "fig_risk")


def fig_predictors():
    sets = {"Azure 2021": runs("results/policies_azure2021_predictors_s*"),
            "Huawei day 30": runs("results/policies_huawei_day30_predictors_s*")}
    sets = {k: v for k, v in sets.items() if v}
    if not sets:
        raise FileNotFoundError("results/policies_*_predictors_s*")
    fams = [("learned: own + app, no lead", "trees (own + app features)"),
            ("learned: LSTM-full (no attention)", "LSTM, no attention"),
            ("learned: STAWarm-full (LSTM + app attention)", "STAWarm"),
            ("learned: ensemble (trees + LSTM-full)", "trees + LSTM ensemble")]
    fig, axes = plt.subplots(1, len(sets), figsize=(3.6 * len(sets), 2.8), squeeze=False)
    grid = np.linspace(0.5, 3.5, 70)
    for ax, (name, fs) in zip(axes[0], sets.items()):
        for k, (fam, lab) in enumerate(fams):
            cur = []
            for f in fs:
                l = pd.read_csv(f / "policies.csv")
                l = l[l.family == fam]
                if len(l):
                    cur.append([at_memory(l, "cold_rate", m) for m in grid])
            if cur:
                ax.plot(grid, 100 * mean_curve(cur), color=C[k % len(C)], ls=["-", "--", "-.", ":", ":", ":"][k],
                        label=f"{lab} ({len(cur)} run{'s' if len(cur) > 1 else ''})", zorder=3)
        ka = mean_by_param(fs, "fixed keep-alive")
        ax.plot(ka.memory_vs_keepalive10, 100 * ka.cold_rate, color=INK2, marker="s", ms=4, ls=":", lw=1.2,
                label="fixed keep-alive", zorder=2)
        ax.axvline(1.0, color=AXIS, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 3.5)
        ax.set_xlabel("Idle memory relative to keep-alive 10 min")
        style(ax, name)
    axes[0][0].set_ylabel("Cold-start rate (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()       # one shared legend above the panels
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.laid_out = True
    save(fig, "fig_predictors")


def fig_headroom():
    fs = runs("results/policies_azure2021_baselines_s*")
    if not fs:
        raise FileNotFoundError("results/policies_azure2021_baselines_s*")
    fams = [("learned: own + app, no lead", "learned (trees)"), ("learned: state-aware (trees)", "state-aware"),
            ("learned: call rate only (no learning)", "call rate only")]
    grid = np.linspace(0.5, 3.5, 70)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for k, (fam, lab) in enumerate(fams):
        cur = [[at_memory(l[l.family == fam], "cold_rate", m) for m in grid]
               for l in (pd.read_csv(f / "policies.csv") for f in fs) if (l.family == fam).any()]
        if cur:
            ax.plot(grid, 100 * mean_curve(cur), color=C[k], ls=["-", "--", "-."][k], label=lab)
    o = mean_by_param(fs, "oracle (perfect predictor)")
    if len(o):
        floor = 0.01                                      # % ; a log axis cannot show zero
        y = 100 * o.cold_rate.to_numpy()
        ax.plot(o.memory_vs_keepalive10, np.maximum(y, floor), color=C[3], marker="D", ms=5, lw=1.2,
                label="perfect predictor" + (f" ({y.max():.0f}%, drawn at axis floor)" if y.max() < floor else ""))
    ka = mean_by_param(fs, "fixed keep-alive")
    ax.plot(ka.memory_vs_keepalive10, 100 * ka.cold_rate, color=INK2, marker="s", ms=4, ls=":", lw=1.2,
            label="fixed keep-alive")
    ax.axvline(1.0, color=AXIS, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 3.5)
    ax.set_xlabel("Idle memory relative to keep-alive 10 min")
    ax.set_ylabel("Cold-start rate (%)")
    style(ax, f"Mean of {len(fs)} seeds")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.09), fontsize=7)   # clear of the floor markers
    save(fig, "fig_headroom")


def fig_bench():
    d = pd.read_csv("results/benchmark_overhead.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, (att, lab) in enumerate(((False, "LSTM, no attention"), (True, "STAWarm (with attention)"))):
        s = d[d.attention == att]
        ax.plot(s.functions, s.inference_s_per_hour, color=C[k], marker=MARK[k], ms=6, label=lab)
        last = s.iloc[-1]
        ax.annotate(f"{last.inference_s_per_hour:.1f} s", (last.functions, last.inference_s_per_hour),
                    xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of functions")
    ax.set_ylabel("CPU time per hour of decisions (s)")
    style(ax)
    ax.legend(frameon=False, loc="upper left")                # the empty corner: both lines rise to the right
    save(fig, "fig_bench")


def main():
    for f in (fig_coldstart_cdf, fig_gaps, fig_signal, fig_pareto, fig_seeds, fig_groups, fig_risk,
              fig_predictors, fig_headroom, fig_bench):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: missing {e.filename or e}")
        except Exception as e:  # keep going; report the problem
            print(f"FAILED {f.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
