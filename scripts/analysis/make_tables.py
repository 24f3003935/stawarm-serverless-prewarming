"""Write every results table of the paper as LaTeX from the run folders.

Run : python scripts/analysis/make_tables.py results
Out : results/tables/*.tex
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = R / "tables"
OUT.mkdir(parents=True, exist_ok=True)
DS = {"azure2021": "Azure 2021", "huawei_day30": "Huawei day 30"}
NAME = {"own + app, no lead": r"Trees (own + application features)",
        "LSTM-full (no attention)": r"LSTM, no attention",
        "STAWarm-full (LSTM + app attention)": r"\sysname{} (LSTM + attention)",
        "ensemble (trees + LSTM-full)": r"Ensemble (trees + LSTM)",
        "call rate only (no learning)": r"Invocation rate only (no learning)",
        "state-aware (trees)": r"State-aware trees"}
PRED = ["own + app, no lead", "LSTM-full (no attention)", "STAWarm-full (LSTM + app attention)",
        "ensemble (trees + LSTM-full)"]
THETAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def at(d, col, m):
    d = d.sort_values("memory_vs_keepalive10")
    x, y = d.memory_vs_keepalive10.to_numpy(), d[col].to_numpy(float)
    return np.nan if (len(x) == 0 or m < x.min() or m > x.max()) else float(np.interp(m, x, y))


def runs(pattern):
    return sorted(p for p in glob.glob(str(R / pattern)) if (Path(p) / "policies.csv").exists())


def seed_of(path):
    return int(Path(path).name.split("_s")[-1].split("_")[0])


def pct(v, nd=2):
    return "--" if v is None or not np.isfinite(v) else f"{100 * v:.{nd}f}"


def pm(v, nd=2):
    v = pd.Series(v).dropna()
    if not len(v):
        return "--"
    return f"{100 * v.mean():.{nd}f}" + (f" $\\pm$ {100 * v.std():.{nd}f}" if len(v) > 1 else "")


def sg(v, nd=1):
    """Signed number with a typographic minus."""
    return f"{v:+.{nd}f}".replace("-", "$-$")


def pval(p):
    return "--" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def holm(ps):
    keys = [k for k in ps if np.isfinite(ps[k])]
    order = sorted(keys, key=lambda k: ps[k])
    out, run = {k: np.nan for k in ps}, 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * ps[k]))
        out[k] = run
    return out


def write(name, text):
    (OUT / f"{name}.tex").write_text(text.strip() + "\n", encoding="utf-8")
    print("wrote", OUT / f"{name}.tex")


def collect(pattern):
    """One row per (run, family) with values at 0.8x / 1.0x and at the histogram's memory."""
    rows = []
    for f in runs(pattern):
        df = pd.read_csv(Path(f) / "policies.csv")
        ka_all = df[df.family == "fixed keep-alive"]
        ka = ka_all[ka_all.param == "10 min"].iloc[0]
        h = df[df.family == "hybrid histogram"].iloc[0]
        base = {"run": Path(f).name, "seed": seed_of(f), "ka": ka.cold_rate, "ka08": at(ka_all, "cold_rate", 0.8),
                "ka_rare": ka.cold_rare, "ka_med": ka.cold_medium, "ka_freq": ka.cold_frequent,
                "ka_slow": ka["cold_lead >5s"], "ka_lat": ka.added_latency_ms_per_call,
                "hist_mem": h.memory_vs_keepalive10, "hist": h.cold_rate, "hist_rare": h.cold_rare,
                "hist_med": h.cold_medium, "hist_freq": h.cold_frequent, "hist_slow": h["cold_lead >5s"],
                "ka60": ka_all[ka_all.param == "60 min"].cold_rate.iloc[0],
                "ka60_mem": ka_all[ka_all.param == "60 min"].memory_vs_keepalive10.iloc[0]}
        for fam in df.family.unique():
            d = df[df.family == fam]
            if fam.startswith("oracle"):
                for _, r in d.iterrows():
                    rows.append({**base, "family": "oracle", "param": r.param, "mem": r.memory_vs_keepalive10,
                                 "cold": r.cold_rate})
                continue
            if not fam.startswith("learned"):
                continue
            fam = fam.replace("learned: ", "")
            r = {**base, "family": fam, "minmem": d.memory_vs_keepalive10.min(),
                 "maxmem": d.memory_vs_keepalive10.max(),
                 "lowest": d.cold_rate.min(),
                 "lowest_mem": d.sort_values("cold_rate").memory_vs_keepalive10.iloc[0],
                 "at_hist": at(d, "cold_rate", h.memory_vs_keepalive10)}
            for m in (0.8, 1.0):
                r[f"c{m}"] = at(d, "cold_rate", m)
            for col, key in (("cold_rare", "rare"), ("cold_medium", "med"), ("cold_frequent", "freq"),
                             ("cold_lead >5s", "slow"), ("added_latency_ms_per_call", "lat")):
                r[key] = at(d, col, 1.0)
            r["sweep"] = d.set_index("param")[["cold_rate", "memory_vs_keepalive10"]]
            rows.append(r)
    return pd.DataFrame(rows)


def ttest_vs_ka(g):
    v, k = g["c1.0"].to_numpy(float), g["ka"].to_numpy(float)
    ok = np.isfinite(v)
    if ok.sum() < 2 or np.std(v[ok] - k[ok]) == 0:
        return np.nan
    return stats.ttest_rel(v[ok], k[ok]).pvalue


def gain(g):
    v = g["c1.0"].dropna()
    return 1 - v.mean() / g.loc[v.index, "ka"].mean()


# ------------------------------------------------------------------------------------------------ data
gpu = {ds: collect(f"policies_{ds}_predictors_s*") for ds in DS}
pkg = {ds: collect(f"policies_{ds}_baselines_s*") for ds in DS}
real = {ds: collect(f"policies_{ds}_d2_load_rand_conservative_s*") for ds in DS}
s11 = {sc: pd.read_csv(R / f"seedstudy_azure2021{sfx}" / "seeds.csv") for sc, sfx in (("1", ""), ("2", "_x2"))
       if (R / f"seedstudy_azure2021{sfx}" / "seeds.csv").exists()}

# ------------------------------------------------------------------------- Table: main (baselines + trees)
lines = []
for ds, dname in DS.items():
    P = pkg[ds]
    first = P.iloc[0]
    ka = P.drop_duplicates("run")
    lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{dname}}} ({ka.run.nunique()} seeds)}} \\")
    lines.append(rf"\quad Keep-alive 10 min (reference) & {pm(ka.ka08)} & {pm(ka.ka)} & -- & -- \\")
    ps = {fam: ttest_vs_ka(g) for fam, g in P.groupby("family")}
    ph = holm(ps)
    for fam in ["call rate only (no learning)", "own + app, no lead", "state-aware (trees)"]:
        g = P[P.family == fam]
        better = int((g["c1.0"] < g["ka"]).sum())
        lines.append(rf"\quad {NAME[fam]} & {pm(g['c0.8'])} & {pm(g['c1.0'])} & "
                     rf"{sg(-100 * gain(g))}\% ({better}/{len(g)}) & {pval(ph[fam])} \\")
    if ds == "azure2021" and "1" in s11:
        s = s11["1"][s11["1"].setting == "coldtime"]
        g = pd.DataFrame({"c1.0": s["cold@1.0"].to_numpy(), "ka": s.keepalive10_cold.to_numpy()})
        p = ttest_vs_ka(g)
        lines.append(rf"\quad Trees, five seeds & {pm(s['cold@0.8'])} & {pm(s['cold@1.0'])} & "
                     rf"{sg(-100 * gain(g))}\% ({int((g['c1.0'] < g['ka']).sum())}/{len(g)}) & {pval(p)} \\")
    if ds == "azure2021":
        lines.append(r"\addlinespace")
write("tab_main", r"""
\begin{table}[t]
  \centering
  \caption{Cold-start rate (\%) at matched idle memory: each curve read at exactly 0.8$\times$ and 1.0$\times$ the memory of a 10-minute keep-alive. Mean $\pm$ standard deviation over seeds; ``vs.\ keep-alive'' is the relative change at 1.0$\times$, with the number of seeds in which the policy is better; $p$ is a paired $t$-test against keep-alive at 1.0$\times$, Holm-corrected over the three policies of each trace (the five-seed row is a separate run family and is not corrected).}
  \label{tab:main}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lcccc@{}}
    \toprule
    Policy & 0.8$\times$ memory & 1.0$\times$ memory & vs.\ keep-alive & $p$ (Holm) \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table}
""")

# ------------------------------------------------------------------------- Table: predictors
lines = []
for ds, dname in DS.items():
    P = gpu[ds]
    ka = P.drop_duplicates("run")
    lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{dname}}} ({ka.run.nunique()} seeds; keep-alive 10 min: "
                 rf"{pct(ka.ka.mean())}\% at 1.0$\times$, {pct(ka.ka08.mean())}\% at 0.8$\times$)}} \\")
    ph = holm({fam: ttest_vs_ka(g) for fam, g in P.groupby("family")})
    base = P[P.family == PRED[0]].set_index("seed")["c1.0"]
    for fam in PRED:
        g = P[P.family == fam]
        better = int((g["c1.0"] < g["ka"]).sum())
        if fam == PRED[0]:
            diff = "--"
        else:
            d = (g.set_index("seed")["c1.0"] - base).dropna()
            p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else np.nan
            diff = f"{sg(100 * d.mean(), 3)} ({int((d < 0).sum())}/{len(d)}; $p$={pval(p)})"
        lines.append(rf"\quad {NAME[fam]} & {pm(g['c0.8'])} & {pm(g['c1.0'])} & "
                     rf"{sg(-100 * gain(g))}\% ({better}/{len(g)}) & {pval(ph[fam])} & {diff} \\")
    if ds == "azure2021":
        lines.append(r"\addlinespace")
write("tab_pred", r"""
\begin{table*}[t]
  \centering
  \caption{Comparison of predictors under a common decision layer, at matched idle memory (cold-start rate in \%, mean $\pm$ standard deviation over seeds). All four use the same 13 inputs and the near-term target. ``vs.\ keep-alive'': relative change at 1.0$\times$ and seeds better than keep-alive; $p$: paired $t$-test against keep-alive, Holm-corrected over the four predictors of each trace; last column: paired difference from the trees at 1.0$\times$ in percentage points (negative is better), seeds better than the trees and a paired $t$-test.}
  \label{tab:pred}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lccccc@{}}
    \toprule
    Predictor & 0.8$\times$ memory & 1.0$\times$ memory & vs.\ keep-alive & $p$ (Holm) & Difference from trees \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# ------------------------------------------------------------------------- Table: AUPRC
rows = []
for ds in DS:
    pr = []
    for f in runs(f"policies_{ds}_predictors_s*"):
        p = pd.read_csv(Path(f) / "prediction.csv")
        pr.append(p)
    p = pd.concat(pr)
    p = p[p.target == "near-term"]
    t = p.groupby(["model", "bucket"]).auprc.agg(["mean", "std"])
    rows.append((ds, t))
buckets = {"azure2021": ["rare", "medium"], "huawei_day30": ["rare", "medium", "frequent"]}
head = " & ".join([f"{DS[ds]}: {b}" for ds in DS for b in buckets[ds]])
lines = []
for fam in PRED:
    cells = []
    for ds, t in rows:
        for b in buckets[ds]:
            if (fam, b) in t.index:
                m, s = t.loc[(fam, b)]
                cells.append(f"{m:.3f} $\\pm$ {s:.3f}")
            else:
                cells.append("--")
    lines.append(f"{NAME[fam]} & " + " & ".join(cells) + r" \\")
write("tab_auprc", r"""
\begin{table*}[t]
  \centering
  \caption{Prediction quality: area under the precision-recall curve against the near-term target on the test period, by frequency class (mean $\pm$ standard deviation over three seeds; 3 million sampled function-seconds per class and run). Azure 2021 has no function in the frequent class.}
  \label{tab:auprc}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}l""" + "c" * len(head.split(" & ")) + r"""@{}}
    \toprule
    Predictor & """ + head + r""" \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# ------------------------------------------------------------------------- Table: histogram at its own memory
lines = []
fams = ["own + app, no lead", "STAWarm-full (LSTM + app attention)", "ensemble (trees + LSTM-full)",
        "LSTM-full (no attention)"]
cells = {}
for ds in DS:
    P = gpu[ds]
    ka = P.drop_duplicates("run")
    cells[ds] = {"mem": f"{ka.hist_mem.mean():.2f}", "hist": pm(ka["hist"]),
                 "ka60": f"{pm(ka.ka60)} @ {ka.ka60_mem.mean():.2f}"}
    for fam in fams:
        cells[ds][fam] = pm(P[P.family == fam].at_hist)
lines.append(r"Hybrid histogram (own memory) & " + " & ".join(cells[ds]["hist"] for ds in DS) + r" \\")
lines.append(r"Keep-alive 60 min (own memory) & " + " & ".join(cells[ds]["ka60"] for ds in DS) + r" \\")
lines.append(r"\midrule")
for fam in fams:
    lines.append(f"{NAME[fam]} & " + " & ".join(cells[ds][fam] for ds in DS) + r" \\")
write("tab_hist", r"""
\begin{table}[t]
  \centering
  \caption{Comparison with the hybrid histogram policy at the histogram's own memory (""" +
      f"{cells['azure2021']['mem']}" + r"""$\times$ on Azure 2021 and """ + f"{cells['huawei_day30']['mem']}" +
      r"""$\times$ on Huawei day 30, mean of three seeds). Cold-start rate in \%, mean $\pm$ standard deviation; the learned curves are read at the histogram's memory in each seed. ``--'': the curve does not reach that memory.}
  \label{tab:hist}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lcc@{}}
    \toprule
    Policy & Azure 2021 & Huawei day 30 \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table}
""")

# ------------------------------------------------------------------------- Table: classes at 1.0x
lines = []
for ds, dname in DS.items():
    P = gpu[ds]
    ka = P.drop_duplicates("run")
    cols = ["rare", "med", "freq", "slow"]
    lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{dname}}}}} \\")
    lines.append(r"\quad Keep-alive 10 min & " + " & ".join(pm(ka["ka_" + c]) for c in cols) +
                 f" & {ka.ka_lat.mean():.1f}" + r" \\")
    lines.append(r"\quad Hybrid histogram$^\dagger$ & " + " & ".join(pm(ka["hist_" + c]) for c in cols) + r" & -- \\")
    for fam in PRED:
        g = P[P.family == fam]
        lines.append(rf"\quad {NAME[fam]} & " + " & ".join(pm(g[c]) for c in cols) + f" & {g.lat.mean():.1f}" + r" \\")
    if ds == "azure2021":
        lines.append(r"\addlinespace")
write("tab_classes", r"""
\begin{table*}[t]
  \centering
  \caption{Cold-start rate (\%) by frequency class and for slow functions (90th-percentile cold-start time over 5\,s), at 1.0$\times$ the memory of the 10-minute keep-alive (mean $\pm$ standard deviation over three seeds), and added latency per invocation in ms (median cold-start time per cold call). $^\dagger$The histogram is shown at its own, larger memory (\cref{tab:hist}). Rare: fewer than 10 active minutes a day in training; medium: 10 to 1,000; frequent: more. Slow functions are few, so their rate varies strongly with the profile assignment of each seed.}
  \label{tab:classes}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lccccc@{}}
    \toprule
    Policy & Rare & Medium & Frequent & Slow ($>$5\,s) & ms/call \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# ------------------------------------------------------------------------- Table: headroom (oracle)
lines = []
orc = {ds: pkg[ds][pkg[ds].family == "oracle"] for ds in DS}
for hold in ["hold 5 s", "hold 15 s", "hold 30 s", "hold 60 s"]:
    c = []
    for ds in DS:
        o = orc[ds][orc[ds].param == hold]
        c.append(f"{o.mem.mean():.2f} $\\pm$ {o.mem.std():.2f}")
    lines.append(f"Perfect predictor, {hold.replace('hold ', 'hold ')} & 0 & " + " & ".join(c) + r" \\")
lines.append(r"\midrule")
for fam in ["own + app, no lead"]:
    c = []
    for ds in DS:
        g = pkg[ds][pkg[ds].family == fam]
        c.append(f"{pm(g.lowest)}\\% @ {g.lowest_mem.mean():.2f}")
    lines.append(r"Trees, lowest cold-start rate attained (\%) at the stated memory & -- & " + " & ".join(c) + r" \\")
write("tab_headroom", r"""
\begin{table}[t]
  \centering
  \caption{Remaining potential for improvement: idle memory (relative to the 10-minute keep-alive, mean $\pm$ standard deviation over three seeds) at which a perfect predictor, which pre-warms exactly $\kappa_i$ seconds before every invocation and retains the instance for the stated time, incurs no cold starts. For comparison, the last row gives the lowest cold-start rate attained by the tree policy over the threshold sweep and the memory at which it is attained.}
  \label{tab:headroom}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lccc@{}}
    \toprule
    Policy & Cold starts & Azure 2021 & Huawei day 30 \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table}
""")

# ------------------------------------------------------------------------- Table: realism
lines = []
for ds, dname in DS.items():
    a, b = pkg[ds][pkg[ds].family == "own + app, no lead"], real[ds][real[ds].family == "own + app, no lead"]
    d = (b.set_index("seed")["c1.0"] - a.set_index("seed")["c1.0"]).dropna()
    lines.append(rf"{dname} & {pm(a['c0.8'])} & {pm(a['c1.0'])} & {pm(b['c0.8'])} & {pm(b['c1.0'])} & "
                 rf"{sg(100 * d.mean(), 3)} & {sg(-100 * gain(b))}\% \\")
write("tab_realism", r"""
\begin{table*}[t]
  \centering
  \caption{Sensitivity to the simplifications of the simulator (tree policy, three seeds). Conservative setting: pre-warm decisions take effect with a delay of 2\,s, loading time of a pre-warm is charged as idle memory, and every cold start draws its duration from a log-normal fitted to the function's median and 90th percentile. Cold-start rate in \%; $\Delta$ is the paired difference at 1.0$\times$ in percentage points; the last column is the change against keep-alive under the realistic setting.}
  \label{tab:realism}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    & \multicolumn{2}{c}{Default simulator} & \multicolumn{2}{c}{Conservative} & & \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-5}
    Trace & 0.8$\times$ & 1.0$\times$ & 0.8$\times$ & 1.0$\times$ & $\Delta$ at 1.0$\times$ & vs.\ keep-alive \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# ------------------------------------------------------------------------- Table: five seeds, cold-start-time input
lines = []
for sc, lab in (("1", "normal"), ("2", r"doubled ($\times$2)")):
    if sc not in s11:
        continue
    s = s11[sc]
    for setting, slab in (("plain", "without"), ("coldtime", "with")):
        x = s[s.setting == setting]
        g = pd.DataFrame({"c1.0": x["cold@1.0"].to_numpy(), "ka": x.keepalive10_cold.to_numpy()})
        lines.append(rf"{lab if setting == 'plain' else ''} & {slab} & {pm(x['cold@0.8'])} & {pm(x['cold@1.0'])} & "
                     rf"{pm(x['cold_slow@1.0'], 1)} & {sg(-100 * gain(g))}\% ({int((g['c1.0'] < g['ka']).sum())}/5) \\")
    p = s.pivot(index="seed", columns="setting")
    d = (p[("cold@1.0", "coldtime")] - p[("cold@1.0", "plain")])
    pv = stats.ttest_1samp(d, 0).pvalue
    lines.append(rf" & difference & & {sg(100 * d.mean(), 3)} (sd {100 * d.std():.3f}; $p$={pval(pv)}) & & \\")
    lines.append(rf" & keep-alive 10 min & & {pct(s.keepalive10_cold.mean())} & {pm(s[s.setting == 'plain'].keepalive10_cold_slow, 1)} & \\")
    if sc == "1":
        lines.append(r"\addlinespace")
write("tab_seeds", r"""
\begin{table*}[t]
  \centering
  \caption{Tree policy on Azure 2021 over five seeds, without and with each function's cold-start time as an input, at normal and doubled cold-start times. Cold-start rate in \% at matched memory (mean $\pm$ standard deviation); slow: functions with a 90th-percentile cold-start time over 5\,s; last column: change against keep-alive at 1.0$\times$ and seeds better. ``Difference'' is with minus without, in percentage points, with a paired $t$-test.}
  \label{tab:seeds}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}llcccc@{}}
    \toprule
    Cold starts & Cold-start-time input & 0.8$\times$ memory & 1.0$\times$ memory & Slow, 1.0$\times$ & vs.\ keep-alive \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# ------------------------------------------------------------------------- Table: overhead
b = pd.read_csv(R / "benchmark_overhead.csv")
lines = []
for _, r in b.iterrows():
    lines.append(f"{int(r.functions):,} & {'yes' if r.attention else 'no'} & {r.train_step_s:.2f}\\,s & "
                 f"{r.inference_s_per_hour:.1f}\\,s & {r.us_per_decision:.2f}\\,$\\mu$s \\\\")
write("tab_bench", r"""
\begin{table}[t]
  \centering
  \caption{Cost of the streaming model at the size used in all experiments (one layer, $d=64$, $K=8$, 13 inputs) on two vCPUs of an Intel Xeon at 2.8\,GHz, random inputs, applications of four functions.}
  \label{tab:bench}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}rlrrr@{}}
    \toprule
    Functions & Attention & \hdr{Train step}{(300\,s window)} & \hdr{Inference}{per hour of trace} & \hdr{Per}{decision} \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table}
""")

# ------------------------------------------------------------------------- Table: cross-function early signal
SIG_FILES = {"azure2021": "Azure 2021", "huawei_day30": "Huawei day 30"}
if all((R / "early_signal" / f"{ds}_controls.csv").exists() for ds in SIG_FILES):
    lines = []
    for ds, dname in SIG_FILES.items():
        e = pd.read_csv(R / "early_signal" / f"{ds}_controls.csv")
        e = e[(e.lead_s == 2) & (e.window_s == 60)]
        order = [b for b in ("rare", "medium", "frequent") if b in set(e.bucket)]
        for i, b in enumerate(order):
            r = e[e.bucket == b].iloc[0]
            lines.append(f"{dname if i == 0 else ''} & {b} & {int(r.n_calls):,} & {100 * r.p_before:.1f} & "
                         f"{r.lift_uniform:.2f} & {r.lift_same_hour:.2f} & {r.lift_app_active:.2f} & "
                         f"{r.hints_per_hit:.0f}" + r" \\")
        if ds == "azure2021":
            lines.append(r"\addlinespace")
    write("tab_signal", r"""
\begin{table}[t]
  \centering
  \caption{Early-signal test for cold-prone invocations (previous invocation of the same function more than 10 min earlier), lead 2\,s, window 60\,s. ``Signal'': share of these invocations preceded by an invocation of another function of the same application within the window. Each lift divides this share by the corresponding share at reference times: random times (uniform), the same time of day on a random day (same hour), and times within 10 min after activity of the application (application active). The last column counts the signals on which a policy would act per cold start avoided.}
  \label{tab:signal}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}llrrrrrr@{}}
    \toprule
    Trace & Class & Invocations & Signal (\%) & \hdr{Lift}{uniform} & \hdr{Lift}{same hour} & \hdr{Lift}{app.\ active} & \hdr{Signals per}{avoided cold start} \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table}
""")

# ------------------------------------------------------------------------- Table: prediction targets
TARGETS = [("own + app, no lead", "Near-term"), ("own + app + lead", "Lead-shifted"),
           ("own + app, union window", "Union")]
SETTINGS = [("", r"Default ($H=60$\,s)"), ("_h10", r"Hold $H=10$\,s"), ("_x2", r"Doubled cold-start durations")]
tg = {sfx: collect(f"policies_azure2021{sfx}_targets_s*") for sfx, _ in SETTINGS}
if all(len(v) for v in tg.values()):
    lines = []
    for sfx, sname in SETTINGS:
        P = tg[sfx]
        ka = P.drop_duplicates("run")
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{sname}}} ({ka.run.nunique()} seeds; keep-alive 10 min: "
                     rf"{pct(ka.ka.mean())}\% at 1.0$\times$)}} \\")
        base = P[P.family == TARGETS[0][0]].set_index("seed")["c1.0"]
        for fam, tname in TARGETS:
            g = P[P.family == fam]
            better = int((g["c1.0"] < g["ka"]).sum())
            if fam == TARGETS[0][0]:
                diff = "--"
            else:
                d = (g.set_index("seed")["c1.0"] - base).dropna()
                p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else np.nan
                diff = f"{sg(100 * d.mean(), 3)} ({int((d < 0).sum())}/{len(d)}; $p$={pval(p)})"
            lines.append(rf"\quad {tname} & {pm(g['c0.8'])} & {pm(g['c1.0'])} & {better}/{len(g)} & {diff} \\")
        if sfx != SETTINGS[-1][0]:
            lines.append(r"\addlinespace")
    write("tab_targets", r"""
\begin{table}[t]
  \centering
  \caption{Tree policy on Azure 2021 trained on the near-term, lead-shifted and union targets (cold-start rate in \%, mean $\pm$ standard deviation over three seeds, at matched idle memory). ``Better'': seeds in which the policy outperforms keep-alive at 1.0$\times$; last column: paired difference from the near-term target at 1.0$\times$ in percentage points (negative is better), seeds better than the near-term target and a paired $t$-test.}
  \label{tab:targets}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lcccc@{}}
    \toprule
    Target & 0.8$\times$ memory & 1.0$\times$ memory & Better & Difference from near-term \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table}
""")

# ------------------------------------------------------------------------- Table: budget controller
bud = sorted(R.glob("budget_azure2021_s*/risk.csv"))
if bud:
    d = pd.concat(pd.read_csv(f) for f in bud)
    d = d[d.sharing != "none (fixed 0.5)"]
    SHARE = [("per function", "Function"), ("per app", "Application"), ("per bucket", "Frequency class")]

    def rng(v, scale=100, nd=0):
        v = pd.Series(v).dropna()
        if not len(v):
            return "--"
        lo, hi = scale * v.min(), scale * v.max()
        return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}--{hi:.{nd}f}"

    lines = []
    for key, sname in SHARE:
        g = d[d.sharing == key]
        low = g[g.memory <= 1.0]
        lines.append(f"{sname} & {len(g)} & {rng(g.memory, 1, 2)} & {len(low)} & {rng(low.fns_within)} & "
                     f"{rng(g.fns_within)} & {rng(g.groups_within)}" + r" \\")
    write("tab_budget", r"""
\begin{table}[t]
  \centering
  \caption{Budget controller with the tree policy on Azure 2021: 12 settings per sharing level (three step sizes, two initial thresholds, budgets of 2\% and 5\%), each run with """ + f"{len(bud)}" + r""" seeds. Memory is relative to the 10-minute keep-alive. ``Functions within budget'': share of functions with at least 20 test invocations whose cold-start rate does not exceed the budget, over the runs at no more than keep-alive memory and over all runs; ``groups within budget'': the same share for the groups that share a threshold. Ranges are over runs.}
  \label{tab:budget}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lrcrccc@{}}
    \toprule
    \hdr{Threshold}{shared per} & Runs & Memory & \hdr{Runs at}{$\le$1.0$\times$} & \hdr{Functions within}{budget, $\le$1.0$\times$ (\%)} & \hdr{Functions within}{budget, all (\%)} & \hdr{Groups within}{budget (\%)} \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table}
""")

# =============================================================================== appendix: per-seed tables
for ds, dname in DS.items():
    P = gpu[ds]
    lines = []
    for seed, g in P.groupby("seed"):
        k = g.iloc[0]
        cells = [f"{pct(k.ka08)} / {pct(k.ka)}"]
        for fam in PRED:
            r = g[g.family == fam].iloc[0]
            cells.append(f"{pct(r['c0.8'])} / {pct(r['c1.0'])}")
        lines.append(f"{seed} & " + " & ".join(cells) + r" \\")
    write(f"app_pred_{ds}", r"""
\begin{table*}[!htbp]
  \centering
  \caption{""" + dname + r""": cold-start rate (\%) per seed at 0.8$\times$ / 1.0$\times$ the memory of the 10-minute keep-alive (keep-alive at 0.8$\times$ is read from its own curve).}
  \label{tab:app-pred-""" + ds + r"""}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}cccccc@{}}
    \toprule
    Seed & Keep-alive & Trees & LSTM, no attention & \sysname & Ensemble \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# per-seed baselines, realism and oracle
lines = []
for ds, dname in DS.items():
    P, Q = pkg[ds], real[ds]
    for seed, g in P.groupby("seed"):
        k = g.iloc[0]
        cr = g[g.family == "call rate only (no learning)"].iloc[0]
        tr = g[g.family == "own + app, no lead"].iloc[0]
        sa = g[g.family == "state-aware (trees)"].iloc[0]
        rl = Q[(Q.seed == seed) & (Q.family == "own + app, no lead")]
        o = g[(g.family == "oracle") & (g.param == "hold 60 s")]
        lines.append(f"{dname if seed == P.seed.min() else ''} & {seed} & {pct(k.ka)} & {pct(k['hist'])} @ {k.hist_mem:.2f} & "
                     f"{pct(cr['c1.0'])} & {pct(tr['c1.0'])} & {pct(sa['c1.0'])} & "
                     f"{pct(rl['c1.0'].iloc[0]) if len(rl) else '--'} & {o.mem.iloc[0]:.2f}" + r" \\")
    if ds == "azure2021":
        lines.append(r"\addlinespace")
write("app_base", r"""
\begin{table*}[!htbp]
  \centering
  \caption{Per-seed baselines (cold-start rate in \%). Learned policies are read at 1.0$\times$ the memory of the 10-minute keep-alive; the histogram is shown at its own memory; the last column is the memory at which the perfect predictor with a 60\,s hold has no cold start.}
  \label{tab:app-base}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}llccccccc@{}}
    \toprule
    Trace & Seed & Keep-alive & Histogram @ memory & Rate only & Trees & State-aware & Trees, conservative & Oracle memory \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# per-seed five-seed study
lines = []
for sc, lab in (("1", "normal"), ("2", r"$\times$2")):
    if sc not in s11:
        continue
    s = s11[sc]
    for seed, g in s.groupby("seed"):
        p, c = g[g.setting == "plain"].iloc[0], g[g.setting == "coldtime"].iloc[0]
        lines.append(f"{lab if seed == 0 else ''} & {seed} & {pct(p['cold@0.8'])} / {pct(p['cold@1.0'])} & "
                     f"{pct(c['cold@0.8'])} / {pct(c['cold@1.0'])} & {pct(p.keepalive10_cold_slow, 1)} & "
                     f"{pct(p['cold_slow@1.0'], 1)} & {pct(c['cold_slow@1.0'], 1)}" + r" \\")
    if sc == "1":
        lines.append(r"\addlinespace")
write("app_seeds", r"""
\begin{table*}[!htbp]
  \centering
  \caption{Tree policy on Azure 2021, per seed: cold-start rate (\%) at 0.8$\times$ / 1.0$\times$ memory without and with the cold-start-time input, and the rate for slow functions ($>$5\,s) under keep-alive and under both settings at 1.0$\times$. Keep-alive has 1.16\% overall in every seed; its slow-function rate changes with the profile assignment.}
  \label{tab:app-seeds}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}llccccc@{}}
    \toprule
    Cold starts & Seed & Without & With & Slow: keep-alive & Slow: without & Slow: with \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# full threshold sweep, seed 0, both traces
for ds, dname in DS.items():
    P = gpu[ds]
    g = P[P.seed == 0]
    lines = []
    for th in THETAS:
        cells = []
        for fam in PRED:
            sw = g[g.family == fam].sweep.iloc[0]
            key = th if th in sw.index else str(th)
            try:
                r = sw.loc[key]
                cells.append(f"{pct(r.cold_rate)} & {r.memory_vs_keepalive10:.2f}")
            except KeyError:
                cells.append("-- & --")
        lines.append(f"{th:g} & " + " & ".join(cells) + r" \\")
    write(f"app_sweep_{ds}", r"""
\begin{table*}[!htbp]
  \centering
  \caption{""" + dname + r""", seed 0: every operating point of the threshold sweep. Cold: cold-start rate in \%; mem.: idle memory relative to the 10-minute keep-alive.}
  \label{tab:app-sweep-""" + ds + r"""}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}c""" + "rr" * 4 + r"""@{}}
    \toprule
    & \multicolumn{2}{c}{Trees} & \multicolumn{2}{c}{LSTM, no attention} & \multicolumn{2}{c}{\sysname} & \multicolumn{2}{c}{Ensemble} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
    $\theta$ & Cold & Mem. & Cold & Mem. & Cold & Mem. & Cold & Mem. \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")

# AUPRC, both targets
lines = []
for ds in DS:
    p = pd.concat(pd.read_csv(Path(f) / "prediction.csv") for f in runs(f"policies_{ds}_predictors_s*"))
    t = p.groupby(["model", "bucket", "target"]).agg(auprc=("auprc", "mean"), base=("base_rate", "mean")).reset_index()
    for b in buckets[ds]:
        for fam in PRED:
            r = t[(t.model == fam) & (t.bucket == b)].set_index("target")
            if not len(r):
                continue
            lines.append(f"{DS[ds] if (b == buckets[ds][0] and fam == PRED[0]) else ''} & "
                         f"{b if fam == PRED[0] else ''} & {NAME[fam]} & {r.loc['near-term', 'auprc']:.3f} & "
                         f"{r.loc['lead-shifted', 'auprc']:.3f} & {100 * r.loc['near-term', 'base']:.2f}" + r" \\")
    if ds == "azure2021":
        lines.append(r"\addlinespace")
write("app_auprc", r"""
\begin{table*}[!htbp]
  \centering
  \caption{AUPRC of each predictor against the near-term target, on which all were trained, and against the lead-shifted target, mean of three seeds; base rate is the share of positive function-seconds (near-term target) in the sample.}
  \label{tab:app-auprc}
  \tablefont
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{@{}lllccc@{}}
    \toprule
    Trace & Class & Predictor & Near-term & Lead-shifted & Base rate (\%) \\
    \midrule
""" + "\n".join("    " + x for x in lines) + r"""
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}
""")
