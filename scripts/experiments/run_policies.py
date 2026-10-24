"""Main policy comparison at one-second resolution (paper, RQ1, RQ2, RQ4, RQ6, RQ7).

Policies on the test period: fixed keep-alive (1-60 min), the hybrid histogram (Shahrad et al.), and
threshold policies driven by a predictor. --variants selects the predictors:
    app_nolead     gradient-boosted trees, own + application features, near-term target
    app_lead       same, lead-shifted target            own_lead   own features only, lead-shifted target
    app_union      same, union-window target            rate       call rate only (no learning)
    state_aware    near-term model for warm instances, lead-shifted model for cold ones (trees)
    lstm_full      LSTM without attention, full-trace training (GPU recommended)
    stawarm_full   STAWarm: LSTM + sparse attention over same-application functions
    ensemble       mean of app_nolead and lstm_full (list both before it)
--oracle adds the perfect predictor; --control-delay, --charge-loading and --random-durations make the
simulator more pessimistic; --lead-feature gives every model each function's cold-start time.
Cold-start durations and memory sizes are drawn from Huawei Region 2 measurements.

Inputs : data/processed/azure2021_events.parquet or huawei_r2_day30_events.parquet,
         data/processed/huawei_r2_cold_start_stats.parquet
Outputs: results/policies_<dataset>[suffixes]_<tag>/policies.csv, prediction.csv, summary.txt, pareto.png
See README.md for the exact commands behind each table of the paper.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from stawarm.analysis import frequency_buckets_from_events
from stawarm.policies.baselines import FixedKeepAlive, HybridHistogram
from stawarm.policies.threshold import StateAwarePolicy, ThresholdPolicy
from stawarm.seconds import app_index, assign_profiles, chunk_targets, events_to_active, train_and_predict
from stawarm.sim.simulator import simulate_gaps, simulate_slots

DAY = 86400
H = 60                   # a pre-warmed instance is held 60 s; targets look 60 s ahead
KA_MIN = 60              # learned policies keep an instance 60 s after each call, then decide
THETAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
ALPHAS = [0.02, 0.05, 0.10]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["azure2021", "huawei_day30"], default="azure2021")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply cold-start durations (sensitivity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=60,
                    help="seconds a pre-warmed instance is held and the target window length (default 60)")
    ap.add_argument("--variants", default="own_lead,app_lead,app_nolead",
                    help="comma list from own_lead, app_lead, app_nolead, app_union")
    ap.add_argument("--lead-feature", action="store_true",
                    help="give every model the function's cold-start lead time as an input feature")
    ap.add_argument("--thetas", default="", help="comma list of thresholds (default 0.02,0.05,0.1,0.2,0.3,0.5)")
    ap.add_argument("--tag", default="", help="suffix for the results folder, so runs don't overwrite each other")
    ap.add_argument("--oracle", action="store_true",
                    help="also simulate a perfect predictor (true future) inside the same decision layer")
    ap.add_argument("--control-delay", type=int, default=0, help="seconds before a pre-warm decision takes effect")
    ap.add_argument("--charge-loading", action="store_true", help="count loading time of pre-warms as idle memory")
    ap.add_argument("--random-durations", action="store_true",
                    help="each start takes a random time drawn from the function's cold-start distribution")
    ap.add_argument("--hidden", type=int, default=128, help="LSTM hidden size (lstm_full / stawarm_full)")
    ap.add_argument("--layers", type=int, default=2, help="LSTM layers (lstm_full / stawarm_full)")
    ap.add_argument("--max-epochs", type=int, default=20, help="epochs with early stopping (lstm_full / stawarm_full)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--risk-on", default="app_lead",
                    help="which variant the risk-controlled runs use (must be in --variants)")
    args = ap.parse_args()
    global H, THETAS
    H = args.horizon
    if args.thetas:
        THETAS = [float(v) for v in args.thetas.split(",")]
    t0 = time.time()
    suffix = ("" if args.scale == 1.0 else f"_x{args.scale:g}") + ("" if H == 60 else f"_h{H}") + \
             (f"_d{args.control_delay}" if args.control_delay else "") + \
             ("_load" if args.charge_loading else "") + ("_rand" if args.random_durations else "") + \
             (f"_{args.tag}" if args.tag else "")
    out = Path(f"results/policies_{args.dataset}{suffix}")
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "azure2021":
        ev = pd.read_parquet("data/processed/azure2021_events.parquet")
        train_end, test_start = 9 * DAY, 10 * DAY             # train days 1-9, warm-up day 10, test 11-14
    else:
        ev = pd.read_parquet("data/processed/huawei_r2_day30_events.parquet")
        train_end, test_start = 14 * 3600, 16 * 3600          # train 0-14 h, warm-up 14-16 h, test 16-24 h
    active = events_to_active(ev)
    N, T = active.shape
    apps = app_index(ev, N)
    bucket_s = frequency_buckets_from_events(ev, 0, train_end)
    bucket = bucket_s.reindex(range(N)).fillna("unseen").to_numpy()
    rate = active[:, :train_end].sum(1) / (train_end / DAY)
    stats = pd.read_parquet("data/processed/huawei_r2_cold_start_stats.parquet")
    prof = assign_profiles(N, stats, seed=args.seed, scale=args.scale)
    lead, mem, d50 = prof["lead_s"].to_numpy(), prof["mem_mb"].to_numpy(), prof["d50_s"].to_numpy()
    # functions grouped by how long their cold start takes (lead time), to see where lead-shifting matters
    lead_group = np.select([lead <= 1, lead <= 5], ["lead 1s", "lead 2-5s"], "lead >5s")
    LEAD_GROUPS = ["lead 1s", "lead 2-5s", "lead >5s"]
    log(f"{args.dataset}: {N} functions x {T:,} s; buckets {pd.Series(bucket).value_counts().to_dict()}; "
        f"lead time median {np.median(lead):.0f} s, max {lead.max()} s; "
        f"lead groups {pd.Series(lead_group).value_counts().to_dict()}; horizon {H} s")

    sim_kw = {"delay": args.control_delay, "charge_loading": args.charge_loading}
    if args.random_durations:
        d90 = prof["d90_s"].to_numpy(); sig = np.log(np.maximum(d90, 1e-3) / np.maximum(d50, 1e-3)) / 1.2816
        sig = np.clip(sig, 0.05, 2.0); dur_rng = np.random.default_rng(args.seed + 1000)
        sim_kw["start_duration"] = lambda rows: np.minimum(
            np.ceil(d50[rows] * np.exp(sig[rows] * dur_rng.standard_normal(len(rows)))), 120).astype(np.int64).clip(1)
    log(f"realism: delay {args.control_delay} s, charge loading {args.charge_loading}, random durations {args.random_durations}")

    warm_start = train_end
    x = active[:, warm_start:].astype(np.int8)
    warm = test_start - warm_start
    rows = []

    def record(res, family, param, extra=None):
        pf = res.per_function()
        inv = pf["invoked"].to_numpy(); cold = pf["cold"].to_numpy()
        row = {"family": family, "param": param, "cold_rate": res.cold_rate,
               "added_latency_ms_per_call": 1000 * float((cold * d50).sum() / max(inv.sum(), 1)),
               "wasted_gb_s": res.wasted_gb_s}
        for b in ["rare", "medium", "frequent"]:
            m = bucket == b
            row[f"cold_{b}"] = float(cold[m].sum() / max(inv[m].sum(), 1)) if m.any() else np.nan
        idle = pf["idle_warm"].to_numpy()
        for g in LEAD_GROUPS:
            m = lead_group == g
            row[f"cold_{g}"] = float(cold[m].sum() / max(inv[m].sum(), 1)) if m.any() else np.nan
            row[f"gb_s_{g}"] = float((idle[m] * mem[m]).sum() / 1024.0)
        row.update(extra or {})
        rows.append(row)

    # ---------------------------------------------------------------- baselines
    for k_min in [1, 2, 5, 10, 20, 60]:
        pol = FixedKeepAlive(k_min * 60)
        pol.reset(N, x.shape[1])
        record(simulate_gaps(x, pol.windows, mem, 1.0, warmup=warm), "fixed keep-alive", f"{k_min} min")
    log("keep-alive done")
    hist = HybridHistogram(range_slots=240, bin_slots=60)
    record(simulate_gaps(active, hist.windows, mem, 1.0, warmup=test_start,
                         prewarm_cost=lead if args.charge_loading else None), "hybrid histogram", "4 h range")
    log("histogram done")

    # ---------------------------------------------------------------- learned policies
    test_target = None
    pred_rows = []
    # target: "lead" = call in [t+lead, t+lead+H-1]; "none" = call in [t+1, t+H];
    #         "union" = call in [t+1, t+lead+H-1] (near term for warm instances + far part for cold ones)
    all_variants = {"own_lead": ("own + lead", False, "lead"), "app_lead": ("own + app + lead", True, "lead"),
                    "app_nolead": ("own + app, no lead", True, "none"),
                    "app_union": ("own + app, union window", True, "union"),
                    # LSTM predictors (no-lead target). Inputs are the function's own features only;
                    # STAWarm adds sparse attention over the LSTM states of same-app functions.
                    "lstm_own": ("LSTM, own (no attention)", False, "none"),
                    "stawarm": ("STAWarm (LSTM + app attention)", True, "none"),
                    # full-trace LSTMs (GPU-friendly): same inputs as the trees, validation early stopping
                    "lstm_full": ("LSTM-full (no attention)", True, "none"),
                    "stawarm_full": ("STAWarm-full (LSTM + app attention)", True, "none"),
                    # naive baseline: each function's training call rate, no learning
                    "rate": ("call rate only (no learning)", False, "none"),
                    # state-aware: near-term model when an instance exists, lead-shifted model when not
                    "state_aware": ("state-aware (trees)", True, "state"),
                    # average of the tree and LSTM-full probabilities (needs app_nolead and lstm_full earlier)
                    "ensemble": ("ensemble (trees + LSTM-full)", True, "none")}
    keys = [k.strip() for k in args.variants.split(",")]
    variants = [(k,) + all_variants[k] for k in keys]
    risk_name = all_variants[args.risk_on][0] if args.risk_on in keys else None
    rng = np.random.default_rng(args.seed)
    keep_P = {}                     # predictions kept for the ensemble
    for key, name, use_app, target in variants:
        log(f"training '{name}'")
        target_lead = np.ones(N, np.int64) if target == "none" else lead
        lead_in = lead if args.lead_feature else None
        P_cold = None
        if key == "ensemble":
            missing = [k for k in ("app_nolead", "lstm_full") if k not in keep_P]
            if missing:
                raise SystemExit(f"ensemble needs {missing} earlier in --variants")
            P = ((keep_P["app_nolead"].astype(np.float32) + keep_P["lstm_full"].astype(np.float32)) / 2).astype(np.float16)
            clf, names = None, ["mean of trees and LSTM-full"]
        elif key == "rate":
            p_rate = 1 - np.exp(-np.maximum(rate, 0) / DAY * H)     # P(call in next H s) at the training rate
            P = np.broadcast_to(p_rate.astype(np.float32)[:, None], (N, T - warm_start))
            clf, names = None, ["log_rate"]
        elif key == "state_aware":
            P, clf, names = train_and_predict(active, apps, rate, np.ones(N, np.int64), H, train_end, warm_start,
                                              True, seed=args.seed, log=log, lead_feature=lead_in)
            log("    second model (lead-shifted target) for instances that must start from cold")
            P_cold, _, _ = train_and_predict(active, apps, rate, lead, H, train_end, warm_start, True,
                                             seed=args.seed, log=log, lead_feature=lead_in)
        elif key in ("lstm_full", "stawarm_full"):
            from stawarm.models.stawarm_stream import train_and_predict_lstm_full
            P, clf, names = train_and_predict_lstm_full(active, apps, rate, target_lead, H, train_end, warm_start,
                                                        attention=(key == "stawarm_full"), seed=args.seed, log=log,
                                                        lead_feature=lead_in, hidden=args.hidden, layers=args.layers,
                                                        max_epochs=args.max_epochs, device=args.device,
                                                        bptt=900 if N <= 800 else 200, block=900 if N <= 800 else 300)
        elif key in ("lstm_own", "stawarm"):
            from stawarm.models.stawarm_stream import train_and_predict_lstm
            P, clf, names = train_and_predict_lstm(active, apps, rate, target_lead, H, train_end, warm_start,
                                                   attention=(key == "stawarm"), seed=args.seed, log=log,
                                                   lead_feature=lead_in, window=900 if N <= 800 else 600)
        else:
            P, clf, names = train_and_predict(active, apps, rate, target_lead, H, train_end, warm_start, use_app,
                                              seed=args.seed, log=log, union=(target == "union"),
                                              lead_feature=lead_in)
        if key in ("app_nolead", "lstm_full") and "ensemble" in keys:
            keep_P[key] = P
        # prediction quality against the physically correct target (real lead), test period only
        if test_target is None:
            hi_t = T - H - int(lead.max())
            test_target = {"lead-shifted": chunk_targets(active, test_start, hi_t, lead, H),
                           "near-term": chunk_targets(active, test_start, hi_t, np.ones(N, np.int64), H)}
        for tname, tt_mat in test_target.items():
            Pt = P[:, test_start - warm_start: test_start - warm_start + tt_mat.shape[1]]
            for b in ["rare", "medium", "frequent"]:
                m = np.flatnonzero(bucket == b)
                if len(m) == 0:
                    continue
                ii = rng.choice(m, 3_000_000); tt = rng.integers(0, tt_mat.shape[1], 3_000_000)
                yy = tt_mat[ii, tt]
                if yy.any() and not yy.all():
                    pred_rows.append({"model": name, "target": tname, "bucket": b,
                                      "auprc": average_precision_score(yy, np.asarray(Pt[ii, tt], np.float32)),
                                      "base_rate": float(yy.mean())})
        for th in THETAS:
            if P_cold is not None:
                pol = StateAwarePolicy(P, P_cold, th, lead=lead, hold=H, keepalive_min=KA_MIN, name=f"{name}@{th}")
            else:
                pol = ThresholdPolicy(P, theta_star=th, lead=lead, hold=H, keepalive_min=KA_MIN, name=f"{name}@{th}")
            record(simulate_slots(x, pol, mem, 1.0, warmup=warm, **sim_kw), f"learned: {name}", th)
        log(f"'{name}' simulated")
        if name == risk_name:
            for alpha in ALPHAS:
                pol = ThresholdPolicy(P, theta_star=0.5, lead=lead, hold=H, keepalive_min=KA_MIN,
                                      alpha=alpha, eta=0.005, theta0=0.5)
                r_all = simulate_slots(x, pol, mem, 1.0, warmup=0, **sim_kw)
                pf = r_all.per_function()
                P_n, miss = pf["invoked"].to_numpy(), pf["cold"].to_numpy()
                bound_ok = bool(np.all((P_n == 0) | (miss / np.maximum(P_n, 1) <= alpha + 1.005 / (0.005 * np.maximum(P_n, 1)))))
                pol = ThresholdPolicy(P, theta_star=0.5, lead=lead, hold=H, keepalive_min=KA_MIN,
                                      alpha=alpha, eta=0.005, theta0=0.5)
                r_test = simulate_slots(x, pol, mem, 1.0, warmup=warm, **sim_kw)
                pt = r_test.per_function()
                busy = pt["invoked"].to_numpy() >= 20        # per-function rates are noisy below 20 calls
                rate_f = pt["cold"].to_numpy()[busy] / pt["invoked"].to_numpy()[busy]
                record(r_test, "risk-controlled", alpha,
                       {"bound_holds": bound_ok, "budget": alpha,
                        "fns_within_budget": float((rate_f <= alpha).mean()) if busy.any() else np.nan,
                        "fns_counted": int(busy.sum()),
                        "worst_fn_cold_rate": float(rate_f.max()) if busy.any() else np.nan})
            log("risk-controlled done")
        del P, P_cold

    if args.oracle:
        # a perfect predictor in the same decision layer: knows whether a call comes from now until the end
        # of the hold (union window). Holds of 5-60 s trace its trade-off curve.
        for Hh in (5, 15, 30, 60):
            Y = np.zeros((N, T - warm_start), np.float16)
            Y[:] = chunk_targets(active, warm_start, T, lead, Hh, union=True)
            pol = ThresholdPolicy(Y, theta_star=0.5, lead=lead, hold=Hh, keepalive_min=KA_MIN, name=f"oracle H={Hh}")
            record(simulate_slots(x, pol, mem, 1.0, warmup=warm, **sim_kw), "oracle (perfect predictor)", f"hold {Hh} s")
            del Y
        log("oracle done")

    df = pd.DataFrame(rows)
    ref = df[(df.family == "fixed keep-alive") & (df.param == "10 min")]["wasted_gb_s"].iloc[0]
    df["memory_vs_keepalive10"] = df["wasted_gb_s"] / ref
    df.to_csv(out / "policies.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "prediction.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["family", "param", "cold_rate", "cold_rare", "cold_medium", "added_latency_ms_per_call",
            "memory_vs_keepalive10"]
    table = df[cols].copy()
    for c in ["cold_rate", "cold_rare", "cold_medium"]:
        table[c] = (100 * table[c]).round(2).astype(str) + "%"
    table["added_latency_ms_per_call"] = table["added_latency_ms_per_call"].round(1)
    table["memory_vs_keepalive10"] = table["memory_vs_keepalive10"].round(2)
    text = [table.to_string(index=False), "", "Prediction quality (AUPRC, real lead-time target, test period):",
            pd.DataFrame(pred_rows).round(4).to_string(index=False)]
    # cold-start rate by lead group: where lead-shifted targets would matter most
    lg = df[df.family.str.startswith("learned") | (df.param == "10 min")].copy()
    lg_cols = ["family", "param"] + [f"cold_{g}" for g in LEAD_GROUPS] + ["memory_vs_keepalive10"]
    for g in LEAD_GROUPS:
        lg[f"cold_{g}"] = (100 * lg[f"cold_{g}"]).round(2).astype(str) + "%"
    lg["memory_vs_keepalive10"] = lg["memory_vs_keepalive10"].round(2)
    counts = pd.Series(lead_group).value_counts().reindex(LEAD_GROUPS).fillna(0).astype(int).to_dict()
    text += ["", f"Cold-start rate by cold-start duration (functions per group: {counts}):",
             lg[lg_cols].to_string(index=False)]
    rc = df[df.family == "risk-controlled"]
    if len(rc):
        r = rc[["budget", "cold_rate", "fns_within_budget", "fns_counted", "worst_fn_cold_rate", "bound_holds",
                "memory_vs_keepalive10"]].copy()
        r["cold_rate"] = (100 * r["cold_rate"]).round(2).astype(str) + "%"
        r["fns_within_budget"] = (100 * r["fns_within_budget"]).round(1).astype(str) + "%"
        r["worst_fn_cold_rate"] = (100 * r["worst_fn_cold_rate"]).round(2).astype(str) + "%"
        r["memory_vs_keepalive10"] = r["memory_vs_keepalive10"].round(2)
        r["fns_counted"] = r["fns_counted"].astype(int)
        r["budget"] = (100 * r["budget"]).round(0).astype(int).astype(str) + "%"
        text += ["", "Risk control: budget vs achieved (functions with >= 20 test calls; "
                 "bound_holds = Proposition 1 bound held for every function over the whole run):",
                 r.to_string(index=False)]
    # best cold-start rate per policy at no more memory than a 10-minute keep-alive
    within = df[df["memory_vs_keepalive10"] <= 1.0].sort_values("cold_rate").groupby("family").head(1)
    text += ["", "Best cold-start rate per policy family using no more memory than a 10-minute keep-alive:",
             within[["family", "param", "cold_rate", "memory_vs_keepalive10"]].round(4).to_string(index=False),
             "", f"finished in {(time.time() - t0) / 60:.1f} min"]
    (out / "summary.txt").write_text("\n".join(text))
    print("\n".join(text))
    plot_pareto(df, out / "pareto.png", args.dataset)
    log(f"wrote {out}")


def plot_pareto(df: pd.DataFrame, path: Path, dataset: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("fixed keep-alive", "#2a78d6", "o"), ("hybrid histogram", "#eb6834", "D"),
              ("learned: own + lead", "#1baf7a", "s"), ("learned: own + app + lead", "#eda100", "^"),
              ("learned: own + app, no lead", "#e87ba4", "v"), ("learned: own + app, union window", "#7b5cc4", "P"),
              ("learned: LSTM, own (no attention)", "#6b6a64", "x"),
              ("learned: STAWarm (LSTM + app attention)", "#0b0b0b", "*")]
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for fam, color, marker in series:
        d = df[df.family == fam].sort_values("memory_vs_keepalive10")
        if d.empty:
            continue
        ax.plot(d["memory_vs_keepalive10"], 100 * d["cold_rate"], color=color, lw=2, marker=marker, ms=6,
                label=fam.replace("learned: ", "learned, "), zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Idle memory-time, relative to a 10-minute keep-alive", color="#52514e")
    ax.set_ylabel("Cold-start rate (%, log scale)", color="#52514e")
    ax.set_title(f"Cold starts vs idle memory, test period ({dataset})", color="#0b0b0b", loc="left", fontsize=11)
    ax.grid(True, color="#e6e5e0", lw=0.8, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#b9b8b2")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=8.5, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
