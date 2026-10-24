"""Budget controller study (paper, RQ5): can an online threshold keep a cold-start budget?

Trains one predictor and then compares, on the same predictions:
  sharing : one threshold per function | per application | per frequency class
  eta     : 0.005 | 0.05 | 0.2
  start   : every threshold starts at 0.5 | cautious: rare and unseen functions start at 0.05
  budget  : alpha = 2% and 5%
For each setting it reports the pooled cold-start rate, idle memory relative to a 10-minute keep-alive,
the share of busy functions (>= 20 test calls) and of groups within budget, and whether the group bound
miss_g <= alpha + (1 + eta*k) / (eta * P_g) held over the whole run (Proposition 1).

Outputs: results/budget_<dataset>[_x<scale>][_<tag>]/risk.csv, summary.txt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stawarm.analysis import frequency_buckets_from_events
from stawarm.policies.baselines import FixedKeepAlive
from stawarm.policies.threshold import ThresholdPolicy
from stawarm.seconds import app_index, assign_profiles, events_to_active, train_and_predict
from stawarm.sim.simulator import simulate_gaps, simulate_slots

DAY = 86400
H = 60
KA_MIN = 60
THETA_STAR = 0.5


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["azure2021", "huawei_day30"], default="azure2021")
    ap.add_argument("--model", choices=["union", "nolead", "lead"], default="union",
                    help="training target of the predictor")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--etas", default="0.005,0.05,0.2")
    ap.add_argument("--alphas", default="0.02,0.05")
    ap.add_argument("--lead-feature", action="store_true",
                    help="give the predictor each function's cold-start lead as an input feature")
    ap.add_argument("--tag", default="", help="suffix for the results folder, so runs don't overwrite each other")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(f"results/budget_{args.dataset}" + ("" if args.scale == 1.0 else f"_x{args.scale:g}") +
               (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "azure2021":
        ev = pd.read_parquet("data/processed/azure2021_events.parquet")
        train_end, test_start = 9 * DAY, 10 * DAY
    else:
        ev = pd.read_parquet("data/processed/huawei_r2_day30_events.parquet")
        train_end, test_start = 14 * 3600, 16 * 3600
    active = events_to_active(ev)
    N, T = active.shape
    apps = app_index(ev, N)
    bucket = frequency_buckets_from_events(ev, 0, train_end).reindex(range(N)).fillna("unseen").to_numpy()
    rate = active[:, :train_end].sum(1) / (train_end / DAY)
    prof = assign_profiles(N, pd.read_parquet("data/processed/huawei_r2_cold_start_stats.parquet"),
                           seed=args.seed, scale=args.scale)
    lead, mem = prof["lead_s"].to_numpy(), prof["mem_mb"].to_numpy()
    x = active[:, train_end:].astype(np.int8)
    warm = test_start - train_end
    log(f"{N} functions; model target '{args.model}'")

    ka = FixedKeepAlive(10 * 60); ka.reset(N, x.shape[1])
    ref = simulate_gaps(x, ka.windows, mem, 1.0, warmup=warm)
    log(f"10-minute keep-alive: {ref.cold_rate:.2%} cold")

    target_lead = np.ones(N, np.int64) if args.model == "nolead" else lead
    P, _, _ = train_and_predict(active, apps, rate, target_lead, H, train_end, train_end, True,
                                seed=args.seed, log=log, union=(args.model == "union"),
                                lead_feature=lead if args.lead_feature else None)
    fixed = simulate_slots(x, ThresholdPolicy(P, THETA_STAR, lead=lead, hold=H, keepalive_min=KA_MIN), mem, 1.0,
                           warmup=warm)
    log(f"fixed threshold {THETA_STAR} (no risk control): {fixed.cold_rate:.2%} cold")

    groupings = {
        "per function": np.arange(N),
        "per app": np.where(apps >= 0, apps, apps.max() + 1 + np.arange(N)),
        "per bucket": pd.factorize(bucket)[0],
    }
    rare = np.isin(bucket, ["rare", "unseen"])
    starts = {"0.5": np.full(N, 0.5), "cautious": np.where(rare, 0.05, 0.5)}
    rows = [dict(sharing="none (fixed 0.5)", eta=np.nan, start="-", alpha=np.nan, **metrics(fixed, ref, None, None))]
    etas = [float(e) for e in args.etas.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    for gname, groups in groupings.items():
        for eta in etas:
            for sname, th0 in starts.items():
                for alpha in alphas:
                    pol = ThresholdPolicy(P, THETA_STAR, lead=lead, hold=H, keepalive_min=KA_MIN, alpha=alpha,
                                          eta=eta, theta0=th0, groups=groups)
                    r = simulate_slots(x, pol, mem, 1.0, warmup=warm)
                    rows.append(dict(sharing=gname, eta=eta, start=sname, alpha=alpha,
                                     **metrics(r, ref, pol, alpha)))
            log(f"{gname}, eta {eta} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "risk.csv", index=False)

    show = df.copy()
    for c in ["cold_rate", "fns_within", "groups_within", "worst_fn"]:
        show[c] = (100 * show[c]).round(1).astype(str) + "%"
    show["memory"] = show["memory"].round(2)
    show["alpha"] = show["alpha"].map(lambda a: "-" if np.isnan(a) else f"{a:.0%}")
    cols = ["sharing", "eta", "start", "alpha", "cold_rate", "memory", "fns_within", "groups_within",
            "worst_fn", "bound_holds", "median_slack"]
    ok = df[(df.fns_within >= 0.9) & (df.memory <= 1.0)].sort_values("memory")
    text = [f"Risk-control study ({args.dataset}, model '{args.model}', scale {args.scale:g}). "
            f"10-minute keep-alive: {ref.cold_rate:.2%} cold = memory 1.00.",
            "fns_within = share of functions with >= 20 test calls whose cold-start rate <= budget; "
            "groups_within = same for groups; median_slack = the bound's allowance above alpha for the median group.",
            "", show[cols].to_string(index=False), "",
            "Configurations keeping >= 90% of busy functions within budget at no more memory than keep-alive:",
            ok[cols].head(10).to_string(index=False) if len(ok) else "  none",
            "", f"finished in {(time.time() - t0) / 60:.1f} min"]
    (out / "summary.txt").write_text("\n".join(text))
    print("\n".join(text))
    log(f"wrote {out}")


def metrics(r, ref, pol, alpha) -> dict:
    inv, cold = r.invoked, r.cold
    busy = inv >= 20
    m = {"cold_rate": r.cold_rate, "memory": r.wasted_gb_s / max(ref.wasted_gb_s, 1e-9)}
    if pol is None:
        return m | {"fns_within": np.nan, "groups_within": np.nan, "worst_fn": np.nan,
                    "bound_holds": None, "median_slack": np.nan}
    rate_f = cold[busy] / inv[busy]
    g = pol.g
    gi = np.bincount(g, weights=inv, minlength=pol.n_groups)
    gc = np.bincount(g, weights=cold, minlength=pol.n_groups)
    gb = gi >= 20
    # the bound, over the whole simulated run (warm-up included), per group
    Pa = np.bincount(g, weights=r.invoked_all, minlength=pol.n_groups)
    Ca = np.bincount(g, weights=r.cold_all, minlength=pol.n_groups)
    slack = (1 + pol.eta * pol.max_step) / (pol.eta * np.maximum(Pa, 1))
    has = Pa > 0
    return m | {"fns_within": float((rate_f <= alpha).mean()) if busy.any() else np.nan,
                "groups_within": float((gc[gb] / gi[gb] <= alpha).mean()) if gb.any() else np.nan,
                "worst_fn": float(rate_f.max()) if busy.any() else np.nan,
                "bound_holds": bool(np.all(Ca[has] / Pa[has] <= alpha + slack[has])),
                "median_slack": float(np.median(slack[has]))}


if __name__ == "__main__":
    main()
