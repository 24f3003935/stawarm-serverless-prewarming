"""Pilot study on the Azure Functions 2019 trace (one-minute slots, 7 days).

Question: do other functions of the same owner carry a usable early signal for pre-warming,
especially for rarely invoked functions? A minute-level, model-agnostic check of the cross-function
hypothesis.

Setup
  * sample owners with 2-20 functions (multi-function applications), about 2,000 functions
  * train days 1-4, validation day 5, test days 6-7 (a weekend, so some drift is included)
  * target: function invoked in the next slot (lead = 1 minute, H = 1)
  * predictors: gradient-boosted trees on (a) the function's own history, (b) own history plus the
    top-3 same-owner neighbours chosen by lagged cross-correlation on training days
  * policies on the test days: fixed keep-alive (several windows), hybrid histogram,
    and threshold policies driven by (a) and (b), swept over the threshold
Outputs go to results/pilot_azure2019/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from stawarm.data.azure2019 import function_index, load_azure2019
from stawarm.data.common import frequency_bucket
from stawarm.features import build_features, candidate_neighbours, future_window_any
from stawarm.policies.baselines import FixedKeepAlive, HybridHistogram
from stawarm.policies.threshold import ThresholdPolicy
from stawarm.sim.simulator import simulate_gaps, simulate_slots

DAY = 1440


def pick_sample(root: str, n_target: int, seed: int) -> pd.MultiIndex:
    idx = function_index(root, 1)
    idx = idx[idx["active_minutes"] > 0]
    owners = idx.groupby(level="HashOwner").size()
    owners = owners[(owners >= 2) & (owners <= 20)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(owners.index.to_numpy())
    chosen, total = [], 0
    for o in order:
        chosen.append(o)
        total += owners[o]
        if total >= n_target:
            break
    sel = idx[idx.index.get_level_values("HashOwner").isin(chosen)]
    return sel.index


def sample_rows(y: np.ndarray, t_lo: int, t_hi: int, n_neg: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """All positive (i, t) in [t_lo, t_hi) plus a uniform sample of negatives."""
    sub = y[:, t_lo:t_hi]
    pi, pt = np.nonzero(sub)
    ni = rng.integers(0, y.shape[0], n_neg)
    nt = rng.integers(0, t_hi - t_lo, n_neg)
    keep = ~sub[ni, nt]
    i = np.concatenate([pi, ni[keep]])
    t = np.concatenate([pt, nt[keep]]) + t_lo
    return i, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/azure2019")
    ap.add_argument("--out", default="results/pilot_azure2019")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    sel = pick_sample(args.root, args.n, args.seed)
    tr = load_azure2019(args.root, range(1, 8), select=sel)
    x = tr.counts
    active = x > 0
    N, T = x.shape
    TRAIN, VAL, TEST = (0, 4 * DAY), (4 * DAY, 5 * DAY), (5 * DAY, 7 * DAY)
    apd = (active[:, :TRAIN[1]].sum(1) / 4.0)
    bucket = frequency_bucket(apd)
    print(f"loaded {N} functions, {T} slots in {time.time()-t0:.0f}s; buckets:",
          pd.Series(bucket).value_counts().to_dict())

    # ---- neighbours from training days only
    nb = candidate_neighbours(active[:, :TRAIN[1]], tr.meta["owner"].to_numpy(), K=3, max_lag=5)
    has_nb = np.array([len(v) > 0 for v in nb])
    print(f"functions with >=1 correlated same-owner neighbour: {has_nb.mean():.1%}")

    # ---- target: invoked in the next slot (lead 1, H 1)
    lead = np.ones(N, np.int64)
    y = future_window_any(active, lead, H=1)

    results = {"n_functions": int(N), "buckets": pd.Series(bucket).value_counts().to_dict(),
               "share_with_neighbour": float(has_nb.mean())}
    probs = {}
    for name, use_nb in (("self", False), ("self+neighbours", True)):
        X, names = build_features(active, 60.0, neighbours=nb if use_nb else None, K=3)
        i_tr, t_tr = sample_rows(y, 60, TRAIN[1] - 1, n_neg=600_000, rng=rng)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                             early_stopping=True, random_state=args.seed)
        clf.fit(X[i_tr, t_tr], y[i_tr, t_tr])
        P = np.zeros((N, T), np.float32)
        for lo in range(VAL[0], T, DAY):          # predict val + test in daily chunks
            hi = min(lo + DAY, T)
            P[:, lo:hi] = clf.predict_proba(X[:, lo:hi].reshape(-1, X.shape[-1]))[:, 1].reshape(N, hi - lo)
        probs[name] = P
        del X
        # prediction quality on test, by frequency bucket (exclude the last slot: no label)
        row = {}
        for b in ["rare", "medium", "frequent", "all"]:
            m = np.ones(N, bool) if b == "all" else (bucket == b)
            yy = y[m, TEST[0]:T - 1].ravel(); pp = P[m, TEST[0]:T - 1].ravel()
            if yy.sum() == 0 or yy.all():
                continue
            row[b] = {"auprc": float(average_precision_score(yy, pp)), "auroc": float(roc_auc_score(yy, pp)),
                      "base_rate": float(yy.mean())}
        results[f"prediction_{name}"] = row
        print(name, json.dumps(row, indent=None))

    def bucket_cols(res):
        pf = res.per_function()
        out = {}
        for b in ["rare", "medium", "frequent"]:
            m = bucket == b
            out[f"cold_{b}"] = float(pf.loc[m, "cold"].sum() / max(pf.loc[m, "invoked"].sum(), 1))
        return out

    np.savez_compressed(out / "probs_test.npz", self=probs["self"][:, TEST[0]:], nb=probs["self+neighbours"][:, TEST[0]:])

    # ---- policies on test days; state warms up on the validation day
    mem = tr.meta["mem_mb"].to_numpy(float)
    xs = x[:, VAL[0]:]
    warm = DAY
    rows = []
    ref = None
    for k in [1, 5, 10, 20, 30, 60, 120, 240]:
        r = simulate_slots(xs, FixedKeepAlive(k), mem, 60.0, warmup=warm)
        if k == 10:
            ref = r
        rows.append({**r.summary(), **bucket_cols(r), "family": "fixed keep-alive", "param": k})
    # histogram learns from day 1 onwards; count only test days
    rh = simulate_gaps(x, HybridHistogram().windows, mem, 60.0, warmup=TEST[0], name="hybrid-histogram")
    rows.append({**rh.summary(), **bucket_cols(rh), "family": "hybrid histogram", "param": 240})
    for name, P in probs.items():
        Ps = P[:, VAL[0]:]
        for th in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            pol = ThresholdPolicy(Ps, theta_star=th, lead=1, hold=1, keepalive_min=1, name=f"{name}@{th}")
            r = simulate_slots(xs, pol, mem, 60.0, warmup=warm)
            rows.append({**r.summary(), **bucket_cols(r), "family": f"threshold ({name})", "param": th})
    # risk control: budget alpha on the model with neighbours
    for alpha in [0.05, 0.10, 0.20]:
        pol = ThresholdPolicy(probs["self+neighbours"][:, VAL[0]:], theta_star=0.9, lead=1, hold=1,
                              keepalive_min=1, alpha=alpha, eta=0.02, theta0=0.5, name=f"aci-{alpha}")
        r = simulate_slots(xs, pol, mem, 60.0, warmup=0)   # controller adapts from day 5
        pf = r.per_function()
        # realised miss rate on test days only
        rt = simulate_slots(xs, ThresholdPolicy(probs["self+neighbours"][:, VAL[0]:], theta_star=0.9, lead=1,
                                                hold=1, keepalive_min=1, alpha=alpha, eta=0.02, theta0=0.5),
                            mem, 60.0, warmup=warm)
        pf_all = r.per_function()             # Proposition 1 covers every invoked slot since the start
        Pa = pf_all["invoked"].to_numpy(); ma = pf_all["cold"].to_numpy()
        eta = 0.02
        ok_bound = (Pa == 0) | (ma / np.maximum(Pa, 1) <= alpha + (1 + eta) / (eta * np.maximum(Pa, 1)))
        pf = rt.per_function()
        P_n = pf["invoked"].to_numpy(); miss = pf["cold"].to_numpy()
        many = P_n >= 100
        within = float((miss[many] / P_n[many] <= alpha).mean()) if many.any() else float("nan")
        rows.append({**rt.summary(), **bucket_cols(rt), "family": "risk-controlled", "param": alpha,
                     "bound_holds_all": bool(ok_bound.all()),
                     "share_fn_within_alpha_100plus": within, "n_fn_100plus": int(many.sum())})
    df = pd.DataFrame(rows)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    df["wasted_vs_keepalive10"] = df["wasted_gb_s"] / ref.wasted_gb_s
    df.to_csv(out / "policies.csv", index=False)
    print(df[["family", "param", "cold_rate_pooled", "cold_pct_p75_fn", "wasted_vs_keepalive10"]].to_string(index=False))

    # per-bucket cold rates for the best comparable operating points
    def bucket_rates(res):
        pf = res.per_function()
        return {b: float(pf.loc[bucket == b, "cold"].sum() / max(pf.loc[bucket == b, "invoked"].sum(), 1))
                for b in ["rare", "medium", "frequent"]}
    results["bucket_cold_rates"] = {
        "keepalive-10": bucket_rates(ref),
        "hybrid-histogram": bucket_rates(rh)}
    json.dump(results, open(out / "summary.json", "w"), indent=2, default=float)
    np.save(out / "bucket.npy", bucket)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
