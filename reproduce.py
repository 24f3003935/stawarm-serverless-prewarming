"""Reproduce the CPU experiments of the paper on a local machine and compare them with the published results.

Experiments that train the neural predictors require a GPU and are run with the notebooks in
``notebooks/kaggle`` (01 and 02). Everything else is run by this script. New results are written to
``reproduction/results`` so that the published files in ``results/`` are never overwritten.

Stages (run in this order; each can be repeated, and finished runs are skipped):
    python reproduce.py env          record the software and hardware environment
    python reproduce.py data         download the public traces and prepare data/processed (once)
    python reproduce.py verify       quick check (about 20-60 min): tree policy, Azure, seed 0
    python reproduce.py cpu          all CPU experiments of the paper (many hours; resumable)
    python reproduce.py summaries    matched-memory summaries of the new runs
    python reproduce.py compare      compare the new runs with the published results
    python reproduce.py tables       LaTeX tables from the new runs
    python reproduce.py figures      figures from the new runs
Optional:
    python reproduce.py benchmark    computational cost of the streaming model
    python reproduce.py pilot        minute-level pilot on Azure Functions 2019 (downloads that trace)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "reproduction"
PAPER = ROOT / "results"
THETAS = "0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
PROCESSED = ["azure2021_events.parquet", "huawei_r2_day30_events.parquet", "huawei_r2_cold_start_stats.parquet"]


def run(script: str, *args: str, cwd: Path = WORK, log: str | None = None) -> None:
    """Run one script with the current interpreter; output goes to the console and to a log file."""
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / script), *args]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8")
    logdir = WORK / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"{log or Path(script).stem}.log"
    print(time.strftime("%H:%M:%S"), " ".join([Path(script).name, *args]), flush=True)
    t0 = time.time()
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            fh.write(line)
            print("    " + line, end="", flush=True)
        code = proc.wait()
    if code != 0:
        sys.exit(f"failed ({code}): {' '.join(cmd)}\nsee {logfile}")
    print(time.strftime("%H:%M:%S"), f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


def prepare_workdir() -> None:
    """reproduction/data/processed holds links to (or copies of) the prepared traces."""
    src = ROOT / "data" / "processed"
    missing = [f for f in PROCESSED if not (src / f).exists()]
    if missing:
        sys.exit(f"missing {missing} in {src}; run: python reproduce.py data")
    dst = WORK / "data" / "processed"
    dst.mkdir(parents=True, exist_ok=True)
    for f in PROCESSED:
        if not (dst / f).exists():
            try:
                os.link(src / f, dst / f)
            except OSError:
                shutil.copy2(src / f, dst / f)
    (WORK / "results").mkdir(parents=True, exist_ok=True)


def done(path: str) -> bool:
    return (WORK / "results" / path).exists()


# ----------------------------------------------------------------------------------------------- stages

def stage_env(_) -> None:
    WORK.mkdir(exist_ok=True)
    run("tools/record_environment.py", str(WORK / "environment.txt"), cwd=ROOT)


def stage_data(args) -> None:
    run("data/fetch_traces.py", "azure2021", cwd=ROOT)
    run("data/fetch_traces.py", "huawei", "--requests", cwd=ROOT)
    run("data/prepare_azure2021.py", cwd=ROOT)
    run("data/prepare_huawei_day30.py", cwd=ROOT)
    run("data/build_cold_start_profiles.py", cwd=ROOT)


def policies(dataset: str, tag: str, folder: str, *extra: str) -> None:
    if done(f"{folder}/policies.csv"):
        print("skip (already done):", folder)
        return
    run("experiments/run_policies.py", "--dataset", dataset, "--risk-on", "none", "--lead-feature",
        "--thetas", THETAS, "--tag", tag, *extra, log=folder)


def stage_verify(_) -> None:
    prepare_workdir()
    policies("azure2021", "verify_s0", "policies_azure2021_verify_s0", "--variants", "app_nolead", "--seed", "0")
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    from compare_results import compare_policies
    d = compare_policies(WORK / "results" / "policies_azure2021_verify_s0", PAPER / "policies_azure2021_baselines_s0")
    d = d[d.family.isin(["learned: own + app, no lead", "fixed keep-alive", "hybrid histogram"])]
    for c in ("new", "paper"):
        d[c] = (100 * d[c]).round(4)
    text = ("Quick check: tree policy, Azure 2021, seed 0, against policies_azure2021_baselines_s0 of the paper "
            "(cold-start rate in %; tree-based runs are deterministic, so differences should be zero with the "
            "same library versions)\n\n" + d.to_string(index=False))
    (WORK / "verify.txt").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n\nwrote {WORK / 'verify.txt'}")


def stage_cpu(_) -> None:
    prepare_workdir()
    seeds = ["0", "1", "2"]
    # baselines, state-aware trees and the perfect predictor; conservative simulator
    for ds in ("azure2021", "huawei_day30"):
        for s in seeds:
            policies(ds, f"baselines_s{s}", f"policies_{ds}_baselines_s{s}",
                     "--variants", "app_nolead,rate,state_aware", "--oracle", "--seed", s)
        for s in seeds:
            policies(ds, f"conservative_s{s}", f"policies_{ds}_d2_load_rand_conservative_s{s}",
                     "--variants", "app_nolead", "--control-delay", "2", "--charge-loading", "--random-durations",
                     "--seed", s)
    # five-seed study, normal and doubled cold-start durations (finished seeds are skipped by the script)
    run("experiments/run_seed_study.py", log="seedstudy_azure2021")
    run("experiments/run_seed_study.py", "--scale", "2", log="seedstudy_azure2021_x2")
    # early-signal test and controls (model-free)
    if not done("early_signal/huawei_day30_controls.csv"):
        run("experiments/run_early_signal_test.py")
        run("experiments/run_early_signal_controls.py")
        run("experiments/run_early_signal_controls.py", "--dataset", "huawei_day30")
    # budget controller
    for s in seeds:
        if not done(f"budget_azure2021_s{s}/risk.csv"):
            run("experiments/run_budget_controller.py", "--model", "nolead", "--lead-feature", "--seed", s,
                "--tag", f"s{s}", log=f"budget_azure2021_s{s}")
    # prediction targets: default settings, 10 s hold, doubled cold-start durations
    for prefix, extra in (("", []), ("_h10", ["--horizon", "10"]), ("_x2", ["--scale", "2"])):
        for s in seeds:
            policies("azure2021", f"targets_s{s}", f"policies_azure2021{prefix}_targets_s{s}",
                     "--variants", "app_nolead,app_lead,app_union", "--seed", s, *extra)


def stage_summaries(_) -> None:
    r = WORK / "results"
    for kind, name in (("predictors", "predictors"), ("baselines", "baselines"),
                       ("d2_load_rand_conservative", "conservative")):
        for ds, short in (("azure2021", "azure"), ("huawei_day30", "huawei")):
            if list(r.glob(f"policies_{ds}_{kind}_s*")):
                run("analysis/summarize_runs.py", f"results/policies_{ds}_{kind}_s*",
                    "--out", f"results/summary_{name}_{short}.txt", log="summaries")
    for prefix, name in (("", "targets_azure"), ("_h10", "targets_h10_azure"), ("_x2", "targets_x2_azure")):
        if list(r.glob(f"policies_azure2021{prefix}_targets_s*")):
            run("analysis/summarize_runs.py", f"results/policies_azure2021{prefix}_targets_s*",
                "--out", f"results/summary_{name}.txt", log="summaries")


def stage_compare(_) -> None:
    run("analysis/compare_results.py", str(WORK / "results"), str(PAPER), "--out", str(WORK / "comparison.txt"),
        cwd=ROOT)


def stage_tables(_) -> None:
    run("analysis/make_tables.py", "results")


def stage_figures(_) -> None:
    prepare_workdir()
    run("analysis/make_figures.py")


def stage_benchmark(_) -> None:
    prepare_workdir()
    run("experiments/run_overhead_benchmark.py")


def stage_pilot(_) -> None:
    run("data/fetch_traces.py", "azure2019", cwd=ROOT)
    run("experiments/run_pilot_azure2019.py", "--root", str(ROOT / "data" / "raw" / "azure2019"),
        "--out", str(WORK / "results" / "pilot_azure2019"))


STAGES = {"env": stage_env, "data": stage_data, "verify": stage_verify, "cpu": stage_cpu,
          "summaries": stage_summaries, "compare": stage_compare, "tables": stage_tables,
          "figures": stage_figures, "benchmark": stage_benchmark, "pilot": stage_pilot}


def main() -> None:
    try:
        sys.stdout.reconfigure(errors="replace")       # consoles that cannot show every character
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    args = ap.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
