# Evaluating attention-guided recurrent neural networks for serverless cold-start mitigation

Code, experiment pipelines and result files for the article *Evaluating attention-guided recurrent neural networks
for serverless cold-start mitigation* (under review).

The repository contains:

* a **per-second simulator** of the zero-to-one cold start, with per-function cold-start durations and memory sizes
  drawn from Huawei Cloud measurements, and optional control delay, charged loading time and random cold-start
  durations;
* **policies**: fixed keep-alive, the hybrid histogram policy of Shahrad et al. (USENIX ATC 2020), a rule based on
  invocation rates only, cost-threshold policies driven by gradient-boosted trees, an LSTM, **STAWarm** (a streaming
  LSTM with sparse attention over the functions of the same application) or an ensemble, a state-aware variant, a
  perfect predictor, and an online cold-start budget controller;
* **experiment pipelines** for every result of the article: a local runner for the CPU experiments and notebooks for
  the GPU experiments;
* the **result files** of all reported experiments (`results/`), from which every table and figure of the article is
  regenerated.

No trace data is included; all traces are public and are downloaded and prepared by the scripts.

## Repository layout

```
reproduce.py                    local runner for the CPU experiments (see "Reproducing the results")
src/stawarm/                    library
  data/                         loaders for Azure Functions 2019 and 2021 and Huawei Cloud 2025
  seconds.py                    per-second features, targets, cold-start profiles, tree predictor
  models/stawarm_stream.py      streaming STAWarm and LSTM; full-trace training with early stopping
  models/stawarm.py             window-based model used by the minute-level pilot
  sim/simulator.py              per-second and event-driven simulators with identical accounting
  policies/                     keep-alive, hybrid histogram, threshold and state-aware policies, budget controller
  analysis.py                   early-signal test and its controls
scripts/
  data/                         download the traces and build data/processed
  experiments/                  one script per experiment (policies, seed study, budget controller, early signal,
                                overhead benchmark, minute-level pilot)
  analysis/                     summaries at matched memory, comparison with published results, tables, figures
  tools/record_environment.py   records package versions and hardware of a run
notebooks/kaggle/               notebooks used to run the experiments on Kaggle (GPU: 01, 02; CPU: 03-05)
results/                        result files of the article
tests/                          unit tests
```

## Installation

Python 3.10 or newer; the article used Python 3.13, and the package versions it used are pinned in
`requirements-lock.txt` (see "Environment" below).

```bash
python -m venv .venv                 # with Python 3.13 to match the article
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt # exact package versions of the environment used for the article
pip install -e ".[dev]"              # this package and pytest; the pinned versions are kept
pytest                               # 35 tests
```

Without the lock file, `pip install -e ".[dev]"` installs compatible versions; results of tree-based experiments
may then differ slightly from the published ones. Extracting the Azure 2021 archive requires
[7-Zip](https://www.7-zip.org) (`7z` on the PATH).

## Data

| Trace | Use | Source |
| --- | --- | --- |
| Azure Functions 2021 | main trace (424 functions, 14 days) | [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset) |
| Huawei Cloud Region 2, 2025 | cold-start profiles (31 days); second provider (day 30) | [Huawei Cloud trace release](https://github.com/sir-lab/data-release) |
| Azure Functions 2019 | minute-level pilot only | [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset) |

```bash
python reproduce.py data
```

downloads the traces into `data/raw/` and writes the three files of `data/processed/` that all experiments use
(`azure2021_events.parquet`, `huawei_r2_day30_events.parquet`, `huawei_r2_cold_start_stats.parquet`). The Huawei
request tables are large; the preparation reduces them to unique (function, second) pairs. If Google Drive or Git
LFS refuses a scripted download, the script prints the link to download in a browser and the folder to save it in.

## Reproducing the results

### CPU experiments (local machine)

`reproduce.py` runs every CPU experiment of the article and writes to `reproduction/results/`, so the published
files in `results/` are never overwritten. Finished runs are skipped, so each stage can be interrupted and restarted.

```bash
python reproduce.py env          # record the environment -> reproduction/environment.txt
python reproduce.py verify       # quick check: tree policy, Azure 2021, seed 0, against the published run
python reproduce.py cpu          # all CPU experiments
python reproduce.py summaries    # matched-memory summaries with Holm-corrected tests
python reproduce.py compare      # differences from the published results -> reproduction/comparison.txt
python reproduce.py tables       # LaTeX tables -> reproduction/results/tables/
python reproduce.py figures      # figures -> reproduction/results/figures/
```

Optional stages: `benchmark` (computational cost of the streaming model) and `pilot` (minute-level pilot on Azure
Functions 2019).

### GPU experiments (Kaggle)

The comparison of predictors trains the LSTM and STAWarm on a GPU. On Kaggle, upload this repository as one dataset
and the three files of `data/processed/` as a second dataset, attach both to the notebook, select the accelerator
named in the notebook and run all cells:

| Notebook | Accelerator | Content |
| --- | --- | --- |
| `00_record_environment` | CPU and GPU | package versions and hardware of the Kaggle environment |
| `01_gpu_model_size_selection` | GPU | validation loss of three model sizes |
| `02_gpu_predictor_comparison` | GPU | trees, LSTM, STAWarm and ensemble, three seeds, both traces |
| `03_cpu_baselines_and_robustness` | CPU | baselines, state-aware trees, perfect predictor, conservative simulator |
| `04_cpu_seed_study` | CPU | five seeds with and without the cold-start-duration feature |
| `05_cpu_early_signal_targets_budget` | CPU | early-signal test, prediction targets, budget controller |

Notebooks 03-05 run the same commands as `reproduce.py cpu`. Each notebook moves the published results aside, writes
its own outputs, records its environment, and compares its outputs with the published results.

### Expected differences

Gradient-boosted trees and all simulations are deterministic for a given seed and library version, so tree-based
results reproduce exactly in the recorded environment. GPU training of the neural predictors is not bit-for-bit
deterministic, so their results vary slightly between runs. Different library versions can change results slightly.

## Result files

`results/` holds the outputs used in the article. Folder names follow the pattern
`<experiment>_<trace>[_options]_<tag>_s<seed>`.

| Files | Article section | Content |
| --- | --- | --- |
| `policies_<trace>_baselines_s0..2` | Comparison with keep-alive; distance from a perfect predictor | trees, invocation rate only, state-aware trees, perfect predictor |
| `policies_<trace>_predictors_s0..2` | Comparison of predictors | trees, LSTM, STAWarm, ensemble |
| `policies_<trace>_d2_load_rand_conservative_s0..2` | Robustness | trees under the conservative simulator |
| `policies_azure2021[_x2]_s0..4_{plain,coldtime}`, `seedstudy_azure2021[_x2]` | Comparison with keep-alive; robustness | five-seed study |
| `policies_azure2021[_h10,_x2]_targets_s0..2` | Choice of prediction target | near-term, lead-shifted and union targets |
| `budget_azure2021_s0..2` | Cold-start budgets | 36 controller settings per seed |
| `early_signal/` | Cross-function early signal | lifts against the three controls |
| `summary_*.txt`, `summary_*.csv` | all | matched-memory summaries with Holm-corrected tests |
| `benchmark_overhead.csv` | Computational cost | CPU time of the streaming model |

Each `policies_*` folder contains `policies.csv` (every policy and threshold: cold-start rates overall, by frequency
class and by cold-start duration, added latency and idle memory), `prediction.csv` (AUPRC by frequency class
against both targets) and `summary.txt`. The tables and figures of the article are regenerated with

```bash
python scripts/analysis/make_tables.py results     # -> results/tables/*.tex
python scripts/analysis/make_figures.py            # -> results/figures/*.pdf, *.png
```

Two figures (the distribution of cold-start durations and of the time between invocations) also need the prepared
traces in `data/processed/`.

## Environment

`environment/` contains the recorded environments of the runs (Python version, package versions, CPU, memory, GPU
and the full list of installed packages), and `requirements-lock.txt` pins the package versions of the environment
used for the article. With these versions, the tree-based experiments reproduce the published cold-start rates
exactly on Linux and on Windows.

## Conventions

* Splits are chronological. Anything fitted (models, neighbours, cold-start profiles, thresholds) uses training or
  validation data only.
* Policies are compared at matched idle memory: each curve is read at exactly 0.8x and 1.0x the memory of a
  10-minute keep-alive, without extrapolation.
* A seed changes both the models' randomness and the cold-start profile assigned to each function.

## Licence

MIT; see [LICENSE](LICENSE). The traces are subject to the licences of their providers.
