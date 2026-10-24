# Recorded environments

| File | Machine | Used for |
| --- | --- | --- |
| `kaggle_cpu.txt` | Kaggle notebook, 4 vCPUs, 31 GB, no accelerator | all CPU experiments; full `pip freeze` |
| `kaggle_gpu.txt` | Kaggle notebook, 2 x NVIDIA T4 (one used per run) | training and evaluation of the neural predictors |

Each file is written by `scripts/tools/record_environment.py`. `requirements-lock.txt` in the repository root pins the
versions of the packages the experiments depend on. With these versions, the tree-based experiments reproduce the
published cold-start rates exactly, including on Windows (`python reproduce.py verify`).
