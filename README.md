# CSE258 Assignment 1 (Goodreads recommender tasks)

All six tasks are solved in a single file, `assignment1.py`, which regenerates the six submitted prediction
files. `writeup.txt` describes the methods task by task; `HISTORY.md` lists how the solution evolved and the
leaderboard score of each submitted version.

## Files

| Path | Contents |
|---|---|
| `assignment1.py` | Final code (version 13). Entry points: `run_rating_task` (Tasks 1, 3), `run_pair_task` (Tasks 2, 4), `run_task5`, `run_task6` |
| `writeup.txt` | Method description (the submitted write-up) |
| `task_1_predictions.csv` … `task_6_predictions.csv` | Submitted predictions |
| `validation_metrics.json`, `training.log` | Validation numbers and the logs of the runs that produced the predictions |
| `validate_submission.py` | Checks row order, ids and value ranges of the prediction files; writes `submission_manifest.json` |
| `requirements.txt` | Python packages (PyTorch with CUDA is needed) |
| `v5/` … `v13/` | Earlier versions of `assignment1.py`, their write-ups and validation numbers |
| `cache_v7/`, `cache_v8/`, `cache_v10/`, `experiments/` | Experiment scripts (their cached inputs and outputs are not in the repository) |
| `HISTORY.md` | Version history and leaderboard scores |

The course data (`assignment1_2026/`), cached features, Bayesian-FM chains and model outputs are not included.

## Running

```bash
pip install -r requirements.txt   # torch: CUDA build, e.g. --index-url https://download.pytorch.org/whl/cu130
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
    python assignment1.py --data-dir assignment1_2026 --output-dir out --bfm-cache chains
```

- `--tasks 1 2 3 4 5 6` selects tasks (default: all). Run one large task at a time: Task 2 and Task 3 peak at about
  17–20 GB of RAM.
- An NVIDIA GPU is required (Bayesian factorization machine, MLP stackers, BERT4Rec, SASRec). A full run takes
  several hours, most of it in the Bayesian-FM chains of Tasks 1 and 3; `--bfm-cache DIR` stores the chains and
  reuses them on later runs.
- GPU sampling is not bit-reproducible (atomic additions), so a re-run reproduces the predictions statistically,
  not digit by digit. All other parts use fixed seeds.

## Method in brief

- **Tasks 1 and 3 (ratings):** an 18-fold out-of-fold feature bank (biases, MF, SVD++, neighbourhoods, EASE
  features) plus several Bayesian factorization machines sampled on the GPU, combined by a Ridge + gradient-boosted
  stacker averaged with an MLP stacker.
- **Tasks 2 and 4 (which of two books):** co-occurrence, spectral, EASE and BPR latent-model features (Task 4 also
  time-aware features and a time-aware BERT4Rec), decided by an antisymmetric GBDT + MLP pair classifier.
- **Task 5 (next book):** the same pair classifier with order-aware features and time-aware SASRec, trained on
  pseudo test sets that hold out each user's last event.
- **Task 6 (time to the final interaction):** median regression of log(1 + gap) on pseudo-labels that mimic the
  test (last-event and cutoff sets), weighted by the predicted gap and bagged.

Details and validation numbers are in `writeup.txt`.

## Leaderboard (public half of the test set)

| Version | Task 1 MSE | Task 2 Acc | Task 3 MSE | Task 4 Acc | Task 5 Acc | Task 6 MAE |
|---|---:|---:|---:|---:|---:|---:|
| v12 (latest scored) | 0.50939 | 0.9519 | 0.65420 | 0.9492 | 0.93052 | 8,126,659 |

The repository contains v13, which differs from v12 only in Task 6 (bagging; not yet scored when this was written).
