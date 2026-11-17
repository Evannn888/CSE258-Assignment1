# Version history

Scores are from the public half of the leaderboard (MSE and MAE: lower is better; accuracy: higher is better).
"–" means the task was not changed or the version was not scored. Every change was first checked on held-out
training data; from v13 on, a change was kept only if it improved on at least two validation folds.

## Leaderboard scores

| Version | Task 1 MSE | Task 2 Acc | Task 3 MSE | Task 4 Acc | Task 5 Acc | Task 6 MAE |
|---|---:|---:|---:|---:|---:|---:|
| starting point | 0.5638 | 0.9381 | 0.7287 | – | – | – |
| v3 | 0.5262 | – | 0.6798 | – | – | – |
| v4 | 0.5260 | 0.9407 | 0.6798 | 0.9375 | 0.9112 | 8,955,691 |
| v5 | 0.5260 | 0.9446 | 0.6798 | 0.9393 | 0.9220 | 8,645,212 |
| v6 | 0.5178 | 0.9490 | 0.6731 | 0.9448 | 0.9272 | 8,620,051 |
| v7 | 0.5168 | 0.9490 | 0.6710 | 0.9450 | 0.9275 | 8,578,307 |
| v8 | 0.5151 | 0.9510 | 0.6665 | 0.9463 | 0.9281 | 8,366,709 |
| v9 | 0.5151 | 0.9519 | 0.6665 | 0.9483 | 0.9296 | 8,366,918 |
| v11 (includes v10) | 0.5099 | 0.9519 | 0.6542 | 0.9492 | 0.9302 | 8,141,390 |
| v12 | 0.5094 | 0.9519 | 0.6542 | 0.9492 | 0.9305 | 8,126,659 |
| v13 | not yet scored | | | | | |

## Changes by version

- **v3** — Ratings: the stacker (Ridge + out-of-fold residual gradient boosting) is trained on about 150k ratings
  instead of 20k; new top-K rating-residual neighbourhood features.
- **v4** — Task 1: base models are fitted out-of-fold and their predictions averaged; an extra stacker on spectral
  coordinates. Other tasks rewritten in a single file.
- **v5** — Tasks 2 and 4: shared co-occurrence and spectral features, four-fold pair training data, the 10-core
  rule; Task 4 adds popularity around the query time and a time-weighted user neighbourhood. Task 5: item
  popularity after the user's last event and pseudo test sets cut at earlier dates. Task 6: last-event
  pseudo-labels (the old model predicted gaps about twice too long).
- **v6** — Ratings: the feature bank is computed out-of-fold for every training rating (18 folds), so the stacker
  learns from about 1.8M / 1.4M rows. Pair tasks: BPR latent models (MF, SVD++-style, FPMC-style) as features.
- **v7** — Ratings: an MLP stacker averaged 50/50 with the gradient-boosted one. Task 4: BERT4Rec. Task 5: SASRec.
  Task 6: time-left scale factors and a reading-lag feature.
- **v8** — EASE everywhere: rating-neighbourhood features for Tasks 1 and 3 (user-user EASE too for Task 3), candidate
  scores for Tasks 2 and 4. Larger pair classifier for Task 5. Task 6: cutoff pseudo-labels and co-reader lags.
  A check on Task-6-like problems built from the Task 1–4 files became the method check for Task 6.
- **v9** — Pair classifiers blend the gradient-boosted model with a GPU MLP. Task 6: relatedness features.
- **v10** — Ratings: Bayesian factorization machine (Gibbs sampling, heteroscedastic noise, ordinal-probit
  variant; time slots for Task 3). Task 4: time-aware BERT4Rec. Task 5: time-aware SASRec. Task 6: second fit
  weighted by the predicted gap, eleven cutoff sets.
- **v11** — The Bayesian factorization machine runs on the GPU (18 folds in one batch: 11 minutes for 18 rank-32 chains instead of about 8 CPU-hours), which
  allowed rank 64 and more chains.
- **v12** — Task 1: a fifth chain with "era" inputs (year a book first appears, median years of book and user).
  Task 5: three SASRec seeds averaged plus a larger SASRec. Task 6: larger trees, stronger L2, re-tuned scale.
- **v13** — Task 6: the final model averages five bootstrap models with the full-data one.

## Tried without gain (selection)

Tuned MF / Funk-SVD, I-AutoRec, Mult-VAE, MLP stackers with learned embeddings, a 5-class MLP stacker, putting the
user's other ratings into the factorization machine, more time inputs for Task 3, LightGCN for Tasks 2 and 4,
predicted-rating features for the pair tasks, sampled-softmax and longer-history SASRec variants, averaging
BERT4Rec seeds, joint inference over a user's test rows, classifier-based distributions for Task 6, external
Goodreads data (not allowed and not used).
