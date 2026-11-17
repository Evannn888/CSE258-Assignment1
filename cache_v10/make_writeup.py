"""v10/writeup.txt from v9's, with the v10 changes.  Numbers come from cache_v10/summary_numbers.json."""
import json
from pathlib import Path
root = Path('/data/UCSD/CSE258/A1')
N = json.loads((root / 'cache_v10/summary_numbers.json').read_text())
s = (root / 'v9/writeup.txt').read_text()
s = s.replace('CSE258 Assignment 1 - write-up (v9)', 'CSE258 Assignment 1 - write-up (v10)')
old = '\nLatent factor models (Tasks 2, 4, 5)\n'
new = f'''
Bayesian factorization machine (Tasks 1 and 3; new in v10)
The matrix factorisations in the feature bank (SGD point estimates) barely beat the biases on these sparse
data (median 24 ratings per user, 18 per book): fold-0 MSE 0.589 against 0.5885 for biases alone. A
factorization machine sampled by Gibbs sampling (Rendle 2012, the libFM MCMC solver, written here in numba)
averages predictions over the posterior instead, and every prior mean and precision is itself sampled, so no
regularisation has to be tuned. Features ("SVD++ flipped", Rendle, Zhang & Koren 2019): user id, book id, the
bag of every book the user interacted with and the bag of every user of the book (existence only: training
rows of any rating plus the test pairs; weights |bag|^-1/2); Task 3 adds the calendar month, the user's day
("session", bias only) and the user in that year. Two changes from plain libFM mattered: user- and
book-specific noise precisions (Student-t-like heteroscedastic noise, prior strength 30; fold-0 MSE 0.5617 ->
0.5451 at rank 8) and an ordinal-probit likelihood for the 1-5 scale as a second model. Chains: rank 32,
500 sweeps (100 burn-in), Gaussian; rank 16, 400 sweeps, ordinal probit; Task 3 also rank 32 ordinal probit
with all time slots. Every chain is run out-of-fold on the same 18 folds as the bank and its posterior mean
is a stacking feature. Alone (out-of-fold over all ratings): Task 1 {N['t1_c0']:.4f} / {N['t1_c1']:.4f}, Task 3
{N['t3_c0']:.4f} / {N['t3_c1']:.4f} / {N['t3_c2']:.4f}, against 0.580 for the best single column of the old bank (Task 1).
Higher rank still helps alone (Task 1 rank 64: 0.5311 vs 0.5338 on fold 0) but costs about an hour per chain.

Latent factor models (Tasks 2, 4, 5)
'''
assert old in s; s = s.replace(old, new, 1)
old = 'rating) added to the blend gave -0.0002 on both tasks (not adopted).\n'
new = f'''rating) added to the blend gave -0.0002 on both tasks (not adopted).
New in v10: the Bayesian FM predictions (above) are added to the bank. Fast stacker on fold 0 (the v8/v9
protocol, 600k rows, two subsamples): Task 1 0.5252 -> 0.5178 (rank-32 chain) -> 0.5174 (both chains); Task 3
0.6608 -> 0.6525 (rank-32 chain). Full production stacker (Ridge + GBDT and MLP, trained on folds 1-17, scored
on fold 0): Task 1 {N['t1_full_old']:.4f} -> {N['t1_full_new']:.4f}; Task 3 {N['t3_full_old']:.4f} -> {N['t3_full_new']:.4f}.
Tried without gain: rating-weighted bags of the user's other ratings inside the FM (with ids it memorises the
row's own rating; without ids it is far worse, 0.71-0.80), user/book attribute slots (activity level, first
year) and an explicit-residual EASE (ridge regression of a rating residual on the user's other residuals:
-0.00003 in the stack).
'''
assert old in s; s = s.replace(old, new, 1)
old = 'Tried without gain: EASE on same-day or same-month "sessions"'
new = f'''New in v10 (Task 4): the BERT4Rec model knows when things happened - every token also gets an embedding of
the gap to the previous and to the next event in the window and of its calendar month, and the masked query
token carries the query time. Alone 89.8% -> 91.3% (fold 0); in the classifier (4 folds, small GBDT) 0.9437 ->
0.9451; full run, fold 0: {N['t4_val']:.4f} (v9 0.9450). A larger model (192 units, 3 blocks) did not help. The
predicted rating of each candidate (Bayesian FM trained on the fold's ratings) and its posterior sd added
+0.04 points for Task 2 and nothing for Task 4 (not adopted).
Tried without gain: EASE on same-day or same-month "sessions"'''
assert old in s; s = s.replace(old, new, 1)
old = 'Tried without gain: SASRec trained with a sampled softmax'
new = f'''New in v10: SASRec
tokens also carry the gap to the previous event and the calendar month (training-file timestamps): alone
87.2% -> 88.0%, in the classifier (cached sets, small GBDT) 0.9319 -> 0.9323; full run: {N['t5_val']:.4f}
(v9 0.9349). Not used in --task5-order-only mode.
Tried without gain: SASRec trained with a sampled softmax'''
assert old in s; s = s.replace(old, new, 1)
old = '''An
absolute-error gradient-boosted model on log(1+gap) predicts the median; predictions are scaled up by
1.25 / 1.20 / 1.10 / 1.05 (time left below 90 days / 1 year / 2 years / more) and capped by the time left.'''
new = '''An
absolute-error gradient-boosted model on log(1+gap) predicts the median. New in v10: the model is fitted twice
and the second fit weights every row by the gap the first fit predicts for it - an error in log space costs
seconds in proportion to the gap, and the metric is in seconds; eleven cutoff sets (14 days to 3 years) instead
of five; leaves of at least 400 rows. Predictions are scaled up by 1.10 / 1.10 / 1.05 / 1.03 (time left below
90 days / 1 year / 2 years / more; v9 1.25 / 1.20 / 1.10 / 1.05) and capped by the time left.'''
assert old in s; s = s.replace(old, new, 1)
old = 'v7 7.683M / 6.375M seconds MAE -> v8 7.518M / 6.237M (-2.2% on both) -> v9 7.500M / 6.222M; gap-sequence,'
new = ('v7 7.683M / 6.375M seconds MAE -> v8 7.518M / 6.237M (-2.2% on both) -> v9 7.500M / 6.222M -> v10\n'
       f"{N['t6_h1']:.4f}M / {N['t6_h2']:.4f}M (weighted second fit, milder scale and larger leaves -1.3% / -2.1%; eleven\n"
       "cutoff sets a further -0.2% / -0.2%); gap-sequence,")
assert old in s; s = s.replace(old, new, 1)
old = 'Validation MAE on last-training-event targets: 6.98M seconds (v7 7.07M, v8 7.00M).'
new = ('''Also tried in v10 without gain: the median of a distribution assembled from 19 classifiers P(gap > t),
those probabilities as extra features, features from other test users who will read the same book, and
weights from the predicted spread instead of the predicted gap.
Validation MAE on last-training-event targets: 7.05M seconds (v9 6.98M; this target is not the test's -
further events still follow it - and it moved the other way from the check on true final gaps above).''')
assert old in s; s = s.replace(old, new, 1)
(root / 'v10/writeup.txt').write_text(s)
print('v10 writeup written:', len(s.splitlines()), 'lines')
