"""Statistical check of the production sampler on fold 0 (compare with the experimental runs)."""
import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import assignment1 as V
import bfm_prod_mod as P
from pathlib import Path
task = int(sys.argv[1]); k = int(sys.argv[2]); iters = int(sys.argv[3]); burn = int(sys.argv[4]); probit = int(sys.argv[5])
time_ = tuple(sys.argv[6].split(',')) if len(sys.argv) > 6 and sys.argv[6] else ()
D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
a, users, items = V.load(D / V.FILES[task]); nu, ni = len(users), len(items)
tests = V.read_test(D / f'task_{task}_test.csv')
qu = np.array([users.get(r['user'], nu) for r in tests], np.int64); qi = np.array([items.get(r['item'], ni) for r in tests], np.int64)
qt = np.array([int(r['timestamp']) for r in tests], np.int64) if task == 3 else np.zeros(len(tests), np.int64)
test_rows = np.column_stack([qu, qi, qt, np.zeros(len(qu), np.int64)])
folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
part = folds[0]; keep = np.ones(len(a), bool); keep[part] = False
t0 = time.time()
m, sd = P.bfm_rating_predictions(a[keep], np.vstack([a[part], test_rows]), nu, ni, 1, k=k, iters=iters, burn=burn, probit=bool(probit), time=time_)
ok = a[part, 3] > 0
print(f'prod task {task} K{k} iters {iters} probit {probit} time {time_}: fold-0 val MSE %.5f (%.0fs)' % (np.mean((np.clip(m[:len(part)][ok], 1, 5) - a[part, 3][ok]) ** 2), time.time() - t0), flush=True)
