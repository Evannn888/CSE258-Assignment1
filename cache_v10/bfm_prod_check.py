"""Check that the production sampler reproduces the experimental one (bfm4) for the same seed."""
import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import assignment1 as V
import numba as nb
from pathlib import Path
import bfm_prod_mod as P
ns = {'bfm_rating_predictions': P.bfm_rating_predictions}
task = int(sys.argv[1]); time_ = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 and sys.argv[2] else ()
probit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
a, users, items = V.load(D / V.FILES[task]); nu, ni = len(users), len(items)
tests = V.read_test(D / f'task_{task}_test.csv')
qu = np.array([users.get(r['user'], nu) for r in tests], np.int64); qi = np.array([items.get(r['item'], ni) for r in tests], np.int64)
qt = np.array([int(r['timestamp']) for r in tests], np.int64) if task == 3 else np.zeros(len(tests), np.int64)
test_rows = np.column_stack([qu, qi, qt, np.zeros(len(qu), np.int64)])
folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
part = folds[0]; keep = np.ones(len(a), bool); keep[part] = False
m, sd = ns['bfm_rating_predictions'](a[keep], np.vstack([a[part], test_rows]), nu, ni, 1, k=4, iters=12, burn=4, probit=bool(probit), time=time_, spare=0)
ok = a[part, 3] > 0
print('prod: fold-0 val MSE %.6f' % np.mean((np.clip(m[:len(part)][ok], 1, 5) - a[part, 3][ok]) ** 2), 'test mean %.6f' % m[len(part):].mean())
np.save('/data/UCSD/CSE258/A1/cache_v10/prod_check.npy', np.r_[m[:len(part)][ok], m[len(part):]])
