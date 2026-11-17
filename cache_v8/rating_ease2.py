"""More EASE rating features on the fold-0 protocol (small stacker), baseline = bank + item EASE(150) [+ user EASE(100) for T3].
Variants: (m) extra item-EASE lambdas 50 and 500; (t) time-decayed item-EASE neighbourhood (tau 30 / 365 days)."""
import sys, numpy as np, numba as nb, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from ease_big import ease_inplace
from rating_ease import feats as ease_feats
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor


@nb.njit(parallel=True, cache=True)
def tfeats(qu, qi, qt, qfold, uptr, ui, ut, ur, ufold, B, ubar, nfold, taus):
    n = len(qu)
    out = np.zeros((n, 2 * len(taus)), np.float32)
    for r in nb.prange(n):
        u, i, k = qu[r], qi[r], qfold[r]
        for q in range(len(taus)):
            imp, num, den = 0.0, 0.0, 0.0
            for y in range(uptr[u], uptr[u + 1]):
                h = ui[y]
                if h == i:
                    continue
                w = np.exp(-abs(ut[y] - qt[r]) / (taus[q] * 86400.0))
                b = B[h, i] * w
                imp += b
                if ur[y] > 0 and ufold[y] != k:
                    num += b * (ur[y] - ubar[u, k])
                    den += abs(b)
            out[r, 2 * q] = imp
            out[r, 2 * q + 1] = num / (den + 0.05) if den > 0 else 0.0
    return out


task = int(sys.argv[1])
D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
a, users, items = V.load(D / V.FILES[task])
nu, ni = len(users), len(items)
tests = V.read_test(D / f'task_{task}_test.csv')
tu = np.array([users[r['user']] for r in tests]); ti = np.array([items[r['item']] for r in tests])
folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
rowfold = np.empty(len(a), np.int64)
for k, p in enumerate(folds):
    rowfold[p] = k
o = np.lexsort((a[:, 2], a[:, 0]))
uptr = np.searchsorted(a[o, 0], np.arange(nu + 1))
ui, ut, ur, uf = a[o, 1], a[o, 2].astype(np.float64), a[o, 3].astype(np.float64), rowfold[o]
nf = V.RATING_FOLDS
pos = a[:, 3] > 0
gm = a[pos, 3].mean()
s_all = np.bincount(a[pos, 0], weights=a[pos, 3], minlength=nu); c_all = np.bincount(a[pos, 0], minlength=nu).astype(float)
ubar = np.zeros((nu, nf + 1))
for k in range(nf):
    m = pos & (rowfold == k)
    ubar[:, k] = (s_all - np.bincount(a[m, 0], weights=a[m, 3], minlength=nu) + 3 * gm) / (c_all - np.bincount(a[m, 0], minlength=nu) + 3)
ubar[:, nf] = (s_all + 3 * gm) / (c_all + 3)
ibar = (np.bincount(a[pos, 1], weights=a[pos, 3], minlength=ni) + 10 * gm) / (np.bincount(a[pos, 1], minlength=ni) + 10)
meta = np.load(f'/data/UCSD/CSE258/A1/cache_v7/t{task}_meta.npz')
rows = meta['rows']
extra = {}
for lam in (50.0, 500.0):
    B = ease_inplace(np.r_[a[:, 0], tu], np.r_[a[:, 1], ti], nu, ni, lam)
    extra[f'm{lam:g}'] = ease_feats(a[rows, 0], a[rows, 1], rowfold[rows], uptr, ui, ur, uf, B, ubar, ibar, nf)
    del B
B = ease_inplace(np.r_[a[:, 0], tu], np.r_[a[:, 1], ti], nu, ni, 150.0)
if task == 3:
    extra['t'] = tfeats(a[rows, 0], a[rows, 1], a[rows, 2].astype(np.float64), rowfold[rows], uptr, ui, ut, ur, uf, B, ubar, nf, np.array([30.0, 365.0]))
del B
V.log('features done')
E = np.load(f'/data/UCSD/CSE258/A1/cache_v8/rating_ease_t{task}_150.npz')['E']
base_extra = [np.nan_to_num(E)]
if task == 3:
    base_extra.append(np.load('/data/UCSD/CSE258/A1/cache_v8/rating_uease_t3_100.npy'))
y, fold = meta['y'], meta['fold']
X = np.load(f'/data/UCSD/CSE258/A1/cache_v7/t{task}_X.npy', mmap_mode='r')
te = np.flatnonzero(fold == 0)
Xte_bank = np.asarray(X[te])
kw = dict(max_iter=500, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=100.0, early_stopping=False, random_state=0)
res = {'base': [], 'both': []}
for seed in (1, 2, 3):
    tr = np.flatnonzero(fold != 0)
    tr = np.sort(np.random.default_rng(seed).choice(tr, 600000, replace=False))
    Xtr_bank = np.asarray(X[tr])
    for tag, keys in (('base', []), ('both', ['m50', 'm500', 't'] if task == 3 else ['m50', 'm500'])):
        A_ = np.hstack([Xtr_bank] + [b[tr] for b in base_extra] + [np.nan_to_num(extra[k][tr]) for k in keys])
        B_ = np.hstack([Xte_bank] + [b[te] for b in base_extra] + [np.nan_to_num(extra[k][te]) for k in keys])
        h = HistGradientBoostingRegressor(**kw).fit(A_, y[tr])
        res[tag].append(np.mean((np.clip(h.predict(B_), 1, 5) - y[te]) ** 2))
        V.log(f'task {task} seed {seed} {tag}: {res[tag][-1]:.5f}')
d = np.array(res['both']) - np.array(res['base'])
V.log(f'task {task}: base mean {np.mean(res["base"]):.5f}, both mean {np.mean(res["both"]):.5f}, paired diff {d.mean():+.5f} (per seed {np.round(d, 5)})')
