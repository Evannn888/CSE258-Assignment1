"""User-user EASE as rating-neighbourhood features (fold-0 protocol, small stacker)."""
import sys, numpy as np, numba as nb, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from ease_big import ease_inplace
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor


@nb.njit(parallel=True, cache=True)
def ufeats(qu, qi, qfold, iptr, iu, ir, ifold, W, ubar, nfold):
    n = len(qu)
    out = np.zeros((n, 4), np.float32)
    nU = W.shape[0]
    for r in nb.prange(n):
        u, i, k = qu[r], qi[r], qfold[r]
        if u >= nU:
            continue
        imp, num, den, cnt = 0.0, 0.0, 0.0, 0
        for y in range(iptr[i], iptr[i + 1]):
            v = iu[y]
            if v == u:
                continue
            w = W[v, u]
            imp += w
            if ir[y] > 0 and ifold[y] != k:
                num += w * (ir[y] - ubar[v, k])
                den += abs(w)
                cnt += 1
        out[r, 0] = imp
        out[r, 1] = num / (den + 0.05)
        out[r, 2] = np.log(den + 1e-6)
        out[r, 3] = np.log1p(cnt)
    return out


if __name__ == '__main__':
    task = int(sys.argv[1]); lam = float(sys.argv[2])
    D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
    a, users, items = V.load(D / V.FILES[task])
    nu, ni = len(users), len(items)
    tests = V.read_test(D / f'task_{task}_test.csv')
    tu = np.array([users[r['user']] for r in tests]); ti = np.array([items[r['item']] for r in tests])
    folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
    rowfold = np.empty(len(a), np.int64)
    for k, p in enumerate(folds):
        rowfold[p] = k
    t0 = time.time()
    W = ease_inplace(np.r_[a[:, 1], ti], np.r_[a[:, 0], tu], ni, nu, lam)     # transpose: items play "users"
    V.log(f'user-EASE {time.time()-t0:.0f}s')
    o = np.lexsort((a[:, 0], a[:, 1]))
    iptr = np.searchsorted(a[o, 1], np.arange(ni + 1))
    nf = V.RATING_FOLDS
    pos = a[:, 3] > 0
    gm = a[pos, 3].mean()
    s_all = np.bincount(a[pos, 0], weights=a[pos, 3], minlength=nu); c_all = np.bincount(a[pos, 0], minlength=nu).astype(float)
    ubar = np.zeros((nu, nf + 1))
    for k in range(nf):
        m = pos & (rowfold == k)
        ubar[:, k] = (s_all - np.bincount(a[m, 0], weights=a[m, 3], minlength=nu) + 3 * gm) / (c_all - np.bincount(a[m, 0], minlength=nu) + 3)
    ubar[:, nf] = (s_all + 3 * gm) / (c_all + 3)
    meta = np.load(f'/data/UCSD/CSE258/A1/cache_v7/t{task}_meta.npz')
    rows = meta['rows']
    U = ufeats(a[rows, 0], a[rows, 1], rowfold[rows], iptr, a[o, 0], a[o, 3].astype(np.float64), rowfold[o], W, ubar, nf)
    del W
    np.save(f'/data/UCSD/CSE258/A1/cache_v8/rating_uease_t{task}_{lam:g}.npy', U)
    E = np.load(f'/data/UCSD/CSE258/A1/cache_v8/rating_ease_t{task}_150.npz')['E']
    y, fold = meta['y'], meta['fold']
    X = np.load(f'/data/UCSD/CSE258/A1/cache_v7/t{task}_X.npy', mmap_mode='r')
    tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)
    tr = np.sort(np.random.default_rng(0).choice(tr, min(len(tr), 600000), replace=False))
    Xtr = np.hstack([np.asarray(X[tr]), E[tr]]); Xte = np.hstack([np.asarray(X[te]), E[te]])
    kw = dict(max_iter=500, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=100.0, early_stopping=False, random_state=0)
    h = HistGradientBoostingRegressor(**kw).fit(np.hstack([Xtr, U[tr]]), y[tr])
    p = np.clip(h.predict(np.hstack([Xte, U[te]])), 1, 5)
    V.log(f'task {task} item-EASE(150) + user-EASE({lam:g}): fold-0 MSE {np.mean((p - y[te]) ** 2):.5f}  (item-EASE only: t1 0.52472, t3 0.66113)')
