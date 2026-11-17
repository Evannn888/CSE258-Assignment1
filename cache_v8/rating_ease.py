"""EASE item-item weights (implicit, from all known interactions incl. test pairs) as rating-neighbourhood features.
Fold-0 protocol on the cached 18-fold OOF bank: train stacker on folds != 0 with/without the new features."""
import sys, numpy as np, numba as nb, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from ease_big import ease_inplace
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor


@nb.njit(parallel=True, cache=True)
def feats(qu, qi, qfold, uptr, ui, ur, ufold, B, ubar, ibar, nfold):
    n = len(qu)
    out = np.full((n, 5), np.nan, np.float32)
    for r in nb.prange(n):
        u, i, k = qu[r], qi[r], qfold[r]
        imp = 0.0; num = 0.0; num2 = 0.0; den = 0.0; cnt = 0
        for y in range(uptr[u], uptr[u + 1]):
            h = ui[y]
            if h == i:
                continue
            b = B[h, i]
            imp += b
            if ur[y] > 0 and ufold[y] != k:
                num += b * (ur[y] - ubar[u, k] if k < nfold else ur[y] - ubar[u, nfold])
                num2 += b * (ur[y] - ibar[h])
                den += abs(b)
                cnt += 1
        out[r, 0] = imp
        out[r, 4] = np.log1p(cnt)
        if den > 0:
            out[r, 1] = num / (den + 0.05)
            out[r, 2] = num2 / (den + 0.05)
            out[r, 3] = np.log(den + 1e-6)
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
    B = ease_inplace(np.r_[a[:, 0], tu], np.r_[a[:, 1], ti], nu, ni, lam)
    V.log(f'EASE {time.time()-t0:.0f}s')
    # user history arrays (train only; ratings + fold)
    o = np.lexsort((a[:, 2], a[:, 0]))
    uptr = np.searchsorted(a[o, 0], np.arange(nu + 1))
    ui, ur, uf = a[o, 1], a[o, 3].astype(np.float64), rowfold[o]
    # per-user mean rating excluding each fold (and with all ratings in column nfold)
    nf = V.RATING_FOLDS
    pos = a[:, 3] > 0
    s_all = np.bincount(a[pos, 0], weights=a[pos, 3], minlength=nu); c_all = np.bincount(a[pos, 0], minlength=nu).astype(float)
    ubar = np.zeros((nu, nf + 1))
    gm = a[pos, 3].mean()
    for k in range(nf):
        m = pos & (rowfold == k)
        s = s_all - np.bincount(a[m, 0], weights=a[m, 3], minlength=nu); c = c_all - np.bincount(a[m, 0], minlength=nu)
        ubar[:, k] = (s + 3 * gm) / (c + 3)
    ubar[:, nf] = (s_all + 3 * gm) / (c_all + 3)
    ibar = (np.bincount(a[pos, 1], weights=a[pos, 3], minlength=ni) + 10 * gm) / (np.bincount(a[pos, 1], minlength=ni) + 10)
    meta = np.load(f'/data/UCSD/CSE258/A1/cache_v7/t{task}_meta.npz')
    rows = meta['rows']
    E = feats(a[rows, 0], a[rows, 1], rowfold[rows], uptr, ui, ur, uf, B, ubar, ibar, nf)
    Et = feats(tu, ti, np.full(len(tu), nf), uptr, ui, ur, uf, B, ubar, ibar, nf)
    np.savez(f'/data/UCSD/CSE258/A1/cache_v8/rating_ease_t{task}_{lam:g}.npz', E=E, Et=Et)
    V.log('features saved')
    del B
    y, fold = meta['y'], meta['fold']
    X = np.load(f'/data/UCSD/CSE258/A1/cache_v7/t{task}_X.npy', mmap_mode='r')
    tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(tr, min(len(tr), 600000), replace=False))
    Xtr = np.asarray(X[tr]); Xte = np.asarray(X[te])
    kw = dict(max_iter=500, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=100.0, early_stopping=False, random_state=0)
    variants = [('base', Xtr, Xte)] if 'nobase' not in sys.argv else []
    variants.append(('+ease', np.hstack([Xtr, E[tr]]), np.hstack([Xte, E[te]])))
    for tag, A_, B_ in variants:
        t0 = time.time()
        h = HistGradientBoostingRegressor(**kw).fit(A_, y[tr])
        p = np.clip(h.predict(B_), 1, 5)
        V.log(f'task {task} lam {lam:g} {tag}: fold-0 MSE {np.mean((p - y[te]) ** 2):.5f} ({time.time()-t0:.0f}s)')
