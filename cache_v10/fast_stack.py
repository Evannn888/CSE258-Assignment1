"""Fast-stacker check (same protocol as v8/v9): GBDT on 600k rows of folds != 0, scored on fold 0, 1-3 subsamples.
usage: fast_stack.py TASK NSEEDS tagA+tagB tagC ...   (each argument = one variant: '+'-joined OOF tags added to the base)"""
import sys, numpy as np, time
from sklearn.ensemble import HistGradientBoostingRegressor
task, nseed, variants = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
C = '/data/UCSD/CSE258/A1/'
m = np.load(C + f'cache_v7/t{task}_meta.npz'); y, fold = m['y'], m['fold']
X = np.load(C + f'cache_v7/t{task}_X.npy', mmap_mode='r')
base = [np.nan_to_num(np.load(C + f'cache_v8/rating_ease_t{task}_150.npz')['E'])]
if task == 3:
    base.append(np.load(C + 'cache_v8/rating_uease_t3_100.npy'))
def oof(tag):
    cols = []
    for k in range(18):
        d = np.load(C + f'cache_v10/oof_{tag}_t{task}_f{k}.npz')
        c = [d['oof']] + ([d['oof_sd']] if 'oof_sd' in d.files and tag.endswith('S') else [])
        cols.append(np.column_stack(c))
    return np.vstack(cols)
tags = sorted({t for v in variants for t in v.split('+') if t and t != 'base'})
F = {}
for t in tags:
    name = t[:-1] if t.endswith('S') else t
    cols = []
    for k in range(18):
        d = np.load(C + f'cache_v10/oof_{name}_t{task}_f{k}.npz')
        cols.append(np.column_stack([d['oof'], d['oof_sd']]) if t.endswith('S') else d['oof'].reshape(len(d['oof']), -1))
    F[t] = np.vstack(cols)
    assert len(F[t]) == len(y), (t, len(F[t]), len(y))
    te_ = fold == 0
    if F[t][:, 0].mean() > 2:
        print(f'{t}: alone fold-0 MSE {np.mean((np.clip(F[t][te_, 0], 1, 5) - y[te_]) ** 2):.5f}, all-fold MSE {np.mean((np.clip(F[t][:, 0], 1, 5) - y) ** 2):.5f}', flush=True)
te = np.flatnonzero(fold == 0); Xte = np.asarray(X[te])
kw = dict(max_iter=500, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=100.0, early_stopping=False, random_state=0)
res = {v: [] for v in variants}
for seed in range(1, nseed + 1):
    tr = np.sort(np.random.default_rng(seed).choice(np.flatnonzero(fold != 0), 600000, replace=False))
    Xtr = np.asarray(X[tr])
    for v in variants:
        ex = [F[t] for t in v.split('+') if t and t != 'base']
        A = np.hstack([Xtr] + [b[tr] for b in base] + [e[tr] for e in ex]); B = np.hstack([Xte] + [b[te] for b in base] + [e[te] for e in ex])
        t0 = time.time()
        h = HistGradientBoostingRegressor(**kw).fit(A, y[tr])
        res[v].append(np.mean((np.clip(h.predict(B), 1, 5) - y[te]) ** 2))
        print(f'task {task} seed {seed} {v}: {res[v][-1]:.5f} ({time.time() - t0:.0f}s)', flush=True)
for v in variants:
    print(f'{v}: mean {np.mean(res[v]):.5f}  diff vs first {np.mean(res[v]) - np.mean(res[variants[0]]):+.5f}')
