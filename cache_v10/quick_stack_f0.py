"""Quick screening on fold-0 rows only: 2-fold CV (x3 splits) of a small GBDT stacker on bank+EASE with extra columns.
usage: quick_stack_f0.py TASK variant ...   variant = '+'-joined tags (fold-0 npz files with va/pred[/oof_sd]); 'base' = none.
A tag ending in ':s' also adds the posterior sd column."""
import sys, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
task = int(sys.argv[1]); variants = sys.argv[2:]
C = '/data/UCSD/CSE258/A1/'
m = np.load(C + f'cache_v7/t{task}_meta.npz'); y, fold, rows = m['y'], m['fold'], m['rows']
X = np.load(C + f'cache_v7/t{task}_X.npy', mmap_mode='r')
te = np.flatnonzero(fold == 0); Xb = np.asarray(X[te]); yt = y[te]; rt = rows[te]
base = [Xb, np.nan_to_num(np.load(C + f'cache_v8/rating_ease_t{task}_150.npz')['E'][te])]
if task == 3:
    base.append(np.load(C + 'cache_v8/rating_uease_t3_100.npy')[te])
cache = {}
def col(tag):
    if tag not in cache:
        name, sd = (tag[:-2], True) if tag.endswith(':s') else (tag, False)
        d = np.load(C + f'cache_v10/oof_{name}_t{task}_f0.npz'); va = d['va']
        lut = np.full(int(max(va.max(), rt.max())) + 1, -1); lut[va] = np.arange(len(va)); idx = lut[rt]; assert (idx >= 0).all()
        p = d['oof'] if d['oof'].ndim == 2 else d['pred'][:len(va)]
        p = p[idx] if p.ndim == 2 else p[idx][:, None]
        if sd:
            p = np.column_stack([p, d['oof_sd'][idx]])
        if p.shape[1] == 1 and p.mean() > 2:
            print(f'  {tag}: alone MSE {np.mean((np.clip(p[:, 0], 1, 5) - yt) ** 2):.5f}', flush=True)
        cache[tag] = p
    return cache[tag]
kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=50.0, early_stopping=False, random_state=0)
res = {}
for v in variants:
    Z = np.hstack(base + [col(t) for t in v.split('+') if t != 'base']); out = []
    for seed in range(3):
        half = np.random.default_rng(seed).permutation(len(te)) % 2; pr = np.zeros(len(te))
        for h in (0, 1):
            g = HistGradientBoostingRegressor(**kw).fit(Z[half != h], yt[half != h]); pr[half == h] = np.clip(g.predict(Z[half == h]), 1, 5)
        out.append(np.mean((pr - yt) ** 2))
    res[v] = np.mean(out)
    print(f'{v}: {res[v]:.5f}  ({res[v] - res[variants[0]]:+.5f} vs {variants[0]})  per split {np.round(out, 5)}', flush=True)
