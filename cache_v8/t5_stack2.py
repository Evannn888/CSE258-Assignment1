import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
z = np.load('cache_v7/t5_sets.npz'); s_old = np.load('cache_v7/t5_sasrec.npz'); s_new = np.load('cache_v8/t5_sasce.npz')
def build(k, mode):
    F, ctx, seen, L = z[f'F{k}'], z[f'ctx{k}'], z[f'seen{k}'], z[f'L{k}']
    parts = [F, L]
    if mode in ('old', 'both'):
        parts.append(s_old[f'S{k}'][..., None])
    if mode in ('new', 'both'):
        parts.append(s_new[f'S{k}'][..., None])
    base = np.concatenate(parts, -1)
    return [(base[:, [0, d]], ctx, seen[:, [0, d]]) for d in (1, 2, 3)]
small = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
big = dict(max_iter=1000, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=30.0)
for mode, kw, tag in (('new', small, 'small'), ('both', small, 'small'), ('both', big, 'big')):
    t0 = time.time()
    X, Y = [], []
    for k in (1, 2, 3, 4):
        for F, ctx, _ in build(k, mode):
            d, s = V.pair_rows(F[:, 0], F[:, 1], ctx)
            X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **kw).fit(np.vstack(X), np.concatenate(Y))
    accs = []
    for F, ctx, seen in build(0, mode):
        m = V.pair_margin(clf, F[:, 0], F[:, 1], ctx) + 100 * (seen[:, 1] - seen[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T5 sasrec={mode} clf={tag}: {np.round(accs, 4)} mean {np.mean(accs):.4f} {time.time()-t0:.0f}s', flush=True)
