import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
from t5_freq import build
grids = {
    'v7 (31 leaves, 500 it, lr .05)': dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0),
    '63 leaves, 800 it': dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=10.0),
    '127 leaves, 1000 it, lr .03': dict(max_iter=1000, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=30.0),
}
train = [x for k in (1, 2, 3, 4) for x in build(k, False)]
val = build(0, False)
X, Y = [], []
for F, ctx, _ in train:
    d, s = V.pair_rows(F[:, 0], F[:, 1], ctx)
    X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
X = np.vstack(X); Y = np.concatenate(Y)
for tag, kw in grids.items():
    t0 = time.time()
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **kw).fit(X, Y)
    accs = []
    for F, ctx, seen in val:
        m = V.pair_margin(clf, F[:, 0], F[:, 1], ctx) + 100 * (seen[:, 1] - seen[:, 0])
        accs.append(np.mean(m > 0))
    print(f'{tag:32s} val {np.round(accs, 4)} mean {np.mean(accs):.4f} {time.time()-t0:.0f}s', flush=True)
