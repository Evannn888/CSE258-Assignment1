import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
import importlib.util
spec = importlib.util.spec_from_file_location('pm', '/data/UCSD/CSE258/A1/cache_v8/pair_mlp_lib.py')
z = np.load('cache_v7/t5_sets.npz'); s_old = np.load('cache_v7/t5_sasrec.npz')
def build(k):
    base = np.concatenate([z[f'F{k}'], z[f'L{k}'], s_old[f'S{k}'][..., None]], -1)
    return [(base[:, [0, d]], z[f'ctx{k}'], z[f'seen{k}'][:, [0, d]]) for d in (1, 2, 3)]
train = [x for k in (1, 2, 3, 4) for x in build(k)]
X, Y = [], []
for F, ctx, _ in train:
    d, s = V.pair_rows(F[:, 0], F[:, 1], ctx)
    X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
X = np.vstack(X).astype(np.float32); Y = np.concatenate(Y).astype(np.float32)
val = build(0)
t0 = time.time()
big = dict(max_iter=1000, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=30.0)
clf = HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **big).fit(X, Y)
V.log(f'gbdt fitted {time.time()-t0:.0f}s')
sys.argv = ['x', '5']
from pair_mlp_lib import mlp_fit_predict
for di, (F, ctx, seen) in enumerate(val):
    d, sy = V.pair_rows(F[:, 0], F[:, 1], ctx)
    Xa, Xb = np.hstack([d, sy]).astype(np.float32), np.hstack([-d, sy]).astype(np.float32)
    mg = 0.5 * (clf.decision_function(Xa) - clf.decision_function(Xb))
    mm = mlp_fit_predict(X, Y, Xa, Xb, epochs=8) if di == 0 else None
    rule = 100 * (seen[:, 1] - seen[:, 0])
    acc = lambda m: np.mean(m + rule > 0)
    if mm is not None:
        print(f'T5 draw {di+1}: gbdt {acc(mg):.4f} mlp {acc(mm):.4f} blend0.3 {acc(0.7*mg+0.3*mm):.4f} blend0.5 {acc(0.5*mg+0.5*mm):.4f}', flush=True)
