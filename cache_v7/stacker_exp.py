"""Rating stacker experiments on cached 18-fold OOF features: train on folds != 0, evaluate on fold 0."""
import sys, json, time, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
task = int(sys.argv[1]); cfgs = json.loads(sys.argv[2])
C = '/data/UCSD/CSE258/A1/cache_v7/'
X = np.load(C + f't{task}_X.npy', mmap_mode='r'); meta = np.load(C + f't{task}_meta.npz'); y, fold = meta['y'], meta['fold']
tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)
Xtr = np.asarray(X[tr]); ytr = y[tr]; Xte = np.asarray(X[te]); yte = y[te]
ridge = V.ChunkedRidge().fit(Xtr, ytr)
oof = np.empty(len(ytr))
for a, b in KFold(5, shuffle=True, random_state=258).split(np.arange(len(ytr))):
    oof[b] = V.ChunkedRidge().fit(Xtr, ytr, rows=a).predict(Xtr, rows=b)
rte = ridge.predict(Xte)
V.log(f'task {task}: train {len(ytr):,} eval {len(yte):,}; ridge only MSE {np.mean((np.clip(rte, 1, 5) - yte) ** 2):.5f}')
for name, kw in cfgs.items():
    t0 = time.time()
    h = HistGradientBoostingRegressor(early_stopping=False, random_state=258, **kw).fit(Xtr, ytr - oof)
    p = rte + h.predict(Xte)
    V.log(f'task {task} {name}: MSE {np.mean((np.clip(p, 1, 5) - yte) ** 2):.5f}  ({time.time() - t0:.0f}s)')
    np.save(C + f'stk_t{task}_{name}.npy', p)
