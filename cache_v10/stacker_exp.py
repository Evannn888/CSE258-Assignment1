"""Rating stacker variants on the production feature set (bank + EASE + Bayesian-FM chains), evaluated on two folds.
usage: stacker_exp.py TASK tags fold[,fold] variant[,variant...]
variants: base, gb_mf05 (max_features 0.5), gb_lr02 (lr 0.02, 3000 rounds), mlp5 (5 seeds), mlp20 (20 epochs)"""
import sys, numpy as np, importlib.util, time
spec = importlib.util.spec_from_file_location('a1v13s', '/data/UCSD/CSE258/A1/v13/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v13s'] = M; spec.loader.exec_module(M)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
task = int(sys.argv[1]); tags = sys.argv[2].split(','); folds = [int(x) for x in sys.argv[3].split(',')]; variants = sys.argv[4].split(',')
C = '/data/UCSD/CSE258/A1/'
m = np.load(C + f'cache_v7/t{task}_meta.npz'); y, fold = m['y'], m['fold']
X = np.load(C + f'cache_v7/t{task}_X.npy', mmap_mode='r')
cols = [np.nan_to_num(np.load(C + f'cache_v8/rating_ease_t{task}_150.npz')['E']).astype(np.float32)]
if task == 3:
    cols.append(np.load(C + 'cache_v8/rating_uease_t3_100.npy').astype(np.float32))
cols += [np.concatenate([np.load(C + f'cache_v10/oof_{t}_t{task}_f{k}.npz')['oof'] for k in range(18)])[:, None].astype(np.float32) for t in tags]
def gbdt(Xtr, ytr, Xte, **kw):
    ridge = M.ChunkedRidge().fit(Xtr, ytr)
    oof = np.empty(len(ytr))
    for a, b in KFold(5, shuffle=True, random_state=M.SEED).split(np.arange(len(ytr))):
        oof[b] = M.ChunkedRidge().fit(Xtr, ytr, rows=a).predict(Xtr, rows=b)
    P = dict(M.META_HGB); P.update(kw)
    h = HistGradientBoostingRegressor(early_stopping=False, random_state=M.SEED, **P).fit(Xtr, ytr - oof)
    return np.clip(ridge.predict(Xte) + h.predict(Xte), 1, 5)
for f in folds:
    tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
    Xtr = np.hstack([np.asarray(X[tr])] + [c[tr] for c in cols]); Xte = np.hstack([np.asarray(X[te])] + [c[te] for c in cols])
    mse = lambda p: np.mean((np.clip(p, 1, 5) - y[te]) ** 2)
    t0 = time.time()
    g0 = gbdt(Xtr, y[tr], Xte); m0 = np.mean(M.mlp_stacker(Xtr, y[tr], [Xte]), 0)
    M.log(f'task {task} fold {f} base: GBDT {mse(g0):.5f} MLP {mse(m0):.5f} 50/50 {mse(0.5 * g0 + 0.5 * m0):.5f} 40/60 {mse(0.4 * g0 + 0.6 * m0):.5f} ({time.time() - t0:.0f}s)')
    for v in variants:
        t0 = time.time()
        g, mm = g0, m0
        if v == 'gb_mf05':
            g = gbdt(Xtr, y[tr], Xte, max_features=0.5)
        elif v == 'gb_lr02':
            g = gbdt(Xtr, y[tr], Xte, learning_rate=0.02, max_iter=3000)
        elif v == 'mlp5':
            mm = np.mean(M.mlp_stacker(Xtr, y[tr], [Xte], seeds=5), 0)
        elif v == 'mlp20':
            mm = np.mean(M.mlp_stacker(Xtr, y[tr], [Xte], epochs=20), 0)
        M.log(f'task {task} fold {f} {v}: GBDT {mse(g):.5f} MLP {mse(mm):.5f} 50/50 {mse(0.5 * g + 0.5 * mm):.5f} (vs base {mse(0.5 * g0 + 0.5 * m0):.5f}) ({time.time() - t0:.0f}s)')
    del Xtr, Xte
