"""Production stacker on folds 1-17 -> fold 0 with bank + EASE + a given list of Bayesian-FM columns.
usage: full_stack_swap.py TASK tag[,tag...]   (prints the 50/50 MSE)"""
import sys, numpy as np, importlib.util, time
spec = importlib.util.spec_from_file_location('a1v10f', '/data/UCSD/CSE258/A1/v10/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v10f'] = M; spec.loader.exec_module(M)
task = int(sys.argv[1]); tags = [t for t in sys.argv[2].split(',') if t]
C = '/data/UCSD/CSE258/A1/'
m = np.load(C + f'cache_v7/t{task}_meta.npz'); y, fold = m['y'], m['fold']
X = np.load(C + f'cache_v7/t{task}_X.npy', mmap_mode='r')
cols = [np.nan_to_num(np.load(C + f'cache_v8/rating_ease_t{task}_150.npz')['E']).astype(np.float32)]
if task == 3:
    cols.append(np.load(C + 'cache_v8/rating_uease_t3_100.npy').astype(np.float32))
cols += [np.concatenate([np.load(C + f'cache_v10/oof_{t}_t{task}_f{k}.npz')['oof'] for k in range(18)])[:, None].astype(np.float32) for t in tags]
tr, te = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
t0 = time.time()
Xtr = np.hstack([np.asarray(X[tr])] + [c[tr] for c in cols]); Xte = np.hstack([np.asarray(X[te])] + [c[te] for c in cols])
model = M.fit_rating_meta(Xtr, y[tr]); g = model(Xte)
mlp = np.mean(M.mlp_stacker(Xtr, y[tr], [Xte]), axis=0)
mse = lambda p: np.mean((np.clip(p, 1, 5) - y[te]) ** 2)
M.log(f'task {task} [{",".join(tags)}]: GBDT {mse(g):.5f}, MLP {mse(mlp):.5f}, 50/50 {mse(0.5 * g + 0.5 * mlp):.5f}, 40/60 {mse(0.4 * g + 0.6 * mlp):.5f} ({time.time() - t0:.0f}s)')
