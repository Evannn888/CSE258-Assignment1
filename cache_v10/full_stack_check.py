"""Full-protocol check of the v10 rating stacker: train the production stacker (Ridge + residual GBDT, MLP, 50/50) on
the bank rows of folds 1-17 and score fold 0, without and with the Bayesian-FM columns.
usage: full_stack_check.py TASK tag[,tag...]"""
import sys, numpy as np, importlib.util, time
spec = importlib.util.spec_from_file_location('a1v10f', '/data/UCSD/CSE258/A1/v10/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v10f'] = M; spec.loader.exec_module(M)
task = int(sys.argv[1]); tags = sys.argv[2].split(',')
base_tags = sys.argv[4].split(',') if len(sys.argv) > 4 else []
C = '/data/UCSD/CSE258/A1/'
m = np.load(C + f'cache_v7/t{task}_meta.npz'); y, fold = m['y'], m['fold']
X = np.load(C + f'cache_v7/t{task}_X.npy', mmap_mode='r')
cols = [np.nan_to_num(np.load(C + f'cache_v8/rating_ease_t{task}_150.npz')['E']).astype(np.float32)]
if task == 3:
    cols.append(np.load(C + 'cache_v8/rating_uease_t3_100.npy').astype(np.float32))
load = lambda t: np.concatenate([np.load(C + f'cache_v10/oof_{t}_t{task}_f{k}.npz')['oof'] for k in range(18)])[:, None].astype(np.float32)
extra = [load(t) for t in tags]
cols += [load(t) for t in base_tags]
tr, te = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
res = {}
which = sys.argv[3] if len(sys.argv) > 3 else 'both'
variants = [('base' + ('+' + '+'.join(base_tags) if base_tags else ''), []), ('+ ' + '+'.join(tags), extra)]
variants = variants if which == 'both' else variants[:1] if which == 'base' else variants[1:]
for name, ex in variants:
    t0 = time.time()
    Xtr = np.hstack([np.asarray(X[tr])] + [c[tr] for c in cols] + [e[tr] for e in ex])
    Xte = np.hstack([np.asarray(X[te])] + [c[te] for c in cols] + [e[te] for e in ex])
    model = M.fit_rating_meta(Xtr, y[tr])
    g = model(Xte)
    mlp = np.mean(M.mlp_stacker(Xtr, y[tr], [Xte]), axis=0)
    blend = np.clip(0.5 * g + 0.5 * mlp, 1, 5)
    mse = lambda p: np.mean((np.clip(p, 1, 5) - y[te]) ** 2)
    res[name] = mse(blend)
    M.log(f'task {task} {name}: GBDT {mse(g):.5f}, MLP {mse(mlp):.5f}, 50/50 {mse(blend):.5f} ({time.time() - t0:.0f}s)')
    del Xtr, Xte
k = list(res)
if len(k) == 2:
    M.log(f'task {task}: full-stacker change {res[k[1]] - res[k[0]]:+.5f}')
