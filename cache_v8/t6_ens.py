import sys, numpy as np, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor
spec = importlib.util.spec_from_file_location('a1v8', '/data/UCSD/CSE258/A1/v8/assignment1.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
def predict(raw, H):
    days = np.where(H > 0, H, np.inf) / M.DAY
    scale = np.select([days < lim for lim, _ in M.T6_SCALE], [f for _, f in M.T6_SCALE])
    p = raw * scale
    return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)
variants = {'v8': {}, 'leaf127': dict(max_leaf_nodes=127), 'leaf31': dict(max_leaf_nodes=31, min_samples_leaf=100),
            'lr02_it2000': dict(learning_rate=0.02, max_iter=2000)}
for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    g_full = M.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    S = M.task6_last_event_sets(g_full, data) + M.task6_cutoff_sets(g_full, data)
    X = np.vstack([s[0] for s in S]); y = np.log1p(np.maximum(np.concatenate([s[1] for s in S]), 0))
    te = data.tests['task_6_test.csv']
    Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
    logs = {}
    for tag, kw in variants.items():
        m = HistGradientBoostingRegressor(**{**M.T6_PARAMS, **kw}).fit(X, y)
        logs[tag] = m.predict(Xt)
        print(f'{name} {tag}: {np.mean(np.abs(predict(np.expm1(logs[tag]), Ht) - truth))/1e6:.4f}M', flush=True)
    ens = np.mean([logs[k] for k in logs], 0)
    print(f'{name} ensemble(all, log-mean): {np.mean(np.abs(predict(np.expm1(ens), Ht) - truth))/1e6:.4f}M', flush=True)
