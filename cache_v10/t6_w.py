"""Task 6 on the harness (files with known final events): does weighting the median regression towards seconds help?
Variants: v9 (unweighted, log target), stage-2 weights = predicted gap^p, weights = time-left^p, raw-seconds target."""
import sys, numpy as np, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor
spec = importlib.util.spec_from_file_location('a1v9w', '/data/UCSD/CSE258/A1/v9/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v9w'] = M; spec.loader.exec_module(M)
DAY = M.DAY

def cap(p, H):
    return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)

def scaled(p, H, table=M.T6_SCALE):
    days = np.where(H > 0, H, np.inf) / DAY
    return p * np.select([days < lim for lim, _ in table], [f for _, f in table])

for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    g_full = M.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    sets = M.task6_last_event_sets(g_full, data) + M.task6_cutoff_sets(g_full, data)
    X = np.vstack([s[0] for s in sets]); y = np.concatenate([s[1] for s in sets]); Hs = np.concatenate([s[2] for s in sets])
    ly = np.log1p(np.maximum(y, 0))
    te = data.tests['task_6_test.csv']
    Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
    def report(tag, raw):
        res = {'noscale': cap(raw, Ht), 'v9scale': cap(scaled(raw, Ht), Ht)}
        for f in (0.9, 1.1, 1.2):
            res[f'x{f}'] = cap(raw * f, Ht)
        print(name, tag, {k: f'{np.mean(np.abs(v - truth)) / 1e6:.4f}M' for k, v in res.items()}, flush=True)
    t0 = time.time()
    m1 = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, ly)
    p1 = np.expm1(m1.predict(X)); raw1 = np.expm1(m1.predict(Xt))
    report('v9 (unweighted log)', raw1)
    for p in (0.5, 1.0):
        w = np.clip(p1, 3600.0, None) ** p
        m = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, ly, sample_weight=w / w.mean())
        report(f'stage-2 weight pred^{p}', np.expm1(m.predict(Xt)))
    for p in (0.5, 1.0):
        w = np.clip(np.where(Hs > 0, Hs, np.median(Hs)), 86400.0, None) ** p
        m = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, ly, sample_weight=w / w.mean())
        report(f'weight H^{p}', np.expm1(m.predict(Xt)))
    kw = dict(M.T6_PARAMS)
    m = HistGradientBoostingRegressor(**kw).fit(X, np.maximum(y, 0))
    report('raw seconds target', m.predict(Xt))
    print(name, f'{time.time() - t0:.0f}s', flush=True)
