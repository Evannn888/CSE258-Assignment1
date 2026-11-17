"""Task 6 harness: stage-2 (weights = stage-1 predicted gap) variants: best scale per time-left bin, IQR weights, capacity."""
import sys, numpy as np, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor
spec = importlib.util.spec_from_file_location('a1v9w', '/data/UCSD/CSE258/A1/v9/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v9w'] = M; spec.loader.exec_module(M)
DAY = M.DAY
def cap(p, H):
    return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)
BINS = [(0, 90), (90, 365), (365, 730), (730, 1e9)]
GRID = [0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]
def bin_of(H):
    days = np.where(H > 0, H, np.inf) / DAY
    return np.select([(days >= a) & (days < b) for a, b in BINS], range(len(BINS)), default=len(BINS) - 1)
def table_scale(H, table):
    return np.asarray(table)[bin_of(H)]
for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    g_full = M.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    sets = M.task6_last_event_sets(g_full, data) + M.task6_cutoff_sets(g_full, data)
    X = np.vstack([s[0] for s in sets]); y = np.concatenate([s[1] for s in sets]); Hs = np.concatenate([s[2] for s in sets])
    ly = np.log1p(np.maximum(y, 0))
    te = data.tests['task_6_test.csv']
    Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
    bt = bin_of(Ht)
    def report(tag, raw):
        out = [f'{np.mean(np.abs(cap(raw * table_scale(Ht, t), Ht) - truth)) / 1e6:.4f}M' for t in ([1, 1, 1, 1], [1.25, 1.2, 1.1, 1.05], [1.1, 1.1, 1.05, 1.03], [1.05] * 4)]
        best = []
        for b in range(len(BINS)):
            m = bt == b
            errs = [np.sum(np.abs(cap(raw[m] * f, Ht[m]) - truth[m])) for f in GRID]
            best.append(GRID[int(np.argmin(errs))])
        tot = np.mean(np.abs(cap(raw * table_scale(Ht, best), Ht) - truth)) / 1e6
        print(name, tag, 'none/v9/mild/1.05:', out, 'best per bin', best, f'-> {tot:.4f}M', 'bin share of error', [round(float(np.sum(np.abs(cap(raw[bt == b], Ht[bt == b]) - truth[bt == b])) / np.sum(np.abs(cap(raw, Ht) - truth))), 3) for b in range(4)], flush=True)
    m1 = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, ly)
    p1 = cap(np.expm1(m1.predict(X)), Hs); r1 = np.expm1(m1.predict(Xt))
    report('v9 unweighted', r1)
    w = np.clip(p1, 86400.0, None); w = w / w.mean()
    m2 = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, ly, sample_weight=w); r2 = np.expm1(m2.predict(Xt))
    report('stage2', r2)
    report('avg(unweighted, stage2) in log', np.exp(0.5 * (np.log1p(r1) + np.log1p(r2))) - 1)
    for tag, kw in (('leaf100', dict(min_samples_leaf=100)), ('leaf400', dict(min_samples_leaf=400)), ('iter2000', dict(max_iter=2000)), ('leaves127', dict(max_leaf_nodes=127))):
        pr = dict(M.T6_PARAMS); pr.update(kw)
        report('stage2 ' + tag, np.expm1(HistGradientBoostingRegressor(**pr).fit(X, ly, sample_weight=w).predict(Xt)))
    # IQR weights: spread of the conditional distribution in seconds
    q = {}
    for a_ in (0.25, 0.75):
        pr = dict(M.T6_PARAMS); pr.update(loss='quantile', quantile=a_, max_iter=400)
        q[a_] = np.expm1(HistGradientBoostingRegressor(**pr).fit(X, ly).predict(X))
    wi = np.clip(cap(q[0.75], Hs) - cap(q[0.25], Hs), 86400.0, None); wi = wi / wi.mean()
    report('IQR weights', np.expm1(HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, ly, sample_weight=wi).predict(Xt)))
