"""Task 6 harness: weights = (predicted gap)^p for the log-target median regression; more p values, a third stage, caps."""
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
        for f in (1.05, 1.1, 1.15):
            res[f'x{f}'] = cap(raw * f, Ht)
        print(name, tag, {k: f'{np.mean(np.abs(v - truth)) / 1e6:.4f}M' for k, v in res.items()}, flush=True)
    m1 = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, ly)
    p1 = cap(np.expm1(m1.predict(X)), Hs)
    report('v9', np.expm1(m1.predict(Xt)))
    def stage(prev, p, floor=3600.0, tag=''):
        w = np.clip(prev, floor, None) ** p
        m = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, ly, sample_weight=w / w.mean())
        report(f'{tag} p={p} floor={floor / 3600:.0f}h', np.expm1(m.predict(Xt)))
        return cap(np.expm1(m.predict(X)), Hs)
    p2 = stage(p1, 1.0, tag='stage2')
    stage(p1, 1.5, tag='stage2')
    stage(p1, 2.0, tag='stage2')
    stage(p1, 1.0, floor=86400.0, tag='stage2')
    stage(p2, 1.0, tag='stage3')
    # weights from the capped, scaled prediction (what is actually submitted)
    stage(cap(scaled(p1, Hs), Hs), 1.0, tag='stage2 scaled-capped')
