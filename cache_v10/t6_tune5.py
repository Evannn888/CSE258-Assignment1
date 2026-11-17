"""Task 6 harness tuning around the v10 model (weighted second fit).  Pseudo sets are built once per cutoff list;
every fit configuration is scored on the Task-6-like problem (true final gaps) of one file."""
import sys, numpy as np, importlib.util, time, json
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor
spec = importlib.util.spec_from_file_location('a1v11t6', '/data/UCSD/CSE258/A1/v11/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v11t6'] = M; spec.loader.exec_module(M)
DAY = M.DAY
def cap(p, H):
    return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)
def scaled(p, H, table):
    days = np.where(H > 0, H, np.inf) / DAY
    return p * np.select([days < lim for lim, _ in table], [f for _, f in table])
V10_SCALE = [(90, 1.08), (365, 1.05), (np.inf, 1.0)]
def build(g_full, data, cutoffs, ks=(1, 2, 3)):
    sets = []
    ends = g_full.uptr[1:]
    for k in ks:
        keep = np.ones(len(g_full.u), bool)
        for j in range(1, k + 1):
            keep[(ends - j)[g_full.ucnt >= j]] = False
        tgt = ends - k
        users = np.flatnonzero(tgt - g_full.uptr[:-1] >= 3)
        g = M.Graph(g_full.u[keep], g_full.i[keep], g_full.t[keep], g_full.r[keep], data.nU, data.nI)
        X, H, tl = M.task6_features(g, users, g_full.i[tgt[users]])
        sets.append((X, g_full.t[tgt[users]] - tl, H, np.full(len(users), k)))
    for dd in cutoffs:
        cut = M.TEND - dd * DAY
        before = np.bincount(g_full.u[g_full.t < cut], minlength=data.nU)
        users = np.flatnonzero((g_full.ucnt > before) & (before >= 4))
        tgt = g_full.uptr[users] + before[users] - 1
        keep = g_full.t < cut
        keep[tgt] = False
        g = M.Graph(g_full.u[keep], g_full.i[keep], g_full.t[keep], g_full.r[keep], data.nU, data.nI)
        X, H, tl = M.task6_features(g, users, g_full.i[tgt], tend=cut)
        sets.append((X, g_full.t[tgt] - tl, H, np.full(len(users), 100 + dd)))
    return sets
name = sys.argv[1]
d = load_raw(name)
data, truth = t6_like(d)
g_full = M.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
te = data.tests['task_6_test.csv']
Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
C11 = (14, 30, 60, 90, 120, 180, 270, 365, 540, 730, 1095)
C21 = (7, 14, 21, 30, 45, 60, 90, 120, 150, 180, 240, 300, 365, 450, 540, 640, 730, 900, 1095, 1460, 1825)
t0 = time.time()
S = {'c11': build(g_full, data, C11)}
print(name, f'sets built {time.time() - t0:.0f}s', flush=True)
def fit_eval(tag, sets, params=None, power=1.0, floor=DAY, drop_k=()):
    P = dict(M.T6_PARAMS); P.update(params or {})
    sets = [s for s in sets if s[3][0] not in drop_k]
    X = np.vstack([s[0] for s in sets]); y = np.log1p(np.maximum(np.concatenate([s[1] for s in sets]), 0)); H = np.concatenate([s[2] for s in sets])
    first = cap(np.expm1(HistGradientBoostingRegressor(**P).fit(X, y).predict(X)), H)
    w = np.maximum(first, floor) ** power
    raw = np.expm1(HistGradientBoostingRegressor(**P).fit(X, y, sample_weight=w / w.mean()).predict(Xt))
    res = {'v10scale': np.mean(np.abs(cap(scaled(raw, Ht, V10_SCALE), Ht) - truth)) / 1e6}
    for f in (1.0, 1.05, 1.1):
        res[f'x{f}'] = np.mean(np.abs(cap(raw * f, Ht) - truth)) / 1e6
    print(name, tag, ' '.join(f'{k} {v:.4f}M' for k, v in res.items()), f'({time.time() - t0:.0f}s)', flush=True)
    return res
B = dict(max_leaf_nodes=511, l2_regularization=1000.0)
fit_eval('v12 leaf200 it1800', S['c11'], dict(B, min_samples_leaf=200, max_iter=1800))
fit_eval('v12 leaf300 it1800', S['c11'], dict(B, min_samples_leaf=300, max_iter=1800))
