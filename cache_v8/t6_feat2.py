"""Harness: v8 Task-6 pipeline (L + cutoff sets) with extra feature groups appended via a wrapper."""
import sys, numpy as np, numba as nb, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor
spec = importlib.util.spec_from_file_location('a1v8', '/data/UCSD/CSE258/A1/v8/assignment1.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
ORIG = M.task6_features


@nb.njit(cache=True)
def gapseq(qu, uptr, ut):
    n = len(qu)
    out = np.full((n, 9), np.nan)
    for r in range(n):
        u = qu[r]
        a, b = uptr[u], uptr[u + 1]
        L = b - a
        if L < 2:
            continue
        lg = np.empty(L - 1)
        for p in range(a + 1, b):
            lg[p - a - 1] = np.log1p(max(ut[p] - ut[p - 1], 0.0) / 86400.0)
        m = len(lg)
        for k in range(5):
            if m - 1 - k >= 0:
                out[r, k] = lg[m - 1 - k]
        out[r, 5] = lg.max()
        c1 = 0; c2 = 0
        for x in lg:
            c1 += x > np.log1p(180.0)
            c2 += x > np.log1p(365.0)
        out[r, 6] = c1
        out[r, 7] = c2
        k = min(m, 10)
        if k >= 3:
            xs = np.arange(k) * 1.0
            ys = lg[m - k:]
            xm = xs.mean(); ym = ys.mean()
            out[r, 8] = ((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum()
    return out


def season(g, qu, qi):
    tl = g.last_time(qu)
    doy = (tl / 86400.0) % 365.25 / 365.25 * 2 * np.pi
    return np.column_stack([np.sin(doy), np.cos(doy)])


def make(groups):
    def f(g, qu, qi, tend=M.TEND):
        X, H, tl = ORIG(g, qu, qi, tend)
        extra = []
        if 'gaps' in groups:
            extra.append(gapseq(qu.astype(np.int64), g.uptr, g.t))
        if 'season' in groups:
            extra.append(season(g, qu, qi))
        return (np.column_stack([X] + extra).astype(np.float32) if extra else X), H, tl
    return f


def predict(raw, H):
    days = np.where(H > 0, H, np.inf) / M.DAY
    scale = np.select([days < lim for lim, _ in M.T6_SCALE], [f for _, f in M.T6_SCALE])
    p = raw * scale
    return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)


for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    g_full = M.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    te = data.tests['task_6_test.csv']
    for groups in ((), ('gaps',), ('season',), ('gaps', 'season')):
        M.task6_features = make(groups)
        S = M.task6_last_event_sets(g_full, data) + M.task6_cutoff_sets(g_full, data)
        X = np.vstack([s[0] for s in S]); y = np.log1p(np.maximum(np.concatenate([s[1] for s in S]), 0))
        Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
        m = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, y)
        print(f'{name} v8{"+" + "+".join(groups) if groups else ""}: {np.mean(np.abs(predict(np.expm1(m.predict(Xt)), Ht) - truth))/1e6:.4f}M', flush=True)
    M.task6_features = ORIG
