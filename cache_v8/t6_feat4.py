"""Harness: v9 Task-6 pipeline + transition co-reader lag: co-readers v for whom the target j came within the next
3 events after one of u's last m books h in v's own timeline; lag t_v(j) - t_v(h) added to t_u(h)."""
import sys, numpy as np, numba as nb, importlib.util
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor
spec = importlib.util.spec_from_file_location('a1v9x', '/data/UCSD/CSE258/A1/v9/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v9x'] = M; spec.loader.exec_module(M)
ORIG = M.task6_features


@nb.njit(cache=True)
def _trans_lag(qu, qi, uptr, ui, ut, iptr, iusers, ipos, m, maxr, nxt):
    n = len(qu)
    out = np.full((n, 5), np.nan)
    buf = np.empty(maxr * m)
    for r in range(n):
        u, j = qu[r], qi[r]
        a, b = uptr[u], uptr[u + 1]
        if b - a == 0:
            continue
        p = ut[b - 1]
        lo = max(a, b - m)
        ja, jb = iptr[j], iptr[j + 1]
        step = max(1, (jb - ja) // maxr)
        k = 0
        for x in range(ja, jb, step):
            v = iusers[x]
            if v == u:
                continue
            pj = uptr[v] + ipos[x]                 # index of v's interaction with j in user-major arrays
            tvj = ut[pj]
            for y in range(lo, b):
                h = ui[y]
                for z in range(max(uptr[v], pj - nxt), pj):
                    if ui[z] == h:
                        if k < len(buf):
                            buf[k] = ut[y] + (tvj - ut[z]) - p
                            k += 1
                        break
        out[r, 0] = np.log1p(k)
        if k > 0:
            e = np.sort(buf[:k])
            for qq, q in enumerate((0.25, 0.5, 0.75)):
                val = e[min(int(q * k), k - 1)]
                out[r, 1 + qq] = np.sign(val) * np.log1p(abs(val) / 86400.0)
            out[r, 4] = np.mean(e < 0)
    return out


def make():
    def f(g, qu, qi, tend=M.TEND):
        X, H, tl = ORIG(g, qu, qi, tend)
        E = _trans_lag(qu.astype(np.int64), qi.astype(np.int64), g.uptr, g.i, g.t, g.iptr, g.iusers, g.ipos, 10, 3000, 3)
        return np.column_stack([X, E]).astype(np.float32), H, tl
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
    for tag, fn in (('v9', ORIG), ('v9+translag', make())):
        M.task6_features = fn
        S = M.task6_last_event_sets(g_full, data) + M.task6_cutoff_sets(g_full, data)
        X = np.vstack([s[0] for s in S]); y = np.log1p(np.maximum(np.concatenate([s[1] for s in S]), 0))
        Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
        mdl = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, y)
        print(f'{name} {tag}: {np.mean(np.abs(predict(np.expm1(mdl.predict(Xt)), Ht) - truth))/1e6:.4f}M', flush=True)
    M.task6_features = ORIG
