"""Harness: v8 Task-6 pipeline + (rel) relatedness of the target to the user's last items, normalised by popularity;
(wlag) co-reader lag weighted by co-reader/user history overlap."""
import sys, numpy as np, numba as nb, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor
spec = importlib.util.spec_from_file_location('a1v8', '/data/UCSD/CSE258/A1/v8/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v8'] = M; spec.loader.exec_module(M)
ORIG = M.task6_features


@nb.njit(cache=True)
def _rel_wlag(qu, qi, uptr, ui, ut, iptr, iusers, itimes, sptr, sitem, stime, icnt, m, maxr):
    """For co-readers v of j: overlap o_v = number of u's last m items v also read.  rel: sum_v o_v normalised;
    wlag: lag estimates weighted by o_v^2 (co-readers sharing more of u's recent books count more)."""
    n = len(qu)
    out = np.full((n, 6), np.nan)
    buf = np.empty(maxr * m); wbuf = np.empty(maxr * m)
    for r in range(n):
        u, j = qu[r], qi[r]
        a, b = uptr[u], uptr[u + 1]
        if b - a == 0:
            continue
        p = ut[b - 1]
        lo = max(a, b - m)
        popsum = 0.0
        for y in range(lo, b):
            popsum += icnt[ui[y]]
        ja, jb = iptr[j], iptr[j + 1]
        step = max(1, (jb - ja) // maxr)
        k = 0; tot = 0.0; last1 = 0.0
        for x in range(ja, jb, step):
            v = iusers[x]
            if v == u:
                continue
            ov = 0
            start = k
            for y in range(lo, b):
                h = ui[y]
                L, R = sptr[v], sptr[v + 1]
                while L < R:
                    M_ = (L + R) // 2
                    if sitem[M_] < h:
                        L = M_ + 1
                    else:
                        R = M_
                if L < sptr[v + 1] and sitem[L] == h:
                    ov += 1
                    if y == b - 1:
                        last1 += 1.0
                    if k < len(buf):
                        buf[k] = ut[y] + (itimes[x] - stime[L]) - p
                        k += 1
            for q in range(start, k):
                wbuf[q] = ov * ov
            tot += ov
        nj = max(jb - ja, 1)
        out[r, 0] = np.log1p(tot) - 0.5 * np.log1p(nj) - 0.5 * np.log1p(popsum)
        out[r, 1] = np.log1p(last1) - 0.5 * np.log1p(nj) - 0.5 * np.log1p(icnt[ui[b - 1]])
        out[r, 2] = tot / (nj * (b - lo))
        if k > 0:
            e = buf[:k]; w = wbuf[:k]
            o = np.argsort(e)
            e, w = e[o], w[o]
            cw = np.cumsum(w) / w.sum()
            for qq, q in enumerate((0.25, 0.5, 0.75)):
                val = e[min(np.searchsorted(cw, q), k - 1)]
                out[r, 3 + qq] = np.sign(val) * np.log1p(abs(val) / 86400.0)
    return out


def rel_wlag(g, qu, qi, m=10, maxr=3000):
    o = np.lexsort((g.i, g.u))
    sptr = np.searchsorted(g.u[o], np.arange(g.nU + 1))
    return _rel_wlag(qu.astype(np.int64), qi.astype(np.int64), g.uptr, g.i, g.t, g.iptr, g.iusers, g.itimes,
                     sptr, g.i[o], g.t[o], g.icnt, m, maxr)


def make(cols):
    def f(g, qu, qi, tend=M.TEND):
        X, H, tl = ORIG(g, qu, qi, tend)
        if not cols:
            return X, H, tl
        E = rel_wlag(g, qu, qi)[:, cols]
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
    for tag, cols in (('v8', []), ('+rel', [0, 1, 2]), ('+wlag', [3, 4, 5]), ('+rel+wlag', [0, 1, 2, 3, 4, 5])):
        M.task6_features = make(cols)
        S = M.task6_last_event_sets(g_full, data) + M.task6_cutoff_sets(g_full, data)
        X = np.vstack([s[0] for s in S]); y = np.log1p(np.maximum(np.concatenate([s[1] for s in S]), 0))
        Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
        mdl = HistGradientBoostingRegressor(**M.T6_PARAMS).fit(X, y)
        print(f'{name} {tag}: {np.mean(np.abs(predict(np.expm1(mdl.predict(Xt)), Ht) - truth))/1e6:.4f}M', flush=True)
    M.task6_features = ORIG
