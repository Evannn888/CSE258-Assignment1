import sys, numpy as np, numba as nb, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1')
from t6_harness import *
from t6_cut import csets


@nb.njit(cache=True)
def _colag(qu, qi, uptr, ui, ut, iptr, iusers, itimes, sptr, sitem, stime, m, maxr):
    """For each query (user u, target item j): over other readers v of j and the user's last m items h,
    est = t_u(h) + (t_v(j) - t_v(h)) - p_u  (when the user would read j if they kept the co-readers' pace)."""
    n = len(qu)
    out = np.full((n, 8), np.nan)
    buf = np.empty(maxr * m)
    wbuf = np.empty(maxr * m)
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
        nv = 0
        for x in range(ja, jb, step):
            v = iusers[x]
            if v == u:
                continue
            tvj = itimes[x]
            sa, sb = sptr[v], sptr[v + 1]
            hit = False
            for y in range(lo, b):
                h = ui[y]
                # binary search h in v's item-sorted list
                L, R = sa, sb
                while L < R:
                    M = (L + R) // 2
                    if sitem[M] < h:
                        L = M + 1
                    else:
                        R = M
                if L < sb and sitem[L] == h:
                    est = ut[y] + (tvj - stime[L]) - p
                    if k < len(buf):
                        buf[k] = est
                        wbuf[k] = 1.0 / (b - y)
                        k += 1
                    hit = True
            if hit:
                nv += 1
        out[r, 0] = np.log1p(nv)
        if k == 0:
            continue
        e = buf[:k]
        w = wbuf[:k]
        o = np.argsort(e)
        e, w = e[o], w[o]
        cw = np.cumsum(w) / w.sum()
        qs = (0.25, 0.5, 0.75)
        for qq in range(3):
            idx = np.searchsorted(cw, qs[qq])
            idx = min(idx, k - 1)
            val = e[idx]
            out[r, 1 + qq] = np.sign(val) * np.log1p(abs(val) / 86400.0)
        out[r, 4] = np.mean(e < 0)
        out[r, 5] = np.log1p(np.median(np.abs(e)) / 86400.0)
        pos = e[e > 0]
        if len(pos) > 0:
            out[r, 6] = np.log1p(np.median(pos) / 86400.0)
        out[r, 7] = np.log1p(k)
    return out


def colag(g, qu, qi, m=10, maxr=3000):
    o = np.lexsort((g.i, g.u))
    sptr = np.searchsorted(g.u[o], np.arange(g.nU + 1))
    return _colag(qu.astype(np.int64), qi.astype(np.int64), g.uptr, g.i, g.t, g.iptr, g.iusers, g.itimes,
                  sptr, g.i[o], g.t[o], m, maxr)


def lsets2(data, ks=(1, 2, 3), extra_fn=None):
    sets = []
    gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    ends = gf.uptr[1:]
    for k in ks:
        keep = np.ones(len(gf.u), bool)
        for jj in range(1, k + 1):
            keep[(ends - jj)[gf.ucnt >= jj]] = False
        tgt = ends - k
        users = np.flatnonzero(tgt - gf.uptr[:-1] >= 3)
        g = Graph(gf.u[keep], gf.i[keep], gf.t[keep], gf.r[keep], data.nU, data.nI)
        qi = gf.i[tgt[users]]
        X, H, tl = feats(g, users, qi, A.TEND, extra_fn(g, users, qi) if extra_fn else None)
        sets.append((X, gf.t[tgt[users]] - tl, H, np.full(len(users), k)))
    return sets, gf


def mae(p, H, y):
    return np.mean(np.abs(finish(p, H) - y))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        d = load_raw(name)
        data, truth = t6_like(d)
        gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
        te = data.tests['task_6_test.csv']
        t0 = time.time()
        Ct = colag(gf, te['user'], te['item'])
        print(name, f'colag {time.time()-t0:.1f}s', 'coverage', np.mean(np.isfinite(Ct[:, 2])).round(3), flush=True)
        Xt0, Ht, _ = feats(gf, te['user'], te['item'], A.TEND)
        Xt1 = np.column_stack([Xt0, Ct])
        half = np.random.default_rng(0).permutation(len(truth)) % 2 == 0
        for tag, Xt in [('oracle base', Xt0), ('oracle +colag', Xt1)]:
            pred = np.zeros(len(truth))
            for part in (half, ~half):
                mdl = HistGradientBoostingRegressor(**PARAMS).fit(Xt[~part], np.log1p(np.maximum(truth[~part], 0)))
                pred[part] = np.expm1(mdl.predict(Xt[part]))
            print(f'  {tag:16s} {mae(pred, Ht, truth)/1e6:.3f}M', flush=True)
        for tag, fn, Xt in [('L base', None, Xt0), ('L +colag', colag, Xt1)]:
            S, _ = lsets2(data, extra_fn=fn)
            mdl = HistGradientBoostingRegressor(**PARAMS).fit(np.vstack([s[0] for s in S]), np.log1p(np.maximum(np.concatenate([s[1] for s in S]), 0)))
            raw = np.expm1(mdl.predict(Xt))
            print(f'  {tag:16s} noscale {mae(raw, Ht, truth)/1e6:.3f}M v7scale {mae(raw*scale_of(Ht, SCALE), Ht, truth)/1e6:.3f}M', flush=True)
