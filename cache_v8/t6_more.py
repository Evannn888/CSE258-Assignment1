import sys, numpy as np, numba as nb
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1')
from t6_colag import *


@nb.njit(cache=True)
def _activity(qu, uptr, ut, ur):
    n = len(qu)
    out = np.full((n, 12), np.nan)
    wins = (0.3, 1.0, 7.0, 30.0, 90.0, 365.0)
    for r in range(n):
        u = qu[r]
        a, b = uptr[u], uptr[u + 1]
        if b - a == 0:
            continue
        p = ut[b - 1]
        for w in range(6):
            c = 0
            for y in range(b - 1, a - 1, -1):
                if p - ut[y] <= wins[w] * 86400.0:
                    c += 1
                else:
                    break
            out[r, w] = np.log1p(c)
        out[r, 6] = np.log1p(max(p - ut[a], 0.0) / 86400.0)       # tenure
        out[r, 7] = ur[b - 1]                                       # last rating
        s = 0.0; z = 0.0; c = 0
        for y in range(max(a, b - 5), b):
            s += ur[y]; z += ur[y] == 0; c += 1
        out[r, 8] = s / c
        out[r, 9] = z / c
        zz = 0.0
        for y in range(a, b):
            zz += ur[y] == 0
        out[r, 10] = zz / (b - a)
        # events in last 365 days relative to user's average yearly rate
        span = max(p - ut[a], 86400.0 * 30)
        out[r, 11] = np.log((np.exp(out[r, 5]) - 1 + 0.5) / ((b - a) * 365.0 * 86400.0 / span + 0.5))
    return out


def activity(g, qu, qi):
    return _activity(qu.astype(np.int64), g.uptr, g.t, g.r)


def item_zero(g, qu, qi):
    z = np.bincount(g.i, weights=(g.r == 0).astype(float), minlength=g.nI)
    return np.column_stack([(z[qi] + 0.5) / (g.icnt[qi] + 2)])


def combo(*fns):
    return lambda g, qu, qi: np.column_stack([f(g, qu, qi) for f in fns])


def csets2(data, extra_fn, cut_days=(30, 90, 180, 365, 730), min_hist=3):
    gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    out = []
    for cd in cut_days:
        C = A.TEND - cd * DAY
        nb_ = np.array([np.searchsorted(gf.t[gf.uptr[u]:gf.uptr[u + 1]], C) for u in range(data.nU)])
        users = np.flatnonzero((gf.ucnt > nb_) & (nb_ >= min_hist + 1))
        pf = gf.uptr[users] + nb_[users] - 1
        keep = gf.t < C
        keep[pf] = False
        g = Graph(gf.u[keep], gf.i[keep], gf.t[keep], gf.r[keep], data.nU, data.nI)
        qi = gf.i[pf]
        X, H, tl = feats(g, users, qi, C, extra_fn(g, users, qi) if extra_fn else None)
        out.append((X, gf.t[pf] - tl, H, np.full(len(users), -cd)))
    return out


def fitp(sets, params=PARAMS):
    X = np.vstack([s[0] for s in sets]); y = np.concatenate([s[1] for s in sets])
    return HistGradientBoostingRegressor(**params).fit(X, np.log1p(np.maximum(y, 0)))


if __name__ == '__main__':
    variants = {
        'colag': colag,
        'colag+act': combo(colag, activity),
        'colag+act+iz': combo(colag, activity, item_zero),
    }
    for name in sys.argv[1:]:
        d = load_raw(name)
        data, truth = t6_like(d)
        gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
        te = data.tests['task_6_test.csv']
        print(name, flush=True)
        for tag, fn in variants.items():
            Xt, Ht, _ = feats(gf, te['user'], te['item'], A.TEND, fn(gf, te['user'], te['item']))
            L, _ = lsets2(data, extra_fn=fn)
            raw = np.expm1(fitp(L).predict(Xt))
            line = f'  {tag:14s} L: {mae(raw*scale_of(Ht, SCALE), Ht, truth)/1e6:.3f}M'
            if tag != 'colag+act+iz':
                C = csets2(data, fn)
                raw2 = np.expm1(fitp(L + C).predict(Xt))
                line += f'  L+C: noscale {mae(raw2, Ht, truth)/1e6:.3f}M v7scale {mae(raw2*scale_of(Ht, SCALE), Ht, truth)/1e6:.3f}M'
            print(line, flush=True)
