import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
sys.path.insert(0, '/data/UCSD/CSE258/A1')
from t6_harness import *


def csets(data, cut_days=(30, 90, 180, 365, 730), min_hist=3, tend=A.TEND):
    gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    out = []
    for cd in cut_days:
        C = tend - cd * DAY
        # last event index before C per user, and whether user has an event >= C
        before = np.array([np.searchsorted(gf.t[gf.uptr[u]:gf.uptr[u + 1]], C) for u in range(data.nU)])
        nb_ = before  # number of events before C
        has_after = gf.ucnt > nb_
        users = np.flatnonzero(has_after & (nb_ >= min_hist + 1))
        pf = gf.uptr[users] + nb_[users] - 1           # pseudo-final index
        keep = gf.t < C
        keep[pf] = False
        g = Graph(gf.u[keep], gf.i[keep], gf.t[keep], gf.r[keep], data.nU, data.nI)
        qi = gf.i[pf]
        X, H, tl = feats(g, users, qi, C)
        out.append((X, gf.t[pf] - tl, H, np.full(len(users), -cd)))
        print(f'  cut {cd}d: {len(users)} users, target median {np.median(gf.t[pf]-tl)/DAY:.1f}d, H median {np.median(H)/DAY:.0f}d', flush=True)
    return out


def fit(sets, indicator=False):
    X = np.vstack([np.column_stack([s[0], (s[3] > 0).astype(np.float32)]) if indicator else s[0] for s in sets])
    y = np.concatenate([s[1] for s in sets])
    return HistGradientBoostingRegressor(**PARAMS).fit(X, np.log1p(np.maximum(y, 0)))


def evaluate(model, Xt, Ht, truth, indicator=False, tag=''):
    if indicator:
        Xt = np.column_stack([Xt, np.zeros(len(Xt), np.float32)])
    raw = np.expm1(model.predict(Xt))
    res = {'noscale': np.mean(np.abs(finish(raw, Ht) - truth)), 'v7scale': np.mean(np.abs(finish(raw * scale_of(Ht, SCALE), Ht) - truth))}
    for f in (0.9, 1.1, 1.2):
        res[f'x{f}'] = np.mean(np.abs(finish(raw * f, Ht) - truth))
    print(f'{tag:28s}', ' '.join(f'{k}={v/1e6:.3f}M' for k, v in res.items()), flush=True)
    return raw


if __name__ == '__main__':
    for name in sys.argv[1:]:
        d = load_raw(name)
        data, truth = t6_like(d)
        print(name, len(truth))
        L, gf = lsets(data)
        C = csets(data)
        te = data.tests['task_6_test.csv']
        Xt, Ht, _ = feats(gf, te['user'], te['item'], A.TEND)
        evaluate(fit(L), Xt, Ht, truth, tag='L1-3 (v7)')
        evaluate(fit(C), Xt, Ht, truth, tag='C only')
        evaluate(fit(L + C), Xt, Ht, truth, tag='L+C')
        evaluate(fit(L + C, True), Xt, Ht, truth, True, tag='L+C indicator')
        evaluate(fit(C[:3]), Xt, Ht, truth, tag='C 30/90/180 only')
