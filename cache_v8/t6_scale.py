import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1')
from t6_colag import *
BUCKETS = [0, 30, 90, 180, 365, 730, 1460, 1e9]
FACT = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0])
for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    te = data.tests['task_6_test.csv']
    Xt = np.column_stack([feats(gf, te['user'], te['item'], A.TEND)[0], colag(gf, te['user'], te['item'])])
    Ht = A.TEND - gf.last_time(te['user'])
    S, _ = lsets2(data, extra_fn=colag)
    mdl = HistGradientBoostingRegressor(**PARAMS).fit(np.vstack([s[0] for s in S]), np.log1p(np.maximum(np.concatenate([s[1] for s in S]), 0)))
    raw = np.expm1(mdl.predict(Xt))
    np.save(f'/data/UCSD/CSE258/A1/cache_v8/t6_raw_{name[:8]}.npy', np.column_stack([raw, Ht, truth]))
    garb = gf.t[gf.uptr[te['user'] + 1] - 1] < 0  # placeholder
    print(name)
    best = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        m = (Ht / DAY >= lo) & (Ht / DAY < hi)
        errs = [np.mean(np.abs(finish(raw[m] * f, Ht[m]) - truth[m])) for f in FACT]
        b = int(np.argmin(errs))
        best.append(FACT[b])
        print(f'  H [{lo:>5},{hi:>5}) n={m.sum():5d} ' + ' '.join(f'{f}:{e/1e6:.2f}' for f, e in zip(FACT, errs)) + f'  best {FACT[b]}')
    # total with bucket-best
    p = raw.copy()
    for (lo, hi), f in zip(zip(BUCKETS[:-1], BUCKETS[1:]), best):
        m = (Ht / DAY >= lo) & (Ht / DAY < hi)
        p[m] *= f
    print(f'  total MAE noscale {mae(raw, Ht, truth)/1e6:.3f}M v7scale {mae(raw*scale_of(Ht, SCALE), Ht, truth)/1e6:.3f}M bucket-best {mae(p, Ht, truth)/1e6:.3f}M')
