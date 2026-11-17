import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1')
from t6_more import *
from functools import partial
grid = {
    'base': {},
    'lr03_it1200': dict(learning_rate=0.03, max_iter=1200),
    'leaf127': dict(max_leaf_nodes=127),
    'msl50': dict(min_samples_leaf=50),
    'msl800': dict(min_samples_leaf=800),
    'l2_100': dict(l2_regularization=100.0),
    'it300': dict(max_iter=300),
}
for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    te = data.tests['task_6_test.csv']
    print(name, flush=True)
    for m in (10, 30):
        fn = partial(colag, m=m)
        Xt, Ht, _ = feats(gf, te['user'], te['item'], A.TEND, fn(gf, te['user'], te['item']))
        S = lsets2(data, extra_fn=fn)[0] + csets2(data, fn)
        for tag, kw in grid.items():
            if m == 30 and tag != 'base':
                continue
            raw = np.expm1(fitp(S, {**PARAMS, **kw}).predict(Xt))
            print(f'  m={m} {tag:12s} v7scale {mae(raw*scale_of(Ht, SCALE), Ht, truth)/1e6:.3f}M noscale {mae(raw, Ht, truth)/1e6:.3f}M', flush=True)
