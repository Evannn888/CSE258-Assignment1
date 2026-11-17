import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1')
from t6_harness import *
from t6_cut import csets, fit, evaluate
res = {}
for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    te = data.tests['task_6_test.csv']
    Xt, Ht, _ = feats(gf, te['user'], te['item'], A.TEND)
    res[name] = (Xt, Ht, truth)
    # 2-fold oracle: train on true final gaps of half the users
    half = np.random.default_rng(0).permutation(len(truth)) % 2 == 0
    pred = np.zeros(len(truth))
    for part in (half, ~half):
        m = HistGradientBoostingRegressor(**PARAMS).fit(Xt[~part], np.log1p(np.maximum(truth[~part], 0)))
        pred[part] = np.expm1(m.predict(Xt[part]))
    print(name, 'oracle 2-fold (true final labels):', f'{np.mean(np.abs(finish(pred, Ht) - truth))/1e6:.3f}M')
    for f in (0.9, 1.1):
        print('   x', f, f'{np.mean(np.abs(finish(pred*f, Ht) - truth))/1e6:.3f}M')
names = list(res)
if len(names) == 2:
    for a, b in [(names[0], names[1]), (names[1], names[0])]:
        Xa, Ha, ya = res[a]; Xb, Hb, yb = res[b]
        m = HistGradientBoostingRegressor(**PARAMS).fit(Xa, np.log1p(np.maximum(ya, 0)))
        p = np.expm1(m.predict(Xb))
        print(f'cross: train truth {a} -> test {b}: {np.mean(np.abs(finish(p, Hb) - yb))/1e6:.3f}M')
