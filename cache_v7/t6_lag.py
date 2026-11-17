import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
data = V.Data(Path('/data/UCSD/CSE258/A1/assignment1_2026'), 'task_6_train.csv', ['task_6_test.csv'])
g_full = V.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
ends = g_full.uptr[1:]
def lag_features(g, users, items):
    first = np.full(g.nI, np.inf); np.minimum.at(first, g.i, g.t)
    age = (g.t - first[g.i]) / V.DAY                                   # how long after the book appeared the user read it
    o = np.lexsort((age, g.u)); su, sa = g.u[o], age[o]
    ptr = np.searchsorted(su, np.arange(g.nU + 1))
    tl = g.last_time(users)
    F = np.full((len(users), 6), np.nan)
    for r, u in enumerate(users):
        a = sa[ptr[u]:ptr[u + 1]]
        if len(a) == 0: continue
        med = a[len(a) // 2]
        F[r, 0] = np.log1p(med); F[r, 1] = np.log1p(a[len(a) // 4]); F[r, 2] = np.log1p(a[3 * len(a) // 4]); F[r, 3] = np.mean(a < 30)
        fi = first[items[r]]
        if np.isfinite(fi):
            for k, lagv in enumerate((med, a[len(a) // 4])):
                F[r, 4 + k] = np.log1p(max(fi + lagv * V.DAY - tl[r], 0) / V.DAY)    # predicted gap from user lag
    return F
sets = {}
for k in (1, 2, 3):
    keep = np.ones(len(g_full.u), bool)
    for j in range(1, k + 1):
        keep[(ends - j)[g_full.ucnt >= j]] = False
    tgt = ends - k; users = np.flatnonzero(tgt - g_full.uptr[:-1] >= 3)
    g = V.Graph(g_full.u[keep], g_full.i[keep], g_full.t[keep], g_full.r[keep], data.nU, data.nI)
    items = g_full.i[tgt[users]]
    X, H, tl = V.task6_features(g, users, items)
    sets[k] = (X, lag_features(g, users, items), g_full.t[tgt[users]] - tl, H)
P = dict(loss='absolute_error', max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=10.0, random_state=V.SEED)
for seed in (1, 2, 3):
    half = np.random.default_rng(seed).permutation(len(sets[1][2])) % 2 == 0
    res = {}
    for label, use in [('base', False), ('lag', True)]:
        mk = (lambda s: np.hstack([s[0], s[1]])) if use else (lambda s: s[0])
        errs = []
        for part in (half, ~half):
            X = np.vstack([mk(sets[1])[~part], mk(sets[2]), mk(sets[3])]); y = np.concatenate([sets[1][2][~part], sets[2][2], sets[3][2]])
            m = HistGradientBoostingRegressor(**P).fit(X, np.log1p(np.maximum(y, 0)))
            p = np.expm1(m.predict(mk(sets[1])[part])); H = sets[1][3][part]
            p = np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)
            errs.append(np.mean(np.abs(p - sets[1][2][part])))
        res[label] = np.mean(errs)
    V.log(f'split seed {seed}: base {res["base"]:,.0f}  +lag {res["lag"]:,.0f}  ({100 * (res["lag"] / res["base"] - 1):+.2f}%)')
