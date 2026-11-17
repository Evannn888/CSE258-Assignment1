import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
data = V.Data(Path('/data/UCSD/CSE258/A1/assignment1_2026'), 'task_6_train.csv', ['task_6_test.csv'])
g_full = V.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
ends = g_full.uptr[1:]
sets = {}
for k in (1, 2, 3, 4):
    keep = np.ones(len(g_full.u), bool)
    for j in range(1, k + 1):
        keep[(ends - j)[g_full.ucnt >= j]] = False
    tgt = ends - k; users = np.flatnonzero(tgt - g_full.uptr[:-1] >= 3)
    g = V.Graph(g_full.u[keep], g_full.i[keep], g_full.t[keep], g_full.r[keep], data.nU, data.nI)
    X, H, tl = V.task6_features(g, users, g_full.i[tgt[users]])
    sets[k] = (X, g_full.t[tgt[users]] - tl, H)
P = dict(loss='absolute_error', max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=10.0, random_state=V.SEED)
bins = [0, 90, 365, 730, 1460, 1e9]
def study(train_ks, test_k):
    X = np.vstack([sets[k][0] for k in train_ks]); y = np.concatenate([sets[k][1] for k in train_ks])
    m = HistGradientBoostingRegressor(**P).fit(X, np.log1p(np.maximum(y, 0)))
    Xe, ye, He = sets[test_k]; p0 = np.expm1(m.predict(Xe)); Hd = He / V.DAY
    fs = np.round(np.arange(0.9, 1.61, 0.05), 2); best = {}
    tot = {f: 0.0 for f in fs}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (Hd >= lo) & (Hd < hi)
        errs = {f: np.abs(np.where(He[mm] > 0, np.minimum(p0[mm] * f, He[mm]), p0[mm] * f) - ye[mm]).sum() for f in fs}
        for f in fs: tot[f] += errs[f]
        best[(lo, hi)] = min(errs, key=errs.get)
        V.log(f'  train L{train_ks} -> L{test_k}  H [{lo:5.0f},{min(hi, 99999):5.0f}) d: best factor {best[(lo, hi)]:.2f}  P(y>pred)={np.mean(ye[mm] > p0[mm]):.3f}')
    n = len(ye)
    perbin = sum(min(np.abs(np.where(He[(Hd >= lo) & (Hd < hi)] > 0, np.minimum(p0[(Hd >= lo) & (Hd < hi)] * f, He[(Hd >= lo) & (Hd < hi)]), p0[(Hd >= lo) & (Hd < hi)] * f) - ye[(Hd >= lo) & (Hd < hi)]).sum() for f in [best[(lo, hi)]]) for lo, hi in zip(bins[:-1], bins[1:]))
    V.log(f'  MAE uniform 1.00 {tot[1.0] / n:,.0f} | 1.10 {tot[1.1] / n:,.0f} | best uniform {min(tot, key=tot.get)} {min(tot.values()) / n:,.0f} | per-bin best {perbin / n:,.0f}')
    return best
b1 = study([2, 3, 4], 1)
b2 = study([3, 4], 2)
