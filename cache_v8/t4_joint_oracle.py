import sys, numpy as np, numba as nb
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from ease_big import ease_inplace
from sklearn.ensemble import HistGradientBoostingClassifier
import csv
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz'); b4 = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
e4 = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t4.npz')
u, i, t, r = z['u'], z['i'], z['t'], z['r']; nU, nI = int(z['nU']), int(z['nI'])
# test structure: how many rows have another row of the same user, and how close in time
te_u, te_t = z['testqu'], z['testqt']
o = np.lexsort((te_t, te_u)); uu, tt = te_u[o], te_t[o]
same = uu[1:] == uu[:-1]
dt = np.abs(np.diff(tt))[same] / 86400
print('test rows', len(te_u), 'rows with another row of same user', np.isin(te_u, uu[1:][same]).sum(), '| nearest-gap quantiles (days)', np.percentile(dt, [10, 25, 50, 75]).round(1))
# oracle on validation folds: feature = EASE affinity between candidate and the TRUE positives of the user's other held rows (same fold)
def oracle_feat(f, B):
    qu, qc, qt = u[z[f'held{f}']], z[f'qc{f}'], z[f'qt{f}']
    pos = qc[:, 0]
    out = np.zeros((len(qu), 3, 2), np.float32)
    by = {}
    for k, us in enumerate(qu):
        by.setdefault(us, []).append(k)
    for us, ks in by.items():
        if len(ks) < 2:
            continue
        for k in ks:
            for k2 in ks:
                if k2 == k:
                    continue
                w = np.exp(-abs(qt[k] - qt[k2]) / (30 * 86400))
                out[k, :, 0] += B[pos[k2], qc[k]]
                out[k, :, 1] += B[pos[k2], qc[k]] * w
    return out
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
S = []; O = []
for f in range(4):
    keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
    pop = np.bincount(i[keep], minlength=nI).astype(float)
    parts = [z[f'F{f}'], z[f'L{f}'], b4[f'S{f}'][..., None]]
    for lam_ in (500, 2000):
        X = e4[f'E{f}_{lam_}']
        parts.append(np.concatenate([X, X / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1).astype(np.float32))
    S.append((np.concatenate(parts, -1), z[f'ctx{f}'], z[f'seen{f}']))
    B = ease_inplace(u[keep], i[keep], nU, nI, 500.0)
    O.append(oracle_feat(f, B)); del B
    multi = np.any(O[-1] != 0, axis=(1, 2))
    print(f'fold {f}: rows with other held rows of same user {multi.mean():.3f}')
for use in (False, True):
    accs = []
    for tf in (0, 1):
        X, Y = [], []
        for k in range(4):
            if k == tf: continue
            F, ctx, _ = S[k]
            F = np.concatenate([F, O[k]], -1) if use else F
            for c in range(1, F.shape[1]):
                d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
                X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
        clf = HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **small).fit(np.vstack(X), np.concatenate(Y))
        F, c, s = S[tf]
        F = np.concatenate([F, O[tf]], -1) if use else F
        m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T4 {"+ ORACLE other-row positives" if use else "v8"}: fold0 {accs[0]:.4f} fold1 {accs[1]:.4f} mean {np.mean(accs):.4f}', flush=True)
