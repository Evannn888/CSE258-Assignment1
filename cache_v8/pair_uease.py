"""User-user EASE score for pair tasks: sum over the candidate's other users v of W[v, u]."""
import sys, numpy as np, numba as nb, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from ease_big import ease_inplace
from sklearn.ensemble import HistGradientBoostingClassifier


@nb.njit(parallel=True, cache=True)
def uscores(qu, qc, iptr, iu, W):
    n, C = qc.shape
    out = np.zeros((n, C, 2), np.float32)
    for r in nb.prange(n):
        u = qu[r]
        for c in range(C):
            j = qc[r, c]
            s = 0.0
            for y in range(iptr[j], iptr[j + 1]):
                v = iu[y]
                if v != u:
                    s += W[v, u]
            out[r, c, 0] = s
            out[r, c, 1] = s / max(iptr[j + 1] - iptr[j], 1) ** 0.5
    return out


task = int(sys.argv[1]); lam = float(sys.argv[2])
z = np.load(f'/data/UCSD/CSE258/A1/cache_v7/pairs_t{task}.npz')
u, i, t, r = z['u'], z['i'], z['t'], z['r']; nU, nI = int(z['nU']), int(z['nI'])
held_all = np.concatenate([z[f'held{f}'] for f in range(4)])
excl = np.ones(len(u), bool); excl[held_all] = False
t0 = time.time()
W = ease_inplace(i[excl], u[excl], nI, nU, lam)
V.log(f'user-EASE {time.time()-t0:.0f}s')
US = []
for f in range(4):
    keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
    o = np.lexsort((u[keep], i[keep]))
    iptr = np.searchsorted(i[keep][o], np.arange(nI + 1))
    US.append(uscores(u[z[f'held{f}']], z[f'qc{f}'], iptr, u[keep][o], W))
    V.log(f'fold {f} standalone acc raw {np.mean(US[-1][:, 0, 0] > US[-1][:, 1, 0]):.4f} norm {np.mean(US[-1][:, 0, 1] > US[-1][:, 1, 1]):.4f}')
del W
eas = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t2_folds_300.npz') if task == 2 else np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t4.npz')
lams = [300] if task == 2 else [500, 2000]
extra = []
if task == 4:
    b4 = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
def build(use_u):
    S = []
    for f in range(4):
        keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
        pop = np.bincount(i[keep], minlength=nI).astype(float)
        parts = [z[f'F{f}'], z[f'L{f}']] + ([b4[f'S{f}'][..., None]] if task == 4 else [])
        for lam_ in lams:
            X = eas[f'E{f}_{lam_}']
            parts.append(np.concatenate([X, X / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1).astype(np.float32))
        if use_u:
            parts.append(US[f])
        S.append((np.concatenate(parts, -1), z[f'ctx{f}'], z[f'seen{f}']))
    return S
for use_u in (True,):
    S = build(use_u)
    accs = []
    for tf in (0, 1):
        X, Y = [], []
        for k in range(4):
            if k == tf: continue
            F, ctx, _ = S[k]
            for c in range(1, F.shape[1]):
                d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
                X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
        clf = HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **small).fit(np.vstack(X), np.concatenate(Y))
        F, c, s = S[tf]
        m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T{task} EASE + user-EASE({lam:g}): fold0 {accs[0]:.4f} fold1 {accs[1]:.4f} mean {np.mean(accs):.4f}', flush=True)
