"""Session EASE for Task 4: EASE on (user, day) sessions -> same-day co-shelving weights; score the candidate from the
user's items on/near the query day.  Per-fold matrices (cheap: 17k items).  Stack test on folds 0/1."""
import sys, numpy as np, numba as nb, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from ease_big import ease_inplace
from sklearn.ensemble import HistGradientBoostingClassifier


@nb.njit(parallel=True, cache=True)
def sess_scores(qu, qc, qt, uptr, ui, ut, B):
    n, C = qc.shape
    out = np.zeros((n, C, 4), np.float32)
    for r in nb.prange(n):
        u = qu[r]
        for y in range(uptr[u], uptr[u + 1]):
            h = ui[y]
            dt = abs(ut[y] - qt[r]) / 86400.0
            for c in range(C):
                b = B[h, qc[r, c]]
                if dt < 1.0:
                    out[r, c, 0] += b
                if dt < 7.0:
                    out[r, c, 1] += b
                out[r, c, 2] += b * np.exp(-dt / 3.0)
                out[r, c, 3] += b
    return out


def session_ids(u, t):
    day = np.floor(t / 86400.0).astype(np.int64)
    key = u.astype(np.int64) * 100000 + (day - day.min())
    _, sid = np.unique(key, return_inverse=True)
    return sid


lam = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz'); b4 = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
e4 = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t4.npz')
u, i, t, r = z['u'], z['i'], z['t'], z['r']; nU, nI = int(z['nU']), int(z['nI'])
SS = []
for f in range(4):
    keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
    sid = session_ids(u[keep], t[keep])
    B = ease_inplace(sid, i[keep], int(sid.max()) + 1, nI, lam)
    g = V.Graph(u[keep], i[keep], t[keep], r[keep], nU, nI)
    S = sess_scores(u[z[f'held{f}']], z[f'qc{f}'], z[f'qt{f}'], g.uptr, g.i, g.t, B)
    del B
    pop = np.maximum(g.icnt[z[f'qc{f}']], 1)[..., None] ** 0.75
    SS.append(np.concatenate([S, S / pop], -1).astype(np.float32))
    V.log(f'fold {f}: sessions {sid.max()+1}, standalone same-day {np.mean(S[:, 0, 0] > S[:, 1, 0]):.4f} (ties {np.mean(S[:, 0, 0] == S[:, 1, 0]):.3f}) all-history-norm {np.mean(SS[-1][:, 0, 7] > SS[-1][:, 1, 7]):.4f}')
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
def build(use):
    out = []
    for f in range(4):
        keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
        pop = np.bincount(i[keep], minlength=nI).astype(float)
        parts = [z[f'F{f}'], z[f'L{f}'], b4[f'S{f}'][..., None]]
        for lam_ in (500, 2000):
            X = e4[f'E{f}_{lam_}']
            parts.append(np.concatenate([X, X / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1).astype(np.float32))
        if use:
            parts.append(SS[f])
        out.append((np.concatenate(parts, -1), z[f'ctx{f}'], z[f'seen{f}']))
    return out
for use in (False, True):
    S_ = build(use)
    accs = []
    for tf in (0, 1):
        X, Y = [], []
        for k in range(4):
            if k == tf: continue
            F, ctx, _ = S_[k]
            for c in range(1, F.shape[1]):
                d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
                X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
        clf = HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **small).fit(np.vstack(X), np.concatenate(Y))
        F, c, s = S_[tf]
        m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T4 v8 features{" + session EASE(%g)" % lam if use else ""}: fold0 {accs[0]:.4f} fold1 {accs[1]:.4f} mean {np.mean(accs):.4f}', flush=True)
