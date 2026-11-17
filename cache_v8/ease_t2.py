import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import assignment1 as V
from ease_big import ease_inplace
from ease_pairs import ease_scores
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t2.npz')
u, i, t, r = z['u'], z['i'], z['t'], z['r']
nU, nI = int(z['nU']), int(z['nI'])
held_all = np.concatenate([z[f'held{f}'] for f in range(4)])
excl = np.ones(len(u), bool); excl[held_all] = False
res = {}
mode = sys.argv[1]
lams = [float(x) for x in sys.argv[2:]]
for lam in lams:
    if mode == 'folds':
        B = ease_inplace(u[excl], i[excl], nU, nI, lam)
        for f in range(4):
            keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
            g = V.Graph(u[keep], i[keep], t[keep], r[keep], nU, nI)
            S = ease_scores(u[z[f'held{f}']], z[f'qc{f}'], z[f'qt{f}'], g.uptr, g.i, g.t, B, np.zeros(0))
            res[f'E{f}_{lam:g}'] = S
            pop = np.bincount(i[keep], minlength=nI).astype(float)
            n = S[..., 0] / pop[z[f'qc{f}']] ** 0.75
            V.log(f'lam {lam:g} fold {f}: raw acc {np.mean(S[:, 0, 0] > S[:, 1, 0]):.4f} norm acc {np.mean(n[:, 0] > n[:, 1]):.4f}')
    else:
        B = ease_inplace(u, i, nU, nI, lam)
        g = V.Graph(u, i, t, r, nU, nI)
        res[f'Etest_{lam:g}'] = ease_scores(z['testqu'], z['testqc'], z['testqt'], g.uptr, g.i, g.t, B, np.zeros(0))
    del B
np.savez(f'/data/UCSD/CSE258/A1/cache_v8/ease_t2_{mode}_{"_".join(sys.argv[2:])}.npz', **res)
V.log('saved')
