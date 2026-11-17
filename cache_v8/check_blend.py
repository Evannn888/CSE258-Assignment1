import sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('a1v9', '/data/UCSD/CSE258/A1/v9/assignment1.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t2.npz'); e = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t2_folds_300.npz')
u, i = z['u'], z['i']; nI = int(z['nI'])
S = []
for f in range(4):
    keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
    pop = np.bincount(i[keep], minlength=nI).astype(float)
    X = e[f'E{f}_300']
    S.append((np.concatenate([z[f'F{f}'], z[f'L{f}'], np.concatenate([X, X / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1)], -1).astype(np.float32), z[f'ctx{f}'], z[f'seen{f}']))
F, ctx, seen = S[0]
m = M.blended_margin(2, [(F_, c) for F_, c, _ in S[1:]], 'small', 400, F[:, 0], F[:, 1], ctx) + 100 * (seen[:, 1] - seen[:, 0])
print('T2 fold0 blended acc', np.mean(m > 0).round(4), '(experiment: gbdt 0.9496, blend0.3 0.9501)')
