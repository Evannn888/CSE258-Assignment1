"""Check the v10 production bert4rec_scores on a cached Task 4 fold (stand-alone pair accuracy)."""
import sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('a1v10b', '/data/UCSD/CSE258/A1/v10/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v10b'] = M; spec.loader.exec_module(M)
f = int(sys.argv[1])
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz')
u, i, t, r, nU, nI = z['u'], z['i'], z['t'], z['r'], int(z['nU']), int(z['nI'])
keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
g = M.Graph(u[keep], i[keep], t[keep], r[keep], nU, nI)
S = M.bert4rec_scores(g, u[z[f'held{f}']], z[f'qt{f}'], z[f'qc{f}'], seed=f)
M.log(f'production time-aware BERT4Rec fold {f}: single acc {np.mean(S[:, 0] > S[:, 1]):.4f}')
