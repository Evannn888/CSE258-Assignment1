import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1')
import sasrec_ce as SC
import assignment1 as V
z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz')
nU, nI = int(z['nU']), int(z['nI'])
cfg = dict(d=128, drop=0.6, epochs=15, wd=0.3)
res = {}
for k in range(5):
    S = SC.train_score(z[f'hu{k}'], z[f'hi{k}'], nU, nI, z[f'users{k}'], z[f'qc{k}'], seed=k, **cfg)
    res[f'S{k}'] = S
    V.log(f'set {k}: standalone {np.mean(S[:, 0] > S[:, 1]):.4f}')
res['test'] = SC.train_score(z['fu'], z['fi'], nU, nI, z['testqu'], z['testqc'], seed=9, **cfg)
np.savez('/data/UCSD/CSE258/A1/cache_v8/t5_sasce.npz', **res)
V.log('saved')
