"""Check the v10 production sasrec_scores on cached Task 5 set 0 (stand-alone pair accuracy; experiment: 0.8798)."""
import sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('a1v10s', '/data/UCSD/CSE258/A1/v10/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v10s'] = M; spec.loader.exec_module(M)
z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz')
nU, nI = int(z['nU']), int(z['nI'])
hu, hi, ht = z['hu0'], z['hi0'], z['ht0']
g = M.Graph(hu, hi, ht, np.zeros(len(hu)), nU, nI)
S = M.sasrec_scores(g, z['users0'], z['qc0'], seed=0)
M.log(f'production time-aware SASRec set 0: single acc {np.mean(S[:, 0] > S[:, 1]):.4f}')
S2 = M.sasrec_scores(g, z['users0'], z['qc0'], seed=0, use_time=False)
M.log(f'production SASRec without time set 0: single acc {np.mean(S2[:, 0] > S2[:, 1]):.4f} (v9 cached 0.8722)')
