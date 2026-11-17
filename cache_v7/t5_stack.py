import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz'); s = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sasrec.npz')
def sets(use_sas):
    out = []
    for k in range(5):
        F = np.concatenate([z[f'F{k}'], z[f'L{k}']], -1)
        if use_sas: F = np.concatenate([F, s[f'S{k}'][..., None]], -1)
        out.append((F, z[f'ctx{k}'], z[f'seen{k}']))
    return out
for use in (False, True):
    S_ = sets(use)
    clf = V.fit_pair_classifier([(F, c) for F, c, _ in S_[1:]], 500)
    F, c, se = S_[0]
    m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (se[:, 1] - se[:, 0])
    V.log(f'task 5 validation {"v6 + SASRec" if use else "v6"}: {np.mean(m > 0):.4f}')
