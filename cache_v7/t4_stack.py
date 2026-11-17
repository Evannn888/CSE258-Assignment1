import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz'); b = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
def sets(use):
    return [(np.concatenate([z[f'F{f}'], z[f'L{f}']] + ([b[f'S{f}'][..., None]] if use else []), -1), z[f'ctx{f}'], z[f'seen{f}']) for f in range(4)]
for use in (False, True):
    S = sets(use)
    for tf in (0, 1):
        clf = V.fit_pair_classifier([S[k][:2] for k in range(4) if k != tf], 400)
        F, c, s = S[tf]
        m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
        V.log(f'task 4 {"v6 + BERT4Rec" if use else "v6"} test fold {tf}: {np.mean(m > 0):.4f}')
