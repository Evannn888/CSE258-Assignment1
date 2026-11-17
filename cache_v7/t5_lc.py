"""Task 5 stacker learning curve: train on 1..4 cutoff sets, evaluate on the last-event set."""
import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz')
S = [(np.concatenate([z[f'F{k}'], z[f'L{k}']], -1), z[f'ctx{k}'], z[f'seen{k}']) for k in range(5)]
F0, c0, s0 = S[0]
for ks in ([1], [1, 2], [1, 2, 3], [1, 2, 3, 4]):
    clf = V.fit_pair_classifier([S[k][:2] for k in ks], 500)
    m = V.pair_margin(clf, F0[:, 0], F0[:, 1], c0) + 100 * (s0[:, 1] - s0[:, 0])
    V.log(f'task 5 stacker on {len(ks)} cutoff set(s): last-event acc {np.mean(m > 0):.4f}')
