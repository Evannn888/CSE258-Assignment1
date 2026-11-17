import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
task = int(sys.argv[1])
z = np.load(f'/data/UCSD/CSE258/A1/cache_v7/pairs_t{task}.npz')
sets = [(np.concatenate([z[f'F{f}'], z[f'L{f}']], -1), z[f'ctx{f}'], z[f'seen{f}']) for f in range(4)]
F0, c0, s0 = sets[0]
for train_folds in ([1], [1, 2], [1, 2, 3]):
    clf = V.fit_pair_classifier([sets[f][:2] for f in train_folds], 400)
    m = V.pair_margin(clf, F0[:, 0], F0[:, 1], c0) + 100 * (s0[:, 1] - s0[:, 0])
    V.log(f'task {task}: stacker trained on {len(train_folds)} fold(s) ({30000 * len(train_folds) * 2:,} pos-neg pairs): fold0 acc {np.mean(m > 0):.4f}')
