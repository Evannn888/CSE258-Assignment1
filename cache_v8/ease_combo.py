import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
def fit(sets_):
    X, Y = [], []
    for F, ctx in sets_:
        for c in range(1, F.shape[1]):
            d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
            X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
    return HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **small).fit(np.vstack(X), np.concatenate(Y))
def run(task, z, extra_parts, lam_sets, tag):
    nI = int(z['nI'])
    S = []
    for f in range(4):
        keep = np.ones(len(z['u']), bool); keep[z[f'held{f}']] = False
        pop = np.bincount(z['i'][keep], minlength=nI).astype(float)
        parts = [z[f'F{f}'], z[f'L{f}']] + [p[f] for p in extra_parts]
        for E, lam in lam_sets:
            X = E[f'E{f}_{lam}']
            parts.append(np.concatenate([X, X / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1).astype(np.float32))
        S.append((np.concatenate(parts, -1), z[f'ctx{f}'], z[f'seen{f}']))
    accs = []
    for tf in (0, 1):
        clf = fit([S[k][:2] for k in range(4) if k != tf])
        F, c, s = S[tf]
        m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T{task} {tag}: fold0 {accs[0]:.4f} fold1 {accs[1]:.4f} mean {np.mean(accs):.4f}', flush=True)
z4 = np.load('cache_v7/pairs_t4.npz'); b4 = np.load('cache_v7/t4_b4r.npz'); e4 = np.load('cache_v8/ease_t4.npz')
b4p = [[b4[f'S{f}'][..., None] for f in range(4)]]
run(4, z4, b4p, [(e4, 500)], 'ease 500')
run(4, z4, b4p, [(e4, 500), (e4, 2000)], 'ease 500+2000')
z2 = np.load('cache_v7/pairs_t2.npz'); e2a = np.load('cache_v8/ease_t2_folds_300.npz'); e2b = np.load('cache_v8/ease_t2_folds_1000_3000.npz')
run(2, z2, [], [(e2a, 300), (e2b, 3000)], 'ease 300+3000')
