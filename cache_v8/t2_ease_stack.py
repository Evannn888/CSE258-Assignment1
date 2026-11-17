import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t2.npz')
e = np.load(sys.argv[1]); LAMS = [int(x) for x in sys.argv[2:]]
nI = int(z['nI'])
def ease_feat(f, lam):
    keep = np.ones(len(z['u']), bool); keep[z[f'held{f}']] = False
    pop = np.bincount(z['i'][keep], minlength=nI).astype(float)
    S = e[f'E{f}_{lam}']
    return np.concatenate([S, S / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1).astype(np.float32)
def sets(lam):
    return [(np.concatenate([z[f'F{f}'], z[f'L{f}']] + ([ease_feat(f, lam)] if lam else []), -1), z[f'ctx{f}'], z[f'seen{f}']) for f in range(4)]
def fit(sets_, kw):
    X, Y = [], []
    for F, ctx in sets_:
        for c in range(1, F.shape[1]):
            d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
            X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
    return HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **kw).fit(np.vstack(X), np.concatenate(Y))
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
big = dict(max_iter=1000, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=30.0)
for lam, kw, tag in [(x, small, "small") for x in LAMS]:
    S = sets(lam)
    accs = []
    for tf in (0, 1):
        clf = fit([S[k][:2] for k in range(4) if k != tf], kw)
        F, c, s = S[tf]
        m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T2 ease={lam} clf={tag}: fold0 {accs[0]:.4f} fold1 {accs[1]:.4f} mean {np.mean(accs):.4f}', flush=True)
