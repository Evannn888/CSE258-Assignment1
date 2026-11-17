import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz'); b = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
e = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t4.npz')
nI = int(z['nI'])
def ease_feat(f):
    keep = np.ones(len(z['u']), bool); keep[z[f'held{f}']] = False
    pop = np.bincount(z['i'][keep], minlength=nI).astype(float)
    qc = z[f'qc{f}']
    S = e[f'E{f}_2000']
    return np.concatenate([S, S / pop[qc][..., None] ** 0.75], -1).astype(np.float32)
def sets(use_ease):
    out = []
    for f in range(4):
        parts = [z[f'F{f}'], z[f'L{f}'], b[f'S{f}'][..., None]]
        if use_ease:
            parts.append(ease_feat(f))
        out.append((np.concatenate(parts, -1), z[f'ctx{f}'], z[f'seen{f}']))
    return out
def fit(sets_, kw):
    X, Y = [], []
    for F, ctx in sets_:
        for c in range(1, F.shape[1]):
            d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
            X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
    return HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **kw).fit(np.vstack(X), np.concatenate(Y))
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
big = dict(max_iter=1000, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=30.0)
for use_ease in (False, True):
    S = sets(use_ease)
    for tag, kw in (('small', small), ('big', big)):
        accs = []
        for tf in (0, 1):
            t0 = time.time()
            clf = fit([S[k][:2] for k in range(4) if k != tf], kw)
            F, c, s = S[tf]
            m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
            accs.append(np.mean(m > 0))
        print(f'T4 ease={use_ease} clf={tag}: fold0 {accs[0]:.4f} fold1 {accs[1]:.4f} mean {np.mean(accs):.4f}', flush=True)
