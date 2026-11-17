"""Task 2/4 stack check: v8 pair features (+EASE) with / without the Bayesian-FM predicted rating and its posterior sd."""
import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
task = int(sys.argv[1])
z = np.load(f'/data/UCSD/CSE258/A1/cache_v7/pairs_t{task}.npz')
nI = int(z['nI'])
if task == 2:
    e = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t2_folds_300.npz'); lams = ['300']
else:
    e = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t4.npz'); lams = ['500', '2000']
    b4r = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
def base(f):
    keep = np.ones(len(z['u']), bool); keep[z[f'held{f}']] = False
    pop = np.bincount(z['i'][keep], minlength=nI).astype(float)
    parts = [z[f'F{f}'], z[f'L{f}']]
    if task == 4:
        parts.append(b4r[f'S{f}'][..., None])
    for lam in lams:
        S = e[f'E{f}_{lam}']
        parts.append(np.concatenate([S, S / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1))
    return np.concatenate(parts, -1).astype(np.float32)
TAG = sys.argv[2] if len(sys.argv) > 2 else 'd64_L3_e35'
def extra(f):
    d = np.load(f'/data/UCSD/CSE258/A1/cache_v10/pairrat_t{task}_f{f}.npz')
    return np.concatenate([np.load(f'/data/UCSD/CSE258/A1/cache_v10/lgcn_t{task}_{TAG}_f{f}.npy')[..., None],
                           np.stack([d['S'], d['SD']], -1)], -1).astype(np.float32)
def fit(sets_, kw):
    X, Y = [], []
    for F, ctx in sets_:
        for c in range(1, F.shape[1]):
            d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
            X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
    return HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **kw).fit(np.vstack(X), np.concatenate(Y))
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
for tag, use in (('base', False), ('+LightGCN+rating', True)):
    S = [(np.concatenate([base(f), extra(f)], -1) if use else base(f), z[f'ctx{f}'], z[f'seen{f}']) for f in range(4)]
    accs = []
    for tf in range(4):
        clf = fit([S[k][:2] for k in range(4) if k != tf], small)
        F, c, s = S[tf]
        m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T{task} {tag}: folds {np.round(accs, 4)} mean {np.mean(accs):.4f}', flush=True)
