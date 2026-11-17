"""Task 4 stack check: v9 features with the time-aware BERT4Rec score added to / replacing the old one."""
import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
tag = sys.argv[1] if len(sys.argv) > 1 else 'gap+month_e120'
z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz'); nI = int(z['nI'])
e = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t4.npz'); b4r = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
def feats(f, mode):
    keep = np.ones(len(z['u']), bool); keep[z[f'held{f}']] = False
    pop = np.bincount(z['i'][keep], minlength=nI).astype(float)
    parts = [z[f'F{f}'], z[f'L{f}']]
    one = lambda suf: np.load(f'/data/UCSD/CSE258/A1/cache_v10/b4rt_gap+month_e120{suf}_f{f}.npy')
    new = (one('') if mode == 'new' else (one('') + one('_r2') + one('_r3')) / 3)[..., None]
    parts.append(new)
    for lam in ('500', '2000'):
        S = e[f'E{f}_{lam}']
        parts.append(np.concatenate([S, S / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1))
    return np.concatenate(parts, -1).astype(np.float32)
def fit(sets_, kw):
    X, Y = [], []
    for F, ctx in sets_:
        for c in range(1, F.shape[1]):
            d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
            X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
    return HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **kw).fit(np.vstack(X), np.concatenate(Y))
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)
for mode in ('new', 'avg3'):
    S = [(feats(f, mode), z[f'ctx{f}'], z[f'seen{f}']) for f in range(4)]
    accs = []
    for tf in range(4):
        clf = fit([S[k][:2] for k in range(4) if k != tf], small)
        F, c, s = S[tf]
        m = V.pair_margin(clf, F[:, 0], F[:, 1], c) + 100 * (s[:, 1] - s[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T4 sequence feature = {mode}: folds {np.round(accs, 4)} mean {np.mean(accs):.4f}', flush=True)
