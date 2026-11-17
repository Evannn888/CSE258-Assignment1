"""Task 6 harness: threshold-classifier probabilities (cross-fitted by user) as extra features of the weighted
median regression."""
import sys, numpy as np, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
spec = importlib.util.spec_from_file_location('a1v9w', '/data/UCSD/CSE258/A1/v9/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v9w'] = M; spec.loader.exec_module(M)
DAY = M.DAY
def cap(p, H):
    return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)
BINS = [(0, 90), (90, 365), (365, 730), (730, 1e9)]
def table_scale(H, table):
    days = np.where(H > 0, H, np.inf) / DAY
    return np.asarray(table)[np.select([(days >= a) & (days < b) for a, b in BINS], range(4), default=3)]
THR = 0.5 * 1.6 ** np.arange(19)
CKW = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=10.0, random_state=258)
PR = dict(M.T6_PARAMS); PR.update(min_samples_leaf=400)

def set_users(g_full, data):
    """User ids of the rows of the pseudo sets, in the order of task6_last_event_sets + task6_cutoff_sets."""
    out = []
    ends = g_full.uptr[1:]
    for k in (1, 2, 3):
        out.append(np.flatnonzero((ends - k) - g_full.uptr[:-1] >= 3))
    for dd in M.T6_CUTOFF_DAYS:
        cut = M.TEND - dd * DAY
        before = np.bincount(g_full.u[g_full.t < cut], minlength=data.nU)
        out.append(np.flatnonzero((g_full.ucnt > before) & (before >= 4)))
    return np.concatenate(out)

def probs(Xa, ya, Ha, Xb, Hb):
    P = np.zeros((len(Xb), len(THR)))
    for k, thr in enumerate(THR):
        m = (Ha > thr * DAY) | (Ha <= 0)
        lab = (ya[m] > thr * DAY).astype(int)
        if lab.min() == lab.max():
            P[:, k] = lab[0]
        else:
            P[:, k] = HistGradientBoostingClassifier(**CKW).fit(Xa[m], lab).predict_proba(Xb)[:, 1]
        P[(Hb > 0) & (Hb <= thr * DAY), k] = 0.0
    return P

for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    g_full = M.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    sets = M.task6_last_event_sets(g_full, data) + M.task6_cutoff_sets(g_full, data)
    X = np.vstack([s[0] for s in sets]); y = np.concatenate([s[1] for s in sets]); Hs = np.concatenate([s[2] for s in sets])
    users = set_users(g_full, data); assert len(users) == len(y)
    ly = np.log1p(np.maximum(y, 0))
    te = data.tests['task_6_test.csv']
    Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
    def report(tag, raw):
        out = [f'{np.mean(np.abs(cap(raw * table_scale(Ht, t), Ht) - truth)) / 1e6:.4f}M' for t in ([1, 1, 1, 1], [1.1, 1.1, 1.05, 1.03], [1.05] * 4, [1.25, 1.2, 1.1, 1.05])]
        print(name, tag, 'none/mild/1.05/v9:', out, flush=True)
    def two_stage(Xa, Xb):
        first = cap(np.expm1(HistGradientBoostingRegressor(**PR).fit(Xa, ly).predict(Xa)), Hs)
        w = np.maximum(first, DAY)
        return np.expm1(HistGradientBoostingRegressor(**PR).fit(Xa, ly, sample_weight=w / w.mean()).predict(Xb))
    t0 = time.time()
    report('weighted two-stage (reference)', two_stage(X, Xt))
    Poof = np.zeros((len(X), len(THR)))
    fold = users % 3
    for f in range(3):
        Poof[fold == f] = probs(X[fold != f], y[fold != f], Hs[fold != f], X[fold == f], Hs[fold == f])
    Pt = probs(X, y, Hs, Xt, Ht)
    print(name, f'cross-fitted classifiers {time.time() - t0:.0f}s', flush=True)
    lg = lambda P: np.log(np.clip(P, 1e-4, 1 - 1e-4) / (1 - np.clip(P, 1e-4, 1 - 1e-4))).astype(np.float32)
    report('two-stage + threshold logits', two_stage(np.hstack([X, lg(Poof)]), np.hstack([Xt, lg(Pt)])))
