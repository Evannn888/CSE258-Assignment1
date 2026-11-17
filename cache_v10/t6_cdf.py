"""Task 6 harness: estimate the whole distribution of the final gap with binary classifiers P(gap > t_k | x)
(log-loss, far more sample-efficient than sign-gradient quantile regression) and predict its median."""
import sys, numpy as np, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
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
THR = 0.5 * 1.6 ** np.arange(19)                      # days: 0.5 ... 2361
CKW = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=10.0, random_state=258)

def cdf_median(P, H):
    """P[:, k] = P(gap > THR[k]); median by log-linear interpolation of the logit; capped by H."""
    P = np.minimum.accumulate(np.clip(P, 1e-4, 1 - 1e-4), axis=1)
    lt = np.log(THR * DAY)
    n = len(P); out = np.empty(n)
    L = np.log(P / (1 - P))
    for r in range(n):
        k = np.searchsorted(-P[r], -0.5)               # first k with P < 0.5
        if k == 0:
            out[r] = 0.5 * THR[0] * DAY
        elif k == len(THR):
            out[r] = THR[-1] * DAY * 1.5
        else:
            a, b = L[r, k - 1], L[r, k]
            out[r] = np.exp(lt[k - 1] + (lt[k] - lt[k - 1]) * (a / (a - b) if a > b else 0.5))
    return out

for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    g_full = M.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    sets = M.task6_last_event_sets(g_full, data) + M.task6_cutoff_sets(g_full, data)
    X = np.vstack([s[0] for s in sets]); y = np.concatenate([s[1] for s in sets]); Hs = np.concatenate([s[2] for s in sets])
    te = data.tests['task_6_test.csv']
    Xt, Ht, _ = M.task6_features(g_full, te['user'], te['item'])
    def report(tag, raw):
        out = [f'{np.mean(np.abs(cap(raw * table_scale(Ht, t), Ht) - truth)) / 1e6:.4f}M' for t in ([1, 1, 1, 1], [1.1, 1.1, 1.05, 1.03], [1.05] * 4, [1.25, 1.2, 1.1, 1.05], [0.95] * 4)]
        print(name, tag, 'none/mild/1.05/v9/0.95:', out, flush=True)
    t0 = time.time()
    P = np.zeros((len(Xt), len(THR)))
    for k, thr in enumerate(THR):
        m = (Hs > thr * DAY) | (Hs <= 0)
        lab = (y[m] > thr * DAY).astype(int)
        if lab.min() == lab.max():
            P[:, k] = lab[0]
            continue
        clf = HistGradientBoostingClassifier(**CKW).fit(X[m], lab)
        P[:, k] = clf.predict_proba(Xt)[:, 1]
        P[(Ht > 0) & (Ht <= thr * DAY), k] = 0.0
    print(name, f'classifiers {time.time() - t0:.0f}s', flush=True)
    med = cdf_median(P, Ht)
    report('CDF median', med)
    np.savez(f'/data/UCSD/CSE258/A1/cache_v10/t6_cdf_{name[:9]}.npz', P=P, med=med, truth=truth, H=Ht)
