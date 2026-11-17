"""Task 6 harness: (a) transductive 'co-final' features: other queries with the same target item and when those users
were last seen; (b) more cutoff pseudo-sets.  Reference = weighted two-stage regression."""
import sys, numpy as np, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
from t6_harness import load_raw, t6_like
from sklearn.ensemble import HistGradientBoostingRegressor
spec = importlib.util.spec_from_file_location('a1v9w', '/data/UCSD/CSE258/A1/v9/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v9w'] = M; spec.loader.exec_module(M)
DAY = M.DAY
def cap(p, H):
    return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)
BINS = [(0, 90), (90, 365), (365, 730), (730, 1e9)]
def table_scale(H, table):
    days = np.where(H > 0, H, np.inf) / DAY
    return np.asarray(table)[np.select([(days >= a) & (days < b) for a, b in BINS], range(4), default=3)]
PR = dict(M.T6_PARAMS); PR.update(min_samples_leaf=400)

def cofinal(qi, tl, nI, icnt):
    """Per query: how many other queries share its target item, and where those users' last events lie relative to mine."""
    o = np.lexsort((tl, qi)); si, st = qi[o], tl[o]
    ptr = np.searchsorted(si, np.arange(nI + 1))
    n = len(qi); F = np.full((n, 7), np.nan, np.float32)
    for r in range(n):
        a, b = ptr[qi[r]], ptr[qi[r] + 1]
        m = b - a - 1
        F[r, 0] = np.log1p(m)
        F[r, 1] = np.log((m + 0.5) / (icnt[qi[r]] + 5.0))
        if m > 0:
            d = (st[a:b] - tl[r]) / DAY                       # includes my own 0
            later = d[d > 0]
            F[r, 2] = len(later) / m
            F[r, 3] = np.sign(np.median(d)) * np.log1p(abs(np.median(d)))
            F[r, 4] = np.log1p(max(d.max(), 0))
            F[r, 5] = np.log1p(np.median(later)) if len(later) else -1.0
            F[r, 6] = np.log1p(np.quantile(later, 0.25)) if len(later) else -1.0
    return F

def build(g_full, data, cutoffs):
    """Pseudo sets as in v9, returning also the target item and the time of the previous event."""
    sets = []
    ends = g_full.uptr[1:]
    for k in (1, 2, 3):
        keep = np.ones(len(g_full.u), bool)
        for j in range(1, k + 1):
            keep[(ends - j)[g_full.ucnt >= j]] = False
        tgt = ends - k
        users = np.flatnonzero(tgt - g_full.uptr[:-1] >= 3)
        g = M.Graph(g_full.u[keep], g_full.i[keep], g_full.t[keep], g_full.r[keep], data.nU, data.nI)
        qi = g_full.i[tgt[users]]
        X, H, tl = M.task6_features(g, users, qi)
        sets.append((X, g_full.t[tgt[users]] - tl, H, cofinal(qi, tl, data.nI, g.icnt)))
    for dd in cutoffs:
        cut = M.TEND - dd * DAY
        before = np.bincount(g_full.u[g_full.t < cut], minlength=data.nU)
        users = np.flatnonzero((g_full.ucnt > before) & (before >= 4))
        tgt = g_full.uptr[users] + before[users] - 1
        keep = g_full.t < cut
        keep[tgt] = False
        g = M.Graph(g_full.u[keep], g_full.i[keep], g_full.t[keep], g_full.r[keep], data.nU, data.nI)
        qi = g_full.i[tgt]
        X, H, tl = M.task6_features(g, users, qi, tend=cut)
        sets.append((X, g_full.t[tgt] - tl, H, cofinal(qi, tl, data.nI, g.icnt)))
    return sets

for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    g_full = M.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    te = data.tests['task_6_test.csv']
    Xt, Ht, tlt = M.task6_features(g_full, te['user'], te['item'])
    Ct = cofinal(te['item'], tlt, data.nI, g_full.icnt)
    def report(tag, raw):
        out = [f'{np.mean(np.abs(cap(raw * table_scale(Ht, t), Ht) - truth)) / 1e6:.4f}M' for t in ([1, 1, 1, 1], [1.1, 1.1, 1.05, 1.03], [1.05] * 4, [1.25, 1.2, 1.1, 1.05])]
        print(name, tag, 'none/mild/1.05/v9:', out, flush=True)
    def two_stage(Xa, ly, Hs, Xb):
        first = cap(np.expm1(HistGradientBoostingRegressor(**PR).fit(Xa, ly).predict(Xa)), Hs)
        w = np.maximum(first, DAY)
        return np.expm1(HistGradientBoostingRegressor(**PR).fit(Xa, ly, sample_weight=w / w.mean()).predict(Xb))
    for tag, cutoffs in (('5 cutoffs', M.T6_CUTOFF_DAYS), ('11 cutoffs', (14, 30, 60, 90, 120, 180, 270, 365, 540, 730, 1095))):
        sets = build(g_full, data, cutoffs)
        X = np.vstack([s[0] for s in sets]); y = np.concatenate([s[1] for s in sets]); Hs = np.concatenate([s[2] for s in sets]); C = np.vstack([s[3] for s in sets])
        ly = np.log1p(np.maximum(y, 0))
        report(f'{tag} ({len(y):,} rows) reference', two_stage(X, ly, Hs, Xt))
        report(f'{tag} + co-final features', two_stage(np.hstack([X, C]), ly, Hs, np.hstack([Xt, Ct])))
        report(f'{tag} + co-final count only', two_stage(np.hstack([X, C[:, :2]]), ly, Hs, np.hstack([Xt, Ct[:, :2]])))
