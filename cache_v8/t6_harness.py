"""Task-6 method harness: turn a file with (nearly) complete histories into a Task-6-like problem
(hold out each user's final event; target = final time - last remaining time) and score methods.
Only used to compare Task-6 METHODS; the Task-6 model itself is trained on task_6_train.csv only."""
import sys, numpy as np, time
from types import SimpleNamespace
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as A
from assignment1 import Graph, DAY, user_gap_stats, time_window_features, item_after_quantiles, item_gap_priors, reading_lag_features
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path


def load_raw(name):
    d = A.Data(Path('/data/UCSD/CSE258/A1/assignment1_2026'), name, ['task_6_test.csv'] if name == 'task_6_train.csv' else [])
    return d


def t6_like(d, min_events=5):
    """Hold out each user's last event (by time).  Returns data namespace + truth."""
    g = Graph(d.u, d.i, d.t, d.r, d.nU, d.nI)
    ends = g.uptr[1:]
    users = np.flatnonzero(g.ucnt >= min_events)
    keep = np.ones(len(g.u), bool)
    keep[ends[users] - 1] = False
    sel = np.zeros(d.nU, bool); sel[users] = True
    keep &= sel[g.u]                       # only users with >= min_events
    final_t = g.t[ends[users] - 1]
    final_i = g.i[ends[users] - 1]
    prev_t = g.t[ends[users] - 2]
    data = SimpleNamespace(u=g.u[keep], i=g.i[keep], t=g.t[keep], r=g.r[keep], nU=d.nU, nI=d.nI,
                           tests={'task_6_test.csv': {'user': users, 'item': final_i}})
    return data, final_t - prev_t


def feats(g, qu, qi, tend, extra=None):
    tl = g.last_time(qu)
    H = tend - tl
    G = user_gap_stats(g.uptr, g.t)[qu]
    F = [G, np.log1p(np.maximum(H, 0) / DAY), (tl - 1.4e9) / 3e7, np.log1p(g.icnt[qi])]
    F.append(time_window_features(g, qi, tl, [(0, 7), (0, 30), (0, 90), (0, 365), (0, 3650), (-365, 0)]))
    F.append(item_after_quantiles(qi, tl, g.iptr, g.itimes))
    F.append(item_gap_priors(g)[qi])
    F.append(reading_lag_features(g, qu, qi, tl))
    if extra is not None:
        F.append(extra)
    return np.column_stack(F).astype(np.float32), H, tl


PARAMS = dict(loss='absolute_error', max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
              l2_regularization=10.0, random_state=258)
SCALE = [(90, 1.25), (365, 1.20), (730, 1.10), (np.inf, 1.05)]


def scale_of(H, table):
    days = np.where(H > 0, H, np.inf) / DAY
    return np.select([days < lim for lim, _ in table], [f for _, f in table])


def finish(p, H):
    return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)


def lsets(data, ks=(1, 2, 3), tend=A.TEND, testfreq=False):
    sets = []
    gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    ends = gf.uptr[1:]
    for k in ks:
        keep = np.ones(len(gf.u), bool)
        for j in range(1, k + 1):
            keep[(ends - j)[gf.ucnt >= j]] = False
        tgt = ends - k
        users = np.flatnonzero(tgt - gf.uptr[:-1] >= 3)
        g = Graph(gf.u[keep], gf.i[keep], gf.t[keep], gf.r[keep], data.nU, data.nI)
        qi = gf.i[tgt[users]]
        extra = None
        if testfreq:
            extra = freq_feats(g, qi, data.nI)
        X, H, tl = feats(g, users, qi, tend, extra)
        sets.append((X, gf.t[tgt[users]] - tl, H, np.full(len(users), k)))
    return sets, gf


def freq_feats(g, qi, nI):
    c = np.bincount(qi, minlength=nI).astype(float)
    return np.column_stack([np.log1p(c[qi]), np.log((c[qi] + 0.5) / (g.icnt[qi] + 5.0))])


def run(data, truth, variant='v7', verbose=True):
    t0 = time.time()
    te = data.tests['task_6_test.csv']
    tf = variant.endswith('+freq')
    sets, gf = lsets(data, testfreq=tf)
    X = np.vstack([s[0] for s in sets]); y = np.concatenate([s[1] for s in sets])
    model = HistGradientBoostingRegressor(**PARAMS).fit(X, np.log1p(np.maximum(y, 0)))
    extra = freq_feats(gf, te['item'], data.nI) if tf else None
    Xt, Ht, _ = feats(gf, te['user'], te['item'], A.TEND, extra)
    raw = np.expm1(model.predict(Xt))
    out = {}
    out['noscale'] = finish(raw, Ht)
    out['v7scale'] = finish(raw * scale_of(Ht, SCALE), Ht)
    for f in (1.1, 1.2, 1.3, 1.5):
        out[f'x{f}'] = finish(raw * f, Ht)
    res = {k: float(np.mean(np.abs(v - truth))) for k, v in out.items()}
    # L1 local validation-style number (model trained on all, scored in-sample on L1 -> optimistic, just for scale)
    if verbose:
        print(variant, {k: f'{v/1e6:.3f}M' for k, v in res.items()}, f'{time.time()-t0:.0f}s', flush=True)
    return res, raw, Ht


if __name__ == '__main__':
    for name in sys.argv[1:]:
        d = load_raw(name)
        data, truth = t6_like(d)
        print(name, 'harness users', len(truth), 'truth median days', np.median(truth) / DAY)
        base = np.mean(np.abs(truth - np.median(truth)))
        # official baseline: user's median gap
        gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
        med = np.array([np.median(np.diff(gf.t[gf.uptr[u]:gf.uptr[u + 1]])) for u in data.tests['task_6_test.csv']['user']])
        print(' global-median MAE', f'{base/1e6:.3f}M', ' official-baseline MAE', f'{np.mean(np.abs(med - truth))/1e6:.3f}M')
        run(data, truth, 'v7')
        run(data, truth, 'v7+freq')
