import sys, numpy as np, calendar, time
from types import SimpleNamespace
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
name = sys.argv[1]
d = V.Data(Path('/data/UCSD/CSE258/A1/assignment1_2026'), name, [])
gfull = V.Graph(d.u, d.i, d.t, d.r, d.nU, d.nI)
ends = gfull.uptr[1:]
users = np.flatnonzero(gfull.ucnt >= 5)
fin = ends[users] - 1
keep = np.ones(len(gfull.u), bool); keep[fin] = False
data = SimpleNamespace(u=gfull.u[keep], i=gfull.i[keep], t=gfull.t[keep], r=gfull.r[keep], nU=d.nU, nI=d.nI)
rng = np.random.default_rng(5)
g = V.Graph(data.u, data.i, data.t, data.r, d.nU, d.nI)
qc = np.column_stack([gfull.i[fin], V.sample_negatives(data.i, gfull.i[fin], 1, rng)])
Ft, ctxt = V.task5_features(g, V.Spectral(g, recency=True), users, qc, V.TEND)
seent = g.seen(users, qc)
last_t = np.bincount(data.u, minlength=d.nU)  # placeholder
umax = np.full(d.nU, -np.inf); np.maximum.at(umax, data.u, data.t)
dates = ['2017-10-01', '2017-09-01', '2017-08-01', '2017-07-01', '2017-06-01', '2017-05-01', '2017-03-01', '2017-01-01',
         '2016-10-01', '2016-07-01', '2016-04-01', '2016-01-01']
S = {}
for s in dates:
    cut = calendar.timegm(time.strptime(s, '%Y-%m-%d'))
    F, ctx, seen, (gk, us, qck, Tu) = V.task5_pseudo_set(data, cut, rng)
    S[s] = (F, ctx, seen, umax[us] >= cut)
    V.log(f'cut {s}: {F.shape[0]} users, active after cut {np.mean(S[s][3]):.3f}')
def evaluate(train, tag):
    clf = V.fit_pair_classifier(train, 500)
    m = V.pair_margin(clf, Ft[:, 0], Ft[:, 1], ctxt) + 100 * (seent[:, 1] - seent[:, 0])
    new1, new2 = g.icnt[qc[:, 0]] == 0, g.icnt[qc[:, 1]] == 0
    m = np.where(new1 & ~new2, 100.0, np.where(new2 & ~new1, -100.0, m))
    print(f'{name} {tag:34s} true-final acc {np.mean(m > 0):.4f}', flush=True)
four = ['2017-08-01', '2017-05-01', '2017-01-01', '2016-07-01']
evaluate([(S[s][0], S[s][1]) for s in four], '4 cutoffs (v8), all users')
evaluate([(S[s][0][S[s][3]], S[s][1][S[s][3]]) for s in four], '4 cutoffs, active-after-cut only')
evaluate([(S[s][0], S[s][1]) for s in dates], '12 cutoffs, all users')
evaluate([(S[s][0][S[s][3]], S[s][1][S[s][3]]) for s in dates], '12 cutoffs, active-after-cut only')
