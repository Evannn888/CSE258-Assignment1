"""Task-5 design harness on a file with complete histories: the true FINAL event is the positive.
Compares which pseudo-sets (last-event set, cutoff sets) train a classifier that is best on true finals.
Only base task-5 features (no latent/SASRec) to keep it fast; used to choose the design, never for training."""
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
# test-like set: positive = true final event, one popularity negative from the remaining interactions
g = V.Graph(data.u, data.i, data.t, data.r, d.nU, d.nI)
qc = np.column_stack([gfull.i[fin], V.sample_negatives(data.i, gfull.i[fin], 1, rng)])
sv = V.Spectral(g, recency=True)
Ft, ctxt = V.task5_features(g, sv, users, qc, V.TEND)
seent = g.seen(users, qc)
V.log(f'{name}: true-final test set {len(users)} users')
cuts = [None] + [calendar.timegm(time.strptime(s, '%Y-%m-%d')) for s in ('2017-08-01', '2017-05-01', '2017-01-01', '2016-07-01')]
sets = []
for cut in cuts:
    F, ctx, seen, _ = V.task5_pseudo_set(data, cut, rng)
    sets.append((F, ctx, seen))
    V.log(f'pseudo set {cut}: {F.shape}')

def evaluate(train_sets, tag):
    clf = V.fit_pair_classifier([(F, c) for F, c, _ in train_sets], 500)
    m = V.pair_margin(clf, Ft[:, 0], Ft[:, 1], ctxt) + 100 * (seent[:, 1] - seent[:, 0])
    new1, new2 = g.icnt[qc[:, 0]] == 0, g.icnt[qc[:, 1]] == 0
    m = np.where(new1 & ~new2, 100.0, np.where(new2 & ~new1, -100.0, m))
    F0, c0, s0 = sets[0]
    m0 = V.pair_margin(clf, F0[:, 0], F0[:, 1], c0) + 100 * (s0[:, 1] - s0[:, 0])
    print(f'{name} {tag:28s} true-final acc {np.mean(m > 0):.4f} | last-event set acc {np.mean(m0 > 0):.4f}', flush=True)

evaluate(sets[1:], 'cutoff sets 1-4 (v8 validation)')
evaluate(sets, 'all sets 0-4 (v8 final)')
evaluate(sets[:1], 'last-event set 0 only')
