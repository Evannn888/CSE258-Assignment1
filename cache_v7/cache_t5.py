"""Cache v6 Task 5 pseudo sets (+ test): features, CPU latent scores, history arrays for GPU models."""
import sys, numpy as np, calendar, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
if __name__ == '__main__':
    data = V.Data(Path('/data/UCSD/CSE258/A1/assignment1_2026'), 'task_5_train.csv', ['task_5_test.csv'])
    rng = np.random.default_rng(V.SEED + 5)
    cuts = [None] + [calendar.timegm(time.strptime(s, '%Y-%m-%d')) for s in ('2017-08-01', '2017-05-01', '2017-01-01', '2016-07-01')]
    sets = [V.task5_pseudo_set(data, cut, rng) for cut in cuts]
    g = V.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    qu, qc, _ = V.test_pairs(data, 'task_5_test.csv')
    lat = V.latent_scores(5, [s[3] for s in sets] + [(g, qu, qc, g.last_time(qu))])
    save = {}
    for k, (F, ctx, seen, (gk, users, qck, Tu)) in enumerate(sets):
        save.update({f'F{k}': F, f'ctx{k}': ctx, f'seen{k}': seen, f'L{k}': lat[k], f'users{k}': users, f'qc{k}': qck, f'Tu{k}': Tu,
                     f'hu{k}': gk.u, f'hi{k}': gk.i, f'ht{k}': gk.t})
    F, ctx = V.task5_features(g, V.Spectral(g, recency=True), qu, qc, V.TEND)
    save.update(dict(testF=F, testctx=ctx, testL=lat[-1], testseen=g.seen(qu, qc), testqu=qu, testqc=qc, testTu=g.last_time(qu),
                     fu=g.u, fi=g.i, ft=g.t, nU=data.nU, nI=data.nI, icnt=g.icnt))
    np.savez('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz', **save)
    V.log('saved')
