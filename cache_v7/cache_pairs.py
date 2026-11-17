"""Cache v6 Task 2/4 folds: held rows, candidates, v6 features (incl. CPU latent scores), test features."""
import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
if __name__ == '__main__':
    task = int(sys.argv[1]); out = Path('/data/UCSD/CSE258/A1/cache_v7')
    D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
    tests = [f'task_{k}_test.csv' for k in V.FILES if V.FILES[k] == V.FILES[task]]
    data = V.Data(D, V.FILES[task], tests)
    rng = np.random.default_rng(V.SEED + task)
    perm = rng.permutation(len(data.u))
    sets, problems, save = [], [], {}
    for f in range(4):
        held = perm[f * 30000:(f + 1) * 30000]
        keep = np.ones(len(data.u), bool); keep[held] = False
        g = V.Graph(data.u[keep], data.i[keep], data.t[keep], data.r[keep], data.nU, data.nI)
        sv = V.Spectral(g)
        qc = np.column_stack([data.i[held], V.sample_negatives(data.i, data.i[held], 2, rng)])
        qt = data.t[held] if task == 4 else np.zeros(len(held))
        F, ctx = V.pair_features(g, sv, task, data.u[held], qc, qt)
        save.update({f'held{f}': held, f'qc{f}': qc, f'qt{f}': qt, f'F{f}': F, f'ctx{f}': ctx, f'seen{f}': g.seen(data.u[held], qc)})
        problems.append((g, data.u[held], qc, qt))
        V.log(f'fold {f} features')
    g = V.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    qu, qc, te = V.test_pairs(data, f'task_{task}_test.csv')
    qt = te['timestamp'] if task == 4 else np.zeros(len(qu))
    problems.append((g, qu, qc, qt))
    lat = V.latent_scores(task, problems)
    for f in range(4):
        save[f'L{f}'] = lat[f]
    F, ctx = V.pair_features(g, V.Spectral(g), task, qu, qc, qt)
    save.update(dict(testF=F, testctx=ctx, testL=lat[-1], testseen=g.seen(qu, qc), testqu=qu, testqc=qc, testqt=qt,
                     u=data.u, i=data.i, t=data.t, r=data.r, nU=data.nU, nI=data.nI,
                     other_test_items=data.tests['task_1_test.csv' if task == 2 else 'task_3_test.csv']['item']))
    np.savez(out / f'pairs_t{task}.npz', **save)
    V.log('saved')
