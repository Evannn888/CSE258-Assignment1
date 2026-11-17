import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1')
from t6_harness import *
for name in ['tasks_3_4_train.csv', 'tasks_1_2_train.csv', 'task_5_train.csv', 'task_6_train.csv']:
    d = load_raw(name)
    g = Graph(d.u, d.i, d.t, d.r, d.nU, d.nI)
    fut = g.t > A.TEND
    nfu = np.bincount(g.u[fut], minlength=d.nU)
    print(name, 'events>TEND', fut.sum(), 'users with >=1', (nfu > 0).sum(), 'with exactly1', (nfu == 1).sum(),
          'years', np.unique((1970 + g.t[fut] / 31557600).astype(int))[:30])
    if name.startswith('tasks'):
        data, truth = t6_like(d)
        users = data.tests['task_6_test.csv']['user']
        ft = g.t[g.uptr[users + 1] - 1]
        bad = ft > A.TEND
        print('  harness finals > TEND:', bad.sum(), 'of', len(users), ' their delta days:', np.sort(truth[bad] / DAY).round(0)[:20])
        res, raw, Ht = run(data, truth, 'v7', verbose=False)
        p = finish(raw * scale_of(Ht, SCALE), Ht)
        e = np.abs(p - truth)
        print(f'  MAE all {e.mean()/1e6:.3f}M  excluding garbage-final users {e[~bad].mean()/1e6:.3f}M  garbage share {e[bad].sum()/e.sum():.3f}')
        big = truth > 3 * 365 * DAY
        print(f'  users with final gap > 3y: {big.mean():.3f} share of total abs error {e[big].sum()/e.sum():.3f}')
        for lo, hi in [(0, 30), (30, 90), (90, 365), (365, 730), (730, 1e9)]:
            m = (Ht / DAY >= lo) & (Ht / DAY < hi)
            print(f'  H in [{lo},{hi}) n={m.sum()} MAE {e[m].mean()/1e6:.2f}M  truth med {np.median(truth[m])/DAY:.0f}d pred med {np.median(p[m])/DAY:.0f}d share {e[m].sum()/e.sum():.3f}')
