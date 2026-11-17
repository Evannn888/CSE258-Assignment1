import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as A
from pathlib import Path
for f in ['tasks_1_2_train.csv', 'tasks_3_4_train.csv', 'task_5_train.csv', 'task_6_train.csv']:
    d = A.Data(Path('assignment1_2026'), f, [])
    g = A.Graph(d.u, d.i, d.t, d.r, d.nU, d.nI)
    c = g.ucnt
    last = g.t[g.uptr[1:] - 1]
    ok = c >= 2
    H = (A.TEND - g.t[g.uptr[1:][ok] - 2]) / A.DAY
    fg = (last[ok] - g.t[g.uptr[1:][ok] - 2]) / A.DAY
    print(f, 'users', d.nU, 'ucnt q', np.percentile(c, [0, 10, 25, 50, 75, 90]).round(1), 'frac>=5', np.mean(c >= 5).round(3))
    print('   last-gap days q', np.percentile(fg, [10, 25, 50, 75, 90]).round(1), ' H(prev) q', np.percentile(H, [10, 25, 50, 75, 90]).round(0))
    print('   last time q', [str(np.datetime64(int(x), 's'))[:10] for x in np.percentile(last, [10, 50, 90, 99])])
