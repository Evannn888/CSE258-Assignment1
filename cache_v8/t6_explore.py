import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as A
from pathlib import Path
d = A.Data(Path('assignment1_2026'), 'task_6_train.csv', ['task_6_test.csv'])
g = A.Graph(d.u, d.i, d.t, d.r, d.nU, d.nI)
te = d.tests['task_6_test.csv']
qu, qi = te['user'], te['item']
tl = g.last_time(qu)
H = (A.TEND - tl) / A.DAY
print('test users', len(qu), 'unique', len(np.unique(qu)), 'train users', d.nU)
print('ucnt quantiles', np.percentile(g.ucnt[qu], [0, 10, 25, 50, 75, 90, 99]))
print('H days quantiles', np.percentile(H, [0, 1, 5, 10, 25, 50, 75, 90, 99]))
for lim in [1, 7, 30, 90, 365, 730, 1460]:
    print(f'frac H<{lim}d', np.mean(H < lim).round(4))
# time distribution overall
t = g.t
print('global time quantiles', [np.datetime64(int(x), 's') for x in np.percentile(t[(t > 9e8)], [1, 10, 50, 90, 99, 99.9])])
print('max t', np.datetime64(int(t.max()), 's'), 'n > TEND', np.sum(t > A.TEND))
# L1 targets
ends = g.uptr[1:]
ok = g.ucnt >= 3
lt = g.t[ends[ok] - 1] - g.t[ends[ok] - 2]
print('L1 gap days quantiles', np.percentile(lt / A.DAY, [10, 25, 50, 75, 90]))
print('L1 zero-gap frac', np.mean(lt < 86400))
# test item seen in train?
print('test item unseen frac', np.mean(g.icnt[qi] == 0), 'test item in own history', np.mean([qi[k] in set(g.i[g.uptr[u]:g.uptr[u+1]]) for k, u in enumerate(qu[:5000])]))
# test item frequency among test
cnt_test = np.bincount(qi, minlength=d.nI)
print('test item count quantiles', np.percentile(cnt_test[qi], [10, 50, 90, 99]))
