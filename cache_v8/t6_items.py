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
ct = np.bincount(qi, minlength=d.nI)
top = np.argsort(-ct)[:25]
D = lambda x: str(np.datetime64(int(x), 'D'))
for j in top:
    a, b = g.iptr[j], g.iptr[j + 1]
    tt = g.itimes[a:b]
    tt = tt[(tt > 9e8) & (tt < 1.52e9)]
    q = np.percentile(tt, [0, 10, 50, 90]) if len(tt) else [np.nan]*4
    m = qi == j
    print(f'item {j:6d} test {ct[j]:4d} train {int(g.icnt[j]):5d} ratio {ct[j]/(g.icnt[j]+1):.2f} first {D(q[0]) if len(tt) else "-"} p10 {D(q[1]) if len(tt) else "-"} med {D(q[2]) if len(tt) else "-"} p90 {D(q[3]) if len(tt) else "-"} | users H med {np.median(H[m]):.0f}d tl med {D(np.median(tl[m]))}')
# overall relationship: ratio of test count to train count
r = ct[qi] / (g.icnt[qi] + 1)
print('ratio quantiles', np.percentile(r, [10, 50, 90, 99]))
# relation between ratio and H
for lo, hi in [(0, .05), (.05, .2), (.2, 1), (1, 1e9)]:
    m = (r >= lo) & (r < hi)
    print(f'ratio [{lo},{hi}) n={m.sum()} H median {np.median(H[m]):.0f}d')
