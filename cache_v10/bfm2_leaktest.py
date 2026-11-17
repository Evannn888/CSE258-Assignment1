import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import bfm2
rng = np.random.default_rng(0)
nu, ni, n = 2000, 3000, 80000
key = np.unique(rng.integers(0, nu, n).astype(np.int64) * ni + rng.integers(0, ni, n)); u, i = key // ni, key % ni
p = rng.permutation(len(u)); u, i = u[p], i[p]
y = rng.normal(0, 1, len(u))                   # pure noise: nothing can be learned
ntr = int(0.9 * len(u))
for side in ('none', 'ux', 'ix', 'ux,ix'):
    des = bfm2.Design(ntr, len(u) - ntr)
    su = des.add_slot(u[:ntr], u[ntr:], nu); si = des.add_slot(i[:ntr], i[ntr:], ni)
    rows = np.arange(ntr)
    cu = np.bincount(u[:ntr], minlength=nu).astype(float); ci = np.bincount(i[:ntr], minlength=ni).astype(float)
    if 'ux' in side:
        des.add_bag(su, u[:ntr], i[:ntr], y[:ntr] * cu[u[:ntr]] ** -0.5, ni, row=rows)
    if 'ix' in side:
        des.add_bag(si, i[:ntr], u[:ntr], y[:ntr] * ci[i[:ntr]] ** -0.5, nu, row=rows)
    pred, sd, trace = bfm2.run(des, y[:ntr], y[ntr:], 8, 150, 50, 1, lo=-100.0, hi=100.0)
    print(side, 'train mse', trace[[10, 50, 100, 149], 0].round(3), 'val mse', np.mean((pred - y[ntr:]) ** 2).round(4), 'alpha', trace[-1, 2].round(3))
