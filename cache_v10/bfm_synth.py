import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import bfm
rng = np.random.default_rng(0)
nu, ni, K, n = 3000, 5000, 4, 150000
P = rng.normal(0, 0.5, (nu, K)); Q = rng.normal(0, 0.5, (ni, K)); bu = rng.normal(0, 0.4, nu); bi = rng.normal(0, 0.3, ni)
Yt = rng.normal(0, 0.5, (ni, K))                       # true implicit item factors
u = rng.integers(0, nu, n * 2); i = (rng.zipf(1.3, n * 2) - 1) % ni
key = np.unique(u.astype(np.int64) * ni + i); u, i = key // ni, key % ni
perm = rng.permutation(len(u)); u, i = u[perm], i[perm]
cnt = np.bincount(u, minlength=nu).astype(float)
PY = np.zeros((nu, K)); np.add.at(PY, u, Yt[i]); PY /= np.sqrt(np.maximum(cnt, 1))[:, None]
for use_bag_truth in (0, 1):
    pu = P + (PY if use_bag_truth else 0)
    ytrue = 3.8 + bu[u] + bi[i] + (pu[u] * Q[i]).sum(1)
    y = ytrue + rng.normal(0, np.sqrt(0.5), len(u))
    ntr = int(len(u) * 0.9)
    ent = np.vstack([u[:ntr], nu + i[:ntr]]).astype(np.int64); q_ent = np.vstack([u[ntr:], nu + i[ntr:]]).astype(np.int64)
    eoff = np.array([0, nu, nu + ni], np.int64)
    print('truth with bag' if use_bag_truth else 'truth without bag', 'rows', len(u), 'oracle test mse', np.mean((y[ntr:] - ytrue[ntr:]) ** 2), 'bias-only oracle', np.mean((y[ntr:] - 3.8 - bu[u[ntr:]] - bi[i[ntr:]]) ** 2))
    for bags in ('', 'u'):
        if bags:
            p, r, c = bfm.make_bag(u, i, nu, ni)
            bag_ptr, bag_ent, bag_c, boff = p, r, c, np.array([0, ni, ni], np.int64)
        else:
            bag_ptr, bag_ent, bag_c, boff = np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0), np.array([0, 0, 0], np.int64)
        for Kfit in (4, 16):
            t0 = time.time()
            pred, pred_c, trace = bfm.gibbs(Kfit, 300, 100, 1, y[:ntr].copy(), ent, eoff, bag_ptr, bag_ent, bag_c, boff, q_ent, y[ntr:].copy(), 0.1, -100.0, 100.0, 0, 0)
            print(f'   model bags={bags!r} K={Kfit}: test mse {np.mean((pred - y[ntr:]) ** 2):.4f}  vs truth-mse {np.mean((pred - ytrue[ntr:]) ** 2):.4f}  train mse last {trace[-1, 0]:.4f}  ({time.time() - t0:.0f}s)')
