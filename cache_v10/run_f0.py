"""Fold-0-only run of the production GPU sampler (for quick comparisons with the CPU fold-0 runs)."""
import sys, time, numpy as np, torch, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('a1v11f', '/data/UCSD/CSE258/A1/v11/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v11f'] = M; spec.loader.exec_module(M)
task, k, iters, burn, probit, tm, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), bool(int(sys.argv[5])), sys.argv[6], int(sys.argv[7])
nu_tau = float(sys.argv[8]) if len(sys.argv) > 8 else 30.0
M._bfm_torch(); torch.use_deterministic_algorithms(False)
a, nu, ni, test_rows, folds = M.rating_problem(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
rowfold = np.empty(len(a), np.int64)
for kk, p in enumerate(folds):
    rowfold[p] = kk
pos = np.flatnonzero(a[:, 3] > 0)
u, i, t, y = a[pos, 0], a[pos, 1], a[pos, 2], a[pos, 3].astype(np.float64)
mask = (rowfold[pos] != 0).astype(np.float32)[None]
Q = np.vstack([a[folds[0]], test_rows])[None]
nU, nI = nu + 1, ni + 1
au, ai = np.r_[a[:, 0], test_rows[:, 0]], np.r_[a[:, 1], test_rows[:, 1]]
pu, eu, cu = M.bfm_bag(au, ai, nU, nI); pi, ei, ci = M.bfm_bag(ai, au, nI, nU)
feat = lambda ptr: np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
slots = [dict(ent=u, q=Q[:, :, 0], n=nU, kind=1, bag=(feat(pu), eu, cu, nI)), dict(ent=i, q=Q[:, :, 1], n=nI, kind=1, bag=(feat(pi), ei, ci, nU))]
for kind in [x for x in tm.split(',') if x and x != 'none']:
    tr, q, m, kd = M.bfm_time_slot(kind, u, t, Q[:, :, 0].reshape(-1), Q[:, :, 2].reshape(-1))
    slots.append(dict(ent=tr, q=q.reshape(1, -1), n=m, kind=kd))
t0 = time.time()
model = M.BayesianFM(y, mask, slots, k, probit=probit, nu_tau=nu_tau, seed=seed)
mean = model.run(iters, burn)[0]
yy = a[folds[0], 3]; ok = yy > 0
print(f'GPU fold 0 task {task} K{k} iters {iters} probit {int(probit)} time {tm} nu {nu_tau} seed {seed}: MSE {np.mean((np.clip(mean[:len(yy)][ok], 1, 5) - yy[ok]) ** 2):.5f} ({time.time() - t0:.0f}s)', flush=True)
