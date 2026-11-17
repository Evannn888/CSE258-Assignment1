"""Run the batched GPU Bayesian FM for a rating task on a set of folds; saves chains in the production cache format
(posterior mean for all rows of fold k, then the test rows)."""
import sys, time, argparse, numpy as np, torch, importlib.util
from pathlib import Path
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import bfm_gpu
spec = importlib.util.spec_from_file_location('a1v10g', '/data/UCSD/CSE258/A1/v10/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v10g'] = M; spec.loader.exec_module(M)
ap = argparse.ArgumentParser()
ap.add_argument('--task', type=int, default=1); ap.add_argument('--folds', type=str, default='all')
ap.add_argument('--k', type=int, default=32); ap.add_argument('--iters', type=int, default=500); ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--probit', type=int, default=0); ap.add_argument('--nu', type=float, default=30.0); ap.add_argument('--time', type=str, default='')
ap.add_argument('--cg', type=int, default=12); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', type=str, default='')
ap.add_argument('--verbose', type=int, default=50)
args = ap.parse_args()
task = args.task
a, nu, ni, test_rows, folds = M.rating_problem(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
fl = list(range(M.RATING_FOLDS)) if args.folds == 'all' else [int(x) for x in args.folds.split(',')]
F = len(fl)
rowfold = np.empty(len(a), np.int64)
for k, p in enumerate(folds):
    rowfold[p] = k
pos = np.flatnonzero(a[:, 3] > 0)
u, i, t, y = a[pos, 0], a[pos, 1], a[pos, 2], a[pos, 3].astype(np.float64)
mask = np.stack([(rowfold[pos] != k).astype(np.float32) for k in fl])
L = max(len(folds[k]) for k in fl) + len(test_rows)
qrows = []
for k in fl:
    q = np.vstack([a[folds[k]], test_rows])
    qrows.append(np.vstack([q, np.repeat(q[-1:], L - len(q), 0)]))
Q = np.stack(qrows)                                           # (F, L, 4)
nU, nI = nu + 1, ni + 1
au, ai = np.r_[a[:, 0], test_rows[:, 0]], np.r_[a[:, 1], test_rows[:, 1]]
pu, eu, cu = M.bfm_bag(au, ai, nU, nI)                         # feature (item) -> users
pi, ei, ci = M.bfm_bag(ai, au, nI, nU)                         # feature (user) -> items
feat_of = lambda ptr: np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
slots = [dict(ent=u, q=Q[:, :, 0], n=nU, kind=1, bag=(feat_of(pu), eu, cu, nI)),
         dict(ent=i, q=Q[:, :, 1], n=nI, kind=1, bag=(feat_of(pi), ei, ci, nU))]
for name in [x for x in args.time.split(',') if x]:
    a_, b_, m, kd = M.bfm_time_slot(name, u, t, Q[:, :, 0].reshape(-1), Q[:, :, 2].reshape(-1))
    slots.append(dict(ent=a_, q=b_.reshape(F, L), n=m, kind=kd))
qy = np.full((F, L), np.nan)
for r, k in enumerate(fl):
    yy = a[folds[k], 3].astype(float)
    qy[r, :len(yy)] = np.where(yy > 0, yy, np.nan)
t0 = time.time()
model = bfm_gpu.BFMGPU(y, mask, slots, args.k, probit=bool(args.probit), nu_tau=args.nu, seed=args.seed, cg_iters=args.cg)
mean, sd = model.run(args.iters, args.burn, verbose=args.verbose, q_y=torch.as_tensor(qy, dtype=torch.float32, device='cuda'))
el = time.time() - t0
mses = []
for r, k in enumerate(fl):
    yy = a[folds[k], 3]; ok = yy > 0
    mses.append(np.mean((np.clip(mean[r, :len(yy)][ok], 1, 5) - yy[ok]) ** 2))
    if args.tag:
        out = Path(f'/data/UCSD/CSE258/A1/cache_v10/gpu_chains/{args.tag}'); out.mkdir(parents=True, exist_ok=True)
        np.save(out / f't{task}_f{k}.npy', mean[r, :len(folds[k]) + len(test_rows)].astype(np.float32))
M.log(f'task {task} K{args.k} iters {args.iters} probit {args.probit} time {args.time!r} cg {args.cg}: folds {fl[:5]}{"..." if F > 5 else ""} '
      f'MSE per fold {np.round(mses, 5)[:6]} mean {np.mean(mses):.5f} ({el:.0f}s)')
