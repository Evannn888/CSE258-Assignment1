"""Bayesian FM v2 on one fold of a rating task.  --feat: i=implicit bags, x=explicit (rating-weighted, self-excluded) bags,
on the user side (u..) and/or the item side (i..), e.g. 'ui,ii,ux,ix'.  --time adds one-hot time slots (Task 3)."""
import sys, time, numpy as np, argparse
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import assignment1 as V
from pathlib import Path
import bfm2, bfm3

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=int, default=1); ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--k', type=int, default=8); ap.add_argument('--iters', type=int, default=300); ap.add_argument('--burn', type=int, default=60)
ap.add_argument('--feat', type=str, default='ui,ii'); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--time', type=str, default=''); ap.add_argument('--tag', type=str, default=''); ap.add_argument('--verbose', type=int, default=0)
ap.add_argument('--center', type=str, default='cross'); ap.add_argument('--probit', type=int, default=0); ap.add_argument('--hetero', type=int, default=0); ap.add_argument('--nu', type=float, default=10.0)
args = ap.parse_args()
task = args.task
D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
a, users, items = V.load(D / V.FILES[task])
nu, ni = len(users), len(items)
tests = V.read_test(D / f'task_{task}_test.csv')
tu = np.array([users[r['user']] for r in tests]); ti = np.array([items[r['item']] for r in tests])
tt = np.array([int(r['timestamp']) for r in tests], np.int64) if task == 3 else np.zeros(len(tests), np.int64)
folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
part = folds[args.fold]
va = part[a[part, 3] > 0]
keep = np.ones(len(a), bool); keep[part] = False
tr = np.flatnonzero(keep & (a[:, 3] > 0))
y = a[tr, 3].astype(np.float64)
u, i = a[tr, 0], a[tr, 1]
qu, qi, qt = np.r_[a[va, 0], tu], np.r_[a[va, 1], ti], np.r_[a[va, 2], tt]
q_y = np.r_[a[va, 3].astype(np.float64), np.full(len(tu), np.nan)]
des = bfm2.Design(len(tr), len(qu))
su = des.add_slot(u, qu, nu); si = des.add_slot(i, qi, ni)
feats = [f for f in args.feat.split(',') if f]
allu, alli = np.r_[a[:, 0], tu], np.r_[a[:, 1], ti]                       # every known interaction (existence only)
key = np.unique(allu.astype(np.int64) * ni + alli); pu_, pi_ = key // ni, key % ni
cu, ci = np.bincount(pu_, minlength=nu).astype(float), np.bincount(pi_, minlength=ni).astype(float)
if 'ui' in feats:
    des.add_bag(su, pu_, pi_, cu[pu_] ** -0.5, ni)
if 'ii' in feats:
    des.add_bag(si, pi_, pu_, ci[pi_] ** -0.5, nu)
# shrunk biases on the training rows for the explicit coefficients
mu = y.mean()
nru, nri = np.bincount(u, minlength=nu).astype(float), np.bincount(i, minlength=ni).astype(float)
bu, bi = np.zeros(nu), np.zeros(ni)
for _ in range(15):
    bi = np.bincount(i, weights=y - mu - bu[u], minlength=ni) / (nri + 5.0)
    bu = np.bincount(u, weights=y - mu - bi[i], minlength=nu) / (nru + 3.0)
rows = np.arange(len(tr))
if args.center == 'cross':          # user-side bag: item-centred ratings; item-side bag: user-centred ratings (no target leak)
    res_u, res_i = y - mu - bi[i], y - mu - bu[u]
elif args.center == 'raw':
    res_u = res_i = y - mu
if 'ux' in feats:
    des.add_bag(su, u, i, res_u * np.maximum(nru[u], 1) ** -0.5, ni, row=rows)
if 'ix' in feats:
    des.add_bag(si, i, u, res_i * np.maximum(nri[i], 1) ** -0.5, nu, row=rows)
if args.time:
    for kind in args.time.split(','):
        span = {'month': 30.44, 'quarter': 91.3, 'year': 365.25}[kind]
        code = lambda t: ((np.clip(t, 1.0e9, V.TEND) - 1.0e9) // (span * 86400)).astype(np.int64)
        ctr, cq = code(a[tr, 2]), code(qt)
        des.add_slot(ctr, cq, int(max(ctr.max(), cq.max())) + 1)
V.log(f'task {task} fold {args.fold}: train {len(tr):,}, val {len(va):,}, K {args.k}, feat {args.feat!r}, time {args.time!r}')
t0 = time.time()
pred, sd, lat, trace = bfm3.run(des, y, q_y, args.k, args.iters, args.burn, args.seed + 1000 * args.fold, probit=args.probit, hetero=args.hetero, nu_tau=args.nu, verbose=args.verbose)
nv = len(va)
mse = np.mean((np.clip(pred[:nv], 1, 5) - q_y[:nv]) ** 2)
V.log(f'task {task} {args.tag} fold {args.fold}: val MSE {mse:.5f}, train mse {trace[-1, 0]:.4f}, alpha {trace[-1, 2]:.3f} ({time.time() - t0:.0f}s)')
if args.tag:
    np.savez(f'/data/UCSD/CSE258/A1/cache_v10/oof_{args.tag}_t{task}_f{args.fold}.npz', oof=pred[:nv].astype(np.float32), test=pred[nv:].astype(np.float32),
             oof_sd=sd[:nv].astype(np.float32), test_sd=sd[nv:].astype(np.float32), oof_lat=lat[:nv].astype(np.float32), test_lat=lat[nv:].astype(np.float32), mse=mse, va=va, pred=pred, trace=trace)
