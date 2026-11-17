"""One fold of the out-of-fold Bayesian-FM predictions for a rating task (rows in the order of cache_v7/t{task}_meta.npz).
Saves oof (the fold's positive-rating rows) and test predictions."""
import sys, time, numpy as np, argparse
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import assignment1 as V
from pathlib import Path
import bfm

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=int, default=1); ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--k', type=int, default=32); ap.add_argument('--iters', type=int, default=200); ap.add_argument('--burn', type=int, default=30)
ap.add_argument('--bags', type=str, default='ui'); ap.add_argument('--probit', type=int, default=0); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--time', type=str, default=''); ap.add_argument('--tag', type=str, required=True); ap.add_argument('--power', type=float, default=0.5)
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
va = part[a[part, 3] > 0]                                     # same order as the bank rows of this fold
keep = np.ones(len(a), bool); keep[part] = False
tr = np.flatnonzero(keep & (a[:, 3] > 0))
y = a[tr, 3].astype(np.float64)
slots_tr = [a[tr, 0], nu + a[tr, 1]]; slots_q = [np.r_[a[va, 0], tu], nu + np.r_[a[va, 1], ti]]
eoff = [0, nu, nu + ni]
qt = np.r_[a[va, 2], tt]
if args.time:
    for kind in args.time.split(','):
        def code(t, kind=kind):
            t = np.clip(t, 1.0e9, V.TEND)
            span = {'month': 30.44, 'quarter': 91.3, 'year': 365.25}[kind]
            return ((t - 1.0e9) // (span * 86400)).astype(np.int64)
        ctr, cq = code(a[tr, 2]), code(qt)
        m = int(max(ctr.max(), cq.max())) + 1
        slots_tr.append(eoff[-1] + ctr); slots_q.append(eoff[-1] + cq); eoff.append(eoff[-1] + m)
S = len(slots_tr)
ent = np.ascontiguousarray(np.vstack(slots_tr).astype(np.int64)); q_ent = np.ascontiguousarray(np.vstack(slots_q).astype(np.int64))
q_y = np.r_[a[va, 3].astype(np.float64), np.full(len(tu), np.nan)]
allu, alli = np.r_[a[:, 0], tu], np.r_[a[:, 1], ti]
ptrs, ents, cs, boff = [np.zeros(1, np.int64)], [], [], [0]
if 'u' in args.bags:
    p, r, c = bfm.make_bag(allu, alli, nu, ni, args.power)
    ptrs.append(p[1:] + ptrs[-1][-1]); ents.append(r); cs.append(c); boff.append(boff[-1] + ni)
else:
    boff.append(boff[-1])
if 'i' in args.bags:
    p, r, c = bfm.make_bag(alli, allu, ni, nu, args.power)
    ptrs.append(p[1:] + ptrs[-1][-1]); ents.append(nu + r); cs.append(c); boff.append(boff[-1] + nu)
else:
    boff.append(boff[-1])
for _ in range(S - 2):
    boff.append(boff[-1])
bag_ptr = np.concatenate(ptrs); bag_ent = np.concatenate(ents) if ents else np.zeros(0, np.int64); bag_c = np.concatenate(cs) if cs else np.zeros(0)
t0 = time.time()
pred, pred_c, trace = bfm.gibbs(args.k, args.iters, args.burn, args.seed + 1000 * args.fold, y, ent, np.array(eoff, np.int64), bag_ptr, bag_ent, bag_c,
                                np.array(boff, np.int64), q_ent, q_y, 0.1, 1.0, 5.0, args.probit, 0)
nv = len(va)
mse = np.mean((np.clip(pred[:nv], 1, 5) - q_y[:nv]) ** 2)
V.log(f'task {task} {args.tag} fold {args.fold}: val MSE {mse:.5f} ({time.time() - t0:.0f}s)')
np.savez(f'/data/UCSD/CSE258/A1/cache_v10/oof_{args.tag}_t{task}_f{args.fold}.npz', oof=pred[:nv].astype(np.float32), test=pred[nv:].astype(np.float32),
         oof_c=pred_c[:nv].astype(np.float32), test_c=pred_c[nv:].astype(np.float32), mse=mse)
