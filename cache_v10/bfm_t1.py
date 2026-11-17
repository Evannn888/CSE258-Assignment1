"""Fold-k check of the Bayesian FM on a rating task: train on the other folds' positive ratings, score fold k."""
import sys, time, numpy as np, argparse
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
import bfm

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=int, default=1); ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--k', type=int, default=32); ap.add_argument('--iters', type=int, default=200); ap.add_argument('--burn', type=int, default=30)
ap.add_argument('--bags', type=str, default='ui'); ap.add_argument('--probit', type=int, default=0); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--time', type=str, default=''); ap.add_argument('--out', type=str, default=''); ap.add_argument('--verbose', type=int, default=10)
ap.add_argument('--power', type=float, default=0.5)
args = ap.parse_args()
task = args.task
D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
a, users, items = V.load(D / V.FILES[task])
nu, ni = len(users), len(items)
tests = V.read_test(D / f'task_{task}_test.csv')
tu = np.array([users[r['user']] for r in tests]); ti = np.array([items[r['item']] for r in tests])
tt = np.array([int(r['timestamp']) for r in tests], np.int64) if task == 3 else np.zeros(len(tests), np.int64)
folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
rowfold = np.empty(len(a), np.int64)
for k, p in enumerate(folds):
    rowfold[p] = k
pos = a[:, 3] > 0
tr = np.flatnonzero(pos & (rowfold != args.fold)); va = np.flatnonzero(pos & (rowfold == args.fold))
y = a[tr, 3].astype(np.float64)
slots_tr = [a[tr, 0], nu + a[tr, 1]]; slots_q = [np.r_[a[va, 0], tu], nu + np.r_[a[va, 1], ti]]
eoff = [0, nu, nu + ni]
qt = np.r_[a[va, 2], tt]
if args.time:                      # extra one-hot slots built from the timestamp (Task 3)
    for kind in args.time.split(','):
        def code(t, kind=kind):
            t = np.clip(t, 1.0e9, V.TEND)
            if kind == 'month':
                return ((t - 1.0e9) // (30.44 * 86400)).astype(np.int64)
            if kind == 'quarter':
                return ((t - 1.0e9) // (91.3 * 86400)).astype(np.int64)
            if kind == 'year':
                return ((t - 1.0e9) // (365.25 * 86400)).astype(np.int64)
            raise ValueError(kind)
        ctr, cq = code(a[tr, 2]), code(qt)
        m = int(max(ctr.max(), cq.max())) + 1
        slots_tr.append(eoff[-1] + ctr); slots_q.append(eoff[-1] + cq); eoff.append(eoff[-1] + m)
S = len(slots_tr)
ent = np.ascontiguousarray(np.vstack(slots_tr).astype(np.int64)); q_ent = np.ascontiguousarray(np.vstack(slots_q).astype(np.int64))
q_y = np.r_[a[va, 3].astype(np.float64), np.full(len(tu), np.nan)]
allu, alli = np.r_[a[:, 0], tu], np.r_[a[:, 1], ti]            # every known interaction (existence only)
ptrs, ents, cs, boff = [np.zeros(1, np.int64)], [], [], [0]
if 'u' in args.bags:               # user slot: bag of the user's items
    p, r, c = bfm.make_bag(allu, alli, nu, ni, args.power)
    ptrs.append(p[1:] + ptrs[-1][-1]); ents.append(r); cs.append(c); boff.append(boff[-1] + ni)
else:
    boff.append(boff[-1])
if 'i' in args.bags:               # item slot: bag of the item's users
    p, r, c = bfm.make_bag(alli, allu, ni, nu, args.power)
    ptrs.append(p[1:] + ptrs[-1][-1]); ents.append(nu + r); cs.append(c); boff.append(boff[-1] + nu)
else:
    boff.append(boff[-1])
for _ in range(S - 2):
    boff.append(boff[-1])
bag_ptr = np.concatenate(ptrs); bag_ent = np.concatenate(ents) if ents else np.zeros(0, np.int64); bag_c = np.concatenate(cs) if cs else np.zeros(0)
V.log(f'task {task} fold {args.fold}: train {len(tr):,}, val {len(va):,}, slots {S}, K {args.k}, bags {args.bags!r}, probit {args.probit}')
t0 = time.time()
pred, pred_c, trace = bfm.gibbs(args.k, args.iters, args.burn, args.seed, y, ent, np.array(eoff, np.int64), bag_ptr, bag_ent, bag_c,
                                np.array(boff, np.int64), q_ent, q_y, 0.1, 1.0, 5.0, args.probit, args.verbose)
nv = len(va)
mse = np.mean((np.clip(pred[:nv], 1, 5) - q_y[:nv]) ** 2); mse_c = np.mean((pred_c[:nv] - q_y[:nv]) ** 2)
V.log(f'done in {time.time() - t0:.0f}s: val MSE {mse:.5f} (avg of clipped samples {mse_c:.5f})')
if args.out:
    np.savez(args.out, pred=pred, pred_c=pred_c, trace=trace, va=va)
