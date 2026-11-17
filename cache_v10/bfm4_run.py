"""Bayesian FM v4 on one fold of a rating task.
--model std : user ids + item ids (+ implicit bags)                       ("Bayesian SVD++ flipped")
--model U   : no user parameters; the user is the bag of his/her rated items (rating-weighted, leave-one-out) + implicit bag;
              items have ids (+ implicit bag)                              ("Bayesian asymmetric SVD", item-based neighbourhood)
--model I   : mirror image: no item parameters; the item is the bag of its raters (user-based neighbourhood)
--time adds one-hot time slots (Task 3)."""
import sys, time, numpy as np, argparse
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import assignment1 as V
from pathlib import Path
import bfm4

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=int, default=1); ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--k', type=int, default=8); ap.add_argument('--iters', type=int, default=300); ap.add_argument('--burn', type=int, default=60)
ap.add_argument('--model', type=str, default='std'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--imp', type=int, default=1)
ap.add_argument('--time', type=str, default=''); ap.add_argument('--tag', type=str, default=''); ap.add_argument('--verbose', type=int, default=0)
ap.add_argument('--probit', type=int, default=0); ap.add_argument('--hetero', type=int, default=1); ap.add_argument('--nu', type=float, default=10.0)
ap.add_argument('--mean', type=int, default=1); ap.add_argument('--full', type=int, default=0); ap.add_argument('--ctx', type=str, default=''); ap.add_argument('--thr', type=float, default=0.0)
args = ap.parse_args()
task = args.task
D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
a, users, items = V.load(D / V.FILES[task])
nu, ni = len(users), len(items)
tests = V.read_test(D / f'task_{task}_test.csv')
tu = np.array([users[r['user']] for r in tests]); ti = np.array([items[r['item']] for r in tests])
tt = np.array([int(r['timestamp']) for r in tests], np.int64) if task == 3 else np.zeros(len(tests), np.int64)
folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
if args.full:                                     # train on every rating, predict the test rows only
    va = np.zeros(0, np.int64); tr = np.flatnonzero(a[:, 3] > 0)
else:
    part = folds[args.fold]
    va = part[a[part, 3] > 0]
    keep = np.ones(len(a), bool); keep[part] = False
    tr = np.flatnonzero(keep & (a[:, 3] > 0))
y = a[tr, 3].astype(np.float64)
u, i = a[tr, 0], a[tr, 1]
qu, qi, qt = np.r_[a[va, 0], tu], np.r_[a[va, 1], ti], np.r_[a[va, 2], tt]
q_y = np.r_[a[va, 3].astype(np.float64), np.full(len(tu), np.nan)]
des = bfm4.Design(len(tr), len(qu))
su = des.add_slot(u, qu, nu, ids=args.model != 'U'); si = des.add_slot(i, qi, ni, ids=args.model != 'I')
allu, alli = np.r_[a[:, 0], tu], np.r_[a[:, 1], ti]                       # every known interaction (existence only)
key = np.unique(allu.astype(np.int64) * ni + alli); pu_, pi_ = key // ni, key % ni
cu, ci = np.bincount(pu_, minlength=nu).astype(float), np.bincount(pi_, minlength=ni).astype(float)
if args.imp:
    des.add_bag(su, pu_, pi_, cu[pu_] ** -0.5, ni)
    des.add_bag(si, pi_, pu_, ci[pi_] ** -0.5, nu)
mu = y.mean()
nru, nri = np.bincount(u, minlength=nu).astype(float), np.bincount(i, minlength=ni).astype(float)
bu, bi = np.zeros(nu), np.zeros(ni)
for _ in range(15):
    bi = np.bincount(i, weights=y - mu - bu[u], minlength=ni) / (nri + 5.0)
    bu = np.bincount(u, weights=y - mu - bi[i], minlength=nu) / (nru + 3.0)
rows = np.arange(len(tr))
if args.model == 'U':
    res = y - mu - bi[i]                          # item-centred: carries the user's own level
    des.add_bag(su, u, i, res * nru[u] ** -0.5, ni, row=rows)
    if args.mean:
        des.add_bag(su, u, np.zeros(len(tr), np.int64), res / (nru[u] + 3.0), 1, row=rows)
if args.model == 'I':
    res = y - mu - bu[u]                          # user-centred: carries the item's own level
    des.add_bag(si, i, u, res * nri[i] ** -0.5, nu, row=rows)
    if args.mean:
        des.add_bag(si, i, np.zeros(len(tr), np.int64), res / (nri[i] + 5.0), 1, row=rows)
if args.ctx:                                   # entity-level attribute slots (shared factors for similar users / items)
    tr_all = np.r_[a[:, 2].astype(np.float64), np.full(len(tu), np.nan)]
    tclip = np.where((tr_all > 1.0e9) & (tr_all < V.TEND + 86400), tr_all, np.nan)
    def first_year(ids, n):
        f = np.full(n, np.inf); ok = ~np.isnan(tclip); np.minimum.at(f, ids[ok], tclip[ok])
        yr = np.where(np.isfinite(f), (f - 1.0e9) // (365.25 * 86400), -1).astype(np.int64) + 1
        return yr
    def med_year(ids, n):
        ok = ~np.isnan(tclip); o = np.lexsort((tclip[ok], ids[ok])); ii = ids[ok][o]; tt_ = tclip[ok][o]
        ptr = np.searchsorted(ii, np.arange(n + 1)); out = np.zeros(n, np.int64)
        for r in range(n):
            if ptr[r + 1] > ptr[r]:
                out[r] = int((tt_[(ptr[r] + ptr[r + 1]) // 2] - 1.0e9) // (365.25 * 86400)) + 1
        return out
    for kind in args.ctx.split(','):
        if kind == 'ubin':
            code = np.minimum(np.log2(np.maximum(cu, 1)).astype(np.int64), 9); src, qsrc = u, qu
        elif kind == 'ibin':
            code = np.minimum(np.log2(np.maximum(ci, 1)).astype(np.int64), 9); src, qsrc = i, qi
        elif kind == 'ucohort':
            code = first_year(allu, nu); src, qsrc = u, qu
        elif kind == 'icohort':
            code = first_year(alli, ni); src, qsrc = i, qi
        elif kind == 'umed':
            code = med_year(allu, nu); src, qsrc = u, qu
        elif kind == 'imed':
            code = med_year(alli, ni); src, qsrc = i, qi
        else:
            raise ValueError(kind)
        des.add_slot(code[src], code[qsrc], int(code.max()) + 1, ids=1)
if args.time:
    T0 = 1.0e9
    clip = lambda t: np.clip(t.astype(np.float64), T0, V.TEND)
    ttr, tq, tall = clip(a[tr, 2]), clip(qt), clip(np.r_[a[:, 2], tt])
    day = lambda t: np.round((t - 27000.0) / 86400.0).astype(np.int64)           # timestamps are dates (Pacific midnight +- 5000 s)
    ufirst = np.full(nu, np.inf); np.minimum.at(ufirst, allu, tall)
    ifirst = np.full(ni, np.inf); np.minimum.at(ifirst, alli, tall)
    edges = np.array([0.5, 1.5, 7.5, 30.5, 90.5, 180.5, 365.5, 730.5, 1460.5])
    def pair_code(x_tr, y_tr, x_q, y_q):
        k = np.r_[x_tr, x_q].astype(np.int64) * 100003 + np.r_[y_tr, y_q].astype(np.int64)
        uq, inv = np.unique(k, return_inverse=True)
        return inv[:len(x_tr)], inv[len(x_tr):], len(uq)
    for kind in args.time.split(','):
        ids = 1
        if kind in ('month', 'quarter', 'year'):
            span = {'month': 30.44, 'quarter': 91.3, 'year': 365.25}[kind]
            ctr, cq = ((ttr - T0) // (span * 86400)).astype(np.int64), ((tq - T0) // (span * 86400)).astype(np.int64)
            m = int(max(ctr.max(), cq.max())) + 1
        elif kind in ('session', 'sessionf'):                 # (user, day): bias only / bias + factors
            ctr, cq, m = pair_code(u, day(ttr), qu, day(tq)); ids = 2 if kind == 'session' else 1
        elif kind in ('uyear', 'uyearf'):                     # (user, year)
            ctr, cq, m = pair_code(u, (ttr - T0) // (365.25 * 86400), qu, (tq - T0) // (365.25 * 86400)); ids = 2 if kind == 'uyear' else 1
        elif kind in ('iyear', 'iyearf'):                     # (item, year)
            ctr, cq, m = pair_code(i, (ttr - T0) // (365.25 * 86400), qi, (tq - T0) // (365.25 * 86400)); ids = 2 if kind == 'iyear' else 1
        elif kind == 'tenure':                                # days since the user's first known event
            ctr, cq, m = np.searchsorted(edges, (ttr - ufirst[u]) / 86400.0), np.searchsorted(edges, (tq - ufirst[qu]) / 86400.0), len(edges) + 1
        elif kind == 'age':                                   # days since the item's first known event
            ctr, cq, m = np.searchsorted(edges, (ttr - ifirst[i]) / 86400.0), np.searchsorted(edges, (tq - ifirst[qi]) / 86400.0), len(edges) + 1
        elif kind == 'ssize':                                 # how many books the user shelved that day
            kall = allu.astype(np.int64) * 100003 + day(tall)
            uq, cnt = np.unique(kall, return_counts=True)
            size = lambda uu, tt_: cnt[np.searchsorted(uq, uu.astype(np.int64) * 100003 + day(tt_))]
            eb = np.array([1.5, 2.5, 5.5, 10.5, 30.5, 100.5])
            ctr, cq, m = np.searchsorted(eb, size(u, ttr)), np.searchsorted(eb, size(qu, tq)), len(eb) + 1
        else:
            raise ValueError(kind)
        des.add_slot(ctr, cq, m, ids=ids)
V.log(f'task {task} fold {args.fold}: train {len(tr):,}, val {len(va):,}, K {args.k}, model {args.model}, time {args.time!r}, ctx {args.ctx!r}, probit {args.probit}, hetero {args.hetero}')
t0 = time.time()
if args.thr > 0:                               # binary probit: P(rating >= thr)
    yb = (y >= args.thr).astype(np.float64); qb = np.where(np.isnan(q_y), np.nan, (q_y >= args.thr).astype(np.float64))
    pred, sd, lat, trace = bfm4.run(des, yb, qb, args.k, args.iters, args.burn, args.seed + 1000 * args.fold, lo=0.0, hi=1.0, probit=1,
                                    hetero=args.hetero, nu_tau=args.nu, verbose=args.verbose)
    nv = len(va)
    V.log(f'task {task} {args.tag} fold {args.fold}: P(r>={args.thr:g}) Brier {np.mean((pred[:nv] - qb[:nv]) ** 2):.5f} (base rate Brier {np.var(qb[:nv]):.5f}) ({time.time() - t0:.0f}s)')
    name = f'/data/UCSD/CSE258/A1/cache_v10/oof_{args.tag}_t{task}_' + ('full' if args.full else f'f{args.fold}') + '.npz'
    np.savez(name, oof=pred[:nv].astype(np.float32), test=pred[nv:].astype(np.float32), oof_sd=sd[:nv].astype(np.float32),
             test_sd=sd[nv:].astype(np.float32), oof_lat=lat[:nv].astype(np.float32), test_lat=lat[nv:].astype(np.float32), va=va, pred=pred)
    sys.exit(0)
pred, sd, lat, trace = bfm4.run(des, y, q_y, args.k, args.iters, args.burn, args.seed + 1000 * args.fold, probit=args.probit, hetero=args.hetero,
                                nu_tau=args.nu, verbose=args.verbose)
nv = len(va)
mse = np.mean((np.clip(pred[:nv], 1, 5) - q_y[:nv]) ** 2) if nv else float('nan')
V.log(f'task {task} {args.tag} fold {args.fold}: val MSE {mse:.5f}, train mse {trace[-1, 0]:.4f}, alpha {trace[-1, 2]:.3f} ({time.time() - t0:.0f}s)')
if args.tag:
    name = f'/data/UCSD/CSE258/A1/cache_v10/oof_{args.tag}_t{task}_' + ('full' if args.full else f'f{args.fold}') + '.npz'
    np.savez(name, oof=pred[:nv].astype(np.float32), test=pred[nv:].astype(np.float32), oof_sd=sd[:nv].astype(np.float32),
             test_sd=sd[nv:].astype(np.float32), oof_lat=lat[:nv].astype(np.float32), test_lat=lat[nv:].astype(np.float32), mse=mse, va=va, pred=pred, trace=trace)
