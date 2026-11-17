"""Bayesian probit models for the pair tasks: real interaction (1) vs popularity-sampled fake (0).
--model mf   : user ids x item ids
--model fism : no user parameters, user = bag of the items in his/her history (the row's own item left out); items have ids
Scores (posterior mean of the latent predictor) for the three candidates of every held-out row of a cached fold."""
import sys, time, numpy as np, argparse
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import assignment1 as V
import bfm4

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=int, default=2); ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--k', type=int, default=16); ap.add_argument('--iters', type=int, default=150); ap.add_argument('--burn', type=int, default=40)
ap.add_argument('--model', type=str, default='mf'); ap.add_argument('--neg', type=int, default=2); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--tag', type=str, default=''); ap.add_argument('--verbose', type=int, default=0)
args = ap.parse_args()
z = np.load(f'/data/UCSD/CSE258/A1/cache_v7/pairs_t{args.task}.npz')
u, i, nU, nI = z['u'], z['i'], int(z['nU']), int(z['nI'])
held = z[f'held{args.fold}']; qc = z[f'qc{args.fold}']
keep = np.ones(len(u), bool); keep[held] = False
pu, pi = u[keep], i[keep]
rng = np.random.default_rng(args.seed)
key = np.sort(pu.astype(np.int64) * nI + pi)
nu_, ni_ = [np.repeat(pu, args.neg)], []
neg = pi[rng.integers(0, len(pi), len(pu) * args.neg)]                       # item of a random interaction
for _ in range(20):                                                          # resample fakes that are real interactions
    k = nu_[0].astype(np.int64) * nI + neg
    pos = np.searchsorted(key, k); bad = (pos < len(key)) & (key[np.minimum(pos, len(key) - 1)] == k)
    if not bad.any():
        break
    neg[bad] = pi[rng.integers(0, len(pi), bad.sum())]
ru, ri = np.r_[pu, nu_[0]], np.r_[pi, neg]
y = np.r_[np.ones(len(pu)), np.zeros(len(neg))]
w = np.r_[np.ones(len(pu)), np.full(len(neg), 1.0 / args.neg)]
qu = np.repeat(u[held], qc.shape[1]); qi = qc.reshape(-1)
des = bfm4.Design(len(y), len(qu))
su = des.add_slot(ru, qu, nU, ids=args.model == 'mf'); si = des.add_slot(ri, qi, nI, ids=True)
if args.model == 'fism':
    cnt = np.bincount(pu, minlength=nU).astype(float)
    des.add_bag(su, pu, pi, np.maximum(cnt[pu], 1) ** -0.5, nI, row=np.arange(len(pu)))
V.log(f'task {args.task} fold {args.fold} model {args.model}: rows {len(y):,}, K {args.k}')
t0 = time.time()
pred, sd, lat, trace = bfm4.run(des, y, np.full(len(qu), np.nan), args.k, args.iters, args.burn, args.seed, lo=0.0, hi=1.0, probit=1, hetero=0,
                                verbose=args.verbose, row_w=w)
S = lat.reshape(qc.shape); P = pred.reshape(qc.shape)
seen = z[f'seen{args.fold}']
acc = np.mean(S[:, 0] > S[:, 1]); acc_seen = np.mean((S[:, 0] - S[:, 1] + 100 * (seen[:, 1] - seen[:, 0])) > 0)
V.log(f'task {args.task} {args.tag} fold {args.fold} {args.model} K{args.k}: pair acc {acc:.4f} (with seen rule {acc_seen:.4f}) ({time.time() - t0:.0f}s)')
if args.tag:
    np.savez(f'/data/UCSD/CSE258/A1/cache_v10/bimp_{args.tag}_t{args.task}_f{args.fold}.npz', S=S.astype(np.float32), P=P.astype(np.float32))
