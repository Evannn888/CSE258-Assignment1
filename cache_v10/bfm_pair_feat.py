"""Does the predicted RATING of a candidate (Bayesian FM trained on the fold's ratings) tell real interactions from
popularity-sampled fakes?  One chain for one cached fold of a pair task."""
import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1'); sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import assignment1 as V
import bfm_prod_mod as P
task, f = int(sys.argv[1]), int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
z = np.load(f'/data/UCSD/CSE258/A1/cache_v7/pairs_t{task}.npz')
u, i, t, r, nU, nI = z['u'], z['i'], z['t'], z['r'], int(z['nU']), int(z['nI'])
held, qc = z[f'held{f}'], z[f'qc{f}']
keep = np.ones(len(u), bool); keep[held] = False
fit = np.column_stack([u[keep], i[keep], t[keep], r[keep]]).astype(np.int64)
qu = np.repeat(u[held], qc.shape[1]); qi = qc.reshape(-1)
# bags from the training interactions only: the candidates are not known interactions.  Trick: give the query rows
# the spare ids for the bag construction by building the bags from `fit` alone.
pos = fit[:, 3] > 0
uu, ii, y = fit[pos, 0], fit[pos, 1], fit[pos, 3].astype(np.float64)
nu1, ni1 = nU + 1, nI + 1
pu, eu, cu = P.bfm_bag(fit[:, 0], fit[:, 1], nu1, ni1)
pi, ei, ci = P.bfm_bag(fit[:, 1], fit[:, 0], ni1, nu1)
t0 = time.time()
m, sd = P.bfm_gibbs(K, 300, 60, 1, y, np.ascontiguousarray(np.vstack([uu, nu1 + ii]).astype(np.int64)), np.array([0, nu1, nu1 + ni1], np.int64),
                    np.array([1, 1], np.int64), np.r_[pu, pi[1:] + pu[-1]].astype(np.int64), np.r_[eu, nu1 + ei].astype(np.int64), np.r_[cu, ci],
                    np.array([0, ni1, ni1 + nu1], np.int64), np.array([0, 1], np.int64),
                    np.ascontiguousarray(np.vstack([qu, nu1 + qi]).astype(np.int64)), 1.0, 5.0, 0, 30.0)
S = m.reshape(qc.shape); SD = sd.reshape(qc.shape)
seen = z[f'seen{f}']
print(f'task {task} fold {f}: predicted-rating pair acc {np.mean(S[:, 0] > S[:, 1]):.4f}; mean rating pos {S[:, 0].mean():.3f} neg {S[:, 1].mean():.3f}; '
      f'posterior sd pos {SD[:, 0].mean():.3f} neg {SD[:, 1].mean():.3f}; sd pair acc (pos smaller) {np.mean(SD[:, 0] < SD[:, 1]):.4f} ({time.time() - t0:.0f}s)', flush=True)
np.savez(f'/data/UCSD/CSE258/A1/cache_v10/pairrat_t{task}_f{f}.npz', S=S.astype(np.float32), SD=SD.astype(np.float32))
