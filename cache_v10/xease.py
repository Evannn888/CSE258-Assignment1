"""Explicit-residual EASE ("ridge regression of rating residuals on the user's other residuals", closed form).
Item-item (mode i): target residual of item i predicted from the user's other residuals; user-user (mode u): transposed.
One fold: the fold's ratings are excluded everywhere; features for the fold's positive rows and for the test rows."""
import sys, time, numpy as np, numba as nb, argparse
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
from scipy.linalg import lapack
from threadpoolctl import threadpool_limits


@nb.njit(parallel=True, cache=True)
def wgram(ptr, idx, val, G, nblk):
    for blk in nb.prange(nblk):
        for u in range(len(ptr) - 1):
            for x in range(ptr[u], ptr[u + 1]):
                i = idx[x]
                if i % nblk != blk:
                    continue
                vx = val[x]
                for y in range(ptr[u], ptr[u + 1]):
                    G[idx[y], i] += vx * val[y]


@nb.njit(parallel=True, cache=True)
def feats(qa, qb, ptr, idx, val, B):
    """qa: row entity (whose other residuals are used), qb: target column."""
    n = len(qa)
    out = np.zeros((n, 3), np.float32)
    for r in nb.prange(n):
        u, i = qa[r], qb[r]
        s, sa, c = 0.0, 0.0, 0
        for x in range(ptr[u], ptr[u + 1]):
            j = idx[x]
            if j == i:
                continue
            b = B[j, i]
            s += val[x] * b
            sa += abs(b)
            c += 1
        out[r, 0] = s
        out[r, 1] = sa
        out[r, 2] = s / (sa + 0.05)
    return out


def solve(G, lam, threads=8):
    n = G.shape[0]
    G[np.diag_indices(n)] += lam
    with threadpool_limits(limits=threads, user_api='blas'):
        c, info = lapack.spotrf(G, lower=1, overwrite_a=1, clean=0); assert info == 0
        P, info = lapack.spotri(c, lower=1, overwrite_c=1); assert info == 0
    V._ease_finish(P)
    return P


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, default=3); ap.add_argument('--fold', type=int, default=0); ap.add_argument('--mode', type=str, default='i')
    ap.add_argument('--lam', type=float, nargs='+', default=[20.0]); ap.add_argument('--tag', type=str, default='xe'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--center', type=str, default='ui')
    args = ap.parse_args()
    task = args.task
    D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
    a, users, items = V.load(D / V.FILES[task])
    nu, ni = len(users), len(items)
    tests = V.read_test(D / f'task_{task}_test.csv')
    tu = np.array([users[r['user']] for r in tests]); ti = np.array([items[r['item']] for r in tests])
    folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
    part = folds[args.fold]
    va = part[a[part, 3] > 0]
    keep = np.ones(len(a), bool); keep[part] = False
    tr = np.flatnonzero(keep & (a[:, 3] > 0))
    u, i, y = a[tr, 0], a[tr, 1], a[tr, 3].astype(np.float64)
    mu = y.mean(); bu, bi = np.zeros(nu), np.zeros(ni)
    cu, ci = np.bincount(u, minlength=nu), np.bincount(i, minlength=ni)
    for _ in range(15):
        bu = np.bincount(u, weights=y - mu - bi[i], minlength=nu) / (cu + 3.0)
        bi = np.bincount(i, weights=y - mu - bu[u], minlength=ni) / (ci + 5.0)
    res = (y - mu - bu[u] - bi[i]) if args.center == 'ui' else (y - mu - bi[i]) if args.center == 'i' else (y - mu - bu[u])
    res = res.astype(np.float32)
    qa, qb = np.r_[a[va, 0], tu], np.r_[a[va, 1], ti]
    if args.mode == 'u':                       # user-user: rows = items, columns = users
        u, i, nu, ni, qa, qb = i, u, ni, nu, qb, qa
    o = np.lexsort((i, u)); ptr = np.searchsorted(u[o], np.arange(nu + 1)); idx = i[o].astype(np.int64); val = res[o]
    out = []
    for lam in args.lam:
        t0 = time.time()
        G = np.zeros((ni, ni), np.float32, order='F')
        wgram(ptr, idx, val, G, 64)
        B = solve(G, lam, args.threads)
        del G
        out.append(feats(qa, qb, ptr, idx, val, B))
        del B
        f = out[-1][:len(va), 0]
        base = mu + bu[a[va, 0]] + bi[a[va, 1]] if True else 0
        yv = a[va, 3].astype(np.float64)
        best = min((np.mean((np.clip(base + s * f, 1, 5) - yv) ** 2), s) for s in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0))
        V.log(f'task {task} {args.tag} mode {args.mode} fold {args.fold} lam {lam:g}: bias-only {np.mean((np.clip(base, 1, 5) - yv) ** 2):.5f}, bias + s*f best {best[0]:.5f} (s={best[1]}) ({time.time() - t0:.0f}s)')
    F = np.hstack(out)
    np.savez(f'/data/UCSD/CSE258/A1/cache_v10/oof_{args.tag}_t{task}_f{args.fold}.npz', oof=F[:len(va)], test=F[len(va):], va=va)
