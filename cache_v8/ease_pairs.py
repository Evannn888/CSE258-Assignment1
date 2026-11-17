"""EASE (Steck 2019) item-item model as extra features for the Task 2/4 pair classifier."""
import sys, numpy as np, numba as nb, time, torch
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
V.torch_setup()


def ease_B(u, i, nU, nI, lam, dev='cpu'):
    import scipy.sparse as sp
    from scipy.linalg import lapack
    X = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(nU, nI))
    X.data[:] = 1.0                                   # binary (duplicates summed then reset)
    G = np.asfortranarray((X.T @ X).toarray().astype(np.float32))
    G[np.diag_indices(nI)] += lam
    c, info = lapack.spotrf(G, lower=1, overwrite_a=1)
    assert info == 0
    P, info = lapack.spotri(c, lower=1, overwrite_c=1)
    assert info == 0
    P = np.tril(P) + np.tril(P, -1).T
    d = np.diag(P).copy()
    P /= -d[None, :]
    P[np.diag_indices(nI)] = 0.0
    return np.ascontiguousarray(P)


@nb.njit(parallel=True, cache=True)
def ease_scores(qu, qc, qt, uptr, ui, ut, B, taus):
    n, C = qc.shape
    out = np.zeros((n, C, 1 + len(taus)), np.float32)
    for r in nb.prange(n):
        u = qu[r]
        for y in range(uptr[u], uptr[u + 1]):
            h = ui[y]
            for c in range(C):
                b = B[h, qc[r, c]]
                out[r, c, 0] += b
                for k in range(len(taus)):
                    out[r, c, 1 + k] += b * np.exp(-abs(ut[y] - qt[r]) / (taus[k] * 86400.0))
    return out


def fold_graphs(z):
    u, i, t, r = z['u'], z['i'], z['t'], z['r']
    nU, nI = int(z['nU']), int(z['nI'])
    for f in range(4):
        keep = np.ones(len(u), bool)
        keep[z[f'held{f}']] = False
        yield f, V.Graph(u[keep], i[keep], t[keep], r[keep], nU, nI), u[z[f'held{f}']]
    yield 'test', V.Graph(u, i, t, r, nU, nI), z['testqu']


if __name__ == '__main__':
    task = int(sys.argv[1])
    lams = [float(x) for x in sys.argv[2:]]
    z = np.load(f'/data/UCSD/CSE258/A1/cache_v7/pairs_t{task}.npz')
    taus = np.array([30.0, 365.0]) if task == 4 else np.zeros(0)
    res = {}
    for f, g, qu in fold_graphs(z):
        qc = z[f'qc{f}'] if f != 'test' else z['testqc']
        qt = z[f'qt{f}'] if f != 'test' else z['testqt']
        for lam in lams:
            t0 = time.time()
            B = ease_B(g.u, g.i, g.nU, g.nI, lam)
            S = ease_scores(qu, qc, qt, g.uptr, g.i, g.t, B, taus)
            del B
            res[f'E{f}_{lam:g}'] = S
            if f != 'test':
                acc = np.mean(S[:, 0, 0] > S[:, 1, 0])
                print(f'fold {f} lam {lam:g}: standalone acc {acc:.4f} ' + (f'tau30 {np.mean(S[:, 0, 1] > S[:, 1, 1]):.4f} tau365 {np.mean(S[:, 0, 2] > S[:, 1, 2]):.4f}' if task == 4 else '') + f' {time.time()-t0:.0f}s', flush=True)
        if f == 1 and len(lams) > 1:
            pass
    np.savez(f'/data/UCSD/CSE258/A1/cache_v8/ease_t{task}.npz', **res)
    print('saved')
