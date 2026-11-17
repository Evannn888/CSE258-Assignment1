"""EASE for large item sets on CPU, in place (one n x n float32 matrix)."""
import sys, numpy as np, numba as nb, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from scipy.linalg import lapack


@nb.njit(parallel=True, cache=True)
def _gram(uptr, ui, n, G, nblk):
    # rows of G split into nblk stripes so threads never write the same row
    for blk in nb.prange(nblk):
        for u in range(len(uptr) - 1):
            a, b = uptr[u], uptr[u + 1]
            for x in range(a, b):
                i = ui[x]
                if i % nblk != blk:
                    continue
                for y in range(a, b):
                    G[ui[y], i] += 1.0          # column i (Fortran: column-major contiguous)


@nb.njit(parallel=True, cache=True)
def _finish(P):
    n = P.shape[0]
    d = np.empty(n, np.float32)
    for j in range(n):
        d[j] = P[j, j]
    # P holds the lower triangle (column-major view): fill upper, then B = -P / diag (column scaling), zero diagonal
    for j in nb.prange(n):
        for i in range(j + 1, n):
            P[j, i] = P[i, j]
    for j in nb.prange(n):
        for i in range(n):
            P[i, j] = -P[i, j] / d[j]
        P[j, j] = 0.0


def ease_inplace(u, i, nU, nI, lam):
    """Binary user-item matrix -> EASE weight matrix B (nI x nI float32, Fortran order)."""
    o = np.lexsort((i, u))
    u, i = u[o], i[o]
    keep = np.r_[True, (u[1:] != u[:-1]) | (i[1:] != i[:-1])]
    u, i = u[keep], i[keep]
    uptr = np.searchsorted(u, np.arange(nU + 1))
    t0 = time.time()
    G = np.zeros((nI, nI), np.float32, order='F')
    _gram(uptr, i, nI, G, 64)
    G[np.diag_indices(nI)] += lam
    V.log(f'gram {time.time()-t0:.0f}s')
    c, info = lapack.spotrf(G, lower=1, overwrite_a=1, clean=0)
    assert info == 0, info
    V.log(f'potrf {time.time()-t0:.0f}s')
    P, info = lapack.spotri(c, lower=1, overwrite_c=1)
    assert info == 0, info
    del c, G
    V.log(f'potri {time.time()-t0:.0f}s')
    _finish(P)
    return P
