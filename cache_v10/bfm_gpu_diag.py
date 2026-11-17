import sys, numpy as np, torch, importlib.util
from pathlib import Path
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v10')
import bfm_gpu
spec = importlib.util.spec_from_file_location('a1v10g', '/data/UCSD/CSE258/A1/v10/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v10g'] = M; spec.loader.exec_module(M)
a, nu, ni, test_rows, folds = M.rating_problem(1, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
rowfold = np.empty(len(a), np.int64)
for k, p in enumerate(folds):
    rowfold[p] = k
pos = np.flatnonzero(a[:, 3] > 0)
u, i, y = a[pos, 0], a[pos, 1], a[pos, 3].astype(np.float64)
mask = (rowfold[pos] != 0).astype(np.float32)[None]
q = np.vstack([a[folds[0]], test_rows])[None]
nU, nI = nu + 1, ni + 1
au, ai = np.r_[a[:, 0], test_rows[:, 0]], np.r_[a[:, 1], test_rows[:, 1]]
pu, eu, cu = M.bfm_bag(au, ai, nU, nI); pi, ei, ci = M.bfm_bag(ai, au, nI, nU)
feat_of = lambda ptr: np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
slots = [dict(ent=u, q=q[:, :, 0], n=nU, kind=1, bag=(feat_of(pu), eu, cu, nI)), dict(ent=i, q=q[:, :, 1], n=nI, kind=1, bag=(feat_of(pi), ei, ci, nU))]
m = bfm_gpu.BFMGPU(y, mask, slots, 8, cg_iters=12)
orig = m.pcg
def pcg_dbg(s, A, rhs, lam, Minv):
    S, al = m.S[s], m.alpha
    At = A.T.contiguous()
    op = lambda P: torch.sparse.mm(S['CT'], At * torch.sparse.mm(S['C'], P)) + lam[None, :] * P
    x = torch.zeros_like(rhs); r = rhs.clone(); zc = Minv * r; p = zc.clone(); rz = (r * zc).sum(0); res = []
    for it in range(200):
        Ap = op(p); ac = rz / (p * Ap).sum(0).clamp_min(1e-30); x += ac * p; r -= ac * Ap
        zc = Minv * r; rz_new = (r * zc).sum(0); p = zc + (rz_new / rz.clamp_min(1e-30)) * p; rz = rz_new
        if it in (4, 9, 19, 49, 99, 199):
            res.append((it + 1, round((r.norm(dim=0) / rhs.norm(dim=0)).max().item(), 5)))
    d = 1 / Minv
    print(f'slot {s}: lam {lam.item():.3g}, alpha {al.item():.3g}, diag min {d.min().item():.3g} median {d.median().item():.3g} max {d.max().item():.3g}; '
          f'A max {A.max().item():.3g}; residual by iteration {res}', flush=True)
    m.cg_rel = 0
    return x
for sweep in range(25):
    m.pcg = (pcg_dbg if sweep in (2, 10, 24) else orig)
    if sweep in (2, 10, 24):
        print('--- sweep', sweep)
    m.sweep()
