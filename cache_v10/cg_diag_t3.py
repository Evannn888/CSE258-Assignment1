"""CG residuals of the collapsed bag solve on Task 3 (fold 0 only, rank 8, probit + time slots)."""
import sys, numpy as np, torch, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('a1v11d', '/data/UCSD/CSE258/A1/v11/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v11d'] = M; spec.loader.exec_module(M)
M._bfm_torch(); torch.use_deterministic_algorithms(False)
task = int(sys.argv[1])
a, nu, ni, test_rows, folds = M.rating_problem(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
rowfold = np.empty(len(a), np.int64)
for k, p in enumerate(folds):
    rowfold[p] = k
pos = np.flatnonzero(a[:, 3] > 0)
u, i, t, y = a[pos, 0], a[pos, 1], a[pos, 2], a[pos, 3].astype(np.float64)
mask = (rowfold[pos] != 0).astype(np.float32)[None]
Q = np.vstack([a[folds[0]], test_rows])[None]
nU, nI = nu + 1, ni + 1
au, ai = np.r_[a[:, 0], test_rows[:, 0]], np.r_[a[:, 1], test_rows[:, 1]]
pu, eu, cu = M.bfm_bag(au, ai, nU, nI); pi, ei, ci = M.bfm_bag(ai, au, nI, nU)
feat = lambda ptr: np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
slots = [dict(ent=u, q=Q[:, :, 0], n=nU, kind=1, bag=(feat(pu), eu, cu, nI)), dict(ent=i, q=Q[:, :, 1], n=nI, kind=1, bag=(feat(pi), ei, ci, nU))]
for kind in [x for x in sys.argv[2].split(',') if x]:
    tr, q, m, kd = M.bfm_time_slot(kind, u, t, Q[:, :, 0].reshape(-1), Q[:, :, 2].reshape(-1))
    slots.append(dict(ent=tr, q=q.reshape(1, -1), n=m, kind=kd))
model = M.BayesianFM(y, mask, slots, 8, probit=bool(int(sys.argv[3])), seed=0)
orig = model._pcg
log = []
def pcg_dbg(S, W, rhs, lam, Minv):
    Wt = W.T.contiguous()
    op = lambda P: torch.sparse.mm(S['CT'], Wt * torch.sparse.mm(S['C'], P)) + lam[None, :] * P
    x = torch.zeros_like(rhs); r = rhs.clone(); zc = Minv * r; p = zc.clone(); rz = (r * zc).sum(0); res = []
    for it in range(60):
        Ap = op(p); ac = rz / (p * Ap).sum(0).clamp_min(1e-30); x += ac * p; r -= ac * Ap
        zc = Minv * r; rz_new = (r * zc).sum(0); p = zc + (rz_new / rz.clamp_min(1e-30)) * p; rz = rz_new
        if it + 1 in (10, 20, 40, 60):
            res.append((it + 1, float((r.norm(dim=0) / rhs.norm(dim=0)).max())))
    log.append(res)
    return orig(S, W, rhs, lam, Minv)
model._pcg = pcg_dbg
for sw in range(15):
    model.sweep()
    if sw in (2, 8, 14):
        print('sweep', sw, 'residuals (iterations, rel.res) of the last 4 solves:', [[(i, round(v, 4)) for i, v in r] for r in log[-4:]], flush=True)
