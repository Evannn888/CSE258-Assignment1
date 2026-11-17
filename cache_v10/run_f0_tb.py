"""Fold-0 GPU runs (production BayesianFM) with temporal slots that carry a bag: the entity (user, period) also gets
the bag of the books the user interacted with in that period (existence only).
usage: run_f0_tb.py TASK K iters burn probit slots seed [folds]
slots: comma list of month, session, uyear, uyear+bag, umonth, umonth+bag, uquarter+bag, iyear, iyear+bag"""
import sys, time, numpy as np, torch, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('a1v12tb', '/data/UCSD/CSE258/A1/v12/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v12tb'] = M; spec.loader.exec_module(M)
task, k, iters, burn, probit, tm, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), bool(int(sys.argv[5])), sys.argv[6], int(sys.argv[7])
fl = [int(x) for x in sys.argv[8].split(',')] if len(sys.argv) > 8 else [0]
M._bfm_torch(); torch.use_deterministic_algorithms(False)
a, nu, ni, test_rows, folds = M.rating_problem(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
rowfold = np.empty(len(a), np.int64)
for kk, p in enumerate(folds):
    rowfold[p] = kk
pos = np.flatnonzero(a[:, 3] > 0)
u, i, t, y = a[pos, 0], a[pos, 1], a[pos, 2], a[pos, 3].astype(np.float64)
F = len(fl)
mask = np.stack([(rowfold[pos] != kk).astype(np.float32) for kk in fl])
L = max(len(folds[kk]) for kk in fl) + len(test_rows)
Q = np.stack([np.vstack([a[folds[kk]], test_rows, np.repeat(test_rows[-1:], L - len(folds[kk]) - len(test_rows), 0)]) for kk in fl])
nU, nI = nu + 1, ni + 1
au, ai, at = np.r_[a[:, 0], test_rows[:, 0]], np.r_[a[:, 1], test_rows[:, 1]], np.r_[a[:, 2], test_rows[:, 2]]
pu, eu, cu = M.bfm_bag(au, ai, nU, nI); pi, ei, ci = M.bfm_bag(ai, au, nI, nU)
feat = lambda ptr: np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
slots = [dict(ent=u, q=Q[:, :, 0], n=nU, kind=1, bag=(feat(pu), eu, cu, nI)), dict(ent=i, q=Q[:, :, 1], n=nI, kind=1, bag=(feat(pi), ei, ci, nU))]
clip = lambda x: np.clip(np.asarray(x, np.float64), 1.0e9, M.TEND)
span = {'uyear': 365.25, 'umonth': 30.44, 'uquarter': 91.3, 'iyear': 365.25}
tt_all = a[:, 2].astype(np.float64)
okt = (tt_all > 1.0e9) & (tt_all < M.TEND + M.DAY)
def ent_year(ids, n, how):
    out = np.zeros(n, np.int64)
    o = np.lexsort((tt_all[okt], ids[okt])); ii, tt = ids[okt][o], tt_all[okt][o]
    ptr = np.searchsorted(ii, np.arange(n + 1))
    for r in range(n):
        if ptr[r + 1] > ptr[r]:
            v = tt[ptr[r]] if how == 'first' else tt[(ptr[r] + ptr[r + 1]) // 2]
            out[r] = int((v - 1.0e9) // (365.25 * M.DAY)) + 1
    return out
CTX = {'icohort': ('i', 'first'), 'imed': ('i', 'med'), 'umed': ('u', 'med'), 'ucohort': ('u', 'first')}
for name in [x for x in tm.split(',') if x and x != 'none']:
    if name in CTX:
        side, how = CTX[name]
        if side == 'i':
            code = ent_year(a[:, 1], nI, how); slots.append(dict(ent=code[i], q=code[Q[:, :, 1]], n=int(code.max()) + 1, kind=1))
        else:
            code = ent_year(a[:, 0], nU, how); slots.append(dict(ent=code[u], q=code[Q[:, :, 0]], n=int(code.max()) + 1, kind=1))
        continue
    base, bag = name.split('+')[0], name.endswith('+bag')
    if base in ('month', 'session', 'uyear') and not bag:
        tr, q, m, kd = M.bfm_time_slot(base, u, t, Q[:, :, 0].reshape(-1), Q[:, :, 2].reshape(-1))
        slots.append(dict(ent=tr, q=q.reshape(F, L), n=m, kind=kd))
        continue
    code = lambda x: ((clip(x) - 1.0e9) // (span[base] * M.DAY)).astype(np.int64)
    who_tr, who_q, who_bag, other_bag, n_other = (u, Q[:, :, 0].reshape(-1), au, ai, nI) if base[0] == 'u' else (i, Q[:, :, 1].reshape(-1), ai, au, nU)
    k_tr, k_q = who_tr.astype(np.int64) * 100003 + code(t), who_q.astype(np.int64) * 100003 + code(Q[:, :, 2].reshape(-1))
    kb = who_bag.astype(np.int64) * 100003 + code(at)
    uq, inv = np.unique(np.r_[k_tr, k_q, kb], return_inverse=True)
    s = dict(ent=inv[:len(k_tr)], q=inv[len(k_tr):len(k_tr) + len(k_q)].reshape(F, L), n=len(uq), kind=1)
    if bag:
        eb = inv[len(k_tr) + len(k_q):]
        pb, rb, cb = M.bfm_bag(eb, other_bag, len(uq), n_other)
        s['bag'] = (feat(pb), rb, cb, n_other)
    slots.append(s)
t0 = time.time()
import os
model = M.BayesianFM(y, mask, slots, k, probit=probit, seed=seed, nu_tau=float(os.environ.get('NU', 30)))
mean = model.run(iters, burn)
mses = []
for r, kk in enumerate(fl):
    yy = a[folds[kk], 3]; ok = yy > 0
    mses.append(np.mean((np.clip(mean[r, :len(yy)][ok], 1, 5) - yy[ok]) ** 2))
print(f'GPU task {task} K{k} iters {iters} probit {int(probit)} slots {tm} seed {seed} folds {fl}: MSE {np.round(mses, 5)} mean {np.mean(mses):.5f} ({time.time() - t0:.0f}s)', flush=True)
