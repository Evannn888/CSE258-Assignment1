"""Run one Bayesian-FM spec through the v11 production function (chains cached in cache_v10/gchains).
usage: run_gchain.py TASK k iters burn probit(0/1) time(comma list or '') seed"""
import sys, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('a1v12r', '/data/UCSD/CSE258/A1/v12/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v12r'] = M; spec.loader.exec_module(M)
task, k, iters, burn, probit, tm, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), bool(int(sys.argv[5])), sys.argv[6], int(sys.argv[7])
era = tuple(x for x in (sys.argv[8] if len(sys.argv) > 8 else '').split(',') if x and x != 'none')
sp = dict(k=k, iters=iters, burn=burn, probit=probit, time=tuple(x for x in tm.split(',') if x and x != 'none'), seed=seed)
if era:
    sp['era'] = era
nu = float(sys.argv[9]) if len(sys.argv) > 9 else 30.0
if nu != 30.0:
    sp['nu'] = nu
a, nu, ni, test_rows, folds = M.rating_problem(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
M.bfm_fold_chains(task, a, nu, ni, test_rows, folds, sp, Path('/data/UCSD/CSE258/A1/cache_v10/gchains'))
print('name:', M.bfm_spec_name(sp))
