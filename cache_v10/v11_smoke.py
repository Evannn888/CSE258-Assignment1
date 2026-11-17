import sys, numpy as np, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('a1v11s', '/data/UCSD/CSE258/A1/v11/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v11s'] = M; spec.loader.exec_module(M)
task = int(sys.argv[1])
a, nu, ni, test_rows, folds = M.rating_problem(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
sp = dict(k=4, iters=6, burn=2, probit=bool(int(sys.argv[2])), time=tuple(x for x in sys.argv[3].split(',') if x) if len(sys.argv) > 3 else ())
out = M.bfm_fold_chains(task, a, nu, ni, test_rows, folds, sp, 1)
print(len(out), [o.shape for o in out[:2]], [len(p) + len(test_rows) for p in folds[:2]], 'finite', all(np.isfinite(o).all() for o in out))
