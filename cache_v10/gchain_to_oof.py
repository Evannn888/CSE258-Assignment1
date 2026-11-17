"""Cached v11 chains -> oof_{tag}_t{task}_f{k}.npz for fast_stack.py.  usage: gchain_to_oof.py TASK NAME TAG"""
import sys, numpy as np, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('a1v11o', '/data/UCSD/CSE258/A1/v11/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v11o'] = M; spec.loader.exec_module(M)
task, name, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]
a, nu, ni, test_rows, folds = M.rating_problem(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
tot, n = 0.0, 0
for k, part in enumerate(folds):
    m = np.load(f'/data/UCSD/CSE258/A1/cache_v10/gchains/{name}_t{task}_f{k}.npy')
    pos = a[part, 3] > 0
    np.savez(f'/data/UCSD/CSE258/A1/cache_v10/oof_{tag}_t{task}_f{k}.npz', oof=m[:len(part)][pos].astype(np.float32), test=m[len(part):].astype(np.float32))
    tot += np.sum((np.clip(m[:len(part)][pos], 1, 5) - a[part, 3][pos]) ** 2); n += pos.sum()
print(f'task {task} {name} -> {tag}: out-of-fold MSE {tot / n:.5f}')
