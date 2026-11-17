"""Convert cached production chains (cache_v10/chains/bfm_t{task}_f{k}_s{j}.npy) into the oof_{tag}_t{task}_f{k}.npz
layout used by fast_stack.py.  usage: chains_to_oof.py TASK SPEC TAG"""
import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1/v10')
import assignment1 as M
from pathlib import Path
task, j, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]
src = (lambda k: f'/data/UCSD/CSE258/A1/cache_v10/chains/bfm_t{task}_f{k}_s{j}.npy') if j.isdigit() else (lambda k: f'/data/UCSD/CSE258/A1/cache_v10/gpu_chains/{j}/t{task}_f{k}.npy')
a, nu, ni, test_rows, folds = M.rating_problem(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'))
tot, n = 0.0, 0
for k, part in enumerate(folds):
    m = np.load(src(k))
    pos = a[part, 3] > 0
    oof = m[:len(part)][pos]
    np.savez(f'/data/UCSD/CSE258/A1/cache_v10/oof_{tag}_t{task}_f{k}.npz', oof=oof.astype(np.float32), test=m[len(part):].astype(np.float32))
    tot += np.sum((np.clip(oof, 1, 5) - a[part, 3][pos]) ** 2); n += pos.sum()
print(f'task {task} spec {j} -> {tag}: out-of-fold MSE {tot / n:.5f} over {n:,} rows')
