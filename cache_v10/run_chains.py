"""Pre-compute the Bayesian-FM chains of v10/assignment1.py into the chain cache (same code path as the pipeline).
usage: PYTHONPATH=v10 python run_chains.py TASK WORKERS spec_index [spec_index ...] [--folds a b c]"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import assignment1 as M
if __name__ == '__main__':
    task, workers = int(sys.argv[1]), int(sys.argv[2])
    rest = sys.argv[3:]
    folds = range(M.RATING_FOLDS)
    if '--folds' in rest:
        folds = [int(x) for x in rest[rest.index('--folds') + 1:]]; rest = rest[:rest.index('--folds')]
    specs = [int(x) for x in rest]
    cache = Path('/data/UCSD/CSE258/A1/cache_v10/chains'); cache.mkdir(exist_ok=True)
    jobs = [(task, Path('/data/UCSD/CSE258/A1/assignment1_2026'), k, j, cache) for j in specs for k in folds]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as ex:
        for _ in ex.map(M._bfm_fold_job, jobs):
            pass
    M.log(f'chains done: task {task} specs {specs}')
