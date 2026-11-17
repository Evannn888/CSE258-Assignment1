"""Cache the v6 18-fold out-of-fold rating feature bank (identical folds/features to run_rating_task)."""
import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
if __name__ == '__main__':
    task = int(sys.argv[1]); out = Path('/data/UCSD/CSE258/A1/cache_v7')
    D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
    a, users, items = V.load(D / V.FILES[task])
    nu, ni = len(users), len(items)
    tests = V.read_test(D / f'task_{task}_test.csv')
    qu = np.array([users.get(r['user'], nu) for r in tests], np.int64); qi = np.array([items.get(r['item'], ni) for r in tests], np.int64)
    qt = np.array([int(r['timestamp']) for r in tests], np.int64) if task == 3 else np.zeros(len(tests), np.int64)
    test_rows = np.column_stack([qu, qi, qt, np.zeros(len(qu), np.int64)])
    folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
    y_folds = [a[p, 3].astype(float) for p in folds]
    y = np.concatenate([yf[yf > 0] for yf in y_folds])
    fold_id = np.concatenate([np.full((yf > 0).sum(), k) for k, yf in enumerate(y_folds)])
    rows = np.concatenate([p[yf > 0] for p, yf in zip(folds, y_folds)])
    jobs = []
    for p in folds:
        keep = np.ones(len(a), bool); keep[p] = False
        jobs.append((task, a[keep], np.vstack([a[p], test_rows]), nu, ni))
    X = None; T = None; r0 = 0
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as ex:
        for k, (block, p, yf) in enumerate(zip(ex.map(V._rating_fold_job, jobs), folds, y_folds)):
            if X is None:
                X = np.lib.format.open_memmap(out / f't{task}_X.npy', 'w+', np.float32, (len(y), block.shape[1] - 1))
                T = np.lib.format.open_memmap(out / f't{task}_test.npy', 'w+', np.float32, (len(folds), len(test_rows), block.shape[1] - 1))
            m = yf > 0
            X[r0:r0 + m.sum()] = block[:len(p)][m, 1:]; r0 += m.sum()
            T[k] = block[len(p):, 1:]
            V.log(f'task {task} fold {k} cached')
    X.flush(); T.flush()
    np.savez(out / f't{task}_meta.npz', y=y, fold=fold_id, rows=rows, users=a[rows, 0], items=a[rows, 1])
    V.log('done')
