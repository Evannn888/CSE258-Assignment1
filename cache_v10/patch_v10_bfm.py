"""Build v10/assignment1.py from v9: add the Bayesian factorization machine and use it in the rating tasks."""
p = '/data/UCSD/CSE258/A1/v10/assignment1.py'
s = open('/data/UCSD/CSE258/A1/v9/assignment1.py').read()
body = open('/data/UCSD/CSE258/A1/cache_v10/bfm_prod.py').read()
body = body.replace("import math\n\n", "")
# 0. module docstring
old = """  Tasks 1 & 3 ....... rating regression: base-model feature bank + EASE neighbourhoods, 18-fold out-of-fold,
                      (Ridge + residual GBDT) & MLP
"""
assert old in s
s = s.replace(old, """  Bayesian FM ....... Gibbs-sampled factorization machine (heteroscedastic / ordinal probit), new in v10
  Tasks 1 & 3 ....... rating regression: base-model feature bank + EASE neighbourhoods + Bayesian FM chains,
                      18-fold out-of-fold, (Ridge + residual GBDT) & MLP
""")
# 1. imports
old = "import argparse\nimport calendar\nimport csv\nimport os\n"
assert old in s; s = s.replace(old, "import argparse\nimport calendar\nimport csv\nimport math\nimport os\n")
# 2. BFM section before the rating section
marker = "# =============================================================================================\n# Tasks 1 and 3: rating prediction\n"
assert s.count(marker) == 1
banner = '''# =============================================================================================
# Bayesian factorization machine (new in v10; used by Tasks 1 and 3)
#
# The point-estimate matrix factorisations of the feature bank barely improve on the biases for these sparse data
# (median 24 ratings per user, 18 per item).  A Gibbs-sampled factorisation machine (Rendle 2012; "SVD++ flipped"
# feature set of Rendle, Zhang & Koren 2019) averages over the posterior instead and needs no tuning of
# regularisation: all prior means / precisions are sampled.  Two changes mattered here: the noise precision is
# user- and item-specific (heteroscedastic), and an ordinal-probit likelihood is available for the 1-5 scale.
# =============================================================================================
'''
body = body.replace('''# ---------------------------------------------------------------------------------------------
# Bayesian factorization machine (Gibbs sampling)
# ---------------------------------------------------------------------------------------------
''', banner)
body = body.replace("known_queries=True, spare=1):", "known_queries=True):").replace("    nU, nI = nu + spare, ni + spare                           # one spare id for users / items unseen in training",
                                                   "    nU, nI = nu + 1, ni + 1                                   # one spare id for users / items unseen in training")
assert 'spare=' not in body and '+ spare' not in body
s = s.replace(marker, body + "\n\n" + marker)
# 3. rating pipeline
old = s[s.index("RATING_FOLDS = 18\n"):s.index("class ChunkedRidge:")]
new = '''RATING_FOLDS = 18
META_HGB = dict(max_leaf_nodes=127, max_iter=1500, learning_rate=0.04, min_samples_leaf=200, l2_regularization=100.0)
# Bayesian FM chains per fold: a rank-32 Gaussian model and a rank-16 ordinal-probit model; Task 3 adds time slots
# and a third chain (rank-32 probit with the user-year slot, the best single model there)
BFM_SPECS = {1: (dict(k=32, iters=500, burn=100), dict(k=16, iters=400, burn=80, probit=True)),
             3: (dict(k=32, iters=500, burn=100, time=('month', 'session')),
                 dict(k=16, iters=400, burn=80, probit=True, time=('month', 'session', 'uyear')),
                 dict(k=32, iters=500, burn=100, probit=True, time=('month', 'session', 'uyear')))}


def rating_problem(task, data_dir):
    """Training rows, test queries and the 18 folds of a rating task (shared by the main process and the workers)."""
    a, users, items = load(data_dir / FILES[task])
    nu, ni = len(users), len(items)
    tests = read_test(data_dir / f'task_{task}_test.csv')
    qu = np.array([users.get(r['user'], nu) for r in tests], np.int64)
    qi = np.array([items.get(r['item'], ni) for r in tests], np.int64)
    qt = np.array([int(r['timestamp']) for r in tests], np.int64) if task == 3 else np.zeros(len(tests), np.int64)
    test_rows = np.column_stack([qu, qi, qt, np.zeros(len(qu), np.int64)])
    folds = np.array_split(np.random.default_rng(11).permutation(len(a)), RATING_FOLDS)
    return a, nu, ni, test_rows, folds


def _rating_fold_job(args):
    task, fit, queries, nu, ni = args
    state = ratings_fit_extended(fit, nu, ni, task)
    X, _ = ratings_extract_extended(state, queries[:, 0], queries[:, 1], queries[:, 2])
    return X.astype(np.float32)


def _bfm_fold_job(args):
    """One Gibbs chain: fit on all folds but k, predict fold k (all its rows) and the test rows.  With a cache
    directory the chain is stored / reused (the chains are deterministic, so a cached file equals a fresh run)."""
    task, data_dir, k, j, cache = args
    path = None if cache is None else cache / f'bfm_t{task}_f{k}_s{j}.npy'
    if path is not None and path.exists():
        return np.load(path)
    a, nu, ni, test_rows, folds = rating_problem(task, data_dir)
    keep = np.ones(len(a), bool)
    keep[folds[k]] = False
    t0 = time.time()
    mean, _ = bfm_rating_predictions(a[keep], np.vstack([a[folds[k]], test_rows]), nu, ni, SEED + 1000 * k + j, **BFM_SPECS[task][j])
    mean = mean.astype(np.float32)
    held = a[folds[k], 3]
    log(f'task {task}: Bayesian FM {j} fold {k}: MSE {np.mean((np.clip(mean[:len(held)][held > 0], 1, 5) - held[held > 0]) ** 2):.5f} '
        f'({time.time() - t0:.0f}s)')
    if path is not None:
        np.save(path, mean)
    return mean


'''
s = s.replace(old, new)
old = s[s.index("def run_rating_task(task, args):"):s.index("# =============================================================================================\n# main\n")]
new = '''def run_rating_task(task, args):
    a, nu, ni, test_rows, folds = rating_problem(task, args.data_dir)
    qu, qi = test_rows[:, 0], test_rows[:, 1]
    y_folds = [a[part, 3].astype(float) for part in folds]
    y = np.concatenate([yf[yf > 0] for yf in y_folds])
    rows = np.concatenate([part[yf > 0] for part, yf in zip(folds, y_folds)])        # same order as the bank rows
    # Bayesian FM chains first (light on memory, one core each); their predictions become stacking features
    specs = BFM_SPECS[task][:args.bfm_specs]
    with ProcessPoolExecutor(max_workers=args.bfm_workers, mp_context=get_context('spawn')) as ex:
        chains = list(ex.map(_bfm_fold_job, [(task, args.data_dir, k, j, args.bfm_cache)
                                             for k in range(RATING_FOLDS) for j in range(len(specs))]))
    P = [np.column_stack(chains[k * len(specs):(k + 1) * len(specs)]) for k in range(RATING_FOLDS)]
    for j in range(len(specs)):
        oof = np.concatenate([P[k][:len(part), j][yf > 0] for k, (part, yf) in enumerate(zip(folds, y_folds))])
        log(f'task {task}: Bayesian FM {j} ({specs[j]}): out-of-fold MSE {np.mean((np.clip(oof, 1, 5) - y) ** 2):.5f}')
    E, Et = rating_ease_features(task, a, folds, rows, qu, qi, nu, ni)
    log(f'task {task}: EASE rating-neighbourhood features done')
    jobs = []
    for part in folds:
        keep = np.ones(len(a), bool)
        keep[part] = False
        jobs.append((task, a[keep], np.vstack([a[part], test_rows]), nu, ni))
    # Column 0 of the bank is used by neither stacker part, so it is dropped; only positive-rating rows are kept.
    X, test_blocks, row = None, [], 0
    with ProcessPoolExecutor(max_workers=args.rating_workers, mp_context=get_context('spawn')) as ex:
        for block, part, yf, Pk in zip(ex.map(_rating_fold_job, jobs), folds, y_folds, P):
            if X is None:
                nb_ = block.shape[1] - 1
                ne_ = nb_ + E.shape[1]
                X = np.empty((len(y), ne_ + len(specs)), np.float32)
            rated = yf > 0
            X[row:row + rated.sum(), :nb_] = block[:len(part)][rated, 1:]
            X[row:row + rated.sum(), ne_:] = Pk[:len(part)][rated]
            row += rated.sum()
            test_blocks.append(np.ascontiguousarray(np.hstack([block[len(part):, 1:], Et, Pk[len(part):]])))
            del block
    X[:, nb_:ne_] = E
    del jobs, E, P
    log(f'task {task}: out-of-fold feature bank done')
    metrics = {'meta_rows': int(len(y)), 'folds': RATING_FOLDS, 'n_features': int(X.shape[1])}
    model = fit_rating_meta(X, y)
    gbdt = np.mean([model(Z) for Z in test_blocks], axis=0)
    mlp = np.mean(mlp_stacker(X, y, test_blocks), axis=0)
    del X
    predictions = np.clip(0.5 * gbdt + 0.5 * mlp, 1, 5)      # GBDT and MLP stackers are equally good; the blend is better
    save_predictions(args.data_dir, args.output_dir, task, [repr(float(v)) for v in predictions])
    return metrics


'''
s = s.replace(old, new)
old = "    ap.add_argument('--rating-workers', type=int, default=4)\n"
assert old in s; s = s.replace(old, old + "    ap.add_argument('--bfm-workers', type=int, default=6, help='parallel Gibbs chains (Tasks 1 and 3)')\n"
                                       "    ap.add_argument('--bfm-cache', type=Path, default=None, help='directory to store / reuse the Gibbs chains')\n"
                                       "    ap.add_argument('--bfm-specs', type=int, default=None, help='use only the first N chain specs (default: all)')\n")
open(p, 'w').write(s)
print('written', len(s.splitlines()), 'lines')
