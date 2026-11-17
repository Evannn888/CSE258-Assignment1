"""v10 Tasks 2/4: predicted rating (Bayesian FM) of every candidate and its posterior sd as two more pair features."""
p = '/data/UCSD/CSE258/A1/v10/assignment1.py'
s = open(p).read()
old = "def run_pair_task(task, data, folds=4, holdout=30000, negatives=2):\n"
new = '''PAIR_RATING_BFM = dict(k=16, iters=300, burn=60)


def _pair_rating_job(args):
    fit, qu, qc, nU, nI, seed = args
    q = np.column_stack([np.repeat(qu, qc.shape[1]), qc.reshape(-1), np.zeros(qc.size, np.int64)])
    mean, sd = bfm_rating_predictions(fit, q, nU, nI, seed, known_queries=False, **PAIR_RATING_BFM)
    return np.stack([mean.reshape(qc.shape), sd.reshape(qc.shape)], -1).astype(np.float32)


def pair_rating_scores(task, problems, nU, nI):
    """Which rating would the user give each candidate, and how sure is the model?  A Bayesian FM is trained on the
    ratings of the problem's interactions (one chain per validation fold and one on all data; the candidates are not
    treated as known interactions).  Real interactions get a slightly higher predicted rating (4.07 vs 3.91) and,
    more usefully, a smaller posterior sd than popularity-sampled fakes."""
    jobs = [(np.column_stack([g.u, g.i, g.t, g.r]).astype(np.int64), qu, qc, nU, nI, SEED + 10 * task + k)
            for k, (g, qu, qc, qt) in enumerate(problems)]
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=get_context('spawn')) as ex:
        return list(ex.map(_pair_rating_job, jobs))


def run_pair_task(task, data, folds=4, holdout=30000, negatives=2):
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = "    log(f'task {task}: latent models trained')\n"
new = "    log(f'task {task}: latent models trained')\n    rate = pair_rating_scores(task, problems, data.nU, data.nI)\n    log(f'task {task}: predicted-rating features done')\n"
assert s.count(old) == 1; s = s.replace(old, new)
old = "    sets = [(np.concatenate([F, latent[f], ease[f]], -1), ctx, seen) for f, (F, ctx, seen) in enumerate(sets)]\n"
new = "    sets = [(np.concatenate([F, latent[f], ease[f], rate[f]], -1), ctx, seen) for f, (F, ctx, seen) in enumerate(sets)]\n"
assert s.count(old) == 1; s = s.replace(old, new)
old = "    F = np.concatenate([F, latent[-1], ease[-1]], -1)\n"
new = "    F = np.concatenate([F, latent[-1], ease[-1], rate[-1]], -1)\n"
assert s.count(old) == 1; s = s.replace(old, new)
open(p, 'w').write(s)
print('pairs patch applied')
