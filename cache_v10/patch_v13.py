"""v13/assignment1.py from v12/assignment1.py."""
import json
flags = json.load(open('/data/UCSD/CSE258/A1/cache_v10/v13_flags.json'))
s = open('/data/UCSD/CSE258/A1/v12/assignment1.py').read()
# ---- (a) Task 6: bootstrap bagging of the final model
old = '''    def fit(sets):
        """Median regression on log(1 + gap), fitted twice: the second pass weights every row by the gap the first
        pass predicts for it, so that the trees spend their splits where an error in log space costs most seconds
        (the metric is the absolute error in seconds; an unweighted fit in log space treats a 1-day and a 1-year
        gap alike).  On the Task-6-like check problems this lowered the error by 1.3% / 2.1%."""
        X = np.vstack([s[0] for s in sets])
        y = np.log1p(np.maximum(np.concatenate([s[1] for s in sets]), 0))
        H = np.concatenate([s[2] for s in sets])
        first = np.expm1(HistGradientBoostingRegressor(**T6_PARAMS).fit(X, y).predict(X))
        w = np.maximum(np.where(H > 0, np.minimum(first, H), first), DAY)
        return HistGradientBoostingRegressor(**T6_PARAMS).fit(X, y, sample_weight=w / w.mean())

    def predict(model, X, H):
        """Gaps grow toward the end of a history, so predictions are scaled up most when little time is left."""
        days = np.where(H > 0, H, np.inf) / DAY
        scale = np.select([days < lim for lim, _ in T6_SCALE], [f for _, f in T6_SCALE])
        p = np.expm1(model.predict(X)) * scale
        return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)
'''
new = '''    def two_stage(X, y, H):
        """Median regression on log(1 + gap), fitted twice: the second pass weights every row by the gap the first
        pass predicts for it, so that the trees spend their splits where an error in log space costs most seconds
        (the metric is the absolute error in seconds; an unweighted fit in log space treats a 1-day and a 1-year
        gap alike).  On the Task-6-like check problems this lowered the error by 1.3% / 2.1%."""
        first = np.expm1(HistGradientBoostingRegressor(**T6_PARAMS).fit(X, y).predict(X))
        w = np.maximum(np.where(H > 0, np.minimum(first, H), first), DAY)
        return HistGradientBoostingRegressor(**T6_PARAMS).fit(X, y, sample_weight=w / w.mean())

    def fit(sets, bags=0):
        """One model on all rows plus `bags` models on bootstrap samples of the rows (v13); the large trees vary
        a lot from sample to sample, and averaging them helped on both check problems (-0.2% / -0.3% with 3)."""
        X = np.vstack([s[0] for s in sets])
        y = np.log1p(np.maximum(np.concatenate([s[1] for s in sets]), 0))
        H = np.concatenate([s[2] for s in sets])
        models = [two_stage(X, y, H)]
        rng = np.random.default_rng(SEED)
        for _ in range(bags):
            r = np.sort(rng.choice(len(y), len(y), replace=True))
            models.append(two_stage(X[r], y[r], H[r]))
        return models

    def predict(models, X, H):
        """Average of the models in log space; gaps grow toward the end of a history, so predictions are scaled up
        most when little time is left, then capped by the time left."""
        days = np.where(H > 0, H, np.inf) / DAY
        scale = np.select([days < lim for lim, _ in T6_SCALE], [f for _, f in T6_SCALE])
        p = np.expm1(np.mean([m.predict(X) for m in models], axis=0)) * scale
        return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)
'''
assert old in s; s = s.replace(old, new)
old = '''    model = fit(lsets + csets)
    te = data.tests['task_6_test.csv']
    X, H, _ = task6_features(g_full, te['user'], te['item'])
    return [repr(float(v)) for v in predict(model, X, H)], metrics
'''
new = f'''    models = fit(lsets + csets, bags=T6_BAGS)
    te = data.tests['task_6_test.csv']
    X, H, _ = task6_features(g_full, te['user'], te['item'])
    return [repr(float(v)) for v in predict(models, X, H)], metrics
'''
assert old in s; s = s.replace(old, new)
old = "T6_CUTOFF_DAYS = (14, 30, 60, 90, 120, 180, 270, 365, 540, 730, 1095)\n"
assert old in s
s = s.replace(old, old + f"T6_BAGS = {flags.get('t6_bags', 5)}                                   # bootstrap models averaged with the full-data one (v13)\n")
# ---- (c) Bayesian FM: more entity attributes (user's first year, activity / popularity level)
old = """        side, how = {'icohort': (1, 'first'), 'imed': (1, 'median'), 'umed': (0, 'median')}[kind]"""
new = """        side, how = {'icohort': (1, 'first'), 'imed': (1, 'median'), 'umed': (0, 'median'), 'ucohort': (0, 'first'),
                     'ubin': (0, 'count'), 'ibin': (1, 'count')}[kind]"""
assert old in s; s = s.replace(old, new)
old = """    has the median of its interactions ('median'), from the training timestamps; 0 = no usable timestamp.\"\"\"
"""
new = """    has the median of its interactions ('median'), from the training timestamps; 0 = no usable timestamp.
    'count' instead gives 1 + floor(log2(number of training interactions)), capped at 14.\"\"\"
    if how == 'count':
        return np.minimum(np.floor(np.log2(np.maximum(np.bincount(a[:, col], minlength=n), 1))).astype(np.int64) + 1, 14)
"""
assert old in s, 'docstring anchor'
s = s.replace(old, new)
# ---- (b) Bayesian FM specs
if flags.get('specs'):
    a = s.index("# Bayesian FM chains (GPU).")
    b = s.index("def rating_problem(task, data_dir):")
    s = s[:a] + open(flags["specs"]).read().strip() + "\n\n\n" + s[b:]
open('/data/UCSD/CSE258/A1/v13/assignment1.py', 'w').write(s)
print('v13 written', len(s.splitlines()), 'lines')
