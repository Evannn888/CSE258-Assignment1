import sys, numpy as np
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8'); sys.path.insert(0, '/data/UCSD/CSE258/A1')
from t6_harness import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
for name in ['tasks_3_4_train.csv', 'tasks_1_2_train.csv']:
    d = load_raw(name)
    gfull = Graph(d.u, d.i, d.t, d.r, d.nU, d.nI)
    data, truth = t6_like(d)
    users = data.tests['task_6_test.csv']['user']
    fin = gfull.t[gfull.uptr[users + 1] - 1]
    bad = fin > A.TEND
    gf = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    past = np.bincount(gf.u[gf.t < 946684800], minlength=d.nU)[users]
    qi = data.tests['task_6_test.csv']['item']
    print(name, 'bad', bad.sum(), 'P(bad | has garbage-past)', bad[past > 0].mean().round(3), 'n', (past > 0).sum(),
          'P(bad | none)', bad[past == 0].mean().round(4))
    print('   final item train count: bad median', np.median(gf.icnt[qi[bad]]), 'others', np.median(gf.icnt[qi[~bad]]))
    # deltas of bad users
    print('   bad deltas (years):', np.sort(truth[bad] / 31557600).round(1)[::5])
    print('   bad final years:', np.unique((1970 + fin[bad] / 31557600).astype(int), return_counts=True))
    X, H, tl = feats(gf, users, qi, A.TEND)
    X = np.column_stack([X, np.log1p(past), gf.r[gf.uptr[users + 1] - 1]])
    half = np.random.default_rng(0).permutation(len(users)) % 2 == 0
    pr = np.zeros(len(users))
    for part in (half, ~half):
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(X[~part], bad[~part])
        pr[part] = c.predict_proba(X[part])[:, 1]
    print('   classifier AUC', roc_auc_score(bad, pr).round(3), 'max prob', pr.max().round(3), 'n prob>0.5', (pr > 0.5).sum(), 'top-20 precision', bad[np.argsort(-pr)[:20]].mean())
