"""Diagnose which Task 5 feature columns break ID-permutation invariance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import exec_interactions as ex

a, _, _ = ex.load_task5_order_only(ex.DATA_DIR / 'task_5_train.csv')
sub = a[a[:, 0] < 3000].copy()
_, ui = np.unique(sub[:, 0], return_inverse=True)
_, ii = np.unique(sub[:, 1], return_inverse=True)
sub[:, 0], sub[:, 1] = ui, ii
nu, ni = ui.max() + 1, ii.max() + 1
fit, va, _ = ex.make_interaction_splits_dual(sub, nu, ni, sequential=True, seed=258)
va = va[:500]


def feats(f, qu, q1, q2):
    st = ex.fit_graph_state(5, f, nu, ni, compute_bt_diag=False, equivariant_mode=True)
    x0, d, c, s = ex.extract_query_features(st, qu, q1, q2, np.zeros(len(qu), np.int64))
    return x0, d, c


x0a, da, ca = feats(fit, va[:, 0], va[:, 1], va[:, 2])
rng = np.random.default_rng(2026)
pu, pi = rng.permutation(nu), rng.permutation(ni)
fp = fit.copy(); fp[:, 0] = pu[fit[:, 0]]; fp[:, 1] = pi[fit[:, 1]]
x0b, db, cb = feats(fp, pu[va[:, 0]], pi[va[:, 1]], pi[va[:, 2]])
diff = np.max(np.abs(da - db), axis=0)
scale = np.max(np.abs(da), axis=0) + 1e-12
print('n_x0 cols', x0a.shape[1], 'n_d cols', da.shape[1])
for k in np.argsort(-diff)[:15]:
    print(f'd col {k}: maxdiff {diff[k]:.4g} (rel {diff[k]/scale[k]:.3g})')

# Re-run with a tighter Lanczos subspace to test whether the drift is numerical.
from scipy.sparse.linalg import svds as _svds
ex.svds = lambda x, k, **kw: _svds(x, k=k, ncv=min(min(x.shape) - 1, 4 * k), **kw)
x0a, da, ca = feats(fit, va[:, 0], va[:, 1], va[:, 2])
x0b, db, cb = feats(fp, pu[va[:, 0]], pi[va[:, 1]], pi[va[:, 2]])
diff = np.max(np.abs(da - db), axis=0)
print('ncv=4k max diff over spectral cols 23..32:', diff[23:33].round(8).tolist())
print('max diff elsewhere:', float(np.max(np.r_[diff[:23], diff[33:]])))
