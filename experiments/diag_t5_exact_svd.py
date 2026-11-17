"""Check Task 5 ID invariance when the truncated SVD is computed exactly (dense)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import exec_interactions as ex


def exact_svds(x, k, **_):
    # Deterministic dense eigendecomposition of the small user Gram matrix.
    g = (x @ x.T).toarray()
    w, u = np.linalg.eigh(g)
    idx = np.argsort(w)[::-1][:k][::-1]
    s = np.sqrt(np.maximum(w[idx], 0.0))
    u = u[:, idx]
    vt = (x.T @ u).T / np.maximum(s, 1e-12)[:, None]
    return u, s, np.asarray(vt)


ex.svds = exact_svds
a, _, _ = ex.load_task5_order_only(ex.DATA_DIR / 'task_5_train.csv')
sub = a[a[:, 0] < 3000].copy()
_, ui = np.unique(sub[:, 0], return_inverse=True)
_, ii = np.unique(sub[:, 1], return_inverse=True)
sub[:, 0], sub[:, 1] = ui, ii
nu, ni = ui.max() + 1, ii.max() + 1
fit, va, _ = ex.make_interaction_splits_dual(sub, nu, ni, sequential=True, seed=258)
va = va[:500]


def feats(f, qu, q1, q2):
    st = ex.fit_graph_state(5, f, nu, ni, equivariant_mode=True)
    return ex.extract_query_features(st, qu, q1, q2, np.zeros(len(qu), np.int64))[1]


rng = np.random.default_rng(2026)
pu, pi = rng.permutation(nu), rng.permutation(ni)
fp = fit.copy(); fp[:, 0] = pu[fit[:, 0]]; fp[:, 1] = pi[fit[:, 1]]
da = feats(fit, va[:, 0], va[:, 1], va[:, 2])
db = feats(fp, pu[va[:, 0]], pi[va[:, 1]], pi[va[:, 2]])
diff = np.max(np.abs(da - db), axis=0)
print('exact-SVD spectral cols 23..32 diff:', diff[23:33].round(10).tolist())
print('exact-SVD overall max diff:', float(diff.max()))

