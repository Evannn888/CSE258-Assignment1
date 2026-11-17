"""Compare fitted Task 5 state arrays under an ID permutation."""
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
rng = np.random.default_rng(2026)
pu, pi = rng.permutation(nu), rng.permutation(ni)
fp = fit.copy(); fp[:, 0] = pu[fit[:, 0]]; fp[:, 1] = pi[fit[:, 1]]
A = ex.fit_graph_state(5, fit, nu, ni, equivariant_mode=True)
B = ex.fit_graph_state(5, fp, nu, ni, equivariant_mode=True)
piu = np.r_[pi, ni]; puu = np.r_[pu, nu]
for key, perm in [('u_vec', puu), ('i_vec', piu), ('u_unit_ext', puu), ('i_unit', piu), ('v_item_ext', piu), ('u_decay', puu)]:
    x, y = A[key], B[key][perm]
    print(key, x.shape, 'maxdiff', float(np.max(np.abs(x - y))))
print('singular values tail', np.sort(np.linalg.norm(A['u_vec'][:-1], axis=0))[:5])

