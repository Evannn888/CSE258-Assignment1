import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import diag_t5_exact_svd as d  # reruns check; reuses its objects
ex = d.ex
A = ex.fit_graph_state(5, d.fit, d.nu, d.ni, equivariant_mode=True)
B = ex.fit_graph_state(5, d.fp, d.nu, d.ni, equivariant_mode=True)
qu, q1, q2 = d.va[:, 0], d.va[:, 1], d.va[:, 2]
fa = ex.a1.spectral_history_pooling(qu, q1, q2, A['u_ptr'], A['u_items'], A['i_unit'])
fb = ex.a1.spectral_history_pooling(d.pu[qu], d.pi[q1], d.pi[q2], B['u_ptr'], B['u_items'], B['i_unit'])
r = int(np.argmax(np.abs(fa - fb).max(1)))
u = qu[r]
ha = A['u_items'][A['u_ptr'][u]:A['u_ptr'][u+1]]
hb = B['u_items'][B['u_ptr'][d.pu[u]]:B['u_ptr'][d.pu[u]+1]]
print('row', r, fa[r], fb[r])
print('history equal under perm:', np.array_equal(d.pi[ha], hb), len(ha))
print('i_unit equal rows:', float(np.abs(A['i_unit'][ha] @ A['i_unit'][q1[r]] - B['i_unit'][hb] @ B['i_unit'][d.pi[q1[r]]]).max()))
print('i_unit dtype', A['i_unit'].dtype, A['i_unit'].shape)
