"""Fold-0 GPU run with the NON-collapsed order (own factors given bags, then bags given own factors by CG; same order
as the CPU sampler) to test whether the collapsed step changes the result."""
import sys, time, numpy as np, torch, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('a1v11n', '/data/UCSD/CSE258/A1/v11/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v11n'] = M; spec.loader.exec_module(M)
src = open('/data/UCSD/CSE258/A1/cache_v10/run_f0.py').read()
# patch the class: replace the collapsed block of sweep() by the non-collapsed one
import inspect, textwrap
code = inspect.getsource(M.BayesianFM.sweep)
old = code[code.index("                if 'Y' in s:"):code.index("                s['a'][f] = af + d\n")]
new = """                prec = lv + aA
                Vn = (lv * mv + aA * Vf + al * G) / prec + torch.randn_like(A) / torch.sqrt(prec)
                dv = Vn - Vf
                s['V'][f] = Vn
                G = G - A * dv
                d = dv
                if 'Y' in s:
                    Yf = s['Y'][f]
                    s['lam_y'][f], s['mu_y'][f] = _t_hyper(Yf, s['lam_y'][f])
                    lam, mu = s['lam_y'][f], s['mu_y'][f]
                    rhs = torch.sparse.mm(s['CT'], (al * G + torch.sqrt(aA.clamp_min(0)) * torch.randn_like(A)).T.contiguous()) \\
                        + (lam[:, None] * (mu[:, None] - Yf)).T + torch.sqrt(lam)[None, :] * torch.randn(s['J'], F, device='cuda')
                    diag = torch.sparse.mm(s['C2T'], aA.T.contiguous()) + lam[None, :]
                    self.cg_iters = 120
                    delta = self._pcg(s, aA, rhs, lam, 1.0 / diag)
                    s['Y'][f] = Yf + delta.T
                    d = d + torch.sparse.mm(s['C'], delta).T
"""
code = code.replace(old, new)
code = textwrap.dedent(code)
ns = vars(M)
exec(compile(code, 'sweep_nc', 'exec'), ns)
M.BayesianFM.sweep = ns['sweep']
sys.argv = ['run_f0.py'] + sys.argv[1:]
exec(compile(src.replace("spec = importlib.util.spec_from_file_location('a1v11f', '/data/UCSD/CSE258/A1/v11/assignment1.py')\nM = importlib.util.module_from_spec(spec); sys.modules['a1v11f'] = M; spec.loader.exec_module(M)\n", "").replace("print(f'GPU", "print(f'GPU non-collapsed"), 'run_f0', 'exec'), {'M': M, '__name__': '__main__'})
