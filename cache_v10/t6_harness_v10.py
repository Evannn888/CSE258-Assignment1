import sys, numpy as np, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
V10 = load_mod('/data/UCSD/CSE258/A1/v10/assignment1.py', 'a1v10h')
for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    t0 = time.time()
    pred, met = V10.run_task6(data)
    p = np.array([float(x) for x in pred])
    print(f'{name} v10: harness MAE {np.mean(np.abs(p - truth))/1e6:.4f}M  (L1 metric {met["validation_mae_L1_seconds"]/1e6:.3f}M) {time.time()-t0:.0f}s', flush=True)
