import sys, numpy as np, importlib.util, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
from t6_harness import load_raw, t6_like
def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
V7 = load_mod('/data/UCSD/CSE258/A1/assignment1.py', 'a1v7')
V8 = load_mod('/data/UCSD/CSE258/A1/v8/assignment1.py', 'a1v8')
for name in sys.argv[1:]:
    d = load_raw(name)
    data, truth = t6_like(d)
    for tag, M in (('v7', V7), ('v8', V8)):
        t0 = time.time()
        pred, met = M.run_task6(data)
        p = np.array([float(x) for x in pred])
        print(f'{name} {tag}: harness MAE {np.mean(np.abs(p - truth))/1e6:.4f}M  (L1 metric {met["validation_mae_L1_seconds"]/1e6:.3f}M) {time.time()-t0:.0f}s', flush=True)
