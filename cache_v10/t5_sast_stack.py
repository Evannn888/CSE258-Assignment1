"""Task 5 stack check (cached v6 features + latent + SASRec): old SASRec vs time-aware SASRec vs both."""
import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
tag = sys.argv[1] if len(sys.argv) > 1 else 'gap+month_e20_d64'
z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz'); s = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sasrec.npz')
new = {k: np.load(f'/data/UCSD/CSE258/A1/cache_v10/sast_{tag}_s{k}.npy') for k in range(5)}
for k in range(5):
    print(f'set {k}: single acc old {np.mean(s[f"S{k}"][:, 0] > s[f"S{k}"][:, 1]):.4f} new {np.mean(new[k][:, 0] > new[k][:, 1]):.4f}', flush=True)
def build(k, mode):
    F, ctx, seen, L = z[f'F{k}'], z[f'ctx{k}'], z[f'seen{k}'], z[f'L{k}']
    parts = [F, L]
    if mode in ('old', 'both'):
        parts.append(s[f'S{k}'][..., None])
    if mode in ('new', 'both'):
        parts.append(new[k][..., None])
    base = np.concatenate(parts, -1)
    return [(base[:, [0, dd]], ctx, seen[:, [0, dd]]) for dd in (1, 2, 3)]
for mode in ('old', 'new', 'both'):
    t0 = time.time()
    train = [x for k in (1, 2, 3, 4) for x in build(k, mode)]
    clf = V.fit_pair_classifier([(F, c) for F, c, _ in train], 500)
    accs = []
    for F, ctx, seen in build(0, mode):
        m = V.pair_margin(clf, F[:, 0], F[:, 1], ctx) + 100 * (seen[:, 1] - seen[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T5 SASRec = {mode}: val acc per negative draw {np.round(accs, 4)} mean {np.mean(accs):.4f} ({time.time() - t0:.0f}s)', flush=True)
