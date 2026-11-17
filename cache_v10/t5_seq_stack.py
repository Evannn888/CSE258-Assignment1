"""Task 5 stack check (cached sets, small GBDT): which SASRec features help.
variants (argv): names among new, avg3, big, new+big, avg3+big"""
import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz')
C = '/data/UCSD/CSE258/A1/cache_v10/'
ld = lambda tag, k: np.load(C + f'sast_{tag}_s{k}.npy')
feats = {'new': lambda k: [ld('gap+month_e20_d64', k)],
         'avg3': lambda k: [(ld('gap+month_e20_d64', k) + ld('gap+month_e20_d64_r2', k) + ld('gap+month_e20_d64_r3', k)) / 3],
         'big': lambda k: [ld('gap+month_e30_d128', k)],
         'L200': lambda k: [ld('gap+month_e20_d64_L200', k)],
         'n16': lambda k: [ld('gap+month_e20_d64_n16', k)],
         'L200n16': lambda k: [ld('gap+month_e20_d64_L200n16', k)],
         'n64': lambda k: [ld('gap+month_e20_d64_n64', k)]}
def build(k, variant):
    F, ctx, seen, L = z[f'F{k}'], z[f'ctx{k}'], z[f'seen{k}'], z[f'L{k}']
    parts = [F, L] + [x[..., None] for name in variant.split('+') for x in feats[name](k)]
    base = np.concatenate(parts, -1)
    return [(base[:, [0, dd]], ctx, seen[:, [0, dd]]) for dd in (1, 2, 3)]
for variant in sys.argv[1:]:
    t0 = time.time()
    train = [x for k in (1, 2, 3, 4) for x in build(k, variant)]
    clf = V.fit_pair_classifier([(F, c) for F, c, _ in train], 500)
    accs = []
    for F, ctx, seen in build(0, variant):
        m = V.pair_margin(clf, F[:, 0], F[:, 1], ctx) + 100 * (seen[:, 1] - seen[:, 0])
        accs.append(np.mean(m > 0))
    print(f'T5 {variant}: val acc per negative draw {np.round(accs, 4)} mean {np.mean(accs):.4f} ({time.time() - t0:.0f}s)', flush=True)
