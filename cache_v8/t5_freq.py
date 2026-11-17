import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
z = np.load('cache_v7/t5_sets.npz')
s = np.load('cache_v7/t5_sasrec.npz')
nI = int(z['nI'])


def cfeat(qpos, qneg, hi):
    c = np.bincount(hi, minlength=nI).astype(float)
    n = len(qpos)
    k = np.bincount(qpos, minlength=nI) + np.bincount(qneg, minlength=nI)
    e = n * c / c.sum()
    f = lambda x: np.column_stack([np.log((k[x] + 0.5) / (e[x] + 0.5)), np.log1p(k[x]), (k[x] - e[x]) / np.maximum(k[x], 1)])
    return np.stack([f(qpos), f(qneg)], 1).astype(np.float32)


def build(k, use_freq):
    F, ctx, seen, L, qc, S = z[f'F{k}'], z[f'ctx{k}'], z[f'seen{k}'], z[f'L{k}'], z[f'qc{k}'], s[f'S{k}']
    base = np.concatenate([F, L, S[..., None]], -1)
    out = []
    for d in (1, 2, 3):
        Fd = base[:, [0, d]]
        if use_freq:
            Fd = np.concatenate([Fd, cfeat(qc[:, 0], qc[:, d], z[f'hi{k}'])], -1)
        out.append((Fd, ctx, seen[:, [0, d]]))
    return out


if __name__ == '__main__':
  for use_freq in (False, True):
    t0 = time.time()
    train = [x for k in (1, 2, 3, 4) for x in build(k, use_freq)]
    clf = V.fit_pair_classifier([(F, c) for F, c, _ in train], 500)
    accs = []
    for F, ctx, seen in build(0, use_freq):
        m = V.pair_margin(clf, F[:, 0], F[:, 1], ctx) + 100 * (seen[:, 1] - seen[:, 0])
        accs.append(np.mean(m > 0))
    print('freq' if use_freq else 'base', 'val acc per draw', np.round(accs, 4), 'mean', np.mean(accs).round(4), f'{time.time()-t0:.0f}s', flush=True)
