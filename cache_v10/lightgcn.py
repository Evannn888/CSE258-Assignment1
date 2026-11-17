"""LightGCN (He et al. 2020) for the pair tasks: BPR with popularity negatives (the test's negative distribution).
Scores the cached candidates of each fold of cache_v7/pairs_t{task}.npz.  usage: lightgcn.py TASK folds d layers epochs"""
import sys, time, numpy as np, torch
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
task = int(sys.argv[1]); folds = [int(x) for x in sys.argv[2].split(',')]
d, L, EP = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
lr, reg, B = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3, float(sys.argv[7]) if len(sys.argv) > 7 else 1e-4, 4096
dev = 'cuda'
z = np.load(f'/data/UCSD/CSE258/A1/cache_v7/pairs_t{task}.npz')
u_all, i_all, nU, nI = z['u'], z['i'], int(z['nU']), int(z['nI'])
for f in folds:
    t0 = time.time()
    torch.manual_seed(f)
    held = z[f'held{f}'] if f >= 0 else np.zeros(0, np.int64)
    keep = np.ones(len(u_all), bool); keep[held] = False
    u, i = u_all[keep], i_all[keep]
    key = np.unique(u.astype(np.int64) * nI + i); uu, ii = key // nI, key % nI
    N = nU + nI
    deg = np.bincount(uu, minlength=nU).astype(float), np.bincount(ii, minlength=nI).astype(float)
    dd = np.r_[deg[0], deg[1]]
    r = np.r_[uu, nU + ii]; c = np.r_[nU + ii, uu]
    w = 1.0 / np.sqrt(np.maximum(dd[r], 1) * np.maximum(dd[c], 1))
    A = torch.sparse_coo_tensor(torch.as_tensor(np.vstack([r, c])), torch.as_tensor(w, dtype=torch.float32), (N, N)).coalesce().to(dev).to_sparse_csr()
    E0 = torch.nn.Parameter(torch.randn(N, d, device=dev) * 0.1)
    opt = torch.optim.Adam([E0], lr=lr)
    pu = torch.as_tensor(u, device=dev); pi = torch.as_tensor(i, device=dev)
    def prop():
        e, acc = E0, E0
        for _ in range(L):
            e = torch.sparse.mm(A, e); acc = acc + e
        return acc / (L + 1)
    n = len(u)
    for ep in range(EP):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for b in range(0, n, B):
            idx = perm[b:b + B]
            bu, bi = pu[idx], pi[idx]
            bj = pi[torch.randint(0, n, (len(idx),), device=dev)]          # item of a random interaction
            E = prop()
            eu, ei, ej = E[bu], E[nU + bi], E[nU + bj]
            x = (eu * (ei - ej)).sum(1)
            loss = torch.nn.functional.softplus(-x).mean() + reg * (E0[bu].pow(2).sum(1) + E0[nU + bi].pow(2).sum(1) + E0[nU + bj].pow(2).sum(1)).mean() / 2
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 10 == 9 or ep == EP - 1:
            with torch.no_grad():
                E = prop()
                qu = torch.as_tensor(u_all[held], device=dev); qc = torch.as_tensor(z[f'qc{f}'], device=dev)
                S = (E[qu][:, None, :] * E[nU + qc]).sum(2).cpu().numpy()
            seen = z[f'seen{f}']
            print(f'fold {f} ep {ep + 1}: loss {tot / n:.4f}, pair acc {np.mean(S[:, 0] > S[:, 1]):.4f} (seen rule {np.mean((S[:, 0] - S[:, 1] + 1e6 * (seen[:, 1] - seen[:, 0])) > 0):.4f}) {time.time() - t0:.0f}s', flush=True)
    np.save(f'/data/UCSD/CSE258/A1/cache_v10/lgcn_t{task}_d{d}_L{L}_e{EP}_f{f}.npy', S.astype(np.float32))
