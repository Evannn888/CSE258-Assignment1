"""I-AutoRec (Sedhain et al. 2015) for explicit positive ratings; quick single-fold check for Task 1/3."""
import sys, numpy as np, time, torch, torch.nn as nn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
V.torch_setup()
dev = 'cuda'


def train_predict(fit, nu, ni, qu, qi, k=500, epochs=60, lr=1e-3, wd=0.0, lam=1.0, batch=256, seed=0, drop=0.0, evalfn=None):
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    pos = fit[:, 3] > 0
    u, i, r = fit[pos, 0], fit[pos, 1], fit[pos, 3].astype(np.float32)
    mu = r.mean()
    # item-major CSR
    o = np.lexsort((u, i))
    u, i, r = u[o], i[o], r[o]
    iptr = np.searchsorted(i, np.arange(ni + 1))
    U = torch.tensor(u, device=dev); R = torch.tensor(r - mu, device=dev)
    enc = nn.Linear(nu, k).to(dev); dec = nn.Linear(k, nu).to(dev)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    items = np.flatnonzero(np.diff(iptr) > 0)
    dropout = nn.Dropout(drop)

    def batch_dense(its):
        a, b = iptr[its], iptr[its + 1]
        lens = b - a
        rows = np.repeat(np.arange(len(its)), lens)
        idx = np.concatenate([np.arange(x, y) for x, y in zip(a, b)])
        X = torch.zeros((len(its), nu), device=dev)
        M = torch.zeros((len(its), nu), device=dev)
        ri = torch.tensor(rows, device=dev); ui = U[torch.tensor(idx, device=dev)]
        X[ri, ui] = R[torch.tensor(idx, device=dev)]
        M[ri, ui] = 1.0
        return X, M

    for ep in range(epochs):
        enc.train(); dec.train()
        p = rs.permutation(items)
        tot = 0.0
        for b0 in range(0, len(p), batch):
            X, M = batch_dense(p[b0:b0 + batch])
            out = dec(torch.sigmoid(enc(dropout(X))))
            loss = (((out - X) * M) ** 2).sum() / M.sum() + lam * 1e-4 * (enc.weight.pow(2).sum() + dec.weight.pow(2).sum()) / len(items) * batch
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if evalfn is not None and (ep % 10 == 9 or ep == epochs - 1):
            print(f'  ep {ep+1} loss {tot/(len(p)/batch):.4f} eval {evalfn(predict(enc, dec, batch_dense, qu, qi, ni, iptr, mu))}', flush=True)
    return predict(enc, dec, batch_dense, qu, qi, ni, iptr, mu)


def predict(enc, dec, batch_dense, qu, qi, ni, iptr, mu):
    enc.eval(); dec.eval()
    out = np.full(len(qu), np.nan, np.float32)
    with torch.no_grad():
        o = np.argsort(qi)
        uq = np.unique(qi[o])
        uq = uq[(uq < ni)]
        for b0 in range(0, len(uq), 512):
            its = uq[b0:b0 + 512]
            X, M = batch_dense(its)
            Y = dec(torch.sigmoid(enc(X))).cpu().numpy() + mu
            pos = {it: j for j, it in enumerate(its)}
            sel = np.flatnonzero(np.isin(qi, its))
            out[sel] = Y[[pos[x] for x in qi[sel]], qu[sel]]
    return out


if __name__ == '__main__':
    task = int(sys.argv[1])
    D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
    a, users, items = V.load(D / V.FILES[task])
    nu, ni = len(users), len(items)
    folds = np.array_split(np.random.default_rng(11).permutation(len(a)), V.RATING_FOLDS)
    part = folds[0]
    keep = np.ones(len(a), bool); keep[part] = False
    q = a[part]; q = q[q[:, 3] > 0]
    y = q[:, 3].astype(float)
    pos = a[keep][:, 3] > 0
    mu = a[keep][pos, 3].mean()
    print('fold0 rated rows', len(y), 'global-mean MSE', np.mean((y - mu) ** 2))
    def ev(p):
        p = np.where(np.isfinite(p), p, mu)
        return f'MSE {np.mean((np.clip(p, 1, 5) - y) ** 2):.4f}'
    cfg = dict(k=int(sys.argv[2]) if len(sys.argv) > 2 else 500, epochs=int(sys.argv[3]) if len(sys.argv) > 3 else 60,
               lam=float(sys.argv[4]) if len(sys.argv) > 4 else 1.0, drop=float(sys.argv[5]) if len(sys.argv) > 5 else 0.0)
    t0 = time.time()
    p = train_predict(a[keep], nu, ni, q[:, 0], q[:, 1], evalfn=ev, **cfg)
    print(cfg, 'final', ev(p), f'{time.time()-t0:.0f}s')
    np.save(f'/data/UCSD/CSE258/A1/cache_v8/autorec_t{task}_f0.npy', p)
