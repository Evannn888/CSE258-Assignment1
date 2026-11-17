"""BERT4Rec trained with sampled softmax over K shared popularity negatives (Task 4 fill-in-the-blank)."""
import sys, numpy as np, time, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
V.torch_setup()
dev = 'cuda'


def b4r_ce_scores(g, qu, qt, qc, seed, d=128, L=100, blocks=2, drop=0.3, epochs=60, lr=1e-3, wd=0.1, K=4096, pmask=0.2,
                  batch=256, evalfn=None, every=20):
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    nI, PAD, MASK = g.nI, 0, g.nI + 1
    starts, lens = g.uptr[:-1], np.diff(g.uptr)

    class B4R(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(nI + 2, d, padding_idx=0)
            self.pos = nn.Embedding(L, d)
            self.bias = nn.Embedding(nI + 2, 1)
            layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=4 * d, dropout=drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)
            nn.init.normal_(self.emb.weight, std=0.02)
            nn.init.zeros_(self.bias.weight)

        def forward(self, s):
            return self.ln(self.enc(self.drop(self.emb(s) * d ** 0.5 + self.pos.weight[None]), src_key_padding_mask=(s == PAD)))
    m = B4R().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=wd)
    gen = torch.Generator(device=dev).manual_seed(seed)
    pool = torch.tensor(g.i + 1, device=dev)
    users = np.flatnonzero(lens >= 2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, pct_start=0.1, total_steps=epochs * ((len(users) + batch - 1) // batch))
    # query windows (fixed)
    Q = np.zeros((len(qu), L), np.int64)
    qpos = np.zeros(len(qu), np.int64)
    for r, (u, t) in enumerate(zip(qu, qt)):
        a, b = g.uptr[u], g.uptr[u + 1]
        p = a + np.searchsorted(g.t[a:b], t)
        left = g.i[max(a, p - L // 2):p] + 1
        right = g.i[p:min(b, p + L - 1 - len(left))] + 1
        seq = np.r_[left, MASK, right]
        Q[r, :len(seq)] = seq
        qpos[r] = len(left)

    def score():
        m.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(Q), 1024):
                Qt = torch.tensor(Q[b:b + 1024], device=dev)
                h = m(Qt)[torch.arange(len(Qt), device=dev), torch.tensor(qpos[b:b + 1024], device=dev)]
                c = torch.tensor(qc[b:b + 1024] + 1, device=dev)
                out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
        return np.vstack(out).astype(np.float32)

    for ep in range(epochs):
        m.train()
        p = rs.permutation(users)
        for b in range(0, len(p), batch):
            ub = p[b:b + batch]
            S = np.zeros((len(ub), L), np.int64)
            for r, u in enumerate(ub):
                o = rs.integers(0, max(lens[u] - L, 0) + 1)
                w = g.i[starts[u] + o:starts[u] + o + L] + 1
                S[r, :len(w)] = w
            S = torch.tensor(S, device=dev)
            msk = (torch.rand(S.shape, device=dev, generator=gen) < pmask) & (S != PAD)
            h = m(torch.where(msk, torch.full_like(S, MASK), S))[msk]
            tgt = S[msk]
            neg = pool[torch.randint(0, len(pool), (K,), device=dev, generator=gen)]
            pl = (h * m.emb(tgt)).sum(-1) + m.bias(tgt)[:, 0]
            nl = (h @ m.emb(neg).T + m.bias(neg)[:, 0][None]).masked_fill(neg[None, :] == tgt[:, None], float('-inf'))
            loss = Fn.cross_entropy(torch.cat([pl[:, None], nl], 1), torch.zeros(len(tgt), dtype=torch.long, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        torch.cuda.synchronize()
        if evalfn is not None and (ep % every == every - 1):
            print(f'   ep {ep+1}: {evalfn(score())}', flush=True)
    out = score()
    del m
    torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz')
    old = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
    u, i, t, r, nU, nI = z['u'], z['i'], z['t'], z['r'], int(z['nU']), int(z['nI'])
    f = 0
    keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
    g = V.Graph(u[keep], i[keep], t[keep], r[keep], nU, nI)
    qu, qc, qt = u[z[f'held{f}']], z[f'qc{f}'], z[f'qt{f}']
    ev = lambda S: ' '.join(f'{np.mean(S[:, 0] > S[:, j]):.4f}' for j in (1, 2))
    print('old BERT4Rec fold 0 standalone', ev(old['S0']), flush=True)
    cfgs = {'d128_drop5_wd3_e120': dict(drop=0.5, wd=0.3, epochs=120, every=40), 'd128_drop3_wd1_e60': dict(), 'd128_drop5_wd3_e60': dict(drop=0.5, wd=0.3), 'd256_drop5_wd3_e60': dict(d=256, drop=0.5, wd=0.3)}
    for name in sys.argv[1:]:
        t0 = time.time()
        S = b4r_ce_scores(g, qu, qt, qc, seed=0, evalfn=ev, **cfgs[name])
        print(name, 'final', ev(S), f'{time.time()-t0:.0f}s', flush=True)
