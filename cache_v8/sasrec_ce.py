"""SASRec trained with sampled softmax over K shared popularity negatives (no logQ correction -> learns the lift
log p(i|h) - log pop(i), which is exactly the log-odds for 'true next item vs popularity negative')."""
import sys, numpy as np, time, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
V.torch_setup()
dev = 'cuda'


class SASRec(nn.Module):
    def __init__(self, nI, d, L, blocks, heads, drop):
        super().__init__()
        self.d, self.L = d, L
        self.emb = nn.Embedding(nI + 1, d, padding_idx=0)
        self.pos = nn.Embedding(L, d)
        self.bias = nn.Embedding(nI + 1, 1)
        layer = nn.TransformerEncoderLayer(d, nhead=heads, dim_feedforward=4 * d, dropout=drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.zeros_(self.bias.weight)

    def forward(self, s):
        x = self.drop(self.emb(s) * self.d ** 0.5 + self.pos.weight[None])
        causal = torch.triu(torch.full((self.L, self.L), float('-inf'), device=s.device), 1)
        return self.ln(self.enc(x, mask=causal, src_key_padding_mask=(s == 0)))


def make_seq(hu, hi, nU, L):
    o = np.lexsort((np.arange(len(hu)), hu))
    hu, hi = hu[o], hi[o]
    ptr = np.searchsorted(hu, np.arange(nU + 1))
    seq = np.zeros((nU, L), np.int64)
    for u in range(nU):
        it = hi[ptr[u]:ptr[u + 1]][-L:] + 1
        if len(it):
            seq[u, -len(it):] = it
    return seq, hi


def train_score(hu, hi, nU, nI, users, cands, seed, d=128, L=100, blocks=2, heads=2, drop=0.2, epochs=30, lr=1e-3,
                K=4096, batch=256, wd=0.0, verbose=False, evalfn=None):
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    seq, hi_sorted = make_seq(hu, hi, nU, L)
    S = torch.tensor(seq, device=dev)
    pool = torch.tensor(hi_sorted + 1, device=dev)
    m = SASRec(nI, d, L, blocks, heads, drop).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=wd)
    gen = torch.Generator(device=dev).manual_seed(seed)
    tu = np.flatnonzero((seq > 0).sum(1) >= 2)
    steps = epochs * ((len(tu) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        m.train()
        p = rs.permutation(tu)
        tot = 0.0
        for b in range(0, len(p), batch):
            s = S[torch.tensor(p[b:b + batch], device=dev)]
            inp = torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], 1)
            h = m(inp)
            valid = (s > 0) & (inp > 0)
            hv = h[valid]                         # (N, d)
            tv = s[valid]                         # (N,)
            neg = pool[torch.randint(0, len(pool), (K,), device=dev, generator=gen)]
            pl = (hv * m.emb(tv)).sum(-1) + m.bias(tv)[:, 0]
            nl = hv @ m.emb(neg).T + m.bias(neg)[:, 0][None]
            nl = nl.masked_fill(neg[None, :] == tv[:, None], float('-inf'))
            logits = torch.cat([pl[:, None], nl], 1)
            loss = Fn.cross_entropy(logits, torch.zeros(len(tv), dtype=torch.long, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        torch.cuda.synchronize()
        if verbose and (evalfn is not None) and (ep % 5 == 4 or ep == epochs - 1):
            print(f'   ep {ep+1} loss {tot / (len(p) / batch):.4f} eval {evalfn(score(m, S, users, cands))}', flush=True)
    out = score(m, S, users, cands)
    del S, m
    torch.cuda.empty_cache()
    return out


def score(m, S, users, cands):
    m.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(users), 1024):
            h = m(S[torch.tensor(users[b:b + 1024], device=dev)])[:, -1]
            c = torch.tensor(cands[b:b + 1024] + 1, device=dev)
            out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    return np.vstack(out).astype(np.float32)


if __name__ == '__main__':
    z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz')
    nU, nI = int(z['nU']), int(z['nI'])
    k = 0
    users, qc = z[f'users{k}'], z[f'qc{k}']
    ev = lambda S: ' '.join(f'{np.mean(S[:, 0] > S[:, j]):.4f}' for j in (1, 2, 3))
    cfgs = {
        'd128_K4096_e30': dict(),
        'd64_K4096_e30': dict(d=64),
        'd256_K4096_e30': dict(d=256, drop=0.3),
    }
    for name in sys.argv[1:]:
        t0 = time.time()
        print(name, flush=True)
        S = train_score(z[f'hu{k}'], z[f'hi{k}'], nU, nI, users, qc, seed=0, verbose=True, evalfn=ev, **cfgs[name])
        print(name, 'standalone acc', ev(S), f'{time.time()-t0:.0f}s', flush=True)
