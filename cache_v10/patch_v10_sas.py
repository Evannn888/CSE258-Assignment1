"""v10 Task 5: time-aware SASRec (gap to the previous event and calendar month of every history token)."""
p = '/data/UCSD/CSE258/A1/v10/assignment1.py'
s = open(p).read()
a = s.index("def sasrec_scores(g, users, cands, seed")
b = s.index("def _b4r_time_inputs(T, n):")
new = '''def sasrec_scores(g, users, cands, seed, d=64, L=100, blocks=2, drop=0.3, epochs=20, lr=1e-3, nneg=4, use_time=True):
    """SASRec (causal self-attention over the last L items) trained with popularity negatives; scores candidates
    for the item following each user's history in graph g.  New in v10 (use_time): every history token also carries
    the bucketised gap to the previous event and its calendar month (the training-file timestamps; Task 5 may use
    them).  Alone 87.2% -> 88.0% of validation pairs."""
    torch = torch_setup()
    import torch.nn as nn
    import torch.nn.functional as Fn
    dev = 'cuda'
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    nU, nI = g.nU, g.nI
    tc = np.clip(g.t, 1.0e9, TEND + DAY)
    seq = np.zeros((nU, L), np.int64)
    gap = np.zeros((nU, L), np.int64)
    mon = np.zeros((nU, L), np.int64)
    for u in range(nU):
        a_, b_ = g.uptr[u], g.uptr[u + 1]
        n = min(b_ - a_, L)
        if n == 0:
            continue
        seq[u, -n:] = g.i[b_ - n:b_] + 1                     # 0 = padding
        if use_time:
            tw = tc[b_ - n:b_]
            mon[u, -n:] = np.clip((tw - 1.0e9) // (30.44 * DAY), 0, 218).astype(np.int64) + 1
            prev = np.r_[tc[b_ - n - 1] if b_ - a_ > n else np.nan, tw[:-1]]
            gb = 1 + np.minimum(np.log2(1.0 + np.maximum(tw - prev, 0) / DAY), 15)
            gap[u, -n:] = np.where(np.isnan(prev), 0, gb).astype(np.int64)

    class SASRec(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(nI + 1, d, padding_idx=0)
            self.pos = nn.Embedding(L, d)
            self.bias = nn.Embedding(nI + 1, 1)
            self.gap, self.month = nn.Embedding(17, d), nn.Embedding(220, d)
            layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=2 * d, dropout=drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)
            nn.init.normal_(self.emb.weight, std=0.02)
            nn.init.zeros_(self.bias.weight)
            nn.init.normal_(self.gap.weight, std=0.02)
            nn.init.normal_(self.month.weight, std=0.02)

        def forward(self, s, gp, mo):
            x = self.emb(s) * d ** 0.5 + self.pos.weight[None]
            if use_time:
                x = x + self.gap(gp) + self.month(mo)
            causal = torch.triu(torch.full((L, L), float('-inf'), device=s.device), 1)
            return self.ln(self.enc(self.drop(x), mask=causal, src_key_padding_mask=(s == 0)))
    S, GP, MO = (torch.tensor(x, device=dev) for x in (seq, gap, mon))
    shift = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
    pool = torch.tensor(g.i + 1, device=dev)
    m = SASRec().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.98))
    gen = torch.Generator(device=dev).manual_seed(seed)
    train_users = np.flatnonzero((seq > 0).sum(1) >= 2)
    for ep in range(epochs):
        p = rs.permutation(train_users)
        for b in range(0, len(p), 256):
            ix = torch.tensor(p[b:b + 256], device=dev)
            s = S[ix]
            inp = shift(s)
            h = m(inp, shift(GP[ix]), shift(MO[ix]))
            valid = (s > 0) & (inp > 0)
            neg = pool[torch.randint(0, len(pool), (*s.shape, nneg), device=dev, generator=gen)]
            ps = (h * m.emb(s)).sum(-1) + m.bias(s)[..., 0]
            ns = (h[:, :, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]
            loss = (Fn.softplus(-(ps[..., None] - ns)) * valid[..., None]).sum() / (valid.sum() * nneg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
    m.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(users), 1024):
            ix = torch.tensor(users[b:b + 1024], device=dev)
            h = m(S[ix], GP[ix], MO[ix])[:, -1]
            c = torch.tensor(cands[b:b + 1024] + 1, device=dev)
            out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    del S, GP, MO, m
    torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float32)


'''
s = s[:a] + new + s[b:]
old = """    seqs = [sasrec_scores(gk, users, qck, seed=k) for k, (_, _, _, (gk, users, qck, _)) in enumerate(sets)]
    seqs.append(sasrec_scores(g, qu, qc, seed=len(sets)))
"""
new2 = """    seqs = [sasrec_scores(gk, users, qck, seed=k, use_time=not order_only) for k, (_, _, _, (gk, users, qck, _)) in enumerate(sets)]
    seqs.append(sasrec_scores(g, qu, qc, seed=len(sets), use_time=not order_only))
"""
assert old in s; s = s.replace(old, new2)
open(p, 'w').write(s)
print('sasrec patch applied')
