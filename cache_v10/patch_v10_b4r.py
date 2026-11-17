"""v10 Task 4: time-aware BERT4Rec (gap-to-neighbour and calendar-month embeddings on every token)."""
p = '/data/UCSD/CSE258/A1/v10/assignment1.py'
s = open(p).read()
a = s.index("def bert4rec_scores(g, qu, qt, qc, seed")
b = s.index("# =============================================================================================\n# Tasks 2 and 4: which of two items is the held-out interaction")
new = '''def _b4r_time_inputs(T, n):
    """T: (B, L) clipped event times of the windows, n: number of real tokens per row.  Returns the bucketised gap to
    the previous and to the next event (0 = none; bucket = 1 + floor(log2(1 + days)), capped at 16) and the calendar
    month of every token (0 = padding)."""
    B, L = T.shape
    valid = np.arange(L)[None, :] < n[:, None]
    bucket = 1 + np.minimum(np.log2(1.0 + np.maximum(np.diff(T, axis=1), 0) / DAY).astype(np.int64), 15)
    pair = valid[:, 1:]
    gp, gn = np.zeros((B, L), np.int64), np.zeros((B, L), np.int64)
    gp[:, 1:] = np.where(pair, bucket, 0)
    gn[:, :-1] = np.where(pair, bucket, 0)
    mo = np.where(valid, np.clip((T - 1.0e9) // (30.44 * DAY), 0, 218).astype(np.int64) + 1, 0)
    return gp, gn, mo


def bert4rec_scores(g, qu, qt, qc, seed, d=128, L=100, blocks=2, drop=0.2, epochs=120, lr=1e-3, nneg=4, pmask=0.2):
    """Task 4 as fill-in-the-blank: a bidirectional transformer trained to recover masked items of users' time-ordered
    histories (popularity negatives); a [MASK] token inserted at the query time is scored against the candidates.
    New in v10: every token also carries when it happened - the gap to the previous and to the next event of the
    window and its calendar month.  Masking hides only the item, so the model knows when the missing interaction took
    place (exactly what the Task 4 query provides): alone 89.8% -> 91.5% of validation pairs."""
    torch = torch_setup()
    import torch.nn as nn
    import torch.nn.functional as Fn
    dev = 'cuda'
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    nI, PAD, MASK = g.nI, 0, g.nI + 1
    starts, lens = g.uptr[:-1], np.diff(g.uptr)
    tc = np.clip(g.t, 1.0e9, TEND + DAY)

    class B4R(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(nI + 2, d, padding_idx=0)
            self.pos = nn.Embedding(L, d)
            self.bias = nn.Embedding(nI + 2, 1)
            self.gap_prev, self.gap_next, self.month = nn.Embedding(17, d), nn.Embedding(17, d), nn.Embedding(220, d)
            layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=4 * d, dropout=drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)
            nn.init.normal_(self.emb.weight, std=0.02)
            nn.init.zeros_(self.bias.weight)
            for e in (self.gap_prev, self.gap_next, self.month):
                nn.init.normal_(e.weight, std=0.02)

        def forward(self, s, gp, gn, mo):
            x = self.emb(s) * d ** 0.5 + self.pos.weight[None] + self.gap_prev(gp) + self.gap_next(gn) + self.month(mo)
            return self.ln(self.enc(self.drop(x), src_key_padding_mask=(s == PAD)))
    m = B4R().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.98))
    gen = torch.Generator(device=dev).manual_seed(seed)
    pool = torch.tensor(g.i + 1, device=dev)
    users = np.flatnonzero(lens >= 2)
    T = lambda x: torch.tensor(x, device=dev)
    for ep in range(epochs):
        p = rs.permutation(users)
        for b in range(0, len(p), 256):
            ub = p[b:b + 256]
            S = np.zeros((len(ub), L), np.int64)
            W = np.zeros((len(ub), L))
            n = np.zeros(len(ub), np.int64)
            for r, u in enumerate(ub):                     # a random window of L consecutive items per user
                o = starts[u] + rs.integers(0, max(lens[u] - L, 0) + 1)
                w = g.i[o:o + L] + 1
                S[r, :len(w)] = w
                W[r, :len(w)] = tc[o:o + len(w)]
                n[r] = len(w)
            gp, gn, mo = _b4r_time_inputs(W, n)
            S = T(S)
            msk = (torch.rand(S.shape, device=dev, generator=gen) < pmask) & (S != PAD)
            h = m(torch.where(msk, torch.full_like(S, MASK), S), T(gp), T(gn), T(mo))[msk]
            tgt = S[msk]
            neg = pool[torch.randint(0, len(pool), (len(tgt), nneg), device=dev, generator=gen)]
            ps = (h * m.emb(tgt)).sum(-1) + m.bias(tgt)[:, 0]
            ns = (h[:, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]
            loss = Fn.softplus(-(ps[:, None] - ns)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
    Q = np.zeros((len(qu), L), np.int64)
    W = np.zeros((len(qu), L))
    n = np.zeros(len(qu), np.int64)
    qpos = np.zeros(len(qu), np.int64)
    for r, (u, t) in enumerate(zip(qu, qt)):            # window around the insertion point of the query time
        a, b = g.uptr[u], g.uptr[u + 1]
        p = a + np.searchsorted(g.t[a:b], t)
        lo = max(a, p - L // 2)
        hi = min(b, p + L - 1 - (p - lo))
        seq = np.r_[g.i[lo:p] + 1, MASK, g.i[p:hi] + 1]
        Q[r, :len(seq)] = seq
        W[r, :len(seq)] = np.maximum.accumulate(np.r_[tc[lo:p], min(max(t, 1.0e9), TEND + DAY), tc[p:hi]])
        n[r] = len(seq)
        qpos[r] = p - lo
    gp, gn, mo = _b4r_time_inputs(W, n)
    m.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(Q), 1024):
            sl = slice(b, b + 1024)
            h = m(T(Q[sl]), T(gp[sl]), T(gn[sl]), T(mo[sl]))[torch.arange(len(Q[sl]), device=dev), T(qpos[sl])]
            c = T(qc[sl] + 1)
            out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float32)


'''
s = s[:a] + new + s[b:]
open(p, 'w').write(s)
print('b4r patch applied')
