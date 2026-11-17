"""Time-aware BERT4Rec for Task 4: every token also carries (a) the bucketised gap to the previous and to the next
event of the window and (b) its calendar month.  The mask replaces only the item id, so the model knows WHEN the
missing interaction happened (the Task 4 query gives exactly that)."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')
torch.backends.mha.set_fastpath_enabled(False)
dev = 'cuda'; d, L, NB, DROP, LR, NNEG, PMASK = 128, 100, 2, 0.2, 1e-3, 4, 0.2
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
MODE = sys.argv[3] if len(sys.argv) > 3 else 'gap+month'
NBUCK, NMON, NFREQ = 17, 220, 16
FREQ = None

def gap_bucket(dt):
    return (1 + np.minimum(np.log2(1.0 + np.maximum(dt, 0) / 86400.0).astype(np.int64), NBUCK - 2))

def time_feats(tw, n):
    """tw: times of the n real tokens of a window (ascending).  Returns previous-gap, next-gap buckets and month index."""
    gp, gn, mo = np.zeros(L, np.int64), np.zeros(L, np.int64), np.zeros(L, np.int64)
    if n > 1:
        b = gap_bucket(np.diff(tw))
        gp[1:n] = b; gn[:n - 1] = b
    mo[:n] = np.clip((tw - 1.0e9) // (30.44 * 86400), 0, NMON - 2).astype(np.int64) + 1
    return gp, gn, mo

def b4r_scores(g, qu, qt, qc, seed):
    torch.manual_seed(seed); rs = np.random.default_rng(seed)
    nI = g.nI; PAD, MASK = 0, nI + 1
    tc = np.clip(g.t, 1.0e9, V.TEND + 86400.0)
    pool = torch.tensor(g.i + 1, device=dev); starts, lens = g.uptr[:-1], np.diff(g.uptr)
    class B4R(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(nI + 2, d, padding_idx=0); self.pos = nn.Embedding(L, d); self.bias = nn.Embedding(nI + 2, 1)
            self.gp = nn.Embedding(NBUCK, d); self.gn = nn.Embedding(NBUCK, d); self.mo = nn.Embedding(NMON, d)
            self.sin = nn.Linear(2 * NFREQ, d)
            layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=4 * d, dropout=DROP, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, NB, enable_nested_tensor=False); self.ln = nn.LayerNorm(d); self.drop = nn.Dropout(DROP)
            nn.init.normal_(self.emb.weight, std=0.02); nn.init.zeros_(self.bias.weight)
            for e in (self.gp, self.gn, self.mo):
                nn.init.normal_(e.weight, std=0.02)
        def forward(self, s, gp, gn, mo, tt=None):
            x = self.emb(s) * d ** 0.5 + self.pos.weight[None]
            if 'sin' in MODE:                     # absolute time at many scales -> attention can read time differences
                ang = tt[..., None] * FREQ
                x = x + self.sin(torch.cat([torch.sin(ang), torch.cos(ang)], -1))
            if 'gap' in MODE:
                x = x + self.gp(gp) + self.gn(gn)
            if 'month' in MODE:
                x = x + self.mo(mo)
            return self.ln(self.enc(self.drop(x), src_key_padding_mask=(s == PAD)))
    global FREQ
    FREQ = (2 * np.pi / torch.tensor(np.geomspace(2.0, 7300.0, NFREQ), dtype=torch.float32, device=dev))      # periods 2 days .. 20 years
    m = B4R().to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.98))
    users = np.flatnonzero(lens >= 2)
    T = lambda a: torch.tensor(a, device=dev)
    days = lambda a: torch.tensor((a - 1.0e9) / 86400.0, dtype=torch.float32, device=dev)
    for ep in range(EPOCHS):
        p = rs.permutation(users)
        for b in range(0, len(p), 256):
            ub = p[b:b + 256]; S = np.zeros((len(ub), L), np.int64); GP = np.zeros_like(S); GN = np.zeros_like(S); MO = np.zeros_like(S)
            TT = np.full((len(ub), L), 1.0e9)
            for r, u in enumerate(ub):
                o = rs.integers(0, max(lens[u] - L, 0) + 1); a_ = starts[u] + o
                w = g.i[a_:a_ + L] + 1; S[r, :len(w)] = w
                GP[r], GN[r], MO[r] = time_feats(tc[a_:a_ + len(w)], len(w)); TT[r, :len(w)] = tc[a_:a_ + len(w)]
            S = T(S)
            msk = (torch.rand(S.shape, device=dev) < PMASK) & (S != PAD)
            h = m(torch.where(msk, torch.full_like(S, MASK), S), T(GP), T(GN), T(MO), days(TT))[msk]; tgt = S[msk]
            neg = pool[torch.randint(0, len(pool), (len(tgt), NNEG), device=dev)]
            loss = Fn.softplus(-(((h * m.emb(tgt)).sum(-1) + m.bias(tgt)[:, 0])[:, None] - ((h[:, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
    n = len(qu)
    Q = np.zeros((n, L), np.int64); GP = np.zeros_like(Q); GN = np.zeros_like(Q); MO = np.zeros_like(Q); qpos = np.zeros(n, np.int64)
    TT = np.full((n, L), 1.0e9)
    for r, (u, t) in enumerate(zip(qu, qt)):
        a, b = g.uptr[u], g.uptr[u + 1]; p = a + np.searchsorted(g.t[a:b], t)
        lo, hi = max(a, p - L // 2), None
        left = g.i[lo:p] + 1; hi = min(b, p + L - 1 - len(left)); right = g.i[p:hi] + 1
        seq = np.r_[left, MASK, right]; Q[r, :len(seq)] = seq; qpos[r] = len(left)
        tw = np.r_[tc[lo:p], min(max(t, 1.0e9), V.TEND + 86400.0), tc[p:hi]]
        tw = np.maximum.accumulate(tw)
        GP[r], GN[r], MO[r] = time_feats(tw, len(seq)); TT[r, :len(seq)] = tw
    m.eval(); out = []
    with torch.no_grad():
        for b in range(0, n, 1024):
            sl = slice(b, b + 1024)
            h = m(T(Q[sl]), T(GP[sl]), T(GN[sl]), T(MO[sl]), days(TT[sl]))[torch.arange(len(Q[sl]), device=dev), T(qpos[sl])]
            c = T(qc[sl] + 1); out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    return np.vstack(out).astype(np.float32)

if __name__ == '__main__':
    folds = [int(x) for x in sys.argv[1].split(',')]
    z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz')
    u, i, t, r, nU, nI = z['u'], z['i'], z['t'], z['r'], int(z['nU']), int(z['nI'])
    for f in folds:
        keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
        g = V.Graph(u[keep], i[keep], t[keep], r[keep], nU, nI)
        S = b4r_scores(g, u[z[f'held{f}']], z[f'qt{f}'], z[f'qc{f}'], seed=f)
        V.log(f'fold {f} mode {MODE} epochs {EPOCHS}: single acc {np.mean(S[:, 0] > S[:, 1]):.4f}')
        np.save(f'/data/UCSD/CSE258/A1/cache_v10/b4rt_{MODE}_e{EPOCHS}_f{f}.npy', S)
