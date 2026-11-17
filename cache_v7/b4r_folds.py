"""Train BERT4Rec per Task 4 fold (and full data) and save candidate scores at the masked query position."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')
torch.backends.mha.set_fastpath_enabled(False)
dev = 'cuda'; d, L, NB, DROP, EPOCHS, LR, NNEG, PMASK = 128, 100, 2, 0.2, 120, 1e-3, 4, 0.2
def b4r_scores(g, qu, qt, qc, seed):
    torch.manual_seed(seed); rs = np.random.default_rng(seed)
    nI = g.nI; PAD, MASK = 0, nI + 1
    pool = torch.tensor(g.i + 1, device=dev); starts, lens = g.uptr[:-1], np.diff(g.uptr)
    class B4R(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(nI + 2, d, padding_idx=0); self.pos = nn.Embedding(L, d); self.bias = nn.Embedding(nI + 2, 1)
            layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=4 * d, dropout=DROP, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, NB, enable_nested_tensor=False); self.ln = nn.LayerNorm(d); self.drop = nn.Dropout(DROP)
            nn.init.normal_(self.emb.weight, std=0.02); nn.init.zeros_(self.bias.weight)
        def forward(self, s):
            return self.ln(self.enc(self.drop(self.emb(s) * d ** 0.5 + self.pos.weight[None]), src_key_padding_mask=(s == PAD)))
    m = B4R().to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.98))
    users = np.flatnonzero(lens >= 2)
    for ep in range(EPOCHS):
        p = rs.permutation(users)
        for b in range(0, len(p), 256):
            ub = p[b:b + 256]; S = np.zeros((len(ub), L), np.int64)
            for r, u in enumerate(ub):
                o = rs.integers(0, max(lens[u] - L, 0) + 1); w = g.i[starts[u] + o:starts[u] + o + L] + 1; S[r, :len(w)] = w
            S = torch.tensor(S, device=dev)
            msk = (torch.rand(S.shape, device=dev) < PMASK) & (S != PAD)
            h = m(torch.where(msk, torch.full_like(S, MASK), S))[msk]; tgt = S[msk]
            neg = pool[torch.randint(0, len(pool), (len(tgt), NNEG), device=dev)]
            loss = Fn.softplus(-(((h * m.emb(tgt)).sum(-1) + m.bias(tgt)[:, 0])[:, None] - ((h[:, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
    Q = np.zeros((len(qu), L), np.int64); qpos = np.zeros(len(qu), np.int64)
    for r, (u, t) in enumerate(zip(qu, qt)):
        a, b = g.uptr[u], g.uptr[u + 1]; p = a + np.searchsorted(g.t[a:b], t)
        left = g.i[max(a, p - L // 2):p] + 1; right = g.i[p:min(b, p + L - 1 - len(left))] + 1
        seq = np.r_[left, MASK, right]; Q[r, :len(seq)] = seq; qpos[r] = len(left)
    m.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(Q), 1024):
            Qt = torch.tensor(Q[b:b + 1024], device=dev); h = m(Qt)[torch.arange(len(Qt), device=dev), torch.tensor(qpos[b:b + 1024], device=dev)]
            c = torch.tensor(qc[b:b + 1024] + 1, device=dev); out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    return np.vstack(out).astype(np.float32)
if __name__ == '__main__':
    z = np.load('/data/UCSD/CSE258/A1/cache_v7/pairs_t4.npz')
    u, i, t, r, nU, nI = z['u'], z['i'], z['t'], z['r'], int(z['nU']), int(z['nI'])
    res = {}
    for f in range(4):
        keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
        g = V.Graph(u[keep], i[keep], t[keep], r[keep], nU, nI)
        res[f'S{f}'] = b4r_scores(g, u[z[f'held{f}']], z[f'qt{f}'], z[f'qc{f}'], seed=f)
        V.log(f'fold {f}: single acc {np.mean(res[f"S{f}"][:, 0] > res[f"S{f}"][:, 1]):.4f}')
    g = V.Graph(u, i, t, r, nU, nI)
    res['test'] = b4r_scores(g, z['testqu'], z['testqt'], z['testqc'], seed=9)
    np.savez('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz', **res); V.log('saved')
