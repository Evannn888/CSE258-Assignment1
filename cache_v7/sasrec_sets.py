"""Train SASRec per Task 5 pseudo set (and full data) and save candidate scores."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')
torch.backends.mha.set_fastpath_enabled(False)
dev = 'cuda'
d, L, NB, DROP, EPOCHS, LR, NNEG = 64, 100, 2, 0.3, 20, 1e-3, 4
class SASRec(nn.Module):
    def __init__(self, nI):
        super().__init__()
        self.emb = nn.Embedding(nI + 1, d, padding_idx=0); self.pos = nn.Embedding(L, d); self.bias = nn.Embedding(nI + 1, 1)
        layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=2 * d, dropout=DROP, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, NB); self.ln = nn.LayerNorm(d); self.drop = nn.Dropout(DROP)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.zeros_(self.bias.weight)
    def forward(self, s):
        x = self.drop(self.emb(s) * d ** 0.5 + self.pos.weight[None])
        causal = torch.triu(torch.full((L, L), float('-inf'), device=s.device), 1)
        return self.ln(self.enc(x, mask=causal, src_key_padding_mask=(s == 0)))
def train_score(hu, hi, nU, nI, users, cands, seed):
    torch.manual_seed(seed); rs = np.random.default_rng(seed)
    o = np.lexsort((np.arange(len(hu)), hu)); hu, hi = hu[o], hi[o]          # rows are already user/time ordered
    ptr = np.searchsorted(hu, np.arange(nU + 1))
    seq = np.zeros((nU, L), np.int64)
    for u in range(nU):
        it = hi[ptr[u]:ptr[u + 1]][-L:] + 1
        if len(it): seq[u, -len(it):] = it
    S = torch.tensor(seq, device=dev); pool = torch.tensor(hi + 1, device=dev)
    m = SASRec(nI).to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.98))
    tu = np.flatnonzero((seq > 0).sum(1) >= 2)
    for ep in range(EPOCHS):
        p = rs.permutation(tu)
        for b in range(0, len(p), 256):
            s = S[torch.tensor(p[b:b + 256], device=dev)]
            inp = torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], 1)
            h = m(inp); valid = (s > 0) & (inp > 0)
            neg = pool[torch.randint(0, len(pool), (*s.shape, NNEG), device=dev)]
            ps = (h * m.emb(s)).sum(-1) + m.bias(s)[..., 0]
            ns = (h[:, :, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]
            loss = (Fn.softplus(-(ps[..., None] - ns)) * valid[..., None]).sum() / (valid.sum() * NNEG)
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
    m.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(users), 1024):
            h = m(S[torch.tensor(users[b:b + 1024], device=dev)])[:, -1]
            c = torch.tensor(cands[b:b + 1024] + 1, device=dev)
            out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    return np.vstack(out).astype(np.float32)
if __name__ == '__main__':
    z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz')
    nU, nI = int(z['nU']), int(z['nI']); res = {}
    for k in range(5):
        res[f'S{k}'] = train_score(z[f'hu{k}'], z[f'hi{k}'], nU, nI, z[f'users{k}'], z[f'qc{k}'], seed=k)
        V.log(f'set {k}: single acc {np.mean(res[f"S{k}"][:, 0] > res[f"S{k}"][:, 1]):.4f}')
    res['test'] = train_score(z['fu'], z['fi'], nU, nI, z['testqu'], z['testqc'], seed=9)
    np.savez('/data/UCSD/CSE258/A1/cache_v7/t5_sasrec.npz', **res)
    V.log('saved')
