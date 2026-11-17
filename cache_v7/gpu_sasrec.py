"""SASRec on GPU for Task 5: single-model accuracy on the last-event pseudo test (same construction as v6)."""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')
from pathlib import Path
dev = 'cuda'; torch.manual_seed(0); np.random.seed(0)
torch.backends.mha.set_fastpath_enabled(False)
d, L, nblocks, drop, epochs, lr, nneg = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
data = V.Data(Path('/data/UCSD/CSE258/A1/assignment1_2026'), 'task_5_train.csv', ['task_5_test.csv'])
rng = np.random.default_rng(V.SEED + 5)
g_all = V.Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
last = g_all.uptr[1:] - 1; users = np.flatnonzero(g_all.ucnt >= 4); held = last[users]
mask = np.ones(len(g_all.u), bool); mask[held] = False
g = V.Graph(g_all.u[mask], g_all.i[mask], g_all.t[mask], g_all.r[mask], data.nU, data.nI)
qc = np.column_stack([g_all.i[held], V.sample_negatives(g.i, g_all.i[held], 3, rng)])
nI = data.nI
# padded sequences (item ids shifted by 1; 0 = padding), last L items
seq = np.zeros((data.nU, L), np.int64)
for u in range(data.nU):
    items = g.i[g.uptr[u]:g.uptr[u + 1]][-L:] + 1
    if len(items): seq[u, -len(items):] = items
pop = np.bincount(g.i, minlength=nI).astype(np.float64)
pool = torch.tensor(g.i + 1, device=dev)                     # popularity negatives (shifted ids)
class SASRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(nI + 1, d, padding_idx=0); self.pos = nn.Embedding(L, d); self.bias = nn.Embedding(nI + 1, 1)
        layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=d * 2, dropout=drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nblocks); self.ln = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.zeros_(self.bias.weight)
    def forward(self, s):
        x = self.emb(s) * (d ** 0.5) + self.pos.weight[None]
        x = self.drop(x)
        causal = torch.triu(torch.full((L, L), float('-inf'), device=s.device), 1)
        h = self.enc(x, mask=causal, src_key_padding_mask=(s == 0))
        return self.ln(h)
m = SASRec().to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.98))
S = torch.tensor(seq, device=dev)
train_users = np.flatnonzero((seq > 0).sum(1) >= 2)
def score(users_, cands):
    out = []
    m.eval()
    with torch.no_grad():
        for b in range(0, len(users_), 1024):
            h = m(S[users_[b:b + 1024]])[:, -1]                               # state after the last item
            c = cands[b:b + 1024] + 1
            out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    m.train()
    return np.vstack(out)
tu, tcand = torch.tensor(users, device=dev), torch.tensor(qc, device=dev)
for ep in range(epochs):
    p = np.random.permutation(train_users)
    for b in range(0, len(p), 256):
        s = S[torch.tensor(p[b:b + 256], device=dev)]
        inp, tgt = s[:, :-1], s[:, 1:]
        inp = torch.cat([torch.zeros_like(inp[:, :1]), inp], 1)[:, :L]       # keep length L: shift right
        tgt = s
        h = m(torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], 1))          # predict item at t from items < t
        valid = (tgt > 0) & (torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], 1) > 0)
        neg = pool[torch.randint(0, len(pool), (*tgt.shape, nneg), device=dev)]
        ps = (h * m.emb(tgt)).sum(-1) + m.bias(tgt)[..., 0]
        ns = (h[:, :, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]
        loss = (Fn.softplus(-(ps[..., None] - ns)) * valid[..., None]).sum() / (valid.sum() * nneg)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    if ep % 5 == 4 or ep == epochs - 1:
        sc = score(tu, tcand)
        V.log(f'nan frac {np.isnan(sc).mean():.4f}'); V.log(f'SASRec d={d} L={L} blocks={nblocks} drop={drop} ep {ep + 1}: last-event acc (pos vs neg1) {np.mean(sc[:, 0] > sc[:, 1]):.4f}')
np.save(f'/data/UCSD/CSE258/A1/cache_v7/sasrec_L_d{d}.npy', score(tu, tcand))
