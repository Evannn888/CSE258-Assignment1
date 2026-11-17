"""BERT4Rec-style masked-item model for Task 4 (fill in the blank at time t); fold-0 single-model accuracy."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')
torch.backends.mha.set_fastpath_enabled(False)
from pathlib import Path
dev = 'cuda'; torch.manual_seed(0)
d, L, NB, DROP, EPOCHS, LR, NNEG, PMASK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8])
task = 4
data = V.Data(Path('/data/UCSD/CSE258/A1/assignment1_2026'), V.FILES[task], ['task_3_test.csv', 'task_4_test.csv'])
rng = np.random.default_rng(V.SEED + task); perm = rng.permutation(len(data.u)); held = perm[:30000]
qc = np.column_stack([data.i[held], V.sample_negatives(data.i, data.i[held], 2, rng)])
qu, qt = data.u[held], data.t[held]
keep = np.ones(len(data.u), bool); keep[held] = False
g = V.Graph(data.u[keep], data.i[keep], data.t[keep], data.r[keep], data.nU, data.nI)
nI = data.nI; PAD, MASK = 0, nI + 1
pool = torch.tensor(g.i + 1, device=dev)
# training windows: users' time-ordered items (+1), random windows of length L
starts_u = g.uptr[:-1]; lens_u = np.diff(g.uptr)
items_all = torch.tensor(g.i + 1, device=dev)
class B4R(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(nI + 2, d, padding_idx=0); self.pos = nn.Embedding(L, d); self.bias = nn.Embedding(nI + 2, 1)
        layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=4 * d, dropout=DROP, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, NB, enable_nested_tensor=False); self.ln = nn.LayerNorm(d); self.drop = nn.Dropout(DROP)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.zeros_(self.bias.weight)
    def forward(self, s):
        x = self.drop(self.emb(s) * d ** 0.5 + self.pos.weight[None])
        return self.ln(self.enc(x, src_key_padding_mask=(s == PAD)))
m = B4R().to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.98))
users = np.flatnonzero(lens_u >= 2)
def batch_windows(ub, rs):
    """Random window of up to L consecutive items per user, left-aligned with padding at the end."""
    S = np.zeros((len(ub), L), np.int64)
    for r, u in enumerate(ub):
        n = lens_u[u]; a = starts_u[u]
        o = rs.integers(0, max(n - L, 0) + 1)
        w = g.i[a + o:a + o + L] + 1; S[r, :len(w)] = w
    return S
rs = np.random.default_rng(0)
# precompute query sequences: window of L around the insertion point of qt with MASK at the gap
Q = np.zeros((len(qu), L), np.int64); qpos = np.zeros(len(qu), np.int64)
for r, (u, t) in enumerate(zip(qu, qt)):
    a, b = g.uptr[u], g.uptr[u + 1]; p = a + np.searchsorted(g.t[a:b], t)
    left = g.i[max(a, p - L // 2):p] + 1; right = g.i[p:min(b, p + L - 1 - len(left))] + 1
    seq = np.r_[left, MASK, right]; Q[r, :len(seq)] = seq; qpos[r] = len(left)
Qt, qpos_t, cand = torch.tensor(Q, device=dev), torch.tensor(qpos, device=dev), torch.tensor(qc + 1, device=dev)
def evaluate():
    m.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(Q), 1024):
            h = m(Qt[b:b + 1024]); hm = h[torch.arange(len(h), device=dev), qpos_t[b:b + 1024]]
            c = cand[b:b + 1024]; out.append(((hm[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    m.train(); s = np.vstack(out); return np.mean(s[:, 0] > s[:, 1]), s
for ep in range(EPOCHS):
    p = rs.permutation(users)
    for b in range(0, len(p), 256):
        S = torch.tensor(batch_windows(p[b:b + 256], rs), device=dev)
        msk = (torch.rand(S.shape, device=dev) < PMASK) & (S != PAD)
        inp = torch.where(msk, torch.full_like(S, MASK), S)
        h = m(inp)[msk]; tgt = S[msk]
        neg = pool[torch.randint(0, len(pool), (len(tgt), NNEG), device=dev)]
        ps = (h * m.emb(tgt)).sum(-1) + m.bias(tgt)[:, 0]
        ns = (h[:, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]
        loss = Fn.softplus(-(ps[:, None] - ns)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        a, s = evaluate(); V.log(f'BERT4Rec d={d} L={L} blocks={NB} mask={PMASK} ep {ep + 1}: fold0 acc {a:.4f}')
np.save('/data/UCSD/CSE258/A1/cache_v7/b4r_t4_fold0.npy', s)
