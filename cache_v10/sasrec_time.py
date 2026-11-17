"""Time-aware SASRec for Task 5: every history token also carries the bucketised gap to the previous event and its
calendar month.  usage: sasrec_time.py sets(e.g. 0 or 0,1,2,3,4) [mode] [epochs] [d]"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')
torch.backends.mha.set_fastpath_enabled(False)
dev = 'cuda'
MODE = sys.argv[2] if len(sys.argv) > 2 else 'gap+month'
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
d = int(sys.argv[4]) if len(sys.argv) > 4 else 64
L, NB, DROP, LR, NNEG = int(os.environ.get('SL', 100)), 2, 0.3, 1e-3, int(os.environ.get('SNNEG', 4))
import os
SEED_OFF = int(os.environ.get('SEED_OFF', 0)); SUF = os.environ.get('SUF', '')
class SASRec(nn.Module):
    def __init__(self, nI):
        super().__init__()
        self.emb = nn.Embedding(nI + 1, d, padding_idx=0); self.pos = nn.Embedding(L, d); self.bias = nn.Embedding(nI + 1, 1)
        self.gap = nn.Embedding(17, d); self.month = nn.Embedding(220, d)
        layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=2 * d, dropout=DROP, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, NB, enable_nested_tensor=False); self.ln = nn.LayerNorm(d); self.drop = nn.Dropout(DROP)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.zeros_(self.bias.weight)
        nn.init.normal_(self.gap.weight, std=0.02); nn.init.normal_(self.month.weight, std=0.02)
    def forward(self, s, gp, mo):
        x = self.emb(s) * d ** 0.5 + self.pos.weight[None]
        if 'gap' in MODE:
            x = x + self.gap(gp)
        if 'month' in MODE:
            x = x + self.month(mo)
        causal = torch.triu(torch.full((L, L), float('-inf'), device=s.device), 1)
        return self.ln(self.enc(self.drop(x), mask=causal, src_key_padding_mask=(s == 0)))
def train_score(hu, hi, ht, nU, nI, users, cands, seed):
    torch.manual_seed(seed); rs = np.random.default_rng(seed)
    ptr = np.searchsorted(hu, np.arange(nU + 1))
    tc = np.clip(ht, 1.0e9, V.TEND + 86400.0)
    seq = np.zeros((nU, L), np.int64); gp = np.zeros((nU, L), np.int64); mo = np.zeros((nU, L), np.int64)
    for u in range(nU):
        a, b = ptr[u], ptr[u + 1]
        n = min(b - a, L)
        if n:
            seq[u, -n:] = hi[b - n:b] + 1
            tw = tc[b - n:b]
            mo[u, -n:] = np.clip((tw - 1.0e9) // (30.44 * 86400), 0, 218).astype(np.int64) + 1
            if n > 1:
                gp[u, -n + 1:] = 1 + np.minimum(np.log2(1.0 + np.maximum(np.diff(tw), 0) / 86400.0).astype(np.int64), 15)
            if b - a > n:                                      # gap to the event just before the window
                gp[u, -n] = 1 + min(int(np.log2(1.0 + max(tw[0] - tc[b - n - 1], 0) / 86400.0)), 15)
    S, GP, MO = (torch.tensor(x, device=dev) for x in (seq, gp, mo))
    pool = torch.tensor(hi + 1, device=dev)
    m = SASRec(nI).to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.98))
    tu = np.flatnonzero((seq > 0).sum(1) >= 2)
    sh = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
    for ep in range(EPOCHS):
        p = rs.permutation(tu)
        for b in range(0, len(p), 256):
            ix = torch.tensor(p[b:b + 256], device=dev)
            s = S[ix]
            inp = sh(s)
            h = m(inp, sh(GP[ix]), sh(MO[ix])); valid = (s > 0) & (inp > 0)
            neg = pool[torch.randint(0, len(pool), (*s.shape, NNEG), device=dev)]
            ps = (h * m.emb(s)).sum(-1) + m.bias(s)[..., 0]
            ns = (h[:, :, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]
            loss = (Fn.softplus(-(ps[..., None] - ns)) * valid[..., None]).sum() / (valid.sum() * NNEG)
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
    m.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(users), 1024):
            ix = torch.tensor(users[b:b + 1024], device=dev)
            h = m(S[ix], GP[ix], MO[ix])[:, -1]
            c = torch.tensor(cands[b:b + 1024] + 1, device=dev)
            out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    return np.vstack(out).astype(np.float32)
if __name__ == '__main__':
    z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz')
    nU, nI = int(z['nU']), int(z['nI'])
    for k in [int(x) for x in sys.argv[1].split(',')]:
        S = train_score(z[f'hu{k}'], z[f'hi{k}'], z[f'ht{k}'], nU, nI, z[f'users{k}'], z[f'qc{k}'], seed=k + SEED_OFF)
        V.log(f'set {k} mode {MODE} epochs {EPOCHS} d{d}: single acc {np.mean(S[:, 0] > S[:, 1]):.4f}')
        np.save(f'/data/UCSD/CSE258/A1/cache_v10/sast_{MODE}_e{EPOCHS}_d{d}{SUF}_s{k}.npy', S)
