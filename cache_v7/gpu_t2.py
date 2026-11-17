"""GPU latent models for Task 2 (fold 0 single-model accuracy): Mult-VAE and big BPR."""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from pathlib import Path
torch.manual_seed(0)
dev = 'cuda'
task = 2
D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
data = V.Data(D, V.FILES[task], ['task_1_test.csv', 'task_2_test.csv'])
rng = np.random.default_rng(V.SEED + task)
perm = rng.permutation(len(data.u))
held = perm[:30000]
qc = np.column_stack([data.i[held], V.sample_negatives(data.i, data.i[held], 2, rng)])
qu = data.u[held]
keep = np.ones(len(data.u), bool); keep[held] = False
fu, fi = data.u[keep], data.i[keep]
nU, nI = data.nU, data.nI
cnt = np.bincount(fi, minlength=nI).astype(np.float64)
logq = torch.tensor(np.log((cnt + 1) / cnt.sum()), dtype=torch.float32, device=dev)
# sparse user-item matrix on GPU (rows = users)
o = np.lexsort((fi, fu))
indptr = torch.tensor(np.searchsorted(fu[o], np.arange(nU + 1)), dtype=torch.long, device=dev)
indices = torch.tensor(fi[o], dtype=torch.long, device=dev)

def acc(score_fn):
    s = score_fn(torch.tensor(qu, device=dev), torch.tensor(qc, device=dev))   # (n, 3)
    s = s.cpu().numpy()
    return np.mean(s[:, 0] > s[:, 1]) + 0.5 * np.mean(s[:, 0] == s[:, 1])

class MultVAE(nn.Module):
    def __init__(self, nI, h=600, z=200, drop=0.5):
        super().__init__()
        self.e1 = nn.Linear(nI, h); self.e2 = nn.Linear(h, 2 * z); self.d1 = nn.Linear(z, h); self.d2 = nn.Linear(h, nI); self.drop = drop; self.z = z
    def forward(self, x, train=True):
        x = Fn.normalize(x, dim=1)
        if train: x = Fn.dropout(x, self.drop)
        h = torch.tanh(self.e1(x)); mu, logvar = self.e2(h).chunk(2, dim=1)
        zz = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if train else mu
        return self.d2(torch.tanh(self.d1(zz))), mu, logvar

def dense_rows(users):
    a, b = indptr[users], indptr[users + 1]
    lens = b - a
    rows = torch.repeat_interleave(torch.arange(len(users), device=dev), lens)
    start = torch.repeat_interleave(a, lens)
    off = torch.arange(int(lens.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
    x = torch.zeros(len(users), nI, device=dev)
    x[rows, indices[start + off]] = 1.0
    return x

def train_vae(epochs=150, bs=512, h=600, z=200, beta_max=0.2, lr=1e-3, wd=0.0):
    m = MultVAE(nI, h, z).to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    users = np.unique(fu); step = 0; total = epochs * int(np.ceil(len(users) / bs))
    for ep in range(epochs):
        m.train(); p = np.random.permutation(users)
        for b in range(0, len(p), bs):
            ub = torch.tensor(p[b:b + bs], device=dev); x = dense_rows(ub)
            logits, mu, logvar = m(x)
            nll = -(Fn.log_softmax(logits, 1) * x).sum(1).mean()
            kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
            beta = min(beta_max, beta_max * step / (0.5 * total)); step += 1
            loss = nll + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
        if ep % 25 == 24 or ep == epochs - 1:
            m.eval()
            def score(u, c):
                out = []
                with torch.no_grad():
                    for b in range(0, len(u), 512):
                        logits, _, _ = m(dense_rows(u[b:b + 512]), train=False)
                        ls = Fn.log_softmax(logits, 1)
                        out.append(torch.gather(ls, 1, c[b:b + 512]) - torch.gather(logq.expand(len(ls), -1), 1, c[b:b + 512]))
                return torch.cat(out)
            V.log(f'VAE h={h} z={z} beta={beta_max} ep {ep + 1}: fold0 acc {acc(score):.4f}')
    return m

if __name__ == '__main__':
    t0 = time.time()
    train_vae()
    V.log(f'time {time.time() - t0:.0f}s')
