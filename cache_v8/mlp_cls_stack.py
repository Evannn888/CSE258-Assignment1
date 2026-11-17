"""Fold-0 protocol on the cached v7 bank: GBDT (cached) + regression MLP + 5-class MLP (expected rating) blends."""
import sys, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
V.torch_setup()
dev = 'cuda'
task = int(sys.argv[1])
C = '/data/UCSD/CSE258/A1/cache_v7/'
X = np.load(C + f't{task}_X.npy', mmap_mode='r'); meta = np.load(C + f't{task}_meta.npz'); y, fold = meta['y'], meta['fold']
tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)
gb = np.load(C + f'stk_t{task}_v6.npy')
mu = np.zeros(X.shape[1]); sq = np.zeros(X.shape[1])
for a in range(0, len(tr), 200000):
    Z = np.asarray(X[tr[a:a + 200000]], np.float64); mu += Z.sum(0); sq += (Z * Z).sum(0)
mu /= len(tr); sd = np.sqrt(np.maximum(sq / len(tr) - mu ** 2, 0)) + 1e-6
def to_gpu(rows):
    out = torch.empty((len(rows), X.shape[1]), dtype=torch.float32, device=dev)
    for a in range(0, len(rows), 200000):
        Z = (np.asarray(X[rows[a:a + 200000]]) - mu) / sd
        out[a:a + 200000] = torch.tensor(np.clip(np.nan_to_num(Z), -8, 8), dtype=torch.float32, device=dev)
    return out
Xtr, Xte = to_gpu(tr), to_gpu(te)
ytr = torch.tensor(y[tr], dtype=torch.float32, device=dev); yte = y[te]
def train(mode, seeds=3, epochs=12, bs=4096):
    pred = np.zeros(len(te))
    for seed in range(seeds):
        torch.manual_seed(seed)
        out_dim = 1 if mode == 'reg' else 5
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 2048), nn.GELU(), nn.Dropout(0.3), nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, out_dim)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * ((len(tr) + bs - 1) // bs))
        gen = torch.Generator(device=dev).manual_seed(seed)
        lv = torch.arange(1, 6, dtype=torch.float32, device=dev)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(tr), device=dev, generator=gen)
            for b in range(0, len(tr), bs):
                idx = perm[b:b + bs]
                o = net(Xtr[idx])
                if mode == 'reg':
                    loss = ((o[:, 0] + 4.0 - ytr[idx]) ** 2).mean()
                else:
                    loss = nn.functional.cross_entropy(o, (ytr[idx] - 1).long())
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            torch.cuda.synchronize()
        net.eval()
        with torch.no_grad():
            o = torch.cat([net(Xte[a:a + 65536]) for a in range(0, len(te), 65536)])
            p = (o[:, 0] + 4.0) if mode == 'reg' else (torch.softmax(o, 1) * lv).sum(1)
            pred += p.cpu().numpy() / seeds
    return pred
mse = lambda p: np.mean((np.clip(p, 1, 5) - yte) ** 2)
t0 = time.time()
reg = train('reg'); V.log(f'task {task} regression MLP {mse(reg):.5f} ({time.time()-t0:.0f}s)')
cls = train('cls'); V.log(f'task {task} 5-class MLP (expected rating) {mse(cls):.5f}')
V.log(f'task {task} GBDT {mse(gb):.5f} | GBDT+reg 50/50 {mse(0.5*gb+0.5*reg):.5f}')
for w in ((0.4, 0.3, 0.3), (0.34, 0.33, 0.33), (0.5, 0.25, 0.25), (0.4, 0.4, 0.2)):
    V.log(f'task {task} GBDT/reg/cls {w}: {mse(w[0]*gb + w[1]*reg + w[2]*cls):.5f}')
