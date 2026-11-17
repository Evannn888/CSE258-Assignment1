"""MLP stacker + user/item embeddings (GPU), fold-0 evaluation."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')
task, EMB, WD_EMB = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
dev = 'cuda'; C = '/data/UCSD/CSE258/A1/cache_v7/'
X = np.load(C + f't{task}_X.npy', mmap_mode='r'); meta = np.load(C + f't{task}_meta.npz')
y, fold, us, it = meta['y'], meta['fold'], meta['users'], meta['items']
nU, nI = int(us.max()) + 1, int(it.max()) + 1
tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)
Xtr = np.asarray(X[tr]); Xte = np.asarray(X[te]); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
prep = lambda Z: torch.tensor(np.clip(np.nan_to_num((Z - mu) / sd), -8, 8), dtype=torch.float32, device=dev)
Ttr, Tte = prep(Xtr), prep(Xte); del Xtr, Xte
Utr, Itr = torch.tensor(us[tr], device=dev), torch.tensor(it[tr], device=dev); Ute, Ite = torch.tensor(us[te], device=dev), torch.tensor(it[te], device=dev)
ytr = torch.tensor(y[tr], dtype=torch.float32, device=dev); yte = y[te]
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ue = nn.Embedding(nU, EMB); self.ie = nn.Embedding(nI, EMB)
        nn.init.normal_(self.ue.weight, std=0.01); nn.init.normal_(self.ie.weight, std=0.01)
        self.mlp = nn.Sequential(nn.Linear(d + 2 * EMB, 2048), nn.GELU(), nn.Dropout(0.3), nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 1))
    def forward(self, x, u, i):
        return self.mlp(torch.cat([x, self.ue(u), self.ie(i)], 1))[:, 0] + 4.0
preds = []
for seed in range(3):
    torch.manual_seed(seed); net = Net(Ttr.shape[1]).to(dev)
    params = [{'params': net.mlp.parameters(), 'weight_decay': 1e-4}, {'params': list(net.ue.parameters()) + list(net.ie.parameters()), 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(params, lr=1e-3); n = len(ytr); E = 12
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=E * (n // 4096 + 1))
    for ep in range(E):
        net.train(); perm = torch.randperm(n, device=dev)
        for b in range(0, n, 4096):
            idx = perm[b:b + 4096]
            ue, ie = net.ue(Utr[idx]), net.ie(Itr[idx])
            loss = ((net(Ttr[idx], Utr[idx], Itr[idx]) - ytr[idx]) ** 2).mean() + WD_EMB * (ue.pow(2).sum(1) + ie.pow(2).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        torch.cuda.synchronize()
    net.eval()
    with torch.no_grad():
        preds.append(torch.cat([net(Tte[b:b + 65536], Ute[b:b + 65536], Ite[b:b + 65536]) for b in range(0, len(Tte), 65536)]).cpu().numpy())
p = np.mean(preds, 0)
V.log(f'task {task} MLP+emb(d={EMB}, l2={WD_EMB}) 3-seed: MSE {np.mean((np.clip(p, 1, 5) - yte) ** 2):.5f}')
np.save(C + f'mlpemb{EMB}_t{task}.npy', p)
