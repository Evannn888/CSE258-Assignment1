"""GPU MLP stacker on cached OOF rating features: train on folds != 0, evaluate fold 0; blend with GBDT."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')
task = int(sys.argv[1]); gbdt_name = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
H1, H2, DROP, EPOCHS, LR, TAG = (int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]), sys.argv[8]) if len(sys.argv) > 8 else (512, 256, 0.2, 8, 2e-3, 'mlp')
torch.manual_seed(0); dev = 'cuda'
C = '/data/UCSD/CSE258/A1/cache_v7/'
X = np.load(C + f't{task}_X.npy', mmap_mode='r'); meta = np.load(C + f't{task}_meta.npz'); y, fold = meta['y'], meta['fold']
tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)
Xtr = np.asarray(X[tr]); Xte = np.asarray(X[te])
mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
def prep(Z):
    Z = (Z - mu) / sd
    return torch.tensor(np.clip(np.nan_to_num(Z), -8, 8), dtype=torch.float32, device=dev)
Ttr, Tte = prep(Xtr), prep(Xte); del Xtr, Xte
ytr = torch.tensor(y[tr], dtype=torch.float32, device=dev); yte = y[te]
preds = []
for seed in range(3):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Ttr.shape[1], H1), nn.GELU(), nn.Dropout(DROP), nn.Linear(H1, H2), nn.GELU(), nn.Dropout(DROP), nn.Linear(H2, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(ytr); epochs = EPOCHS; steps = epochs * (n // 4096 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for b in range(0, n, 4096):
            idx = perm[b:b + 4096]
            loss = ((net(Ttr[idx])[:, 0] + 4.0 - ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        torch.cuda.synchronize()
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(Tte[b:b + 65536])[:, 0] + 4.0 for b in range(0, len(Tte), 65536)]).cpu().numpy()
    preds.append(p)
    if False: V.log(f'task {task} MLP seed {seed}: MSE {np.mean((np.clip(p, 1, 5) - yte) ** 2):.5f}')
p_mlp = np.mean(preds, 0)
V.log(f'task {task} {TAG} H={H1},{H2} drop={DROP} ep={EPOCHS} lr={LR} 3-seed avg: MSE {np.mean((np.clip(p_mlp, 1, 5) - yte) ** 2):.5f}')
np.save(C + f'{TAG}_t{task}.npy', p_mlp)
if gbdt_name:
    p_g = np.load(C + f'stk_t{task}_{gbdt_name}.npy')
    for w in [0.0, 0.2, 0.3, 0.4, 0.5]:
        V.log(f'task {task} blend GBDT({gbdt_name})*{1 - w:.1f} + MLP*{w:.1f}: MSE {np.mean((np.clip((1 - w) * p_g + w * p_mlp, 1, 5) - yte) ** 2):.5f}')
