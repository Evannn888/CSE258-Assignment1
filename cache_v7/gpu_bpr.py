"""GPU BPR with popularity negatives (single-model fold-0 accuracy), Task 2 or 4."""
import sys, time, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from torch._native import registry as _native
_native.deregister_op_overrides(disable_dsl_names='triton')   # avoid Triton JIT (no Python headers on this machine)
from pathlib import Path
dev = 'cuda'
task, k, epochs, lr, wd, nneg, bs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
torch.manual_seed(0)
D = Path('/data/UCSD/CSE258/A1/assignment1_2026')
data = V.Data(D, V.FILES[task], [f for f in ['task_1_test.csv', 'task_2_test.csv'] if task == 2] or ['task_3_test.csv', 'task_4_test.csv'])
rng = np.random.default_rng(V.SEED + task)
perm = rng.permutation(len(data.u)); held = perm[:30000]
qc = np.column_stack([data.i[held], V.sample_negatives(data.i, data.i[held], 2, rng)])
qu = data.u[held]
keep = np.ones(len(data.u), bool); keep[held] = False
fu = torch.tensor(data.u[keep], device=dev); fi = torch.tensor(data.i[keep], device=dev)
nU, nI = data.nU, data.nI
P = torch.nn.Parameter(torch.randn(nU, k, device=dev) * 0.01); Q = torch.nn.Parameter(torch.randn(nI, k, device=dev) * 0.01); B = torch.nn.Parameter(torch.zeros(nI, device=dev))
opt = torch.optim.Adam([P, Q, B], lr=lr)
tq, tc = torch.tensor(qu, device=dev), torch.tensor(qc, device=dev)
n = len(fu)
for ep in range(epochs):
    order = torch.randperm(n, device=dev)
    for s in range(0, n, bs):
        idx = order[s:s + bs]; u = fu[idx]; i = fi[idx]
        j = fi[torch.randint(0, n, (len(idx), nneg), device=dev)]       # popularity negatives
        pu = P[u]; si = (pu * Q[i]).sum(1) + B[i]; sj = torch.einsum('bk,bnk->bn', pu, Q[j]) + B[j]
        loss = Fn.softplus(-(si[:, None] - sj)).mean() + wd * (pu.pow(2).sum(1).mean() + Q[i].pow(2).sum(1).mean() + Q[j].pow(2).sum(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    if ep % 10 == 9 or ep == epochs - 1:
        with torch.no_grad():
            s = (P[tq][:, None, :] * Q[tc]).sum(2) + B[tc]
            s = s.cpu().numpy()
        V.log(f'BPR k={k} lr={lr} wd={wd} nneg={nneg} ep {ep + 1}: fold0 acc {np.mean(s[:, 0] > s[:, 1]):.4f}')
