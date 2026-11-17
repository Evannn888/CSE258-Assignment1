"""MLP pair classifier on the same pair rows as the GBDT; antisymmetrised; blended with the GBDT margin."""
import sys, numpy as np, time, torch, torch.nn as nn
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
from sklearn.ensemble import HistGradientBoostingClassifier
V.torch_setup()
dev = 'cuda'
task = int(sys.argv[1])
z = np.load(f'/data/UCSD/CSE258/A1/cache_v7/pairs_t{task}.npz')
u, i = z['u'], z['i']; nI = int(z['nI'])
if task == 2:
    e = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t2_folds_300.npz'); lams = [300]
else:
    e = np.load('/data/UCSD/CSE258/A1/cache_v8/ease_t4.npz'); lams = [500, 2000]; b4 = np.load('/data/UCSD/CSE258/A1/cache_v7/t4_b4r.npz')
S = []
for f in range(4):
    keep = np.ones(len(u), bool); keep[z[f'held{f}']] = False
    pop = np.bincount(i[keep], minlength=nI).astype(float)
    parts = [z[f'F{f}'], z[f'L{f}']] + ([b4[f'S{f}'][..., None]] if task == 4 else [])
    for lam in lams:
        X = e[f'E{f}_{lam}']
        parts.append(np.concatenate([X, X / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1).astype(np.float32))
    S.append((np.concatenate(parts, -1), z[f'ctx{f}'], z[f'seen{f}']))
small = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0)


def rows(sets):
    X, Y = [], []
    for F, ctx, _ in sets:
        for c in range(1, F.shape[1]):
            d, s = V.pair_rows(F[:, 0], F[:, c], ctx)
            X += [np.hstack([d, s]), np.hstack([-d, s])]; Y += [np.ones(len(d)), np.zeros(len(d))]
    return np.vstack(X).astype(np.float32), np.concatenate(Y).astype(np.float32)


def mlp_fit_predict(X, Y, Xa, Xb, seeds=3, epochs=20, h1=512, h2=256, drop=0.2, lr=2e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    prep = lambda Z: torch.tensor(np.clip((Z - mu) / sd, -8, 8), dtype=torch.float32, device=dev)
    Xt, Yt, A, B = prep(X), torch.tensor(Y, device=dev), prep(Xa), prep(Xb)
    out = np.zeros(len(Xa))
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], h1), nn.GELU(), nn.Dropout(drop), nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop), nn.Linear(h2, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        bs = 2048; steps = epochs * ((len(Xt) + bs - 1) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        gen = torch.Generator(device=dev).manual_seed(seed)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xt), device=dev, generator=gen)
            for b in range(0, len(Xt), bs):
                idx = perm[b:b + bs]
                loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[idx])[:, 0], Yt[idx])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            out += 0.5 * (net(A)[:, 0] - net(B)[:, 0]).cpu().numpy() / seeds
    return out


for tf in (0, 1):
    X, Y = rows([S[k] for k in range(4) if k != tf])
    F, c, s = S[tf]
    d, sy = V.pair_rows(F[:, 0], F[:, 1], c)
    Xa, Xb = np.hstack([d, sy]).astype(np.float32), np.hstack([-d, sy]).astype(np.float32)
    t0 = time.time()
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=V.SEED, **small).fit(X, Y)
    mg = 0.5 * (clf.decision_function(Xa) - clf.decision_function(Xb))
    mm = mlp_fit_predict(X, Y, Xa, Xb)
    rule = 100 * (s[:, 1] - s[:, 0])
    acc = lambda m: np.mean(m + rule > 0)
    res = {'gbdt': acc(mg), 'mlp': acc(mm)}
    for w in (0.3, 0.5):
        res[f'blend{w}'] = acc((1 - w) * mg + w * mm)
    print(f'T{task} test fold {tf}: ' + ' '.join(f'{k} {v:.4f}' for k, v in res.items()) + f' ({time.time()-t0:.0f}s)', flush=True)
