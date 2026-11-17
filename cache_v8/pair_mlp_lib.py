import numpy as np, torch, torch.nn as nn, sys
sys.path.insert(0, '/data/UCSD/CSE258/A1')
import assignment1 as V
V.torch_setup()
dev = 'cuda'

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


