import sys, numpy as np, time
sys.path.insert(0, '/data/UCSD/CSE258/A1/cache_v8')
import sasrec_ce as SC
z = np.load('/data/UCSD/CSE258/A1/cache_v7/t5_sets.npz')
nU, nI = int(z['nU']), int(z['nI'])
users, qc = z['users0'], z['qc0']
ev = lambda S: ' '.join(f'{np.mean(S[:, 0] > S[:, j]):.4f}' for j in (1, 2, 3))
cfgs = {
    'd64_drop5_e10': dict(d=64, drop=0.5, epochs=10),
    'd128_drop5_wd_e10': dict(d=128, drop=0.5, epochs=10, wd=0.1),
    'd128_drop3_e6': dict(d=128, drop=0.3, epochs=6),
    'd256_drop5_wd_e10': dict(d=256, drop=0.5, epochs=10, wd=0.1),
    'd128_drop5_wd_e20': dict(d=128, drop=0.5, epochs=20, wd=0.1),
    'd128_drop6_wd3_e15': dict(d=128, drop=0.6, epochs=15, wd=0.3),
    'd128_drop5_wd_e10_L200': dict(d=128, drop=0.5, epochs=10, wd=0.1, L=200),
}
for name in sys.argv[1:]:
    t0 = time.time()
    S = SC.train_score(z['hu0'], z['hi0'], nU, nI, users, qc, seed=0, **cfgs[name])
    print(name, 'standalone acc', ev(S), f'{time.time()-t0:.0f}s', flush=True)
    np.save(f'/data/UCSD/CSE258/A1/cache_v8/sasce_{name}_S0.npy', S)
