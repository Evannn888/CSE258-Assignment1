"""Pair tasks: production classifier (GBDT + MLP blend, v12 code) on cached folds, each fold scored by a classifier
trained on the other three.  usage: pair_prod_check.py TASK variant[,variant...]  variants: base, lgcn, rate, lgcn+rate"""
import sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('a1v12p', '/data/UCSD/CSE258/A1/v12/assignment1.py')
M = importlib.util.module_from_spec(spec); sys.modules['a1v12p'] = M; spec.loader.exec_module(M)
task = int(sys.argv[1]); variants = sys.argv[2].split(',')
C = '/data/UCSD/CSE258/A1/'
z = np.load(C + f'cache_v7/pairs_t{task}.npz'); nI = int(z['nI'])
if task == 2:
    e = np.load(C + 'cache_v8/ease_t2_folds_300.npz'); lams = ['300']
else:
    e = np.load(C + 'cache_v8/ease_t4.npz'); lams = ['500', '2000']; b4r = np.load(C + 'cache_v7/t4_b4r.npz')
def feats(f, variant):
    keep = np.ones(len(z['u']), bool); keep[z[f'held{f}']] = False
    pop = np.bincount(z['i'][keep], minlength=nI).astype(float)
    parts = [z[f'F{f}'], z[f'L{f}']]
    if task == 4:
        parts.append(np.load(C + f'cache_v10/b4rt_gap+month_e120_f{f}.npy')[..., None])
    for lam in lams:
        S = e[f'E{f}_{lam}']
        parts.append(np.concatenate([S, S / np.maximum(pop[z[f'qc{f}']], 1)[..., None] ** 0.75], -1))
    if 'lgcn' in variant:
        parts.append(np.load(C + f'cache_v10/lgcn_t{task}_d64_L3_e35_f{f}.npy')[..., None])
    if 'rate' in variant:
        d = np.load(C + f'cache_v10/pairrat_t{task}_f{f}.npz'); parts.append(np.stack([d['S'], d['SD']], -1))
    return np.concatenate(parts, -1).astype(np.float32)
for variant in variants:
    S = [(feats(f, variant), z[f'ctx{f}'], z[f'seen{f}']) for f in range(4)]
    accs = []
    for tf in range(4):
        F, c, s = S[tf]
        m = M.blended_margin(task, [(S[k][0], S[k][1]) for k in range(4) if k != tf], 'small', 400, F[:, 0], F[:, 1], c)
        accs.append(np.mean((m + 100 * (s[:, 1] - s[:, 0])) > 0))
    print(f'T{task} {variant}: folds {np.round(accs, 4)} mean {np.mean(accs):.4f}', flush=True)
