import time, numpy as np, torch
dev = 'cuda'
F, n, E, J = 18, 1834017, 51008, 30910
rng = np.random.default_rng(0)
ent = rng.integers(0, E, n)
vals = torch.randn(F, n, device=dev)
ent_t = torch.as_tensor(ent, device=dev)
perm = torch.as_tensor(np.argsort(ent, kind='stable'), device=dev)
cnt = np.bincount(ent, minlength=E); ends = np.r_[0, np.cumsum(cnt)]
lo_t, hi_t = torch.as_tensor(ends[:-1], device=dev), torch.as_tensor(ends[1:], device=dev)
def seg_cumsum(v):
    cs = torch.nn.functional.pad(torch.cumsum(v[:, perm], dim=1, dtype=torch.float64), (1, 0))
    return (cs[:, hi_t] - cs[:, lo_t]).float()
idx = (torch.arange(F, device=dev)[:, None] * E + ent_t[None, :]).reshape(-1)
def seg_atomic(v):
    return torch.zeros(F * E, device=dev).index_add_(0, idx, v.reshape(-1)).view(F, E)
lens = torch.as_tensor(cnt, device=dev)
def seg_reduce(v):
    return torch.segment_reduce(v[:, perm], 'sum', lengths=lens.expand(F, -1).contiguous(), axis=1)
def bench(fn, *a, rep=20):
    fn(*a); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(rep):
        r = fn(*a)
    torch.cuda.synchronize(); return (time.time() - t0) / rep * 1000, r
for name, fn in (('cumsum fp64', seg_cumsum), ('index_add atomic', seg_atomic), ('segment_reduce', seg_reduce)):
    try:
        ms, r = bench(fn, vals)
        print(f'{name}: {ms:.2f} ms', flush=True)
        if name == 'cumsum fp64':
            ref = r
        else:
            print('   max diff vs cumsum', (r - ref).abs().max().item())
    except Exception as ex:
        print(name, 'failed', ex)
# gather
a = torch.randn(F, E, device=dev)
ms, _ = bench(lambda: torch.index_select(a, 1, ent_t)); print(f'gather (F,n) from (F,E): {ms:.2f} ms')
ms, _ = bench(lambda: vals * vals + vals); print(f'elementwise (F,n): {ms:.2f} ms')
# sparse CSR SpMM: C (E x J) nnz 1.9M
nnz = 1900000
r = np.sort(rng.integers(0, E, nnz)); c = rng.integers(0, J, nnz)
key = np.unique(r.astype(np.int64) * J + c); r, c = key // J, key % J
crow = np.r_[0, np.cumsum(np.bincount(r, minlength=E))]
C = torch.sparse_csr_tensor(torch.as_tensor(crow, device=dev), torch.as_tensor(c, device=dev), torch.rand(len(c), device=dev), size=(E, J))
CT = C.to_sparse_coo().t().coalesce().to_sparse_csr()
X = torch.randn(J, F, device=dev)
ms, _ = bench(lambda: torch.sparse.mm(C, X)); print(f'SpMM C(E x J, nnz {len(c)}) @ (J x {F}): {ms:.2f} ms')
Y = torch.randn(E, F, device=dev)
ms, _ = bench(lambda: torch.sparse.mm(CT, Y)); print(f'SpMM CT @ (E x {F}): {ms:.2f} ms')
r1 = torch.sparse.mm(C, X); r2 = torch.sparse.mm(C, X); print('SpMM deterministic:', torch.equal(r1, r2))
ms, _ = bench(lambda: torch.randn(F, n, device=dev)); print(f'randn (F,n): {ms:.2f} ms')
