"""v12/assignment1.py from v11/assignment1.py (applied in order; each block is optional via flags in cache_v10/v12_flags.json)."""
import json
flags = json.load(open('/data/UCSD/CSE258/A1/cache_v10/v12_flags.json'))
s = open('/data/UCSD/CSE258/A1/v11/assignment1.py').read()
# ---- (a) era slots for the Bayesian FM: entity attributes from the training timestamps
old = '''    for kind in spec.get('time', ()):
        tr, q, m, kd = bfm_time_slot(kind, u, t, Q[:, :, 0].reshape(-1), Q[:, :, 2].reshape(-1))
        slots.append(dict(ent=tr, q=q.reshape(F, L), n=m, kind=kd))
'''
new = '''    for kind in spec.get('time', ()):
        tr, q, m, kd = bfm_time_slot(kind, u, t, Q[:, :, 0].reshape(-1), Q[:, :, 2].reshape(-1))
        slots.append(dict(ent=tr, q=q.reshape(F, L), n=m, kind=kd))
    for kind in spec.get('era', ()):                          # shared factors for books / users of the same era
        side, how = {'icohort': (1, 'first'), 'imed': (1, 'median'), 'umed': (0, 'median')}[kind]
        code = bfm_era_codes(a, (nU, nI)[side], side, how)
        slots.append(dict(ent=code[(u, i)[side]], q=code[Q[:, :, side]], n=int(code.max()) + 1, kind=1))
'''
assert old in s; s = s.replace(old, new)
old = "def bfm_spec_name(spec):\n    t = '-'.join(spec.get('time', ())) or 'none'\n"
new = '''def bfm_era_codes(a, n, col, how):
    """Year (1 = first year of the data) in which an entity (users: col 0, books: col 1) first appears ('first') or
    has the median of its interactions ('median'), from the training timestamps; 0 = no usable timestamp."""
    t = a[:, 2].astype(np.float64)
    ok = (t > 1.0e9) & (t < TEND + DAY)
    o = np.lexsort((t[ok], a[ok, col]))
    ids, tt = a[ok, col][o], t[ok][o]
    ptr = np.searchsorted(ids, np.arange(n + 1))
    has = ptr[1:] > ptr[:-1]
    pick = ptr[:-1] if how == 'first' else (ptr[:-1] + ptr[1:]) // 2
    code = np.zeros(n, np.int64)
    code[has] = ((tt[np.minimum(pick[has], len(tt) - 1)] - 1.0e9) // (365.25 * DAY)).astype(np.int64) + 1
    return code


def bfm_spec_name(spec):
    t = '-'.join(spec.get('time', ()) + spec.get('era', ())) or 'none'
    if spec.get('nu', 30.0) != 30.0:
        t += f"_nu{spec['nu']:g}"
'''
assert old in s; s = s.replace(old, new)
# ---- (b) Task 6: larger trees, stronger L2, re-tuned scale table
if flags.get('t6'):
    old = s[s.index("T6_SCALE = [(90, 1.10), (365, 1.10), (730, 1.05), (np.inf, 1.03)]"):s.index("def task6_last_event_sets")]
    new = flags['t6'] + "\n\n\n"
    s = s.replace(old, new)
# ---- (c) Task 5: two sequence features - three seeds of the time-aware SASRec averaged, and a larger one
if flags.get('t5'):
    old = """    seqs = [sasrec_scores(gk, users, qck, seed=k, use_time=not order_only) for k, (_, _, _, (gk, users, qck, _)) in enumerate(sets)]
    seqs.append(sasrec_scores(g, qu, qc, seed=len(sets), use_time=not order_only))
"""
    new = """    def seq(gk, users, qck, k):
        \"\"\"Two sequence features: three seeds of the time-aware SASRec averaged (less noisy than one), and a larger
        SASRec (128 units, 30 epochs) - worse alone, but it adds to the classifier (0.9323 -> 0.9330 on cached sets).\"\"\"
        small = np.mean([sasrec_scores(gk, users, qck, seed=k + 100 * r, use_time=not order_only) for r in range(3)], axis=0)
        big = sasrec_scores(gk, users, qck, seed=k + 1000, d=128, epochs=30, use_time=not order_only)
        return np.stack([small, big], -1)
    seqs = [seq(gk, users, qck, k) for k, (_, _, _, (gk, users, qck, _)) in enumerate(sets)]
    seqs.append(seq(g, qu, qc, len(sets)))
"""
    assert old in s; s = s.replace(old, new)
    for old, new in (("    sets = [(np.concatenate([F, latent[k], seqs[k][..., None]], -1), ctx, seen) for k, (F, ctx, seen, _) in enumerate(sets)]\n",
                      "    sets = [(np.concatenate([F, latent[k], seqs[k]], -1), ctx, seen) for k, (F, ctx, seen, _) in enumerate(sets)]\n"),
                     ("    F = np.concatenate([F, latent[-1], seqs[-1][..., None]], -1)\n",
                      "    F = np.concatenate([F, latent[-1], seqs[-1]], -1)\n")):
        assert old in s, old
        s = s.replace(old, new)
# ---- (e) Bayesian FM: per-entity sums of slots with few entities by a sparse product (index_add serialises when
# thousands of rows add into the same few addresses: the era slots made a chain three times slower)
old = """            s['idx'] = (torch.arange(F, device=dev)[:, None] * s['n'] + s['ent'][None, :]).reshape(-1)
"""
new = """            s['idx'] = (torch.arange(F, device=dev)[:, None] * s['n'] + s['ent'][None, :]).reshape(-1)
            if s['n'] <= 5000:                       # few entities: sum by a sparse product instead of atomic adds
                ent_np = np.asarray(sl['ent'], np.int64)
                s['R'] = _t_csr(ent_np, np.arange(len(ent_np)), np.ones(len(ent_np)), (s['n'], len(ent_np)))
"""
assert old in s; s = s.replace(old, new)
old = """    def segsum(self, s, vals):
        S = self.S[s]
        return torch.zeros(self.F * S['n'], device='cuda').index_add_(0, S['idx'], vals.reshape(-1)).view(self.F, S['n'])
"""
new = """    def segsum(self, s, vals):
        S = self.S[s]
        if 'R' in S:
            return torch.sparse.mm(S['R'], vals.T.contiguous()).T
        return torch.zeros(self.F * S['n'], device='cuda').index_add_(0, S['idx'], vals.reshape(-1)).view(self.F, S['n'])
"""
assert old in s; s = s.replace(old, new)
# ---- (d) Bayesian FM specs
a = s.index("# Bayesian FM chains (GPU).")
b = s.index("def rating_problem(task, data_dir):")
s = s[:a] + open('/data/UCSD/CSE258/A1/cache_v10/v12_specs.py').read().strip() + "\n\n\n" + s[b:]
open('/data/UCSD/CSE258/A1/v12/assignment1.py', 'w').write(s)
print('v12 written', len(s.splitlines()), 'lines')
