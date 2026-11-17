"""v11/assignment1.py from v10/assignment1.py: the Bayesian FM chains run on the GPU (all 18 folds in one batch)
instead of numba on the CPU.  BFM_SPECS are filled in from cache_v10/v11_specs.py."""
import re
src = open('/data/UCSD/CSE258/A1/v10/assignment1.py').read()
specs = open('/data/UCSD/CSE258/A1/cache_v10/v11_specs.py').read().strip()
s = src
# ---- docstring
old = "  Bayesian FM ....... Gibbs-sampled factorization machine (heteroscedastic / ordinal probit), new in v10\n"
assert old in s
s = s.replace(old, "  Bayesian FM ....... Gibbs-sampled factorization machine (heteroscedastic / ordinal probit) on the GPU,\n"
                   "                      one chain per fold, all 18 folds in one batch (new in v10, GPU in v11)\n")
# ---- replace the numba sampler by the GPU sampler
a = s.index("_SQ2 = math.sqrt(2.0)\n")
b = s.index("def bfm_bag(ent, feat, n_ent, n_feat):")
c = s.index("def bfm_rating_predictions(")
d = s.index("# =============================================================================================\n# Tasks 1 and 3: rating prediction")
helpers = s[b:c]                                   # bfm_bag, bfm_time_slot (unchanged)
gpu = r'''torch = None                  # bound to the configured torch module by _bfm_torch() (the sampler needs it at module level)


def _bfm_torch():
    global torch
    if torch is None:
        torch = torch_setup()
    return torch


def _t_gamma(conc, rate):
    return torch._standard_gamma(conc) / rate


def _t_ncdf(x):
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


def _t_npdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _t_trunc_normal(a, b):
    """Draws from N(0, 1) restricted to (a, b) (float64 tensors), by the inverse cdf on the side away from the mode."""
    flip = a > 0
    a2, b2 = torch.where(flip, -b, a), torch.where(flip, -a, b)
    pa, pb = _t_ncdf(a2), _t_ncdf(b2)
    x = torch.special.ndtri((pa + (pb - pa) * torch.rand_like(pa)).clamp(1e-300, 1 - 1e-16))
    x = torch.minimum(torch.maximum(x, a2), b2)
    x = torch.where(pb - pa < 1e-300, torch.where(b2 < 0, b2, a2), x)
    return torch.where(flip, -x, x)


def _t_hyper(theta, lam):
    """Gibbs step for the Normal mean / Gamma precision of one parameter group, per fold (theta (F, P)); unit hyper-priors
    as in libFM."""
    p = theta.shape[1]
    mu = theta.sum(1) / (p + 1) + torch.randn_like(lam) / torch.sqrt((p + 1) * lam)
    q = mu * mu + ((theta - mu[:, None]) ** 2).sum(1)
    return _t_gamma(torch.full_like(lam, (p + 2) / 2), (1 + q) / 2), mu


def _t_csr(rows, cols, vals, shape):
    o = np.lexsort((cols, rows))
    crow = np.r_[0, np.cumsum(np.bincount(rows, minlength=shape[0]))]
    return torch.sparse_csr_tensor(torch.as_tensor(crow, device='cuda'), torch.as_tensor(cols[o], device='cuda'),
                                   torch.as_tensor(vals[o], dtype=torch.float32, device='cuda'), size=shape)


class BayesianFM:
    """Gibbs sampler for a factorization machine over 'slots', one chain per fold, all folds in the same tensors.

    Row i has one entity per slot; entity r of slot s has a bias w[r] (kind 1 and 2) and - kind 1, or a slot with a bag -
    a factor vector a[r] = V[r] + sum_j C[r, j] Y[j] (own factors + a bag of shared feature vectors).  Prediction:
        w0 + sum_s w_s[ent_s] + sum_f sum_{s<s'} a_s[ent_s, f] a_s'[ent_s', f]
    Every Gaussian prior has its own sampled mean and precision (per slot / bag and factor).  Noise: precision
    alpha * tau[user] * tau[item] with tau ~ Gamma(nu/2, nu/2), and a Gaussian likelihood or (probit) an ordinal probit
    on the integer ratings (Albert-Chib latent variables, cutpoints by a damped Newton step on the marginal likelihood).
    Folds: mask[k, i] = 0 removes row i from chain k's likelihood.  Factor dimensions are sampled one at a time
    (libFM order); within a dimension the bag vectors of a slot are drawn jointly from their Gaussian conditional
    with the entities' own factors integrated out (perturbation + preconditioned conjugate gradients: well
    conditioned, ~1e-3 relative residual in 20 steps), then the own factors given the bags."""

    def __init__(self, y, mask, slots, K, probit=False, nu_tau=30.0, lo=1.0, hi=5.0, seed=0, cg_iters=20):
        torch.manual_seed(seed)
        dev = 'cuda'
        self.K, self.probit, self.nu, self.lo, self.hi, self.cg_iters = K, probit, nu_tau, lo, hi, cg_iters
        T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
        self.mask = T(mask)
        F, n = self.mask.shape
        self.F, self.n = F, n
        self.y = T(y)
        self.ntr = self.mask.sum(1)
        self.S = []
        for sl in slots:
            s = dict(kind=sl['kind'], n=int(sl['n']), ent=T(sl['ent'], torch.int64), q=T(sl['q'], torch.int64))
            s['idx'] = (torch.arange(F, device=dev)[:, None] * s['n'] + s['ent'][None, :]).reshape(-1)
            s['w'] = torch.zeros(F, s['n'], device=dev)
            s['lam_w'], s['mu_w'] = torch.ones(F, device=dev), torch.zeros(F, device=dev)
            s['fac'] = sl['kind'] == 1 or sl.get('bag') is not None
            if sl['kind'] == 1:
                s['V'] = torch.randn(K, F, s['n'], device=dev) * 0.1
                s['lam_v'], s['mu_v'] = torch.ones(K, F, device=dev), torch.zeros(K, F, device=dev)
            if sl.get('bag') is not None:
                feat, ent, coef, J = sl['bag']
                s['J'] = J
                s['C'] = _t_csr(ent, feat, coef, (s['n'], J))
                s['CT'] = _t_csr(feat, ent, coef, (J, s['n']))
                s['C2T'] = _t_csr(feat, ent, coef ** 2, (J, s['n']))
                s['Y'] = torch.randn(K, F, J, device=dev) * 0.1
                s['lam_y'], s['mu_y'] = torch.ones(K, F, device=dev), torch.zeros(K, F, device=dev)
            if s['fac']:
                s['a'] = torch.zeros(K, F, s['n'], device=dev)
            self.S.append(s)
        self._rebuild_a()
        self.tau = [torch.ones(F, self.S[s]['n'], device=dev) for s in (0, 1)]
        self.cnt = [self.segsum(0, self.mask), self.segsum(1, self.mask)]
        self.tw = torch.ones(F, n, device=dev)
        self.om = self.mask.clone()
        self.alpha = torch.ones(F, device=dev)
        self.ncat = int(hi - lo) + 1
        self.cat = (self.y - lo).long().clamp(0, self.ncat - 1)
        if probit:
            c = torch.cumsum(torch.bincount(self.cat, minlength=self.ncat).double() / n, 0)[:-1]
            big = torch.tensor([1e9], device=dev, dtype=torch.float64)
            cut = torch.cat([-big, 1.5 * torch.special.ndtri(c), big])
            self.cut = cut[None, :].repeat(F, 1)
            lb, ub = (cut[self.cat] / 1.5).clamp(-8, 8), (cut[self.cat + 1] / 1.5).clamp(-8, 8)
            z0 = 1.5 * (_t_npdf(lb) - _t_npdf(ub)) / (_t_ncdf(ub) - _t_ncdf(lb)).clamp_min(1e-12)
            self.z = z0.float()[None, :].repeat(F, 1)
            self.w0 = torch.zeros(F, device=dev)
        else:
            self.z = self.y[None, :].expand(F, n)
            self.w0 = (self.y[None, :] * self.mask).sum(1) / self.ntr
        self.e = self.z - self.predict_rows()

    def segsum(self, s, vals):
        S = self.S[s]
        return torch.zeros(self.F * S['n'], device='cuda').index_add_(0, S['idx'], vals.reshape(-1)).view(self.F, S['n'])

    def gather(self, s, x):
        return torch.index_select(x, 1, self.S[s]['ent'])

    def _rebuild_a(self):
        for s in self.S:
            if s['fac']:
                for f in range(self.K):
                    a = s['V'][f].clone() if 'V' in s else torch.zeros(self.F, s['n'], device='cuda')
                    if 'Y' in s:
                        a += torch.sparse.mm(s['C'], s['Y'][f].T.contiguous()).T
                    s['a'][f] = a

    def _fm(self, g):
        """Prediction from per-slot gathers g(slot index, (F, n_ent) tensor)."""
        p = self.w0[:, None] + sum(g(k, s['w']) for k, s in enumerate(self.S) if s['kind'] in (1, 2))
        fs = [k for k, s in enumerate(self.S) if s['fac']]
        for f in range(self.K):
            t1, t2 = 0, 0
            for k in fs:
                v = g(k, self.S[k]['a'][f])
                t1, t2 = t1 + v, t2 + v * v
            p = p + 0.5 * (t1 * t1 - t2)
        return p

    def predict_rows(self):
        return self._fm(self.gather)

    def predict_queries(self):
        g = lambda k, x: torch.gather(x, 1, self.S[k]['q'])
        p = self._fm(g)
        if self.probit:                     # expected rating under the ordinal probit
            sq = torch.sqrt(g(0, self.tau[0]) * g(1, self.tau[1])).double()
            pd, ex = p.double(), torch.full_like(p, self.lo, dtype=torch.float64)
            for c in range(1, self.ncat):
                ex += _t_ncdf(sq * (pd - self.cut[:, c:c + 1]))
            p = ex.float()
        return p

    def _pcg(self, S, W, rhs, lam, Minv):
        """Solves (C^T diag(W) C + lam) x = rhs for every fold (columns of rhs) by preconditioned CG."""
        Wt = W.T.contiguous()
        op = lambda P: torch.sparse.mm(S['CT'], Wt * torch.sparse.mm(S['C'], P)) + lam[None, :] * P
        x, r = torch.zeros_like(rhs), rhs.clone()
        zc = Minv * r
        p, rz = zc.clone(), (r * zc).sum(0)
        for _ in range(self.cg_iters):
            Ap = op(p)
            ac = rz / (p * Ap).sum(0).clamp_min(1e-30)
            x += ac * p
            r -= ac * Ap
            zc = Minv * r
            rz_new = (r * zc).sum(0)
            p = zc + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        return x

    def sweep(self):
        F = self.F
        al = self.alpha[:, None]
        if not self.probit:                                  # noise precision
            self.alpha = _t_gamma((1 + self.ntr) / 2, (1 + (self.om * self.e * self.e).sum(1)) / 2)
            al = self.alpha[:, None]
        if self.nu > 0:                                      # user and item precision factors
            for s in (0, 1):
                G = self.segsum(s, self.mask * al * self.gather(1 - s, self.tau[1 - s]) * self.e * self.e)
                self.tau[s] = _t_gamma((self.nu + self.cnt[s]) / 2, (self.nu + G) / 2)
            self.tw = self.gather(0, self.tau[0]) * self.gather(1, self.tau[1])
            self.om = self.mask * self.tw
        if not self.probit:                                  # global bias (the cutpoints carry it for probit)
            se, sw = (self.om * self.e).sum(1), self.om.sum(1)
            prec = self.alpha * sw + 1e-6
            new = self.alpha * (se + self.w0 * sw) / prec + torch.randn(F, device='cuda') / torch.sqrt(prec)
            self.e -= (new - self.w0)[:, None]
            self.w0 = new
        for k, s in enumerate(self.S):                       # entity biases
            if s['kind'] in (1, 2):
                s['lam_w'], s['mu_w'] = _t_hyper(s['w'], s['lam_w'])
                A, G = self.segsum(k, self.om), self.segsum(k, self.om * self.e)
                prec = al * A + s['lam_w'][:, None]
                new = (al * (G + s['w'] * A) + (s['lam_w'] * s['mu_w'])[:, None]) / prec + torch.randn_like(A) / torch.sqrt(prec)
                d = new - s['w']
                s['w'] = new
                self.e -= self.gather(k, d)
        fs = [k for k, s in enumerate(self.S) if s['fac']]
        for f in range(self.K):                              # factors, one dimension at a time
            tot = sum(self.gather(k, self.S[k]['a'][f]) for k in fs)
            for k in fs:
                s = self.S[k]
                af = s['a'][f]
                o = tot - self.gather(k, af)                 # what multiplies this slot's vector in each row
                wo = self.om * o
                A, G = self.segsum(k, wo * o), self.segsum(k, wo * self.e)
                Vf = s['V'][f]
                s['lam_v'][f], s['mu_v'][f] = _t_hyper(Vf, s['lam_v'][f])
                lv, mv = s['lam_v'][f][:, None], s['mu_v'][f][:, None]
                aA = al * A
                if 'Y' in s:
                    Yf = s['Y'][f]
                    s['lam_y'][f], s['mu_y'][f] = _t_hyper(Yf, s['lam_y'][f])
                    lam, mu = s['lam_y'][f], s['mu_y'][f]
                    P = aA * lv / (aA + lv)                  # data precision on the bag part, own factor integrated out
                    Pt = P * (Vf - mv) + lv / (aA + lv) * al * G
                    rhs = torch.sparse.mm(s['CT'], (Pt + torch.sqrt(P) * torch.randn_like(P)).T.contiguous()) \
                        + (lam[:, None] * (mu[:, None] - Yf)).T + torch.sqrt(lam)[None, :] * torch.randn(s['J'], F, device='cuda')
                    diag = torch.sparse.mm(s['C2T'], P.T.contiguous()) + lam[None, :]
                    delta = self._pcg(s, P, rhs, lam, 1.0 / diag)   # exact draw of the change of the bag vectors
                    s['Y'][f] = Yf + delta.T
                    cd = torch.sparse.mm(s['C'], delta).T
                else:
                    cd = 0
                prec = lv + aA
                Vn = (lv * mv + aA * (Vf - cd) + al * G) / prec + torch.randn_like(A) / torch.sqrt(prec)
                d = Vn - Vf + cd
                s['V'][f] = Vn
                s['a'][f] = af + d
                dg = self.gather(k, d)
                self.e -= o * dg
                tot = tot + dg
        if self.probit:
            self._probit_step()

    def _probit_step(self):
        F, nc = self.F, self.ncat
        m_ = (self.z - self.e).double()
        sq = torch.sqrt(self.tw).double()
        msk, omd = self.mask.double(), self.om.double()
        cat = self.cat[None, :].expand(F, -1)
        up, dn = (cat + 1 < nc).double() * msk, (cat >= 1).double() * msk
        for _ in range(2):                                   # cutpoints: damped Newton steps on the marginal likelihood
            lb = sq * (torch.gather(self.cut, 1, cat) - m_)
            ub = sq * (torch.gather(self.cut, 1, cat + 1) - m_)
            P = torch.where(lb > 0, _t_ncdf(-lb) - _t_ncdf(-ub), _t_ncdf(ub) - _t_ncdf(lb)).clamp_min(1e-10)
            ru, rl = _t_npdf(ub) / P, _t_npdf(lb) / P
            gk = torch.zeros(F, nc + 1, device='cuda', dtype=torch.float64)
            hk = torch.zeros_like(gk)
            gk.scatter_add_(1, cat + 1, up * sq * ru)
            hk.scatter_add_(1, cat + 1, up * omd * (-ub * ru - ru * ru))
            gk.scatter_add_(1, cat, -dn * sq * rl)
            hk.scatter_add_(1, cat, dn * omd * (lb * rl - rl * rl))
            ok = hk[:, 1:nc] < -1e-9
            step = (-gk[:, 1:nc] / torch.where(ok, hk[:, 1:nc], torch.ones_like(hk[:, 1:nc]))).clamp(-0.2, 0.2)
            self.cut[:, 1:nc] += torch.where(ok, 0.7 * step, torch.zeros_like(step))
            for c in range(2, nc):
                self.cut[:, c] = torch.maximum(self.cut[:, c], self.cut[:, c - 1] + 1e-3)
        lb = sq * (torch.gather(self.cut, 1, cat) - m_)
        ub = sq * (torch.gather(self.cut, 1, cat + 1) - m_)
        e_new = torch.where(msk > 0, _t_trunc_normal(lb, ub) / sq.clamp_min(1e-12), self.e.double())
        self.e = e_new.float()
        self.z = (m_ + e_new).float()

    def run(self, iters, burn, refresh=10):
        """Posterior mean of the query predictions over the sweeps after burn-in."""
        acc, nacc = None, 0
        for it in range(iters):
            self.sweep()
            if (it + 1) % refresh == 0:                     # remove float32 drift of the incremental updates
                self._rebuild_a()
                self.e = self.z - self.predict_rows()
            if it >= burn:
                p = self.predict_queries().double()
                acc = p if acc is None else acc + p
                nacc += 1
        return (acc / nacc).float().cpu().numpy()


'''
fold = r'''def bfm_spec_name(spec):
    t = '-'.join(spec.get('time', ())) or 'none'
    return f"gbfm_k{spec['k']}_{'probit' if spec.get('probit') else 'gauss'}_{t}_i{spec['iters']}_b{spec['burn']}_s{spec.get('seed', 0)}"


def bfm_fold_chains(task, a, nu, ni, test_rows, folds, spec, cache=None):
    """All out-of-fold chains of one Bayesian-FM spec in one GPU run: chain k is trained on the positive ratings
    outside fold k and predicts every row of fold k, then the test rows.  Users / items: ids plus the bag of everything
    they interacted with (existence only: training rows of any rating plus the test pairs); optional time slots.
    With a cache directory the chains are stored / reused (GPU sampling is not bit-reproducible: index_add and sparse
    products accumulate in a varying order, so a re-run agrees only statistically)."""
    name = bfm_spec_name(spec)
    paths = None if cache is None else [cache / f'{name}_t{task}_f{k}.npy' for k in range(len(folds))]
    if paths is not None and all(p.exists() for p in paths):
        return [np.load(p) for p in paths]
    _bfm_torch()
    det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)          # deterministic index_add is ~30x slower
    F = len(folds)
    rowfold = np.empty(len(a), np.int64)
    for k, part in enumerate(folds):
        rowfold[part] = k
    pos = np.flatnonzero(a[:, 3] > 0)
    u, i, t, y = a[pos, 0], a[pos, 1], a[pos, 2], a[pos, 3].astype(np.float64)
    mask = np.stack([(rowfold[pos] != k).astype(np.float32) for k in range(F)])
    L = max(len(p) for p in folds) + len(test_rows)
    Q = np.stack([np.vstack([a[p], test_rows, np.repeat(test_rows[-1:], L - len(p) - len(test_rows), 0)]) for p in folds])
    nU, nI = nu + 1, ni + 1                                   # one spare id for users / items unseen in training
    au, ai = np.r_[a[:, 0], test_rows[:, 0]], np.r_[a[:, 1], test_rows[:, 1]]
    pu, eu, cu = bfm_bag(au, ai, nU, nI)                      # item feature -> users holding it
    pi, ei, ci = bfm_bag(ai, au, nI, nU)                      # user feature -> items holding it
    feat = lambda ptr: np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    slots = [dict(ent=u, q=Q[:, :, 0], n=nU, kind=1, bag=(feat(pu), eu, cu, nI)),
             dict(ent=i, q=Q[:, :, 1], n=nI, kind=1, bag=(feat(pi), ei, ci, nU))]
    for kind in spec.get('time', ()):
        tr, q, m, kd = bfm_time_slot(kind, u, t, Q[:, :, 0].reshape(-1), Q[:, :, 2].reshape(-1))
        slots.append(dict(ent=tr, q=q.reshape(F, L), n=m, kind=kd))
    t0 = time.time()
    model = BayesianFM(y, mask, slots, spec['k'], probit=spec.get('probit', False), nu_tau=spec.get('nu', 30.0), seed=spec.get('seed', 0))
    mean = model.run(spec['iters'], spec['burn'])
    del model
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(det, warn_only=True)
    out = [mean[k, :len(p) + len(test_rows)] for k, p in enumerate(folds)]
    oof = np.concatenate([o[:len(p)][a[p, 3] > 0] for o, p in zip(out, folds)])
    yy = np.concatenate([a[p, 3][a[p, 3] > 0] for p in folds])
    log(f'task {task}: Bayesian FM {spec}: out-of-fold MSE {np.mean((np.clip(oof, 1, 5) - yy) ** 2):.5f} ({time.time() - t0:.0f}s)')
    if paths is not None:
        for p, o in zip(paths, out):
            np.save(p, o)
    return out


'''
s = s[:a] + gpu + helpers + fold + s[d:]
# ---- specs, fold job, pipeline
a = s.index("# Bayesian FM chains per fold:")
b = s.index("def rating_problem(task, data_dir):")
s = s[:a] + specs + "\n\n\n" + s[b:]
a = s.index("def _bfm_fold_job(args):")
b = s.index("class ChunkedRidge:")
s = s[:a] + s[b:]
old = s[s.index("    # Bayesian FM chains first (light on memory"):s.index("    E, Et = rating_ease_features(task, a, folds, rows, qu, qi, nu, ni)")]
new = '''    # Bayesian FM chains first (GPU, all folds of a spec in one batch); their predictions become stacking features
    specs = BFM_SPECS[task][:args.bfm_specs]
    chains = [bfm_fold_chains(task, a, nu, ni, test_rows, folds, sp, args.bfm_cache) for sp in specs]
    P = [np.column_stack([c[k] for c in chains]) for k in range(RATING_FOLDS)]
    del chains
'''
s = s.replace(old, new)
s = s.replace("    ap.add_argument('--bfm-workers', type=int, default=6, help='parallel Gibbs chains (Tasks 1 and 3)')\n", "")
s = s.replace("    ap.add_argument('--bfm-cache', type=Path, default=None, help='directory to store / reuse the Gibbs chains')\n",
              "    ap.add_argument('--bfm-cache', type=Path, default=None, help='directory to store / reuse the Bayesian-FM chains')\n")
assert 'bfm_gibbs' not in s and '_bfm_fold_job' not in s and 'bfm_rating_predictions' not in s, 'leftover CPU sampler'
open('/data/UCSD/CSE258/A1/v11/assignment1.py', 'w').write(s)
print('v11 written', len(s.splitlines()), 'lines')
