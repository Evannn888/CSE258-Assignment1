"""Batched GPU Gibbs sampler for the Bayesian factorization machine: one chain per fold, all folds at once.

Same model as bfm_gibbs (assignment1.py, v10): slots with bias / factors, bags of shared feature vectors on the user
and item slots, user- and item-specific noise precisions, Gaussian or ordinal-probit likelihood.  Differences in the
sampling scheme (same posterior): the bag vectors of one factor dimension are drawn jointly from their Gaussian
conditional (perturbation + preconditioned conjugate gradients) instead of one coordinate at a time, and every fold's
chain runs in the same tensors (rows of the fold's own part get weight 0)."""
import math, time
import numpy as np, torch

SQ2 = math.sqrt(2.0)


def _gamma(conc, rate):
    return torch._standard_gamma(conc) / rate


def _ncdf(x):
    return 0.5 * torch.erfc(-x / SQ2)


def _npdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _trunc_std_normal(a, b):
    """Draws from N(0, 1) restricted to (a, b) (float64 tensors)."""
    flip = a > 0
    a2, b2 = torch.where(flip, -b, a), torch.where(flip, -a, b)
    pa, pb = _ncdf(a2), _ncdf(b2)
    u = pa + (pb - pa) * torch.rand_like(pa)
    x = torch.special.ndtri(u.clamp(1e-300, 1 - 1e-16))
    x = torch.minimum(torch.maximum(x, a2), b2)
    x = torch.where(pb - pa < 1e-300, torch.where(b2 < 0, b2, a2), x)
    return torch.where(flip, -x, x)


def _hyper(theta, lam):
    """Normal mean / Gamma precision of one parameter group per fold (theta (F, P)), unit hyper-priors."""
    p = theta.shape[1]
    mu = theta.sum(1) / (p + 1) + torch.randn_like(lam) / torch.sqrt((p + 1) * lam)
    q = mu * mu + ((theta - mu[:, None]) ** 2).sum(1)
    return _gamma(torch.full_like(lam, (p + 2) / 2), (1 + q) / 2), mu


def csr(rows, cols, vals, shape, dev):
    o = np.lexsort((cols, rows))
    crow = np.r_[0, np.cumsum(np.bincount(rows, minlength=shape[0]))]
    return torch.sparse_csr_tensor(torch.as_tensor(crow, device=dev), torch.as_tensor(cols[o], device=dev),
                                   torch.as_tensor(vals[o], dtype=torch.float32, device=dev), size=shape)


class BFMGPU:
    """slots: list of dict(ent=(n,) int, q=(F, L) int, n=#entities, kind=1 factors+bias / 2 bias only,
    bag=None or (feat (m,) int, ent (m,) int, coef (m,) float, J))."""

    def __init__(self, y, mask, slots, K, probit=False, nu_tau=30.0, lo=1.0, hi=5.0, seed=0, cg_iters=12, dev='cuda'):
        torch.manual_seed(seed)
        self.dev, self.K, self.probit, self.nu, self.lo, self.hi, self.cg_iters = dev, K, probit, nu_tau, lo, hi, cg_iters
        T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
        self.mask = T(mask)                                    # (F, n) 1 = training row of that fold's chain
        F, n = self.mask.shape
        self.F, self.n = F, n
        self.y = T(y)
        self.ntr = self.mask.sum(1)
        self.S = []
        for sl in slots:
            s = dict(kind=sl['kind'], n=int(sl['n']), ent=T(sl['ent'], torch.int64), q=T(sl['q'], torch.int64))
            s['idx'] = (torch.arange(F, device=dev)[:, None] * s['n'] + s['ent'][None, :]).reshape(-1)
            s['qidx'] = s['q']
            s['w'] = torch.zeros(F, s['n'], device=dev)
            s['lam_w'], s['mu_w'] = torch.ones(F, device=dev), torch.zeros(F, device=dev)
            s['fac'] = sl['kind'] == 1 or sl.get('bag') is not None
            if sl['kind'] == 1:
                s['V'] = torch.randn(K, F, s['n'], device=dev) * 0.1
                s['lam_v'], s['mu_v'] = torch.ones(K, F, device=dev), torch.zeros(K, F, device=dev)
            if sl.get('bag') is not None:
                feat, ent, coef, J = sl['bag']
                s['J'] = J
                s['C'] = csr(ent, feat, coef, (s['n'], J), dev)
                s['CT'] = csr(feat, ent, coef, (J, s['n']), dev)
                s['C2T'] = csr(feat, ent, coef ** 2, (J, s['n']), dev)
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
            freq = torch.bincount(self.cat, minlength=self.ncat).double() / n
            c = torch.cumsum(freq, 0)[:-1]
            cut = torch.cat([torch.tensor([-1e9], device=dev, dtype=torch.float64), 1.5 * torch.special.ndtri(c),
                             torch.tensor([1e9], device=dev, dtype=torch.float64)])
            self.cut = cut[None, :].repeat(F, 1)                                   # (F, ncat + 1) float64
            lb, ub = (cut[self.cat] / 1.5).clamp(-8, 8), (cut[self.cat + 1] / 1.5).clamp(-8, 8)
            z0 = 1.5 * (_npdf(lb) - _npdf(ub)) / (_ncdf(ub) - _ncdf(lb)).clamp_min(1e-12)
            self.z = z0.float()[None, :].repeat(F, 1)
            self.w0 = torch.zeros(F, device=dev)
        else:
            self.z = self.y[None, :].expand(F, n)
            self.w0 = (self.y[None, :] * self.mask).sum(1) / self.ntr
        self.e = self.z - self.predict_rows()

    # ------------------------------------------------------------------ helpers
    def segsum(self, s, vals):
        S = self.S[s]
        return torch.zeros(self.F * S['n'], device=self.dev).index_add_(0, S['idx'], vals.reshape(-1)).view(self.F, S['n'])

    def gather(self, s, x):                       # x (F, n_ent) -> (F, n)
        return torch.index_select(x, 1, self.S[s]['ent'])

    def _rebuild_a(self):
        for s in self.S:
            if not s['fac']:
                continue
            for f in range(self.K):
                a = s['V'][f].clone() if 'V' in s else torch.zeros(self.F, s['n'], device=self.dev)
                if 'Y' in s:
                    a += torch.sparse.mm(s['C'], s['Y'][f].T.contiguous()).T
                s['a'][f] = a

    def predict_rows(self):
        p = self.w0[:, None] + sum(self.gather(k, s['w']) for k, s in enumerate(self.S) if s['kind'] in (1, 2))
        fs = [k for k, s in enumerate(self.S) if s['fac']]
        for f in range(self.K):
            t1, t2 = 0, 0
            for k in fs:
                v = self.gather(k, self.S[k]['a'][f])
                t1 = t1 + v
                t2 = t2 + v * v
            p = p + 0.5 * (t1 * t1 - t2)
        return p

    def predict_queries(self):
        F = self.F
        g = lambda s, x: torch.gather(x, 1, s['q'])
        p = self.w0[:, None] + sum(g(s, s['w']) for s in self.S if s['kind'] in (1, 2))
        for f in range(self.K):
            t1, t2 = 0, 0
            for s in self.S:
                if s['fac']:
                    v = g(s, s['a'][f])
                    t1 = t1 + v
                    t2 = t2 + v * v
            p = p + 0.5 * (t1 * t1 - t2)
        if self.probit:
            sq = torch.sqrt(g(self.S[0], self.tau[0]) * g(self.S[1], self.tau[1])).double()
            pd = p.double()
            ex = torch.full_like(pd, self.lo)
            for c in range(1, self.ncat):
                ex += _ncdf(sq * (pd - self.cut[:, c:c + 1]))
            p = ex.float()
        return p

    def pcg(self, s, W, rhs, lam, Minv):
        """Solve (C^T diag(W) C + lam) x = rhs for every fold (columns), W (F, n_ent), rhs (J, F)."""
        S = self.S[s]
        Wt = W.T.contiguous()
        op = lambda P: torch.sparse.mm(S['CT'], Wt * torch.sparse.mm(S['C'], P)) + lam[None, :] * P
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        zc = Minv * r
        p = zc.clone()
        rz = (r * zc).sum(0)
        for _ in range(self.cg_iters):
            Ap = op(p)
            ac = rz / (p * Ap).sum(0).clamp_min(1e-30)
            x += ac * p
            r -= ac * Ap
            zc = Minv * r
            rz_new = (r * zc).sum(0)
            p = zc + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        self.cg_rel = (r.norm(dim=0) / rhs.norm(dim=0).clamp_min(1e-30)).max().item()
        return x

    # ------------------------------------------------------------------ one sweep
    def sweep(self):
        F, dev = self.F, self.dev
        al = lambda: self.alpha[:, None]
        if not self.probit:
            sse = (self.om * self.e * self.e).sum(1)
            self.alpha = _gamma((1 + self.ntr) / 2, (1 + sse) / 2)
        if self.nu > 0:
            for s in (0, 1):
                other = self.gather(1 - s, self.tau[1 - s])
                G = self.segsum(s, self.mask * al() * other * self.e * self.e)
                self.tau[s] = _gamma((self.nu + self.cnt[s]) / 2, (self.nu + G) / 2)
            self.tw = self.gather(0, self.tau[0]) * self.gather(1, self.tau[1])
            self.om = self.mask * self.tw
        if not self.probit:
            se, sw = (self.om * self.e).sum(1), self.om.sum(1)
            prec = self.alpha * sw + 1e-6
            new = self.alpha * (se + self.w0 * sw) / prec + torch.randn(F, device=dev) / torch.sqrt(prec)
            self.e -= (new - self.w0)[:, None]
            self.w0 = new
        for k, s in enumerate(self.S):
            if s['kind'] not in (1, 2):
                continue
            s['lam_w'], s['mu_w'] = _hyper(s['w'], s['lam_w'])
            A, G = self.segsum(k, self.om), self.segsum(k, self.om * self.e)
            prec = al() * A + s['lam_w'][:, None]
            new = (al() * (G + s['w'] * A) + (s['lam_w'] * s['mu_w'])[:, None]) / prec + torch.randn_like(A) / torch.sqrt(prec)
            d = new - s['w']
            s['w'] = new
            self.e -= self.gather(k, d)
        fs = [k for k, s in enumerate(self.S) if s['fac']]
        for f in range(self.K):
            tot = sum(self.gather(k, self.S[k]['a'][f]) for k in fs)
            for k in fs:
                s = self.S[k]
                af = s['a'][f]
                o = tot - self.gather(k, af)
                wo = self.om * o
                A, G = self.segsum(k, wo * o), self.segsum(k, wo * self.e)
                if 'Y' in s:
                    # bag vectors first, with the entity's own factor integrated out (its prior caps how strongly a
                    # heavy user pins the bag: the system stays well conditioned), then the own factor given the bag
                    Vf, Yf = s['V'][f], s['Y'][f]
                    s['lam_v'][f], s['mu_v'][f] = _hyper(Vf, s['lam_v'][f])
                    s['lam_y'][f], s['mu_y'][f] = _hyper(Yf, s['lam_y'][f])
                    lv, mv = s['lam_v'][f][:, None], s['mu_v'][f][:, None]
                    lam, mu = s['lam_y'][f], s['mu_y'][f]
                    aA = al() * A
                    P = aA * lv / (aA + lv)
                    Pt = P * (Vf - mv) + lv / (aA + lv) * al() * G
                    rhs = torch.sparse.mm(s['CT'], (Pt + torch.sqrt(P) * torch.randn_like(P)).T.contiguous()) \
                        + (lam[:, None] * (mu[:, None] - Yf)).T + torch.sqrt(lam)[None, :] * torch.randn(s['J'], F, device=dev)
                    diag = torch.sparse.mm(s['C2T'], P.T.contiguous()) + lam[None, :]
                    delta = self.pcg(k, P, rhs, lam, 1.0 / diag)
                    s['Y'][f] = Yf + delta.T
                    cd = torch.sparse.mm(s['C'], delta).T
                    prec = lv + aA
                    Vn = (lv * mv + aA * (Vf - cd) + al() * G) / prec + torch.randn_like(A) / torch.sqrt(prec)
                    d = Vn - Vf + cd
                    s['V'][f] = Vn
                else:
                    s['lam_v'][f], s['mu_v'][f] = _hyper(s['V'][f], s['lam_v'][f])
                    lam, mu = s['lam_v'][f][:, None], s['mu_v'][f][:, None]
                    prec = al() * A + lam
                    Vf = s['V'][f]
                    new = (al() * (G + Vf * A) + lam * mu) / prec + torch.randn_like(A) / torch.sqrt(prec)
                    d = new - Vf
                    s['V'][f] = new
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
        msk = self.mask.double()
        cat = self.cat[None, :].expand(F, -1)
        for rep in range(2):
            lb = sq * (torch.gather(self.cut, 1, cat) - m_)
            ub = sq * (torch.gather(self.cut, 1, cat + 1) - m_)
            P = torch.where(lb > 0, _ncdf(-lb) - _ncdf(-ub), _ncdf(ub) - _ncdf(lb)).clamp_min(1e-10)
            ru, rl = _npdf(ub) / P, _npdf(lb) / P
            omd = self.om.double()
            up = (cat + 1 < nc).double() * msk
            dn = (cat >= 1).double() * msk
            gk = torch.zeros(F, nc + 1, device=self.dev, dtype=torch.float64)
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
        x = _trunc_std_normal(lb, ub)
        e_new = torch.where(msk > 0, x / sq.clamp_min(1e-12), self.e.double())
        self.e = e_new.float()
        self.z = (m_ + e_new).float()

    def run(self, iters, burn, recompute_every=10, verbose=0, q_y=None):
        acc, acc2, nacc = None, None, 0
        t0 = time.time()
        for it in range(iters):
            self.sweep()
            if (it + 1) % recompute_every == 0:          # remove float32 drift of the incremental updates
                self._rebuild_a()
                self.e = self.z - self.predict_rows()
            if it >= burn:
                p = self.predict_queries()
                acc = p.double() if acc is None else acc + p
                acc2 = p.double() ** 2 if acc2 is None else acc2 + p.double() ** 2
                nacc += 1
            if verbose and (it % verbose == 0 or it == iters - 1):
                msg = f'sweep {it}: alpha {self.alpha.mean().item():.3f}, cg rel.res {getattr(self, "cg_rel", 0):.2e}, {time.time() - t0:.0f}s'
                if q_y is not None and nacc:
                    m = (acc / nacc).float()
                    ok = ~torch.isnan(q_y)
                    msg += f', val MSE running mean {(((m.clamp(self.lo, self.hi) - q_y) ** 2)[ok]).mean().item():.5f}'
                print(msg, flush=True)
        m = acc / nacc
        return m.float().cpu().numpy(), torch.sqrt((acc2 / nacc - m * m).clamp_min(0)).float().cpu().numpy()
