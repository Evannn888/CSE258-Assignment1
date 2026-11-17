"""Bayesian factorization machine v3 (numba Gibbs sampler).

  * relational slots, bags with self-exclusion (as v2)
  * likelihood: Gaussian, or ordinal probit (Albert-Chib latent variables; cutpoints follow a damped Newton step on the
    marginal likelihood each sweep, which is where their posterior concentrates with >1M ratings)
  * optional heteroscedastic noise: precision of a row = alpha * tau[user] * tau[item], tau ~ Gamma(nu/2, nu/2)
Returns the posterior mean / sd of the expected rating for the query rows and the mean of the latent predictor.
"""
import math
import numpy as np, numba as nb

SQ2 = math.sqrt(2.0)
ISQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


@nb.njit(cache=True, inline='always')
def ncdf(x):
    return 0.5 * math.erfc(-x / SQ2)


@nb.njit(cache=True, inline='always')
def npdf(x):
    return ISQ2PI * math.exp(-0.5 * x * x)


@nb.njit(cache=True)
def ndtri(p):
    """Acklam's inverse normal cdf (relative error ~1e-9)."""
    if p <= 0.0:
        return -40.0
    if p >= 1.0:
        return 40.0
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q - 2.400758277161838e+00) * q - 2.549732539343734e+00) * q
                 + 4.374664141464968e+00) * q + 2.938163982698783e+00) / \
               ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0)
    if p <= 1.0 - 0.02425:
        q = p - 0.5
        r = q * q
        return (((((-3.969683028665376e+01 * r + 2.209460984245205e+02) * r - 2.759285104469687e+02) * r + 1.383577518672690e+02) * r
                 - 3.066479806614716e+01) * r + 2.506628277459239e+00) * q / \
               (((((-5.447609879822406e+01 * r + 1.615858368580409e+02) * r - 1.556989798598866e+02) * r + 6.680131188771972e+01) * r
                 - 1.328068155288572e+01) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q - 2.400758277161838e+00) * q - 2.549732539343734e+00) * q
              + 4.374664141464968e+00) * q + 2.938163982698783e+00) / \
            ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0)


@nb.njit(cache=True)
def trunc_std_normal(a, b):
    """One draw from N(0,1) restricted to (a, b)."""
    flip = a > 0.0
    if flip:
        a, b = -b, -a
    pa, pb = ncdf(a), ncdf(b)
    if pb - pa < 1e-300:
        x = b if b < 0.0 else a
    else:
        x = ndtri(pa + (pb - pa) * np.random.rand())
        if x < a:
            x = a
        if x > b:
            x = b
    return -x if flip else x


@nb.njit(cache=True)
def _hyper(theta, lam, mu, a_l, b_l, g0):
    p = len(theta)
    s = 0.0
    for x in theta:
        s += x
    mu = s / (p + g0) + np.random.randn() / np.sqrt((p + g0) * lam)
    q = g0 * mu * mu
    for x in theta:
        q += (x - mu) * (x - mu)
    lam = np.random.gamma((a_l + p + 1.0) / 2.0, 2.0 / (b_l + q))
    return lam, mu


@nb.njit(cache=True)
def gibbs3(K, n_iter, burn, seed, y, ent, eoff, bag_ptr, bag_ent, bag_c, bag_row, goff, gslot, self_feat, self_c,
           q_ent, q_y, init_std, lo, hi, probit, hetero, nu_tau, slot_id, verbose):
    np.random.seed(seed)
    S, n = ent.shape
    E = eoff[S]
    NG = len(gslot)
    NB = goff[NG]
    nq = q_ent.shape[1]
    has_self = np.zeros(S, np.bool_)
    for s in range(S):
        for i in range(n):
            if self_feat[s, i] >= 0:
                has_self[s] = True
                break
    V = np.random.randn(K, E) * init_std
    for s in range(S):
        if slot_id[s] == 0:                         # slot without id parameters: bags only
            V[:, eoff[s]:eoff[s + 1]] = 0.0
    Y = np.random.randn(K, max(NB, 1)) * init_std
    a = V.copy()
    for j in range(NB):
        for p in range(bag_ptr[j], bag_ptr[j + 1]):
            for f in range(K):
                a[f, bag_ent[p]] += bag_c[p] * Y[f, j]
    w = np.zeros(E)
    ncat = int(hi - lo) + 1
    cut = np.zeros(ncat + 1)
    cut[0], cut[ncat] = -1e9, 1e9
    z = y.copy()
    w0 = 0.0
    if probit:
        cnt = np.zeros(ncat)
        for i in range(n):
            cnt[int(y[i] - lo)] += 1
        c = 0.0
        for k in range(1, ncat):
            c += cnt[k - 1] / n
            cut[k] = 1.5 * ndtri(c)                 # start wider than N(0,1): latent variance = signal + noise
        for i in range(n):
            k = int(y[i] - lo)
            lb, ub = cut[k] / 1.5, cut[k + 1] / 1.5
            pl, pu = ncdf(lb), ncdf(ub)
            z[i] = 1.5 * (npdf(lb) - npdf(ub)) / max(pu - pl, 1e-12)
    else:
        for i in range(n):
            w0 += y[i]
        w0 /= n
    tau = np.ones(E)                                # per-entity precision factors (slots 0 and 1 only)
    om = np.ones(n)                                 # row precision factor
    e = np.empty(n)
    for i in range(n):
        pred = w0
        for f in range(K):
            t1, t2 = 0.0, 0.0
            for s in range(S):
                v = a[f, ent[s, i]]
                if self_feat[s, i] >= 0:
                    v -= self_c[s, i] * Y[f, self_feat[s, i]]
                t1 += v
                t2 += v * v
            pred += 0.5 * (t1 * t1 - t2)
        e[i] = z[i] - pred
    alpha = 1.0
    lam_w, mu_w = np.ones(S), np.zeros(S)
    lam_v, mu_v = np.ones((S, K)), np.zeros((S, K))
    lam_y, mu_y = np.ones((NG, K)), np.zeros((NG, K))
    A, G, da = np.zeros(E), np.zeros(E), np.zeros(E)
    obuf, sc = np.empty(n), np.zeros(n)
    acc, acc2, accm = np.zeros(nq), np.zeros(nq), np.zeros(nq)
    trace = np.zeros((n_iter, 4))
    gk, hk = np.zeros(ncat + 1), np.zeros(ncat + 1)
    nacc = 0
    for it in range(n_iter):
        # ---- noise precision
        if not probit:
            sse = 0.0
            for i in range(n):
                sse += om[i] * e[i] * e[i]
            alpha = np.random.gamma((1.0 + n) / 2.0, 2.0 / (1.0 + sse))
        if hetero:
            for s in range(2):
                o_s = 1 - s
                for r in range(eoff[s], eoff[s + 1]):
                    A[r] = 0.0
                    G[r] = 0.0
                for i in range(n):
                    r = ent[s, i]
                    A[r] += 1.0
                    G[r] += alpha * tau[ent[o_s, i]] * e[i] * e[i]
                for r in range(eoff[s], eoff[s + 1]):
                    tau[r] = np.random.gamma((nu_tau + A[r]) / 2.0, 2.0 / (nu_tau + G[r]))
            for i in range(n):
                om[i] = tau[ent[0, i]] * tau[ent[1, i]]
        # ---- global bias (fixed at 0 for probit: the cutpoints carry the location)
        if not probit:
            se, sw = 0.0, 0.0
            for i in range(n):
                se += om[i] * e[i]
                sw += om[i]
            prec = alpha * sw + 1e-6
            new = alpha * (se + w0 * sw) / prec + np.random.randn() / np.sqrt(prec)
            d = new - w0
            w0 = new
            for i in range(n):
                e[i] -= d
        # ---- entity biases
        for s in range(S):
            if slot_id[s] == 0:
                continue
            lam_w[s], mu_w[s] = _hyper(w[eoff[s]:eoff[s + 1]], lam_w[s], mu_w[s], 1.0, 1.0, 1.0)
            for r in range(eoff[s], eoff[s + 1]):
                A[r] = 0.0
                G[r] = 0.0
            for i in range(n):
                r = ent[s, i]
                A[r] += om[i]
                G[r] += om[i] * e[i]
            for r in range(eoff[s], eoff[s + 1]):
                prec = alpha * A[r] + lam_w[s]
                new = (alpha * (G[r] + w[r] * A[r]) + lam_w[s] * mu_w[s]) / prec + np.random.randn() / np.sqrt(prec)
                da[r] = new - w[r]
                w[r] = new
            for i in range(n):
                e[i] -= da[ent[s, i]]
        # ---- factors
        for f in range(K):
            af, Yf = a[f], Y[f]
            for s in range(S):
                lo_e, hi_e = eoff[s], eoff[s + 1]
                hs = has_self[s]
                for r in range(lo_e, hi_e):
                    A[r] = 0.0
                    G[r] = 0.0
                    da[r] = 0.0
                for i in range(n):
                    o = 0.0
                    for s2 in range(S):
                        if s2 != s:
                            v = af[ent[s2, i]]
                            if self_feat[s2, i] >= 0:
                                v -= self_c[s2, i] * Yf[self_feat[s2, i]]
                            o += v
                    obuf[i] = o
                    r = ent[s, i]
                    A[r] += om[i] * o * o
                    G[r] += om[i] * o * e[i]
                    if hs:
                        sc[i] = 0.0
                if slot_id[s] != 0:
                    lam_v[s, f], mu_v[s, f] = _hyper(V[f, lo_e:hi_e], lam_v[s, f], mu_v[s, f], 1.0, 1.0, 1.0)
                    lv, mv = lam_v[s, f], mu_v[s, f]
                    for r in range(lo_e, hi_e):
                        prec = alpha * A[r] + lv
                        new = (alpha * (G[r] + V[f, r] * A[r]) + lv * mv) / prec + np.random.randn() / np.sqrt(prec)
                        d = new - V[f, r]
                        V[f, r] = new
                        da[r] = d
                        G[r] -= A[r] * d
                for g in range(NG):
                    if gslot[g] != s:
                        continue
                    lam_y[g, f], mu_y[g, f] = _hyper(Yf[goff[g]:goff[g + 1]], lam_y[g, f], mu_y[g, f], 1.0, 1.0, 1.0)
                    ly, my = lam_y[g, f], mu_y[g, f]
                    for j in range(goff[g], goff[g + 1]):
                        s1, s2_ = 0.0, 0.0
                        for p in range(bag_ptr[j], bag_ptr[j + 1]):
                            r, c, t = bag_ent[p], bag_c[p], bag_row[p]
                            if t >= 0:
                                ot = obuf[t]
                                s2_ += c * c * (A[r] - om[t] * ot * ot)
                                s1 += c * (G[r] - om[t] * ot * (e[t] - ot * da[r]))
                            else:
                                s2_ += c * c * A[r]
                                s1 += c * G[r]
                        if s2_ < 0.0:
                            s2_ = 0.0
                        prec = alpha * s2_ + ly
                        new = (alpha * (s1 + Yf[j] * s2_) + ly * my) / prec + np.random.randn() / np.sqrt(prec)
                        d = new - Yf[j]
                        Yf[j] = new
                        for p in range(bag_ptr[j], bag_ptr[j + 1]):
                            r, c, t = bag_ent[p], bag_c[p], bag_row[p]
                            if t >= 0:
                                ot = obuf[t]
                                G[r] -= c * d * (A[r] - om[t] * ot * ot)
                                sc[t] = c * d
                            else:
                                G[r] -= c * d * A[r]
                            da[r] += c * d
                if hs:
                    for i in range(n):
                        e[i] -= obuf[i] * (da[ent[s, i]] - sc[i])
                else:
                    for i in range(n):
                        e[i] -= obuf[i] * da[ent[s, i]]
                for r in range(lo_e, hi_e):
                    af[r] += da[r]
        # ---- probit: cutpoints (Newton on the marginal likelihood) and latent targets
        if probit:
            for rep in range(2):
                for k in range(ncat + 1):
                    gk[k] = 0.0
                    hk[k] = 0.0
                for i in range(n):
                    k = int(y[i] - lo)
                    sq = math.sqrt(om[i])
                    m_ = z[i] - e[i]
                    lb, ub = sq * (cut[k] - m_), sq * (cut[k + 1] - m_)
                    if lb > 0.0:
                        P = ncdf(-lb) - ncdf(-ub)
                    else:
                        P = ncdf(ub) - ncdf(lb)
                    if P < 1e-10:
                        P = 1e-10
                    if k + 1 < ncat:
                        r_ = npdf(ub) / P
                        gk[k + 1] += sq * r_
                        hk[k + 1] += om[i] * (-ub * r_ - r_ * r_)
                    if k >= 1:
                        r_ = npdf(lb) / P
                        gk[k] -= sq * r_
                        hk[k] += om[i] * (lb * r_ - r_ * r_)
                for k in range(1, ncat):
                    if hk[k] < -1e-9:
                        step = -gk[k] / hk[k]
                        if step > 0.2:
                            step = 0.2
                        if step < -0.2:
                            step = -0.2
                        cut[k] += 0.7 * step
                for k in range(2, ncat):
                    if cut[k] < cut[k - 1] + 1e-3:
                        cut[k] = cut[k - 1] + 1e-3
            for i in range(n):
                k = int(y[i] - lo)
                sq = math.sqrt(om[i])
                m_ = z[i] - e[i]
                x = trunc_std_normal(sq * (cut[k] - m_), sq * (cut[k + 1] - m_))
                e[i] = x / sq
                z[i] = m_ + e[i]
        # ---- predictions for the query rows
        if it >= burn:
            nacc += 1
        sse_q, sse_1, nlab = 0.0, 0.0, 0
        for i in range(nq):
            pred = w0
            for s in range(S):
                pred += w[q_ent[s, i]]
            for f in range(K):
                t1, t2 = 0.0, 0.0
                for s in range(S):
                    v = a[f, q_ent[s, i]]
                    t1 += v
                    t2 += v * v
                pred += 0.5 * (t1 * t1 - t2)
            lat = pred
            if probit:
                sq = math.sqrt(tau[q_ent[0, i]] * tau[q_ent[1, i]]) if hetero else 1.0
                ex = lo
                for k in range(1, ncat):
                    ex += ncdf(sq * (pred - cut[k]))
                pred = ex
            if not np.isnan(q_y[i]):
                d0 = min(max(pred, lo), hi) - q_y[i]
                sse_1 += d0 * d0
            if it >= burn:
                acc[i] += pred
                acc2[i] += pred * pred
                accm[i] += lat
                if not np.isnan(q_y[i]):
                    nlab += 1
                    d1 = min(max(acc[i] / nacc, lo), hi) - q_y[i]
                    sse_q += d1 * d1
        tr_sse = 0.0
        for i in range(n):
            tr_sse += e[i] * e[i]
        trace[it, 0] = tr_sse / n
        if nlab > 0:
            trace[it, 1] = sse_q / nlab
            trace[it, 3] = sse_1 / nlab
        elif it < burn:
            nl = 0
            for i in range(nq):
                if not np.isnan(q_y[i]):
                    nl += 1
            trace[it, 3] = sse_1 / max(nl, 1)
        trace[it, 2] = alpha
        if verbose and (it % verbose == 0 or it == n_iter - 1):
            print(it, 'alpha', alpha, 'train mse', tr_sse / n, 'val: running mean', trace[it, 1], 'this sample', trace[it, 3], 'cut', cut[1], cut[2], cut[3], cut[4])
    m = acc / max(nacc, 1)
    sd = np.sqrt(np.maximum(acc2 / max(nacc, 1) - m * m, 0.0))
    return m, sd, accm / max(nacc, 1), trace


def run(design, y, q_y, K, n_iter, burn, seed, init_std=0.1, lo=1.0, hi=5.0, probit=0, hetero=0, nu_tau=10.0, slot_id=None, verbose=0):
    d = design.build()
    return gibbs3(K, n_iter, burn, seed, np.asarray(y, np.float64), d['ent'], d['eoff'], d['bag_ptr'], d['bag_ent'], d['bag_c'], d['bag_row'],
                  d['goff'], d['gslot'], d['self_feat'], d['self_c'], d['q_ent'], np.asarray(q_y, np.float64), init_std, lo, hi,
                  probit, hetero, nu_tau, np.ones(len(d['eoff']) - 1, np.int64) if slot_id is None else np.asarray(slot_id, np.int64), verbose)

from bfm2 import Design
