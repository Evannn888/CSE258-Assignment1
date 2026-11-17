import numpy as np, numba as nb
DAY = 86400.0
TEND = 1509522985.0
# ---------------------------------------------------------------------------------------------
# Bayesian factorization machine (Gibbs sampling)
# ---------------------------------------------------------------------------------------------
import math

_SQ2 = math.sqrt(2.0)
_ISQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


@nb.njit(cache=True, inline='always')
def _ncdf(x):
    return 0.5 * math.erfc(-x / _SQ2)


@nb.njit(cache=True, inline='always')
def _npdf(x):
    return _ISQ2PI * math.exp(-0.5 * x * x)


@nb.njit(cache=True)
def _ndtri(p):
    """Inverse normal cdf (Acklam's rational approximation, relative error about 1e-9)."""
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
def _trunc_std_normal(a, b):
    """One draw from N(0, 1) restricted to (a, b)."""
    flip = a > 0.0
    if flip:
        a, b = -b, -a
    pa, pb = _ncdf(a), _ncdf(b)
    if pb - pa < 1e-300:
        x = b if b < 0.0 else a
    else:
        x = _ndtri(pa + (pb - pa) * np.random.rand())
        if x < a:
            x = a
        if x > b:
            x = b
    return -x if flip else x


@nb.njit(cache=True)
def _bfm_hyper(theta, lam):
    """Gibbs step for the Normal mean / Gamma precision of one parameter group (unit hyper-priors, as in libFM)."""
    p = len(theta)
    s = 0.0
    for x in theta:
        s += x
    mu = s / (p + 1.0) + np.random.randn() / np.sqrt((p + 1.0) * lam)
    q = mu * mu
    for x in theta:
        q += (x - mu) * (x - mu)
    lam = np.random.gamma((p + 2.0) / 2.0, 2.0 / (1.0 + q))
    return lam, mu


@nb.njit(cache=True)
def bfm_gibbs(K, n_iter, burn, seed, y, ent, eoff, kind, bag_ptr, bag_ent, bag_c, goff, gslot, q_ent, lo, hi, probit, nu_tau):
    """Gibbs sampler for a factorization machine over 'slots'.  Row i has one entity per slot (ent[s, i], global ids,
    slot s owns eoff[s]:eoff[s+1]); an entity r has a bias w[r] and a factor vector
        a[r] = V[r] + sum_j c_rj Y[j]          (its own factors + a bag of shared feature vectors)
    kind[s]: 1 = bias + factors, 2 = bias only, 0 = neither (bag only).  Prediction:
        w0 + sum_s w[ent_s] + sum_f sum_{s<s'} a[ent_s, f] a[ent_s', f]
    Bag features come in groups (goff, slot gslot[g]) stored column-wise: feature j touches entities
    bag_ent[bag_ptr[j]:bag_ptr[j+1]] with coefficients bag_c.  Every Gaussian prior has its own sampled mean and
    precision (per slot for biases, per slot/group and factor for vectors).  Noise: precision alpha * tau[user] *
    tau[item] with tau ~ Gamma(nu/2, nu/2) (heteroscedastic; slots 0 and 1), Gaussian likelihood or - probit=1 -
    an ordinal probit on the integer targets lo..hi (latent variables as in Albert & Chib; the cutpoints take a damped
    Newton step on the marginal likelihood in every sweep).  Returns the posterior mean and sd of the expected
    target for the query rows, averaged over the sweeps after burn-in."""
    np.random.seed(seed)
    S, n = ent.shape
    E = eoff[S]
    NG = len(gslot)
    NB = goff[NG]
    nq = q_ent.shape[1]
    V = np.random.randn(K, E) * 0.1
    fac = np.zeros(S, np.bool_)
    for s in range(S):
        if kind[s] != 1:
            V[:, eoff[s]:eoff[s + 1]] = 0.0
        else:
            fac[s] = True
    for g in range(NG):
        fac[gslot[g]] = True
    Y = np.random.randn(K, max(NB, 1)) * 0.1
    a = V.copy()
    for j in range(NB):
        for p in range(bag_ptr[j], bag_ptr[j + 1]):
            for f in range(K):
                a[f, bag_ent[p]] += bag_c[p] * Y[f, j]
    w = np.zeros(E)
    ncat = int(hi - lo) + 1
    cut = np.zeros(ncat + 1)
    cut[0], cut[ncat] = -1e9, 1e9
    z = y.copy()                                   # regression target (the latent variable for probit)
    w0 = 0.0
    if probit:
        cnt = np.zeros(ncat)
        for i in range(n):
            cnt[int(y[i] - lo)] += 1
        c = 0.0
        for k in range(1, ncat):
            c += cnt[k - 1] / n
            cut[k] = 1.5 * _ndtri(c)
        for i in range(n):
            k = int(y[i] - lo)
            lb, ub = cut[k] / 1.5, cut[k + 1] / 1.5
            z[i] = 1.5 * (_npdf(lb) - _npdf(ub)) / max(_ncdf(ub) - _ncdf(lb), 1e-12)
    else:
        for i in range(n):
            w0 += y[i]
        w0 /= n
    tau = np.ones(E)
    om = np.ones(n)                                # relative precision of each row
    e = np.empty(n)                                # residual z - prediction
    for i in range(n):
        pred = w0
        for f in range(K):
            t1, t2 = 0.0, 0.0
            for s in range(S):
                v = a[f, ent[s, i]]
                t1 += v
                t2 += v * v
            pred += 0.5 * (t1 * t1 - t2)
        e[i] = z[i] - pred
    alpha = 1.0
    lam_w, mu_w = np.ones(S), np.zeros(S)
    lam_v, mu_v = np.ones((S, K)), np.zeros((S, K))
    lam_y, mu_y = np.ones((NG, K)), np.zeros((NG, K))
    A, G, da = np.zeros(E), np.zeros(E), np.zeros(E)
    obuf, tot = np.empty(n), np.empty(n)
    acc, acc2 = np.zeros(nq), np.zeros(nq)
    gk, hk = np.zeros(ncat + 1), np.zeros(ncat + 1)
    nacc = 0
    for it in range(n_iter):
        if not probit:                             # noise precision (fixed to 1 for probit)
            sse = 0.0
            for i in range(n):
                sse += om[i] * e[i] * e[i]
            alpha = np.random.gamma((1.0 + n) / 2.0, 2.0 / (1.0 + sse))
        if nu_tau > 0.0:                           # user and item precision factors
            for s in range(2):
                for r in range(eoff[s], eoff[s + 1]):
                    A[r] = 0.0
                    G[r] = 0.0
                for i in range(n):
                    r = ent[s, i]
                    A[r] += 1.0
                    G[r] += alpha * tau[ent[1 - s, i]] * e[i] * e[i]
                for r in range(eoff[s], eoff[s + 1]):
                    tau[r] = np.random.gamma((nu_tau + A[r]) / 2.0, 2.0 / (nu_tau + G[r]))
            for i in range(n):
                om[i] = tau[ent[0, i]] * tau[ent[1, i]]
        if not probit:                             # global bias (the cutpoints carry the location for probit)
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
        for s in range(S):                         # entity biases
            if kind[s] == 0:
                continue
            lam_w[s], mu_w[s] = _bfm_hyper(w[eoff[s]:eoff[s + 1]], lam_w[s])
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
        for f in range(K):                         # factors, one dimension and one slot at a time
            af, Yf = a[f], Y[f]
            for i in range(n):                     # sum of the row's slot vectors in this dimension
                t_ = 0.0
                for s in range(S):
                    if fac[s]:
                        t_ += af[ent[s, i]]
                tot[i] = t_
            for s in range(S):
                if not fac[s]:
                    continue
                lo_e, hi_e = eoff[s], eoff[s + 1]
                for r in range(lo_e, hi_e):
                    A[r] = 0.0
                    G[r] = 0.0
                    da[r] = 0.0
                for i in range(n):                 # o = what multiplies this slot's vector in row i
                    r = ent[s, i]
                    o = tot[i] - af[r]
                    obuf[i] = o
                    A[r] += om[i] * o * o
                    G[r] += om[i] * o * e[i]
                if kind[s] == 1:
                    lam_v[s, f], mu_v[s, f] = _bfm_hyper(V[f, lo_e:hi_e], lam_v[s, f])
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
                    lam_y[g, f], mu_y[g, f] = _bfm_hyper(Yf[goff[g]:goff[g + 1]], lam_y[g, f])
                    ly, my = lam_y[g, f], mu_y[g, f]
                    for j in range(goff[g], goff[g + 1]):
                        s1, s2_ = 0.0, 0.0
                        for p in range(bag_ptr[j], bag_ptr[j + 1]):
                            r, c = bag_ent[p], bag_c[p]
                            s2_ += c * c * A[r]
                            s1 += c * G[r]
                        prec = alpha * s2_ + ly
                        new = (alpha * (s1 + Yf[j] * s2_) + ly * my) / prec + np.random.randn() / np.sqrt(prec)
                        d = new - Yf[j]
                        Yf[j] = new
                        for p in range(bag_ptr[j], bag_ptr[j + 1]):
                            r, c = bag_ent[p], bag_c[p]
                            G[r] -= c * d * A[r]
                            da[r] += c * d
                for i in range(n):
                    d = da[ent[s, i]]
                    e[i] -= obuf[i] * d
                    tot[i] += d
                for r in range(lo_e, hi_e):
                    af[r] += da[r]
        if probit:                                 # cutpoints, then the latent targets
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
                        P = _ncdf(-lb) - _ncdf(-ub)
                    else:
                        P = _ncdf(ub) - _ncdf(lb)
                    if P < 1e-10:
                        P = 1e-10
                    if k + 1 < ncat:
                        r_ = _npdf(ub) / P
                        gk[k + 1] += sq * r_
                        hk[k + 1] += om[i] * (-ub * r_ - r_ * r_)
                    if k >= 1:
                        r_ = _npdf(lb) / P
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
                x = _trunc_std_normal(sq * (cut[k] - m_), sq * (cut[k + 1] - m_))
                e[i] = x / sq
                z[i] = m_ + e[i]
        if it >= burn:                             # predictions for the query rows
            nacc += 1
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
                if probit:                         # expected category under the ordinal probit
                    sq = math.sqrt(tau[q_ent[0, i]] * tau[q_ent[1, i]])
                    ex = lo
                    for k in range(1, ncat):
                        ex += _ncdf(sq * (pred - cut[k]))
                    pred = ex
                acc[i] += pred
                acc2[i] += pred * pred
    m = acc / max(nacc, 1)
    return m, np.sqrt(np.maximum(acc2 / max(nacc, 1) - m * m, 0.0))


def bfm_bag(ent, feat, n_ent, n_feat):
    """Column-wise bag (feature -> entities) with coefficients |bag of the entity|^-1/2; duplicate pairs are merged."""
    key = np.unique(ent.astype(np.int64) * n_feat + feat)
    r, j = key // n_feat, key % n_feat
    c = np.bincount(r, minlength=n_ent).astype(np.float64)[r] ** -0.5
    o = np.argsort(j, kind='stable')
    return np.searchsorted(j[o], np.arange(n_feat + 1)), r[o], c[o]


def bfm_time_slot(kind, u, t, qu, qt):
    """Context slots from the timestamp: 'month' (calendar month; bias + factors), 'uyear' (the user in that year;
    bias + factors) and 'session' (the user on that day - timestamps are dates, Pacific midnight +- 5000 s; bias only).
    Returns the entity of every training row, of every query row, the number of entities and the slot kind."""
    c = lambda x: np.clip(np.asarray(x, np.float64), 1.0e9, TEND)
    if kind == 'month':
        a, b = ((c(t) - 1.0e9) // (30.44 * DAY)).astype(np.int64), ((c(qt) - 1.0e9) // (30.44 * DAY)).astype(np.int64)
        return a, b, int(max(a.max(), b.max())) + 1, 1
    if kind == 'uyear':
        code = lambda x: ((c(x) - 1.0e9) // (365.25 * DAY)).astype(np.int64)
    else:
        code = lambda x: np.round((c(x) - 27000.0) / DAY).astype(np.int64)
    key = np.r_[u, qu].astype(np.int64) * 100003 + np.r_[code(t), code(qt)]
    uq, inv = np.unique(key, return_inverse=True)
    return inv[:len(u)], inv[len(u):], len(uq), 1 if kind == 'uyear' else 2


def bfm_rating_predictions(fit, queries, nu, ni, seed, k=32, iters=500, burn=100, nu_tau=30.0, probit=False, thr=0.0, time=(), spare=1):
    """Bayesian 'SVD++ flipped': users and items with ids, each also carrying the bag of everything it interacted with
    (existence only: training rows of all ratings plus the query pairs), optional time slots.  Trained on the positive
    ratings of `fit`; returns posterior mean and sd of the rating (or of P(rating >= thr)) for `queries`."""
    pos = fit[:, 3] > 0
    u, i, t, y = fit[pos, 0], fit[pos, 1], fit[pos, 2], fit[pos, 3].astype(np.float64)
    qu, qi, qt = queries[:, 0], queries[:, 1], queries[:, 2]
    nU, nI = nu + spare, ni + spare                           # one spare id for users / items unseen in training
    ent, q_ent, eoff, kind = [u, nU + i], [qu, nU + qi], [0, nU, nU + nI], [1, 1]
    au, ai = np.r_[fit[:, 0], qu], np.r_[fit[:, 1], qi]
    pu, eu, cu = bfm_bag(au, ai, nU, nI)                      # user slot: the user's items
    pi, ei, ci = bfm_bag(ai, au, nI, nU)                      # item slot: the item's users
    for name in time:
        a_, b_, m, kd = bfm_time_slot(name, u, t, qu, qt)
        ent.append(eoff[-1] + a_); q_ent.append(eoff[-1] + b_); eoff.append(eoff[-1] + m); kind.append(kd)
    lo, hi = 1.0, 5.0
    if thr > 0:
        y, lo, hi, probit = (y >= thr).astype(np.float64), 0.0, 1.0, True
    return bfm_gibbs(k, iters, burn, seed, y, np.ascontiguousarray(np.vstack(ent).astype(np.int64)), np.array(eoff, np.int64),
                     np.array(kind, np.int64), np.r_[pu, pi[1:] + pu[-1]].astype(np.int64), np.r_[eu, nU + ei].astype(np.int64),
                     np.r_[cu, ci], np.array([0, nI, nI + nU], np.int64), np.array([0, 1], np.int64),
                     np.ascontiguousarray(np.vstack(q_ent).astype(np.int64)), lo, hi, int(probit), float(nu_tau))
