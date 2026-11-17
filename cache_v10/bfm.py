"""Bayesian factorization machine with relational blocks, Gibbs sampling (numba).

Model (S slots, e.g. user / item / time-bin):  each row i has one entity per slot; the slot vector of entity r is
    a_s(r) = V_s[r] + sum_j C_s[r, j] * Y_s[j]          (id factors + optional bag of implicit features)
    yhat_i = w0 + sum_s w_s[r_s(i)] + sum_f sum_{s<s'} a_s,f(r_s(i)) * a_s',f(r_s'(i))
With slots (user + items-of-user bag, item + users-of-item bag) this is Bayesian "SVD++ flipped" (Rendle et al. 2019)
without the within-block constants.  All Gaussian priors have Gamma/Normal hyperpriors that are sampled, so there
is nothing to tune except the rank and the number of iterations.
"""
import math
import numpy as np, numba as nb


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
def gibbs(K, n_iter, burn, seed, y, ent, eoff, bag_ptr, bag_ent, bag_c, boff, q_ent, q_y, init_std, lo, hi, probit, verbose):
    """ent[s, i]: global entity id of row i in slot s; eoff[s]..eoff[s+1]: entity range of slot s.
    Bag features are global ids boff[s]..boff[s+1] for slot s; CSC (bag_ptr, bag_ent, bag_c) lists entities per feature.
    probit=1: ordinal probit on integer ratings lo..hi (Albert-Chib augmentation, cutpoints by random-walk MH)."""
    np.random.seed(seed)
    S, n = ent.shape
    E, NB = eoff[S], boff[S]
    nq = q_ent.shape[1]
    V = np.random.randn(K, E) * init_std
    Y = np.random.randn(K, max(NB, 1)) * init_std
    a = V.copy()
    for j in range(NB):
        for p in range(bag_ptr[j], bag_ptr[j + 1]):
            for f in range(K):
                a[f, bag_ent[p]] += bag_c[p] * Y[f, j]
    w = np.zeros(E)
    z = y.copy()                                   # regression target (latent variable for probit)
    ncat = int(hi - lo) + 1
    cut = np.zeros(ncat + 1)
    cut[0], cut[ncat] = -1e10, 1e10
    if probit:
        cnt = np.zeros(ncat)
        for i in range(n):
            cnt[int(y[i] - lo)] += 1
        c = 0.0
        for k in range(1, ncat):
            c += cnt[k - 1] / n
            # inverse normal cdf by bisection
            a_, b_ = -8.0, 8.0
            for _ in range(60):
                m_ = 0.5 * (a_ + b_)
                if 0.5 * (1 + math.erf(m_ / np.sqrt(2.0))) < c:
                    a_ = m_
                else:
                    b_ = m_
            cut[k] = 0.5 * (a_ + b_)
        for i in range(n):
            k = int(y[i] - lo)
            l_, h_ = max(cut[k], -3.0), min(cut[k + 1], 3.0)
            z[i] = 0.5 * (l_ + h_)
    w0 = 0.0
    for i in range(n):
        w0 += z[i]
    w0 /= n
    e = np.empty(n)
    for i in range(n):
        lin, s1, s2 = w0, np.zeros(K), np.zeros(K)
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
    lam_y, mu_y = np.ones((S, K)), np.zeros((S, K))
    A, G, da = np.zeros(E), np.zeros(E), np.zeros(E)
    obuf = np.empty(n)
    acc = np.zeros(nq)                 # running sum of predictions (expected rating)
    acc_c = np.zeros(nq)               # running sum of clipped predictions
    trace = np.zeros((n_iter, 3))
    nacc = 0
    for it in range(n_iter):
        # ---- noise precision (fixed to 1 for probit)
        if not probit:
            sse = 0.0
            for i in range(n):
                sse += e[i] * e[i]
            alpha = np.random.gamma((1.0 + n) / 2.0, 2.0 / (1.0 + sse))
        # ---- global bias
        se = 0.0
        for i in range(n):
            se += e[i]
        prec = alpha * n + 1e-6
        new = alpha * (se + w0 * n) / prec + np.random.randn() / np.sqrt(prec)
        d = new - w0
        w0 = new
        for i in range(n):
            e[i] -= d
        # ---- entity biases
        for s in range(S):
            lam_w[s], mu_w[s] = _hyper(w[eoff[s]:eoff[s + 1]], lam_w[s], mu_w[s], 1.0, 1.0, 1.0)
            for r in range(eoff[s], eoff[s + 1]):
                A[r] = 0.0
                G[r] = 0.0
            for i in range(n):
                r = ent[s, i]
                A[r] += 1.0
                G[r] += e[i]
            for r in range(eoff[s], eoff[s + 1]):
                prec = alpha * A[r] + lam_w[s]
                new = (alpha * (G[r] + w[r] * A[r]) + lam_w[s] * mu_w[s]) / prec + np.random.randn() / np.sqrt(prec)
                da[r] = new - w[r]
                w[r] = new
            for i in range(n):
                e[i] -= da[ent[s, i]]
        # ---- factors
        for f in range(K):
            af = a[f]
            for s in range(S):
                lo_e, hi_e = eoff[s], eoff[s + 1]
                for r in range(lo_e, hi_e):
                    A[r] = 0.0
                    G[r] = 0.0
                    da[r] = 0.0
                for i in range(n):
                    o = 0.0
                    for s2 in range(S):
                        if s2 != s:
                            o += af[ent[s2, i]]
                    obuf[i] = o
                    r = ent[s, i]
                    A[r] += o * o
                    G[r] += o * e[i]
                lam_v[s, f], mu_v[s, f] = _hyper(V[f, lo_e:hi_e], lam_v[s, f], mu_v[s, f], 1.0, 1.0, 1.0)
                lv, mv = lam_v[s, f], mu_v[s, f]
                for r in range(lo_e, hi_e):
                    prec = alpha * A[r] + lv
                    new = (alpha * (G[r] + V[f, r] * A[r]) + lv * mv) / prec + np.random.randn() / np.sqrt(prec)
                    d = new - V[f, r]
                    V[f, r] = new
                    da[r] = d
                    G[r] -= A[r] * d
                if boff[s + 1] > boff[s]:
                    lam_y[s, f], mu_y[s, f] = _hyper(Y[f, boff[s]:boff[s + 1]], lam_y[s, f], mu_y[s, f], 1.0, 1.0, 1.0)
                    ly, my = lam_y[s, f], mu_y[s, f]
                    for j in range(boff[s], boff[s + 1]):
                        s1, s2_ = 0.0, 0.0
                        for p in range(bag_ptr[j], bag_ptr[j + 1]):
                            r, c = bag_ent[p], bag_c[p]
                            s2_ += c * c * A[r]
                            s1 += c * G[r]
                        prec = alpha * s2_ + ly
                        new = (alpha * (s1 + Y[f, j] * s2_) + ly * my) / prec + np.random.randn() / np.sqrt(prec)
                        d = new - Y[f, j]
                        Y[f, j] = new
                        for p in range(bag_ptr[j], bag_ptr[j + 1]):
                            r, c = bag_ent[p], bag_c[p]
                            da[r] += c * d
                            G[r] -= A[r] * c * d
                for i in range(n):
                    e[i] -= obuf[i] * da[ent[s, i]]
                for r in range(lo_e, hi_e):
                    af[r] += da[r]
        # ---- probit: resample latent targets and cutpoints
        if probit:
            # cutpoints: uniform conditional between neighbouring latent values (Albert & Chib)
            mx = np.full(ncat, -1e10)
            mn = np.full(ncat, 1e10)
            for i in range(n):
                k = int(y[i] - lo)
                if z[i] > mx[k]:
                    mx[k] = z[i]
                if z[i] < mn[k]:
                    mn[k] = z[i]
            for k in range(1, ncat):
                l_, h_ = mx[k - 1], mn[k]
                if h_ > l_:
                    cut[k] = l_ + (h_ - l_) * np.random.rand()
            for i in range(n):
                k = int(y[i] - lo)
                m_ = z[i] - e[i]                       # current prediction
                l_, h_ = cut[k] - m_, cut[k + 1] - m_
                # truncated standard normal on (l_, h_) by inverse cdf with clamping
                pl = 0.5 * (1 + math.erf(l_ / np.sqrt(2.0)))
                ph = 0.5 * (1 + math.erf(h_ / np.sqrt(2.0)))
                if ph - pl < 1e-12:
                    x = l_ if l_ > 0 else h_
                    if x < -30:
                        x = h_
                    if x > 30:
                        x = l_
                else:
                    u_ = pl + (ph - pl) * np.random.rand()
                    u_ = min(max(u_, 1e-15), 1 - 1e-15)
                    a_, b_ = max(l_, -9.0), min(h_, 9.0)
                    for _ in range(40):
                        mm = 0.5 * (a_ + b_)
                        if 0.5 * (1 + math.erf(mm / np.sqrt(2.0))) < u_:
                            a_ = mm
                        else:
                            b_ = mm
                    x = 0.5 * (a_ + b_)
                znew = m_ + x
                e[i] += znew - z[i]
                z[i] = znew
        # ---- predictions for the query rows
        if it >= burn:
            nacc += 1
        sse_q, sse_c, nlab = 0.0, 0.0, 0
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
            if probit:
                ex = lo
                for k in range(1, ncat):
                    ex += 0.5 * (1 + math.erf((pred - cut[k]) / np.sqrt(2.0)))     # P(rating > k-th cut)
                pred = ex
            pc = min(max(pred, lo), hi)
            if it >= burn:
                acc[i] += pred
                acc_c[i] += pc
                if not np.isnan(q_y[i]):
                    nlab += 1
                    d1 = min(max(acc[i] / nacc, lo), hi) - q_y[i]
                    d2 = acc_c[i] / nacc - q_y[i]
                    sse_q += d1 * d1
                    sse_c += d2 * d2
        tr_sse = 0.0
        for i in range(n):
            tr_sse += e[i] * e[i]
        trace[it, 0] = tr_sse / n
        if nlab > 0:
            trace[it, 1] = sse_q / nlab
            trace[it, 2] = sse_c / nlab
        if verbose and (it % verbose == 0 or it == n_iter - 1):
            print(it, 'alpha', alpha, 'train mse', tr_sse / n, 'val mse avg', trace[it, 1], 'val mse clipped-avg', trace[it, 2])
    return acc / max(nacc, 1), acc_c / max(nacc, 1), trace


def make_bag(rows_ent, rows_feat, n_ent, n_feat, power=0.5):
    """CSC over bag features: for feature j the entities that carry it, coefficient |N(r)|^-power."""
    key = np.unique(rows_ent.astype(np.int64) * n_feat + rows_feat)
    r, j = key // n_feat, key % n_feat
    cnt = np.bincount(r, minlength=n_ent).astype(np.float64)
    c = cnt[r] ** -power
    o = np.argsort(j, kind='stable')
    ptr = np.searchsorted(j[o], np.arange(n_feat + 1))
    return ptr.astype(np.int64), r[o].astype(np.int64), c[o]
