"""Bayesian factorization machine v2: relational slots + bags with self-exclusion (Gibbs sampling, numba).

Row i has one entity per slot.  The slot vector of the row is
    a_s(i) = V_s[r] + sum_{p in bag(r)} c_p * Y[feat_p]  -  self_c[s, i] * Y[self_feat[s, i]]
i.e. the entity's full bag minus (for explicit, rating-weighted bag entries) the entry that was built from this very
row, so a training row never sees its own rating.  Prediction:
    yhat_i = w0 + sum_s w_s[r_s(i)] + sum_f sum_{s<s'} a_s,f(i) * a_s',f(i)
Bag features are organised in groups (contiguous feature ranges, one slot each); every group and every factor has its
own Normal mean / Gamma precision hyper-parameters, which are sampled.
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
def gibbs2(K, n_iter, burn, seed, y, ent, eoff, bag_ptr, bag_ent, bag_c, bag_row, goff, gslot, self_feat, self_c,
           q_ent, q_y, init_std, lo, hi, verbose):
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
    Y = np.random.randn(K, max(NB, 1)) * init_std
    a = V.copy()
    for j in range(NB):
        for p in range(bag_ptr[j], bag_ptr[j + 1]):
            for f in range(K):
                a[f, bag_ent[p]] += bag_c[p] * Y[f, j]
    w = np.zeros(E)
    w0 = 0.0
    for i in range(n):
        w0 += y[i]
    w0 /= n
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
        e[i] = y[i] - pred
    alpha = 1.0
    lam_w, mu_w = np.ones(S), np.zeros(S)
    lam_v, mu_v = np.ones((S, K)), np.zeros((S, K))
    lam_y, mu_y = np.ones((NG, K)), np.zeros((NG, K))
    A, G, da = np.zeros(E), np.zeros(E), np.zeros(E)
    obuf, sc = np.empty(n), np.zeros(n)
    acc, acc2 = np.zeros(nq), np.zeros(nq)
    trace = np.zeros((n_iter, 3))
    nacc = 0
    for it in range(n_iter):
        sse = 0.0
        for i in range(n):
            sse += e[i] * e[i]
        alpha = np.random.gamma((1.0 + n) / 2.0, 2.0 / (1.0 + sse))
        se = 0.0
        for i in range(n):
            se += e[i]
        prec = alpha * n + 1e-6
        new = alpha * (se + w0 * n) / prec + np.random.randn() / np.sqrt(prec)
        d = new - w0
        w0 = new
        for i in range(n):
            e[i] -= d
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
                    A[r] += o * o
                    G[r] += o * e[i]
                    if hs:
                        sc[i] = 0.0
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
                                s2_ += c * c * (A[r] - ot * ot)
                                s1 += c * (G[r] - ot * (e[t] - ot * da[r]))
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
                                G[r] -= c * d * (A[r] - ot * ot)
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
        if it >= burn:
            nacc += 1
        sse_q, nlab = 0.0, 0
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
            if it >= burn:
                acc[i] += pred
                acc2[i] += pred * pred
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
        trace[it, 2] = alpha
        if verbose and (it % verbose == 0 or it == n_iter - 1):
            print(it, 'alpha', alpha, 'train mse', tr_sse / n, 'val mse of running mean', trace[it, 1])
    m = acc / max(nacc, 1)
    sd = np.sqrt(np.maximum(acc2 / max(nacc, 1) - m * m, 0.0))
    return m, sd, trace


class Design:
    """Collects slots and bag groups and produces the arrays for gibbs2."""

    def __init__(self, n_rows, n_query):
        self.n, self.nq = n_rows, n_query
        self.ent, self.q_ent, self.eoff = [], [], [0]
        self.groups = []                       # (slot, n_feat, ent(local), feat, coef, row)
        self.self_feat, self.self_c = [], []

    def add_slot(self, rows_ent, query_ent, n_ent):
        self.ent.append(self.eoff[-1] + np.asarray(rows_ent, np.int64))
        self.q_ent.append(self.eoff[-1] + np.asarray(query_ent, np.int64))
        self.eoff.append(self.eoff[-1] + int(n_ent))
        self.self_feat.append(np.full(self.n, -1, np.int64)); self.self_c.append(np.zeros(self.n))
        return len(self.ent) - 1

    def add_bag(self, slot, ent, feat, coef, n_feat, row=None):
        """Entries (entity of the slot, feature, coefficient); row = index of the training row an entry was built from
        (it is then excluded from that row), or None."""
        self.groups.append((slot, int(n_feat), np.asarray(ent, np.int64), np.asarray(feat, np.int64), np.asarray(coef, np.float64),
                            None if row is None else np.asarray(row, np.int64)))

    def build(self):
        ptrs, ents, cs, rows, goff, gslot = [np.zeros(1, np.int64)], [], [], [], [0], []
        for slot, nf, ent, feat, coef, row in self.groups:
            o = np.argsort(feat, kind='stable')
            ptr = np.searchsorted(feat[o], np.arange(nf + 1))
            ptrs.append(ptr[1:] + ptrs[-1][-1]); ents.append(self.eoff[slot] + ent[o]); cs.append(coef[o])
            rows.append(np.full(len(o), -1, np.int64) if row is None else row[o])
            if row is not None:
                self.self_feat[slot][row] = goff[-1] + feat
                self.self_c[slot][row] = coef
            goff.append(goff[-1] + nf); gslot.append(slot)
        z = lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt)
        return dict(ent=np.ascontiguousarray(np.vstack(self.ent)), eoff=np.array(self.eoff, np.int64), bag_ptr=np.concatenate(ptrs),
                    bag_ent=z(ents, np.int64), bag_c=z(cs, np.float64), bag_row=z(rows, np.int64), goff=np.array(goff, np.int64),
                    gslot=np.array(gslot, np.int64), self_feat=np.ascontiguousarray(np.vstack(self.self_feat)),
                    self_c=np.ascontiguousarray(np.vstack(self.self_c)), q_ent=np.ascontiguousarray(np.vstack(self.q_ent)))


def run(design, y, q_y, K, n_iter, burn, seed, init_std=0.1, lo=1.0, hi=5.0, verbose=0):
    d = design.build()
    return gibbs2(K, n_iter, burn, seed, np.asarray(y, np.float64), d['ent'], d['eoff'], d['bag_ptr'], d['bag_ent'], d['bag_c'], d['bag_row'],
                  d['goff'], d['gslot'], d['self_feat'], d['self_c'], d['q_ent'], np.asarray(q_y, np.float64), init_std, lo, hi, verbose)
