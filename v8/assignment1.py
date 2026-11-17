"""CSE258 Assignment 1 (Goodreads) -- all six tasks in one file.

    python assignment1.py --data-dir assignment1_2026 --output-dir . [--tasks 1 2 3 4 5 6]

Layout
  common ............ CSV I/O, CSR helpers, numba kernels, spectral embedding, pair classifier
  latent models ..... BPR-trained latent factor models (MF, SVD++-style, FPMC-style) on popularity negatives
  EASE .............. closed-form item-item (and user-user) regression (Steck 2019), inverted in place with LAPACK
  Tasks 2 & 4 ....... "which item is the held-out positive": co-occurrence/spectral/EASE features + latent scores -> GBDT
  Task 5 ............ next item: same, plus sequence order and item popularity after the user's last event
  Task 6 ............ time to the final interaction: gap/item-time/co-reader-lag features -> absolute-error GBDT,
                      trained on last-event and cutoff pseudo-labels
  GPU models ........ PyTorch MLP stacker (Tasks 1/3), BERT4Rec (Task 4) and SASRec (Task 5) sequence models
  Tasks 1 & 3 ....... rating regression: base-model feature bank + EASE neighbourhoods, 18-fold out-of-fold,
                      (Ridge + residual GBDT) & MLP

Every model is trained only on the designated training CSV; test files are read for their queries
(user/item ids, timestamps, candidate pairs) and never for labels.
"""
import argparse
import calendar
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numba as nb
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

DAY = 86400.0
TEND = 1509522985.0          # end of the data collection (2017-11-01); used for "time left" features
SEED = 258


# =============================================================================================
# Common: data loading and writing
# =============================================================================================
class Data:
    """One training file plus its test files, sharing a single user/item id map.
    Unseen test users/items get fresh ids, so they simply have zero training counts."""

    def __init__(self, data_dir, train_name, test_names):
        self.users, self.items = {}, {}
        with open(data_dir / train_name, newline='') as f:
            rows = list(csv.reader(f))[1:]
        self.u = np.array([self.uid(r[0]) for r in rows], np.int64)
        self.i = np.array([self.iid(r[1]) for r in rows], np.int64)
        self.t = np.array([float(r[2]) for r in rows])
        self.r = np.array([float(r[3]) for r in rows])
        self.tests = {}
        for name in test_names:
            with open(data_dir / name, newline='') as f:
                rd = csv.reader(f)
                header = next(rd)
                rows = list(rd)
            cols = {}
            for j, h in enumerate(header[:-1]):          # last column is the one to predict
                vals = [r[j] for r in rows]
                if h == 'user':
                    cols[h] = np.array([self.uid(v) for v in vals], np.int64)
                elif h.startswith('item'):
                    cols[h] = np.array([self.iid(v) for v in vals], np.int64)
                else:
                    cols[h] = np.array([float(v) for v in vals])
            self.tests[name] = cols
        self.nU, self.nI = len(self.users), len(self.items)
        log(f'{train_name}: {len(self.u):,} rows, {self.nU:,} users, {self.nI:,} items')

    def uid(self, x):
        return self.users.setdefault(x, len(self.users))

    def iid(self, x):
        return self.items.setdefault(x, len(self.items))


def save_predictions(data_dir, out_dir, task, values):
    name = f'task_{task}_test.csv'
    with open(data_dir / name, newline='') as src, open(out_dir / f'task_{task}_predictions.csv', 'w', newline='') as dst:
        rd, wr = csv.reader(src), csv.writer(dst, lineterminator='\n')
        wr.writerow(next(rd))
        n = 0
        for row, v in zip(rd, values, strict=True):
            wr.writerow(row + [v])
            n += 1
    log(f'wrote task_{task}_predictions.csv ({n:,} rows)')


def test_pairs(data, name):
    te = data.tests[name]
    return te['user'], np.column_stack([te['item1'], te['item2']]), te


def pick(margins, data, name):
    """Turn margins (>0 means item1) into the item-id strings expected in the 'which' column."""
    inv = {v: k for k, v in data.items.items()}
    te = data.tests[name]
    return [inv[a] if m > 0 else inv[b] for m, a, b in zip(margins, te['item1'], te['item2'])]


# =============================================================================================
# Common: graph structures and numba kernels
# =============================================================================================
class Graph:
    """User-major (time ordered) and item-major views of a set of interactions."""

    def __init__(self, u, i, t, r, nU, nI):
        o = np.lexsort((t, u))
        self.u, self.i, self.t, self.r = u[o], i[o], t[o].astype(np.float64), r[o].astype(np.float64)
        self.nU, self.nI = nU, nI
        self.uptr = np.searchsorted(self.u, np.arange(nU + 1))
        L = np.diff(self.uptr)
        self.upos = np.arange(len(self.u)) - np.repeat(self.uptr[:-1], L)       # position inside user history
        oi = np.lexsort((self.t, self.i))
        self.iptr = np.searchsorted(self.i[oi], np.arange(nI + 1))
        self.iusers, self.itimes, self.ipos = self.u[oi], self.t[oi], self.upos[oi]
        self.ucnt = L.astype(np.float64)
        self.icnt = np.diff(self.iptr).astype(np.float64)
        self.gtimes = np.sort(self.t)

    def last_time(self, users):
        return self.t[np.maximum(self.uptr[users + 1] - 1, 0)]

    def seen(self, qu, qc):
        """1 where the candidate already is in the user's history (then it cannot be the held-out positive)."""
        out = np.zeros(qc.shape, np.float32)
        for k, us in enumerate(qu):
            h = self.i[self.uptr[us]:self.uptr[us + 1]]
            out[k] = np.isin(qc[k], h)
        return out


@nb.njit(cache=True)
def window_counts(items, T, iptr, itimes, gtimes, lo, hi):
    """Interactions of each item (and of all items) inside [T+lo, T+hi]."""
    n = len(items)
    c = np.zeros(n)
    g = np.zeros(n)
    for k in range(n):
        a, b = iptr[items[k]], iptr[items[k] + 1]
        c[k] = np.searchsorted(itimes[a:b], T[k] + hi) - np.searchsorted(itimes[a:b], T[k] + lo)
        g[k] = np.searchsorted(gtimes, T[k] + hi) - np.searchsorted(gtimes, T[k] + lo)
    return c, g


@nb.njit(cache=True)
def item_time_stats(items, T, iptr, itimes):
    """First / median / last / 10th-pct item time relative to T (days) and share of the item's events after T."""
    n = len(items)
    out = np.full((n, 5), np.nan)
    for k in range(n):
        a, b = iptr[items[k]], iptr[items[k] + 1]
        m = b - a
        if m == 0:
            continue
        arr = itimes[a:b]
        out[k, 0] = (arr[0] - T[k]) / 86400.0
        out[k, 1] = (arr[m // 2] - T[k]) / 86400.0
        out[k, 2] = (arr[m - 1] - T[k]) / 86400.0
        out[k, 3] = (m - np.searchsorted(arr, T[k])) / m
        out[k, 4] = (arr[int(0.1 * m)] - T[k]) / 86400.0
    return out


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def time_window_features(g, items, T, windows):
    """Log-lift of an item's popularity inside time windows around T versus its overall popularity."""
    cnt, N = g.icnt[items], float(len(g.u))
    base = np.log((cnt + 1) / N)
    F = []
    for lo, hi in windows:
        c, tot = window_counts(items, T, g.iptr, g.itimes, g.gtimes, lo * DAY, hi * DAY)
        F += [np.log((c + (cnt + 1) / N * 100) / (tot + 100)) - base, np.log1p(c)]
    ts = item_time_stats(items, T, g.iptr, g.itimes)
    F += [ts[:, 3]] + [slog(ts[:, j]) for j in (0, 1, 2, 4)]
    return np.column_stack(F)


@nb.njit(cache=True)
def cooc_features(mode, qu, qc, qt, uptr, uitems, utimes, urats, iptr, iusers, itimes, ipos, ucnt, icnt):
    """Item co-occurrence evidence for each candidate, aggregated over the user's history.

    A neighbour v shares a history item j with user u and also interacted with candidate c; it votes
    1/(|H(v)|+2).  Votes are normalised by the popularity of j and c (negatives are sampled by popularity,
    so the model needs lift, not raw co-counts).  The mode decides the extra weighting:
      2: no query time (Task 2)    4: known query time qt (Task 4)    5: ordered history, next item (Task 5)
    Returns (n, C, 16)."""
    n, C = qc.shape
    nU = len(ucnt)
    vote = np.zeros((C, nU))
    vtime = np.zeros((C, nU))
    vpos = np.zeros((C, nU), np.int64)
    out = np.zeros((n, C, 16))
    for r in range(n):
        u = qu[r]
        a, b = uptr[u], uptr[u + 1]
        L = b - a
        for c in range(C):
            it = qc[r, c]
            for p in range(iptr[it], iptr[it + 1]):
                v = iusers[p]
                vote[c, v] = 1.0 / (ucnt[v] + 2.0)
                vtime[c, v] = itimes[p]
                vpos[c, v] = ipos[p]
        acc = np.zeros((C, 13))
        for p in range(a, b):
            j = uitems[p]
            inv_cj = 1.0 / (icnt[j] + 5.0)
            inv_sq = 1.0 / np.sqrt(icnt[j] + 5.0)
            w_r = 0.6 + 0.12 * urats[p]
            u_dir = 1.0
            if mode == 5:                       # recency by position from the end of the history
                back = b - 1 - p
                w_a = 1.0 / (back + 1.0)
                w_b = 1.0 / np.sqrt(back + 1.0)
                w_c = 1.0 if back < 3 else 0.0
            elif mode == 4:                     # closeness to the query time
                dt = abs(qt[r] - utimes[p])
                w_a = np.exp(-dt / (2 * 86400.0))
                w_b = np.exp(-dt / (14 * 86400.0))
                w_c = np.exp(-dt / (180 * 86400.0))
                u_dir = 1.0 if qt[r] >= utimes[p] else -1.0
            else:
                w_a = 1.0
                w_b = 1.0
                w_c = 1.0
            for c in range(C):
                it = qc[r, c]
                if it == j:
                    continue
                w = 0.0
                x1 = 0.0
                x2 = 0.0
                x3 = 0.0
                x4 = 0.0
                for q in range(iptr[j], iptr[j + 1]):
                    v = iusers[q]
                    vw = vote[c, v]
                    if vw <= 0.0 or v == u:
                        continue
                    w += vw
                    if mode == 5:               # did v take the candidate soon AFTER the shared item?
                        dp = vpos[c, v] - ipos[q]
                        if dp > 0:
                            x1 += vw / (dp + 1.5)
                            if dp <= 2:
                                x2 += vw / dp
                        elif dp == -1:
                            x4 += vw
                        if abs(dp) <= 3:
                            x3 += vw
                    else:                       # did v take both items close in time?
                        dtv = abs(vtime[c, v] - itimes[q])
                        if dtv <= 7 * 86400.0:
                            x1 += vw
                        if dtv <= 30 * 86400.0:
                            x2 += vw
                        if dtv <= 180 * 86400.0:
                            x3 += vw
                        if mode == 4:
                            vdir = 1.0 if vtime[c, v] >= itimes[q] else -1.0
                            if vdir == u_dir and dtv <= 90 * 86400.0:
                                x4 += vw
                        elif abs(vtime[c, v] - utimes[p]) <= 90 * 86400.0:
                            x4 += vw
                if w > 0.0:
                    v1 = w * inv_cj
                    ci = icnt[it]
                    acc[c, 0] += 1.0
                    acc[c, 1] += v1
                    acc[c, 2] += w * inv_sq
                    acc[c, 3] += v1 * w_r
                    acc[c, 4] += w_a * v1
                    acc[c, 5] += w_b * v1
                    acc[c, 6] += w_c * v1
                    acc[c, 7] = max(acc[c, 7], v1 / (ci + 2.0))
                    cs = w * inv_sq / np.sqrt(ci + 5.0)
                    acc[c, 8] += cs * cs
                    acc[c, 9] += x1 * inv_cj * (w_b if mode == 5 else 1.0)
                    acc[c, 10] += x2 * inv_cj * (w_a if mode == 5 else 1.0)
                    acc[c, 11] += x3 * inv_cj * (w_a if mode == 5 else 1.0)
                    acc[c, 12] += x4 * inv_cj * (w_a if mode == 5 else (w_b if mode == 4 else 1.0))
        for c in range(C):
            it = qc[r, c]
            for p in range(iptr[it], iptr[it + 1]):
                vote[c, iusers[p]] = 0.0
            ci = icnt[it]
            o = out[r, c]
            o[0] = np.log((acc[c, 1] + 1e-5) / (ci + 0.5))
            o[1] = np.log((acc[c, 1] + 1e-4) / (ci + 5.0))
            o[2] = np.log((acc[c, 2] + 1e-4) / (ci + 2.0))
            o[3] = np.log((acc[c, 3] + 1e-5) / (ci + 1.0))
            o[4] = np.log((acc[c, 4] + 1e-6) / (ci + 1.0))
            o[5] = np.log((acc[c, 5] + 1e-6) / (ci + 1.0))
            o[6] = np.log((acc[c, 6] + 3e-6) / (ci + 1.0))
            o[7] = np.log(acc[c, 7] + 1e-6)
            o[8] = acc[c, 8]
            for k in range(4):
                o[9 + k] = np.log((acc[c, 9 + k] + 1e-6) / (ci + 1.0))
            o[13] = acc[c, 0] / max(L, 1)
            o[14] = np.log(ci + 1.0)
            o[15] = 1.0 if ci == 0 else 0.0
    return out


def cooc(g, mode, qu, qc, qt):
    return cooc_features(mode, qu, qc, qt.astype(np.float64), g.uptr, g.i, g.t, g.r, g.iptr, g.iusers,
                         g.itimes, g.ipos, g.ucnt, g.icnt)


# ---------------------------------------------------------------------------------------------
# Spectral (truncated SVD) embedding of the normalised user-item matrix
# ---------------------------------------------------------------------------------------------
def unit_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > 1e-7, x / np.maximum(n, 1e-7), 0.0)


class Spectral:
    def __init__(self, g, k=64, recency=False):
        inv_u, inv_i = 1 / np.sqrt(g.ucnt + 5.0), 1 / np.sqrt(g.icnt + 10.0)
        w = inv_u[g.u] * inv_i[g.i]
        if recency:                                    # later positions count more (Task 5)
            w = w * (0.5 + 0.5 * (g.upos + 1.0) / g.ucnt[g.u])
        X = sp.csr_matrix((w, (g.u, g.i)), shape=(g.nU, g.nI))
        U, s, Vt = svds(X, k=k, v0=np.full(min(X.shape), 1 / np.sqrt(min(X.shape))), random_state=SEED)
        o = np.argsort(-s)
        U, s, Vt = U[:, o], s[o], Vt[o]
        self.u_vec = (U * s).astype(np.float32)                         # user coordinates
        self.i_vec = (Vt.T * inv_i[:, None]).astype(np.float32)         # popularity-normalised item coordinates
        self.u_unit = unit_rows(U * np.sqrt(s)).astype(np.float32)
        cent = np.zeros((g.nI, k))
        np.add.at(cent, g.i, self.u_unit[g.u])
        self.i_unit = unit_rows(cent).astype(np.float32)               # item = centroid of its users
        self.v_item = (Vt.T * np.sqrt(s) * inv_i[:, None]).astype(np.float32)

    def features(self, g, qu, items):
        lift = np.einsum('ij,ij->i', self.u_vec[qu], self.i_vec[items])
        comm = np.einsum('ij,ij->i', self.u_unit[qu], self.i_unit[items])
        pooled = spectral_pool(qu, items, g.uptr, g.i, self.i_unit, self.v_item)
        return np.column_stack([lift, comm, pooled])


@nb.njit(cache=True)
def spectral_pool(qu, items, uptr, uitems, i_unit, v_item):
    """Similarity of the candidate to each history item, pooled (sum, sum^2, sum^3, max, recency), plus a
    decayed profile of the last 15 items and the very last item."""
    n = len(qu)
    k = i_unit.shape[1]
    out = np.zeros((n, 7))
    for r in range(n):
        a, b = uptr[qu[r]], uptr[qu[r] + 1]
        if b == a:
            continue
        c = items[r]
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        mx = -1.0
        rec = 0.0
        for p in range(a, b):
            j = uitems[p]
            if j == c:
                continue
            d = 0.0
            for f in range(k):
                d += i_unit[j, f] * i_unit[c, f]
            s1 += d
            mx = max(mx, d)
            if d > 0:
                s2 += d * d
                s3 += d * d * d
                rec += d * d / (b - 1 - p + 2.0)
        prof = np.zeros(k)
        wsum = 0.0
        for p in range(max(a, b - 15), b):
            w = 0.75 ** (b - 1 - p)
            prof += w * v_item[uitems[p]]
            wsum += w
        dd = 0.0
        dl = 0.0
        for f in range(k):
            dd += prof[f] / wsum * v_item[c, f]
            dl += v_item[uitems[b - 1], f] * v_item[c, f]
        sq = np.sqrt(b - a)
        out[r, 0] = s1 / sq
        out[r, 1] = s2 / sq
        out[r, 2] = s3 / sq
        out[r, 3] = mx
        out[r, 4] = rec
        out[r, 5] = dd
        out[r, 6] = dl
    return out


# ---------------------------------------------------------------------------------------------
# Pairwise classifier shared by Tasks 2, 4, 5
# ---------------------------------------------------------------------------------------------
def pair_rows(Fa, Fb, ctx):
    """Antisymmetric part (Fa-Fb) and symmetric part (|Fa-Fb|, Fa+Fb, context)."""
    d = np.nan_to_num(Fa - Fb)
    s = np.nan_to_num(np.hstack([np.abs(Fa - Fb), Fa + Fb, ctx]))
    return d, s


PAIR_CLF = {'small': dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0),
            'large': dict(learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=200, l2_regularization=30.0)}


def fit_pair_classifier(sets, max_iter, size='small'):
    """sets: list of (F (n, 1+k, f), ctx); column 0 is the positive. Both orientations are used."""
    X, Y = [], []
    for F, ctx in sets:
        for c in range(1, F.shape[1]):
            d, s = pair_rows(F[:, 0], F[:, c], ctx)
            X += [np.hstack([d, s]), np.hstack([-d, s])]
            Y += [np.ones(len(d)), np.zeros(len(d))]
    clf = HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=False, random_state=SEED, **PAIR_CLF[size])
    return clf.fit(np.vstack(X), np.concatenate(Y))


def pair_margin(clf, Fa, Fb, ctx):
    """> 0 means item a is the positive.  Exactly antisymmetric under swapping a and b."""
    d, s = pair_rows(Fa, Fb, ctx)
    return 0.5 * (clf.decision_function(np.hstack([d, s])) - clf.decision_function(np.hstack([-d, s])))


def sample_negatives(pool, positives, k, rng):
    """Popularity-proportional negatives (an item drawn from a random interaction), never equal to the positive.
    This matches the test files: candidate popularity follows the interaction distribution exactly."""
    neg = pool[rng.integers(0, len(pool), (len(positives), k))]
    bad = neg == positives[:, None]
    while bad.any():
        neg[bad] = pool[rng.integers(0, len(pool), bad.sum())]
        bad = neg == positives[:, None]
    return neg


# =============================================================================================
# Latent factor models trained with BPR on popularity-sampled negatives (Tasks 2, 4, 5)
#
# Test negatives are items of random interactions, so every model ranks a training interaction above an item drawn
# the same way: loss -log sigmoid(s(u,i) - s(u,j)), j ~ popularity.  Their scores become extra stacking features.
#   mf    s = b_i [+ b_i(month)] + p_u . q_i
#   hist  s = b_i [+ b_i(month)] + q_i . (p_u [+ p_u(year)] + |H(u)|^-1/2 sum_{j in H(u)} y_j)      (SVD++ style)
#   ctx   s = b_i [+ b_i(month)] + p_u . q_i + c_i . weighted mean of a_l over the items next to the query in the
#         user's timeline (before and after for Task 4, the last items for Task 5)                 (FPMC style)
# =============================================================================================
def month_bin(t):
    """Monthly bins from 2005-01 (0 = earlier, 161 = later)."""
    return np.clip(np.floor((np.asarray(t, float) - 1104537600.0) / (30.44 * DAY)).astype(np.int64) + 1, 0, 161)


def year_bin(t):
    return np.clip(((np.asarray(t, float) - 946684800.0) // (365.25 * DAY)).astype(np.int64), -1, 18) + 1


@nb.njit(cache=True, fastmath=True)
def _bpr_mf_epoch(us, its, tb, order, negpool, P, Q, B, BT, lr, reg, regb, use_t, seed):
    np.random.seed(seed)
    k = P.shape[1]
    for t in range(len(order)):
        n = order[t]
        u, i = us[n], its[n]
        j = negpool[np.random.randint(0, len(negpool))]
        while j == i:
            j = negpool[np.random.randint(0, len(negpool))]
        x = B[i] - B[j]
        if use_t:
            x += BT[i, tb[n]] - BT[j, tb[n]]
        for f in range(k):
            x += P[u, f] * (Q[i, f] - Q[j, f])
        g = 0.0 if x > 30 else 1.0 / (1.0 + np.exp(x))
        for f in range(k):
            pu, qi, qj = P[u, f], Q[i, f], Q[j, f]
            P[u, f] += lr * (g * (qi - qj) - reg * pu)
            Q[i, f] += lr * (g * pu - reg * qi)
            Q[j, f] += lr * (-g * pu - reg * qj)
        B[i] += lr * (g - regb * B[i])
        B[j] += lr * (-g - regb * B[j])
        if use_t:
            c = tb[n]
            BT[i, c] += lr * (g - reg * BT[i, c])
            BT[j, c] += lr * (-g - reg * BT[j, c])


@nb.njit(cache=True, fastmath=True)
def _bpr_hist_epoch(uorder, uptr, uitems, tb, yr, negpool, P, PY, Q, Y, B, BT, lr, reg, regb, use_t, use_py, seed):
    """User blocks: the history sum h_u is computed once per user; the target item is removed from it."""
    np.random.seed(seed)
    k = Q.shape[1]
    h = np.zeros(k)
    gh = np.zeros(k)
    r = np.zeros(k)
    for uu in range(len(uorder)):
        u = uorder[uu]
        a, b = uptr[u], uptr[u + 1]
        if b == a:
            continue
        norm = max(b - a - 1, 1) ** -0.5
        h[:] = 0.0
        gh[:] = 0.0
        for p in range(a, b):
            h += Y[uitems[p]]
        idx = np.arange(a, b)
        np.random.shuffle(idx)
        for p in idx:
            i = uitems[p]
            j = negpool[np.random.randint(0, len(negpool))]
            while j == i:
                j = negpool[np.random.randint(0, len(negpool))]
            for f in range(k):
                r[f] = P[u, f] + norm * (h[f] - Y[i, f]) + (PY[yr[p], f] if use_py else 0.0)
            x = B[i] - B[j]
            if use_t:
                x += BT[i, tb[p]] - BT[j, tb[p]]
            for f in range(k):
                x += r[f] * (Q[i, f] - Q[j, f])
            g = 0.0 if x > 30 else 1.0 / (1.0 + np.exp(x))
            for f in range(k):
                qi, qj = Q[i, f], Q[j, f]
                d = g * (qi - qj)
                Q[i, f] += lr * (g * r[f] - reg * qi)
                Q[j, f] += lr * (-g * r[f] - reg * qj)
                P[u, f] += lr * (d - reg * P[u, f])
                if use_py:
                    PY[yr[p], f] += lr * (d - reg * PY[yr[p], f])
                gh[f] += d
            B[i] += lr * (g - regb * B[i])
            B[j] += lr * (-g - regb * B[j])
            if use_t:
                c = tb[p]
                BT[i, c] += lr * (g - reg * BT[i, c])
                BT[j, c] += lr * (-g - reg * BT[j, c])
        for p in range(a, b):
            jj = uitems[p]
            for f in range(k):
                Y[jj, f] += lr * (norm * gh[f] / (b - a) - reg * Y[jj, f])


@nb.njit(cache=True, fastmath=True)
def _bpr_ctx_epoch(us, its, tb, ctx, cw, order, negpool, P, Q, A, C, B, BT, lr, reg, regb, seed):
    np.random.seed(seed)
    k = Q.shape[1]
    z = np.zeros(k)
    for t in range(len(order)):
        n = order[t]
        u, i = us[n], its[n]
        j = negpool[np.random.randint(0, len(negpool))]
        while j == i:
            j = negpool[np.random.randint(0, len(negpool))]
        z[:] = 0.0
        ws = 0.0
        for c in range(ctx.shape[1]):
            if ctx[n, c] >= 0:
                ws += cw[n, c]
                z += cw[n, c] * A[ctx[n, c]]
        if ws > 0:
            z /= ws
        x = B[i] - B[j] + BT[i, tb[n]] - BT[j, tb[n]]
        for f in range(k):
            x += (C[i, f] - C[j, f]) * z[f] + P[u, f] * (Q[i, f] - Q[j, f])
        g = 0.0 if x > 30 else 1.0 / (1.0 + np.exp(x))
        for f in range(k):
            ci, cj = C[i, f], C[j, f]
            C[i, f] += lr * (g * z[f] - reg * ci)
            C[j, f] += lr * (-g * z[f] - reg * cj)
            if ws > 0:
                for c in range(ctx.shape[1]):
                    l = ctx[n, c]
                    if l >= 0:
                        A[l, f] += lr * (g * (ci - cj) * cw[n, c] / ws - reg * A[l, f])
            pu, qi, qj = P[u, f], Q[i, f], Q[j, f]
            P[u, f] += lr * (g * (qi - qj) - reg * pu)
            Q[i, f] += lr * (g * pu - reg * qi)
            Q[j, f] += lr * (-g * pu - reg * qj)
        B[i] += lr * (g - regb * B[i])
        B[j] += lr * (-g - regb * B[j])
        c0 = tb[n]
        BT[i, c0] += lr * (g - reg * BT[i, c0])
        BT[j, c0] += lr * (-g - reg * BT[j, c0])


CTX_OFFSETS = {4: ([-1, 1, -2, 2], [1.0, 1.0, 0.5, 0.5]), 5: ([-1, -2, -3, -4], [1.0, 0.5, 0.33, 0.25])}


def timeline_context(u, i, mode):
    """For each interaction (rows sorted by user, time): its neighbours in the user's timeline."""
    offs, ws = CTX_OFFSETS[mode]
    ctx = np.full((len(u), 4), -1, np.int64)
    cw = np.zeros((len(u), 4))
    for c, (off, w) in enumerate(zip(offs, ws)):
        src = np.arange(len(u)) + off
        ok = (src >= 0) & (src < len(u))
        ok[ok] &= u[src[ok]] == u[ok]
        ctx[ok, c] = i[src[ok]]
        cw[ok, c] = w
    return ctx, cw


def query_context(g, qu, qt, mode):
    """Task 4: the history items nearest in time before/after qt.  Task 5: the user's last items."""
    ctx = np.full((len(qu), 4), -1, np.int64)
    cw = np.zeros((len(qu), 4))
    for r, us in enumerate(qu):
        a, b = g.uptr[us], g.uptr[us + 1]
        if mode == 4:
            p = a + np.searchsorted(g.t[a:b], qt[r])
            cand = [(p - 1, 1.0), (p, 1.0), (p - 2, 0.5), (p + 1, 0.5)]
        else:
            cand = [(b - 1, 1.0), (b - 2, 0.5), (b - 3, 0.33), (b - 4, 0.25)]
        for c, (pp, w) in enumerate(cand):
            if a <= pp < b:
                ctx[r, c] = g.i[pp]
                cw[r, c] = w
    return ctx, cw


def fit_latent(kind, g, negpool, k=128, epochs=60, lr=0.05, reg=0.01, regb=0.001, time_bias=False, user_year=False,
               ctx_mode=4, seed=0):
    """Train one latent model on the interactions of graph g; returns a scoring function score(qu, items, qt)."""
    rng = np.random.default_rng(seed)
    nU, nI = g.nU, g.nI
    P, Q, B = rng.standard_normal((nU, k)) * 0.01, rng.standard_normal((nI, k)) * 0.01, np.zeros(nI)
    BT = np.zeros((nI, 162))
    if kind == 'ctx':                       # Task 5 knows only the time of the PREVIOUS event
        t_used = np.where(np.r_[True, g.u[1:] != g.u[:-1]], g.t, np.r_[g.t[0], g.t[:-1]]) if ctx_mode == 5 else g.t
    else:
        t_used = g.t
    tb = month_bin(t_used)
    if kind == 'mf':
        for ep in range(epochs):
            _bpr_mf_epoch(g.u, g.i, tb, rng.permutation(len(g.u)), negpool, P, Q, B, BT, lr, reg, regb, time_bias, seed * 1000 + ep)
    elif kind == 'hist':
        Y = rng.standard_normal((nI, k)) * 0.01
        if user_year:
            keys, yr = np.unique(g.u * 20 + year_bin(g.t), return_inverse=True)
        else:
            keys, yr = np.zeros(1, np.int64), np.zeros(len(g.u), np.int64)
        PY = np.zeros((len(keys), k))
        for ep in range(epochs):
            _bpr_hist_epoch(rng.permutation(nU), g.uptr, g.i, tb, yr, negpool, P, PY, Q, Y, B, BT, lr, reg, regb,
                            time_bias, user_year, seed * 1000 + ep)
    else:
        A, C = rng.standard_normal((nI, k)) * 0.01, rng.standard_normal((nI, k)) * 0.01
        ctx, cw = timeline_context(g.u, g.i, ctx_mode)
        for ep in range(epochs):
            _bpr_ctx_epoch(g.u, g.i, tb, ctx, cw, rng.permutation(len(g.u)), negpool, P, Q, A, C, B, BT, lr, reg, regb, seed * 1000 + ep)

    def score(qu, items, qt):
        r = P[qu].copy()
        if kind == 'hist':
            L = g.ucnt[qu]
            H = np.stack([Y[g.i[g.uptr[x]:g.uptr[x + 1]]].sum(0) for x in qu])
            r += H * np.maximum(L, 1)[:, None] ** -0.5           # query items are never in the history
            if user_year:
                key = qu * 20 + year_bin(qt)
                pos = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
                r += np.where((keys[pos] == key)[:, None], PY[pos], 0.0)
        s = B[items] + np.einsum('nk,nk->n', r, Q[items])
        if kind == 'ctx':
            ctx_q, cw_q = query_context(g, qu, qt, ctx_mode)
            Z = np.einsum('nc,nck->nk', cw_q, np.where(ctx_q[..., None] >= 0, A[np.maximum(ctx_q, 0)], 0.0))
            s += np.einsum('nk,nk->n', Z / np.maximum(cw_q.sum(1, keepdims=True), 1e-9), C[items])
        if time_bias or kind == 'ctx':
            s += BT[items, month_bin(qt)]
        return s
    return score


LATENT_SPECS = {
    2: [('mf', {}), ('hist', {})],
    4: [('mf', dict(time_bias=True)), ('hist', dict(time_bias=True, user_year=True)), ('ctx', dict(k=64, epochs=40, ctx_mode=4))],
    5: [('mf', {}), ('ctx', dict(k=64, epochs=40, ctx_mode=5))],
}


def _latent_job(args):
    """Worker: train one latent model on (u, i, t) and score candidate columns; run in a process pool."""
    kind, opts, (u, i, t, nU, nI), qu, qc, qt, seed = args
    g = Graph(u, i, t, np.zeros(len(u)), nU, nI)
    score = fit_latent(kind, g, g.i, seed=seed, **opts)
    return np.column_stack([score(qu, qc[:, c], qt) for c in range(qc.shape[1])]).astype(np.float32)


def latent_scores(task, problems, workers=6):
    """problems: list of (g, qu, qc, qt).  Returns, per problem, an (n, C, n_models) array of model scores."""
    jobs = [(kind, opts, (g.u, g.i, g.t, g.nU, g.nI), qu, qc, qt, 17 * p + m)
            for p, (g, qu, qc, qt) in enumerate(problems) for m, (kind, opts) in enumerate(LATENT_SPECS[task])]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context('spawn')) as ex:
        out = list(ex.map(_latent_job, jobs))
    M = len(LATENT_SPECS[task])
    return [np.stack(out[p * M:(p + 1) * M], -1) for p in range(len(problems))]


# =============================================================================================
# GPU models (PyTorch): an MLP stacker for Tasks 1/3, BERT4Rec for Task 4 and SASRec for Task 5
#
# Determinism: cuBLAS workspace is fixed and deterministic kernels are requested, so reruns reproduce the outputs on the
# same machine.  torch is imported lazily, so the CPU-only tasks do not need it.
# =============================================================================================
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def torch_setup():
    import torch
    try:                                    # newer torch routes some ops to Triton JIT kernels; use the plain CUDA ones
        from torch._native import registry
        registry.deregister_op_overrides(disable_dsl_names='triton')
    except Exception:
        pass
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    return torch


def mlp_stacker(X, y, test_blocks, seeds=3, epochs=12, batch=4096):
    """Two-hidden-layer MLP (2048-512, GELU, dropout 0.3) on standardised features; predictions averaged over seeds."""
    torch = torch_setup()
    import torch.nn as nn
    dev = 'cuda'
    mu = np.zeros(X.shape[1]); sq = np.zeros(X.shape[1])
    for a in range(0, len(X), 200000):
        Z = X[a:a + 200000].astype(np.float64)
        mu += Z.sum(0); sq += (Z * Z).sum(0)
    mu /= len(X)
    sd = np.sqrt(np.maximum(sq / len(X) - mu ** 2, 0)) + 1e-6
    m_t, s_t = torch.tensor(mu, dtype=torch.float32, device=dev), torch.tensor(sd, dtype=torch.float32, device=dev)

    def to_gpu(Z):
        out = torch.empty(Z.shape, dtype=torch.float32, device=dev)
        for a in range(0, len(Z), 200000):
            out[a:a + 200000] = torch.clamp(torch.nan_to_num((torch.tensor(Z[a:a + 200000], device=dev) - m_t) / s_t), -8, 8)
        return out
    Xg, yg = to_gpu(X), torch.tensor(y, dtype=torch.float32, device=dev)
    tests = [to_gpu(T) for T in test_blocks]
    preds = [np.zeros(len(T)) for T in test_blocks]
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], 2048), nn.GELU(), nn.Dropout(0.3), nn.Linear(2048, 512), nn.GELU(),
                            nn.Dropout(0.3), nn.Linear(512, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = epochs * ((len(y) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
        gen = torch.Generator(device=dev).manual_seed(seed)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(y), device=dev, generator=gen)
            for b in range(0, len(y), batch):
                idx = perm[b:b + batch]
                loss = ((net(Xg[idx])[:, 0] + 4.0 - yg[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            torch.cuda.synchronize()
        net.eval()
        with torch.no_grad():
            for k, T in enumerate(tests):
                preds[k] += torch.cat([net(T[a:a + 65536])[:, 0] + 4.0 for a in range(0, len(T), 65536)]).cpu().numpy() / seeds
    del Xg, tests
    torch.cuda.empty_cache()
    return preds


def sasrec_scores(g, users, cands, seed, d=64, L=100, blocks=2, drop=0.3, epochs=20, lr=1e-3, nneg=4):
    """SASRec (causal self-attention over the last L items) trained with popularity negatives; scores candidates
    for the item following each user's history in graph g."""
    torch = torch_setup()
    import torch.nn as nn
    import torch.nn.functional as Fn
    dev = 'cuda'
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    nU, nI = g.nU, g.nI
    seq = np.zeros((nU, L), np.int64)
    for u in range(nU):
        it = g.i[g.uptr[u]:g.uptr[u + 1]][-L:] + 1          # 0 = padding
        if len(it):
            seq[u, -len(it):] = it

    class SASRec(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(nI + 1, d, padding_idx=0)
            self.pos = nn.Embedding(L, d)
            self.bias = nn.Embedding(nI + 1, 1)
            layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=2 * d, dropout=drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)
            nn.init.normal_(self.emb.weight, std=0.02)
            nn.init.zeros_(self.bias.weight)

        def forward(self, s):
            x = self.drop(self.emb(s) * d ** 0.5 + self.pos.weight[None])
            causal = torch.triu(torch.full((L, L), float('-inf'), device=s.device), 1)
            return self.ln(self.enc(x, mask=causal, src_key_padding_mask=(s == 0)))
    S = torch.tensor(seq, device=dev)
    pool = torch.tensor(g.i + 1, device=dev)
    m = SASRec().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.98))
    gen = torch.Generator(device=dev).manual_seed(seed)
    train_users = np.flatnonzero((seq > 0).sum(1) >= 2)
    for ep in range(epochs):
        p = rs.permutation(train_users)
        for b in range(0, len(p), 256):
            s = S[torch.tensor(p[b:b + 256], device=dev)]
            inp = torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], 1)
            h = m(inp)
            valid = (s > 0) & (inp > 0)
            neg = pool[torch.randint(0, len(pool), (*s.shape, nneg), device=dev, generator=gen)]
            ps = (h * m.emb(s)).sum(-1) + m.bias(s)[..., 0]
            ns = (h[:, :, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]
            loss = (Fn.softplus(-(ps[..., None] - ns)) * valid[..., None]).sum() / (valid.sum() * nneg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
    m.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(users), 1024):
            h = m(S[torch.tensor(users[b:b + 1024], device=dev)])[:, -1]
            c = torch.tensor(cands[b:b + 1024] + 1, device=dev)
            out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    del S, m
    torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float32)


def bert4rec_scores(g, qu, qt, qc, seed, d=128, L=100, blocks=2, drop=0.2, epochs=120, lr=1e-3, nneg=4, pmask=0.2):
    """Task 4 as fill-in-the-blank: a bidirectional transformer trained to recover masked items of users' time-ordered
    histories (popularity negatives); a [MASK] token inserted at the query time is scored against the candidates."""
    torch = torch_setup()
    import torch.nn as nn
    import torch.nn.functional as Fn
    dev = 'cuda'
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    nI, PAD, MASK = g.nI, 0, g.nI + 1
    starts, lens = g.uptr[:-1], np.diff(g.uptr)

    class B4R(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(nI + 2, d, padding_idx=0)
            self.pos = nn.Embedding(L, d)
            self.bias = nn.Embedding(nI + 2, 1)
            layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=4 * d, dropout=drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)
            nn.init.normal_(self.emb.weight, std=0.02)
            nn.init.zeros_(self.bias.weight)

        def forward(self, s):
            return self.ln(self.enc(self.drop(self.emb(s) * d ** 0.5 + self.pos.weight[None]), src_key_padding_mask=(s == PAD)))
    m = B4R().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.98))
    gen = torch.Generator(device=dev).manual_seed(seed)
    pool = torch.tensor(g.i + 1, device=dev)
    users = np.flatnonzero(lens >= 2)
    for ep in range(epochs):
        p = rs.permutation(users)
        for b in range(0, len(p), 256):
            ub = p[b:b + 256]
            S = np.zeros((len(ub), L), np.int64)
            for r, u in enumerate(ub):                     # a random window of L consecutive items per user
                o = rs.integers(0, max(lens[u] - L, 0) + 1)
                w = g.i[starts[u] + o:starts[u] + o + L] + 1
                S[r, :len(w)] = w
            S = torch.tensor(S, device=dev)
            msk = (torch.rand(S.shape, device=dev, generator=gen) < pmask) & (S != PAD)
            h = m(torch.where(msk, torch.full_like(S, MASK), S))[msk]
            tgt = S[msk]
            neg = pool[torch.randint(0, len(pool), (len(tgt), nneg), device=dev, generator=gen)]
            ps = (h * m.emb(tgt)).sum(-1) + m.bias(tgt)[:, 0]
            ns = (h[:, None, :] * m.emb(neg)).sum(-1) + m.bias(neg)[..., 0]
            loss = Fn.softplus(-(ps[:, None] - ns)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
    Q = np.zeros((len(qu), L), np.int64)
    qpos = np.zeros(len(qu), np.int64)
    for r, (u, t) in enumerate(zip(qu, qt)):            # window around the insertion point of the query time
        a, b = g.uptr[u], g.uptr[u + 1]
        p = a + np.searchsorted(g.t[a:b], t)
        left = g.i[max(a, p - L // 2):p] + 1
        right = g.i[p:min(b, p + L - 1 - len(left))] + 1
        seq = np.r_[left, MASK, right]
        Q[r, :len(seq)] = seq
        qpos[r] = len(left)
    m.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(Q), 1024):
            Qt = torch.tensor(Q[b:b + 1024], device=dev)
            h = m(Qt)[torch.arange(len(Qt), device=dev), torch.tensor(qpos[b:b + 1024], device=dev)]
            c = torch.tensor(qc[b:b + 1024] + 1, device=dev)
            out.append(((h[:, None, :] * m.emb(c)).sum(2) + m.bias(c)[..., 0]).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float32)


# =============================================================================================
# Tasks 2 and 4: which of two items is the held-out interaction
# =============================================================================================
@nb.njit(cache=True)
def user_span_features(qu, qc, uptr, utimes, iptr, itimes):
    """Task 2 (no query time): candidate activity during the user's own active period."""
    n, C = qc.shape
    out = np.zeros((n, C, 4))
    for r in range(n):
        a, b = uptr[qu[r]], uptr[qu[r] + 1]
        L = b - a
        if L == 0:
            continue
        tmin, tmax, tmed = utimes[a], utimes[b - 1], utimes[a + L // 2]
        q25, q75 = utimes[a + L // 4], utimes[a + 3 * L // 4]
        for c in range(C):
            ia, ib = iptr[qc[r, c]], iptr[qc[r, c] + 1]
            m = ib - ia
            if m == 0:
                continue
            ts = itimes[ia:ib]
            s1 = np.searchsorted(ts, tmax + 30 * 86400.0, side='right') - np.searchsorted(ts, tmin - 30 * 86400.0)
            s2 = np.searchsorted(ts, q75 + 60 * 86400.0, side='right') - np.searchsorted(ts, q25 - 60 * 86400.0)
            s3 = np.searchsorted(ts, tmed + 180 * 86400.0, side='right') - np.searchsorted(ts, tmed - 180 * 86400.0)
            out[r, c, 0] = np.log((s1 + 0.3) / (m + 3.0))
            out[r, c, 1] = np.log((s2 + 0.3) / (m + 3.0))
            out[r, c, 2] = np.log((s3 + 0.3) / (m + 3.0))
            out[r, c, 3] = 1.0 if tmax < ts[0] else 0.0
    return out


@nb.njit(cache=True)
def query_time_features(qc, qt, iptr, itimes):
    """Task 4: candidate activity in windows around the known query time."""
    n, C = qc.shape
    radii = np.array([2.0, 7.0, 21.0, 60.0, 180.0, 540.0, 1460.0]) * 86400.0
    out = np.zeros((n, C, 10))
    for r in range(n):
        t = qt[r]
        for c in range(C):
            ia, ib = iptr[qc[r, c]], iptr[qc[r, c] + 1]
            m = ib - ia
            if m == 0:
                continue
            ts = itimes[ia:ib]
            for k in range(7):
                cnt = np.searchsorted(ts, t + radii[k], side='right') - np.searchsorted(ts, t - radii[k])
                out[r, c, k] = np.log((cnt + 0.3) / (m + 3.0))
            if t < ts[0]:
                out[r, c, 7] = 1.0
                out[r, c, 8] = np.log1p((ts[0] - t) / 86400.0)
            out[r, c, 9] = max(0.0, 0.02 - np.searchsorted(ts, t) / m) * 50.0
    return out


@nb.njit(cache=True)
def user_knn(qu, qc, qt, uptr, uitems, utimes, iptr, iusers, idf, unorm, tau):
    """Task 4: user-based neighbourhood.  Overlap between u and every other user v (idf weighted, and
    additionally weighted by how close u's shared item is to the query time); candidate scores aggregate the
    similarities of the candidate's users.  Returns (n, C, 7)."""
    n, C = qc.shape
    nU = len(uptr) - 1
    ov = np.zeros((nU, 2))
    stamp = np.full(nU, -1, np.int64)
    out = np.zeros((n, C, 7))
    for r in range(n):
        u = qu[r]
        for p in range(uptr[u], uptr[u + 1]):
            j = uitems[p]
            rec = np.exp(-abs(utimes[p] - qt[r]) / (tau * 86400.0))
            for q in range(iptr[j], iptr[j + 1]):
                v = iusers[q]
                if v == u:
                    continue
                if stamp[v] != r:
                    stamp[v] = r
                    ov[v, 0] = 0.0
                    ov[v, 1] = 0.0
                ov[v, 0] += idf[j]
                ov[v, 1] += idf[j] * rec
        for c in range(C):
            it = qc[r, c]
            m = iptr[it + 1] - iptr[it]
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            mx = 0.0
            nm = 0.0
            sr = 0.0
            for q in range(iptr[it], iptr[it + 1]):
                v = iusers[q]
                if v == u or stamp[v] != r:
                    continue
                den = np.sqrt((unorm[u] + 5.0) * (unorm[v] + 5.0))
                sim = ov[v, 0] / den
                nm += 1.0
                s1 += sim
                s2 += sim * sim
                s3 += sim * sim * sim
                mx = max(mx, sim)
                sr += (ov[v, 1] / den) ** 2
            out[r, c, 0] = np.log(s1 / (m + 3.0) + 1e-3)
            out[r, c, 1] = np.log(s2 + 1e-4)
            out[r, c, 2] = np.log(s3 + 1e-5)
            out[r, c, 3] = mx
            out[r, c, 4] = np.log((nm + 0.1) / (m + 3.0))
            out[r, c, 5] = np.log(sr / (m + 3.0) + 1e-6)
            out[r, c, 6] = np.log(s2 / (m + 3.0) + 1e-6)
    return out


def pair_features(g, sv, task, qu, qc, qt):
    F = [cooc(g, task, qu, qc, qt)]
    if task == 2:
        F.append(user_span_features(qu, qc, g.uptr, g.t, g.iptr, g.itimes))
    else:
        F.append(query_time_features(qc, qt, g.iptr, g.itimes))
        idf = np.log1p(g.nU / (g.icnt + 5.0))
        unorm = np.bincount(g.u, weights=idf[g.i], minlength=g.nU)
        F.append(user_knn(qu, qc, qt, g.uptr, g.i, g.t, g.iptr, g.iusers, idf, unorm, 365.0))
    F.append(np.stack([sv.features(g, qu, qc[:, c]) for c in range(qc.shape[1])], 1))
    return np.concatenate(F, -1).astype(np.float32), np.log1p(g.ucnt[qu])[:, None].astype(np.float32)


# ---------------------------------------------------------------------------------------------
# EASE (Steck 2019): closed-form item-item regression  B = I - P diag(1/diag P),  P = (X'X + lam I)^-1
# ---------------------------------------------------------------------------------------------
EASE_LAMBDA = {2: (300.0,), 4: (500.0, 2000.0)}
EASE_TAUS = {2: (), 4: (30.0, 365.0)}


@nb.njit(parallel=True, cache=True)
def _ease_gram(uptr, ui, G, nblk):
    for blk in nb.prange(nblk):                      # each thread owns the columns i with i % nblk == blk
        for u in range(len(uptr) - 1):
            for x in range(uptr[u], uptr[u + 1]):
                i = ui[x]
                if i % nblk != blk:
                    continue
                for y in range(uptr[u], uptr[u + 1]):
                    G[ui[y], i] += 1.0


@nb.njit(parallel=True, cache=True)
def _ease_finish(P):
    n = P.shape[0]
    d = np.empty(n, np.float32)
    for j in range(n):
        d[j] = P[j, j]
    for j in nb.prange(n):                           # LAPACK left the lower triangle: mirror it
        for i in range(j + 1, n):
            P[j, i] = P[i, j]
    for j in nb.prange(n):
        for i in range(n):
            P[i, j] = -P[i, j] / d[j]
        P[j, j] = 0.0


def ease_matrix(u, i, nU, nI, lam):
    """EASE weights from binary interactions, computed in place (one nI x nI float32 matrix, ~10 GB for Task 2)."""
    from scipy.linalg import lapack
    o = np.lexsort((i, u))
    u, i = u[o], i[o]
    first = np.r_[True, (u[1:] != u[:-1]) | (i[1:] != i[:-1])]
    u, i = u[first], i[first]
    G = np.zeros((nI, nI), np.float32, order='F')
    _ease_gram(np.searchsorted(u, np.arange(nU + 1)), i, G, 64)
    G[np.diag_indices(nI)] += lam
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=8, user_api='blas'):          # the run sets OPENBLAS_NUM_THREADS=1
        c, info = lapack.spotrf(G, lower=1, overwrite_a=1, clean=0)
        assert info == 0
        P, info = lapack.spotri(c, lower=1, overwrite_c=1)
        assert info == 0
    del c, G
    _ease_finish(P)
    return P


@nb.njit(parallel=True, cache=True)
def _ease_scores(qu, qc, qt, uptr, ui, ut, B, taus):
    n, C = qc.shape
    out = np.zeros((n, C, 1 + len(taus)), np.float32)
    for r in nb.prange(n):
        u = qu[r]
        for y in range(uptr[u], uptr[u + 1]):
            h = ui[y]
            for c in range(C):
                b = B[h, qc[r, c]]
                out[r, c, 0] += b
                for k in range(len(taus)):
                    out[r, c, 1 + k] += b * np.exp(-abs(ut[y] - qt[r]) / (taus[k] * 86400.0))
    return out


def ease_features(g, B, task, qu, qc, qt):
    """EASE score of each candidate from the user's history (Task 4 also with history weighted by closeness to the
    query time), raw and divided by popularity^0.75 (negatives are popularity-matched, so the lift matters)."""
    S = _ease_scores(qu, qc, qt, g.uptr, g.i, g.t, B, np.array(EASE_TAUS[task], np.float64))
    return np.concatenate([S, S / np.maximum(g.icnt[qc], 1.0)[..., None] ** 0.75], -1).astype(np.float32)


def deficit_rule(margins, qc, seen, item_train_count, item_other_test_count, core=10):
    """The task 1-4 datasets are 10-cores: every item has >= 10 interactions in train + both test files.
    An item with deficit d = 10 - train - (other task's test) must be the positive in at least d of the rows
    where it is a candidate.  Enforce this by flipping its most favourable rows when the model disagrees."""
    deficit = np.maximum(core - item_train_count - item_other_test_count, 0)
    rows = {}
    for k in range(len(qc)):
        for c in range(2):
            if deficit[qc[k, c]] > 0 and not seen[k, c]:
                rows.setdefault(qc[k, c], []).append((margins[k] if c == 0 else -margins[k], k, c))
    changed = 0
    for item, lst in rows.items():
        lst.sort(reverse=True)
        for rank, (score, k, c) in enumerate(lst[:deficit[item]]):
            if score <= 0:
                margins[k] = 50.0 if c == 0 else -50.0
                changed += 1
    return margins, changed


def run_pair_task(task, data, folds=4, holdout=30000, negatives=2):
    """Validation folds: hold out random interactions, draw popularity negatives, build features on the rest.
    Latent models are trained separately on every fold's remaining interactions (and on all data for the test)."""
    rng = np.random.default_rng(SEED + task)
    perm = rng.permutation(len(data.u))
    sets, problems = [], []
    for f in range(folds):
        held = perm[f * holdout:(f + 1) * holdout]
        keep = np.ones(len(data.u), bool)
        keep[held] = False
        g = Graph(data.u[keep], data.i[keep], data.t[keep], data.r[keep], data.nU, data.nI)
        sv = Spectral(g)
        qc = np.column_stack([data.i[held], sample_negatives(data.i, data.i[held], negatives, rng)])
        qt = data.t[held] if task == 4 else np.zeros(len(held))
        F, ctx = pair_features(g, sv, task, data.u[held], qc, qt)
        sets.append((F, ctx, g.seen(data.u[held], qc)))
        problems.append((g, data.u[held], qc, qt))
        log(f'task {task} fold {f}: {F.shape}')
    g = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    name = f'task_{task}_test.csv'
    qu, qc, te = test_pairs(data, name)
    qt = te['timestamp'] if task == 4 else np.zeros(len(qu))
    problems.append((g, qu, qc, qt))
    latent = latent_scores(task, problems)
    if task == 4:                       # fill-in-the-blank sequence model, one per fold and one on all data
        latent = [np.concatenate([lat, bert4rec_scores(gp, up, tp, cp, seed=k)[..., None]], -1)
                  for k, (lat, (gp, up, cp, tp)) in enumerate(zip(latent, problems))]
    log(f'task {task}: latent models trained')
    # EASE: weights without the validation interactions (for the folds) and on all data (for the test)
    held_all = perm[:folds * holdout]
    keep = np.ones(len(data.u), bool)
    keep[held_all] = False
    ease = [[] for _ in problems]
    for lam in EASE_LAMBDA[task]:
        if data.nI <= 20000:            # Task 4: cheap, so one matrix per fold (each leaves out only that fold)
            for f, (gp, up, cp, tp) in enumerate(problems[:-1]):
                B = ease_matrix(gp.u, gp.i, data.nU, data.nI, lam)
                ease[f].append(ease_features(gp, B, task, up, cp, tp))
                del B
        else:                           # Task 2: 51k items (~5 min per matrix), one matrix without any fold
            B = ease_matrix(data.u[keep], data.i[keep], data.nU, data.nI, lam)
            for f, (gp, up, cp, tp) in enumerate(problems[:-1]):
                ease[f].append(ease_features(gp, B, task, up, cp, tp))
            del B
        B = ease_matrix(data.u, data.i, data.nU, data.nI, lam)
        ease[-1].append(ease_features(g, B, task, qu, qc, qt))
        del B
    ease = [np.concatenate(e, -1) for e in ease]
    log(f'task {task}: EASE features done')
    sets = [(np.concatenate([F, latent[f], ease[f]], -1), ctx, seen) for f, (F, ctx, seen) in enumerate(sets)]
    clf = fit_pair_classifier([(F, c) for F, c, _ in sets[1:]], 400)
    F, ctx, seen = sets[0]
    m = pair_margin(clf, F[:, 0], F[:, 1], ctx) + 100 * (seen[:, 1] - seen[:, 0])
    metrics = {'validation_accuracy_fold0': float(np.mean(m > 0))}
    log(f'task {task} validation accuracy (fold 0, trained on {folds - 1} folds): {metrics["validation_accuracy_fold0"]:.4f}')
    clf = fit_pair_classifier([(F, c) for F, c, _ in sets], 400)
    F, ctx = pair_features(g, Spectral(g), task, qu, qc, qt)
    F = np.concatenate([F, latent[-1], ease[-1]], -1)
    # a candidate already in the user's history -- training file or the other task's test file -- is the negative
    ot = data.tests['task_1_test.csv' if task == 2 else 'task_3_test.csv']
    known = set(zip(ot['user'].tolist(), ot['item'].tolist()))
    extra = np.array([[(u, c) in known for c in row] for u, row in zip(qu.tolist(), qc.tolist())], np.float32)
    seen = np.maximum(g.seen(qu, qc), extra)
    metrics['other_test_seen_rows'] = int(extra.any(1).sum())
    m = pair_margin(clf, F[:, 0], F[:, 1], ctx)
    m = np.where(seen[:, 0] != seen[:, 1], 100.0 * (seen[:, 1] - seen[:, 0]), m)
    other = ot['item']
    m, changed = deficit_rule(m, qc, seen, np.bincount(data.i, minlength=data.nI), np.bincount(other, minlength=data.nI))
    metrics['deficit_rule_rows'] = changed
    log(f'task {task}: 10-core deficit rule set {changed} rows')
    return pick(m, data, name), metrics


# =============================================================================================
# Task 5: next item (ordered histories)
# =============================================================================================
AFTER_WINDOWS = [(0, 7), (0, 30), (0, 90), (0, 365), (0, 3650), (-30, 0), (-365, 0), (-3650, 0)]


def position_stats(g):
    """Order-only item statistics: mean relative position in histories and how often the item is among the
    last 1/3/5 items of a history."""
    rel = (g.upos + 1.0) / g.ucnt[g.u]
    back = g.ucnt[g.u] - g.upos
    cnt = np.bincount(g.i, minlength=g.nI).astype(float)
    rel_mean = (np.bincount(g.i, weights=rel, minlength=g.nI) + 2 * 0.58) / (cnt + 2)
    rel4 = (np.bincount(g.i, weights=rel ** 4, minlength=g.nI) + 2 * 0.24) / (cnt + 2)
    rates = [np.log((np.bincount(g.i, weights=(back <= k).astype(float), minlength=g.nI) + a) / (cnt + 3))
             for k, a in [(1, 0.3), (3, 0.6), (5, 1.0)]]
    last3 = back <= 3
    end_era = np.bincount(g.u[last3], weights=rel_mean[g.i[last3]], minlength=g.nU) / np.maximum(
        np.bincount(g.u[last3], minlength=g.nU), 1)
    return rel_mean, rel4, rates, end_era


def task5_features(g, sv, qu, qc, data_end, order_only=False):
    """Per-candidate features (n, C, f) and per-query context.  Tu = time of the user's last known event.
    order_only=True drops every timestamp-based feature (g.t then holds row order, not time)."""
    Tu = g.last_time(qu)
    rel_mean, rel4, rates, end_era = position_stats(g)
    co = cooc(g, 5, qu, qc, np.zeros(len(qu)))
    F = []
    for c in range(qc.shape[1]):
        it = qc[:, c]
        parts = [rel_mean[it], rel4[it]] + [x[it] for x in rates] + [np.abs(end_era[qu] - rel_mean[it])]
        if not order_only:
            parts.append(time_window_features(g, it, Tu, AFTER_WINDOWS))
        F.append(np.column_stack(parts + [sv.features(g, qu, it), co[:, c]]))
    a, b = g.uptr[qu], g.uptr[qu + 1]
    ctx = [np.log1p(b - a)]
    if not order_only:
        gap = (Tu - g.t[np.maximum(b - 2, a)]) / DAY
        ctx += [(Tu - 1.4e9) / 3e7, np.log1p(np.maximum(gap, 0)), np.log1p(np.maximum(data_end - Tu, 0) / DAY)]
    return np.stack(F, 1).astype(np.float32), np.column_stack(ctx).astype(np.float32)


def task5_pseudo_set(data, cutoff, rng, negatives=3, order_only=False):
    """Hold out every user's last event (before `cutoff`, if given): that event plays the role of the test item."""
    keep = data.t < cutoff if cutoff else np.ones(len(data.u), bool)
    g_all = Graph(data.u[keep], data.i[keep], data.t[keep], data.r[keep], data.nU, data.nI)
    last = g_all.uptr[1:] - 1
    users = np.flatnonzero(g_all.ucnt >= 4)
    held = last[users]
    mask = np.ones(len(g_all.u), bool)
    mask[held] = False
    g = Graph(g_all.u[mask], g_all.i[mask], g_all.t[mask], g_all.r[mask], data.nU, data.nI)
    sv = Spectral(g, recency=True)
    qc = np.column_stack([g_all.i[held], sample_negatives(g.i, g_all.i[held], negatives, rng)])
    F, ctx = task5_features(g, sv, users, qc, cutoff or TEND, order_only)
    return F, ctx, g.seen(users, qc), (g, users, qc, g.last_time(users))


def run_task5(data, order_only=False):
    """order_only=True: use the within-user order of the training file but no timestamp values at all."""
    rng = np.random.default_rng(SEED + 5)
    if order_only:
        data.t = np.arange(len(data.u), dtype=np.float64)          # row order (file is sorted by user, then time)
        cuts = [None]
    else:
        cuts = [None] + [calendar.timegm(time.strptime(s, '%Y-%m-%d')) for s in ('2017-08-01', '2017-05-01', '2017-01-01', '2016-07-01')]
    sets = []
    for cut in cuts:
        sets.append(task5_pseudo_set(data, cut, rng, order_only=order_only))
        log(f'task 5 pseudo set (cutoff {cut}): {sets[-1][0].shape}')
    g = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    name = 'task_5_test.csv'
    qu, qc, _ = test_pairs(data, name)
    latent = latent_scores(5, [s[3] for s in sets] + [(g, qu, qc, g.last_time(qu))])
    seqs = [sasrec_scores(gk, users, qck, seed=k) for k, (_, _, _, (gk, users, qck, _)) in enumerate(sets)]
    seqs.append(sasrec_scores(g, qu, qc, seed=len(sets)))
    log('task 5: latent models and SASRec trained')
    sets = [(np.concatenate([F, latent[k], seqs[k][..., None]], -1), ctx, seen) for k, (F, ctx, seen, _) in enumerate(sets)]
    F, ctx, seen = sets[0]
    if order_only:          # validation: two halves of the users
        half = np.arange(len(F)) % 2 == 0
        clf = fit_pair_classifier([(F[~half], ctx[~half])], 500)
        F, ctx, seen = F[half], ctx[half], seen[half]
    else:                   # validation: train on the cutoff sets, evaluate on the last-event set
        clf = fit_pair_classifier([(F, c) for F, c, _ in sets[1:]], 1000, 'large')
    m = pair_margin(clf, F[:, 0], F[:, 1], ctx) + 100 * (seen[:, 1] - seen[:, 0])
    metrics = {'validation_accuracy_last_event': float(np.mean(m > 0)), 'order_only': order_only}
    log(f'task 5 validation accuracy (last-event pseudo test): {metrics["validation_accuracy_last_event"]:.4f}')
    clf = fit_pair_classifier([(F, c) for F, c, _ in sets], 1000, 'large')
    sv = Spectral(g, recency=True)
    F, ctx = task5_features(g, sv, qu, qc, TEND, order_only)
    F = np.concatenate([F, latent[-1], seqs[-1][..., None]], -1)
    seen = g.seen(qu, qc)
    m = pair_margin(clf, F[:, 0], F[:, 1], ctx)
    m = np.where(seen[:, 0] != seen[:, 1], 100.0 * (seen[:, 1] - seen[:, 0]), m)
    new1, new2 = g.icnt[qc[:, 0]] == 0, g.icnt[qc[:, 1]] == 0       # negatives are always training items
    m = np.where(new1 & ~new2, 100.0, np.where(new2 & ~new1, -100.0, m))
    return pick(m, data, name), metrics


# =============================================================================================
# Task 6: time until the user's final interaction
# =============================================================================================
@nb.njit(cache=True)
def user_gap_stats(uptr, ut):
    nU = len(uptr) - 1
    out = np.full((nU, 12), np.nan)
    for u in range(nU):
        a, b = uptr[u], uptr[u + 1]
        L = b - a
        if L < 2:
            continue
        g = np.empty(L - 1)
        for p in range(a + 1, b):
            g[p - a - 1] = ut[p] - ut[p - 1]
        lg = np.log1p(np.maximum(g, 0.0))
        sg = np.sort(g)
        out[u, 0] = np.log1p(L)
        out[u, 1] = np.mean(lg)
        out[u, 2] = np.std(lg)
        out[u, 3] = np.log1p(max(sg[(L - 1) // 2], 0.0))
        out[u, 4] = np.log1p(max(sg[int(0.8 * (L - 2))], 0.0))
        out[u, 5] = np.log1p(max(sg[int(0.2 * (L - 2))], 0.0))
        out[u, 6] = lg[-1]
        out[u, 7] = np.mean(lg[-3:])
        out[u, 8] = np.log1p(max(ut[b - 1] - ut[a], 0.0))
        out[u, 9] = np.mean(g == 0)
        out[u, 10] = np.log1p(max(np.mean(g), 0.0))
        out[u, 11] = np.mean(g < 86400.0)
    return out


@nb.njit(cache=True)
def item_after_quantiles(items, T, iptr, itimes):
    """Quantiles of the item's interaction times after T (days after T) and how many there are."""
    n = len(items)
    out = np.full((n, 5), np.nan)
    for k in range(n):
        a, b = iptr[items[k]], iptr[items[k] + 1]
        j = np.searchsorted(itimes[a:b], T[k])
        m = b - a - j
        out[k, 4] = np.log1p(m)
        for qq, q in enumerate((0.1, 0.25, 0.5, 0.75)):
            if m > 0:
                out[k, qq] = np.log1p(max(itimes[a + j + int(q * (m - 1))] - T[k], 0.0) / 86400.0)
    return out


def item_gap_priors(g):
    """Per item: typical (log) gap that preceded other users' interactions with it, its spread, the same
    relative to each user's own typical gap, and how often the item was a user's last event."""
    same = np.r_[False, g.u[1:] == g.u[:-1]]
    lg = np.log1p(np.maximum(g.t - np.r_[0, g.t[:-1]], 0))
    ok = same
    cnt = np.bincount(g.i[ok], minlength=g.nI).astype(float)
    s1 = np.bincount(g.i[ok], weights=lg[ok], minlength=g.nI)
    s2 = np.bincount(g.i[ok], weights=lg[ok] ** 2, minlength=g.nI)
    umean = np.bincount(g.u[ok], weights=lg[ok], minlength=g.nU) / np.maximum(np.bincount(g.u[ok], minlength=g.nU), 1)
    s3 = np.bincount(g.i[ok], weights=lg[ok] - umean[g.u[ok]], minlength=g.nI)
    prior, mu = 10.0, lg[ok].mean()
    mean = (s1 + prior * mu) / (cnt + prior)
    sd = np.sqrt(np.maximum((s2 + prior * (mu ** 2 + 4)) / (cnt + prior) - mean ** 2, 0))
    is_last = np.r_[g.u[1:] != g.u[:-1], True].astype(float)
    last_rate = (np.bincount(g.i, weights=is_last, minlength=g.nI) + 0.5) / (g.icnt + 5)
    return np.column_stack([mean, sd, s3 / (cnt + prior), np.log(last_rate), np.log1p(cnt)])


@nb.njit(cache=True)
def _coreader_lag(qu, qi, uptr, ui, ut, iptr, iusers, itimes, sptr, sitem, stime, m, maxr):
    """For a user u (last event at p) and target item j: every other reader v of j who also read one of u's last m
    items h gives an estimate of when u reads j if u keeps v's pace:  t_u(h) + (t_v(j) - t_v(h)) - p.
    Returns weighted quantiles of these estimated gaps (weights favour u's most recent items) and support counts."""
    n = len(qu)
    out = np.full((n, 8), np.nan)
    buf = np.empty(maxr * m)
    wbuf = np.empty(maxr * m)
    for r in range(n):
        u, j = qu[r], qi[r]
        a, b = uptr[u], uptr[u + 1]
        if b - a == 0:
            continue
        p = ut[b - 1]
        lo = max(a, b - m)
        ja, jb = iptr[j], iptr[j + 1]
        step = max(1, (jb - ja) // maxr)
        k = 0
        nv = 0
        for x in range(ja, jb, step):
            v = iusers[x]
            if v == u:
                continue
            hit = False
            for y in range(lo, b):
                h = ui[y]
                L, R = sptr[v], sptr[v + 1]
                while L < R:                                   # binary search h in v's item-sorted history
                    M = (L + R) // 2
                    if sitem[M] < h:
                        L = M + 1
                    else:
                        R = M
                if L < sptr[v + 1] and sitem[L] == h:
                    if k < len(buf):
                        buf[k] = ut[y] + (itimes[x] - stime[L]) - p
                        wbuf[k] = 1.0 / (b - y)
                        k += 1
                    hit = True
            if hit:
                nv += 1
        out[r, 0] = np.log1p(nv)
        if k == 0:
            continue
        e, w = buf[:k], wbuf[:k]
        o = np.argsort(e)
        e, w = e[o], w[o]
        cw = np.cumsum(w) / w.sum()
        for qq, q in enumerate((0.25, 0.5, 0.75)):
            val = e[min(np.searchsorted(cw, q), k - 1)]
            out[r, 1 + qq] = np.sign(val) * np.log1p(abs(val) / 86400.0)
        out[r, 4] = np.mean(e < 0)
        out[r, 5] = np.log1p(np.median(np.abs(e)) / 86400.0)
        pos = e[e > 0]
        if len(pos) > 0:
            out[r, 6] = np.log1p(np.median(pos) / 86400.0)
        out[r, 7] = np.log1p(k)
    return out


def coreader_lag_features(g, qu, qi, m=10, maxr=3000):
    o = np.lexsort((g.i, g.u))
    sptr = np.searchsorted(g.u[o], np.arange(g.nU + 1))
    return _coreader_lag(qu.astype(np.int64), qi.astype(np.int64), g.uptr, g.i, g.t, g.iptr, g.iusers, g.itimes,
                         sptr, g.i[o], g.t[o], m, maxr)


def task6_features(g, qu, qi, tend=TEND):
    """Context: the user's gap profile, time left until the end of the data (tend), when the target item is read
    by others relative to the user's last event, and when co-readers moved from the user's recent books to it."""
    tl = g.last_time(qu)
    H = tend - tl
    G = user_gap_stats(g.uptr, g.t)[qu]
    F = [G, np.log1p(np.maximum(H, 0) / DAY), (tl - 1.4e9) / 3e7, np.log1p(g.icnt[qi])]
    F.append(time_window_features(g, qi, tl, [(0, 7), (0, 30), (0, 90), (0, 365), (0, 3650), (-365, 0)]))
    F.append(item_after_quantiles(qi, tl, g.iptr, g.itimes))
    F.append(item_gap_priors(g)[qi])
    F.append(reading_lag_features(g, qu, qi, tl))
    F.append(coreader_lag_features(g, qu, qi))
    return np.column_stack(F).astype(np.float32), H, tl


def reading_lag_features(g, qu, qi, tl):
    """How long after a book first appears this user usually reads it; first appearance of the target + that lag
    gives a direct estimate of when the user reads the target book."""
    first = np.full(g.nI, np.inf)
    np.minimum.at(first, g.i, g.t)
    age = (g.t - first[g.i]) / DAY
    o = np.lexsort((age, g.u))
    sa = age[o]
    ptr = np.searchsorted(g.u[o], np.arange(g.nU + 1))
    F = np.full((len(qu), 6), np.nan)
    for r, u in enumerate(qu):
        a = sa[ptr[u]:ptr[u + 1]]
        if len(a) == 0:
            continue
        med, q25 = a[len(a) // 2], a[len(a) // 4]
        F[r, :4] = np.log1p(med), np.log1p(q25), np.log1p(a[3 * len(a) // 4]), np.mean(a < 30)
        if np.isfinite(first[qi[r]]):
            F[r, 4] = np.log1p(max(first[qi[r]] + med * DAY - tl[r], 0) / DAY)
            F[r, 5] = np.log1p(max(first[qi[r]] + q25 * DAY - tl[r], 0) / DAY)
    return F


T6_SCALE = [(90, 1.25), (365, 1.20), (730, 1.10), (np.inf, 1.05)]     # (time left < days, factor)
T6_CUTOFF_DAYS = (30, 90, 180, 365, 730)
T6_PARAMS = dict(loss='absolute_error', max_iter=1200, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
                 l2_regularization=100.0, random_state=SEED)


def task6_last_event_sets(g_full, data):
    """L_k: drop each user's last k-1 events (k = 1, 2, 3); the new last event is the target, its gap from the
    previous event the label.  Every user takes part, but k-1 >= 1 further events still follow the target."""
    sets = []
    ends = g_full.uptr[1:]
    for k in (1, 2, 3):
        keep = np.ones(len(g_full.u), bool)
        for j in range(1, k + 1):
            keep[(ends - j)[g_full.ucnt >= j]] = False
        tgt = ends - k
        users = np.flatnonzero(tgt - g_full.uptr[:-1] >= 3)
        g = Graph(g_full.u[keep], g_full.i[keep], g_full.t[keep], g_full.r[keep], data.nU, data.nI)
        X, H, tl = task6_features(g, users, g_full.i[tgt[users]])
        sets.append((X, g_full.t[tgt[users]] - tl, H))
        log(f'task 6 pseudo set L{k}: {len(users):,} users')
    return sets


def task6_cutoff_sets(g_full, data):
    """C_d: pretend the data ends d days before TEND.  For users with a training event after that cutoff (so their
    history before it is complete), the last event before the cutoff is the target: exactly one event remains
    between the previous event and the end, as in the test.  Features use only data before the cutoff."""
    sets = []
    for d in T6_CUTOFF_DAYS:
        cut = TEND - d * DAY
        before = np.bincount(g_full.u[g_full.t < cut], minlength=data.nU)
        users = np.flatnonzero((g_full.ucnt > before) & (before >= 4))
        tgt = g_full.uptr[users] + before[users] - 1
        keep = g_full.t < cut
        keep[tgt] = False
        g = Graph(g_full.u[keep], g_full.i[keep], g_full.t[keep], g_full.r[keep], data.nU, data.nI)
        X, H, tl = task6_features(g, users, g_full.i[tgt], tend=cut)
        sets.append((X, g_full.t[tgt] - tl, H))
        log(f'task 6 cutoff set {d}d: {len(users):,} users')
    return sets


def run_task6(data):
    """Two kinds of pseudo-labels (see above): last-event sets cover every user, cutoff sets reproduce the test's
    'final event before the end of the data' condition.  Training on both was better than either alone when the
    method was checked on files whose users' real final events are known (Tasks 1-4 training data, final event
    held out); those files were used only for that check, never for training this model."""
    g_full = Graph(data.u, data.i, data.t, data.r, data.nU, data.nI)
    lsets = task6_last_event_sets(g_full, data)
    csets = task6_cutoff_sets(g_full, data)

    def fit(sets):
        X = np.vstack([s[0] for s in sets])
        y = np.concatenate([s[1] for s in sets])
        return HistGradientBoostingRegressor(**T6_PARAMS).fit(X, np.log1p(np.maximum(y, 0)))

    def predict(model, X, H):
        """Gaps grow toward the end of a history, so predictions are scaled up most when little time is left."""
        days = np.where(H > 0, H, np.inf) / DAY
        scale = np.select([days < lim for lim, _ in T6_SCALE], [f for _, f in T6_SCALE])
        p = np.expm1(model.predict(X)) * scale
        return np.maximum(np.where(H > 0, np.minimum(p, H), p), 0)

    # sanity validation: 2-fold over L1 users (L2/L3 and the cutoff sets always in training)
    X1, y1, H1 = lsets[0]
    half = np.random.default_rng(SEED).permutation(len(y1)) % 2 == 0
    errs = []
    for part in (half, ~half):
        model = fit([(X1[~part], y1[~part], None)] + lsets[1:] + csets)
        errs.append(np.mean(np.abs(predict(model, X1[part], H1[part]) - y1[part])))
    metrics = {'validation_mae_L1_seconds': float(np.mean(errs))}
    log(f'task 6 validation MAE on last-train-event targets: {metrics["validation_mae_L1_seconds"]:,.0f} s')
    model = fit(lsets + csets)
    te = data.tests['task_6_test.csv']
    X, H, _ = task6_features(g_full, te['user'], te['item'])
    return [repr(float(v)) for v in predict(model, X, H)], metrics


# =============================================================================================
# Tasks 1 and 3: rating prediction
#
# Positive ratings are the target (the leaderboard skips ground-truth zeros); zero ratings stay implicit interactions.
# Base feature bank (unchanged since v3): shrunk biases, explicit MF, user-block SVD++, spectral residual profiles,
# reliable top-K item neighbourhoods, cumulative-logit rating styles, inferred reading stages (Task 1) and temporal
# session/drift random effects (Task 3).  New in v6: the bank is computed out-of-fold for EVERY training rating
# (18 folds; each base fit leaves out 1/18), so the stacker learns from ~1.8M rows instead of 150k.  The stacker is
# a Ridge anchor plus a residual gradient-boosted tree, averaged 50/50 with an MLP stacker trained on the same rows;
# test features come from all 18 fold models (averaged).  New in v8: EASE rating-neighbourhood features (item-item
# for both tasks, user-user as well for Task 3) are appended to the bank.
# =============================================================================================
from numba import njit

def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def load(path):
    users, items, rows = ({}, {}, [])
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            u, i = (row['user'], row['item'])
            if u not in users:
                users[u] = len(users)
            if i not in items:
                items[i] = len(items)
            rows.append((users[u], items[i], int(row['timestamp']), int(row['rating'])))
    a = np.asarray(rows, dtype=np.int64)
    log(f'{path.name}: {len(a):,} rows, {len(users):,} users, {len(items):,} items')
    return (a, users, items)

def read_test(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def als_biases_custom(a, nu, ni, u_cnt, i_cnt, su=8.0, si=12.0, n_iter=15):
    mu = float(a[:, 3].mean())
    bu = np.zeros(nu, dtype=np.float64)
    bi = np.zeros(ni, dtype=np.float64)
    for _ in range(n_iter):
        bu = np.bincount(a[:, 0], weights=a[:, 3] - mu - bi[a[:, 1]], minlength=nu) / (u_cnt + su)
        bi = np.bincount(a[:, 1], weights=a[:, 3] - mu - bu[a[:, 0]], minlength=ni) / (i_cnt + si)
    return (mu, bu, bi)

@njit(cache=True)
def exact_2hop_uu_ii(q_users, q_items, q_times, has_time, u_ptr, u_items, u_times, u_res, i_ptr, i_users, i_times, i_res, u_cnt, i_cnt):
    n = len(q_users)
    out = np.zeros((n, 15), dtype=np.float32)
    in_ui = np.zeros(len(u_cnt), dtype=np.float32)
    res_ui = np.zeros(len(u_cnt), dtype=np.float32)
    uu_dot = np.zeros(len(u_cnt), dtype=np.float32)
    inv_u = 1.0 / np.sqrt(u_cnt.astype(np.float32) + 5.0)
    inv_i = 1.0 / np.sqrt(i_cnt.astype(np.float32) + 5.0)
    for idx in range(n):
        u, i = (q_users[idx], q_items[idx])
        t = q_times[idx] if has_time else 0
        if u >= len(u_cnt) or i >= len(i_cnt):
            continue
        istart, iend = (i_ptr[i], i_ptr[i + 1])
        ustart, uend = (u_ptr[u], u_ptr[u + 1])
        if istart == iend or ustart == uend:
            continue
        for p in range(istart, iend):
            up = i_users[p]
            in_ui[up] = inv_u[up]
            res_ui[up] = i_res[p]
        ii_num, ii_den, ii_num2, ii_den2, ii_num3, ii_den3 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ii_t3, ii_d3, ii_t30, ii_d30, ii_t365, ii_d365 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        agree_num, agree_den = (0.0, 0.0)
        max_sim_r, max_sim = (0.0, 0.0)
        norm_i = inv_i[i]
        norm_u = inv_u[u]
        for p in range(ustart, uend):
            j = u_items[p]
            if j == i:
                continue
            r_uj = u_res[p]
            inv_j = inv_i[j]
            jstart, jend = (i_ptr[j], i_ptr[j + 1])
            dot_ij = 0.0
            for q in range(jstart, jend):
                up = i_users[q]
                w_up = in_ui[up]
                if w_up > 0.0:
                    dot_ij += w_up
                    uu_dot[up] += inv_j
                    prod = r_uj * i_res[q]
                    agree_num += prod * res_ui[up] * norm_i * inv_j
                    agree_den += abs(prod) * norm_i * inv_j
            if dot_ij > 0.0:
                sim = dot_ij * norm_i * inv_j
                ii_num += sim * r_uj
                ii_den += sim
                sim2 = sim * sim
                ii_num2 += sim2 * r_uj
                ii_den2 += sim2
                sim3 = sim2 * sim
                ii_num3 += sim3 * r_uj
                ii_den3 += sim3
                if sim > max_sim:
                    max_sim = sim
                    max_sim_r = r_uj
                if has_time:
                    dt = float(abs(t - u_times[p]))
                    w3 = np.exp(-dt / (3.0 * 86400.0))
                    w30 = np.exp(-dt / (30.0 * 86400.0))
                    w365 = np.exp(-dt / (365.0 * 86400.0))
                    ii_t3 += sim * w3 * r_uj
                    ii_d3 += sim * w3
                    ii_t30 += sim * w30 * r_uj
                    ii_d30 += sim * w30
                    ii_t365 += sim * w365 * r_uj
                    ii_d365 += sim * w365
        uu_num, uu_den, uu_num2, uu_den2, uu_num3, uu_den3 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        uu_tnum, uu_tden = (0.0, 0.0)
        max_usim_r, max_usim = (0.0, 0.0)
        for p in range(istart, iend):
            up = i_users[p]
            d_u = uu_dot[up]
            if d_u > 0.0:
                sim_u = d_u * norm_u * inv_u[up]
                r_upi = res_ui[up]
                uu_num += sim_u * r_upi
                uu_den += sim_u
                sim_u2 = sim_u * sim_u
                uu_num2 += sim_u2 * r_upi
                uu_den2 += sim_u2
                sim_u3 = sim_u2 * sim_u
                uu_num3 += sim_u3 * r_upi
                uu_den3 += sim_u3
                if sim_u > max_usim:
                    max_usim = sim_u
                    max_usim_r = r_upi
                if has_time:
                    dt_u = float(abs(t - i_times[p]))
                    wt_u = np.exp(-dt_u / (90.0 * 86400.0))
                    uu_tnum += sim_u * wt_u * r_upi
                    uu_tden += sim_u * wt_u
                uu_dot[up] = 0.0
            in_ui[up] = 0.0
            res_ui[up] = 0.0
        out[idx, 0] = ii_num / (ii_den + 0.5)
        out[idx, 1] = ii_num / (ii_den + 2.5)
        out[idx, 2] = ii_num2 / (ii_den2 + 0.15)
        out[idx, 3] = ii_num3 / (ii_den3 + 0.03)
        out[idx, 4] = max_sim_r * (max_sim / (max_sim + 0.15))
        out[idx, 5] = agree_num / (agree_den + 1.0)
        out[idx, 6] = uu_num / (uu_den + 0.5)
        out[idx, 7] = uu_num / (uu_den + 2.5)
        out[idx, 8] = uu_num2 / (uu_den2 + 0.15)
        out[idx, 9] = uu_num3 / (uu_den3 + 0.03)
        out[idx, 10] = max_usim_r * (max_usim / (max_usim + 0.15))
        if has_time:
            out[idx, 11] = ii_t3 / (ii_d3 + 0.2)
            out[idx, 12] = ii_t30 / (ii_d30 + 0.3)
            out[idx, 13] = ii_t365 / (ii_d365 + 0.5)
            out[idx, 14] = uu_tnum / (uu_tden + 0.3)
    return out

@njit(cache=True)
def temporal_kernel_residuals(ids, times, ptr, sorted_times, sorted_res, taus, shrinks):
    n = len(ids)
    k = len(taus)
    out = np.zeros((n, k), dtype=np.float64)
    for idx in range(n):
        e, t = (ids[idx], times[idx])
        if e >= len(ptr) - 1:
            continue
        start, end = (ptr[e], ptr[e + 1])
        if start == end:
            continue
        pos = np.searchsorted(sorted_times[start:end], t) + start
        left, right = (max(start, pos - 100), min(end, pos + 100))
        for m in range(k):
            tau = taus[m]
            w_sum, r_sum = (0.0, 0.0)
            for p in range(left, right):
                dt = abs(t - sorted_times[p])
                w = np.exp(-dt / tau)
                w_sum += w
                r_sum += w * sorted_res[p]
            out[idx, m] = r_sum / (w_sum + shrinks[m])
    return out

import time as ratings_time

import numpy as ratings_np

from numba import njit as ratings_njit

from scipy.sparse import csr_matrix as ratings_csr_matrix

from scipy.sparse.linalg import svds as ratings_svds

ratings_SEED_PRIMARY = 258

ratings_RATING_CONFIGS = {1: {'best_stage_name': 'stage5_plus_conf_gating', 'best_l2': 40.0, 'best_eta': 1.0, 'best_hgb_cfg': {'max_leaf_nodes': 15, 'min_samples_leaf': 200, 'l2_regularization': 100.0, 'max_iter': 150, 'learning_rate': 0.03}, 'best_epochs': {'mf32': 28, 'mf64': 23, 'svdpp32': 3, 'svdpp64': 28, 'alpha32': 0.0, 'alpha64': 0.0}}, 3: {'best_stage_name': 'stage5_plus_conf_gating', 'best_l2': 3500.0, 'best_eta': 1.0, 'best_hgb_cfg': {'max_leaf_nodes': 15, 'min_samples_leaf': 200, 'l2_regularization': 100.0, 'max_iter': 150, 'learning_rate': 0.03}, 'best_epochs': {'mf32': 28, 'mf64': 11, 'svdpp32': 26, 'svdpp64': 28, 'alpha32': 0.0, 'alpha64': 0.0}}}

@ratings_njit(cache=True)
def ratings_mf_pos_epoch(a_rated, order, p, q, bu, bi, mu, lr, reg_b, reg_pq):
    k = p.shape[1]
    for idx in order:
        u = a_rated[idx, 0]
        i = a_rated[idx, 1]
        r = a_rated[idx, 3]
        dot = 0.0
        for d in range(k):
            dot += p[u, d] * q[i, d]
        err = r - (mu + bu[u] + bi[i] + dot)
        bu[u] += lr * (err - reg_b * bu[u])
        bi[i] += lr * (err - reg_b * bi[i])
        for d in range(k):
            pu = p[u, d]
            qi = q[i, d]
            p[u, d] += lr * (err * qi - reg_pq * pu)
            q[i, d] += lr * (err * pu - reg_pq * qi)

class ratings_ExplicitPosMF:

    def __init__(self, nu, ni, k, mu, bu_init=None, bi_init=None, seed=ratings_SEED_PRIMARY):
        rng = ratings_np.random.default_rng(seed)
        self.nu = nu
        self.ni = ni
        self.k = k
        self.mu = float(mu)
        self.p = rng.normal(0.0, 0.02, (nu, k)).astype(ratings_np.float32)
        self.q = rng.normal(0.0, 0.02, (ni, k)).astype(ratings_np.float32)
        self.bu = bu_init.astype(ratings_np.float32).copy() if bu_init is not None else ratings_np.zeros(nu, dtype=ratings_np.float32)
        self.bi = bi_init.astype(ratings_np.float32).copy() if bi_init is not None else ratings_np.zeros(ni, dtype=ratings_np.float32)

    def predict_components(self, q_u, q_i):
        u_valid = q_u < self.nu
        i_valid = q_i < self.ni
        both_valid = u_valid & i_valid
        u_safe = ratings_np.where(u_valid, q_u, 0)
        i_safe = ratings_np.where(i_valid, q_i, 0)
        b_u = ratings_np.where(u_valid, self.bu[u_safe], 0.0)
        b_i = ratings_np.where(i_valid, self.bi[i_safe], 0.0)
        dot = ratings_np.where(both_valid, ratings_np.sum(self.p[u_safe] * self.q[i_safe], axis=1), 0.0)
        pred = ratings_np.clip(self.mu + b_u + b_i + dot, 1.0, 5.0)
        return (pred.astype(ratings_np.float64), dot.astype(ratings_np.float64))

    def predict(self, q_u, q_i):
        return self.predict_components(q_u, q_i)[0]

def ratings_train_explicit_pos_mf(a_rated, nu, ni, k=32, lr0=0.012, decay=0.97, reg_b=0.02, reg_pq=0.055, max_epochs=28, bu_init=None, bi_init=None, tune_pos=None, seed=ratings_SEED_PRIMARY):
    mu = float(a_rated[:, 3].mean())
    model = ratings_ExplicitPosMF(nu, ni, k, mu, bu_init=bu_init, bi_init=bi_init, seed=seed)
    rng = ratings_np.random.default_rng(seed)
    best_mse = float('inf')
    best_epoch = max_epochs
    stale = 0
    best_state = None
    for ep in range(1, max_epochs + 1):
        lr = lr0 * decay ** (ep - 1)
        ratings_mf_pos_epoch(a_rated, rng.permutation(len(a_rated)), model.p, model.q, model.bu, model.bi, model.mu, lr, reg_b, reg_pq)
        if tune_pos is not None:
            pred = model.predict(tune_pos[:, 0], tune_pos[:, 1])
            mse = float(ratings_np.mean((pred - tune_pos[:, 3]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_epoch = ep
                stale = 0
                best_state = (model.p.copy(), model.q.copy(), model.bu.copy(), model.bi.copy())
            else:
                stale += 1
                if stale >= 5:
                    break
    if best_state is not None:
        model.p, model.q, model.bu, model.bi = best_state
    elif tune_pos is not None:
        pred = model.predict(tune_pos[:, 0], tune_pos[:, 1])
        best_mse = float(ratings_np.mean((pred - tune_pos[:, 3]) ** 2))
    return (model, best_epoch, best_mse)

@ratings_njit(cache=True)
def ratings_svdpp_user_block_epoch(u_order, u_pos_ptr, u_pos_items, u_pos_ratings, u_imp_ptr, u_imp_items, p, q, y, bu, bi, mu, lr_b, lr, reg_b, reg_p, reg_q, reg_y, ep_shrink_p, ep_shrink_q, ep_shrink_y, alpha):
    """Exact User-Block CSR Mini-Batch SVD++ in Numba.
    r_hat_{ui} = mu + b_u + b_i + q_i^T (p_u + h_u),  h_u = |N(u)|^{-1/2} sum_{j in N(u)} y_j
    Accumulates G_u = (1 / Z_u) sum_{i in R+(u)} e_{ui} q_i in float64,
    updates p_u, q_i, y_j AFTER computing all block errors,
    and applies explicit per-epoch L2 shrinkage on p, q, y.
    """
    k = p.shape[1]
    hu = ratings_np.zeros(k, dtype=ratings_np.float64)
    pu_hu = ratings_np.zeros(k, dtype=ratings_np.float64)
    Gu = ratings_np.zeros(k, dtype=ratings_np.float64)
    for idx_u in range(len(u_order)):
        u = u_order[idx_u]
        rs, re = (u_pos_ptr[u], u_pos_ptr[u + 1])
        n_pos = re - rs
        if n_pos == 0:
            continue
        ims, ime = (u_imp_ptr[u], u_imp_ptr[u + 1])
        n_imp = ime - ims
        inv_sqrt_imp = 1.0 / ratings_np.sqrt(float(n_imp)) if n_imp > 0 else 0.0
        for d in range(k):
            hu[d] = 0.0
        for p_idx in range(ims, ime):
            j = u_imp_items[p_idx]
            for d in range(k):
                hu[d] += y[j, d]
        for d in range(k):
            hu[d] *= inv_sqrt_imp
            pu_hu[d] = p[u, d] + hu[d]
            Gu[d] = 0.0
        Z_u = float(n_pos) ** alpha if alpha > 0.0 else 1.0
        W_u = float(n_pos) / Z_u
        errs = ratings_np.empty(n_pos, dtype=ratings_np.float64)
        err_sum = 0.0
        for idx in range(n_pos):
            i = u_pos_items[rs + idx]
            r = u_pos_ratings[rs + idx]
            dot = 0.0
            for d in range(k):
                dot += q[i, d] * pu_hu[d]
            e = r - (mu + bu[u] + bi[i] + dot)
            errs[idx] = e
            err_sum += e
            we = e / Z_u
            for d in range(k):
                Gu[d] += we * q[i, d]
        bu[u] += lr_b * (err_sum - 8.0 * bu[u]) / (float(n_pos) + 8.0)
        for idx in range(n_pos):
            i = u_pos_items[rs + idx]
            e = errs[idx]
            we = e / Z_u
            bi[i] += lr_b * (e - reg_b * bi[i])
            denom_q = 1.0 + lr * (1.0 / Z_u) * reg_q
            for d in range(k):
                q[i, d] = (q[i, d] + lr * we * pu_hu[d]) / denom_q
        denom_p = 1.0 + lr * W_u * reg_p
        for d in range(k):
            p[u, d] = (p[u, d] + lr * Gu[d]) / denom_p
        if n_imp > 0:
            sy = lr * inv_sqrt_imp
            denom_y = 1.0 + lr * W_u * reg_y * inv_sqrt_imp
            for p_idx in range(ims, ime):
                j = u_imp_items[p_idx]
                for d in range(k):
                    y[j, d] = (y[j, d] + sy * Gu[d]) / denom_y
    if ep_shrink_p > 0.0:
        p *= ratings_np.float32(1.0 - lr * ep_shrink_p)
    if ep_shrink_q > 0.0:
        q *= ratings_np.float32(1.0 - lr * ep_shrink_q)
    if ep_shrink_y > 0.0:
        y *= ratings_np.float32(1.0 - lr * ep_shrink_y)

@ratings_njit(cache=True)
def ratings_svdpp_predict_components(q_u, q_i, u_imp_ptr, u_imp_items, p, q, y, bu, bi, mu):
    """Returns (clipped_pred, dot_pu_qi, dot_hu_qi)."""
    nu, k = p.shape
    ni = q.shape[0]
    h_all = ratings_np.zeros((nu, k), dtype=ratings_np.float32)
    for u in range(nu):
        ims, ime = (u_imp_ptr[u], u_imp_ptr[u + 1])
        n_imp = ime - ims
        if n_imp > 0:
            inv_s = ratings_np.float32(1.0 / ratings_np.sqrt(float(n_imp)))
            for p_idx in range(ims, ime):
                j = u_imp_items[p_idx]
                for d in range(k):
                    h_all[u, d] += y[j, d]
            for d in range(k):
                h_all[u, d] *= inv_s
    n = len(q_u)
    pred = ratings_np.empty(n, dtype=ratings_np.float64)
    dot_p = ratings_np.zeros(n, dtype=ratings_np.float64)
    dot_h = ratings_np.zeros(n, dtype=ratings_np.float64)
    for idx in range(n):
        u, i = (q_u[idx], q_i[idx])
        b_u = bu[u] if u < nu else 0.0
        b_i = bi[i] if i < ni else 0.0
        dp = 0.0
        dh = 0.0
        if u < nu and i < ni:
            for d in range(k):
                qi_d = q[i, d]
                dp += p[u, d] * qi_d
                dh += h_all[u, d] * qi_d
        dot_p[idx] = dp
        dot_h[idx] = dh
        val = mu + b_u + b_i + dp + dh
        if val < 1.0:
            val = 1.0
        elif val > 5.0:
            val = 5.0
        pred[idx] = val
    return (pred, dot_p, dot_h)

class ratings_UserBlockSVDpp:

    def __init__(self, nu, ni, k, mu, bu_init, bi_init, u_imp_ptr, u_imp_items, seed=ratings_SEED_PRIMARY):
        rng = ratings_np.random.default_rng(seed)
        self.nu = nu
        self.ni = ni
        self.k = k
        self.mu = float(mu)
        self.p = rng.normal(0.0, 0.015, (nu, k)).astype(ratings_np.float32)
        self.q = rng.normal(0.0, 0.015, (ni, k)).astype(ratings_np.float32)
        self.y = ratings_np.zeros((ni, k), dtype=ratings_np.float32)
        self.bu = bu_init.astype(ratings_np.float32).copy()
        self.bi = bi_init.astype(ratings_np.float32).copy()
        self.u_imp_ptr = u_imp_ptr
        self.u_imp_items = u_imp_items

    def predict_components(self, q_u, q_i):
        return ratings_svdpp_predict_components(q_u, q_i, self.u_imp_ptr, self.u_imp_items, self.p, self.q, self.y, self.bu, self.bi, self.mu)

    def predict(self, q_u, q_i):
        return self.predict_components(q_u, q_i)[0]

def ratings_train_user_block_svdpp(u_pos_ptr, u_pos_items, u_pos_ratings, u_imp_ptr, u_imp_items, nu, ni, mu0, bu0, bi0, k=32, alpha=0.0, lr_b0=0.04, lr0=0.012, decay=0.95, reg_b=0.05, reg_p=0.06, reg_q=0.06, reg_y=0.05, ep_shrink=0.003, max_epochs=25, tune_pos=None, seed=ratings_SEED_PRIMARY):
    model = ratings_UserBlockSVDpp(nu, ni, k, mu0, bu0, bi0, u_imp_ptr, u_imp_items, seed=seed)
    rng = ratings_np.random.default_rng(seed)
    best_mse = float('inf')
    best_epoch = max_epochs
    stale = 0
    best_state = None
    for ep in range(1, max_epochs + 1):
        d_factor = decay ** (ep - 1)
        ratings_svdpp_user_block_epoch(rng.permutation(nu).astype(ratings_np.int32), u_pos_ptr, u_pos_items, u_pos_ratings, u_imp_ptr, u_imp_items, model.p, model.q, model.y, model.bu, model.bi, model.mu, lr_b0 * d_factor, lr0 * d_factor, reg_b, reg_p, reg_q, reg_y, ep_shrink, ep_shrink, ep_shrink, alpha)
        if tune_pos is not None:
            pred = model.predict(tune_pos[:, 0], tune_pos[:, 1])
            mse = float(ratings_np.mean((pred - tune_pos[:, 3]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_epoch = ep
                stale = 0
                best_state = (model.p.copy(), model.q.copy(), model.y.copy(), model.bu.copy(), model.bi.copy())
            else:
                stale += 1
                if stale >= 4:
                    break
    if best_state is not None:
        model.p, model.q, model.y, model.bu, model.bi = best_state
    elif tune_pos is not None:
        pred = model.predict(tune_pos[:, 0], tune_pos[:, 1])
        best_mse = float(ratings_np.mean((pred - tune_pos[:, 3]) ** 2))
    return (model, best_epoch, best_mse)

def ratings_compute_spectral_features(a_rated, a_implicit, nu, ni, u_pos_cnt, i_pos_cnt, u_imp_cnt, i_imp_cnt, res, res_n, u_std, q_u, q_i, seed=ratings_SEED_PRIMARY):
    """SVD on degree-normalized a_implicit, aggregating a_rated residuals (raw and std-normalized)."""
    inv_u = 1.0 / ratings_np.sqrt(u_imp_cnt + 10.0)
    inv_i = 1.0 / ratings_np.sqrt(i_imp_cnt + 10.0)
    x_imp = ratings_csr_matrix((inv_u[a_implicit[:, 0]] * inv_i[a_implicit[:, 1]], (a_implicit[:, 0], a_implicit[:, 1])), shape=(nu, ni), dtype=ratings_np.float32)
    u_imp, s_imp, vt_imp = ratings_svds(x_imp, k=48, random_state=seed)
    v_i = (vt_imp.T * s_imp ** 0.5).astype(ratings_np.float32)
    v_i /= ratings_np.linalg.norm(v_i, axis=1, keepdims=True) + 1e-08
    u_u = (u_imp * s_imp ** 0.5).astype(ratings_np.float32)
    u_u /= ratings_np.linalg.norm(u_u, axis=1, keepdims=True) + 1e-08
    u_prof_sum = ratings_np.zeros((nu, 48), dtype=ratings_np.float32)
    ratings_np.add.at(u_prof_sum, a_rated[:, 0], res[:, None] * v_i[a_rated[:, 1]])
    u_prof5 = ratings_np.vstack([u_prof_sum / (u_pos_cnt[:, None] + 5.0), ratings_np.zeros((1, 48), ratings_np.float32)])
    u_prof1 = ratings_np.vstack([u_prof_sum / (u_pos_cnt[:, None] + 1.5), ratings_np.zeros((1, 48), ratings_np.float32)])
    i_prof_sum = ratings_np.zeros((ni, 48), dtype=ratings_np.float32)
    ratings_np.add.at(i_prof_sum, a_rated[:, 1], res[:, None] * u_u[a_rated[:, 0]])
    i_prof5 = ratings_np.vstack([i_prof_sum / (i_pos_cnt[:, None] + 5.0), ratings_np.zeros((1, 48), ratings_np.float32)])
    i_prof1 = ratings_np.vstack([i_prof_sum / (i_pos_cnt[:, None] + 1.5), ratings_np.zeros((1, 48), ratings_np.float32)])
    u_profn_sum = ratings_np.zeros((nu, 48), dtype=ratings_np.float32)
    ratings_np.add.at(u_profn_sum, a_rated[:, 0], res_n[:, None] * v_i[a_rated[:, 1]])
    u_profn5 = ratings_np.vstack([u_profn_sum / (u_pos_cnt[:, None] + 5.0), ratings_np.zeros((1, 48), ratings_np.float32)])
    i_profn_sum = ratings_np.zeros((ni, 48), dtype=ratings_np.float32)
    ratings_np.add.at(i_profn_sum, a_rated[:, 1], res_n[:, None] * u_u[a_rated[:, 0]])
    i_profn5 = ratings_np.vstack([i_profn_sum / (i_pos_cnt[:, None] + 5.0), ratings_np.zeros((1, 48), ratings_np.float32)])
    v_i_pad = ratings_np.vstack([v_i, ratings_np.zeros((1, 48), ratings_np.float32)])
    u_u_pad = ratings_np.vstack([u_u, ratings_np.zeros((1, 48), ratings_np.float32)])
    vi_q = v_i_pad[q_i]
    uu_q = u_u_pad[q_u]
    feat_ii5 = ratings_np.sum(u_prof5[q_u] * vi_q, axis=1)
    feat_ii1 = ratings_np.sum(u_prof1[q_u] * vi_q, axis=1)
    feat_uu5 = ratings_np.sum(i_prof5[q_i] * uu_q, axis=1)
    feat_uu1 = ratings_np.sum(i_prof1[q_i] * uu_q, axis=1)
    feat_iin5 = ratings_np.sum(u_profn5[q_u] * vi_q, axis=1) * u_std[q_u]
    feat_uun5 = ratings_np.sum(i_profn5[q_i] * uu_q, axis=1) * u_std[q_u]
    spec_cos = ratings_np.sum(uu_q * vi_q, axis=1)
    return ratings_np.column_stack([feat_ii5, feat_ii1, feat_uu5, feat_uu1, feat_iin5, feat_uun5, spec_cos])

@ratings_njit(cache=True)
def ratings__insert_top5(w, t1, t2, t3, t4, t5):
    if w > t1:
        return (w, t1, t2, t3, t4)
    elif w > t2:
        return (t1, w, t2, t3, t4)
    elif w > t3:
        return (t1, t2, w, t3, t4)
    elif w > t4:
        return (t1, t2, t3, w, t4)
    elif w > t5:
        return (t1, t2, t3, t4, w)
    return (t1, t2, t3, t4, t5)

@ratings_njit(cache=True)
def ratings_fused_knn_and_support(q_u, q_i, u_ptr, u_items, u_res, u_resn, i_ptr, i_users, i_res, i_resn, u_pos_cnt, i_pos_cnt, u_imp_cnt, i_imp_cnt):
    """Fused single-pass 2-hop neighborhood feature extractor on a_rated CSR.
    Computes:
      - 19 cosine/dot kNN features on raw positive residuals `u_res, i_res`
      - 19 cosine/dot kNN features on user-std-normalized residuals `u_resn, i_resn`
      - 18 explicit neighborhood support/confidence/variance/concentration features:
        [ii_sum_w, ii_neff, ii_cnt, ii_max_sim, ii_top1_conc, ii_top5_conc, ii_res_var, ii_pos_sum_w, ii_struct_neff,
         uu_sum_w, uu_neff, uu_cnt, uu_max_sim, uu_top1_conc, uu_top5_conc, uu_res_var, uu_pos_sum_w, uu_struct_neff]
    """
    n = len(q_u)
    nu = len(u_pos_cnt)
    ni = len(i_pos_cnt)
    f_raw = ratings_np.zeros((n, 19), dtype=ratings_np.float64)
    f_norm = ratings_np.zeros((n, 19), dtype=ratings_np.float64)
    f_supp = ratings_np.zeros((n, 18), dtype=ratings_np.float64)
    present = ratings_np.zeros(nu, dtype=ratings_np.bool_)
    target_r = ratings_np.zeros(nu, dtype=ratings_np.float64)
    target_rn = ratings_np.zeros(nu, dtype=ratings_np.float64)
    dot = ratings_np.zeros(nu, dtype=ratings_np.float64)
    dot_n = ratings_np.zeros(nu, dtype=ratings_np.float64)
    su = ratings_np.zeros(nu, dtype=ratings_np.float64)
    sv = ratings_np.zeros(nu, dtype=ratings_np.float64)
    su_n = ratings_np.zeros(nu, dtype=ratings_np.float64)
    sv_n = ratings_np.zeros(nu, dtype=ratings_np.float64)
    cnt = ratings_np.zeros(nu, dtype=ratings_np.float64)
    dotw = ratings_np.zeros(nu, dtype=ratings_np.float64)
    dotw_n = ratings_np.zeros(nu, dtype=ratings_np.float64)
    cw = ratings_np.zeros(nu, dtype=ratings_np.float64)
    inv_i_imp = 1.0 / ratings_np.sqrt(i_imp_cnt.astype(ratings_np.float64) + 10.0)
    inv_u_imp = 1.0 / ratings_np.sqrt(u_imp_cnt.astype(ratings_np.float64) + 5.0)
    for z in range(n):
        u = q_u[z]
        i = q_i[z]
        if u >= nu or i >= ni:
            continue
        istart, iend = (i_ptr[i], i_ptr[i + 1])
        ustart, uend = (u_ptr[u], u_ptr[u + 1])
        if istart == iend or ustart == uend:
            continue
        for p in range(istart, iend):
            v = i_users[p]
            present[v] = True
            target_r[v] = i_res[p]
            target_rn[v] = i_resn[p]
        nums = ratings_np.zeros(11, dtype=ratings_np.float64)
        dens = ratings_np.zeros(11, dtype=ratings_np.float64)
        nums_n = ratings_np.zeros(11, dtype=ratings_np.float64)
        dens_n = ratings_np.zeros(11, dtype=ratings_np.float64)
        ii_sum_w = 0.0
        ii_sum_w2 = 0.0
        ii_cnt = 0.0
        ii_max_sim = 0.0
        ii_t1, ii_t2, ii_t3, ii_t4, ii_t5 = (0.0, 0.0, 0.0, 0.0, 0.0)
        ii_we = 0.0
        ii_we2 = 0.0
        ii_struct_w = 0.0
        ii_struct_w2 = 0.0
        norm_i = 1.0 / ratings_np.sqrt(float(i_imp_cnt[i]) + 5.0)
        for p in range(ustart, uend):
            j = u_items[p]
            if j == i:
                continue
            ru = float(u_res[p])
            run = float(u_resn[p])
            w_j = inv_i_imp[j]
            norm_j = 1.0 / ratings_np.sqrt(float(i_imp_cnt[j]) + 5.0)
            ij = 0.0
            ii_s = 0.0
            jj_s = 0.0
            ij_n = 0.0
            ii_sn = 0.0
            jj_sn = 0.0
            nc = 0.0
            struct_dot = 0.0
            jstart, jend = (i_ptr[j], i_ptr[j + 1])
            for q in range(jstart, jend):
                v = i_users[q]
                if present[v]:
                    rv = float(i_res[q])
                    rvn = float(i_resn[q])
                    ri = target_r[v]
                    rin = target_rn[v]
                    dot[v] += ru * rv
                    su[v] += ru * ru
                    sv[v] += rv * rv
                    dot_n[v] += run * rvn
                    su_n[v] += run * run
                    sv_n[v] += rvn * rvn
                    cnt[v] += 1.0
                    dotw[v] += ru * rv * w_j
                    dotw_n[v] += run * rvn * w_j
                    cw[v] += w_j
                    ij += ri * rv
                    ii_s += ri * ri
                    jj_s += rv * rv
                    ij_n += rin * rvn
                    ii_sn += rin * rin
                    jj_sn += rvn * rvn
                    nc += 1.0
                    struct_dot += inv_u_imp[v]
            if struct_dot > 0.0:
                ssim = struct_dot * norm_i * norm_j
                ii_struct_w += ssim
                ii_struct_w2 += ssim * ssim
            if nc < 2.0:
                continue
            cos = ij / (ratings_np.sqrt(ii_s * jj_s) + 1e-08)
            cos_n = ij_n / (ratings_np.sqrt(ii_sn * jj_sn) + 1e-08)
            s5 = cos * nc / (nc + 5.0)
            as5 = abs(s5)
            ii_sum_w += as5
            ii_sum_w2 += as5 * as5
            ii_cnt += 1.0
            if as5 > ii_max_sim:
                ii_max_sim = as5
            ii_t1, ii_t2, ii_t3, ii_t4, ii_t5 = ratings__insert_top5(as5, ii_t1, ii_t2, ii_t3, ii_t4, ii_t5)
            ii_we += as5 * ru
            ii_we2 += as5 * ru * ru
            for k in range(4):
                shrink = (2.0, 5.0, 10.0, 30.0)[k]
                s = cos * nc / (nc + shrink)
                nums[k] += s * ru
                dens[k] += abs(s)
                ps = s if s > 0.0 else 0.0
                ps2 = ps * ps
                nums[4 + k] += ps2 * ru
                dens[4 + k] += ps2
                sn = cos_n * nc / (nc + shrink)
                nums_n[k] += sn * run
                dens_n[k] += abs(sn)
                psn = sn if sn > 0.0 else 0.0
                psn2 = psn * psn
                nums_n[4 + k] += psn2 * run
                dens_n[4 + k] += psn2
            s_dot = ij / (nc + 10.0)
            nums[8] += s_dot * ru
            dens[8] += abs(s_dot)
            sn_dot = ij_n / (nc + 10.0)
            nums_n[8] += sn_dot * run
            dens_n[8] += abs(sn_dot)
        uu_nums = ratings_np.zeros(8, dtype=ratings_np.float64)
        uu_dens = ratings_np.zeros(8, dtype=ratings_np.float64)
        uu_nums_n = ratings_np.zeros(8, dtype=ratings_np.float64)
        uu_dens_n = ratings_np.zeros(8, dtype=ratings_np.float64)
        uu_sum_w = 0.0
        uu_sum_w2 = 0.0
        uu_cnt = 0.0
        uu_max_sim = 0.0
        uu_t1, uu_t2, uu_t3, uu_t4, uu_t5 = (0.0, 0.0, 0.0, 0.0, 0.0)
        uu_we = 0.0
        uu_we2 = 0.0
        uu_struct_w = 0.0
        uu_struct_w2 = 0.0
        norm_u = inv_u_imp[u]
        for p in range(istart, iend):
            v = i_users[p]
            r = target_r[v]
            rn = target_rn[v]
            nc = cnt[v]
            if cw[v] > 0.0:
                usim = cw[v] * norm_u * inv_u_imp[v]
                uu_struct_w += usim
                uu_struct_w2 += usim * usim
            if nc >= 2.0:
                cos = dot[v] / (ratings_np.sqrt(su[v] * sv[v]) + 1e-08)
                cos_n = dot_n[v] / (ratings_np.sqrt(su_n[v] * sv_n[v]) + 1e-08)
                s5 = cos * nc / (nc + 5.0)
                as5 = abs(s5)
                uu_sum_w += as5
                uu_sum_w2 += as5 * as5
                uu_cnt += 1.0
                if as5 > uu_max_sim:
                    uu_max_sim = as5
                uu_t1, uu_t2, uu_t3, uu_t4, uu_t5 = ratings__insert_top5(as5, uu_t1, uu_t2, uu_t3, uu_t4, uu_t5)
                uu_we += as5 * r
                uu_we2 += as5 * r * r
                for k in range(4):
                    shrink = (2.0, 5.0, 10.0, 30.0)[k]
                    s = cos * nc / (nc + shrink)
                    uu_nums[k] += s * r
                    uu_dens[k] += abs(s)
                    ps = s if s > 0.0 else 0.0
                    ps2 = ps * ps
                    uu_nums[4 + k] += ps2 * r
                    uu_dens[4 + k] += ps2
                    sn = cos_n * nc / (nc + shrink)
                    uu_nums_n[k] += sn * rn
                    uu_dens_n[k] += abs(sn)
                    psn = sn if sn > 0.0 else 0.0
                    psn2 = psn * psn
                    uu_nums_n[4 + k] += psn2 * rn
                    uu_dens_n[4 + k] += psn2
                s_dot = dot[v] / (nc + 10.0)
                nums[9] += s_dot * r
                dens[9] += abs(s_dot)
                sn_dot = dot_n[v] / (nc + 10.0)
                nums_n[9] += sn_dot * rn
                dens_n[9] += abs(sn_dot)
                s_dw = dotw[v] / (cw[v] + 2.0)
                nums[10] += s_dw * r
                dens[10] += abs(s_dw)
                sn_dw = dotw_n[v] / (cw[v] + 2.0)
                nums_n[10] += sn_dw * rn
                dens_n[10] += abs(sn_dw)
            present[v] = False
            target_r[v] = 0.0
            target_rn[v] = 0.0
            dot[v] = 0.0
            dot_n[v] = 0.0
            su[v] = 0.0
            sv[v] = 0.0
            su_n[v] = 0.0
            sv_n[v] = 0.0
            cnt[v] = 0.0
            dotw[v] = 0.0
            dotw_n[v] = 0.0
            cw[v] = 0.0
        for k in range(9):
            f_raw[z, k] = nums[k] / (dens[k] + 1.0)
            f_norm[z, k] = nums_n[k] / (dens_n[k] + 1.0)
        for k in range(8):
            f_raw[z, 9 + k] = uu_nums[k] / (uu_dens[k] + 1.0)
            f_norm[z, 9 + k] = uu_nums_n[k] / (uu_dens_n[k] + 1.0)
        f_raw[z, 17] = nums[9] / (dens[9] + 1.0)
        f_raw[z, 18] = nums[10] / (dens[10] + 1.0)
        f_norm[z, 17] = nums_n[9] / (dens_n[9] + 1.0)
        f_norm[z, 18] = nums_n[10] / (dens_n[10] + 1.0)
        ii_neff = ii_sum_w * ii_sum_w / (ii_sum_w2 + 1e-08)
        ii_mean_e = ii_we / (ii_sum_w + 1e-08)
        ii_var_e = max(0.0, ii_we2 / (ii_sum_w + 1e-08) - ii_mean_e * ii_mean_e) * (ii_cnt / (ii_cnt + 2.0))
        ii_s_neff = ii_struct_w * ii_struct_w / (ii_struct_w2 + 1e-08)
        f_supp[z, 0] = ii_sum_w
        f_supp[z, 1] = ii_neff
        f_supp[z, 2] = ii_cnt
        f_supp[z, 3] = ii_max_sim
        f_supp[z, 4] = ii_t1 / (ii_sum_w + 1e-08)
        f_supp[z, 5] = (ii_t1 + ii_t2 + ii_t3 + ii_t4 + ii_t5) / (ii_sum_w + 1e-08)
        f_supp[z, 6] = ii_var_e
        f_supp[z, 7] = dens[5]
        f_supp[z, 8] = ii_s_neff
        uu_neff = uu_sum_w * uu_sum_w / (uu_sum_w2 + 1e-08)
        uu_mean_e = uu_we / (uu_sum_w + 1e-08)
        uu_var_e = max(0.0, uu_we2 / (uu_sum_w + 1e-08) - uu_mean_e * uu_mean_e) * (uu_cnt / (uu_cnt + 2.0))
        uu_s_neff = uu_struct_w * uu_struct_w / (uu_struct_w2 + 1e-08)
        f_supp[z, 9] = uu_sum_w
        f_supp[z, 10] = uu_neff
        f_supp[z, 11] = uu_cnt
        f_supp[z, 12] = uu_max_sim
        f_supp[z, 13] = uu_t1 / (uu_sum_w + 1e-08)
        f_supp[z, 14] = (uu_t1 + uu_t2 + uu_t3 + uu_t4 + uu_t5) / (uu_sum_w + 1e-08)
        f_supp[z, 15] = uu_var_e
        f_supp[z, 16] = uu_dens[5]
        f_supp[z, 17] = uu_s_neff
    return (f_raw, f_norm, f_supp)

@ratings_njit(cache=True)
def ratings_task3_session_and_kernel_features(q_ids, q_times, ptr, sorted_times, sorted_res, taus, shrinks, is_user):
    """Computes multi-scale exponential kernel residuals for taus = [60s, 600s, 3600s, 86400s, 7d, 30d, 365d, 3650d]
    separating past (t_j < t), future (t_j > t), and combined, outputting smoothed mean, numerator, support sum_w, and n_eff.
    When is_user=True, also extracts exact session features:
      - same-second, same-minute, same-hour, same-day residual means (2 shrinkages) and counts
      - nearest past and nearest future rating residual, log1p(dt), damped residual, and presence flags
      - within-1-hour and within-1-day residual standard deviation
      - user temporal rank quantile rank_u(t) / (L_u + 1)
    """
    n = len(q_ids)
    n_taus = len(taus)
    n_kcols = n_taus * 12
    n_scols = 27 if is_user else 1
    out = ratings_np.zeros((n, n_kcols + n_scols), dtype=ratings_np.float64)
    for idx in range(n):
        e = q_ids[idx]
        t = q_times[idx]
        if e >= len(ptr) - 1:
            continue
        start, end = (ptr[e], ptr[e + 1])
        L = end - start
        if L == 0:
            continue
        pos = ratings_np.searchsorted(sorted_times[start:end], t) + start
        out[idx, n_kcols] = float(pos - start + 1) / float(L + 1)
        left = max(start, pos - 400)
        right = min(end, pos + 400)
        for m in range(n_taus):
            tau = taus[m]
            sh = shrinks[m]
            w_past, w2_past, r_past = (0.0, 0.0, 0.0)
            w_fut, w2_fut, r_fut = (0.0, 0.0, 0.0)
            w_all, w2_all, r_all = (0.0, 0.0, 0.0)
            for p in range(left, right):
                tp = sorted_times[p]
                rp = sorted_res[p]
                dt = float(abs(t - tp))
                w = ratings_np.exp(-dt / tau)
                w2 = w * w
                wr = w * rp
                w_all += w
                w2_all += w2
                r_all += wr
                if tp < t:
                    w_past += w
                    w2_past += w2
                    r_past += wr
                elif tp > t:
                    w_fut += w
                    w2_fut += w2
                    r_fut += wr
            base_c = m * 12
            out[idx, base_c + 0] = r_past / (w_past + sh)
            out[idx, base_c + 1] = r_fut / (w_fut + sh)
            out[idx, base_c + 2] = r_all / (w_all + sh)
            out[idx, base_c + 3] = r_all / (w_all + 0.35 * sh)
            out[idx, base_c + 4] = r_past / (1.0 + 0.2 * w_past)
            out[idx, base_c + 5] = r_fut / (1.0 + 0.2 * w_fut)
            out[idx, base_c + 6] = ratings_np.log1p(w_past)
            out[idx, base_c + 7] = ratings_np.log1p(w_fut)
            out[idx, base_c + 8] = ratings_np.log1p(w_all)
            out[idx, base_c + 9] = w_past * w_past / (w2_past + 1e-08)
            out[idx, base_c + 10] = w_fut * w_fut / (w2_fut + 1e-08)
            out[idx, base_c + 11] = w_all * w_all / (w2_all + 1e-08)
        if not is_user:
            continue
        s_sec_r, s_sec_c = (0.0, 0.0)
        s_min_r, s_min_c = (0.0, 0.0)
        s_hr_r, s_hr_r2, s_hr_c = (0.0, 0.0, 0.0)
        s_day_r, s_day_r2, s_day_c = (0.0, 0.0, 0.0)
        best_prev_dt = 1e+18
        best_prev_r = 0.0
        has_prev = 0.0
        best_next_dt = 1e+18
        best_next_r = 0.0
        has_next = 0.0
        best_any_dt = 1e+18
        best_any_r = 0.0
        for p in range(left, right):
            tp = sorted_times[p]
            rp = sorted_res[p]
            dt = abs(t - tp)
            fdt = float(dt)
            if fdt < best_any_dt:
                best_any_dt = fdt
                best_any_r = rp
            if tp < t:
                dt_p = float(t - tp)
                if dt_p < best_prev_dt:
                    best_prev_dt = dt_p
                    best_prev_r = rp
                    has_prev = 1.0
            elif tp > t:
                dt_n = float(tp - t)
                if dt_n < best_next_dt:
                    best_next_dt = dt_n
                    best_next_r = rp
                    has_next = 1.0
            if dt == 0:
                s_sec_r += rp
                s_sec_c += 1.0
            if dt <= 60:
                s_min_r += rp
                s_min_c += 1.0
            if dt <= 3600:
                s_hr_r += rp
                s_hr_r2 += rp * rp
                s_hr_c += 1.0
            if dt <= 86400:
                s_day_r += rp
                s_day_r2 += rp * rp
                s_day_c += 1.0
        sc = n_kcols + 1
        out[idx, sc + 0] = s_sec_r / (s_sec_c + 0.5)
        out[idx, sc + 1] = s_sec_r / (s_sec_c + 2.0)
        out[idx, sc + 2] = ratings_np.log1p(s_sec_c)
        out[idx, sc + 3] = s_min_r / (s_min_c + 0.5)
        out[idx, sc + 4] = s_min_r / (s_min_c + 2.0)
        out[idx, sc + 5] = ratings_np.log1p(s_min_c)
        out[idx, sc + 6] = s_hr_r / (s_hr_c + 0.6)
        out[idx, sc + 7] = s_hr_r / (s_hr_c + 2.5)
        out[idx, sc + 8] = ratings_np.log1p(s_hr_c)
        out[idx, sc + 9] = s_day_r / (s_day_c + 0.8)
        out[idx, sc + 10] = s_day_r / (s_day_c + 3.0)
        out[idx, sc + 11] = ratings_np.log1p(s_day_c)
        if s_hr_c >= 2.0:
            hr_m = s_hr_r / s_hr_c
            hr_v = max(0.0, s_hr_r2 / s_hr_c - hr_m * hr_m)
            out[idx, sc + 12] = ratings_np.sqrt(hr_v) * (s_hr_c / (s_hr_c + 2.0))
        if s_day_c >= 2.0:
            day_m = s_day_r / s_day_c
            day_v = max(0.0, s_day_r2 / s_day_c - day_m * day_m)
            out[idx, sc + 13] = ratings_np.sqrt(day_v) * (s_day_c / (s_day_c + 2.0))
        out[idx, sc + 14] = has_prev
        if has_prev > 0.0:
            out[idx, sc + 15] = best_prev_r
            out[idx, sc + 16] = ratings_np.log1p(best_prev_dt)
            out[idx, sc + 17] = best_prev_r / (1.0 + ratings_np.sqrt(best_prev_dt / 3600.0))
            out[idx, sc + 18] = best_prev_r / (1.0 + best_prev_dt / 86400.0)
        else:
            out[idx, sc + 16] = 20.0
        out[idx, sc + 19] = has_next
        if has_next > 0.0:
            out[idx, sc + 20] = best_next_r
            out[idx, sc + 21] = ratings_np.log1p(best_next_dt)
            out[idx, sc + 22] = best_next_r / (1.0 + ratings_np.sqrt(best_next_dt / 3600.0))
            out[idx, sc + 23] = best_next_r / (1.0 + best_next_dt / 86400.0)
        else:
            out[idx, sc + 21] = 20.0
        if best_any_dt < 1e+17:
            out[idx, sc + 24] = best_any_r / (1.0 + ratings_np.sqrt(best_any_dt / 600.0))
            out[idx, sc + 25] = ratings_np.log1p(best_any_dt)
        else:
            out[idx, sc + 25] = 20.0
    return out

def ratings_fit_user_temporal_drift(a_rated, nu, res, u_pos_cnt):
    """Fits per-user temporal drift on a_rated:
    d_u(t) = sign(t - t_bar_u) * |(t - t_bar_u) / 86400|^0.4
    alpha_u = sum_j d_u(t_uj) e_uj / (lambda + sum_j d_u(t_uj)^2), lambda = 10.0 (and 30.0).
    """
    u_idx = a_rated[:, 0]
    t_vals = a_rated[:, 2].astype(ratings_np.float64)
    t_sum = ratings_np.bincount(u_idx, weights=t_vals, minlength=nu)
    t_bar = ratings_np.where(u_pos_cnt > 0, t_sum / ratings_np.maximum(u_pos_cnt, 1), 0.0)
    dt_days = (t_vals - t_bar[u_idx]) / 86400.0
    d_uj = ratings_np.sign(dt_days) * ratings_np.abs(dt_days) ** 0.4
    num = ratings_np.bincount(u_idx, weights=d_uj * res.astype(ratings_np.float64), minlength=nu)
    den = ratings_np.bincount(u_idx, weights=d_uj * d_uj, minlength=nu)
    alpha_10 = num / (den + 10.0)
    alpha_30 = num / (den + 30.0)
    return (t_bar, alpha_10, alpha_30)

def ratings_eval_user_temporal_drift(q_u, q_t, nu, t_bar, alpha_10, alpha_30, u_pos_cnt):
    u_valid = q_u < nu
    u_safe = ratings_np.where(u_valid, q_u, 0)
    has_pos = u_valid & (u_pos_cnt[u_safe] > 0)
    dt_days = ratings_np.where(has_pos, (q_t.astype(ratings_np.float64) - t_bar[u_safe]) / 86400.0, 0.0)
    d_ut = ratings_np.clip(ratings_np.sign(dt_days) * ratings_np.abs(dt_days) ** 0.4, -15.0, 15.0)
    a10 = ratings_np.where(has_pos, alpha_10[u_safe], 0.0)
    a30 = ratings_np.where(has_pos, alpha_30[u_safe], 0.0)
    drift_pred10 = a10 * d_ut
    drift_pred30 = a30 * d_ut
    return ratings_np.column_stack([drift_pred10, drift_pred30, d_ut, a10])

def ratings_fit_rating_base(fit_rows, nu, ni, task, tune_rows=None, seed=ratings_SEED_PRIMARY, fixed_epochs=None, run_svdpp_grid=True):
    """Fits all rating supervision models strictly on a_rated = fit_rows[fit_rows[:, 3] > 0],
    while preserving a_implicit = fit_rows for implicit graph counts, zero rates, spectral SVD, and SVD++ N(u).
    """
    assert fixed_epochs is not None and tune_rows is None and (not run_svdpp_grid), 'Production fits require frozen epochs and no tuning rows'
    t0 = ratings_time.time()
    a_implicit = fit_rows
    a_rated = fit_rows[fit_rows[:, 3] > 0]
    tune_pos = None
    if tune_rows is not None:
        tune_pos = tune_rows[tune_rows[:, 3] > 0]
    u_imp_cnt = ratings_np.bincount(a_implicit[:, 0], minlength=nu)
    i_imp_cnt = ratings_np.bincount(a_implicit[:, 1], minlength=ni)
    u_pos_cnt = ratings_np.bincount(a_rated[:, 0], minlength=nu)
    i_pos_cnt = ratings_np.bincount(a_rated[:, 1], minlength=ni)
    mu0, bu0, bi0 = als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 1.5, 2.5, n_iter=20)
    mu1, bu1, bi1 = als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 3.0, 5.0, n_iter=20)
    mu2, bu2, bi2 = als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 8.0, 12.0, n_iter=20)
    mu3, bu3, bi3 = als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 25.0, 35.0, n_iter=20)
    mu4, bu4, bi4 = als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 60.0, 80.0, n_iter=20)
    res = (a_rated[:, 3] - (mu2 + bu2[a_rated[:, 0]] + bi2[a_rated[:, 1]])).astype(ratings_np.float32)
    prior_var = float(ratings_np.mean(res ** 2))
    u_var = ratings_np.bincount(a_rated[:, 0], weights=res ** 2, minlength=nu)
    u_std = ratings_np.append(ratings_np.sqrt((u_var + 5.0 * prior_var) / (u_pos_cnt + 5.0)), ratings_np.sqrt(prior_var)).astype(ratings_np.float64)
    i_var = ratings_np.bincount(a_rated[:, 1], weights=res ** 2, minlength=ni)
    i_std = ratings_np.append(ratings_np.sqrt((i_var + 5.0 * prior_var) / (i_pos_cnt + 5.0)), ratings_np.sqrt(prior_var)).astype(ratings_np.float64)
    res_n = (res / (u_std[a_rated[:, 0]] + 0.1)).astype(ratings_np.float32)
    u_zero = u_imp_cnt - u_pos_cnt
    i_zero = i_imp_cnt - i_pos_cnt
    p0_global = float(ratings_np.sum(u_zero) / max(len(a_implicit), 1))
    pu_z = ratings_np.append((u_zero + 2.0 * p0_global) / (u_imp_cnt + 2.0), p0_global)
    pi_z = ratings_np.append((i_zero + 10.0 * p0_global) / (i_imp_cnt + 10.0), p0_global)
    u_order = ratings_np.lexsort((a_rated[:, 2], a_rated[:, 0]))
    u_pos_ptr = ratings_np.r_[0, u_pos_cnt.cumsum()].astype(ratings_np.int32)
    u_pos_items = a_rated[u_order, 1].astype(ratings_np.int32)
    u_pos_times = a_rated[u_order, 2].astype(ratings_np.int64)
    u_pos_ratings = a_rated[u_order, 3].astype(ratings_np.float32)
    u_pos_res = res[u_order]
    u_pos_resn = res_n[u_order]
    i_order = ratings_np.lexsort((a_rated[:, 2], a_rated[:, 1]))
    i_pos_ptr = ratings_np.r_[0, i_pos_cnt.cumsum()].astype(ratings_np.int32)
    i_pos_users = a_rated[i_order, 0].astype(ratings_np.int32)
    i_pos_times = a_rated[i_order, 2].astype(ratings_np.int64)
    i_pos_res = res[i_order]
    i_pos_resn = res_n[i_order]
    u_imp_order = ratings_np.lexsort((a_implicit[:, 2], a_implicit[:, 0]))
    u_imp_ptr = ratings_np.r_[0, u_imp_cnt.cumsum()].astype(ratings_np.int32)
    u_imp_items = a_implicit[u_imp_order, 1].astype(ratings_np.int32)
    u_imp_times = a_implicit[u_imp_order, 2].astype(ratings_np.int64)
    ep_mf32 = fixed_epochs['mf32'] if fixed_epochs else 28
    mf32, best_ep_mf32, mse_mf32 = ratings_train_explicit_pos_mf(a_rated, nu, ni, k=32, lr0=0.012, decay=0.97, reg_b=0.02, reg_pq=0.055, max_epochs=ep_mf32, bu_init=None, bi_init=None, tune_pos=tune_pos if not fixed_epochs else None, seed=seed)
    ep_mf64 = fixed_epochs['mf64'] if fixed_epochs else 25
    mf64, best_ep_mf64, mse_mf64 = ratings_train_explicit_pos_mf(a_rated, nu, ni, k=64, lr0=0.015, decay=0.96, reg_b=0.03, reg_pq=0.07, max_epochs=ep_mf64, bu_init=bu2, bi_init=bi2, tune_pos=tune_pos if not fixed_epochs else None, seed=seed + 1)
    svdpp_grid_results = {}
    ep_s32 = fixed_epochs['svdpp32'] if fixed_epochs else 25
    best_alpha32 = fixed_epochs.get('alpha32', 0.0) if fixed_epochs else 0.0
    lr32 = 0.009 if best_alpha32 == 0.0 else 0.025 if best_alpha32 == 0.5 else 0.06
    svdpp32, best_ep_svdpp32, mse_svdpp32 = ratings_train_user_block_svdpp(u_pos_ptr, u_pos_items, u_pos_ratings, u_imp_ptr, u_imp_items, nu, ni, mu2, bu2, bi2, k=32, alpha=best_alpha32, lr_b0=0.05, lr0=lr32, decay=0.96, reg_b=0.08, reg_p=0.06, reg_q=0.06, reg_y=0.06, ep_shrink=0.002, max_epochs=ep_s32, tune_pos=None, seed=seed + 32)
    if tune_pos is not None:
        mse_svdpp32 = float(ratings_np.mean((svdpp32.predict(tune_pos[:, 0], tune_pos[:, 1]) - tune_pos[:, 3]) ** 2))
    ep_s64 = fixed_epochs['svdpp64'] if fixed_epochs else 25
    best_alpha64 = fixed_epochs.get('alpha64', 0.0) if fixed_epochs else 0.0
    lr64 = 0.009 if best_alpha64 == 0.0 else 0.025 if best_alpha64 == 0.5 else 0.06
    svdpp64, best_ep_svdpp64, mse_svdpp64 = ratings_train_user_block_svdpp(u_pos_ptr, u_pos_items, u_pos_ratings, u_imp_ptr, u_imp_items, nu, ni, mu2, bu2, bi2, k=64, alpha=best_alpha64, lr_b0=0.05, lr0=lr64, decay=0.96, reg_b=0.08, reg_p=0.07, reg_q=0.07, reg_y=0.07, ep_shrink=0.002, max_epochs=ep_s64, tune_pos=None, seed=seed + 64)
    if tune_pos is not None:
        mse_svdpp64 = float(ratings_np.mean((svdpp64.predict(tune_pos[:, 0], tune_pos[:, 1]) - tune_pos[:, 3]) ** 2))
    t_bar, alpha_10, alpha_30 = (None, None, None)
    if task == 3:
        t_bar, alpha_10, alpha_30 = ratings_fit_user_temporal_drift(a_rated, nu, res, u_pos_cnt)
    return {'task': task, 'nu': nu, 'ni': ni, 'seed': seed, 'a_rated': a_rated, 'a_implicit': a_implicit, 'u_imp_cnt': u_imp_cnt, 'i_imp_cnt': i_imp_cnt, 'u_pos_cnt': u_pos_cnt, 'i_pos_cnt': i_pos_cnt, 'biases': ((mu0, bu0, bi0), (mu1, bu1, bi1), (mu2, bu2, bi2), (mu3, bu3, bi3), (mu4, bu4, bi4)), 'res': res, 'res_n': res_n, 'u_std': u_std, 'i_std': i_std, 'pu_z': pu_z, 'pi_z': pi_z, 'u_pos_ptr': u_pos_ptr, 'u_pos_items': u_pos_items, 'u_pos_times': u_pos_times, 'u_pos_ratings': u_pos_ratings, 'u_pos_res': u_pos_res, 'u_pos_resn': u_pos_resn, 'i_pos_ptr': i_pos_ptr, 'i_pos_users': i_pos_users, 'i_pos_times': i_pos_times, 'i_pos_res': i_pos_res, 'i_pos_resn': i_pos_resn, 'u_imp_ptr': u_imp_ptr, 'u_imp_items': u_imp_items, 'u_imp_times': u_imp_times, 'mf32': mf32, 'mf64': mf64, 'svdpp32': svdpp32, 'svdpp64': svdpp64, 'best_epochs': {'mf32': int(best_ep_mf32), 'mf64': int(best_ep_mf64), 'svdpp32': int(best_ep_svdpp32), 'svdpp64': int(best_ep_svdpp64), 'alpha32': float(best_alpha32), 'alpha64': float(best_alpha64)}, 'base_tune_mses': {'mf32_tune_pos_mse': float(mse_mf32), 'mf64_tune_pos_mse': float(mse_mf64), 'svdpp32_tune_pos_mse': float(mse_svdpp32), 'svdpp64_tune_pos_mse': float(mse_svdpp64)}, 'svdpp_grid_results': svdpp_grid_results, 't_bar': t_bar, 'alpha_10': alpha_10, 'alpha_30': alpha_30, 'fit_elapsed_sec': round(ratings_time.time() - t0, 2)}

def ratings_build_rating_features(state, q_u, q_i, q_t):
    """Builds the complete feature matrix for queries (q_u, q_i, q_t) and returns
    (X, col_groups, diagnostics) to support exact progressive stage ablations.
    """
    task = state['task']
    nu, ni = (state['nu'], state['ni'])
    n = len(q_u)
    (mu0, bu0, bi0), (mu1, bu1, bi1), (mu2, bu2, bi2), (mu3, bu3, bi3), (mu4, bu4, bi4) = state['biases']
    bu0_e, bi0_e = (ratings_np.append(bu0, 0.0), ratings_np.append(bi0, 0.0))
    bu1_e, bi1_e = (ratings_np.append(bu1, 0.0), ratings_np.append(bi1, 0.0))
    bu2_e, bi2_e = (ratings_np.append(bu2, 0.0), ratings_np.append(bi2, 0.0))
    bu3_e, bi3_e = (ratings_np.append(bu3, 0.0), ratings_np.append(bi3, 0.0))
    bu4_e, bi4_e = (ratings_np.append(bu4, 0.0), ratings_np.append(bi4, 0.0))
    base0 = mu0 + bu0_e[q_u] + bi0_e[q_i]
    base1 = mu1 + bu1_e[q_u] + bi1_e[q_i]
    base2 = mu2 + bu2_e[q_u] + bi2_e[q_i]
    base3 = mu3 + bu3_e[q_u] + bi3_e[q_i]
    base4 = mu4 + bu4_e[q_u] + bi4_e[q_i]
    u_pos_cnt_e = ratings_np.append(state['u_pos_cnt'], 0)
    i_pos_cnt_e = ratings_np.append(state['i_pos_cnt'], 0)
    u_imp_cnt_e = ratings_np.append(state['u_imp_cnt'], 0)
    i_imp_cnt_e = ratings_np.append(state['i_imp_cnt'], 0)
    log_upos = ratings_np.log1p(u_pos_cnt_e[q_u])
    log_ipos = ratings_np.log1p(i_pos_cnt_e[q_i])
    log_uimp = ratings_np.log1p(u_imp_cnt_e[q_u])
    log_iimp = ratings_np.log1p(i_imp_cnt_e[q_i])
    u_no_pos = (u_pos_cnt_e[q_u] == 0).astype(ratings_np.float64)
    i_no_pos = (i_pos_cnt_e[q_i] == 0).astype(ratings_np.float64)
    u_no_imp = (u_imp_cnt_e[q_u] == 0).astype(ratings_np.float64)
    i_no_imp = (i_imp_cnt_e[q_i] == 0).astype(ratings_np.float64)
    u_std_q = state['u_std'][q_u]
    i_std_q = state['i_std'][q_i]
    pu_z_q = state['pu_z'][q_u]
    pi_z_q = state['pi_z'][q_i]
    mf32_pred, mf32_dot = state['mf32'].predict_components(q_u, q_i)
    mf64_pred, mf64_dot = state['mf64'].predict_components(q_u, q_i)
    mf32_res = mf32_pred - base2
    mf64_res = mf64_pred - base2
    f_spec = ratings_compute_spectral_features(state['a_rated'], state['a_implicit'], nu, ni, state['u_pos_cnt'], state['i_pos_cnt'], state['u_imp_cnt'], state['i_imp_cnt'], state['res'], state['res_n'], state['u_std'], q_u, q_i, seed=state['seed'])
    f_2hop_raw = exact_2hop_uu_ii(q_u, q_i, q_t, task == 3, state['u_pos_ptr'], state['u_pos_items'], state['u_pos_times'], state['u_pos_res'], state['i_pos_ptr'], state['i_pos_users'], state['i_pos_times'], state['i_pos_res'], state['u_imp_cnt'], state['i_imp_cnt']).astype(ratings_np.float64)
    f_2hop_use = f_2hop_raw[:, :11] if task == 1 else f_2hop_raw
    stage1_list = [ratings_np.ones(n, dtype=ratings_np.float64), base0, base1, base2, base3, base4, bu2_e[q_u], bi2_e[q_i], pu_z_q, pi_z_q, base2 * pu_z_q, base2 * pi_z_q, u_no_pos, i_no_pos, u_no_imp, i_no_imp, log_upos, log_ipos, log_uimp, log_iimp, mf32_pred, mf64_pred, mf32_dot, mf64_dot, f_spec, f_2hop_use, u_std_q * bi2_e[q_i], i_std_q * bu2_e[q_u], u_std_q * f_2hop_raw[:, 0], u_std_q * f_2hop_raw[:, 6], u_std_q * mf32_res, u_std_q * mf64_res, i_std_q * f_2hop_raw[:, 0], i_std_q * f_2hop_raw[:, 6], i_std_q * mf32_res, i_std_q * mf64_res]
    if task == 3:
        taus_7 = ratings_np.array([600.0, 3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0, 3650 * 86400.0])
        u_sh_7 = ratings_np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
        i_sh_7 = ratings_np.array([3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0])
        u_tk7 = temporal_kernel_residuals(q_u, q_t, state['u_pos_ptr'], state['u_pos_times'], state['u_pos_res'].astype(ratings_np.float64), taus_7, u_sh_7)
        i_tk7 = temporal_kernel_residuals(q_i, q_t, state['i_pos_ptr'], state['i_pos_times'], state['i_pos_res'].astype(ratings_np.float64), taus_7, i_sh_7)
        stage1_list.extend([u_tk7, i_tk7, u_std_q[:, None] * u_tk7[:, 1:4]])
    x_stage1 = ratings_np.column_stack(stage1_list)
    svdpp32_pred, svdpp32_dp, svdpp32_dh = state['svdpp32'].predict_components(q_u, q_i)
    svdpp64_pred, svdpp64_dp, svdpp64_dh = state['svdpp64'].predict_components(q_u, q_i)
    svdpp32_res = svdpp32_pred - base2
    svdpp64_res = svdpp64_pred - base2
    x_stage2_only = ratings_np.column_stack([svdpp32_pred, svdpp64_pred, svdpp32_dp, svdpp32_dh, svdpp64_dp, svdpp64_dh, u_std_q * svdpp32_res, i_std_q * svdpp32_res, u_std_q * svdpp64_res, i_std_q * svdpp64_res])
    f_knn_raw, f_knn_norm, f_knn_supp = ratings_fused_knn_and_support(q_u, q_i, state['u_pos_ptr'], state['u_pos_items'], state['u_pos_res'], state['u_pos_resn'], state['i_pos_ptr'], state['i_pos_users'], state['i_pos_res'], state['i_pos_resn'], state['u_pos_cnt'], state['i_pos_cnt'], state['u_imp_cnt'], state['i_imp_cnt'])
    f_knn_norm_rescaled = f_knn_norm * u_std_q[:, None]
    f_2hop_norm = exact_2hop_uu_ii(q_u, q_i, q_t, task == 3, state['u_pos_ptr'], state['u_pos_items'], state['u_pos_times'], state['u_pos_resn'], state['i_pos_ptr'], state['i_pos_users'], state['i_pos_times'], state['i_pos_resn'], state['u_imp_cnt'], state['i_imp_cnt']).astype(ratings_np.float64)
    f_2hop_norm_use = (f_2hop_norm[:, :11] if task == 1 else f_2hop_norm) * u_std_q[:, None]
    ii_sum_w = f_knn_supp[:, 0]
    ii_neff = f_knn_supp[:, 1]
    ii_cnt = f_knn_supp[:, 2]
    ii_max_sim = f_knn_supp[:, 3]
    ii_top1_c = f_knn_supp[:, 4]
    ii_top5_c = f_knn_supp[:, 5]
    ii_var_e = f_knn_supp[:, 6]
    ii_s_neff = f_knn_supp[:, 8]
    uu_sum_w = f_knn_supp[:, 9]
    uu_neff = f_knn_supp[:, 10]
    uu_cnt = f_knn_supp[:, 11]
    uu_max_sim = f_knn_supp[:, 12]
    uu_top1_c = f_knn_supp[:, 13]
    uu_top5_c = f_knn_supp[:, 14]
    uu_var_e = f_knn_supp[:, 15]
    uu_s_neff = f_knn_supp[:, 17]
    f_supp_clean = ratings_np.column_stack([ratings_np.log1p(ii_sum_w), ratings_np.log1p(ii_neff), ratings_np.log1p(ii_cnt), ii_max_sim, ii_top1_c, ii_top5_c, ii_var_e, ratings_np.log1p(ii_s_neff), ratings_np.log1p(uu_sum_w), ratings_np.log1p(uu_neff), ratings_np.log1p(uu_cnt), uu_max_sim, uu_top1_c, uu_top5_c, uu_var_e, ratings_np.log1p(uu_s_neff)])
    x_stage3_only = ratings_np.column_stack([f_knn_raw, f_knn_norm, f_knn_norm_rescaled, f_2hop_norm_use, f_supp_clean])
    if task == 3:
        taus_8 = ratings_np.array([60.0, 600.0, 3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0, 3650 * 86400.0])
        u_sh_8 = ratings_np.array([0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
        i_sh_8 = ratings_np.array([2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0])
        u_sess_raw = ratings_task3_session_and_kernel_features(q_u, q_t, state['u_pos_ptr'], state['u_pos_times'], state['u_pos_res'].astype(ratings_np.float64), taus_8, u_sh_8, True)
        u_sess_norm = ratings_task3_session_and_kernel_features(q_u, q_t, state['u_pos_ptr'], state['u_pos_times'], state['u_pos_resn'].astype(ratings_np.float64), taus_8, u_sh_8, False)
        i_sess_raw = ratings_task3_session_and_kernel_features(q_i, q_t, state['i_pos_ptr'], state['i_pos_times'], state['i_pos_res'].astype(ratings_np.float64), taus_8, i_sh_8, False)
        u_norm_comb = ratings_np.column_stack([u_sess_norm[:, m * 12 + 2] * u_std_q for m in range(8)])
        for m in range(8):
            for c_off in (9, 10, 11):
                u_sess_raw[:, m * 12 + c_off] = ratings_np.log1p(u_sess_raw[:, m * 12 + c_off])
                i_sess_raw[:, m * 12 + c_off] = ratings_np.log1p(i_sess_raw[:, m * 12 + c_off])
        f_drift = ratings_eval_user_temporal_drift(q_u, q_t, nu, state['t_bar'], state['alpha_10'], state['alpha_30'], state['u_pos_cnt'])
        imp_rank_q = ratings_np.zeros(n, dtype=ratings_np.float64)
        u_imp_ptr = state['u_imp_ptr']
        u_imp_times = state['u_imp_times']
        for idx in range(n):
            u = q_u[idx]
            if u < nu:
                s, e = (u_imp_ptr[u], u_imp_ptr[u + 1])
                if e > s:
                    p_pos = ratings_np.searchsorted(u_imp_times[s:e], q_t[idx])
                    imp_rank_q[idx] = float(p_pos + 1) / float(e - s + 1)
        x_stage4_only = ratings_np.column_stack([u_sess_raw, u_norm_comb, i_sess_raw, f_drift, imp_rank_q])
    else:
        x_stage4_only = ratings_np.empty((n, 0), dtype=ratings_np.float64)
    log_ii_neff = ratings_np.log1p(ii_neff)
    log_uu_neff = ratings_np.log1p(uu_neff)
    gate_ii = ii_neff / (ii_neff + 5.0)
    gate_uu = uu_neff / (uu_neff + 5.0)
    gate_ii_s = ii_s_neff / (ii_s_neff + 10.0)
    gate_uu_s = uu_s_neff / (uu_s_neff + 10.0)
    f_neigh_ii = f_knn_raw[:, 1]
    f_neigh_uu = f_knn_raw[:, 10]
    f_2hop_ii = f_2hop_raw[:, 0]
    f_2hop_uu = f_2hop_raw[:, 6]
    gating_cols = [f_neigh_ii * log_ii_neff, f_neigh_uu * log_uu_neff, f_neigh_ii * gate_ii, f_neigh_uu * gate_uu, f_2hop_ii * ratings_np.log1p(ii_s_neff), f_2hop_uu * ratings_np.log1p(uu_s_neff), f_2hop_ii * gate_ii_s, f_2hop_uu * gate_uu_s, f_neigh_ii / (1.0 + ii_var_e), f_neigh_uu / (1.0 + uu_var_e), mf32_res * log_upos, mf32_res * log_ipos, mf64_res * log_upos, mf64_res * log_ipos, svdpp32_res * log_uimp, svdpp32_res * log_iimp, svdpp64_res * log_uimp, svdpp64_res * log_iimp, svdpp32_pred - mf32_pred, ratings_np.abs(svdpp32_pred - mf32_pred), svdpp64_pred - mf64_pred, ratings_np.abs(svdpp64_pred - mf64_pred), ratings_np.abs(mf64_res), ratings_np.abs(svdpp64_res), ratings_np.abs(f_neigh_ii - f_neigh_uu)]
    if task == 3:
        for m in range(4):
            sess_mean = u_sess_raw[:, m * 12 + 2]
            sess_log_neff = u_sess_raw[:, m * 12 + 11]
            gating_cols.append(sess_mean * sess_log_neff)
            gating_cols.append(sess_mean * u_std_q)
    x_stage5_only = ratings_np.column_stack(gating_cols)
    c0 = 52
    c1 = x_stage1.shape[1]
    c2 = c1 + x_stage2_only.shape[1]
    c3 = c2 + x_stage3_only.shape[1]
    c4 = c3 + x_stage4_only.shape[1]
    c5 = c4 + x_stage5_only.shape[1]
    X_full = ratings_np.column_stack([x_stage1, x_stage2_only, x_stage3_only, x_stage4_only, x_stage5_only])
    col_groups = {}
    if task == 3:
        col_groups['stage0_static_no_time'] = slice(0, c0)
    col_groups.update({'stage1_pos_base': slice(0, c1), 'stage2_plus_svdpp': slice(0, c2), 'stage3_plus_knn_support': slice(0, c3), 'stage4_plus_session_drift': slice(0, c4), 'stage5_plus_conf_gating': slice(0, c5)})
    return (X_full, col_groups)

import time as ratings_time

import numpy as ratings_np

from numba import njit as ratings_njit

from scipy.special import expit as ratings_expit

ratings_V3_CONFIGS = {1: {'kind': 'ridge_hgb', 'reg': 500.0, 'eta': 1.0, 'hgb': {'max_leaf_nodes': 31, 'max_iter': 400, 'learning_rate': 0.04, 'min_samples_leaf': 250, 'l2_regularization': 100.0, 'early_stopping': False, 'random_state': 258}, 'tune_mse': 0.5306373790895934, 'use_stage': True}, 3: {'kind': 'ridge_hgb', 'reg': 500.0, 'eta': 0.5, 'hgb': {'max_leaf_nodes': 63, 'max_iter': 400, 'learning_rate': 0.04, 'min_samples_leaf': 250, 'l2_regularization': 100.0, 'early_stopping': False, 'random_state': 258}, 'tune_mse': 0.6600240009778012}}

@ratings_njit(cache=True)
def ratings_nk__summarize_neighbors(sim, values, adjusted):
    result = ratings_np.zeros(42)
    col = 0
    for variant in range(3):
        for positive in range(2):
            w = sim[variant].copy()
            if positive:
                for j in range(len(w)):
                    if w[j] < 0:
                        w[j] = 0.0
            order = ratings_np.argsort(-ratings_np.abs(w))
            for k_index in range(3):
                k = (5, 20, len(w))[k_index]
                numerator = 0.0
                denominator = 0.0
                adjust = 0.0
                for j in order[:min(k, len(order))]:
                    numerator += w[j] * values[j]
                    denominator += abs(w[j])
                    adjust += w[j] * adjusted[j]
                result[col] = numerator / (denominator + 0.1)
                col += 1
                result[col] = numerator / (denominator + 1.0)
                col += 1
                if variant == 1:
                    result[36 + positive * 3 + k_index] = adjust / (denominator + 1.0)
    return result

@ratings_njit(cache=True)
def ratings_nk_topk_residual_features(qu, qi, up, ui, ur, ip, iu, ir, ue, ie):
    nu = len(up) - 1
    ni = len(ip) - 1
    out = ratings_np.zeros((len(qu), 84))
    present = ratings_np.zeros(nu, ratings_np.bool_)
    target = ratings_np.zeros(nu)
    dot = ratings_np.zeros(nu)
    sx = ratings_np.zeros(nu)
    sy = ratings_np.zeros(nu)
    xx = ratings_np.zeros(nu)
    yy = ratings_np.zeros(nu)
    count = ratings_np.zeros(nu)
    for z in range(len(qu)):
        u = qu[z]
        i = qi[z]
        if u >= nu or i >= ni:
            continue
        us, uep = (up[u], up[u + 1])
        is_, iep = (ip[i], ip[i + 1])
        if us == uep or is_ == iep:
            continue
        for p in range(is_, iep):
            v = iu[p]
            present[v] = True
            target[v] = ir[p]
        n_h = uep - us
        sims = ratings_np.zeros((3, n_h))
        values = ratings_np.zeros(n_h)
        adjusted = ratings_np.zeros(n_h)
        for n, p in enumerate(range(us, uep)):
            j = ui[p]
            ru = ur[p]
            values[n] = ru
            if j == i:
                continue
            ds = 0.0
            xs = 0.0
            ys = 0.0
            x2 = 0.0
            y2 = 0.0
            nc = 0.0
            for pos in range(ip[j], ip[j + 1]):
                v = iu[pos]
                if present[v] and v != u:
                    a = target[v]
                    b = ir[pos]
                    ds += a * b
                    xs += a
                    ys += b
                    x2 += a * a
                    y2 += b * b
                    nc += 1
                    dot[v] += ru * b
                    sx[v] += ru
                    sy[v] += b
                    xx[v] += ru * ru
                    yy[v] += b * b
                    count[v] += 1
            if nc >= 3:
                sims[0, n] = ds / (ratings_np.sqrt(ie[i] * ie[j]) + 1e-08) * nc / (nc + 5)
                covariance = ds - xs * ys / nc
                variance = max(0.0, x2 - xs * xs / nc) * max(0.0, y2 - ys * ys / nc)
                sims[1, n] = covariance / (ratings_np.sqrt(variance) + 1e-08) * nc / (nc + 10)
                sims[2, n] = ratings_np.exp(-(x2 + y2 - 2 * ds) / (nc * 0.5)) * nc / (nc + 10)
                adjusted[n] = ru + (xs - ys) / (nc + 5)
        out[z, :42] = ratings_nk__summarize_neighbors(sims, values, adjusted)
        nr = iep - is_
        sims = ratings_np.zeros((3, nr))
        values = ratings_np.zeros(nr)
        adjusted = ratings_np.zeros(nr)
        for n, p in enumerate(range(is_, iep)):
            v = iu[p]
            values[n] = target[v]
            nc = count[v]
            if nc >= 3 and v != u:
                ds = dot[v]
                xs = sx[v]
                ys = sy[v]
                x2 = xx[v]
                y2 = yy[v]
                sims[0, n] = ds / (ratings_np.sqrt(ue[u] * ue[v]) + 1e-08) * nc / (nc + 5)
                covariance = ds - xs * ys / nc
                variance = max(0.0, x2 - xs * xs / nc) * max(0.0, y2 - ys * ys / nc)
                sims[1, n] = covariance / (ratings_np.sqrt(variance) + 1e-08) * nc / (nc + 10)
                sims[2, n] = ratings_np.exp(-(x2 + y2 - 2 * ds) / (nc * 0.5)) * nc / (nc + 10)
                adjusted[n] = target[v] + (xs - ys) / (nc + 5)
            present[v] = False
            target[v] = 0.0
            dot[v] = 0.0
            sx[v] = 0.0
            sy[v] = 0.0
            xx[v] = 0.0
            yy[v] = 0.0
            count[v] = 0.0
        out[z, 42:] = ratings_nk__summarize_neighbors(sims, values, adjusted)
    return out

def ratings_nk_extract_topk_features(state, qu, qi):
    a = state['a_rated']
    r = state['res']
    ue = ratings_np.bincount(a[:, 0], weights=r * r, minlength=state['nu'])
    ie = ratings_np.bincount(a[:, 1], weights=r * r, minlength=state['ni'])
    return ratings_nk_topk_residual_features(qu, qi, state['u_pos_ptr'], state['u_pos_items'], state['u_pos_res'], state['i_pos_ptr'], state['i_pos_users'], state['i_pos_res'], ue, ie)

def ratings_ord_fit_ordinal(fit, nu, ni, reg_u=3.0, reg_i=5.0, iterations=20):
    a = fit[fit[:, 3] > 0]
    uu, ii = (a[:, 0], a[:, 1])
    uc = ratings_np.bincount(uu, minlength=nu)
    ic = ratings_np.bincount(ii, minlength=ni)
    mu = ratings_np.zeros(4)
    bu = ratings_np.zeros((nu, 4))
    bi = ratings_np.zeros((ni, 4))
    for k in range(4):
        target = (a[:, 3] >= k + 2).astype(float)
        prior = target.mean()
        mu[k] = ratings_np.log(prior / (1 - prior))
        for it in range(iterations):
            prob = ratings_expit(mu[k] + bu[uu, k] + bi[ii, k])
            h = prob * (1 - prob)
            gu = ratings_np.bincount(uu, weights=target - prob, minlength=nu) - reg_u * bu[:, k]
            hu = ratings_np.bincount(uu, weights=h, minlength=nu) + reg_u
            bu[:, k] += ratings_np.clip(gu / hu, -1.0, 1.0)
            prob = ratings_expit(mu[k] + bu[uu, k] + bi[ii, k])
            h = prob * (1 - prob)
            gi = ratings_np.bincount(ii, weights=target - prob, minlength=ni) - reg_i * bi[:, k]
            hi = ratings_np.bincount(ii, weights=h, minlength=ni) + reg_i
            bi[:, k] += ratings_np.clip(gi / hi, -1.0, 1.0)
    return dict(mu=mu, bu=bu, bi=bi)

def ratings_ord_ordinal_features(state, qu, qi):
    bu = ratings_np.vstack([state['bu'], ratings_np.zeros((1, 4))])[qu]
    bi = ratings_np.vstack([state['bi'], ratings_np.zeros((1, 4))])[qi]
    prob = ratings_expit(state['mu'][None, :] + bu + bi)
    mean = 1 + prob.sum(axis=1)
    return ratings_np.column_stack([mean, prob, bu, bi])

ratings_jt_DAY = 86400.0

@ratings_njit(cache=True)
def ratings_jt_categorical_sweep(y, prediction, ids, effects, counts, penalty):
    sums = ratings_np.zeros(len(effects))
    for r in range(len(y)):
        k = ids[r]
        sums[k] += y[r] - prediction[r] + effects[k]
    newer = sums / (counts + penalty)
    for r in range(len(y)):
        k = ids[r]
        prediction[r] += newer[k] - effects[k]
    return newer

@ratings_njit(cache=True)
def ratings_jt_slope_sweep(y, prediction, users, z, slopes, counts_z2, penalty):
    sums = ratings_np.zeros(len(slopes))
    for r in range(len(y)):
        u = users[r]
        sums[u] += z[r] * (y[r] - prediction[r] + slopes[u] * z[r])
    newer = sums / (counts_z2 + penalty)
    for r in range(len(y)):
        u = users[r]
        prediction[r] += (newer[u] - slopes[u]) * z[r]
    return newer

def ratings_jt_time_code(ids, times, days):
    bins = ratings_np.clip(ratings_np.floor((times - 946684800) / (ratings_jt_DAY * days)), -10000, 10000).astype(ratings_np.int64)
    return ids.astype(ratings_np.int64) * 20001 + bins + 10000

def ratings_jt_make_group(ids, times, days):
    keys, inverse = ratings_np.unique(ratings_jt_time_code(ids, times, days), return_inverse=True)
    return (keys, inverse.astype(ratings_np.int32), ratings_np.bincount(inverse).astype(ratings_np.float64))

ratings_jt_DEFAULT_CONFIGS = [{'user_days': 1.0, 'item_days': 90.0, 'session_reg': 1.0, 'item_reg': 20.0, 'slope_reg': 20.0}, {'user_days': 1.0, 'item_days': 30.0, 'session_reg': 4.0, 'item_reg': 10.0, 'slope_reg': 80.0}, {'user_days': 7.0, 'item_days': 90.0, 'session_reg': 2.0, 'item_reg': 20.0, 'slope_reg': 20.0}, {'user_days': 30.0, 'item_days': 365.0, 'session_reg': 4.0, 'item_reg': 30.0, 'slope_reg': 80.0}, {'user_days': 90.0, 'item_days': 90.0, 'session_reg': 8.0, 'item_reg': 10.0, 'slope_reg': 20.0}, {'user_days': 365.0, 'item_days': 365.0, 'session_reg': 12.0, 'item_reg': 30.0, 'slope_reg': 80.0}]

def ratings_jt_fit_temporal_models(fit_rows, nu, ni, configs=None):
    a = fit_rows[fit_rows[:, 3] > 0]
    u, i, t, y = (a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(ratings_np.float64))
    uc = ratings_np.bincount(u, minlength=nu).astype(ratings_np.float64)
    ic = ratings_np.bincount(i, minlength=ni).astype(ratings_np.float64)
    tc = ratings_np.bincount(u, weights=t.astype(float), minlength=nu) / ratings_np.maximum(uc, 1)
    raw = ratings_np.sign((t - tc[u]) / ratings_jt_DAY) * ratings_np.abs((t - tc[u]) / ratings_jt_DAY) ** 0.4
    zc = ratings_np.bincount(u, weights=raw, minlength=nu) / ratings_np.maximum(uc, 1)
    z = raw - zc[u]
    z2 = ratings_np.bincount(u, weights=z * z, minlength=nu)
    mu = float(y.mean())
    states = []
    for cfg in configs or ratings_jt_DEFAULT_CONFIGS:
        keys_u, gu, cgu = ratings_jt_make_group(u, t, cfg['user_days'])
        keys_i, gi, cgi = ratings_jt_make_group(i, t, cfg['item_days'])
        keys_long, gl, cgl = ratings_jt_make_group(u, t, 365.0)
        bu, bi, slope = (ratings_np.zeros(nu), ratings_np.zeros(ni), ratings_np.zeros(nu))
        eu, ei, el = (ratings_np.zeros(len(keys_u)), ratings_np.zeros(len(keys_i)), ratings_np.zeros(len(keys_long)))
        pred = ratings_np.full(len(y), mu)
        for _ in range(25):
            bu = ratings_jt_categorical_sweep(y, pred, u, bu, uc, 5.0)
            bi = ratings_jt_categorical_sweep(y, pred, i, bi, ic, 8.0)
            slope = ratings_jt_slope_sweep(y, pred, u, z, slope, z2, cfg['slope_reg'])
            el = ratings_jt_categorical_sweep(y, pred, gl, el, cgl, 15.0)
            ei = ratings_jt_categorical_sweep(y, pred, gi, ei, cgi, cfg['item_reg'])
            eu = ratings_jt_categorical_sweep(y, pred, gu, eu, cgu, cfg['session_reg'])
        states.append(dict(config=cfg, mu=mu, bu=bu, bi=bi, tc=tc, zc=zc, slope=slope, keys_u=keys_u, effect_u=eu, count_u=cgu, keys_i=keys_i, effect_i=ei, count_i=cgi, keys_long=keys_long, effect_long=el, count_long=cgl))
    return dict(nu=nu, ni=ni, models=states)

def ratings_jt_lookup_group(keys, effects, counts, query_keys):
    where = ratings_np.searchsorted(keys, query_keys)
    good = where < len(keys)
    safe = ratings_np.minimum(where, len(keys) - 1)
    good &= keys[safe] == query_keys
    return (ratings_np.where(good, effects[safe], 0.0), ratings_np.where(good, counts[safe], 0.0))

def ratings_jt_temporal_features(state, qu, qi, qt):
    nu, ni = (state['nu'], state['ni'])
    cols = []
    for s in state['models']:
        cfg = s['config']
        bu = ratings_np.r_[s['bu'], 0.0][qu]
        bi = ratings_np.r_[s['bi'], 0.0][qi]
        tc = ratings_np.r_[s['tc'], 0.0][qu]
        zc = ratings_np.r_[s['zc'], 0.0][qu]
        dt = (qt - tc) / ratings_jt_DAY
        z = ratings_np.clip(ratings_np.sign(dt) * ratings_np.abs(dt) ** 0.4 - zc, -30, 30)
        trend = ratings_np.r_[s['slope'], 0.0][qu] * z
        eu, cu = ratings_jt_lookup_group(s['keys_u'], s['effect_u'], s['count_u'], ratings_jt_time_code(qu, qt, cfg['user_days']))
        ei, ci = ratings_jt_lookup_group(s['keys_i'], s['effect_i'], s['count_i'], ratings_jt_time_code(qi, qt, cfg['item_days']))
        el, cl = ratings_jt_lookup_group(s['keys_long'], s['effect_long'], s['count_long'], ratings_jt_time_code(qu, qt, 365.0))
        pred = s['mu'] + bu + bi + trend + eu + ei + el
        cols.extend([pred, eu, ei, el, trend, ratings_np.log1p(cu), ratings_np.log1p(ci), eu * ratings_np.log1p(cu), ei * ratings_np.log1p(ci), eu * (cu / (cu + 2.0))])
    return ratings_np.column_stack(cols)

def ratings_jt_fit_features(fit_rows, nu, ni):
    return ratings_jt_fit_temporal_models(fit_rows, nu, ni)

def ratings_jt_extract_features(state, query_users, query_items, query_times):
    return ratings_jt_temporal_features(state, query_users, query_items, query_times)

@ratings_njit(cache=True)
def ratings_lf_mf_epoch(arr, order, p, q, bu, bi, mu, lr, rp, rq, rb, scales):
    for idx in order:
        u, i, r = (arr[idx, 0], arr[idx, 1], arr[idx, 3])
        dot = 0.0
        for k in range(p.shape[1]):
            dot += p[u, k] * q[i, k]
        sc = scales[u]
        err = r - (mu + bu[u] + bi[i] + sc * dot)
        bu[u] += lr * (err - rb * bu[u])
        bi[i] += lr * (err - rb * bi[i])
        for k in range(p.shape[1]):
            pu, qi = (p[u, k], q[i, k])
            p[u, k] += lr * (err * sc * qi - rp * pu)
            q[i, k] += lr * (err * sc * pu - rq * qi)

@ratings_njit(cache=True)
def ratings_lf_fast_svdpp_epoch(uperm, ptr, its, rats, iptr, iits, p, q, y, bu, bi, mu, lr, rp, rq, ry, rb):
    k = p.shape[1]
    h = ratings_np.empty(k, ratings_np.float64)
    h0 = ratings_np.empty(k, ratings_np.float64)
    for u in uperm:
        s, e = (ptr[u], ptr[u + 1])
        ns, ne = (iptr[u], iptr[u + 1])
        if e <= s or ne <= ns:
            continue
        norm = 1.0 / ratings_np.sqrt(ne - ns)
        for d in range(k):
            z = 0.0
            for ix in range(ns, ne):
                z += y[iits[ix], d]
            h[d] = z * norm
            h0[d] = h[d]
        for ix in range(s, e):
            i = its[ix]
            r = rats[ix]
            pred = mu + bu[u] + bi[i]
            for d in range(k):
                pred += q[i, d] * (p[u, d] + h[d])
            err = r - pred
            bu[u] += lr * (err - rb * bu[u])
            bi[i] += lr * (err - rb * bi[i])
            for d in range(k):
                pu, qi = (p[u, d], q[i, d])
                q[i, d] += lr * (err * (pu + h[d]) - rq * qi)
                p[u, d] += lr * (err * qi - rp * pu)
                h[d] += lr * (err * qi - ry * h[d])
        shrink = (1.0 - lr * ry) ** (e - s)
        for ix in range(ns, ne):
            j = iits[ix]
            for d in range(k):
                y[j, d] = shrink * y[j, d] + (h[d] - shrink * h0[d]) * norm

@ratings_njit(cache=True)
def ratings_lf_implicit_vectors(ptr, its, y):
    out = ratings_np.zeros((len(ptr) - 1, y.shape[1]), ratings_np.float32)
    for u in range(len(out)):
        s, e = (ptr[u], ptr[u + 1])
        if e > s:
            norm = 1.0 / ratings_np.sqrt(e - s)
            for ix in range(s, e):
                for d in range(y.shape[1]):
                    out[u, d] += y[its[ix], d] * norm
    return out

class ratings_lf_Model:

    def __init__(self, p, q, bu, bi, mu, scales, iptr=None, iits=None, y=None):
        self.p, self.q, self.bu, self.bi, self.mu, self.scales = (p, q, bu, bi, mu, scales)
        self.iptr, self.iits, self.y = (iptr, iits, y)

    def predict(self, u, i, raw=False):
        uv = u < len(self.p)
        iv = i < len(self.q)
        uu = ratings_np.minimum(u, len(self.p) - 1)
        ii = ratings_np.minimum(i, len(self.q) - 1)
        pp = self.p if self.y is None else self.p + ratings_lf_implicit_vectors(self.iptr, self.iits, self.y)
        dot = ratings_np.sum(pp[uu] * self.q[ii], axis=1) * self.scales[uu]
        pred = self.mu + ratings_np.where(uv, self.bu[uu], 0) + ratings_np.where(iv, self.bi[ii], 0) + ratings_np.where(uv & iv, dot, 0)
        return pred if raw else ratings_np.clip(pred, 1, 5)

    def state(self):
        return dict(p=self.p, q=self.q, bu=self.bu, bi=self.bi, mu=self.mu, scales=self.scales, iptr=self.iptr, iits=self.iits, y=self.y)

def ratings_lf_train(fit, nu, ni, valid, config, seed=258, fixed_epochs=None):
    pos = fit[fit[:, 3] > 0]
    rng = ratings_np.random.default_rng(seed)
    uc = ratings_np.bincount(pos[:, 0], minlength=nu)
    ic = ratings_np.bincount(pos[:, 1], minlength=ni)
    mu, bu, bi = als_biases_custom(pos, nu, ni, uc, ic, 3.0, 5.0, 20)
    p = rng.normal(0, config['init'], (nu, config['k'])).astype(ratings_np.float32)
    q = rng.normal(0, config['init'], (ni, config['k'])).astype(ratings_np.float32)
    scales = ratings_np.ones(nu, ratings_np.float32)
    if config.get('scale', False):
        resid = pos[:, 3] - mu - bu[pos[:, 0]] - bi[pos[:, 1]]
        vv = ratings_np.bincount(pos[:, 0], weights=resid ** 2, minlength=nu)
        scales = ratings_np.sqrt((vv + 10 * 0.55) / (uc + 10)).astype(ratings_np.float32)
    model = ratings_lf_Model(p, q, bu.astype(ratings_np.float32), bi.astype(ratings_np.float32), mu, scales)
    if config['kind'] == 'svdpp':
        order = ratings_np.argsort(pos[:, 0], kind='stable')
        ptr = ratings_np.r_[0, uc.cumsum()]
        its = pos[order, 1]
        rats = pos[order, 3]
        iu = ratings_np.bincount(fit[:, 0], minlength=nu)
        io = ratings_np.argsort(fit[:, 0], kind='stable')
        model.iptr = ratings_np.r_[0, iu.cumsum()]
        model.iits = fit[io, 1]
        model.y = ratings_np.zeros((ni, config['k']), ratings_np.float32)
    best = 10000000000.0
    best_state = None
    best_ep = 0
    stale = 0
    hist = []
    maxep = fixed_epochs or config.get('epochs', 100)
    tv = valid[valid[:, 3] > 0] if valid is not None else None
    for ep in range(1, maxep + 1):
        lr = config['lr'] * config.get('decay', 0.99) ** (ep - 1)
        if config['kind'] == 'mf':
            ratings_lf_mf_epoch(pos, rng.permutation(len(pos)), p, q, model.bu, model.bi, mu, lr, config['reg'], config['reg'], config.get('reg_b', 0.02), scales)
        else:
            ratings_lf_fast_svdpp_epoch(rng.permutation(nu), ptr, its, rats, model.iptr, model.iits, p, q, model.y, model.bu, model.bi, mu, lr, config['reg'], config['reg'], config['reg_y'], config.get('reg_b', 0.02))
        if tv is not None and (ep % 5 == 0 or ep == maxep):
            pred = model.predict(tv[:, 0], tv[:, 1])
            mse = float(ratings_np.mean((pred - tv[:, 3]) ** 2))
            hist.append([ep, mse])
            print(config['name'], ep, round(mse, 7), flush=True)
            if mse < best:
                best = mse
                best_ep = ep
                stale = 0
                best_state = [model.p.copy(), model.q.copy(), model.bu.copy(), model.bi.copy(), None if model.y is None else model.y.copy()]
            else:
                stale += 1
            if fixed_epochs is None and stale >= 4:
                break
    if best_state is not None:
        model.p, model.q, model.bu, model.bi, model.y = best_state
    return (model, dict(best_epoch=best_ep or maxep, best_tune_mse=best, history=hist, config=config))

@ratings_njit(cache=True)
def ratings_stg__fit_stages(ptr, items, times, rating, res, emb, nu):
    k = emb.shape[1]
    bounds = ratings_np.zeros((nu, 8, 2), ratings_np.int64)
    stats = ratings_np.zeros((nu, 8, 5))
    centers = ratings_np.zeros((nu, 8, k))
    for u in range(nu):
        us, ue = (ptr[u], ptr[u + 1])
        n = ue - us
        if n == 0:
            continue
        high = ratings_np.mean(rating[us:ue] >= 4)
        offset = 0
        for parts in (3, 5):
            for s in range(parts):
                lo = us + n * s // parts
                hi = us + n * (s + 1) // parts
                idx = offset + s
                nn = hi - lo
                if nn == 0:
                    continue
                bounds[u, idx, 0] = times[lo] - 43200
                bounds[u, idx, 1] = times[hi - 1] + 43200
                rr = res[lo:hi]
                stats[u, idx, 0] = nn / n
                stats[u, idx, 1] = ratings_np.sum(rr) / (nn + 3.0)
                stats[u, idx, 2] = ratings_np.median(rr) * nn / (nn + 3.0)
                stats[u, idx, 3] = (ratings_np.sum(rating[lo:hi] >= 4) - high * nn) / (nn + 3.0)
                stats[u, idx, 4] = ratings_np.sum(rr * rr) / (nn + 3.0)
                for ix in range(lo, hi):
                    for d in range(k):
                        centers[u, idx, d] += emb[items[ix], d]
                norm = ratings_np.sqrt(ratings_np.sum(centers[u, idx] ** 2))
                if norm > 1e-07:
                    centers[u, idx] /= norm
            offset += parts
    return (bounds, stats, centers)

def ratings_stg_fit_stage_model(fit, nu, ni, embeddings=None, biases=None):
    pos = fit[fit[:, 3] > 0]
    uc = ratings_np.bincount(pos[:, 0], minlength=nu)
    ic = ratings_np.bincount(pos[:, 1], minlength=ni)
    if biases is None:
        biases = als_biases_custom(pos, nu, ni, uc, ic, 8.0, 12.0, 20)
    mu, bu, bi = biases
    res = pos[:, 3] - mu - bu[pos[:, 0]] - bi[pos[:, 1]]
    emb = ratings_np.zeros((ni, 1), ratings_np.float32) if embeddings is None else embeddings.astype(ratings_np.float32).copy()
    norm = ratings_np.linalg.norm(emb, axis=1, keepdims=True)
    emb = ratings_np.divide(emb, norm, out=ratings_np.zeros_like(emb), where=norm > 1e-07)
    order = ratings_np.lexsort((pos[:, 2], pos[:, 0]))
    ptr = ratings_np.r_[0, uc.cumsum()]
    bounds, stats, centers = ratings_stg__fit_stages(ptr, pos[order, 1], pos[order, 2], pos[order, 3], res[order], emb, nu)
    all_times = ratings_np.sort(fit[:, 2])
    io = ratings_np.lexsort((fit[:, 2], fit[:, 1]))
    ics = ratings_np.bincount(fit[:, 1], minlength=ni)
    return dict(bounds=bounds, stats=stats, centers=centers, emb=emb, iptr=ratings_np.r_[0, ics.cumsum()], itimes=fit[io, 2], global_times=all_times)

@ratings_njit(cache=True)
def ratings_stg_stage_features(state_bounds, stats, centers, emb, iptr, itimes, global_times, qu, qi):
    out = ratings_np.zeros((len(qu), 20))
    nu = stats.shape[0]
    ni = len(iptr) - 1
    for z in range(len(qu)):
        u, i = (qu[z], qi[z])
        if u >= nu:
            continue
        is_, ie = (iptr[i], iptr[i + 1]) if i < ni else (0, 0)
        offset = 0
        for parts in (3, 5):
            tw = ratings_np.zeros(parts)
            ew = ratings_np.zeros(parts)
            cw = ratings_np.zeros(parts)
            lift = ratings_np.zeros(parts)
            for s in range(parts):
                idx = offset + s
                prior = stats[u, idx, 0]
                if prior <= 0:
                    continue
                lo, hi = state_bounds[u, idx]
                p = max(1e-09, (ratings_np.searchsorted(global_times, hi, side='right') - ratings_np.searchsorted(global_times, lo)) / len(global_times))
                c = ratings_np.searchsorted(itimes[is_:ie], hi, side='right') - ratings_np.searchsorted(itimes[is_:ie], lo)
                ll = ratings_np.log((c + 3.0 * p) / ((ie - is_ + 3.0) * p))
                ll = max(-6.0, min(6.0, ll))
                lift[s] = ll
                cosine = 0.0
                if i < ni:
                    for d in range(emb.shape[1]):
                        cosine += emb[i, d] * centers[u, idx, d]
                tw[s] = prior * ratings_np.exp(ll)
                ew[s] = prior * ratings_np.exp(3.0 * cosine)
                cw[s] = prior * ratings_np.exp(ll + 3.0 * cosine)
            st, se, sc = (ratings_np.sum(tw), ratings_np.sum(ew), ratings_np.sum(cw))
            if st <= 0:
                offset += parts
                continue
            tw /= st
            ew /= se
            cw /= sc
            b = 0 if parts == 3 else 10
            for s in range(parts):
                idx = offset + s
                out[z, b] += tw[s] * stats[u, idx, 1]
                out[z, b + 1] += tw[s] * stats[u, idx, 2]
                out[z, b + 2] += cw[s] * stats[u, idx, 1]
                out[z, b + 3] += cw[s] * stats[u, idx, 2]
                out[z, b + 4] += ew[s] * stats[u, idx, 1]
                out[z, b + 5] += tw[s] * stats[u, idx, 3]
                out[z, b + 6] += tw[s] * stats[u, idx, 4]
                out[z, b + 8] += tw[s] * (s + 0.5) / parts
                out[z, b + 9] += tw[s] * lift[s]
            out[z, b + 7] = ratings_np.max(tw) - ratings_np.min(tw)
            offset += parts
    return out

def ratings_stg_extract_stage_features(state, qu, qi):
    return ratings_stg_stage_features(state['bounds'], state['stats'], state['centers'], state['emb'], state['iptr'], state['itimes'], state['global_times'], qu, qi)

def ratings_fit_extended(fit, nu, ni, task):
    old_config = ratings_RATING_CONFIGS[task]
    old_state = ratings_fit_rating_base(fit, nu, ni, task, seed=258, fixed_epochs=old_config['best_epochs'], run_svdpp_grid=False)
    regs = [(1.0, 2.0), (3.0, 5.0), (10.0, 15.0)] if task == 1 else [(1.0, 2.0)]
    ordinal = [ratings_ord_fit_ordinal(fit, nu, ni, ru, ri) for ru, ri in regs]
    joint = ratings_jt_fit_features(fit, nu, ni) if task == 3 else None
    cfg = dict(name='fastpp64_reg08', kind='svdpp', k=64, init=0.05, reg=0.08, reg_y=0.02, lr=0.007, decay=0.99, epochs=80)
    latent, _ = ratings_lf_train(fit, nu, ni, None, cfg, fixed_epochs=80)
    stage = ratings_stg_fit_stage_model(fit, nu, ni, old_state['svdpp64'].y, old_state['biases'][2]) if ratings_V3_CONFIGS[task].get('use_stage', False) else None
    return dict(old_state=old_state, ordinal=ordinal, joint=joint, latent=latent, stage=stage)

def ratings_extract_extended(state, qu, qi, qt):
    x, groups = ratings_build_rating_features(state['old_state'], qu, qi, qt)
    x = x[:, groups['stage5_plus_conf_gating']]
    blocks = [x, ratings_nk_extract_topk_features(state['old_state'], qu, qi)]
    blocks.extend((ratings_ord_ordinal_features(s, qu, qi) for s in state['ordinal']))
    if state['joint'] is not None:
        blocks.append(ratings_jt_extract_features(state['joint'], qu, qi, qt))
    blocks.append(state['latent'].predict(qu, qi, raw=True)[:, None])
    if state.get('stage') is not None:
        blocks.append(ratings_stg_extract_stage_features(state['stage'], qu, qi))
    out = ratings_np.column_stack(blocks).astype(ratings_np.float32)
    if not ratings_np.all(ratings_np.isfinite(out)):
        raise ValueError('Nonfinite v3 rating features')
    return (out, int(x.shape[1]))

RATING_EASE_LAMBDA = 150.0
RATING_USER_EASE_LAMBDA = {3: 100.0}          # user-user EASE helped Task 3 (17k items, ~80 ratings each), not Task 1


@nb.njit(parallel=True, cache=True)
def _user_ease_rating_features(qu, qi, qfold, iptr, iu, ir, ifold, W, ubar):
    """User-user EASE weights W (from which books users share): implicit affinity of the user to the target's
    other readers and the W-weighted average of those readers' centred ratings (own fold excluded)."""
    n = len(qu)
    out = np.zeros((n, 4), np.float32)
    nU = W.shape[0]
    for r in nb.prange(n):
        u, i, k = qu[r], qi[r], qfold[r]
        if u >= nU or i >= len(iptr) - 1:
            continue
        imp, num, den, cnt = 0.0, 0.0, 0.0, 0
        for y in range(iptr[i], iptr[i + 1]):
            v = iu[y]
            if v == u:
                continue
            w = W[v, u]
            imp += w
            if ir[y] > 0 and ifold[y] != k:
                num += w * (ir[y] - ubar[v, k])
                den += abs(w)
                cnt += 1
        out[r, 0] = imp
        out[r, 1] = num / (den + 0.05)
        out[r, 2] = np.log(den + 1e-6)
        out[r, 3] = np.log1p(cnt)
    return out


@nb.njit(parallel=True, cache=True)
def _ease_rating_features(qu, qi, qfold, uptr, ui, ur, ufold, B, ubar, ibar, nfold):
    """Item-item EASE weights (learned from which books users have, not from ratings) as a rating neighbourhood:
    implicit affinity of the target to the user's other books, and the B-weighted average of the user's
    user- and item-centred ratings of those books.  Ratings of the query's own fold are never used."""
    n = len(qu)
    out = np.full((n, 5), np.nan, np.float32)
    nI = B.shape[0]
    for r in nb.prange(n):
        u, i, k = qu[r], qi[r], qfold[r]
        if u >= len(uptr) - 1 or i >= nI:
            continue
        imp, num, num2, den, cnt = 0.0, 0.0, 0.0, 0.0, 0
        for y in range(uptr[u], uptr[u + 1]):
            h = ui[y]
            if h == i:
                continue
            b = B[h, i]
            imp += b
            if ur[y] > 0 and ufold[y] != k:
                num += b * (ur[y] - ubar[u, k])
                num2 += b * (ur[y] - ibar[h])
                den += abs(b)
                cnt += 1
        out[r, 0] = imp
        out[r, 4] = np.log1p(cnt)
        if den > 0:
            out[r, 1] = num / (den + 0.05)
            out[r, 2] = num2 / (den + 0.05)
            out[r, 3] = np.log(den + 1e-6)
    return out


def rating_ease_features(task, a, folds, rows, qu, qi, nu, ni):
    """Features for the out-of-fold rows (fold of each row = its own fold) and the test rows (all ratings)."""
    ok = (qu < nu) & (qi < ni)
    B = ease_matrix(np.r_[a[:, 0], qu[ok]], np.r_[a[:, 1], qi[ok]], nu, ni, RATING_EASE_LAMBDA)
    nf = len(folds)
    rowfold = np.empty(len(a), np.int64)
    for k, part in enumerate(folds):
        rowfold[part] = k
    o = np.lexsort((a[:, 2], a[:, 0]))
    uptr = np.searchsorted(a[o, 0], np.arange(nu + 1))
    pos = a[:, 3] > 0
    gm = a[pos, 3].mean()
    s_all = np.bincount(a[pos, 0], weights=a[pos, 3], minlength=nu)
    c_all = np.bincount(a[pos, 0], minlength=nu).astype(float)
    ubar = np.zeros((nu, nf + 1))
    for k in range(nf):
        m = pos & (rowfold == k)
        ubar[:, k] = (s_all - np.bincount(a[m, 0], weights=a[m, 3], minlength=nu) + 3 * gm) / (
            c_all - np.bincount(a[m, 0], minlength=nu) + 3)
    ubar[:, nf] = (s_all + 3 * gm) / (c_all + 3)
    ibar = (np.bincount(a[pos, 1], weights=a[pos, 3], minlength=ni) + 10 * gm) / (np.bincount(a[pos, 1], minlength=ni) + 10)
    args = (a[o, 1], a[o, 3].astype(np.float64), rowfold[o], B, ubar, ibar, nf)
    E = _ease_rating_features(a[rows, 0], a[rows, 1], rowfold[rows], uptr, *args)
    Et = _ease_rating_features(qu, qi, np.full(len(qu), nf), uptr, *args)
    del B, args
    if task in RATING_USER_EASE_LAMBDA:              # same construction with users and items swapped
        W = ease_matrix(np.r_[a[:, 1], qi[ok]], np.r_[a[:, 0], qu[ok]], ni, nu, RATING_USER_EASE_LAMBDA[task])
        oi = np.lexsort((a[:, 0], a[:, 1]))
        iptr = np.searchsorted(a[oi, 1], np.arange(ni + 1))
        uargs = (iptr, a[oi, 0], a[oi, 3].astype(np.float64), rowfold[oi], W, ubar)
        E = np.hstack([E, _user_ease_rating_features(a[rows, 0], a[rows, 1], rowfold[rows], *uargs)])
        Et = np.hstack([Et, _user_ease_rating_features(qu, qi, np.full(len(qu), nf), *uargs)])
        del W, uargs
    return np.nan_to_num(E), np.nan_to_num(Et)          # NaN only for users without other ratings


RATING_FOLDS = 18
META_HGB = dict(max_leaf_nodes=127, max_iter=1500, learning_rate=0.04, min_samples_leaf=200, l2_regularization=100.0)


def _rating_fold_job(args):
    task, fit, queries, nu, ni = args
    state = ratings_fit_extended(fit, nu, ni, task)
    X, _ = ratings_extract_extended(state, queries[:, 0], queries[:, 1], queries[:, 2])
    return X.astype(np.float32)


class ChunkedRidge:
    """Ridge regression on standardised columns with intercept mean(y), as in the v3/v4 stacker.  Statistics are
    accumulated in float64 chunks over (optional) row indices, so the float32 feature matrix is never copied whole."""

    def __init__(self, l2=500.0, chunk=100000):
        self.l2, self.chunk = l2, chunk

    def _chunks(self, X, rows):
        rows = np.arange(len(X)) if rows is None else rows
        for a in range(0, len(rows), self.chunk):
            r = rows[a:a + self.chunk]
            yield r, X[r].astype(np.float64)

    def fit(self, X, y, rows=None):
        d = X.shape[1]
        n, s1, s2 = 0, np.zeros(d), np.zeros(d)
        for r, Z in self._chunks(X, rows):
            n += len(r)
            s1 += Z.sum(0)
            s2 += (Z * Z).sum(0)
        self.mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - self.mean ** 2, 0))
        self.scale = np.where(std > 1e-7, std, 1.0)
        self.b = float(np.mean(y if rows is None else y[rows]))
        A, rhs = self.l2 * np.eye(d), np.zeros(d)
        for r, Z in self._chunks(X, rows):
            Z = (Z - self.mean) / self.scale
            A += Z.T @ Z
            rhs += Z.T @ (y[r] - self.b)
        self.w = np.linalg.solve(A, rhs)
        return self

    def predict(self, X, rows=None):
        out = [self.b + ((Z - self.mean) / self.scale) @ self.w for _, Z in self._chunks(X, rows)]
        return np.concatenate(out) if out else np.zeros(0)


def fit_rating_meta(X, y):
    """Ridge anchor + gradient-boosted residuals; the Ridge residual targets are 5-fold out-of-fold."""
    ridge = ChunkedRidge().fit(X, y)
    oof = np.empty(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(y))):
        oof[va] = ChunkedRidge().fit(X, y, rows=tr).predict(X, rows=va)
    hgb = HistGradientBoostingRegressor(early_stopping=False, random_state=SEED, **META_HGB).fit(X, y - oof)
    return lambda Z: np.clip(ridge.predict(Z) + hgb.predict(Z), 1, 5)


def run_rating_task(task, args):
    a, users, items = load(args.data_dir / FILES[task])
    nu, ni = len(users), len(items)
    tests = read_test(args.data_dir / f'task_{task}_test.csv')
    qu = np.array([users.get(r['user'], nu) for r in tests], np.int64)
    qi = np.array([items.get(r['item'], ni) for r in tests], np.int64)
    qt = np.array([int(r['timestamp']) for r in tests], np.int64) if task == 3 else np.zeros(len(tests), np.int64)
    test_rows = np.column_stack([qu, qi, qt, np.zeros(len(qu), np.int64)])
    folds = np.array_split(np.random.default_rng(11).permutation(len(a)), RATING_FOLDS)
    y_folds = [a[part, 3].astype(float) for part in folds]
    y = np.concatenate([yf[yf > 0] for yf in y_folds])
    rows = np.concatenate([part[yf > 0] for part, yf in zip(folds, y_folds)])        # same order as the bank rows
    E, Et = rating_ease_features(task, a, folds, rows, qu, qi, nu, ni)
    log(f'task {task}: EASE rating-neighbourhood features done')
    jobs = []
    for part in folds:
        keep = np.ones(len(a), bool)
        keep[part] = False
        jobs.append((task, a[keep], np.vstack([a[part], test_rows]), nu, ni))
    # Column 0 of the bank is used by neither stacker part, so it is dropped; only positive-rating rows are kept.
    X, test_blocks, row = None, [], 0
    with ProcessPoolExecutor(max_workers=args.rating_workers, mp_context=get_context('spawn')) as ex:
        for block, part, yf in zip(ex.map(_rating_fold_job, jobs), folds, y_folds):
            if X is None:
                nb_ = block.shape[1] - 1
                X = np.empty((len(y), nb_ + E.shape[1]), np.float32)
            rated = yf > 0
            X[row:row + rated.sum(), :nb_] = block[:len(part)][rated, 1:]
            row += rated.sum()
            test_blocks.append(np.ascontiguousarray(np.hstack([block[len(part):, 1:], Et])))
            del block
    X[:, nb_:] = E
    del jobs, E
    log(f'task {task}: out-of-fold feature bank done')
    metrics = {'meta_rows': int(len(y)), 'folds': RATING_FOLDS, 'n_features': int(X.shape[1])}
    model = fit_rating_meta(X, y)
    gbdt = np.mean([model(Z) for Z in test_blocks], axis=0)
    mlp = np.mean(mlp_stacker(X, y, test_blocks), axis=0)
    del X
    predictions = np.clip(0.5 * gbdt + 0.5 * mlp, 1, 5)      # GBDT and MLP stackers are equally good; the blend is better
    save_predictions(args.data_dir, args.output_dir, task, [repr(float(v)) for v in predictions])
    return metrics


# =============================================================================================
# main
# =============================================================================================
FILES = {1: 'tasks_1_2_train.csv', 2: 'tasks_1_2_train.csv', 3: 'tasks_3_4_train.csv', 4: 'tasks_3_4_train.csv',
         5: 'task_5_train.csv', 6: 'task_6_train.csv'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path('assignment1_2026'))
    ap.add_argument('--output-dir', type=Path, default=Path('.'))
    ap.add_argument('--tasks', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--rating-workers', type=int, default=4)
    ap.add_argument('--task5-order-only', action='store_true', help='Task 5 without any timestamp features')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    loaded = {}
    for task in sorted(args.tasks):
        if task in (1, 3):
            metrics[task] = run_rating_task(task, args)
            continue
        train = FILES[task]
        if train not in loaded:
            loaded.clear()
            tests = [f'task_{k}_test.csv' for k in FILES if FILES[k] == train]
            loaded[train] = Data(args.data_dir, train, tests)
        data = loaded[train]
        if task in (2, 4):
            preds, metrics[task] = run_pair_task(task, data)
        elif task == 5:
            preds, metrics[task] = run_task5(data, args.task5_order_only)
        else:
            preds, metrics[task] = run_task6(data)
        save_predictions(args.data_dir, args.output_dir, task, preds)
    log(f'done: {metrics}')


if __name__ == '__main__':
    main()
