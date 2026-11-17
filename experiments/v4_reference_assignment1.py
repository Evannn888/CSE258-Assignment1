from __future__ import annotations


# CSE258 A1: positive-conditional ratings, symmetric pair models, and terminal-time prediction.
# Rebuilds all outputs from designated training CSVs; no research cache is required.


from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
SEED = 258
T_MIN_VALID = 946684800
T_MAX_VALID = 1509522985

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

def write_predictions(source, target, predictions):
    with open(source, newline='') as src, open(target, 'w', newline='') as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator='\n')
        header = next(reader)
        writer.writerow(header)
        count = 0
        for row, prediction in zip(reader, predictions, strict=True):
            assert len(row) == len(header) - 1
            if header[-1] == 'which':
                assert prediction in row[-2:]
            else:
                val = float(prediction)
                assert np.isfinite(val)
            writer.writerow(row + [prediction])
            count += 1
    log(f'Wrote {target}: {count:,} predictions')

def read_test(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def interaction_matrix(a, nu, ni):
    x = csr_matrix((np.ones(len(a), np.float32), (a[:, 0], a[:, 1])), shape=(nu, ni))
    x.sort_indices()
    return x

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

@njit(cache=True)
def bpr_pop_epoch(a, order, p, q, ptr, seen, seed, lr, reg):
    np.random.seed(seed)
    n_rows = len(a)
    for idx in order:
        u, i = (a[idx, 0], a[idx, 1])
        j = a[np.random.randint(n_rows), 1]
        start, end = (ptr[u], ptr[u + 1])
        while True:
            pos = np.searchsorted(seen[start:end], j)
            if pos == end - start or seen[start + pos] != j:
                break
            j = a[np.random.randint(n_rows), 1]
        margin = 0.0
        for k in range(p.shape[1]):
            margin += p[u, k] * (q[i, k] - q[j, k])
        grad = 1.0 / (1.0 + np.exp(min(30.0, max(-30.0, margin))))
        for k in range(p.shape[1]):
            pu, qi, qj = (p[u, k], q[i, k], q[j, k])
            p[u, k] += lr * (grad * (qi - qj) - reg * pu)
            q[i, k] += lr * (grad * pu - reg * qi)
            q[j, k] += lr * (-grad * pu - reg * qj)

def train_bpr_pop(a, nu, ni, epochs=25):
    rng = np.random.default_rng(SEED)
    p = rng.normal(0, 0.05, (nu, 32)).astype(np.float32)
    q = rng.normal(0, 0.05, (ni, 32)).astype(np.float32)
    x = interaction_matrix(a, nu, ni)
    for epoch in range(1, epochs + 1):
        bpr_pop_epoch(a, rng.permutation(len(a)), p, q, x.indptr, x.indices, SEED + epoch, 0.025 * 0.97 ** (epoch - 1), 0.01)
    return (p, q)

@njit(cache=True)
def spectral_history_pooling(q_u, q_i1, q_i2, u_ptr, u_items, i_unit):
    n = len(q_u)
    dim = i_unit.shape[1]
    out = np.zeros((n, 5), dtype=np.float32)
    for idx in range(n):
        u, i1, i2 = (q_u[idx], q_i1[idx], q_i2[idx])
        if u >= len(u_ptr) - 1:
            continue
        s, e = (u_ptr[u], u_ptr[u + 1])
        if s == e:
            continue
        v1 = i_unit[i1]
        v2 = i_unit[i2]
        sum1_p1, sum2_p1 = (0.0, 0.0)
        sum1_p2, sum2_p2 = (0.0, 0.0)
        sum1_p3, sum2_p3 = (0.0, 0.0)
        max1, max2 = (-1.0, -1.0)
        rec1_p2, rec2_p2 = (0.0, 0.0)
        L = e - s
        for p in range(s, e):
            j = u_items[p]
            if j == i1 or j == i2:
                continue
            vj = i_unit[j]
            d1, d2 = (0.0, 0.0)
            for k in range(dim):
                d1 += vj[k] * v1[k]
                d2 += vj[k] * v2[k]
            if d1 > max1:
                max1 = d1
            if d2 > max2:
                max2 = d2
            sum1_p1 += d1
            sum2_p1 += d2
            if d1 > 0.0:
                d1_2 = d1 * d1
                sum1_p2 += d1_2
                sum1_p3 += d1_2 * d1
                rec1_p2 += d1_2 / float(L - 1 - (p - s) + 2)
            if d2 > 0.0:
                d2_2 = d2 * d2
                sum2_p2 += d2_2
                sum2_p3 += d2_2 * d2
                rec2_p2 += d2_2 / float(L - 1 - (p - s) + 2)
        fL = np.sqrt(float(L))
        out[idx, 0] = (sum1_p1 - sum2_p1) / fL
        out[idx, 1] = (sum1_p2 - sum2_p2) / fL
        out[idx, 2] = (sum1_p3 - sum2_p3) / fL
        out[idx, 3] = max1 - max2
        out[idx, 4] = rec1_p2 - rec2_p2
    return out

@njit(cache=True)
def task2_pair_features(q_users, q_i1, q_i2, u_ptr, u_items, u_times, u_rats, i_ptr, i_users, i_times, u_cnt, i_cnt):
    n = len(q_users)
    out = np.zeros((n, 18), dtype=np.float32)
    in_u1 = np.zeros(len(u_cnt), dtype=np.float32)
    in_u2 = np.zeros(len(u_cnt), dtype=np.float32)
    t_u1 = np.zeros(len(u_cnt), dtype=np.int64)
    t_u2 = np.zeros(len(u_cnt), dtype=np.int64)
    inv_u = 1.0 / (u_cnt.astype(np.float32) + 2.0)
    for idx in range(n):
        u, i1, i2 = (q_users[idx], q_i1[idx], q_i2[idx])
        if u >= len(u_cnt):
            continue
        ustart, uend = (u_ptr[u], u_ptr[u + 1])
        if ustart == uend:
            continue
        L_u = uend - ustart
        c1 = float(i_cnt[i1]) if i1 < len(i_cnt) else 0.0
        c2 = float(i_cnt[i2]) if i2 < len(i_cnt) else 0.0
        ut_min, ut_max = (u_times[ustart], u_times[uend - 1])
        ut_med = u_times[ustart + L_u // 2]
        ut_q25 = u_times[ustart + L_u // 4]
        ut_q75 = u_times[ustart + 3 * L_u // 4]
        if i1 < len(i_cnt) and c1 > 0:
            ts1 = i_times[i_ptr[i1]:i_ptr[i1 + 1]]
            in_span1 = np.searchsorted(ts1, ut_max + 30 * 86400, side='right') - np.searchsorted(ts1, ut_min - 30 * 86400)
            in_iqr1 = np.searchsorted(ts1, ut_q75 + 60 * 86400, side='right') - np.searchsorted(ts1, ut_q25 - 60 * 86400)
            in_med1 = np.searchsorted(ts1, ut_med + 180 * 86400, side='right') - np.searchsorted(ts1, ut_med - 180 * 86400)
            out[idx, 0] += np.log((in_span1 + 0.3) / (c1 + 3.0))
            out[idx, 1] += np.log((in_iqr1 + 0.3) / (c1 + 3.0))
            out[idx, 2] += np.log((in_med1 + 0.3) / (c1 + 3.0))
            if ut_max < ts1[0]:
                out[idx, 3] -= 1.0
        if i2 < len(i_cnt) and c2 > 0:
            ts2 = i_times[i_ptr[i2]:i_ptr[i2 + 1]]
            in_span2 = np.searchsorted(ts2, ut_max + 30 * 86400, side='right') - np.searchsorted(ts2, ut_min - 30 * 86400)
            in_iqr2 = np.searchsorted(ts2, ut_q75 + 60 * 86400, side='right') - np.searchsorted(ts2, ut_q25 - 60 * 86400)
            in_med2 = np.searchsorted(ts2, ut_med + 180 * 86400, side='right') - np.searchsorted(ts2, ut_med - 180 * 86400)
            out[idx, 0] -= np.log((in_span2 + 0.3) / (c2 + 3.0))
            out[idx, 1] -= np.log((in_iqr2 + 0.3) / (c2 + 3.0))
            out[idx, 2] -= np.log((in_med2 + 0.3) / (c2 + 3.0))
            if ut_max < ts2[0]:
                out[idx, 3] += 1.0
        if i1 < len(i_cnt):
            for p in range(i_ptr[i1], i_ptr[i1 + 1]):
                up = i_users[p]
                in_u1[up] = inv_u[up]
                t_u1[up] = i_times[p]
        if i2 < len(i_cnt):
            for p in range(i_ptr[i2], i_ptr[i2 + 1]):
                up = i_users[p]
                in_u2[up] = inv_u[up]
                t_u2[up] = i_times[p]
        seen1, seen2 = (0.0, 0.0)
        s1_tl, s2_tl, s1_sq, s2_sq, s1_rat, s2_rat = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s1_tn7, s2_tn7, s1_tn30, s2_tn30, s1_tn180, s2_tn180 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s1_ut_sync, s2_ut_sync, l1_max, l2_max, c1_cos2, c2_cos2, hits1, hits2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for pos_idx in range(ustart, uend):
            j = u_items[pos_idx]
            if j == i1:
                seen1 = 1.0
                continue
            if j == i2:
                seen2 = 1.0
                continue
            t_uj = u_times[pos_idx]
            w_r = 0.6 + 0.12 * float(u_rats[pos_idx])
            cj = float(i_cnt[j])
            inv_cj = 1.0 / (cj + 5.0)
            inv_sq_cj = 1.0 / np.sqrt(cj + 5.0)
            jstart, jend = (i_ptr[j], i_ptr[j + 1])
            w1, w2, tn7_1, tn7_2, tn30_1, tn30_2, tn180_1, tn180_2, sync1, sync2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for q in range(jstart, jend):
                up = i_users[q]
                t_upj = i_times[q]
                u1_w = in_u1[up]
                if u1_w > 0.0:
                    w1 += u1_w
                    dt_up = abs(t_u1[up] - t_upj)
                    if dt_up <= 7 * 86400:
                        tn7_1 += u1_w
                    if dt_up <= 30 * 86400:
                        tn30_1 += u1_w
                    if dt_up <= 180 * 86400:
                        tn180_1 += u1_w
                    if abs(t_u1[up] - t_uj) <= 90 * 86400:
                        sync1 += u1_w
                u2_w = in_u2[up]
                if u2_w > 0.0:
                    w2 += u2_w
                    dt_up = abs(t_u2[up] - t_upj)
                    if dt_up <= 7 * 86400:
                        tn7_2 += u2_w
                    if dt_up <= 30 * 86400:
                        tn30_2 += u2_w
                    if dt_up <= 180 * 86400:
                        tn180_2 += u2_w
                    if abs(t_u2[up] - t_uj) <= 90 * 86400:
                        sync2 += u2_w
            if w1 > 0.0:
                hits1 += 1.0
                v_tl = w1 * inv_cj
                s1_tl += v_tl
                s1_sq += w1 * inv_sq_cj
                s1_rat += v_tl * w_r
                s1_tn7 += tn7_1 * inv_cj
                s1_tn30 += tn30_1 * inv_cj
                s1_tn180 += tn180_1 * inv_cj
                s1_ut_sync += sync1 * inv_cj
                lift1 = v_tl / (c1 + 2.0)
                if lift1 > l1_max:
                    l1_max = lift1
                cos1 = w1 * inv_sq_cj / np.sqrt(c1 + 5.0)
                c1_cos2 += cos1 * cos1
            if w2 > 0.0:
                hits2 += 1.0
                v_tl = w2 * inv_cj
                s2_tl += v_tl
                s2_sq += w2 * inv_sq_cj
                s2_rat += v_tl * w_r
                s2_tn7 += tn7_2 * inv_cj
                s2_tn30 += tn30_2 * inv_cj
                s2_tn180 += tn180_2 * inv_cj
                s2_ut_sync += sync2 * inv_cj
                lift2 = v_tl / (c2 + 2.0)
                if lift2 > l2_max:
                    l2_max = lift2
                cos2 = w2 * inv_sq_cj / np.sqrt(c2 + 5.0)
                c2_cos2 += cos2 * cos2
        if i1 < len(i_cnt):
            for p in range(i_ptr[i1], i_ptr[i1 + 1]):
                in_u1[i_users[p]] = 0.0
        if i2 < len(i_cnt):
            for p in range(i_ptr[i2], i_ptr[i2 + 1]):
                in_u2[i_users[p]] = 0.0
        fL = float(L_u)
        out[idx, 4] = np.log((s1_tl + 1e-05) / (c1 + 0.5)) - np.log((s2_tl + 1e-05) / (c2 + 0.5))
        out[idx, 5] = np.log((s1_tl + 1e-07) / (c1 + 2.0)) - np.log((s2_tl + 1e-07) / (c2 + 2.0))
        out[idx, 6] = np.log((s1_tl + 0.0001) / (c1 + 5.0)) - np.log((s2_tl + 0.0001) / (c2 + 5.0))
        out[idx, 7] = np.log((s1_sq + 0.0001) / (c1 + 2.0)) - np.log((s2_sq + 0.0001) / (c2 + 2.0))
        out[idx, 8] = np.log((s1_rat + 1e-05) / (c1 + 1.0)) - np.log((s2_rat + 1e-05) / (c2 + 1.0))
        out[idx, 9] = np.log((s1_tn7 + 1e-06) / (c1 + 1.0)) - np.log((s2_tn7 + 1e-06) / (c2 + 1.0))
        out[idx, 10] = np.log((s1_tn30 + 1e-06) / (c1 + 1.0)) - np.log((s2_tn30 + 1e-06) / (c2 + 1.0))
        out[idx, 11] = np.log((s1_tn180 + 1e-05) / (c1 + 1.0)) - np.log((s2_tn180 + 1e-05) / (c2 + 1.0))
        out[idx, 12] = np.log((s1_ut_sync + 1e-05) / (c1 + 2.0)) - np.log((s2_ut_sync + 1e-05) / (c2 + 2.0))
        out[idx, 13] = np.log(l1_max + 1e-06) - np.log(l2_max + 1e-06)
        out[idx, 14] = c1_cos2 - c2_cos2
        out[idx, 15] = hits1 / fL - hits2 / fL
        out[idx, 16] = np.log(c1 + 1.0) - np.log(c2 + 1.0)
        out[idx, 17] = seen2 - seen1
    return out

@njit(cache=True)
def task4_pair_features(q_users, q_i1, q_i2, q_times, u_ptr, u_items, u_times, u_rats, i_ptr, i_users, i_times_sorted, u_cnt, i_cnt):
    n = len(q_users)
    out = np.zeros((n, 23), dtype=np.float32)
    in_u1 = np.zeros(len(u_cnt), dtype=np.float32)
    in_u2 = np.zeros(len(u_cnt), dtype=np.float32)
    t_u1 = np.zeros(len(u_cnt), dtype=np.int64)
    t_u2 = np.zeros(len(u_cnt), dtype=np.int64)
    inv_u = 1.0 / (u_cnt.astype(np.float32) + 2.0)
    radii = np.array([2 * 86400, 7 * 86400, 21 * 86400, 60 * 86400, 180 * 86400, 540 * 86400, 1460 * 86400], dtype=np.int64)
    for idx in range(n):
        u, i1, i2, t = (q_users[idx], q_i1[idx], q_i2[idx], q_times[idx])
        c1 = float(i_cnt[i1]) if i1 < len(i_cnt) else 0.0
        c2 = float(i_cnt[i2]) if i2 < len(i_cnt) else 0.0
        if i1 < len(i_cnt) and c1 > 0:
            ts1 = i_times_sorted[i_ptr[i1]:i_ptr[i1 + 1]]
            for r_idx in range(7):
                rad = radii[r_idx]
                cnt1 = np.searchsorted(ts1, t + rad, side='right') - np.searchsorted(ts1, t - rad)
                out[idx, r_idx] += np.log((cnt1 + 0.3) / (c1 + 3.0))
            if t < ts1[0]:
                out[idx, 7] -= 1.0
                out[idx, 8] -= np.log1p(float(ts1[0] - t) / 86400.0)
            q_pos1 = float(np.searchsorted(ts1, t)) / c1
            if q_pos1 < 0.02:
                out[idx, 9] -= (0.02 - q_pos1) * 50.0
        if i2 < len(i_cnt) and c2 > 0:
            ts2 = i_times_sorted[i_ptr[i2]:i_ptr[i2 + 1]]
            for r_idx in range(7):
                rad = radii[r_idx]
                cnt2 = np.searchsorted(ts2, t + rad, side='right') - np.searchsorted(ts2, t - rad)
                out[idx, r_idx] -= np.log((cnt2 + 0.3) / (c2 + 3.0))
            if t < ts2[0]:
                out[idx, 7] += 1.0
                out[idx, 8] += np.log1p(float(ts2[0] - t) / 86400.0)
            q_pos2 = float(np.searchsorted(ts2, t)) / c2
            if q_pos2 < 0.02:
                out[idx, 9] += (0.02 - q_pos2) * 50.0
        if u >= len(u_cnt):
            continue
        ustart, uend = (u_ptr[u], u_ptr[u + 1])
        if ustart == uend:
            continue
        fL = float(uend - ustart)
        if i1 < len(i_cnt):
            for p in range(i_ptr[i1], i_ptr[i1 + 1]):
                up = i_users[p]
                in_u1[up] = inv_u[up]
                t_u1[up] = i_times_sorted[p]
        if i2 < len(i_cnt):
            for p in range(i_ptr[i2], i_ptr[i2 + 1]):
                up = i_users[p]
                in_u2[up] = inv_u[up]
                t_u2[up] = i_times_sorted[p]
        seen1, seen2 = (0.0, 0.0)
        s1_tl, s2_tl, s1_sq, s2_sq, s1_rat, s2_rat = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s1_1d, s2_1d, s1_14d, s2_14d, s1_180d, s2_180d, s1_dir, s2_dir = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        l1_max, l2_max, c1_cos2, c2_cos2, hits1, hits2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        best_dt = 1e+18
        near_lift1, near_lift2 = (0.0, 0.0)
        for pos_idx in range(ustart, uend):
            j = u_items[pos_idx]
            if j == i1:
                seen1 = 1.0
                continue
            if j == i2:
                seen2 = 1.0
                continue
            t_uj = u_times[pos_idx]
            dt = float(abs(t - t_uj))
            u_dir = 1 if t >= t_uj else -1
            w_1d = np.exp(-dt / (2.0 * 86400.0))
            w_14d = np.exp(-dt / (14.0 * 86400.0))
            w_180d = np.exp(-dt / (180.0 * 86400.0))
            w_r = 0.6 + 0.12 * float(u_rats[pos_idx])
            cj = float(i_cnt[j])
            inv_cj = 1.0 / (cj + 5.0)
            inv_sq_cj = 1.0 / np.sqrt(cj + 5.0)
            jstart, jend = (i_ptr[j], i_ptr[j + 1])
            w1, w2, dir1, dir2 = (0.0, 0.0, 0.0, 0.0)
            for q in range(jstart, jend):
                up = i_users[q]
                t_upj = i_times_sorted[q]
                u1_w = in_u1[up]
                if u1_w > 0.0:
                    w1 += u1_w
                    up_dir1 = 1 if t_u1[up] >= t_upj else -1
                    if up_dir1 == u_dir and abs(t_u1[up] - t_upj) <= 90 * 86400:
                        dir1 += u1_w
                u2_w = in_u2[up]
                if u2_w > 0.0:
                    w2 += u2_w
                    up_dir2 = 1 if t_u2[up] >= t_upj else -1
                    if up_dir2 == u_dir and abs(t_u2[up] - t_upj) <= 90 * 86400:
                        dir2 += u2_w
            v1 = w1 * inv_cj
            v2 = w2 * inv_cj
            if dt < best_dt:
                best_dt = dt
                near_lift1 = v1
                near_lift2 = v2
            if w1 > 0.0:
                hits1 += 1.0
                s1_tl += v1
                s1_sq += w1 * inv_sq_cj
                s1_rat += v1 * w_r
                s1_1d += w_1d * v1
                s1_14d += w_14d * v1
                s1_180d += w_180d * v1
                s1_dir += w_14d * dir1 * inv_cj
                lift1 = v1 / (c1 + 2.0)
                if lift1 > l1_max:
                    l1_max = lift1
                cos1 = w1 * inv_sq_cj / np.sqrt(c1 + 5.0)
                c1_cos2 += cos1 * cos1
            if w2 > 0.0:
                hits2 += 1.0
                s2_tl += v2
                s2_sq += w2 * inv_sq_cj
                s2_rat += v2 * w_r
                s2_1d += w_1d * v2
                s2_14d += w_14d * v2
                s2_180d += w_180d * v2
                s2_dir += w_14d * dir2 * inv_cj
                lift2 = v2 / (c2 + 2.0)
                if lift2 > l2_max:
                    l2_max = lift2
                cos2 = w2 * inv_sq_cj / np.sqrt(c2 + 5.0)
                c2_cos2 += cos2 * cos2
        if i1 < len(i_cnt):
            for p in range(i_ptr[i1], i_ptr[i1 + 1]):
                in_u1[i_users[p]] = 0.0
        if i2 < len(i_cnt):
            for p in range(i_ptr[i2], i_ptr[i2 + 1]):
                in_u2[i_users[p]] = 0.0
        out[idx, 10] = np.log((s1_tl + 3e-06) / (c1 + 0.5)) - np.log((s2_tl + 3e-06) / (c2 + 0.5))
        out[idx, 11] = np.log((s1_tl + 1e-07) / (c1 + 2.0)) - np.log((s2_tl + 1e-07) / (c2 + 2.0))
        out[idx, 12] = np.log((s1_sq + 0.0001) / (c1 + 2.0)) - np.log((s2_sq + 0.0001) / (c2 + 2.0))
        out[idx, 13] = np.log((s1_rat + 3e-06) / (c1 + 1.0)) - np.log((s2_rat + 3e-06) / (c2 + 1.0))
        out[idx, 14] = np.log((s1_1d + 1e-07) / (c1 + 1.0)) - np.log((s2_1d + 1e-07) / (c2 + 1.0))
        out[idx, 15] = np.log((s1_14d + 1e-06) / (c1 + 1.0)) - np.log((s2_14d + 1e-06) / (c2 + 1.0))
        out[idx, 16] = np.log((s1_180d + 3e-06) / (c1 + 1.0)) - np.log((s2_180d + 3e-06) / (c2 + 1.0))
        out[idx, 17] = np.log((s1_dir + 1e-07) / (c1 + 1.0)) - np.log((s2_dir + 1e-07) / (c2 + 1.0))
        out[idx, 18] = np.log((near_lift1 + 1e-06) / (c1 + 1.0)) - np.log((near_lift2 + 1e-06) / (c2 + 1.0))
        out[idx, 19] = np.log(l1_max + 1e-06) - np.log(l2_max + 1e-06)
        out[idx, 20] = c1_cos2 - c2_cos2
        out[idx, 21] = hits1 / fL - hits2 / fL
        out[idx, 22] = seen2 - seen1
    return out

@njit(cache=True)
def task5_seq_features(q_users, q_i1, q_i2, u_ptr, u_items, u_rats, i_ptr, i_users, i_pos, u_cnt, i_cnt, item_rel_mean, item_rel4_mean, item_last1_rate, item_last3_rate, item_last5_rate, u_end_era):
    n = len(q_users)
    out = np.zeros((n, 24), dtype=np.float32)
    in_u1 = np.zeros(len(u_cnt), dtype=np.float32)
    in_u2 = np.zeros(len(u_cnt), dtype=np.float32)
    pos_u1 = np.zeros(len(u_cnt), dtype=np.int32)
    pos_u2 = np.zeros(len(u_cnt), dtype=np.int32)
    inv_u = 1.0 / (u_cnt.astype(np.float32) + 2.0)
    for idx in range(n):
        u, i1, i2 = (q_users[idx], q_i1[idx], q_i2[idx])
        c1 = float(i_cnt[i1]) if i1 < len(i_cnt) else 0.0
        c2 = float(i_cnt[i2]) if i2 < len(i_cnt) else 0.0
        rel1 = item_rel_mean[i1] if i1 < len(i_cnt) and c1 > 0 else 0.6
        rel2 = item_rel_mean[i2] if i2 < len(i_cnt) and c2 > 0 else 0.6
        r4_1 = item_rel4_mean[i1] if i1 < len(i_cnt) and c1 > 0 else 0.25
        r4_2 = item_rel4_mean[i2] if i2 < len(i_cnt) and c2 > 0 else 0.25
        l1_1 = item_last1_rate[i1] if i1 < len(i_cnt) and c1 > 0 else -2.5
        l1_2 = item_last1_rate[i2] if i2 < len(i_cnt) and c2 > 0 else -2.5
        l3_1 = item_last3_rate[i1] if i1 < len(i_cnt) and c1 > 0 else -1.8
        l3_2 = item_last3_rate[i2] if i2 < len(i_cnt) and c2 > 0 else -1.8
        l5_1 = item_last5_rate[i1] if i1 < len(i_cnt) and c1 > 0 else -1.4
        l5_2 = item_last5_rate[i2] if i2 < len(i_cnt) and c2 > 0 else -1.4
        out[idx, 0] = rel1 - rel2
        out[idx, 1] = r4_1 - r4_2
        out[idx, 2] = l1_1 - l1_2
        out[idx, 3] = l3_1 - l3_2
        out[idx, 4] = l5_1 - l5_2
        if u >= len(u_cnt):
            continue
        ustart, uend = (u_ptr[u], u_ptr[u + 1])
        if ustart == uend:
            continue
        L_u = uend - ustart
        fL = float(L_u)
        era_u = u_end_era[u]
        out[idx, 5] = abs(era_u - rel2) - abs(era_u - rel1)
        if i1 < len(i_cnt):
            for p in range(i_ptr[i1], i_ptr[i1 + 1]):
                up = i_users[p]
                in_u1[up] = inv_u[up]
                pos_u1[up] = i_pos[p]
        if i2 < len(i_cnt):
            for p in range(i_ptr[i2], i_ptr[i2 + 1]):
                up = i_users[p]
                in_u2[up] = inv_u[up]
                pos_u2[up] = i_pos[p]
        seen1, seen2 = (0.0, 0.0)
        s1_tl, s2_tl, s1_sq, s2_sq, s1_rat, s2_rat = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s1_rec, s2_rec, s1_last1, s2_last1, s1_last3, s2_last3 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s1_fwd, s2_fwd, s1_step1, s2_step1, s1_near, s2_near, s1_bwd1, s2_bwd1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        l1_max, l2_max, c1_cos2, c2_cos2, hits1, hits2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for pos_idx in range(ustart, uend):
            j = u_items[pos_idx]
            if j == i1:
                seen1 = 1.0
                continue
            if j == i2:
                seen2 = 1.0
                continue
            d_rec = L_u - 1 - (pos_idx - ustart)
            w_rec = 1.0 / (float(d_rec) + 1.0)
            w_rec2 = 1.0 / (float(d_rec) + 1.0) ** 0.5
            w_r = 0.6 + 0.12 * float(u_rats[pos_idx])
            cj = float(i_cnt[j])
            inv_cj = 1.0 / (cj + 5.0)
            inv_sq_cj = 1.0 / np.sqrt(cj + 5.0)
            jstart, jend = (i_ptr[j], i_ptr[j + 1])
            w1, w2, fwd1, fwd2, st1, st2, nr1, nr2, bw1, bw2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for q in range(jstart, jend):
                up = i_users[q]
                u1_w = in_u1[up]
                if u1_w > 0.0:
                    w1 += u1_w
                    dp = pos_u1[up] - i_pos[q]
                    if dp > 0:
                        fwd1 += u1_w / (float(dp) + 1.5)
                        if dp == 1:
                            st1 += u1_w
                        elif dp == 2:
                            st1 += 0.5 * u1_w
                    elif dp == -1:
                        bw1 += u1_w
                    if abs(dp) <= 3:
                        nr1 += u1_w
                u2_w = in_u2[up]
                if u2_w > 0.0:
                    w2 += u2_w
                    dp = pos_u2[up] - i_pos[q]
                    if dp > 0:
                        fwd2 += u2_w / (float(dp) + 1.5)
                        if dp == 1:
                            st2 += u2_w
                        elif dp == 2:
                            st2 += 0.5 * u2_w
                    elif dp == -1:
                        bw2 += u2_w
                    if abs(dp) <= 3:
                        nr2 += u2_w
            if w1 > 0.0:
                hits1 += 1.0
                v1 = w1 * inv_cj
                s1_tl += v1
                s1_sq += w1 * inv_sq_cj
                s1_rat += v1 * w_r
                s1_rec += w_rec * v1
                if d_rec == 0:
                    s1_last1 += v1
                if d_rec < 3:
                    s1_last3 += v1
                lift1 = v1 / (c1 + 2.0)
                if lift1 > l1_max:
                    l1_max = lift1
                cos1 = w1 * inv_sq_cj / np.sqrt(c1 + 5.0)
                c1_cos2 += cos1 * cos1
                s1_fwd += w_rec2 * fwd1 * inv_cj
                s1_step1 += w_rec * st1 * inv_cj
                s1_near += w_rec * nr1 * inv_cj
                s1_bwd1 += w_rec * bw1 * inv_cj
            if w2 > 0.0:
                hits2 += 1.0
                v2 = w2 * inv_cj
                s2_tl += v2
                s2_sq += w2 * inv_sq_cj
                s2_rat += v2 * w_r
                s2_rec += w_rec * v2
                if d_rec == 0:
                    s2_last1 += v2
                if d_rec < 3:
                    s2_last3 += v2
                lift2 = v2 / (c2 + 2.0)
                if lift2 > l2_max:
                    l2_max = lift2
                cos2 = w2 * inv_sq_cj / np.sqrt(c2 + 5.0)
                c2_cos2 += cos2 * cos2
                s2_fwd += w_rec2 * fwd2 * inv_cj
                s2_step1 += w_rec * st2 * inv_cj
                s2_near += w_rec * nr2 * inv_cj
                s2_bwd1 += w_rec * bw2 * inv_cj
        if i1 < len(i_cnt):
            for p in range(i_ptr[i1], i_ptr[i1 + 1]):
                in_u1[i_users[p]] = 0.0
        if i2 < len(i_cnt):
            for p in range(i_ptr[i2], i_ptr[i2 + 1]):
                in_u2[i_users[p]] = 0.0
        out[idx, 6] = np.log((s1_tl + 1e-05) / (c1 + 0.5)) - np.log((s2_tl + 1e-05) / (c2 + 0.5))
        out[idx, 7] = np.log((s1_tl + 1e-06) / (c1 + 2.0)) - np.log((s2_tl + 1e-06) / (c2 + 2.0))
        out[idx, 8] = np.log((s1_tl + 5e-05) / (c1 + 5.0)) - np.log((s2_tl + 5e-05) / (c2 + 5.0))
        out[idx, 9] = np.log((s1_sq + 0.0001) / (c1 + 2.0)) - np.log((s2_sq + 0.0001) / (c2 + 2.0))
        out[idx, 10] = np.log((s1_rat + 1e-05) / (c1 + 1.0)) - np.log((s2_rat + 1e-05) / (c2 + 1.0))
        out[idx, 11] = np.log((s1_rec + 3e-06) / (c1 + 1.0)) - np.log((s2_rec + 3e-06) / (c2 + 1.0))
        out[idx, 12] = np.log((s1_last1 + 1e-06) / (c1 + 1.0)) - np.log((s2_last1 + 1e-06) / (c2 + 1.0))
        out[idx, 13] = np.log((s1_last3 + 3e-06) / (c1 + 1.0)) - np.log((s2_last3 + 3e-06) / (c2 + 1.0))
        out[idx, 14] = np.log((s1_fwd + 1e-06) / (c1 + 1.0)) - np.log((s2_fwd + 1e-06) / (c2 + 1.0))
        out[idx, 15] = np.log((s1_step1 + 3e-07) / (c1 + 1.0)) - np.log((s2_step1 + 3e-07) / (c2 + 1.0))
        out[idx, 16] = np.log((s1_near + 1e-06) / (c1 + 1.0)) - np.log((s2_near + 1e-06) / (c2 + 1.0))
        out[idx, 17] = np.log((s1_bwd1 + 3e-07) / (c1 + 1.0)) - np.log((s2_bwd1 + 3e-07) / (c2 + 1.0))
        out[idx, 18] = np.log(l1_max + 1e-06) - np.log(l2_max + 1e-06)
        out[idx, 19] = c1_cos2 - c2_cos2
        out[idx, 20] = hits1 / fL - hits2 / fL
        out[idx, 21] = np.log(c1 + 1.0) - np.log(c2 + 1.0)
        out[idx, 22] = (1.0 if c1 == 0.0 else 0.0) - (1.0 if c2 == 0.0 else 0.0)
        out[idx, 23] = seen2 - seen1
    return out

def fit_pairwise_logistic(x_train, l2=10.0):
    w = np.zeros(x_train.shape[1], dtype=np.float64)
    for _ in range(12):
        z = np.clip(x_train @ w, -20.0, 20.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        grad = x_train.T @ (1.0 - prob) - l2 * w
        hess = x_train.T * (prob * (1.0 - prob)) @ x_train + l2 * np.eye(len(w))
        w += np.linalg.solve(hess, grad)
    return w

def apply_delta_params(feats, params):
    valid_tp, u_med, u_q15, u_q25, u_rec2, u_rec3, u_rec5, u_last1, upper_arr, low, na, nt, rem_p15, rem_p25, rem_p30 = feats
    s_med, s_q15, s_q25, s_rec2, s_rec3, s_rec5, s_last1, cap_f, harm_f, w_harm, w_it, w_p15, w_p25, w_unrel = params
    u_base = np.maximum(0.0, s_med * u_med + s_q15 * u_q15 + s_q25 * u_q25 + s_rec2 * u_rec2 + s_rec3 * u_rec3 + s_rec5 * u_rec5 + s_last1 * u_last1)
    ub_cap = np.maximum(low, cap_f * upper_arr)
    pred_clip = np.clip(u_base, low, ub_cap)
    span = np.maximum(1.0, harm_f * upper_arr - low)
    excess = np.maximum(0.0, u_base - low)
    pred_harm = np.clip(low + excess * span / (excess + span), low, ub_cap)
    pred = (1.0 - w_harm) * pred_clip + w_harm * pred_harm
    has_rem = valid_tp & (na >= 1)
    it_est = np.clip(w_p15 * rem_p15 + w_p25 * rem_p25 + (1.0 - w_p15 - w_p25) * rem_p30, low, ub_cap)
    conf = na / (na + 3.0) * (na / np.maximum(1.0, nt)) ** 0.5
    w_eff = np.where(low > u_base, np.maximum(w_unrel, w_it * conf), w_it * conf)
    pred = np.where(has_rem, (1.0 - w_eff) * pred + w_eff * it_est, pred)
    return np.where(valid_tp, pred, u_base)
import time as ratings_time
from pathlib import Path as ratings_Path
import numpy as ratings_np
from numba import njit as ratings_njit
from scipy.sparse import csr_matrix as ratings_csr_matrix
from scipy.sparse.linalg import svds as ratings_svds
from sklearn.ensemble import HistGradientBoostingRegressor as ratings_HistGradientBoostingRegressor
from sklearn.model_selection import KFold as ratings_KFold
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

class ratings_StandardizedRidge:

    def __init__(self, l2=10.0):
        self.l2 = float(l2)
        self.mean = None
        self.scale = None
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X_feat = X[:, 1:]
        self.mean = ratings_np.mean(X_feat, axis=0)
        std = ratings_np.std(X_feat, axis=0)
        self.scale = ratings_np.where(std > 1e-07, std, 1.0)
        Z = (X_feat - self.mean) / self.scale
        y_mean = float(ratings_np.mean(y))
        y_centered = y - y_mean
        d = Z.shape[1]
        A = Z.T @ Z + self.l2 * ratings_np.eye(d, dtype=ratings_np.float64)
        rhs = Z.T @ y_centered
        self.w = ratings_np.linalg.solve(A, rhs)
        self.b = y_mean
        return self

    def predict_raw(self, X):
        Z = (X[:, 1:] - self.mean) / self.scale
        return self.b + Z @ self.w

    def predict(self, X):
        return ratings_np.clip(self.predict_raw(X), 1.0, 5.0)

def ratings_predict_stacked(ridge_mod, hgb_mod, eta, X):
    m0 = ridge_mod.predict_raw(X)
    if eta > 0.0 and hgb_mod is not None:
        m0 = m0 + eta * hgb_mod.predict(X[:, 1:])
    return ratings_np.clip(m0, 1.0, 5.0)

def ratings_fit_fixed_rating_combiner(X, y, config, seed=ratings_SEED_PRIMARY):
    """Train fixed Ridge and HGB on positive labels, with OOF Ridge residuals."""
    pos = y > 0
    Xp, yp = (X[pos], y[pos].astype(ratings_np.float64))
    ridge = ratings_StandardizedRidge(l2=config['best_l2']).fit(Xp, yp)
    hgb = None
    if config['best_eta'] > 0:
        oof = ratings_np.empty(len(yp), dtype=ratings_np.float64)
        for fit_idx, val_idx in ratings_KFold(n_splits=5, shuffle=True, random_state=seed).split(Xp):
            fold_ridge = ratings_StandardizedRidge(l2=config['best_l2']).fit(Xp[fit_idx], yp[fit_idx])
            oof[val_idx] = fold_ridge.predict_raw(Xp[val_idx])
        hgb = ratings_HistGradientBoostingRegressor(**config['best_hgb_cfg'], random_state=seed).fit(Xp[:, 1:], yp - oof)
    return (ridge, hgb)
import csv as interactions_csv
import time as interactions_time
from pathlib import Path as interactions_Path
import numpy as interactions_np
from numba import njit as interactions_njit
from scipy.optimize import minimize as interactions_minimize
from scipy.sparse import csr_matrix as interactions_csr_matrix
from scipy.sparse.linalg import svds as interactions_svds
from scipy.special import expit as interactions_expit
from sklearn.ensemble import HistGradientBoostingClassifier as interactions_HistGradientBoostingClassifier
import math as delta_math
import time as delta_time
from datetime import datetime as delta_datetime, timezone as delta_timezone
from pathlib import Path as delta_Path
import numpy as delta_np
from numba import njit as delta_njit
from scipy.optimize import minimize as delta_minimize
from sklearn.ensemble import HistGradientBoostingRegressor as delta_HistGradientBoostingRegressor


# Production ratings: fixed validated configuration.


import time as ratings_time
from pathlib import Path as ratings_Path
import numpy as ratings_np
from numba import njit as ratings_njit
from scipy.special import expit as ratings_expit
from sklearn.ensemble import HistGradientBoostingRegressor as ratings_HistGradientBoostingRegressor
from sklearn.model_selection import KFold as ratings_KFold
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

def ratings_fit_meta(X, y, config):
    mask = y > 0
    xp = X[mask].astype(ratings_np.float64)
    yp = y[mask].astype(ratings_np.float64)
    ridge = ratings_StandardizedRidge(config['reg']).fit(xp, yp)
    hgb = None
    if config['eta']:
        oof = ratings_np.empty(len(yp))
        for tr, va in ratings_KFold(5, shuffle=True, random_state=258).split(xp):
            rr = ratings_StandardizedRidge(config['reg']).fit(xp[tr], yp[tr])
            oof[va] = rr.predict_raw(xp[va])
        hgb = ratings_HistGradientBoostingRegressor(**config['hgb']).fit(xp[:, 1:], yp - oof)
    return (ridge, hgb)

def ratings_predict_meta(bundle, X, config):
    ridge, hgb = bundle
    pred = ridge.predict_raw(X.astype(ratings_np.float64))
    if hgb is not None:
        pred += config['eta'] * hgb.predict(X[:, 1:])
    return ratings_np.clip(pred, 1, 5)

def ratings_production_rating_task(task, a, users, items, args):
    """Frozen v3 settings: 200k base holdout, 150k meta-fit, 25k audit."""
    start = ratings_time.time()
    nu, ni = (len(users), len(items))
    config = ratings_V3_CONFIGS[task]
    order = ratings_np.random.default_rng(258).permutation(len(a))
    valid, fit = (a[order[:200000]], a[order[200000:]])
    log(f'Task {task}: fitting v3 validation base and ordinal/neighbor/latent feature bank')
    state = ratings_fit_extended(fit, nu, ni, task)
    X, width = ratings_extract_extended(state, valid[:, 0], valid[:, 1], valid[:, 2])
    y = valid[:, 3]
    audited = ratings_fit_meta(X[:150000], y[:150000], config)
    pred = ratings_predict_meta(audited, X[175000:], config)
    truth = y[175000:]
    mask = truth > 0
    old_config = ratings_RATING_CONFIGS[task]
    br, bh = ratings_fit_fixed_rating_combiner(X[:20000, :width].astype(ratings_np.float64), y[:20000], old_config)
    baseline = ratings_predict_stacked(br, bh, old_config['best_eta'], X[175000:, :width].astype(ratings_np.float64))
    difference = (pred[mask] - truth[mask]) ** 2 - (baseline[mask] - truth[mask]) ** 2
    metrics = dict(model='v3_large_meta_ordinal_neighborhood_fast_svdpp', seed=258, audit_positive_mse=float(ratings_np.mean((pred[mask] - truth[mask]) ** 2)), audit_baseline20k_positive_mse=float(ratings_np.mean((baseline[mask] - truth[mask]) ** 2)), audit_positive_rows=int(mask.sum()), meta_fit_rows=150000, base_holdout_rows=200000, audit_paired_mse_change=float(difference.mean()), audit_paired_standard_error=float(difference.std(ddof=1) / ratings_np.sqrt(len(difference))), fixed_config=config, n_features=int(X.shape[1]), caveat='Historical internal validation only; not a leaderboard estimate')
    log(f"Task {task}: frozen-config audit MSE={metrics['audit_positive_mse']:.6f}, baseline20k={metrics['audit_baseline20k_positive_mse']:.6f}")
    final_meta = ratings_fit_meta(X, y, config)
    del state, X, audited, br, bh
    log(f'Task {task}: refitting full v3 feature bank')
    state = ratings_fit_extended(a, nu, ni, task)
    tests = read_test(args.data_dir / f'task_{task}_test.csv')
    qu = ratings_np.array([users.get(r['user'], nu) for r in tests], dtype=ratings_np.int64)
    qi = ratings_np.array([items.get(r['item'], ni) for r in tests], dtype=ratings_np.int64)
    qt = ratings_np.array([int(r['timestamp']) for r in tests], dtype=ratings_np.int64) if task == 3 else ratings_np.zeros(len(tests), dtype=ratings_np.int64)
    Xt, _ = ratings_extract_extended(state, qu, qi, qt)
    predictions = ratings_predict_meta(final_meta, Xt, config)
    write_predictions(args.data_dir / f'task_{task}_test.csv', args.output_dir / f'task_{task}_predictions.csv', predictions)
    metrics.update(test_rows=len(tests), elapsed_seconds=round(ratings_time.time() - start, 2), prediction_min=float(predictions.min()), prediction_max=float(predictions.max()))
    return metrics


# Production interactions: fixed validated configuration.


import csv as interactions_csv
import time as interactions_time
from pathlib import Path as interactions_Path
import numpy as interactions_np
from numba import njit as interactions_njit
from scipy.optimize import minimize as interactions_minimize
from scipy.sparse import csr_matrix as interactions_csr_matrix
from scipy.sparse.linalg import svds as interactions_svds
from scipy.special import expit as interactions_expit
from sklearn.ensemble import HistGradientBoostingClassifier as interactions_HistGradientBoostingClassifier

def interactions_log(message):
    log(message)

def interactions_load_task5_order_only(path):
    """Dedicated loader for Task 5 that ONLY parses user, item, int(rating) and NEVER reads timestamp."""
    users, items, rows = ({}, {}, [])
    with open(path, newline='') as f:
        for row in interactions_csv.DictReader(f):
            u, i = (row['user'], row['item'])
            if u not in users:
                users[u] = len(users)
            if i not in items:
                items[i] = len(items)
            rows.append((users[u], items[i], 0, int(row['rating'])))
    a = interactions_np.asarray(rows, dtype=interactions_np.int64)
    interactions_log(f'[Pure-Order Loader] {interactions_Path(path).name}: {len(a):,} rows, {len(users):,} users, {len(items):,} items')
    return (a, users, items)

def interactions_make_interaction_splits_dual(a, nu, ni, sequential=False, seed=258):
    """Construct validation splits under BOTH negative sampling mechanisms:
    - pool_A: negatives sampled from fit_rows[:, 1] (after holding out positives)
    - pool_B: negatives sampled from original_rows[:, 1] (before holding out positives)
    When seed=258, pool_A matches assignment1.make_interaction_split bit-for-bit.
    """
    rng = interactions_np.random.default_rng(seed)
    if sequential:
        last = interactions_np.full(nu, -1, interactions_np.int64)
        interactions_np.maximum.at(last, a[:, 0], interactions_np.arange(len(a)))
        held = last[last >= 0]
    else:
        held = rng.choice(len(a), min(30000, len(a) // 10), replace=False)
    mask = interactions_np.ones(len(a), dtype=bool)
    mask[held] = False
    fit_rows, positive = (a[mask], a[held])
    neg_A = fit_rows[rng.integers(0, len(fit_rows), len(positive)), 1].copy()
    for n in range(len(positive)):
        while neg_A[n] == positive[n, 1]:
            neg_A[n] = fit_rows[rng.integers(0, len(fit_rows)), 1]
    va_A = interactions_np.column_stack([positive[:, 0], positive[:, 1], neg_A, positive[:, 2]])
    rng_B = interactions_np.random.default_rng(seed + 10000)
    neg_B = a[rng_B.integers(0, len(a), len(positive)), 1].copy()
    for n in range(len(positive)):
        while neg_B[n] == positive[n, 1]:
            neg_B[n] = a[rng_B.integers(0, len(a)), 1]
    va_B = interactions_np.column_stack([positive[:, 0], positive[:, 1], neg_B, positive[:, 2]])
    return (fit_rows, va_A, va_B)

@interactions_njit(cache=True)
def interactions_compute_seen_and_zero_flags(q_u, q_i1, q_i2, u_ptr, u_items, i_cnt):
    """Compute:
    - seen_override = (seen2 - seen1) ONLY when one candidate is in H(u) (0 if both or neither)
    - zero_diff = (c1==0) - (c2==0)
    - zero_sum = (c1==0) + (c2==0)
    - zero_neq = 1.0 if (c1==0) != (c2==0) else 0.0
    """
    n = len(q_u)
    nu = len(u_ptr) - 1
    ni = len(i_cnt)
    seen_override = interactions_np.zeros(n, dtype=interactions_np.float64)
    zero_diff = interactions_np.zeros(n, dtype=interactions_np.float64)
    zero_sum = interactions_np.zeros(n, dtype=interactions_np.float64)
    zero_neq = interactions_np.zeros(n, dtype=interactions_np.float64)
    for idx in range(n):
        u, i1, i2 = (q_u[idx], q_i1[idx], q_i2[idx])
        c1 = i_cnt[i1] if i1 < ni else 0
        c2 = i_cnt[i2] if i2 < ni else 0
        z1 = 1.0 if c1 == 0 else 0.0
        z2 = 1.0 if c2 == 0 else 0.0
        zero_diff[idx] = z1 - z2
        zero_sum[idx] = z1 + z2
        zero_neq[idx] = 1.0 if z1 != z2 else 0.0
        if u < nu:
            s1, s2 = (0.0, 0.0)
            for p in range(u_ptr[u], u_ptr[u + 1]):
                j = u_items[p]
                if j == i1:
                    s1 = 1.0
                if j == i2:
                    s2 = 1.0
            seen_override[idx] = s2 - s1
    return (seen_override, zero_diff, zero_sum, zero_neq)

@interactions_njit(cache=True)
def interactions_extract_bt_pairs(u_ptr, u_items):
    """Deterministically sample directed pairs (a < b) from each user sequence at
    lags 1, 3, 10 plus first-to-last anchor, weighted by 1 / sqrt(n_pairs(u)).
    """
    total = 0
    nu = len(u_ptr) - 1
    for u in range(nu):
        L = u_ptr[u + 1] - u_ptr[u]
        if L < 2:
            continue
        np_u = L - 1 + max(0, L - 3) + max(0, L - 10) + (1 if L > 2 else 0)
        total += np_u
    earlier = interactions_np.empty(total, dtype=interactions_np.int32)
    later = interactions_np.empty(total, dtype=interactions_np.int32)
    weights = interactions_np.empty(total, dtype=interactions_np.float64)
    u_owner = interactions_np.empty(total, dtype=interactions_np.int32)
    pos = 0
    for u in range(nu):
        s, e = (u_ptr[u], u_ptr[u + 1])
        L = e - s
        if L < 2:
            continue
        start_pos = pos
        for k in range(L - 1):
            a_i, b_i = (u_items[s + k], u_items[s + k + 1])
            if a_i != b_i:
                earlier[pos] = a_i
                later[pos] = b_i
                u_owner[pos] = u
                pos += 1
        for k in range(L - 3):
            a_i, b_i = (u_items[s + k], u_items[s + k + 3])
            if a_i != b_i:
                earlier[pos] = a_i
                later[pos] = b_i
                u_owner[pos] = u
                pos += 1
        for k in range(L - 10):
            a_i, b_i = (u_items[s + k], u_items[s + k + 10])
            if a_i != b_i:
                earlier[pos] = a_i
                later[pos] = b_i
                u_owner[pos] = u
                pos += 1
        if L > 2:
            a_i, b_i = (u_items[s], u_items[e - 1])
            if a_i != b_i:
                earlier[pos] = a_i
                later[pos] = b_i
                u_owner[pos] = u
                pos += 1
        np_u = pos - start_pos
        if np_u > 0:
            weights[start_pos:pos] = 1.0 / interactions_np.sqrt(float(np_u))
    return (earlier[:pos], later[:pos], weights[:pos], u_owner[:pos])

def interactions__solve_bt_lbfgs(ear, lat, w, n_items, reg=1.0, maxiter=25):

    def loss_and_grad(era):
        z = era[lat] - era[ear]
        loss = float(interactions_np.sum(w * interactions_np.logaddexp(0.0, -z)) + 0.5 * reg * interactions_np.sum(era * era))
        g = w * interactions_expit(-z)
        grad = interactions_np.bincount(ear, weights=g, minlength=n_items)
        grad -= interactions_np.bincount(lat, weights=g, minlength=n_items)
        grad += reg * era
        return (loss, grad)
    res = interactions_minimize(loss_and_grad, interactions_np.zeros(n_items, dtype=interactions_np.float64), method='L-BFGS-B', jac=True, options={'maxiter': maxiter})
    return (res.x, res)

def interactions_fit_order_tendency(u_ptr, u_items, n_items, reg=1.0, maxiter=25, compute_diagnostics=False):
    """Fit Bradley-Terry order_tendency axis e_i and curvature proxy J_i."""
    ear, lat, w, u_owner = interactions_extract_bt_pairs(u_ptr, u_items)
    era, res = interactions__solve_bt_lbfgs(ear, lat, w, n_items, reg=reg, maxiter=maxiter)
    z = era[lat] - era[ear]
    p = interactions_expit(z)
    cw = w * p * (1.0 - p)
    J = reg + interactions_np.bincount(ear, weights=cw, minlength=n_items) + interactions_np.bincount(lat, weights=cw, minlength=n_items)
    pair_counts = interactions_np.bincount(ear, minlength=n_items) + interactions_np.bincount(lat, minlength=n_items)
    diag = {}
    if compute_diagnostics:
        in_sample_acc = float(interactions_np.mean((z > 0) + 0.5 * (z == 0)))
        u_split = u_owner % 5 != 0
        era_tr, _ = interactions__solve_bt_lbfgs(ear[u_split], lat[u_split], w[u_split], n_items, reg=reg, maxiter=maxiter)
        ho = ~u_split
        z_ho = era_tr[lat[ho]] - era_tr[ear[ho]]
        seen_tr = interactions_np.bincount(ear[u_split], minlength=n_items) + interactions_np.bincount(lat[u_split], minlength=n_items) > 0
        both_seen = seen_tr[ear[ho]] & seen_tr[lat[ho]]
        oos_acc_all = float(interactions_np.mean((z_ho > 0) + 0.5 * (z_ho == 0)))
        oos_acc_both_seen = float(interactions_np.mean((z_ho[both_seen] > 0) + 0.5 * (z_ho[both_seen] == 0)))
        p_min = interactions_np.minimum(ear, lat).astype(interactions_np.int64)
        p_max = interactions_np.maximum(ear, lat).astype(interactions_np.int64)
        keys = p_min * interactions_np.int64(n_items) + p_max
        dirs = (ear < lat).astype(interactions_np.int64)
        order_k = interactions_np.argsort(keys)
        keys_s = keys[order_k]
        dirs_s = dirs[order_k]
        _, idx_start, counts = interactions_np.unique(keys_s, return_index=True, return_counts=True)
        multi = counts > 1
        dir_sum = interactions_np.add.reduceat(dirs_s, idx_start)[multi]
        cnt_multi = counts[multi]
        conflicts = int(interactions_np.sum((dir_sum > 0) & (dir_sum < cnt_multi)))
        n_multi = int(len(cnt_multi))
        diag = {'n_directed_pairs': int(len(ear)), 'n_unique_unordered_pairs': int(len(counts)), 'n_multi_observation_pairs': n_multi, 'n_conflicting_pairs': conflicts, 'bidirectional_conflict_ratio': float(conflicts / max(1, n_multi)), 'in_sample_pair_accuracy': in_sample_acc, 'oos_cross_user_pair_accuracy_all': oos_acc_all, 'oos_cross_user_pair_accuracy_both_seen': oos_acc_both_seen, 'n_oos_pairs_all': int(interactions_np.sum(ho)), 'n_oos_pairs_both_seen': int(interactions_np.sum(both_seen)), 'lbfgs_iterations': int(res.nit), 'lbfgs_final_loss': float(res.fun), 'era_std': float(interactions_np.std(era)), 'mean_item_pair_count': float(interactions_np.mean(pair_counts)), 'items_with_pairs': int(interactions_np.sum(pair_counts > 0))}
    return (era, J, pair_counts, diag)

@interactions_njit(cache=True)
def interactions_task5_bt_features(q_u, q_i1, q_i2, u_ptr, u_items, i_cnt, era, J, reg=1.0):
    """Compute candidate features relative to user u's recent order_tendency E_u:
    decayed mean of last 3, 5, 10 items, 90th percentile q90(u), recent trend E_{u,last3} - E_{u,first3},
    e_i - E_u, |e_i - E_u|, max(0, e_i - E_u), max(0, E_u - e_i), indicator e_i > q90(u),
    e_i - e_{i_{u,L}}, reliability log(J_i / lambda), and unknown-item flags (e_i=0, conf=0, missing=1 when c_i=0).
    """
    n = len(q_u)
    nu = len(u_ptr) - 1
    ni = len(i_cnt)
    d_out = interactions_np.zeros((n, 16), dtype=interactions_np.float64)
    s_out = interactions_np.zeros((n, 16), dtype=interactions_np.float64)
    u_ctx = interactions_np.zeros((n, 4), dtype=interactions_np.float64)
    E_dec3 = interactions_np.zeros(nu + 1, dtype=interactions_np.float64)
    E_dec5 = interactions_np.zeros(nu + 1, dtype=interactions_np.float64)
    E_dec10 = interactions_np.zeros(nu + 1, dtype=interactions_np.float64)
    q90_u = interactions_np.zeros(nu + 1, dtype=interactions_np.float64)
    trend_u = interactions_np.zeros(nu + 1, dtype=interactions_np.float64)
    e_last_u = interactions_np.zeros(nu + 1, dtype=interactions_np.float64)
    u_conf = interactions_np.zeros(nu + 1, dtype=interactions_np.float64)
    for u in range(nu):
        s, e = (u_ptr[u], u_ptr[u + 1])
        L = e - s
        if L == 0:
            continue
        seq_e = interactions_np.empty(L, dtype=interactions_np.float64)
        c_sum = 0.0
        for p in range(L):
            it = u_items[s + p]
            if it < ni and i_cnt[it] > 0:
                seq_e[p] = era[it]
                c_sum += interactions_np.log(J[it] / reg)
            else:
                seq_e[p] = 0.0
        u_conf[u] = c_sum / float(L)
        e_last_u[u] = seq_e[L - 1]
        w3, s3 = (0.0, 0.0)
        for k in range(min(L, 3)):
            w = 0.8 ** k
            s3 += w * seq_e[L - 1 - k]
            w3 += w
        E_dec3[u] = s3 / w3
        w5, s5 = (0.0, 0.0)
        for k in range(min(L, 5)):
            w = 0.8 ** k
            s5 += w * seq_e[L - 1 - k]
            w5 += w
        E_dec5[u] = s5 / w5
        w10, s10 = (0.0, 0.0)
        for k in range(min(L, 10)):
            w = 0.85 ** k
            s10 += w * seq_e[L - 1 - k]
            w10 += w
        E_dec10[u] = s10 / w10
        m_last3, m_first3 = (0.0, 0.0)
        k3 = min(L, 3)
        for k in range(k3):
            m_last3 += seq_e[L - 1 - k]
            m_first3 += seq_e[k]
        trend_u[u] = (m_last3 - m_first3) / float(k3)
        sorted_e = interactions_np.sort(seq_e)
        idx90 = 0.9 * float(L - 1)
        lo = int(interactions_np.floor(idx90))
        hi = int(interactions_np.ceil(idx90))
        q90_u[u] = sorted_e[lo] + (idx90 - float(lo)) * (sorted_e[hi] - sorted_e[lo])
    for idx in range(n):
        u, i1, i2 = (q_u[idx], q_i1[idx], q_i2[idx])
        u_eff = u if u < nu else nu
        ed3 = E_dec3[u_eff]
        ed5 = E_dec5[u_eff]
        ed10 = E_dec10[u_eff]
        q90 = q90_u[u_eff]
        tr_u = trend_u[u_eff]
        elast = e_last_u[u_eff]
        u_ctx[idx, 0] = ed3
        u_ctx[idx, 1] = q90
        u_ctx[idx, 2] = tr_u
        u_ctx[idx, 3] = u_conf[u_eff]
        f_cand = interactions_np.zeros((2, 16), dtype=interactions_np.float64)
        for side in range(2):
            item = i1 if side == 0 else i2
            ci = float(i_cnt[item]) if item < ni else 0.0
            if item < ni and ci > 0.0:
                ei = era[item]
                conf_i = interactions_np.log(J[item] / reg)
                miss_i = 0.0
                rel_i = conf_i / (conf_i + 1.0)
            else:
                ei = 0.0
                conf_i = 0.0
                miss_i = 1.0
                rel_i = 0.0
            diff3 = ei - ed3
            diff5 = ei - ed5
            diff10 = ei - ed10
            difflast = ei - elast
            f_cand[side, 0] = diff3
            f_cand[side, 1] = abs(diff3)
            f_cand[side, 2] = max(0.0, diff3)
            f_cand[side, 3] = max(0.0, -diff3)
            f_cand[side, 4] = abs(diff5)
            f_cand[side, 5] = max(0.0, diff5)
            f_cand[side, 6] = max(0.0, -diff5)
            f_cand[side, 7] = abs(diff10)
            f_cand[side, 8] = max(0.0, diff10)
            f_cand[side, 9] = max(0.0, -diff10)
            f_cand[side, 10] = 1.0 if miss_i == 0.0 and ei > q90 else 0.0
            f_cand[side, 11] = abs(difflast)
            f_cand[side, 12] = max(0.0, difflast)
            f_cand[side, 13] = max(0.0, -difflast)
            f_cand[side, 14] = rel_i * diff3
            f_cand[side, 15] = conf_i
        for k in range(16):
            d_out[idx, k] = f_cand[0, k] - f_cand[1, k]
            s_out[idx, k] = f_cand[0, k] + f_cand[1, k]
    return (d_out, s_out, u_ctx)

@interactions_njit(cache=True)
def interactions_task4_session_window_features(q_u, q_i1, q_i2, q_t, u_ptr, u_items, u_times, i_ptr, i_users, i_times_sorted, u_cnt, i_cnt, all_times_sorted, i_unit):
    """Compute:
    - Global-event-normalized asymmetric window lifts for h in {1h, 1d, 7d, 30d, 180d}:
      L^-, L^+, L^+ - L^-, and window support.
    - Multi-scale session similarity for tau in {600s, 3600s, 86400s, 7d, 30d} using both
      co-occurrence kernel K_co(i, j) and spectral unit kernel K_spec(i, j).
    - Nearest past (t_uj <= t) and nearest future (t_uj > t) item co-occurrence lift and spectral similarity.
    - Same-hour / same-day user activity counts and session support.
    """
    n = len(q_u)
    d_out = interactions_np.zeros((n, 34), dtype=interactions_np.float64)
    s_out = interactions_np.zeros((n, 34), dtype=interactions_np.float64)
    u_ctx = interactions_np.zeros((n, 14), dtype=interactions_np.float64)
    nu = len(u_cnt)
    ni = len(i_cnt)
    n_total = float(len(all_times_sorted))
    denom_q = n_total + float(ni)
    dim = i_unit.shape[1]
    in_u1 = interactions_np.zeros(nu, dtype=interactions_np.float64)
    in_u2 = interactions_np.zeros(nu, dtype=interactions_np.float64)
    inv_u = 1.0 / (u_cnt.astype(interactions_np.float64) + 2.0)
    hs = interactions_np.array([3600, 86400, 7 * 86400, 30 * 86400, 180 * 86400], dtype=interactions_np.int64)
    taus = interactions_np.array([600.0, 3600.0, 86400.0, 7.0 * 86400.0, 30.0 * 86400.0], dtype=interactions_np.float64)
    alpha = 10.0
    lam_tau = 0.5
    for idx in range(n):
        u, i1, i2, t = (q_u[idx], q_i1[idx], q_i2[idx], q_t[idx])
        f_cand = interactions_np.zeros((2, 34), dtype=interactions_np.float64)
        pos_t_left = interactions_np.searchsorted(all_times_sorted, t, side='left')
        pos_t_right = interactions_np.searchsorted(all_times_sorted, t, side='right')
        for w_idx in range(5):
            h = hs[w_idx]
            N_minus = float(pos_t_left - interactions_np.searchsorted(all_times_sorted, t - h, side='left'))
            N_plus = float(interactions_np.searchsorted(all_times_sorted, t + h, side='right') - pos_t_right)
            if w_idx < 3:
                u_ctx[idx, 9 + w_idx] = interactions_np.log1p(N_minus + N_plus)
            for side in range(2):
                item = i1 if side == 0 else i2
                ci = float(i_cnt[item]) if item < ni else 0.0
                qi = (ci + 1.0) / denom_q
                log_qi = interactions_np.log(qi)
                if item < ni and ci > 0.0:
                    ts = i_times_sorted[i_ptr[item]:i_ptr[item + 1]]
                    p_left = interactions_np.searchsorted(ts, t, side='left')
                    p_right = interactions_np.searchsorted(ts, t, side='right')
                    n_m = float(p_left - interactions_np.searchsorted(ts, t - h, side='left'))
                    n_p = float(interactions_np.searchsorted(ts, t + h, side='right') - p_right)
                else:
                    n_m = 0.0
                    n_p = 0.0
                L_m = interactions_np.log((n_m + alpha * qi) / (N_minus + alpha)) - log_qi
                L_p = interactions_np.log((n_p + alpha * qi) / (N_plus + alpha)) - log_qi
                base_col = w_idx * 4
                f_cand[side, base_col + 0] = L_m
                f_cand[side, base_col + 1] = L_p
                f_cand[side, base_col + 2] = L_p - L_m
                f_cand[side, base_col + 3] = interactions_np.log1p(n_m + n_p)
        if u < nu:
            ustart, uend = (u_ptr[u], u_ptr[u + 1])
            if uend > ustart:
                c1 = float(i_cnt[i1]) if i1 < ni else 0.0
                c2 = float(i_cnt[i2]) if i2 < ni else 0.0
                if i1 < ni:
                    for p in range(i_ptr[i1], i_ptr[i1 + 1]):
                        in_u1[i_users[p]] = inv_u[i_users[p]]
                if i2 < ni:
                    for p in range(i_ptr[i2], i_ptr[i2 + 1]):
                        in_u2[i_users[p]] = inv_u[i_users[p]]
                v1_vec = i_unit[i1]
                v2_vec = i_unit[i2]
                W_tau = interactions_np.zeros(5, dtype=interactions_np.float64)
                num_co = interactions_np.zeros((2, 5), dtype=interactions_np.float64)
                num_sp = interactions_np.zeros((2, 5), dtype=interactions_np.float64)
                best_past_dt = 1e+18
                best_fut_dt = 1e+18
                past_lift1, past_lift2 = (0.0, 0.0)
                fut_lift1, fut_lift2 = (0.0, 0.0)
                past_sp1, past_sp2 = (0.0, 0.0)
                fut_sp1, fut_sp2 = (0.0, 0.0)
                same_hr = 0.0
                same_day = 0.0
                for pos_idx in range(ustart, uend):
                    j = u_items[pos_idx]
                    if j == i1 or j == i2:
                        continue
                    t_uj = u_times[pos_idx]
                    dt = float(abs(t - t_uj))
                    if dt <= 3600.0:
                        same_hr += 1.0
                    if dt <= 86400.0:
                        same_day += 1.0
                    cj = float(i_cnt[j])
                    inv_cj = 1.0 / (cj + 5.0)
                    jstart, jend = (i_ptr[j], i_ptr[j + 1])
                    w1, w2 = (0.0, 0.0)
                    for q in range(jstart, jend):
                        up = i_users[q]
                        w1 += in_u1[up]
                        w2 += in_u2[up]
                    v1_co = w1 * inv_cj
                    v2_co = w2 * inv_cj
                    k_co1 = w1 / interactions_np.sqrt((c1 + 5.0) * (cj + 5.0))
                    k_co2 = w2 / interactions_np.sqrt((c2 + 5.0) * (cj + 5.0))
                    vj_vec = i_unit[j]
                    k_sp1, k_sp2 = (0.0, 0.0)
                    for k in range(dim):
                        k_sp1 += v1_vec[k] * vj_vec[k]
                        k_sp2 += v2_vec[k] * vj_vec[k]
                    for m in range(5):
                        wm = interactions_np.exp(-dt / taus[m])
                        W_tau[m] += wm
                        num_co[0, m] += wm * k_co1
                        num_co[1, m] += wm * k_co2
                        num_sp[0, m] += wm * k_sp1
                        num_sp[1, m] += wm * k_sp2
                    if t_uj <= t:
                        dt_p = float(t - t_uj)
                        if dt_p < best_past_dt:
                            best_past_dt = dt_p
                            past_lift1 = v1_co
                            past_lift2 = v2_co
                            past_sp1 = k_sp1
                            past_sp2 = k_sp2
                    else:
                        dt_f = float(t_uj - t)
                        if dt_f < best_fut_dt:
                            best_fut_dt = dt_f
                            fut_lift1 = v1_co
                            fut_lift2 = v2_co
                            fut_sp1 = k_sp1
                            fut_sp2 = k_sp2
                if i1 < ni:
                    for p in range(i_ptr[i1], i_ptr[i1 + 1]):
                        in_u1[i_users[p]] = 0.0
                if i2 < ni:
                    for p in range(i_ptr[i2], i_ptr[i2 + 1]):
                        in_u2[i_users[p]] = 0.0
                for m in range(5):
                    denom_m = lam_tau + W_tau[m]
                    f_cand[0, 20 + m] = interactions_np.log(num_co[0, m] / denom_m + 1e-05)
                    f_cand[1, 20 + m] = interactions_np.log(num_co[1, m] / denom_m + 1e-05)
                    f_cand[0, 25 + m] = num_sp[0, m] / denom_m
                    f_cand[1, 25 + m] = num_sp[1, m] / denom_m
                    u_ctx[idx, 2 + m] = interactions_np.log1p(W_tau[m])
                f_cand[0, 30] = interactions_np.log((past_lift1 + 1e-06) / (c1 + 1.0))
                f_cand[1, 30] = interactions_np.log((past_lift2 + 1e-06) / (c2 + 1.0))
                f_cand[0, 31] = interactions_np.log((fut_lift1 + 1e-06) / (c1 + 1.0))
                f_cand[1, 31] = interactions_np.log((fut_lift2 + 1e-06) / (c2 + 1.0))
                f_cand[0, 32] = past_sp1
                f_cand[1, 32] = past_sp2
                f_cand[0, 33] = fut_sp1
                f_cand[1, 33] = fut_sp2
                u_ctx[idx, 0] = interactions_np.log1p(same_hr)
                u_ctx[idx, 1] = interactions_np.log1p(same_day)
                u_ctx[idx, 7] = interactions_np.log1p(best_past_dt / 3600.0) if best_past_dt < 1e+17 else 10.0
                u_ctx[idx, 8] = interactions_np.log1p(best_fut_dt / 3600.0) if best_fut_dt < 1e+17 else 10.0
                u_ctx[idx, 12] = 1.0 if best_past_dt < 1e+17 else 0.0
                u_ctx[idx, 13] = 1.0 if best_fut_dt < 1e+17 else 0.0
        for k in range(34):
            d_out[idx, k] = f_cand[0, k] - f_cand[1, k]
            s_out[idx, k] = f_cand[0, k] + f_cand[1, k]
    return (d_out, s_out, u_ctx)

@interactions_njit(cache=True)
def interactions_task2_stage_features(q_u, q_i1, q_i2, u_ptr, u_items, u_times, i_ptr, i_times, i_cnt, all_times_sorted, i_unit):
    """Split user u's timestamped history into S=3 equal-count temporal segments,
    compute segment unit-spectral centroids m_{us} and segment time bounds, and evaluate:
    T_{ui} = sum_{s=0}^2 pi_{us} exp(2.0 <v_i, m_{us}>) f_hat_i(t_bar_{us}) / (g_hat(t_bar_{us}) + eps).
    """
    n = len(q_u)
    d_out = interactions_np.zeros((n, 9), dtype=interactions_np.float64)
    s_out = interactions_np.zeros((n, 9), dtype=interactions_np.float64)
    nu = len(u_ptr) - 1
    ni = len(i_cnt)
    n_total = float(len(all_times_sorted))
    dim = i_unit.shape[1]
    W90 = 90 * 86400
    W30 = 30 * 86400
    kappa = 5.0
    for idx in range(n):
        u, i1, i2 = (q_u[idx], q_i1[idx], q_i2[idx])
        if u >= nu:
            continue
        ustart, uend = (u_ptr[u], u_ptr[u + 1])
        L_raw = uend - ustart
        if L_raw == 0:
            continue
        valid_pos = interactions_np.empty(L_raw, dtype=interactions_np.int32)
        L_eff = 0
        for p in range(ustart, uend):
            j = u_items[p]
            if j != i1 and j != i2:
                valid_pos[L_eff] = p
                L_eff += 1
        if L_eff == 0:
            continue
        m_us = interactions_np.zeros((3, dim), dtype=interactions_np.float64)
        pi_us = interactions_np.zeros(3, dtype=interactions_np.float64)
        t_bar = interactions_np.zeros(3, dtype=interactions_np.int64)
        t_min_s = interactions_np.zeros(3, dtype=interactions_np.int64)
        t_max_s = interactions_np.zeros(3, dtype=interactions_np.int64)
        g_A = interactions_np.zeros(3, dtype=interactions_np.float64)
        g_B = interactions_np.zeros(3, dtype=interactions_np.float64)
        for s in range(3):
            s_lo = s * L_eff // 3
            s_hi = (s + 1) * L_eff // 3
            if s_hi <= s_lo:
                continue
            pi_us[s] = float(s_hi - s_lo) / float(L_eff)
            for p in range(s_lo, s_hi):
                j = u_items[valid_pos[p]]
                vj = i_unit[j]
                for k in range(dim):
                    m_us[s, k] += vj[k]
            norm_m = 0.0
            for k in range(dim):
                norm_m += m_us[s, k] * m_us[s, k]
            norm_m = interactions_np.sqrt(norm_m) + 1e-08
            for k in range(dim):
                m_us[s, k] /= norm_m
            tb = u_times[valid_pos[(s_lo + s_hi) // 2]]
            tmin = u_times[valid_pos[s_lo]]
            tmax = u_times[valid_pos[s_hi - 1]]
            t_bar[s] = tb
            t_min_s[s] = tmin
            t_max_s[s] = tmax
            n_gA = interactions_np.searchsorted(all_times_sorted, tb + W90, side='right') - interactions_np.searchsorted(all_times_sorted, tb - W90, side='left')
            g_A[s] = (float(n_gA) + 1.0) / (n_total + 1.0)
            n_gB = interactions_np.searchsorted(all_times_sorted, tmax + W30, side='right') - interactions_np.searchsorted(all_times_sorted, tmin - W30, side='left')
            g_B[s] = (float(n_gB) + 1.0) / (n_total + 1.0)
        f_cand = interactions_np.zeros((2, 9), dtype=interactions_np.float64)
        for side in range(2):
            item = i1 if side == 0 else i2
            ci = float(i_cnt[item]) if item < ni else 0.0
            vi = i_unit[item]
            T_A = 0.0
            T_B = 0.0
            T_spec = 0.0
            max_sim = -1.0
            min_sim = 1.0
            lift_A = 0.0
            lift_B = 0.0
            max_lift_A = 0.0
            max_term_A = 0.0
            for s in range(3):
                if pi_us[s] <= 0.0:
                    continue
                sim = 0.0
                for k in range(dim):
                    sim += vi[k] * m_us[s, k]
                if sim > max_sim:
                    max_sim = sim
                if sim < min_sim:
                    min_sim = sim
                exp_sim = interactions_np.exp(2.0 * sim)
                if item < ni and ci > 0.0:
                    ts = i_times[i_ptr[item]:i_ptr[item + 1]]
                    n_iA = interactions_np.searchsorted(ts, t_bar[s] + W90, side='right') - interactions_np.searchsorted(ts, t_bar[s] - W90, side='left')
                    f_iA = (float(n_iA) + kappa * g_A[s]) / (ci + kappa)
                    R_A = f_iA / (g_A[s] + 1e-08)
                    n_iB = interactions_np.searchsorted(ts, t_max_s[s] + W30, side='right') - interactions_np.searchsorted(ts, t_min_s[s] - W30, side='left')
                    f_iB = (float(n_iB) + kappa * g_B[s]) / (ci + kappa)
                    R_B = f_iB / (g_B[s] + 1e-08)
                else:
                    R_A = 1.0
                    R_B = 1.0
                term_A = pi_us[s] * exp_sim * R_A
                term_B = pi_us[s] * exp_sim * R_B
                T_A += term_A
                T_B += term_B
                T_spec += pi_us[s] * exp_sim
                lift_A += pi_us[s] * R_A
                lift_B += pi_us[s] * R_B
                if R_A > max_lift_A:
                    max_lift_A = R_A
                if term_A > max_term_A:
                    max_term_A = term_A
            f_cand[side, 0] = interactions_np.log(T_A + 1e-06)
            f_cand[side, 1] = interactions_np.log(T_B + 1e-06)
            f_cand[side, 2] = interactions_np.log(T_spec + 1e-06)
            f_cand[side, 3] = max_sim
            f_cand[side, 4] = min_sim
            f_cand[side, 5] = interactions_np.log(lift_A + 1e-06)
            f_cand[side, 6] = interactions_np.log(lift_B + 1e-06)
            f_cand[side, 7] = interactions_np.log(max_lift_A + 1e-06)
            f_cand[side, 8] = max_term_A / (T_A + 1e-08)
        for k in range(9):
            d_out[idx, k] = f_cand[0, k] - f_cand[1, k]
            s_out[idx, k] = f_cand[0, k] + f_cand[1, k]
    return (d_out, s_out)

def interactions_fit_equivariant_pair_embedding(a_fit, nu, ni, u_imp, s_imp, vt_imp, inv_i, steps=20, lr=0.05, reg=0.01):
    """Deterministic mean-negative pair embedding; this is a surrogate, not exact BPR."""
    signs = interactions_np.where(interactions_np.sum(u_imp ** 3, axis=0) >= 0, 1.0, -1.0)
    u_c = u_imp * signs[None, :]
    v_c = vt_imp.T * signs[None, :]
    k = min(32, u_c.shape[1])
    P = (u_c[:, -k:] * s_imp[-k:] ** 0.5).astype(interactions_np.float64)
    Q = (v_c[:, -k:] * s_imp[-k:] ** 0.5 * inv_i[:, None] ** 0.5).astype(interactions_np.float64)
    u_idx, i_idx = (a_fit[:, 0], a_fit[:, 1])
    i_cnt = interactions_np.bincount(i_idx, minlength=ni).astype(interactions_np.float64)
    pop_prob = i_cnt / max(1.0, float(interactions_np.sum(i_cnt)))
    for _ in range(steps):
        q_bar = pop_prob @ Q
        margin = interactions_np.sum(P[u_idx] * (Q[i_idx] - q_bar[None, :]), axis=1)
        g = interactions_expit(-interactions_np.clip(margin, -20.0, 20.0))
        G_mat = interactions_csr_matrix((g, (u_idx, i_idx)), shape=(nu, ni))
        grad_P = G_mat @ Q - interactions_np.asarray(G_mat.sum(axis=1)) * q_bar[None, :] - reg * P
        grad_Q = G_mat.T @ P - interactions_np.outer(pop_prob, interactions_np.sum(g[:, None] * P[u_idx], axis=0)) - reg * Q
        P += lr * (grad_P / (interactions_np.linalg.norm(grad_P, axis=0, keepdims=True) + 1e-08))
        Q += lr * (grad_Q / (interactions_np.linalg.norm(grad_Q, axis=0, keepdims=True) + 1e-08))
    P_ext = interactions_np.vstack([P, interactions_np.zeros((1, k), dtype=interactions_np.float64)])
    Q_ext = interactions_np.vstack([Q, interactions_np.zeros((1, k), dtype=interactions_np.float64)])
    return (P_ext, Q_ext)

def interactions_stable_unit_rows(values, threshold=1e-07):
    """Zero numerical null-space projections before normalization.

    Two successive normalizations otherwise amplify floating point noise for
    disconnected components absent from the retained spectral subspace.
    """
    norms = interactions_np.linalg.norm(values, axis=1, keepdims=True)
    return interactions_np.divide(values, norms, out=interactions_np.zeros_like(values), where=norms > threshold)

def interactions_fit_graph_state(task, a_fit, nu, ni, bpr_epochs=25, compute_bt_diag=False, equivariant_mode=False):
    """Fit all graph/spectral/BPR/order_tendency structures on a_fit once, so multiple
    query sets (e.g. va_A, va_B, test, single-row) can be scored against the exact same state.
    """
    u_cnt = interactions_np.bincount(a_fit[:, 0], minlength=nu)
    i_cnt = interactions_np.bincount(a_fit[:, 1], minlength=ni)
    inv_u = 1.0 / interactions_np.sqrt(u_cnt + 5.0)
    inv_i = 1.0 / interactions_np.sqrt(i_cnt + 10.0)
    state = {'task': task, 'a_fit': a_fit, 'nu': nu, 'ni': ni, 'u_cnt': u_cnt, 'i_cnt': i_cnt, 'inv_u': inv_u, 'inv_i': inv_i}
    if task in (2, 4):
        u_order = interactions_np.lexsort((a_fit[:, 2], a_fit[:, 0]))
        u_ptr = interactions_np.r_[0, u_cnt.cumsum()]
        u_items = a_fit[u_order, 1].astype(interactions_np.int32)
        u_times = a_fit[u_order, 2]
        u_rats = a_fit[u_order, 3]
        i_order = interactions_np.lexsort((a_fit[:, 2], a_fit[:, 1]))
        i_ptr = interactions_np.r_[0, i_cnt.cumsum()]
        i_users = a_fit[i_order, 0].astype(interactions_np.int32)
        i_times = a_fit[i_order, 2]
        all_times_sorted = interactions_np.sort(a_fit[:, 2])
        x_imp = interactions_csr_matrix((inv_u[a_fit[:, 0]] * inv_i[a_fit[:, 1]], (a_fit[:, 0], a_fit[:, 1])), shape=(nu, ni))
        u_imp, s_imp, vt_imp = interactions_svds(x_imp, k=64, random_state=SEED)
        u_vec = interactions_np.vstack([(u_imp * s_imp).astype(interactions_np.float32), interactions_np.zeros((1, 64), interactions_np.float32)])
        i_vec = interactions_np.vstack([(vt_imp.T * inv_i[:, None]).astype(interactions_np.float32), interactions_np.zeros((1, 64), interactions_np.float32)])
        u_emb = (u_imp * s_imp ** 0.5).astype(interactions_np.float32)
        u_unit = interactions_stable_unit_rows(u_emb)
        i_cent = interactions_np.zeros((ni + 1, 64), dtype=interactions_np.float32)
        interactions_np.add.at(i_cent, a_fit[:, 1], u_unit[a_fit[:, 0]])
        i_unit = interactions_stable_unit_rows(i_cent)
        u_unit_ext = interactions_np.vstack([u_unit, interactions_np.zeros((1, 64), interactions_np.float32)])
        p, q = train_bpr_pop(a_fit, nu, ni, bpr_epochs)
        p = interactions_np.vstack([p, interactions_np.zeros((1, 32), interactions_np.float32)])
        q = interactions_np.vstack([q, interactions_np.zeros((1, 32), interactions_np.float32)])
        state.update({'u_ptr': u_ptr, 'u_items': u_items, 'u_times': u_times, 'u_rats': u_rats, 'i_ptr': i_ptr, 'i_users': i_users, 'i_times': i_times, 'all_times_sorted': all_times_sorted, 'u_vec': u_vec, 'i_vec': i_vec, 'u_unit_ext': u_unit_ext, 'i_unit': i_unit, 'p_bpr': p, 'q_bpr': q})
        return state
    u_order = interactions_np.argsort(a_fit[:, 0], kind='mergesort')
    u_ptr = interactions_np.r_[0, u_cnt.cumsum()]
    u_items = a_fit[u_order, 1].astype(interactions_np.int32)
    u_rats = a_fit[u_order, 3]
    u_pos = interactions_np.zeros(len(a_fit), dtype=interactions_np.int32)
    u_rel_pos = interactions_np.zeros(len(a_fit), dtype=interactions_np.float64 if equivariant_mode else interactions_np.float32)
    last1_cnt = interactions_np.zeros(ni, dtype=interactions_np.float64 if equivariant_mode else interactions_np.float32)
    last3_cnt = interactions_np.zeros(ni, dtype=interactions_np.float64 if equivariant_mode else interactions_np.float32)
    last5_cnt = interactions_np.zeros(ni, dtype=interactions_np.float64 if equivariant_mode else interactions_np.float32)
    for u in range(nu):
        s, e = (u_ptr[u], u_ptr[u + 1])
        L = e - s
        if L > 0:
            p_idx = interactions_np.arange(L, dtype=interactions_np.int32)
            u_pos[s:e] = p_idx
            u_rel_pos[s:e] = (p_idx + 1.0) / float(L)
            last1_cnt[u_items[e - 1]] += 1.0
            for p_i in range(max(s, e - 3), e):
                last3_cnt[u_items[p_i]] += 1.0
            for p_i in range(max(s, e - 5), e):
                last5_cnt[u_items[p_i]] += 1.0
    row_pos = interactions_np.zeros(len(a_fit), dtype=interactions_np.int32)
    row_rel_pos = interactions_np.zeros(len(a_fit), dtype=interactions_np.float64 if equivariant_mode else interactions_np.float32)
    row_pos[u_order] = u_pos
    row_rel_pos[u_order] = u_rel_pos
    item_rel_sum = interactions_np.bincount(a_fit[:, 1], weights=row_rel_pos, minlength=ni)
    item_rel_mean = ((item_rel_sum + 2.0 * 0.58) / (i_cnt + 2.0)).astype(interactions_np.float32)
    item_rel4_sum = interactions_np.bincount(a_fit[:, 1], weights=row_rel_pos ** 4, minlength=ni)
    item_rel4_mean = ((item_rel4_sum + 2.0 * 0.24) / (i_cnt + 2.0)).astype(interactions_np.float32)
    item_last1_rate = interactions_np.log((last1_cnt + 0.3) / (i_cnt + 3.0)).astype(interactions_np.float32)
    item_last3_rate = interactions_np.log((last3_cnt + 0.6) / (i_cnt + 3.0)).astype(interactions_np.float32)
    item_last5_rate = interactions_np.log((last5_cnt + 1.0) / (i_cnt + 3.0)).astype(interactions_np.float32)
    u_end_era = interactions_np.full(nu + 1, 0.6, dtype=interactions_np.float32)
    for u in range(nu):
        s, e = (u_ptr[u], u_ptr[u + 1])
        if e > s:
            u_end_era[u] = interactions_np.mean(item_rel_mean[u_items[max(s, e - 3):e]])
    i_order = interactions_np.argsort(a_fit[:, 1], kind='mergesort')
    i_ptr = interactions_np.r_[0, i_cnt.cumsum()]
    i_users = a_fit[i_order, 0].astype(interactions_np.int32)
    i_pos = row_pos[i_order]
    rec_weights = (0.5 + 0.5 * row_rel_pos) * inv_u[a_fit[:, 0]] * inv_i[a_fit[:, 1]]
    x_imp = interactions_csr_matrix((rec_weights, (a_fit[:, 0], a_fit[:, 1])), shape=(nu, ni))
    v0 = 1.0 / interactions_np.sqrt(u_cnt + 5.0)
    u_imp, s_imp, vt_imp = interactions_svds(x_imp, k=64, v0=v0, random_state=SEED)
    order_sv = interactions_np.argsort(s_imp)
    u_imp, s_imp, vt_imp = (u_imp[:, order_sv], s_imp[order_sv], vt_imp[order_sv, :])
    u_vec = interactions_np.vstack([(u_imp * s_imp).astype(interactions_np.float64), interactions_np.zeros((1, 64), interactions_np.float64)])
    i_vec = interactions_np.vstack([(vt_imp.T * inv_i[:, None]).astype(interactions_np.float64), interactions_np.zeros((1, 64), interactions_np.float64)])
    u_emb = (u_imp * s_imp ** 0.5).astype(interactions_np.float64)
    u_unit = interactions_stable_unit_rows(u_emb)
    i_cent = interactions_np.zeros((ni + 1, 64), dtype=interactions_np.float64)
    interactions_np.add.at(i_cent, a_fit[:, 1], u_unit[a_fit[:, 0]])
    i_unit = interactions_stable_unit_rows(i_cent).astype(interactions_np.float32)
    u_unit_ext = interactions_np.vstack([u_unit, interactions_np.zeros((1, 64), interactions_np.float64)])
    v_item = (vt_imp.T * s_imp ** 0.5 * inv_i[:, None]).astype(interactions_np.float64)
    u_decay = interactions_np.zeros((nu + 1, 64), dtype=interactions_np.float64)
    u_last1 = interactions_np.zeros((nu + 1, 64), dtype=interactions_np.float64)
    for u in range(nu):
        s, e = (u_ptr[u], u_ptr[u + 1])
        if e > s:
            u_last1[u] = v_item[u_items[e - 1]]
            start_k = max(s, e - 15)
            w_sum = 0.0
            for p_idx in range(e - 1, start_k - 1, -1):
                w = 0.75 ** (e - 1 - p_idx)
                u_decay[u] += w * v_item[u_items[p_idx]]
                w_sum += w
            u_decay[u] /= w_sum
    v_item_ext = interactions_np.vstack([v_item, interactions_np.zeros((1, 64), interactions_np.float64)])
    p_eq, q_eq = interactions_fit_equivariant_pair_embedding(a_fit, nu, ni, u_imp, s_imp, vt_imp, inv_i)
    era, J, pair_counts, bt_diag = interactions_fit_order_tendency(u_ptr, u_items, ni, reg=1.0, maxiter=25, compute_diagnostics=compute_bt_diag)
    state.update({'u_ptr': u_ptr, 'u_items': u_items, 'u_rats': u_rats, 'i_ptr': i_ptr, 'i_users': i_users, 'i_pos': i_pos, 'item_rel_mean': item_rel_mean, 'item_rel4_mean': item_rel4_mean, 'item_last1_rate': item_last1_rate, 'item_last3_rate': item_last3_rate, 'item_last5_rate': item_last5_rate, 'u_end_era': u_end_era, 'u_vec': u_vec, 'i_vec': i_vec, 'u_unit_ext': u_unit_ext, 'i_unit': i_unit, 'u_decay': u_decay, 'u_last1': u_last1, 'v_item_ext': v_item_ext, 'p_bpr': p_eq, 'q_bpr': q_eq, 'era': era, 'J': J, 'bt_pair_counts': pair_counts, 'bt_diag': bt_diag})
    return state

def interactions_extract_query_features(state, q_u, q_i1, q_i2, q_t, precomputed_base=None, precomputed_nb=None):
    """Extract full antisymmetric features d, symmetric context features ctx,
    base features x0, and clean seen_override for any query set (q_u, q_i1, q_i2, q_t).
    """
    task = state['task']
    a_fit = state['a_fit']
    nu, ni = (state['nu'], state['ni'])
    u_cnt, i_cnt = (state['u_cnt'], state['i_cnt'])
    u_ptr, u_items = (state['u_ptr'], state['u_items'])
    seen_override, zero_diff, zero_sum, zero_neq = interactions_compute_seen_and_zero_flags(q_u, q_i1, q_i2, u_ptr, u_items, i_cnt)
    if precomputed_base is not None:
        x0 = precomputed_base.astype(interactions_np.float64)
    else:
        spec_lift = interactions_np.sum(state['u_vec'][q_u] * (state['i_vec'][q_i1] - state['i_vec'][q_i2]), axis=1)
        comm_sim = interactions_np.sum(state['u_unit_ext'][q_u] * (state['i_unit'][q_i1] - state['i_unit'][q_i2]), axis=1)
        f_pool = spectral_history_pooling(q_u, q_i1, q_i2, u_ptr, u_items, state['i_unit'])
        bpr_score = interactions_np.sum(state['p_bpr'][q_u] * (state['q_bpr'][q_i1] - state['q_bpr'][q_i2]), axis=1)
        if task == 2:
            f_pair = task2_pair_features(q_u, q_i1, q_i2, u_ptr, u_items, state['u_times'], state['u_rats'], state['i_ptr'], state['i_users'], state['i_times'], u_cnt, i_cnt)
            x0 = interactions_np.column_stack([f_pair[:, :17], spec_lift, comm_sim, f_pool, bpr_score]).astype(interactions_np.float64)
        elif task == 4:
            f_pair = task4_pair_features(q_u, q_i1, q_i2, q_t, u_ptr, u_items, state['u_times'], state['u_rats'], state['i_ptr'], state['i_users'], state['i_times'], u_cnt, i_cnt)
            x0 = interactions_np.column_stack([f_pair[:, :22], spec_lift, comm_sim, f_pool, bpr_score]).astype(interactions_np.float64)
        else:
            f_seq = task5_seq_features(q_u, q_i1, q_i2, u_ptr, u_items, state['u_rats'], state['i_ptr'], state['i_users'], state['i_pos'], u_cnt, i_cnt, state['item_rel_mean'], state['item_rel4_mean'], state['item_last1_rate'], state['item_last3_rate'], state['item_last5_rate'], state['u_end_era'])
            spec_decay = interactions_np.sum(state['u_decay'][q_u] * (state['v_item_ext'][q_i1] - state['v_item_ext'][q_i2]), axis=1)
            spec_last1 = interactions_np.sum(state['u_last1'][q_u] * (state['v_item_ext'][q_i1] - state['v_item_ext'][q_i2]), axis=1)
            x0 = interactions_np.column_stack([f_seq[:, :23], spec_lift, comm_sim, f_pool, spec_decay, spec_last1, bpr_score]).astype(interactions_np.float64)
    if precomputed_nb is not None:
        d_nb = precomputed_nb[0].astype(interactions_np.float64)
        s_nb = precomputed_nb[1].astype(interactions_np.float64)
    else:
        d_nb, s_nb = interactions_build_neighbor_features(a_fit, nu, ni, q_u, q_i1, q_i2, q_t, task)
        d_nb = d_nb.astype(interactions_np.float64)
        s_nb = s_nb.astype(interactions_np.float64)
    if task == 2:
        d_new, s_new = interactions_task2_stage_features(q_u, q_i1, q_i2, u_ptr, u_items, state['u_times'], state['i_ptr'], state['i_times'], i_cnt, state['all_times_sorted'], state['i_unit'])
        d_full = interactions_np.column_stack([x0, d_nb, d_new, zero_diff])
        s_extra = s_new
    elif task == 4:
        d_new, s_new, u_ctx = interactions_task4_session_window_features(q_u, q_i1, q_i2, q_t, u_ptr, u_items, state['u_times'], state['i_ptr'], state['i_users'], state['i_times'], u_cnt, i_cnt, state['all_times_sorted'], state['i_unit'])
        d_full = interactions_np.column_stack([x0, d_nb, d_new, zero_diff])
        s_extra = interactions_np.column_stack([s_new, u_ctx])
    else:
        d_new, s_new, u_ctx = interactions_task5_bt_features(q_u, q_i1, q_i2, u_ptr, u_items, i_cnt, state['era'], state['J'], reg=1.0)
        d_full = interactions_np.column_stack([x0, d_nb, d_new])
        s_extra = interactions_np.column_stack([s_new, u_ctx])
    uc_ext = interactions_np.r_[u_cnt, 0]
    ic_ext = interactions_np.r_[i_cnt, 0]
    lc1 = interactions_np.log1p(ic_ext[q_i1])
    lc2 = interactions_np.log1p(ic_ext[q_i2])
    base_ctx = interactions_np.column_stack([interactions_np.log1p(uc_ext[q_u]), lc1 + lc2, interactions_np.abs(lc1 - lc2), interactions_np.minimum(lc1, lc2), interactions_np.maximum(lc1, lc2), zero_sum, zero_neq])
    ctx_full = interactions_np.column_stack([base_ctx, s_nb, s_extra, interactions_np.abs(d_full)]).astype(interactions_np.float64)
    return (x0, d_full, ctx_full, seen_override)

def interactions_expand(d, ctx, sign=1.0):
    return interactions_np.column_stack([sign * d, ctx])

def interactions_fit_antisymmetric_hgb(d_tr, ctx_tr, leaves=15, reg=10.0, lr=0.06, max_iter=250):
    model = interactions_HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=leaves, learning_rate=lr, l2_regularization=reg, min_samples_leaf=40, early_stopping=False, random_state=258)
    x_dup = interactions_np.vstack([interactions_expand(d_tr, ctx_tr, 1.0), interactions_expand(d_tr, ctx_tr, -1.0)])
    y_dup = interactions_np.r_[interactions_np.ones(len(d_tr)), interactions_np.zeros(len(d_tr))]
    model.fit(x_dup, y_dup)
    return model

def interactions_score_antisymmetric_hgb(model, d, ctx):
    return 0.5 * (model.decision_function(interactions_expand(d, ctx, 1.0)) - model.decision_function(interactions_expand(d, ctx, -1.0)))

def interactions_compute_bucket_breakdown(base_scores, cand_scores, i_cnt, q_i1, q_i2):
    """Report accuracy by min(c1, c2) frequency buckets {0, 1, 2-5, 6-20, >20},
    sample counts, and paired flip counts / standard errors.
    """
    ni = len(i_cnt)
    ic_ext = interactions_np.r_[i_cnt, 0]
    c1 = ic_ext[q_i1]
    c2 = ic_ext[q_i2]
    min_c = interactions_np.minimum(c1, c2)
    tie_positive_first = interactions_np.random.default_rng(258).integers(0, 2, len(base_scores)).astype(bool)
    b_ok = (base_scores > 0) | (base_scores == 0) & tie_positive_first
    c_ok = (cand_scores > 0) | (cand_scores == 0) & tie_positive_first

    def stats_for_mask(mask):
        cnt = int(interactions_np.sum(mask))
        if cnt == 0:
            return {'count': 0, 'baseline_acc': None, 'candidate_acc': None, 'improvement': None, 'gained': 0, 'lost': 0, 'paired_se': None}
        bm = b_ok[mask]
        cm = c_ok[mask]
        diff = cm.astype(interactions_np.float64) - bm.astype(interactions_np.float64)
        se = float(diff.std(ddof=1) / interactions_np.sqrt(cnt)) if cnt > 1 else 0.0
        return {'count': cnt, 'baseline_acc': float(interactions_np.mean(bm)), 'candidate_acc': float(interactions_np.mean(cm)), 'improvement': float(interactions_np.mean(diff)), 'gained': int(interactions_np.sum(cm & ~bm)), 'lost': int(interactions_np.sum(bm & ~cm)), 'paired_se': se}
    buckets = {'0': min_c == 0, '1': min_c == 1, '2-5': (min_c >= 2) & (min_c <= 5), '6-20': (min_c >= 6) & (min_c <= 20), '>20': min_c > 20}
    return {'overall': stats_for_mask(interactions_np.ones(len(min_c), dtype=bool)), 'by_min_c_bucket': {k: stats_for_mask(m) for k, m in buckets.items()}}

@interactions_njit(cache=True)
def interactions_neighborhood(qu, qi1, qi2, qt, uptr, ui, ut, iptr, iu, ipos, it, norm, task):
    out = interactions_np.zeros((len(qu), 26), interactions_np.float32)
    symmetric = interactions_np.zeros((len(qu), 26), interactions_np.float32)
    nusers = len(uptr) - 1
    stamps = interactions_np.full(nusers, -1, interactions_np.int64)
    ov = interactions_np.zeros((nusers, 5), interactions_np.float32)
    lastpos = interactions_np.zeros(nusers, interactions_np.int32)
    nitems = len(iptr) - 1
    for row in range(len(qu)):
        u = qu[row]
        i1 = qi1[row]
        i2 = qi2[row]
        if u >= nusers:
            continue
        start = uptr[u]
        end = uptr[u + 1]
        L = end - start
        if L == 0:
            continue
        for pos in range(start, end):
            j = ui[pos]
            if j == i1 or j == i2:
                continue
            cj = iptr[j + 1] - iptr[j]
            idf = interactions_np.log(1 + nusers / (cj + 5))
            rec = 1.0 / interactions_np.sqrt(end - pos)
            if task == 4:
                rec = interactions_np.exp(-abs(float(ut[pos] - qt[row])) / (86400 * 365))
            for ptr in range(iptr[j], iptr[j + 1]):
                v = iu[ptr]
                if v == u:
                    continue
                if stamps[v] != row:
                    stamps[v] = row
                    ov[v, :] = 0
                    lastpos[v] = -1
                ov[v, 0] += 1
                ov[v, 1] += idf
                ov[v, 2] += idf * rec
                ov[v, 3] += 1.0 / interactions_np.sqrt(cj + 5)
                ov[v, 4] += rec
                if ipos[ptr] > lastpos[v]:
                    lastpos[v] = ipos[ptr]
        fs = interactions_np.zeros((2, 26), interactions_np.float64)
        for side in range(2):
            item = i1 if side == 0 else i2
            if item >= nitems:
                continue
            count = iptr[item + 1] - iptr[item]
            best = interactions_np.zeros(10, interactions_np.float64)
            nmatch = 0
            for ptr in range(iptr[item], iptr[item + 1]):
                v = iu[ptr]
                if stamps[v] != row:
                    continue
                nmatch += 1
                nv = uptr[v + 1] - uptr[v]
                cosine = ov[v, 0] / interactions_np.sqrt((L + 3.0) * (nv + 3.0))
                idfcos = ov[v, 1] / interactions_np.sqrt((norm[u] + 8.0) * (norm[v] + 8.0))
                recent = ov[v, 2] / interactions_np.sqrt((norm[u] + 8.0) * (norm[v] + 8.0))
                rare = ov[v, 3] / interactions_np.sqrt((L + 3.0) * (nv + 3.0))
                for k, sim in enumerate((cosine, idfcos, recent, rare)):
                    fs[side, k * 4] += sim
                    fs[side, k * 4 + 1] += sim * sim
                    fs[side, k * 4 + 2] += sim * sim * sim
                    if sim > fs[side, k * 4 + 3]:
                        fs[side, k * 4 + 3] = sim
                if task == 5:
                    dist = ipos[ptr] - lastpos[v]
                    if dist > 0:
                        fs[side, 16] += idfcos * idfcos / (dist + 1)
                        if dist <= 3:
                            fs[side, 17] += idfcos * idfcos
                    elif dist >= -3:
                        fs[side, 18] += idfcos * idfcos
                elif task == 4:
                    dt = abs(float(it[ptr] - qt[row])) / 86400
                    fs[side, 16] += idfcos * idfcos * interactions_np.exp(-dt / 7)
                    fs[side, 17] += idfcos * idfcos * interactions_np.exp(-dt / 90)
                    fs[side, 18] += idfcos * idfcos * interactions_np.exp(-dt / 365)
                else:
                    fs[side, 16] += idfcos * idfcos / (nv + 5.0)
                    fs[side, 17] += idfcos * idfcos * ov[v, 0]
                    fs[side, 18] += idfcos * idfcos * ov[v, 1]
                if idfcos > best[0]:
                    best[0] = idfcos
                    best.sort()
            fs[side, 19] = interactions_np.sum(best)
            fs[side, 20] = interactions_np.sum(best[5:])
            fs[side, 21] = interactions_np.sum(best[7:])
            fs[side, 22] = nmatch / (count + 2.0)
            fs[side, 23] = interactions_np.sum(best ** 2)
            fs[side, 24] = interactions_np.sum(best ** 3)
            fs[side, 25] = nmatch
            for k in range(19):
                if k % 4 == 3 and k < 16:
                    fs[side, k] = interactions_np.log(fs[side, k] + 0.001)
                else:
                    fs[side, k] = interactions_np.log((fs[side, k] + 0.001) / (count + 3.0))
            fs[side, 25] = interactions_np.log((nmatch + 0.1) / (count + 3.0))
        for k in range(26):
            out[row, k] = fs[0, k] - fs[1, k]
            symmetric[row, k] = fs[0, k] + fs[1, k]
    return (out, symmetric)

def interactions_build_neighbor_features(a, nu, ni, qu, qi1, qi2, qt, task):
    uc = interactions_np.bincount(a[:, 0], minlength=nu)
    ic = interactions_np.bincount(a[:, 1], minlength=ni)
    uorder = interactions_np.argsort(a[:, 0], kind='stable') if task == 5 else interactions_np.lexsort((a[:, 2], a[:, 0]))
    uptr = interactions_np.r_[0, uc.cumsum()]
    ui = a[uorder, 1]
    ut = a[uorder, 2]
    position = interactions_np.empty(len(a), interactions_np.int64)
    position[uorder] = interactions_np.arange(len(a)) - interactions_np.repeat(uptr[:-1], uc)
    iorder = interactions_np.argsort(a[:, 1], kind='stable')
    iptr = interactions_np.r_[0, ic.cumsum()]
    norm = interactions_np.bincount(a[:, 0], weights=interactions_np.log(1 + nu / (ic[a[:, 1]] + 5)), minlength=nu)
    return interactions_neighborhood(qu, qi1, qi2, qt, uptr, ui, ut, iptr, a[iorder, 0], position[iorder], a[iorder, 2], norm, task)

def interactions_verify_task5_invariance(a_full):
    """On a 500-row validation slice, verify that:
    (a) randomly permuting the order of user blocks in fit_rows, and
    (b) randomly permuting user/item integer IDs
    leave the extracted features and predictions invariant (within 1e-5).
    """
    interactions_log('[Task 5 Invariance Check] Starting mandatory invariance verification on 500-row validation slice...')
    t0 = interactions_time.time()
    sub_mask = a_full[:, 0] < 3000
    a_sub = a_full[sub_mask].copy()
    uniq_u, u_inv = interactions_np.unique(a_sub[:, 0], return_inverse=True)
    uniq_i, i_inv = interactions_np.unique(a_sub[:, 1], return_inverse=True)
    a_sub[:, 0] = u_inv
    a_sub[:, 1] = i_inv
    nu, ni = (len(uniq_u), len(uniq_i))
    fit_rows, va_A, _ = interactions_make_interaction_splits_dual(a_sub, nu, ni, sequential=True, seed=258)
    va_slice = va_A[:500].copy()

    def run_pipeline_on_slice(fit_r, q_u, q_i1, q_i2):
        st = interactions_fit_graph_state(5, fit_r, nu, ni, compute_bt_diag=False, equivariant_mode=True)
        _, d_feat, ctx_feat, seen_ov = interactions_extract_query_features(st, q_u, q_i1, q_i2, interactions_np.zeros(len(q_u), dtype=interactions_np.int64))
        rms = interactions_np.sqrt(interactions_np.mean(d_feat[:250] ** 2, axis=0)) + 1e-08
        w = fit_pairwise_logistic(interactions_np.clip(d_feat[:250] / rms, -8.0, 8.0), l2=25.0)
        preds = interactions_np.clip(d_feat / rms, -8.0, 8.0) @ w + 100.0 * seen_ov
        return (d_feat, ctx_feat, preds)
    d0, c0, p0 = run_pipeline_on_slice(fit_rows, va_slice[:, 0], va_slice[:, 1], va_slice[:, 2])
    rng = interactions_np.random.default_rng(2026)
    u_order = interactions_np.argsort(fit_rows[:, 0], kind='mergesort')
    u_cnt = interactions_np.bincount(fit_rows[:, 0], minlength=nu)
    u_ptr = interactions_np.r_[0, u_cnt.cumsum()]
    u_perm_blocks = rng.permutation(nu)
    blocks = [fit_rows[u_order[u_ptr[u]:u_ptr[u + 1]]] for u in u_perm_blocks if u_ptr[u + 1] > u_ptr[u]]
    fit_block_perm = interactions_np.vstack(blocks)
    d_blk, c_blk, p_blk = run_pipeline_on_slice(fit_block_perm, va_slice[:, 0], va_slice[:, 1], va_slice[:, 2])
    perm_u = rng.permutation(nu)
    perm_i = rng.permutation(ni)
    fit_id_perm = fit_rows.copy()
    fit_id_perm[:, 0] = perm_u[fit_rows[:, 0]]
    fit_id_perm[:, 1] = perm_i[fit_rows[:, 1]]
    d_id, c_id, p_id = run_pipeline_on_slice(fit_id_perm, perm_u[va_slice[:, 0]], perm_i[va_slice[:, 1]], perm_i[va_slice[:, 2]])
    diff_block_feat = float(max(interactions_np.max(interactions_np.abs(d0 - d_blk)), interactions_np.max(interactions_np.abs(c0 - c_blk))))
    diff_block_pred = float(interactions_np.max(interactions_np.abs(p0 - p_blk)))
    diff_id_feat = float(max(interactions_np.max(interactions_np.abs(d0 - d_id)), interactions_np.max(interactions_np.abs(c0 - c_id))))
    diff_id_pred = float(interactions_np.max(interactions_np.abs(p0 - p_id)))
    assert diff_block_feat < 1e-05 and diff_block_pred < 1e-05, f'Block invariance failed: {diff_block_feat}, {diff_block_pred}'
    assert diff_id_feat < 1e-05 and diff_id_pred < 1e-05, f'ID invariance failed: {diff_id_feat}, {diff_id_pred}'
    res = {'n_validation_slice_rows': 500, 'n_subgraph_users': int(nu), 'n_subgraph_items': int(ni), 'user_block_permutation_max_feature_diff': diff_block_feat, 'user_block_permutation_max_prediction_diff': diff_block_pred, 'user_item_id_permutation_max_feature_diff': diff_id_feat, 'user_item_id_permutation_max_prediction_diff': diff_id_pred, 'passed_within_1e_5': True, 'elapsed_seconds': float(interactions_time.time() - t0)}
    interactions_log(f'[Task 5 Invariance Check] PASSED: {res}')
    return res

def interactions_fit_fixed_pair_combiner(d, context, config):
    rms = interactions_np.sqrt(interactions_np.mean(d * d, axis=0)) + 1e-08
    linear = fit_pairwise_logistic(interactions_np.clip(d / rms, -8, 8), l2=10.0)
    hgb = interactions_fit_antisymmetric_hgb(d, context, leaves=config['leaves'], reg=10.0)
    return (rms, linear, hgb)

def interactions_predict_pair_combiner(bundle, d, context, config):
    rms, linear, hgb = bundle
    z_linear = interactions_np.clip(d / rms, -8, 8) @ linear
    z_hgb = interactions_score_antisymmetric_hgb(hgb, d, context)
    return (1 - config['blend']) * z_linear + config['blend'] * z_hgb

def interactions_oriented_pair_accuracy(scores):
    tie_positive_first = interactions_np.random.default_rng(258).integers(0, 2, len(scores)).astype(bool)
    return float(interactions_np.mean((scores > 0) | (scores == 0) & tie_positive_first))

def interactions_production_interaction_task(task, a, users, items, args):
    started = interactions_time.time()
    nu, ni = (len(users), len(items))
    checks = interactions_verify_task5_invariance(a) if task == 5 else {}
    fit, va_a, va_b = interactions_make_interaction_splits_dual(a, nu, ni, sequential=task == 5)
    state = interactions_fit_graph_state(task, fit, nu, ni, equivariant_mode=task == 5)
    queries = (va_a, va_b)
    feature_sets = [interactions_extract_query_features(state, v[:, 0], v[:, 1], v[:, 2], v[:, 3]) for v in queries]
    x0, d, context, seen = feature_sets[1]
    idx = interactions_np.random.default_rng(258).permutation(len(d))
    tr, tune, audit = interactions_np.split(idx, [len(d) // 2, 3 * len(d) // 4])
    config = {'leaves': 15, 'blend': 0.5}
    tuning = [{'selection': 'frozen after tune-only validation', 'leaves': 15, 'blend': 0.5}] if task == 5 else []
    bundle = interactions_fit_fixed_pair_combiner(d[tr], context[tr], config)
    audits = {}
    for label, v, features in zip(('after_holdout', 'before_holdout'), queries, feature_sets):
        bx, dx, cx, sx = features
        b = bundle if label == 'before_holdout' else interactions_fit_fixed_pair_combiner(dx[tr], cx[tr], config)
        score = interactions_predict_pair_combiner(b, dx[audit], cx[audit], config) + 100 * sx[audit]
        rms0 = interactions_np.sqrt(interactions_np.mean(bx[tr] ** 2, axis=0)) + 1e-08
        w0 = fit_pairwise_logistic(interactions_np.clip(bx[tr] / rms0, -8, 8))
        old_score = interactions_np.clip(bx[audit] / rms0, -8, 8) @ w0 + 100 * sx[audit]
        audits[label] = interactions_compute_bucket_breakdown(old_score, score, state['i_cnt'], v[audit, 1], v[audit, 2])
        audits[label]['exact_ties'] = {'baseline': int(interactions_np.sum(old_score == 0)), 'candidate': int(interactions_np.sum(score == 0))}
        reverse = interactions_predict_pair_combiner(b, -dx[audit], cx[audit], config) - 100 * sx[audit]
        checks[label + '_swap_max_error'] = float(interactions_np.max(interactions_np.abs(score + reverse)))
        assert checks[label + '_swap_max_error'] < 1e-08
    full_bundle = interactions_fit_fixed_pair_combiner(d, context, config)
    del state, feature_sets, d, context, x0
    full_state = interactions_fit_graph_state(task, a, nu, ni, equivariant_mode=task == 5)
    tests = read_test(args.data_dir / f'task_{task}_test.csv')
    qu = interactions_np.array([users.get(r['user'], nu) for r in tests], dtype=interactions_np.int64)
    q1 = interactions_np.array([items.get(r['item1'], ni) for r in tests], dtype=interactions_np.int64)
    q2 = interactions_np.array([items.get(r['item2'], ni) for r in tests], dtype=interactions_np.int64)
    qt = interactions_np.array([int(r['timestamp']) for r in tests], dtype=interactions_np.int64) if task == 4 else interactions_np.zeros(len(tests), dtype=interactions_np.int64)
    _, dt, ct, st = interactions_extract_query_features(full_state, qu, q1, q2, qt)
    margins = interactions_predict_pair_combiner(full_bundle, dt, ct, config) + 100 * st
    _, dr, cr, sr = interactions_extract_query_features(full_state, qu[:3], q1[:3], q2[:3], qt[:3])
    short = interactions_predict_pair_combiner(full_bundle, dr, cr, config) + 100 * sr
    checks['single_batch_max_error'] = float(interactions_np.max(interactions_np.abs(short - margins[:3])))
    assert checks['single_batch_max_error'] < 1e-06
    _, swapped_d, swapped_c, swapped_s = interactions_extract_query_features(full_state, qu[:3], q2[:3], q1[:3], qt[:3])
    checks['swapped_query_feature_error'] = float(max(interactions_np.max(interactions_np.abs(dr + swapped_d)), interactions_np.max(interactions_np.abs(cr - swapped_c)), interactions_np.max(interactions_np.abs(sr + swapped_s))))
    assert checks['swapped_query_feature_error'] < 1e-05
    if not interactions_np.all(interactions_np.isfinite(margins)):
        raise ValueError('Invalid pairwise margins')
    predictions = [r['item1'] if m >= 0 else r['item2'] for r, m in zip(tests, margins)]
    write_predictions(args.data_dir / f'task_{task}_test.csv', args.output_dir / f'task_{task}_predictions.csv', predictions)
    metrics = {'model': 'symmetric_context_hgb_linear_graph', 'fixed_config': config, 'audit': audits, 'checks': checks, 'test_rows': len(tests), 'tune_only_selection': tuning, 'seed': 258, 'test_exact_ties': int(interactions_np.sum(margins == 0)), 'elapsed_seconds': round(interactions_time.time() - started, 2), 'negative_sampling': 'before-holdout pool for final fit; both pools audited', 'zero_frequency_hard_override': False}
    log(f"Task {task} audit: { {k: v['overall'] for k, v in audits.items()}}")
    return metrics


# Production delta: fixed validated configuration.


import math as delta_math
import time as delta_time
from datetime import datetime as delta_datetime, timezone as delta_timezone
from pathlib import Path as delta_Path
import numpy as delta_np
from numba import njit as delta_njit
from scipy.optimize import minimize as delta_minimize
from sklearn.ensemble import HistGradientBoostingRegressor as delta_HistGradientBoostingRegressor
delta_SEED = 258
delta_DAY = 86400.0
delta_T_MIN_VALID = T_MIN_VALID
delta_T_MAX_VALID = T_MAX_VALID
delta_CUTOFF_SPECS = [('2016-01', 1451606400), ('2016-07', 1467331200), ('2017-01', 1483228800), ('2017-07', 1498867200)]
delta_HGB_PARAMS = dict(loss='absolute_error', early_stopping=False, random_state=258, max_leaf_nodes=15, l2_regularization=50.0, max_iter=300, learning_rate=0.04, min_samples_leaf=100)

def delta_extract_cutoff_delta_features(histories, item_ts, query_u, query_i, cutoff=delta_T_MAX_VALID):
    n = len(query_u)
    valid_tp = delta_np.zeros(n, dtype=bool)
    u_med = delta_np.zeros(n, dtype=delta_np.float64)
    u_q15 = delta_np.zeros(n, dtype=delta_np.float64)
    u_q25 = delta_np.zeros(n, dtype=delta_np.float64)
    u_rec2 = delta_np.zeros(n, dtype=delta_np.float64)
    u_rec3 = delta_np.zeros(n, dtype=delta_np.float64)
    u_rec5 = delta_np.zeros(n, dtype=delta_np.float64)
    u_last1 = delta_np.zeros(n, dtype=delta_np.float64)
    upper_arr = delta_np.zeros(n, dtype=delta_np.float64)
    lower2 = delta_np.zeros(n, dtype=delta_np.float64)
    na_arr = delta_np.zeros(n, dtype=delta_np.float64)
    ntot_arr = delta_np.zeros(n, dtype=delta_np.float64)
    rem_p15 = delta_np.zeros(n, dtype=delta_np.float64)
    rem_p25 = delta_np.zeros(n, dtype=delta_np.float64)
    rem_p30 = delta_np.zeros(n, dtype=delta_np.float64)
    for idx in range(n):
        u = query_u[idx]
        it = query_i[idx]
        h = histories[u]
        tp = float(h[-1][1])
        valid_tp[idx] = delta_T_MIN_VALID <= tp <= cutoff
        raw_t = [x[1] for x in h]
        clean_d = [float(raw_t[k + 1] - raw_t[k]) for k in range(len(raw_t) - 1) if delta_T_MIN_VALID <= raw_t[k] <= cutoff and delta_T_MIN_VALID <= raw_t[k + 1] <= cutoff]
        if not clean_d:
            clean_d = delta_np.maximum(0.0, delta_np.diff(raw_t)).tolist()
        u_med[idx] = delta_np.median(clean_d)
        u_q15[idx] = delta_np.percentile(clean_d, 15)
        u_q25[idx] = delta_np.percentile(clean_d, 25)
        u_rec2[idx] = delta_np.median(clean_d[-2:])
        u_rec3[idx] = delta_np.median(clean_d[-3:])
        u_rec5[idx] = delta_np.median(clean_d[-5:])
        u_last1[idx] = clean_d[-1]
        if valid_tp[idx]:
            upper_arr[idx] = max(0.0, float(cutoff) - tp)
            ts = item_ts[it] if it < len(item_ts) else []
            n_tot = len(ts)
            ntot_arr[idx] = n_tot
            if n_tot >= 1:
                lower2[idx] = max(0.0, float(ts[0] if n_tot < 20 else ts[int(0.02 * n_tot)]) - tp)
                pos = int(delta_np.searchsorted(ts, tp))
                na = n_tot - pos
                na_arr[idx] = na
                if na >= 1:
                    rem = delta_np.asarray(ts[pos:], dtype=delta_np.float64) - tp
                    rem_p15[idx] = delta_np.percentile(rem, 15)
                    rem_p25[idx] = delta_np.percentile(rem, 25)
                    rem_p30[idx] = delta_np.percentile(rem, 30)
    return (valid_tp, u_med, u_q15, u_q25, u_rec2, u_rec3, u_rec5, u_last1, upper_arr, lower2, na_arr, ntot_arr, rem_p15, rem_p25, rem_p30)

def delta_posterior_median_grid(edges: delta_np.ndarray, positive_mass: delta_np.ndarray, log_item_lift: delta_np.ndarray, log_post_survival: delta_np.ndarray, zero_prob: float, zero_log_lift: float, zero_log_survival: float, beta: float=1.0, gamma: float=1.0) -> float:
    """Discrete-grid conditional posterior median over [0, H].

    positive_mass[k] = P(edges[k] < D <= edges[k+1]), excluding D=0.
    log_item_lift and log_post_survival are evaluated at interval midpoints.
    """
    if edges[-1] <= 0:
        return 0.0
    nodes = delta_np.r_[0.0, (edges[:-1] + edges[1:]) / 2.0]
    mass = delta_np.r_[zero_prob, positive_mass]
    logm = delta_np.full(len(mass), -delta_np.inf, dtype=delta_np.float64)
    present = mass > 0
    logm[present] = delta_np.log(mass[present])
    if beta != 0.0:
        logm += beta * delta_np.r_[zero_log_lift, log_item_lift]
    if gamma != 0.0:
        logm += gamma * delta_np.r_[zero_log_survival, log_post_survival]
    good = delta_np.isfinite(logm)
    if not good.any():
        raise ValueError('No posterior support: use a documented fallback')
    weights = delta_np.zeros_like(logm)
    weights[good] = delta_np.exp(logm[good] - delta_np.max(logm[good]))
    median_idx = int(delta_np.searchsorted(delta_np.cumsum(weights), 0.5 * weights.sum()))
    median_idx = min(median_idx, len(nodes) - 1)
    return float(nodes[median_idx])

@delta_njit(cache=True)
def delta__norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + delta_math.erf(x * 0.7071067811865475))

@delta_njit(cache=True)
def delta__quantile_sorted(sorted_arr: delta_np.ndarray, q: float) -> float:
    n = len(sorted_arr)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_arr[0])
    pos = q * (n - 1)
    lo = int(delta_math.floor(pos))
    hi = int(delta_math.ceil(pos))
    if lo == hi:
        return float(sorted_arr[lo])
    frac = pos - lo
    return float(sorted_arr[lo] * (1.0 - frac) + sorted_arr[hi] * frac)

@delta_njit(cache=True)
def delta__eval_posterior_quantiles_numba(edges: delta_np.ndarray, pos_mass: delta_np.ndarray, log_lift_unit: delta_np.ndarray, log_surv: delta_np.ndarray, zero_prob: float, zero_log_lift_unit: float, zero_log_surv: float, beta_i: float, gamma: float) -> tuple[float, float, float]:
    k_len = len(pos_mass)
    if edges[k_len] <= 0.0:
        return (0.0, 0.0, 0.0)
    logm = delta_np.full(k_len + 1, -1e+300, dtype=delta_np.float64)
    if zero_prob > 0.0:
        logm[0] = delta_math.log(zero_prob) + beta_i * zero_log_lift_unit + gamma * zero_log_surv
    for k in range(k_len):
        if pos_mass[k] > 0.0:
            logm[k + 1] = delta_math.log(pos_mass[k]) + beta_i * log_lift_unit[k] + gamma * log_surv[k]
    max_log = -1e+300
    for k in range(k_len + 1):
        if logm[k] > max_log:
            max_log = logm[k]
    if max_log <= -1e+290:
        return (0.0, 0.0, 0.0)
    w_sum = 0.0
    weights = delta_np.zeros(k_len + 1, dtype=delta_np.float64)
    for k in range(k_len + 1):
        if logm[k] > -1e+290:
            w = delta_math.exp(logm[k] - max_log)
            weights[k] = w
            w_sum += w
    if w_sum <= 0.0:
        return (0.0, 0.0, 0.0)
    q25_t = 0.25 * w_sum
    q50_t = 0.5 * w_sum
    q75_t = 0.75 * w_sum
    cs = 0.0
    q25, q50, q75 = (-1.0, -1.0, -1.0)
    for k in range(k_len + 1):
        cs += weights[k]
        node_val = 0.0 if k == 0 else 0.5 * (edges[k - 1] + edges[k])
        if q25 < 0.0 and cs >= q25_t:
            q25 = node_val
        if q50 < 0.0 and cs >= q50_t:
            q50 = node_val
        if q75 < 0.0 and cs >= q75_t:
            q75 = node_val
            break
    if q25 < 0.0:
        q25 = edges[k_len]
    if q50 < 0.0:
        q50 = edges[k_len]
    if q75 < 0.0:
        q75 = edges[k_len]
    return (q25, q50, q75)

@delta_njit(cache=True)
def delta__activity_bucket(hlen: int) -> int:
    if hlen <= 4:
        return 0
    if hlen <= 8:
        return 1
    if hlen <= 15:
        return 2
    if hlen <= 30:
        return 3
    if hlen <= 60:
        return 4
    return 5

def delta_fit_activity_priors(histories: list[list[tuple[int, int]]], train_users: delta_np.ndarray, cutoff: int) -> dict[str, delta_np.ndarray]:
    """Estimate activity-group log-gap priors strictly from train_users at cutoff."""
    n_buckets = 6
    mu_lists = [[] for _ in range(n_buckets)]
    hi_lists = [[] for _ in range(n_buckets)]
    sig_lists = [[] for _ in range(n_buckets)]
    zero_num = delta_np.zeros(n_buckets, dtype=delta_np.float64)
    zero_den = delta_np.zeros(n_buckets, dtype=delta_np.float64)
    for u in train_users:
        h = histories[int(u)]
        if len(h) < 2:
            continue
        b = delta__activity_bucket(len(h))
        clean_t = [x[1] for x in h if delta_T_MIN_VALID <= x[1] <= cutoff]
        if len(clean_t) < 2:
            clean_t = [x[1] for x in h]
        gaps = delta_np.maximum(0.0, delta_np.diff(delta_np.asarray(clean_t, dtype=delta_np.float64)))
        if len(gaps) == 0:
            continue
        zero_num[b] += float(delta_np.sum(gaps <= 60.0))
        zero_den[b] += float(len(gaps))
        pos_g = gaps[gaps > 60.0]
        if len(pos_g) == 0:
            pos_g = delta_np.maximum(61.0, gaps)
        z = delta_np.log1p(pos_g)
        mu_lists[b].append(0.5 * float(delta_np.median(z)) + 0.5 * float(delta_np.percentile(z, 75)))
        hi_lists[b].append(float(delta_np.percentile(z, 90)))
        if len(z) >= 2:
            sig_lists[b].append(float(delta_np.std(z)))
    act_mu = delta_np.full(n_buckets, 15.1, dtype=delta_np.float64)
    act_hi = delta_np.full(n_buckets, 16.2, dtype=delta_np.float64)
    act_sig = delta_np.full(n_buckets, 0.9, dtype=delta_np.float64)
    act_pi0 = delta_np.full(n_buckets, 0.0018, dtype=delta_np.float64)
    for b in range(n_buckets):
        if mu_lists[b]:
            act_mu[b] = float(delta_np.median(mu_lists[b]))
        if hi_lists[b]:
            act_hi[b] = float(delta_np.median(hi_lists[b]))
        if sig_lists[b]:
            act_sig[b] = float(delta_np.clip(delta_np.median(sig_lists[b]), 0.65, 1.15))
        if zero_den[b] > 0:
            act_pi0[b] = float((zero_num[b] + 1.0) / (zero_den[b] + 500.0))
    return {'act_mu': act_mu, 'act_hi': act_hi, 'act_sig': act_sig, 'act_pi0': act_pi0}

@delta_njit(cache=True)
def delta__compute_grid_and_rich_features_numba(query_u: delta_np.ndarray, query_i: delta_np.ndarray, u_raw_ptr: delta_np.ndarray, u_raw_ts: delta_np.ndarray, u_clean_ptr: delta_np.ndarray, u_clean_ts: delta_np.ndarray, i_ptr: delta_np.ndarray, i_ts: delta_np.ndarray, global_ts: delta_np.ndarray, cutoff: int, act_mu: delta_np.ndarray, act_hi: delta_np.ndarray, act_sig: delta_np.ndarray, act_pi0: delta_np.ndarray, k_grid: int, kappa_main: float, beta_tuned: float, gamma_tuned: float) -> tuple[delta_np.ndarray, delta_np.ndarray, delta_np.ndarray, delta_np.ndarray, delta_np.ndarray]:
    """Compute rich soft features + posterior grid summaries in one Numba pass.

    Returns:
      rich_feats: (n_queries, 101)
      grid_f_rat: (n_queries, k_grid)
      grid_pos_mass: (n_queries, k_grid)
      grid_log_surv: (n_queries, k_grid)
      grid_meta: (n_queries, 5) -> [H, ci, zero_prob, zero_f_rat, zero_log_surv]
    """
    n_q = len(query_u)
    n_items = len(i_ptr) - 1
    n_glob = float(max(1, len(global_ts)))
    quantiles = delta_np.array([0.0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0], dtype=delta_np.float64)
    rec_ks = delta_np.array([1, 2, 3, 5, 10, 20], dtype=delta_np.int64)
    win_days = delta_np.array([7.0, 30.0, 90.0, 365.0, 730.0], dtype=delta_np.float64)
    rich_feats = delta_np.zeros((n_q, 101), dtype=delta_np.float64)
    grid_f_rat = delta_np.zeros((n_q, k_grid), dtype=delta_np.float64)
    grid_pos_mass = delta_np.zeros((n_q, k_grid), dtype=delta_np.float64)
    grid_log_surv = delta_np.zeros((n_q, k_grid), dtype=delta_np.float64)
    grid_meta = delta_np.zeros((n_q, 5), dtype=delta_np.float64)
    u_grid = delta_np.linspace(0.0, 1.0, k_grid + 1)
    bw30 = int(30.0 * delta_DAY)
    bw120 = int(120.0 * delta_DAY)
    for idx in range(n_q):
        u = query_u[idx]
        it = query_i[idx]
        r_start, r_end = (u_raw_ptr[u], u_raw_ptr[u + 1])
        c_start, c_end = (u_clean_ptr[u], u_clean_ptr[u + 1])
        raw_len = r_end - r_start
        clean_len = c_end - c_start
        tp = float(u_raw_ts[r_end - 1])
        t_first = float(u_clean_ts[c_start]) if clean_len >= 1 else float(u_raw_ts[r_start])
        valid_tp = tp >= delta_T_MIN_VALID and tp <= cutoff
        H = max(0.0, float(cutoff) - tp) if valid_tp else 0.0
        H_days = H / delta_DAY
        if clean_len >= 2:
            n_gaps = clean_len - 1
            gaps_sec = delta_np.empty(n_gaps, dtype=delta_np.float64)
            for g_i in range(n_gaps):
                d_val = float(u_clean_ts[c_start + g_i + 1] - u_clean_ts[c_start + g_i])
                gaps_sec[g_i] = d_val if d_val > 0.0 else 0.0
        elif raw_len >= 2:
            n_gaps = raw_len - 1
            gaps_sec = delta_np.empty(n_gaps, dtype=delta_np.float64)
            for g_i in range(n_gaps):
                d_val = float(u_raw_ts[r_start + g_i + 1] - u_raw_ts[r_start + g_i])
                gaps_sec[g_i] = d_val if d_val > 0.0 else 0.0
        else:
            n_gaps = 1
            gaps_sec = delta_np.zeros(1, dtype=delta_np.float64)
        gaps_days = gaps_sec / delta_DAY
        sorted_gaps_days = delta_np.sort(gaps_days)
        g_mean = float(delta_np.mean(gaps_days))
        g_std = float(delta_np.std(gaps_days))
        g_log1p_mean = 0.0
        lt1, lt7, lt30, n_zero60 = (0.0, 0.0, 0.0, 0.0)
        pos_count = 0
        for g_i in range(n_gaps):
            gd = gaps_days[g_i]
            gs = gaps_sec[g_i]
            g_log1p_mean += delta_math.log1p(gd)
            if gd < 1.0:
                lt1 += 1.0
            if gd < 7.0:
                lt7 += 1.0
            if gd < 30.0:
                lt30 += 1.0
            if gs <= 60.0:
                n_zero60 += 1.0
            else:
                pos_count += 1
        g_log1p_mean /= float(n_gaps)
        lt1 /= float(n_gaps)
        lt7 /= float(n_gaps)
        lt30 /= float(n_gaps)
        b_act = delta__activity_bucket(raw_len)
        pi_u = (n_zero60 + 5.0 * act_pi0[b_act]) / (float(n_gaps) + 5.0)
        col = 0
        rich_feats[idx, col] = float(raw_len)
        col += 1
        rich_feats[idx, col] = float(clean_len)
        col += 1
        rich_feats[idx, col] = H_days
        col += 1
        rich_feats[idx, col] = max(0.0, (tp - t_first) / delta_DAY)
        col += 1
        rich_feats[idx, col] = max(0.0, (float(cutoff) - t_first) / delta_DAY)
        col += 1
        rich_feats[idx, col] = 1.0 if valid_tp else 0.0
        col += 1
        rich_feats[idx, col] = g_mean
        col += 1
        rich_feats[idx, col] = g_std
        col += 1
        rich_feats[idx, col] = g_log1p_mean
        col += 1
        rich_feats[idx, col] = lt1
        col += 1
        rich_feats[idx, col] = lt7
        col += 1
        rich_feats[idx, col] = lt30
        col += 1
        rich_feats[idx, col] = pi_u
        col += 1
        for q_i in range(9):
            rich_feats[idx, col] = delta__quantile_sorted(sorted_gaps_days, quantiles[q_i])
            col += 1
        u_med_days = rich_feats[idx, col - 5]
        u_q75_days = rich_feats[idx, col - 4]
        for rk_i in range(6):
            rk = rec_ks[rk_i]
            start_g = max(0, n_gaps - rk)
            sub_g = gaps_days[start_g:]
            sub_sorted = delta_np.sort(sub_g)
            m_val = float(delta_np.mean(sub_g))
            med_val = delta__quantile_sorted(sub_sorted, 0.5)
            std_val = float(delta_np.std(sub_g))
            l_val = 0.0
            for s_i in range(len(sub_g)):
                l_val += delta_math.log1p(sub_g[s_i])
            l_val /= float(len(sub_g))
            rich_feats[idx, col] = m_val
            col += 1
            rich_feats[idx, col] = med_val
            col += 1
            rich_feats[idx, col] = std_val
            col += 1
            rich_feats[idx, col] = l_val
            col += 1
        u_last1_days = rich_feats[idx, col - 23]
        u_rec3_days = rich_feats[idx, col - 15]
        u_rec5_days = rich_feats[idx, col - 11]
        for w_i in range(5):
            thresh = tp - win_days[w_i] * delta_DAY
            cnt_w = 0
            for r_p in range(r_start, r_end):
                if u_raw_ts[r_p] >= thresh:
                    cnt_w += 1
            rich_feats[idx, col] = float(cnt_w)
            col += 1
        denom_h = H_days + 1.0
        rich_feats[idx, col] = u_med_days / denom_h
        col += 1
        rich_feats[idx, col] = u_q75_days / denom_h
        col += 1
        rich_feats[idx, col] = u_rec3_days / denom_h
        col += 1
        rich_feats[idx, col] = u_rec5_days / denom_h
        col += 1
        rich_feats[idx, col] = u_last1_days / denom_h
        col += 1
        if it < n_items:
            istart, iend = (i_ptr[it], i_ptr[it + 1])
            ci = iend - istart
        else:
            istart, iend = (0, 0)
            ci = 0
        rich_feats[idx, col] = float(ci)
        col += 1
        rich_feats[idx, col] = delta_math.log1p(float(ci))
        col += 1
        if ci > 0 and valid_tp:
            its_sub = i_ts[istart:iend]
            for q_i in range(9):
                q_ts = delta__quantile_sorted(its_sub, quantiles[q_i])
                rich_feats[idx, col] = (q_ts - tp) / delta_DAY
                col += 1
            pos_idx = int(delta_np.searchsorted(its_sub, int(tp)))
            na = ci - pos_idx
            rich_feats[idx, col] = float(na) / float(ci)
            col += 1
            rich_feats[idx, col] = (float(na) + 1.0) / (float(ci) + 4.0)
            col += 1
            if na > 0:
                fut_sub = its_sub[pos_idx:]
                for q_i in range(9):
                    q_fut = delta__quantile_sorted(fut_sub, quantiles[q_i])
                    rich_feats[idx, col] = (q_fut - tp) / delta_DAY
                    col += 1
            else:
                for _ in range(9):
                    rich_feats[idx, col] = delta_np.nan
                    col += 1
            for w_i in range(5):
                w_sec = int(win_days[w_i] * delta_DAY)
                c_fwd = int(delta_np.searchsorted(its_sub, int(tp) + w_sec, side='right')) - pos_idx
                c_bwd = pos_idx - int(delta_np.searchsorted(its_sub, int(tp) - w_sec, side='left'))
                rich_feats[idx, col] = float(c_fwd) / float(ci)
                col += 1
                rich_feats[idx, col] = float(c_bwd) / float(ci)
                col += 1
        else:
            for _ in range(30):
                rich_feats[idx, col] = delta_np.nan
                col += 1
        if pos_count > 0:
            z_pos = delta_np.empty(pos_count, dtype=delta_np.float64)
            p_k = 0
            for g_i in range(n_gaps):
                if gaps_sec[g_i] > 60.0:
                    z_pos[p_k] = delta_math.log1p(gaps_sec[g_i])
                    p_k += 1
        else:
            z_pos = delta_np.empty(n_gaps, dtype=delta_np.float64)
            for g_i in range(n_gaps):
                z_pos[g_i] = delta_math.log1p(max(61.0, gaps_sec[g_i]))
        nz = len(z_pos)
        z_sorted = delta_np.sort(z_pos)
        z_med = delta__quantile_sorted(z_sorted, 0.5)
        z_q75 = delta__quantile_sorted(z_sorted, 0.75)
        z_q90 = delta__quantile_sorted(z_sorted, 0.9)
        rec_n = min(5, nz)
        z_rec_sub = delta_np.sort(z_pos[nz - rec_n:])
        z_rec = delta__quantile_sorted(z_rec_sub, 0.5)
        mu_all = (float(nz) * (0.5 * z_med + 0.5 * z_q75) + 3.0 * act_mu[b_act]) / (float(nz) + 3.0)
        mu_hi = (float(nz) * z_q90 + 3.0 * act_hi[b_act]) / (float(nz) + 3.0)
        mu_rec = (float(rec_n) * z_rec + 2.0 * mu_all) / (float(rec_n) + 2.0)
        z_std = float(delta_np.std(z_pos)) if nz >= 2 else act_sig[b_act]
        sig = min(1.05, max(0.5, (0.5 * z_std * float(nz) + act_sig[b_act] * 6.0) / (float(nz) + 6.0)))
        if valid_tp and H > 0.0:
            log1p_H = delta_math.log1p(H)
            edges = delta_np.empty(k_grid + 1, dtype=delta_np.float64)
            z_edges = delta_np.empty(k_grid + 1, dtype=delta_np.float64)
            cdf_edges = delta_np.empty(k_grid + 1, dtype=delta_np.float64)
            for k in range(k_grid + 1):
                ze = u_grid[k] * log1p_H
                z_edges[k] = ze
                edges[k] = delta_math.expm1(ze)
                cdf_edges[k] = 0.45 * delta__norm_cdf((ze - mu_all) / sig) + 0.3 * delta__norm_cdf((ze - mu_rec) / sig) + 0.25 * delta__norm_cdf((ze - mu_hi) / sig)
            pos_mass = delta_np.empty(k_grid, dtype=delta_np.float64)
            log_surv = delta_np.empty(k_grid, dtype=delta_np.float64)
            f_rat_arr = delta_np.ones(k_grid, dtype=delta_np.float64)
            log_lift_k1 = delta_np.zeros(k_grid, dtype=delta_np.float64)
            log_lift_main = delta_np.zeros(k_grid, dtype=delta_np.float64)
            cdf_H = 0.45 * delta__norm_cdf((log1p_H - mu_all) / sig) + 0.3 * delta__norm_cdf((log1p_H - mu_rec) / sig) + 0.25 * delta__norm_cdf((log1p_H - mu_hi) / sig)
            zero_surv = (1.0 - pi_u) * max(1e-12, 1.0 - cdf_H)
            zero_log_surv = delta_math.log(zero_surv)
            zero_f_rat = 1.0
            if ci > 0:
                its_sub = i_ts[istart:iend]
                t0_int = int(tp)
                lo1_0 = max(delta_T_MIN_VALID, t0_int - bw30)
                hi1_0 = min(cutoff, t0_int + bw30)
                lo2_0 = max(delta_T_MIN_VALID, t0_int - bw120)
                hi2_0 = min(cutoff, t0_int + bw120)
                ic1_0 = float(delta_np.searchsorted(its_sub, hi1_0, side='right') - delta_np.searchsorted(its_sub, lo1_0, side='left'))
                gc1_0 = float(max(1, delta_np.searchsorted(global_ts, hi1_0, side='right') - delta_np.searchsorted(global_ts, lo1_0, side='left')))
                ic2_0 = float(delta_np.searchsorted(its_sub, hi2_0, side='right') - delta_np.searchsorted(its_sub, lo2_0, side='left'))
                gc2_0 = float(max(1, delta_np.searchsorted(global_ts, hi2_0, side='right') - delta_np.searchsorted(global_ts, lo2_0, side='left')))
                zero_f_rat = 0.55 * (ic1_0 / float(ci) / (gc1_0 / n_glob)) + 0.45 * (ic2_0 / float(ci) / (gc2_0 / n_glob))
            zero_lift_k1 = (float(ci) * zero_f_rat + 1.0) / (float(ci) + 1.0) if ci > 0 else 1.0
            zero_lift_main = (float(ci) * zero_f_rat + kappa_main) / (float(ci) + kappa_main) if ci > 0 else 1.0
            zero_log_lift_k1 = delta_math.log(max(1e-08, zero_lift_k1))
            zero_log_lift_main = delta_math.log(max(1e-08, zero_lift_main))
            for k in range(k_grid):
                dm = cdf_edges[k + 1] - cdf_edges[k]
                pm_k = (1.0 - pi_u) * (dm if dm > 1e-15 else 1e-15)
                pos_mass[k] = pm_k
                mid_d = 0.5 * (edges[k] + edges[k + 1])
                rem_d = max(0.0, H - mid_d)
                z_rem = delta_math.log1p(rem_d)
                cdf_rem = 0.45 * delta__norm_cdf((z_rem - mu_all) / sig) + 0.3 * delta__norm_cdf((z_rem - mu_rec) / sig) + 0.25 * delta__norm_cdf((z_rem - mu_hi) / sig)
                s_k = (1.0 - pi_u) * max(1e-12, 1.0 - cdf_rem)
                log_surv[k] = delta_math.log(s_k)
                if ci > 0:
                    its_sub = i_ts[istart:iend]
                    tn = int(tp + mid_d)
                    lo1 = max(delta_T_MIN_VALID, tn - bw30)
                    hi1 = min(cutoff, tn + bw30)
                    lo2 = max(delta_T_MIN_VALID, tn - bw120)
                    hi2 = min(cutoff, tn + bw120)
                    ic1 = float(delta_np.searchsorted(its_sub, hi1, side='right') - delta_np.searchsorted(its_sub, lo1, side='left'))
                    gc1 = float(max(1, delta_np.searchsorted(global_ts, hi1, side='right') - delta_np.searchsorted(global_ts, lo1, side='left')))
                    ic2 = float(delta_np.searchsorted(its_sub, hi2, side='right') - delta_np.searchsorted(its_sub, lo2, side='left'))
                    gc2 = float(max(1, delta_np.searchsorted(global_ts, hi2, side='right') - delta_np.searchsorted(global_ts, lo2, side='left')))
                    fr = 0.55 * (ic1 / float(ci) / (gc1 / n_glob)) + 0.45 * (ic2 / float(ci) / (gc2 / n_glob))
                    f_rat_arr[k] = fr
                    l_k1 = (float(ci) * fr + 1.0) / (float(ci) + 1.0)
                    l_main = (float(ci) * fr + kappa_main) / (float(ci) + kappa_main)
                    log_lift_k1[k] = delta_math.log(max(1e-08, l_k1))
                    log_lift_main[k] = delta_math.log(max(1e-08, l_main))
                grid_f_rat[idx, k] = f_rat_arr[k]
                grid_pos_mass[idx, k] = pos_mass[k]
                grid_log_surv[idx, k] = log_surv[k]
            grid_meta[idx, 0] = H
            grid_meta[idx, 1] = float(ci)
            grid_meta[idx, 2] = pi_u
            grid_meta[idx, 3] = zero_f_rat
            grid_meta[idx, 4] = zero_log_surv
            att_k1 = float(ci) / (float(ci) + 1.0)
            att_main = float(ci) / (float(ci) + kappa_main)
            _, med_b0_g1, _ = delta__eval_posterior_quantiles_numba(edges, pos_mass, log_lift_k1, log_surv, pi_u, zero_log_lift_k1, zero_log_surv, 0.0, 1.0)
            _, med_b05_g05, _ = delta__eval_posterior_quantiles_numba(edges, pos_mass, log_lift_k1, log_surv, pi_u, zero_log_lift_k1, zero_log_surv, 0.5 * att_k1, 0.5)
            q25_b1_g1, med_b1_g1, q75_b1_g1 = delta__eval_posterior_quantiles_numba(edges, pos_mass, log_lift_k1, log_surv, pi_u, zero_log_lift_k1, zero_log_surv, 1.0 * att_k1, 1.0)
            _, med_b1_g0, _ = delta__eval_posterior_quantiles_numba(edges, pos_mass, log_lift_k1, log_surv, pi_u, zero_log_lift_k1, zero_log_surv, 1.0 * att_k1, 0.0)
            q25_tun, med_tun, q75_tun = delta__eval_posterior_quantiles_numba(edges, pos_mass, log_lift_main, log_surv, pi_u, zero_log_lift_main, zero_log_surv, beta_tuned * att_main, gamma_tuned)
            rich_feats[idx, col] = med_b0_g1 / delta_DAY
            col += 1
            rich_feats[idx, col] = med_b05_g05 / delta_DAY
            col += 1
            rich_feats[idx, col] = med_b1_g1 / delta_DAY
            col += 1
            rich_feats[idx, col] = med_b1_g0 / delta_DAY
            col += 1
            rich_feats[idx, col] = q25_b1_g1 / delta_DAY
            col += 1
            rich_feats[idx, col] = q75_b1_g1 / delta_DAY
            col += 1
            rich_feats[idx, col] = (q75_b1_g1 - q25_b1_g1) / delta_DAY
            col += 1
            rich_feats[idx, col] = med_tun / delta_DAY
            col += 1
            rich_feats[idx, col] = q25_tun / delta_DAY
            col += 1
            rich_feats[idx, col] = q75_tun / delta_DAY
            col += 1
            rich_feats[idx, col] = (med_b1_g1 - med_b0_g1) / delta_DAY
            col += 1
            rich_feats[idx, col] = (med_b1_g1 - med_b1_g0) / delta_DAY
            col += 1
            rich_feats[idx, col] = med_tun / delta_DAY / denom_h
            col += 1
        else:
            fallback_s = u_med_days * delta_DAY
            grid_meta[idx, 0] = 0.0
            grid_meta[idx, 1] = float(ci)
            grid_meta[idx, 2] = pi_u
            grid_meta[idx, 3] = 1.0
            grid_meta[idx, 4] = 0.0
            for _ in range(6):
                rich_feats[idx, col] = u_med_days
                col += 1
            rich_feats[idx, col] = 0.0
            col += 1
            for _ in range(3):
                rich_feats[idx, col] = u_med_days
                col += 1
            rich_feats[idx, col] = 0.0
            col += 1
            rich_feats[idx, col] = 0.0
            col += 1
            rich_feats[idx, col] = 0.5
            col += 1
    return (rich_feats, grid_f_rat, grid_pos_mass, grid_log_surv, grid_meta)

@delta_njit(cache=True)
def delta_eval_cached_posterior_grid(grid_f_rat: delta_np.ndarray, grid_pos_mass: delta_np.ndarray, grid_log_surv: delta_np.ndarray, grid_meta: delta_np.ndarray, fallback_med_sec: delta_np.ndarray, beta: float, gamma: float, kappa: float) -> delta_np.ndarray:
    """Fast evaluation of posterior_median_grid from cached per-query grid arrays."""
    n_q, k_grid = grid_pos_mass.shape
    u_grid = delta_np.linspace(0.0, 1.0, k_grid + 1)
    preds = delta_np.empty(n_q, dtype=delta_np.float64)
    edges = delta_np.empty(k_grid + 1, dtype=delta_np.float64)
    log_lift = delta_np.empty(k_grid, dtype=delta_np.float64)
    for idx in range(n_q):
        H = grid_meta[idx, 0]
        if H <= 0.0:
            preds[idx] = fallback_med_sec[idx]
            continue
        ci = grid_meta[idx, 1]
        pi_u = grid_meta[idx, 2]
        zero_f_rat = grid_meta[idx, 3]
        zero_log_surv = grid_meta[idx, 4]
        log1p_H = delta_math.log1p(H)
        for k in range(k_grid + 1):
            edges[k] = delta_math.expm1(u_grid[k] * log1p_H)
        if ci > 0.0 and beta != 0.0:
            beta_i = beta * (ci / (ci + kappa))
            zero_lift = (ci * zero_f_rat + kappa) / (ci + kappa)
            zero_log_lift = delta_math.log(max(1e-08, zero_lift))
            for k in range(k_grid):
                lk = (ci * grid_f_rat[idx, k] + kappa) / (ci + kappa)
                log_lift[k] = delta_math.log(max(1e-08, lk))
        else:
            beta_i = 0.0
            zero_log_lift = 0.0
            for k in range(k_grid):
                log_lift[k] = 0.0
        _, q50, _ = delta__eval_posterior_quantiles_numba(edges, grid_pos_mass[idx], log_lift, grid_log_surv[idx], pi_u, zero_log_lift, zero_log_surv, beta_i, gamma)
        preds[idx] = q50
    return preds

def delta_run_numerical_verification_checks(histories: list[list[tuple[int, int]]], item_ts_list: list[list[int]], global_ts: delta_np.ndarray, query_u: delta_np.ndarray, query_i: delta_np.ndarray, cutoff: int, priors: dict[str, delta_np.ndarray]) -> dict:
    """Execute mandatory verification checks from Section 6.2 - 6.4."""
    H_test = 120.0 * delta_DAY
    k_grid = 96
    edges_1s = delta_np.expm1(delta_np.linspace(0.0, delta_np.log1p(H_test), k_grid + 1))
    mid_1s = 0.5 * (edges_1s[:-1] + edges_1s[1:])
    idx_h2_1s = int(delta_np.searchsorted(edges_1s, 0.5 * H_test)) - 1
    grid_res_1s = float(edges_1s[idx_h2_1s + 1] - edges_1s[idx_h2_1s])
    edges_1h = 3600.0 * delta_np.expm1(delta_np.linspace(0.0, delta_np.log1p(H_test / 3600.0), k_grid + 1))
    mid_1h = 0.5 * (edges_1h[:-1] + edges_1h[1:])
    idx_h2_1h = int(delta_np.searchsorted(edges_1h, 0.5 * H_test)) - 1
    grid_res_1h = float(edges_1h[idx_h2_1h + 1] - edges_1h[idx_h2_1h])
    exp_records = []
    for mean_days in [10.0, 30.0, 60.0, 120.0, 300.0, 1000.0]:
        lam = 1.0 / (mean_days * delta_DAY)
        pm_1s = delta_np.exp(-lam * edges_1s[:-1]) - delta_np.exp(-lam * edges_1s[1:])
        ls_1s = -lam * (H_test - mid_1s)
        med_1s = delta_posterior_median_grid(edges_1s, pm_1s, delta_np.zeros(k_grid), ls_1s, 0.0, 0.0, -lam * H_test, beta=0.0, gamma=1.0)
        pm_1h = delta_np.exp(-lam * edges_1h[:-1]) - delta_np.exp(-lam * edges_1h[1:])
        ls_1h = -lam * (H_test - mid_1h)
        med_1h = delta_posterior_median_grid(edges_1h, pm_1h, delta_np.zeros(k_grid), ls_1h, 0.0, 0.0, -lam * H_test, beta=0.0, gamma=1.0)
        err_1s = abs(med_1s - 0.5 * H_test)
        err_1h = abs(med_1h - 0.5 * H_test)
        assert err_1s <= grid_res_1s * 1.05, f'Exponential check failed on 1s grid for mean_days={mean_days}'
        assert err_1h <= grid_res_1h * 1.05, f'Exponential check failed on 1h grid for mean_days={mean_days}'
        exp_records.append({'mean_wait_days': mean_days, 'target_half_H_days': 0.5 * H_test / delta_DAY, 'median_log1p_sec_days': med_1s / delta_DAY, 'abs_err_log1p_sec_days': err_1s / delta_DAY, 'within_grid_res_log1p_sec': bool(err_1s <= grid_res_1s), 'median_log1p_hour_days': med_1h / delta_DAY, 'abs_err_log1p_hour_days': err_1h / delta_DAY, 'within_grid_res_log1p_hour': bool(err_1h <= grid_res_1h)})
    pm_unif = delta_np.full(k_grid, 0.4 / k_grid, dtype=delta_np.float64)
    zero_med = delta_posterior_median_grid(edges_1s, pm_unif, delta_np.ones(k_grid), delta_np.ones(k_grid), 0.6, 1.0, 1.0, beta=0.0, gamma=0.0)
    assert zero_med == 0.0, f'Zero-atom check failed: got {zero_med}'
    u_raw_ptr, u_raw_ts, u_clean_ptr, u_clean_ts = delta__pack_user_histories(histories, cutoff)
    i_ptr, i_ts = delta__pack_item_timestamps(item_ts_list)
    sub_u = query_u[:4000]
    sub_i = query_i[:4000]
    rf_64, _, _, _, _ = delta__compute_grid_and_rich_features_numba(sub_u, sub_i, u_raw_ptr, u_raw_ts, u_clean_ptr, u_clean_ts, i_ptr, i_ts, global_ts, cutoff, priors['act_mu'], priors['act_hi'], priors['act_sig'], priors['act_pi0'], 64, 0.8, 1.2, 1.0)
    rf_96, _, _, _, _ = delta__compute_grid_and_rich_features_numba(sub_u, sub_i, u_raw_ptr, u_raw_ts, u_clean_ptr, u_clean_ts, i_ptr, i_ts, global_ts, cutoff, priors['act_mu'], priors['act_hi'], priors['act_sig'], priors['act_pi0'], 96, 0.8, 1.2, 1.0)
    rf_128, _, _, _, _ = delta__compute_grid_and_rich_features_numba(sub_u, sub_i, u_raw_ptr, u_raw_ts, u_clean_ptr, u_clean_ts, i_ptr, i_ts, global_ts, cutoff, priors['act_mu'], priors['act_hi'], priors['act_sig'], priors['act_pi0'], 128, 0.8, 1.2, 1.0)
    diff_64_128_days = delta_np.abs(rf_64[:, 95] - rf_128[:, 95])
    diff_96_128_days = delta_np.abs(rf_96[:, 95] - rf_128[:, 95])
    H_sub_days = delta_np.maximum(1.0, rf_96[:, 2])
    return {'exponential_special_case': {'passed': True, 'H_days': H_test / delta_DAY, 'grid_res_at_H_half_log1p_sec_days': grid_res_1s / delta_DAY, 'grid_res_at_H_half_log1p_hour_days': grid_res_1h / delta_DAY, 'results_by_lambda': exp_records}, 'zero_atom_special_case': {'passed': bool(zero_med == 0.0), 'pi_u': 0.6, 'beta': 0.0, 'gamma': 0.0, 'returned_median': zero_med}, 'grid_convergence_K64_K96_K128': {'n_queries_checked': int(len(sub_u)), 'K64_vs_K128_median_abs_diff_days': float(delta_np.median(diff_64_128_days)), 'K64_vs_K128_median_rel_diff_pct_of_H': float(delta_np.median(diff_64_128_days / H_sub_days) * 100.0), 'K96_vs_K128_median_abs_diff_days': float(delta_np.median(diff_96_128_days)), 'K96_vs_K128_median_rel_diff_pct_of_H': float(delta_np.median(diff_96_128_days / H_sub_days) * 100.0)}}

def delta__pack_user_histories(histories: list[list[tuple[int, int]]], cutoff: int) -> tuple[delta_np.ndarray, delta_np.ndarray, delta_np.ndarray, delta_np.ndarray]:
    nu = len(histories)
    u_raw_ptr = delta_np.zeros(nu + 1, dtype=delta_np.int64)
    u_clean_ptr = delta_np.zeros(nu + 1, dtype=delta_np.int64)
    raw_list = []
    clean_list = []
    for u, h in enumerate(histories):
        for _, t in h:
            raw_list.append(t)
            if delta_T_MIN_VALID <= t <= cutoff:
                clean_list.append(t)
        u_raw_ptr[u + 1] = len(raw_list)
        u_clean_ptr[u + 1] = len(clean_list)
    return (u_raw_ptr, delta_np.asarray(raw_list, dtype=delta_np.int64), u_clean_ptr, delta_np.asarray(clean_list, dtype=delta_np.int64))

def delta__pack_item_timestamps(item_ts_list: list[list[int]]) -> tuple[delta_np.ndarray, delta_np.ndarray]:
    ni = len(item_ts_list)
    i_ptr = delta_np.zeros(ni + 1, dtype=delta_np.int64)
    flat = []
    for i, ts in enumerate(item_ts_list):
        if ts:
            flat.extend(ts)
        i_ptr[i + 1] = len(flat)
    return (i_ptr, delta_np.asarray(flat, dtype=delta_np.int64))

def delta_build_clean_structures(histories: list[list[tuple[int, int]]], ni: int, cutoff: int) -> tuple[list[list[int]], delta_np.ndarray]:
    """Build sorted item_ts and global_ts strictly from events <= cutoff."""
    item_ts = [[] for _ in range(ni)]
    global_ts = []
    for h in histories:
        for i, t in h:
            if delta_T_MIN_VALID <= t <= cutoff:
                item_ts[i].append(t)
                global_ts.append(t)
    for ts in item_ts:
        if ts:
            ts.sort()
    global_ts_arr = delta_np.sort(delta_np.asarray(global_ts, dtype=delta_np.int64))
    return (item_ts, global_ts_arr)

def delta_make_terminal_examples(allhist: list[list[tuple[int, int]]], ni: int, cutoff: int, user_group: delta_np.ndarray, user_mask: delta_np.ndarray | None=None) -> dict:
    """Construct leakage-free terminal dataset at cutoff C.

    Eligibility:
      - At least 1 observed training event with t > C (used ONLY for eligibility)
      - At least 3 observed events with t <= C
    Holdout:
      - Hold out each eligible user's final event <= C as (i_target, t_target)
      - Remove ALL eligible users' held-out target events before constructing
        item_ts and global_ts so no target timestamps ever leak into features.
    """
    nu = len(allhist)
    h_le = [[ev for ev in h if ev[1] <= cutoff] for h in allhist]
    eligible = delta_np.array([len(h_le[u]) >= 3 and allhist[u][-1][1] > cutoff for u in range(nu)], dtype=bool)
    if user_mask is not None:
        selected_mask = eligible & user_mask
    else:
        selected_mask = eligible
    qu = delta_np.where(selected_mask)[0].astype(delta_np.int64)
    qi = delta_np.array([h_le[u][-1][0] for u in qu], dtype=delta_np.int64)
    truth_sec = delta_np.array([max(0.0, float(h_le[u][-1][1] - h_le[u][-2][1])) for u in qu], dtype=delta_np.float64)
    groups_q = user_group[qu]
    obs_histories = [h_le[u][:-1] if eligible[u] else h_le[u] for u in range(nu)]
    item_ts, global_ts = delta_build_clean_structures(obs_histories, ni, cutoff)
    vf_tuple = delta_extract_cutoff_delta_features(obs_histories, item_ts, qu, qi, cutoff)
    return {'cutoff': cutoff, 'obs_histories': obs_histories, 'item_ts': item_ts, 'global_ts': global_ts, 'query_u': qu, 'query_i': qi, 'truth_sec': truth_sec, 'truth_days': truth_sec / delta_DAY, 'groups': groups_q, 'vf_tuple': vf_tuple, 'vf_array': delta_np.asarray(vf_tuple, dtype=delta_np.float64)}

def delta_slice_vf_tuple(vf_tuple: tuple, mask_or_idx: delta_np.ndarray) -> tuple:
    return tuple((arr[mask_or_idx] for arr in vf_tuple))

def delta_concat_vf_tuples(vf_list: list[tuple]) -> tuple:
    return tuple((delta_np.concatenate([vf[k] for vf in vf_list], axis=0) for k in range(len(vf_list[0]))))

def delta_build_full_feature_matrix(rich_feats: delta_np.ndarray, vf_tuple: tuple, pred_term_nm_sec: delta_np.ndarray, post_med_tuned_sec: delta_np.ndarray) -> delta_np.ndarray:
    """Combine rich soft features, scaled base features, terminal NM, and posterior median."""
    valid_tp, u_med, _, _, _, _, _, _, upper_arr, low, na, nt, rem_p15, rem_p25, rem_p30 = vf_tuple
    vf_scaled = delta_np.column_stack([valid_tp.astype(delta_np.float64), u_med / delta_DAY, vf_tuple[2] / delta_DAY, vf_tuple[3] / delta_DAY, vf_tuple[4] / delta_DAY, vf_tuple[5] / delta_DAY, vf_tuple[6] / delta_DAY, vf_tuple[7] / delta_DAY, upper_arr / delta_DAY, low / delta_DAY, na, nt, rem_p15 / delta_DAY, rem_p25 / delta_DAY, rem_p30 / delta_DAY])
    conf = na / (na + 3.0) * (na / delta_np.maximum(1.0, nt)) ** 0.5
    it_est_days = (0.35 * rem_p15 + 0.35 * rem_p25 + 0.3 * rem_p30) / delta_DAY
    term_nm_days = pred_term_nm_sec / delta_DAY
    post_tun_days = post_med_tuned_sec / delta_DAY
    H_days = delta_np.maximum(1.0, upper_arr / delta_DAY)
    extra_cols = delta_np.column_stack([term_nm_days, post_tun_days, 0.5 * (term_nm_days + post_tun_days), term_nm_days - post_tun_days, term_nm_days / H_days, post_tun_days / H_days, it_est_days, conf])
    return delta_np.column_stack([rich_feats, vf_scaled, extra_cols])

def delta_postprocess_terminal_predictions(raw_pred_sec: delta_np.ndarray, vf_tuple: tuple) -> delta_np.ndarray:
    """Clip to [0, H] only when t0 in [T_MIN_VALID, cutoff] is valid; fallback to user median otherwise."""
    valid_tp = vf_tuple[0]
    u_med_sec = vf_tuple[1]
    H_sec = vf_tuple[8]
    clipped = delta_np.clip(raw_pred_sec, 0.0, H_sec)
    return delta_np.where(valid_tp, clipped, delta_np.maximum(0.0, u_med_sec))

def delta_fixed_features(ds, prior_users):
    """Identical rich-feature definitions at every training/test cutoff.

    The rich bank includes a fixed beta=1.2/kappa=.8 posterior. A separate
    selected posterior uses beta=1/kappa=.5. Keeping these definitions separate
    fixes the research prototype's train/test semantic mismatch.
    """
    priors = delta_fit_activity_priors(ds['obs_histories'], prior_users, ds['cutoff'])
    raw_ptr, raw_ts, clean_ptr, clean_ts = delta__pack_user_histories(ds['obs_histories'], ds['cutoff'])
    ip, it = delta__pack_item_timestamps(ds['item_ts'])
    rich, fr, pm, ls, meta = delta__compute_grid_and_rich_features_numba(ds['query_u'], ds['query_i'], raw_ptr, raw_ts, clean_ptr, clean_ts, ip, it, ds['global_ts'], ds['cutoff'], priors['act_mu'], priors['act_hi'], priors['act_sig'], priors['act_pi0'], 96, 0.8, 1.2, 1.0)
    post = delta_postprocess_terminal_predictions(delta_eval_cached_posterior_grid(fr, pm, ls, meta, ds['vf_tuple'][1], 1.0, 1.0, 0.5), ds['vf_tuple'])
    return (rich, post)

def delta_fit_terminal_params(datasets, params, group=None):
    vfs, ys = ([], [])
    for ds in datasets:
        mask = delta_np.ones(len(ds['query_u']), dtype=bool) if group is None else ds['groups'] == group
        vfs.append(delta_slice_vf_tuple(ds['vf_tuple'], mask))
        ys.append(ds['truth_sec'][mask])
    vf = delta_concat_vf_tuples(vfs)
    y = delta_np.concatenate(ys)
    fit = delta_minimize(lambda p: delta_np.mean(delta_np.abs(apply_delta_params(vf, p) - y)) / delta_DAY, params, method='Nelder-Mead', options={'maxiter': 1200})
    if not delta_np.all(delta_np.isfinite(fit.x)):
        raise ValueError('Task 6 fitted non-finite parameters')
    return fit.x

def delta_assemble_terminal(ds, params):
    term = delta_postprocess_terminal_predictions(apply_delta_params(ds['vf_tuple'], params), ds['vf_tuple'])
    x = delta_build_full_feature_matrix(ds['rich'], ds['vf_tuple'], term, ds['post'])
    anchor = 0.65 * term + 0.35 * ds['post']
    return (x, anchor, term)

def delta_predict_terminal(ds, params, model):
    x, anchor, term = delta_assemble_terminal(ds, params)
    residual = delta_postprocess_terminal_predictions(anchor + model.predict(x) * delta_DAY, ds['vf_tuple'])
    ensemble = delta_postprocess_terminal_predictions(0.8 * residual + 0.05 * term + 0.15 * ds['post'], ds['vf_tuple'])
    return (ensemble, term, residual)

def delta_production_delta_task(a, users, items, args):
    """Validate the locked terminal model, refit, and write Task 6 predictions."""
    started = delta_time.time()
    nu, ni = (len(users), len(items))
    histories = [[] for _ in range(nu)]
    for u, i, t, _ in a:
        histories[int(u)].append((int(i), int(t)))
    if any((len(h) < 3 for h in histories)):
        raise ValueError('Task 6 needs at least three historical events per user')
    if any((delta_np.any(delta_np.diff([t for _, t in h]) < 0) for h in histories)):
        raise ValueError('Task 6 histories are not chronological')
    perm = delta_np.random.default_rng(delta_SEED).permutation(nu)
    groups = delta_np.zeros(nu, dtype=delta_np.int64)
    groups[perm[int(0.6 * nu):int(0.8 * nu)]] = 1
    groups[perm[int(0.8 * nu):]] = 2
    train_users = delta_np.where(groups == 0)[0]
    all_users = delta_np.arange(nu, dtype=delta_np.int64)
    penult = [h[:-1] for h in histories]
    pit, _ = delta_build_clean_structures(penult, ni, delta_T_MAX_VALID)
    pi = delta_np.array([h[-1][0] for h in histories], dtype=delta_np.int64)
    py = delta_np.array([h[-1][1] - h[-2][1] for h in histories], dtype=delta_np.float64)
    pf = delta_extract_cutoff_delta_features(penult, pit, all_users, pi, delta_T_MAX_VALID)
    init = delta_np.array([0.13, 0.05, 0.08, 0.05, 0.12, 0.04, 0.05, 0.5, 0.65, 0.35, 0.26, 0.45, 0.35, 0.7])
    pf_train = delta_slice_vf_tuple(pf, train_users)
    legacy = delta_minimize(lambda p: delta_np.mean(delta_np.abs(apply_delta_params(pf_train, p) - py[train_users])) / delta_DAY, init, method='Nelder-Mead', options={'maxiter': 800}).x
    del penult, pit
    datasets = []
    for name, cutoff in delta_CUTOFF_SPECS:
        ds = delta_make_terminal_examples(histories, ni, cutoff, groups)
        ds['name'] = name
        ds['rich'], ds['post'] = delta_fixed_features(ds, train_users)
        datasets.append(ds)
        log(f"Task 6 terminal examples {name}: {len(ds['query_u']):,}")
    early = datasets[:3]
    init_terminal = delta_np.array([1.2, -0.15, 0.25, 0.35, 0.8, 0.05, 0.4, 0.86, -0.8, 0.42, 0.95, -0.5, 1.1, 0.9])
    early_params = delta_fit_terminal_params(early, init_terminal, group=0)
    xx, yy = ([], [])
    for ds in early:
        x, anchor, _ = delta_assemble_terminal(ds, early_params)
        mask = ds['groups'] == 0
        xx.append(x[mask])
        yy.append((ds['truth_sec'][mask] - anchor[mask]) / delta_DAY)
    model = delta_HistGradientBoostingRegressor(**delta_HGB_PARAMS).fit(delta_np.vstack(xx), delta_np.concatenate(yy))
    del xx, yy
    audit_metrics = {}
    for ds in datasets:
        pred, term, residual = delta_predict_terminal(ds, early_params, model)
        old = delta_postprocess_terminal_predictions(apply_delta_params(ds['vf_tuple'], legacy), ds['vf_tuple'])
        mask = ds['groups'] == 2
        y = ds['truth_sec'][mask]
        audit_metrics[ds['name']] = {'users': int(mask.sum()), 'legacy_mae_seconds': float(delta_np.mean(delta_np.abs(old[mask] - y))), 'terminal_nm_mae_seconds': float(delta_np.mean(delta_np.abs(term[mask] - y))), 'posterior_mae_seconds': float(delta_np.mean(delta_np.abs(ds['post'][mask] - y))), 'residual_hgb_mae_seconds': float(delta_np.mean(delta_np.abs(residual[mask] - y))), 'ensemble_mae_seconds': float(delta_np.mean(delta_np.abs(pred[mask] - y)))}
    check_ds = datasets[2]
    priors = delta_fit_activity_priors(check_ds['obs_histories'], train_users, check_ds['cutoff'])
    checks = delta_run_numerical_verification_checks(check_ds['obs_histories'], check_ds['item_ts'], check_ds['global_ts'], check_ds['query_u'], check_ds['query_i'], check_ds['cutoff'], priors)
    full_params = delta_fit_terminal_params(datasets, early_params)
    xx, yy = ([], [])
    for ds in datasets:
        ds['rich'], ds['post'] = delta_fixed_features(ds, all_users)
        x, anchor, _ = delta_assemble_terminal(ds, full_params)
        xx.append(x)
        yy.append((ds['truth_sec'] - anchor) / delta_DAY)
    model = delta_HistGradientBoostingRegressor(**delta_HGB_PARAMS).fit(delta_np.vstack(xx), delta_np.concatenate(yy))
    del xx, yy
    test = read_test(args.data_dir / 'task_6_test.csv')
    qu = delta_np.array([users[r['user']] for r in test], dtype=delta_np.int64)
    qi = delta_np.array([items.get(r['item'], ni) for r in test], dtype=delta_np.int64)
    item_ts, global_ts = delta_build_clean_structures(histories, ni, delta_T_MAX_VALID)
    test_ds = dict(cutoff=delta_T_MAX_VALID, obs_histories=histories, item_ts=item_ts, global_ts=global_ts, query_u=qu, query_i=qi, vf_tuple=delta_extract_cutoff_delta_features(histories, item_ts, qu, qi, delta_T_MAX_VALID))
    test_ds['rich'], test_ds['post'] = delta_fixed_features(test_ds, all_users)
    predictions, _, _ = delta_predict_terminal(test_ds, full_params, model)
    if not delta_np.all(delta_np.isfinite(predictions)) or delta_np.any(predictions < 0):
        raise ValueError('Task 6 generated invalid predictions')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.data_dir / 'task_6_test.csv', args.output_dir / 'task_6_predictions.csv', predictions)
    valid = test_ds['vf_tuple'][0]
    metrics = {'validation_protocol': 'fixed user groups; terminal events at historical cutoffs; model fit before July 2017', 'validation_population_caveat': 'historical eligibility selects users active after each cutoff; MAE is not a full-test estimate', 'cross_cutoff_audit': audit_metrics, 'numerical_checks': checks, 'fixed_config': {'grid_K': 96, 'posterior_beta': 1.0, 'posterior_gamma': 1.0, 'posterior_kappa': 0.5, 'rich_beta': 1.2, 'rich_gamma': 1.0, 'rich_kappa': 0.8, 'hgb': delta_HGB_PARAMS, 'weights': {'residual_hgb': 0.8, 'terminal_nm': 0.05, 'posterior': 0.15}}, 'prediction_count': len(predictions), 'invalid_time_fallback_count': int((~valid).sum()), 'test_prediction_quantiles_days': delta_np.quantile(predictions / delta_DAY, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(), 'elapsed_seconds': float(delta_time.time() - started)}
    log(f"Task 6 July user-isolated terminal audit: {audit_metrics['2017-07']}")
    return metrics

# Final entry point. All six tasks use only the designated training file.
def _finite_json(value):
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description='CSE258 A1: fixed validated models and CSV generation')
    parser.add_argument('--data-dir', type=Path, default=Path('assignment1_2026'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--tasks', nargs='+', type=int, default=[1,2,3,4,5,6], choices=range(1,7))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir/'validation_metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    started = time.time()
    tasks = set(args.tasks)
    for name, group in [('tasks_1_2_train.csv',[1,2]), ('tasks_3_4_train.csv',[3,4]),
                        ('task_5_train.csv',[5]), ('task_6_train.csv',[6])]:
        active = [task for task in group if task in tasks]
        if not active:
            continue
        loader = interactions_load_task5_order_only if active == [5] else load
        a, users, items = loader(args.data_dir/name)
        for task in active:
            if task in (1,3):
                result = ratings_production_rating_task(task,a,users,items,args)
            elif task in (2,4,5):
                result = interactions_production_interaction_task(task,a,users,items,args)
            else:
                result = delta_production_delta_task(a,users,items,args)
            metrics[str(task)] = _finite_json(result)
            metrics_path.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        del a, users, items
    log(f'Completed tasks {sorted(tasks)} in {time.time()-started:.2f}s; leaderboard labels were not accessed.')


if __name__ == '__main__':
    main()

