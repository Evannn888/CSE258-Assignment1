"""Production interaction models; task-specific data only, no artifact/cache dependency."""
import csv
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
import assignment1 as base

def log(message):
    base.log(message)

def load_task5_order_only(path):
    """Dedicated loader for Task 5 that ONLY parses user, item, int(rating) and NEVER reads timestamp."""
    users, items, rows = {}, {}, []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            u, i = row["user"], row["item"]
            if u not in users:
                users[u] = len(users)
            if i not in items:
                items[i] = len(items)
            # Column 2 is set to constant 0; row['timestamp'] is never read.
            rows.append((users[u], items[i], 0, int(row["rating"])))
    a = np.asarray(rows, dtype=np.int64)
    log(f"[Pure-Order Loader] {Path(path).name}: {len(a):,} rows, {len(users):,} users, {len(items):,} items")
    return a, users, items

def make_interaction_splits_dual(a, nu, ni, sequential=False, seed=258):
    """Construct validation splits under BOTH negative sampling mechanisms:
    - pool_A: negatives sampled from fit_rows[:, 1] (after holding out positives)
    - pool_B: negatives sampled from original_rows[:, 1] (before holding out positives)
    When seed=258, pool_A matches assignment1.make_interaction_split bit-for-bit.
    """
    rng = np.random.default_rng(seed)
    if sequential:
        last = np.full(nu, -1, np.int64)
        np.maximum.at(last, a[:, 0], np.arange(len(a)))
        held = last[last >= 0]
    else:
        held = rng.choice(len(a), min(30000, len(a) // 10), replace=False)
    mask = np.ones(len(a), dtype=bool)
    mask[held] = False
    fit_rows, positive = a[mask], a[held]

    # pool_A uses rng directly so seed=258 matches assignment1.py bit-for-bit
    neg_A = fit_rows[rng.integers(0, len(fit_rows), len(positive)), 1].copy()
    for n in range(len(positive)):
        while neg_A[n] == positive[n, 1]:
            neg_A[n] = fit_rows[rng.integers(0, len(fit_rows)), 1]
    va_A = np.column_stack([positive[:, 0], positive[:, 1], neg_A, positive[:, 2]])

    # pool_B samples from original_rows[:, 1] before holding out positives
    rng_B = np.random.default_rng(seed + 10000)
    neg_B = a[rng_B.integers(0, len(a), len(positive)), 1].copy()
    for n in range(len(positive)):
        while neg_B[n] == positive[n, 1]:
            neg_B[n] = a[rng_B.integers(0, len(a)), 1]
    va_B = np.column_stack([positive[:, 0], positive[:, 1], neg_B, positive[:, 2]])

    return fit_rows, va_A, va_B

@njit(cache=True)
def compute_seen_and_zero_flags(q_u, q_i1, q_i2, u_ptr, u_items, i_cnt):
    """Compute:
    - seen_override = (seen2 - seen1) ONLY when one candidate is in H(u) (0 if both or neither)
    - zero_diff = (c1==0) - (c2==0)
    - zero_sum = (c1==0) + (c2==0)
    - zero_neq = 1.0 if (c1==0) != (c2==0) else 0.0
    """
    n = len(q_u)
    nu = len(u_ptr) - 1
    ni = len(i_cnt)
    seen_override = np.zeros(n, dtype=np.float64)
    zero_diff = np.zeros(n, dtype=np.float64)
    zero_sum = np.zeros(n, dtype=np.float64)
    zero_neq = np.zeros(n, dtype=np.float64)

    for idx in range(n):
        u, i1, i2 = q_u[idx], q_i1[idx], q_i2[idx]
        c1 = i_cnt[i1] if i1 < ni else 0
        c2 = i_cnt[i2] if i2 < ni else 0
        z1 = 1.0 if c1 == 0 else 0.0
        z2 = 1.0 if c2 == 0 else 0.0
        zero_diff[idx] = z1 - z2
        zero_sum[idx] = z1 + z2
        zero_neq[idx] = 1.0 if z1 != z2 else 0.0

        if u < nu:
            s1, s2 = 0.0, 0.0
            for p in range(u_ptr[u], u_ptr[u + 1]):
                j = u_items[p]
                if j == i1:
                    s1 = 1.0
                if j == i2:
                    s2 = 1.0
            # Only non-zero when exactly one candidate is in H(u)
            seen_override[idx] = s2 - s1
    return seen_override, zero_diff, zero_sum, zero_neq

@njit(cache=True)
def extract_bt_pairs(u_ptr, u_items):
    """Deterministically sample directed pairs (a < b) from each user sequence at
    lags 1, 3, 10 plus first-to-last anchor, weighted by 1 / sqrt(n_pairs(u)).
    """
    total = 0
    nu = len(u_ptr) - 1
    for u in range(nu):
        L = u_ptr[u + 1] - u_ptr[u]
        if L < 2:
            continue
        np_u = (L - 1) + max(0, L - 3) + max(0, L - 10) + (1 if L > 2 else 0)
        total += np_u
    earlier = np.empty(total, dtype=np.int32)
    later = np.empty(total, dtype=np.int32)
    weights = np.empty(total, dtype=np.float64)
    u_owner = np.empty(total, dtype=np.int32)
    pos = 0
    for u in range(nu):
        s, e = u_ptr[u], u_ptr[u + 1]
        L = e - s
        if L < 2:
            continue
        start_pos = pos
        for k in range(L - 1):
            a_i, b_i = u_items[s + k], u_items[s + k + 1]
            if a_i != b_i:
                earlier[pos] = a_i
                later[pos] = b_i
                u_owner[pos] = u
                pos += 1
        for k in range(L - 3):
            a_i, b_i = u_items[s + k], u_items[s + k + 3]
            if a_i != b_i:
                earlier[pos] = a_i
                later[pos] = b_i
                u_owner[pos] = u
                pos += 1
        for k in range(L - 10):
            a_i, b_i = u_items[s + k], u_items[s + k + 10]
            if a_i != b_i:
                earlier[pos] = a_i
                later[pos] = b_i
                u_owner[pos] = u
                pos += 1
        if L > 2:
            a_i, b_i = u_items[s], u_items[e - 1]
            if a_i != b_i:
                earlier[pos] = a_i
                later[pos] = b_i
                u_owner[pos] = u
                pos += 1
        np_u = pos - start_pos
        if np_u > 0:
            weights[start_pos:pos] = 1.0 / np.sqrt(float(np_u))
    return earlier[:pos], later[:pos], weights[:pos], u_owner[:pos]

def _solve_bt_lbfgs(ear, lat, w, n_items, reg=1.0, maxiter=25):
    def loss_and_grad(era):
        z = era[lat] - era[ear]
        loss = float(np.sum(w * np.logaddexp(0.0, -z)) + 0.5 * reg * np.sum(era * era))
        g = w * expit(-z)
        grad = np.bincount(ear, weights=g, minlength=n_items)
        grad -= np.bincount(lat, weights=g, minlength=n_items)
        grad += reg * era
        return loss, grad

    res = minimize(
        loss_and_grad,
        np.zeros(n_items, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter},
    )
    return res.x, res

def fit_order_tendency(u_ptr, u_items, n_items, reg=1.0, maxiter=25, compute_diagnostics=False):
    """Fit Bradley-Terry order_tendency axis e_i and curvature proxy J_i."""
    ear, lat, w, u_owner = extract_bt_pairs(u_ptr, u_items)
    era, res = _solve_bt_lbfgs(ear, lat, w, n_items, reg=reg, maxiter=maxiter)
    z = era[lat] - era[ear]
    p = expit(z)
    cw = w * p * (1.0 - p)
    J = reg + np.bincount(ear, weights=cw, minlength=n_items) + np.bincount(lat, weights=cw, minlength=n_items)
    pair_counts = np.bincount(ear, minlength=n_items) + np.bincount(lat, minlength=n_items)

    diag = {}
    if compute_diagnostics:
        in_sample_acc = float(np.mean((z > 0) + 0.5 * (z == 0)))
        # Cross-user out-of-sample pairwise ordering accuracy (hold out 20% of users)
        u_split = (u_owner % 5 != 0)
        era_tr, _ = _solve_bt_lbfgs(ear[u_split], lat[u_split], w[u_split], n_items, reg=reg, maxiter=maxiter)
        ho = ~u_split
        z_ho = era_tr[lat[ho]] - era_tr[ear[ho]]
        seen_tr = (np.bincount(ear[u_split], minlength=n_items) + np.bincount(lat[u_split], minlength=n_items)) > 0
        both_seen = seen_tr[ear[ho]] & seen_tr[lat[ho]]
        oos_acc_all = float(np.mean((z_ho > 0) + 0.5 * (z_ho == 0)))
        oos_acc_both_seen = float(np.mean((z_ho[both_seen] > 0) + 0.5 * (z_ho[both_seen] == 0)))

        # Bidirectional conflict ratio on repeated unordered item pairs
        p_min = np.minimum(ear, lat).astype(np.int64)
        p_max = np.maximum(ear, lat).astype(np.int64)
        keys = p_min * np.int64(n_items) + p_max
        dirs = (ear < lat).astype(np.int64)
        order_k = np.argsort(keys)
        keys_s = keys[order_k]
        dirs_s = dirs[order_k]
        _, idx_start, counts = np.unique(keys_s, return_index=True, return_counts=True)
        multi = counts > 1
        dir_sum = np.add.reduceat(dirs_s, idx_start)[multi]
        cnt_multi = counts[multi]
        conflicts = int(np.sum((dir_sum > 0) & (dir_sum < cnt_multi)))
        n_multi = int(len(cnt_multi))

        diag = {
            "n_directed_pairs": int(len(ear)),
            "n_unique_unordered_pairs": int(len(counts)),
            "n_multi_observation_pairs": n_multi,
            "n_conflicting_pairs": conflicts,
            "bidirectional_conflict_ratio": float(conflicts / max(1, n_multi)),
            "in_sample_pair_accuracy": in_sample_acc,
            "oos_cross_user_pair_accuracy_all": oos_acc_all,
            "oos_cross_user_pair_accuracy_both_seen": oos_acc_both_seen,
            "n_oos_pairs_all": int(np.sum(ho)),
            "n_oos_pairs_both_seen": int(np.sum(both_seen)),
            "lbfgs_iterations": int(res.nit),
            "lbfgs_final_loss": float(res.fun),
            "era_std": float(np.std(era)),
            "mean_item_pair_count": float(np.mean(pair_counts)),
            "items_with_pairs": int(np.sum(pair_counts > 0)),
        }
    return era, J, pair_counts, diag

@njit(cache=True)
def task5_bt_features(q_u, q_i1, q_i2, u_ptr, u_items, i_cnt, era, J, reg=1.0):
    """Compute candidate features relative to user u's recent order_tendency E_u:
    decayed mean of last 3, 5, 10 items, 90th percentile q90(u), recent trend E_{u,last3} - E_{u,first3},
    e_i - E_u, |e_i - E_u|, max(0, e_i - E_u), max(0, E_u - e_i), indicator e_i > q90(u),
    e_i - e_{i_{u,L}}, reliability log(J_i / lambda), and unknown-item flags (e_i=0, conf=0, missing=1 when c_i=0).
    """
    n = len(q_u)
    nu = len(u_ptr) - 1
    ni = len(i_cnt)
    d_out = np.zeros((n, 16), dtype=np.float64)
    s_out = np.zeros((n, 16), dtype=np.float64)
    u_ctx = np.zeros((n, 4), dtype=np.float64)

    E_dec3 = np.zeros(nu + 1, dtype=np.float64)
    E_dec5 = np.zeros(nu + 1, dtype=np.float64)
    E_dec10 = np.zeros(nu + 1, dtype=np.float64)
    q90_u = np.zeros(nu + 1, dtype=np.float64)
    trend_u = np.zeros(nu + 1, dtype=np.float64)
    e_last_u = np.zeros(nu + 1, dtype=np.float64)
    u_conf = np.zeros(nu + 1, dtype=np.float64)

    for u in range(nu):
        s, e = u_ptr[u], u_ptr[u + 1]
        L = e - s
        if L == 0:
            continue
        seq_e = np.empty(L, dtype=np.float64)
        c_sum = 0.0
        for p in range(L):
            it = u_items[s + p]
            if it < ni and i_cnt[it] > 0:
                seq_e[p] = era[it]
                c_sum += np.log(J[it] / reg)
            else:
                seq_e[p] = 0.0
        u_conf[u] = c_sum / float(L)
        e_last_u[u] = seq_e[L - 1]

        w3, s3 = 0.0, 0.0
        for k in range(min(L, 3)):
            w = 0.8 ** k
            s3 += w * seq_e[L - 1 - k]
            w3 += w
        E_dec3[u] = s3 / w3

        w5, s5 = 0.0, 0.0
        for k in range(min(L, 5)):
            w = 0.8 ** k
            s5 += w * seq_e[L - 1 - k]
            w5 += w
        E_dec5[u] = s5 / w5

        w10, s10 = 0.0, 0.0
        for k in range(min(L, 10)):
            w = 0.85 ** k
            s10 += w * seq_e[L - 1 - k]
            w10 += w
        E_dec10[u] = s10 / w10

        m_last3, m_first3 = 0.0, 0.0
        k3 = min(L, 3)
        for k in range(k3):
            m_last3 += seq_e[L - 1 - k]
            m_first3 += seq_e[k]
        trend_u[u] = (m_last3 - m_first3) / float(k3)

        sorted_e = np.sort(seq_e)
        idx90 = 0.9 * float(L - 1)
        lo = int(np.floor(idx90))
        hi = int(np.ceil(idx90))
        q90_u[u] = sorted_e[lo] + (idx90 - float(lo)) * (sorted_e[hi] - sorted_e[lo])

    for idx in range(n):
        u, i1, i2 = q_u[idx], q_i1[idx], q_i2[idx]
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

        f_cand = np.zeros((2, 16), dtype=np.float64)
        for side in range(2):
            item = i1 if side == 0 else i2
            ci = float(i_cnt[item]) if item < ni else 0.0
            if item < ni and ci > 0.0:
                ei = era[item]
                conf_i = np.log(J[item] / reg)
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
            f_cand[side, 10] = 1.0 if (miss_i == 0.0 and ei > q90) else 0.0
            f_cand[side, 11] = abs(difflast)
            f_cand[side, 12] = max(0.0, difflast)
            f_cand[side, 13] = max(0.0, -difflast)
            f_cand[side, 14] = rel_i * diff3
            f_cand[side, 15] = conf_i

        for k in range(16):
            d_out[idx, k] = f_cand[0, k] - f_cand[1, k]
            s_out[idx, k] = f_cand[0, k] + f_cand[1, k]
    return d_out, s_out, u_ctx

@njit(cache=True)
def task4_session_window_features(
    q_u, q_i1, q_i2, q_t, u_ptr, u_items, u_times, i_ptr, i_users, i_times_sorted,
    u_cnt, i_cnt, all_times_sorted, i_unit
):
    """Compute:
    - Global-event-normalized asymmetric window lifts for h in {1h, 1d, 7d, 30d, 180d}:
      L^-, L^+, L^+ - L^-, and window support.
    - Multi-scale session similarity for tau in {600s, 3600s, 86400s, 7d, 30d} using both
      co-occurrence kernel K_co(i, j) and spectral unit kernel K_spec(i, j).
    - Nearest past (t_uj <= t) and nearest future (t_uj > t) item co-occurrence lift and spectral similarity.
    - Same-hour / same-day user activity counts and session support.
    """
    n = len(q_u)
    d_out = np.zeros((n, 34), dtype=np.float64)
    s_out = np.zeros((n, 34), dtype=np.float64)
    u_ctx = np.zeros((n, 14), dtype=np.float64)

    nu = len(u_cnt)
    ni = len(i_cnt)
    n_total = float(len(all_times_sorted))
    denom_q = n_total + float(ni)
    dim = i_unit.shape[1]

    in_u1 = np.zeros(nu, dtype=np.float64)
    in_u2 = np.zeros(nu, dtype=np.float64)
    inv_u = 1.0 / (u_cnt.astype(np.float64) + 2.0)

    hs = np.array([3600, 86400, 7 * 86400, 30 * 86400, 180 * 86400], dtype=np.int64)
    taus = np.array([600.0, 3600.0, 86400.0, 7.0 * 86400.0, 30.0 * 86400.0], dtype=np.float64)
    alpha = 10.0
    lam_tau = 0.5

    for idx in range(n):
        u, i1, i2, t = q_u[idx], q_i1[idx], q_i2[idx], q_t[idx]
        f_cand = np.zeros((2, 34), dtype=np.float64)

        pos_t_left = np.searchsorted(all_times_sorted, t, side="left")
        pos_t_right = np.searchsorted(all_times_sorted, t, side="right")

        for w_idx in range(5):
            h = hs[w_idx]
            N_minus = float(pos_t_left - np.searchsorted(all_times_sorted, t - h, side="left"))
            N_plus = float(np.searchsorted(all_times_sorted, t + h, side="right") - pos_t_right)
            if w_idx < 3:
                u_ctx[idx, 9 + w_idx] = np.log1p(N_minus + N_plus)

            for side in range(2):
                item = i1 if side == 0 else i2
                ci = float(i_cnt[item]) if item < ni else 0.0
                qi = (ci + 1.0) / denom_q
                log_qi = np.log(qi)
                if item < ni and ci > 0.0:
                    ts = i_times_sorted[i_ptr[item]:i_ptr[item + 1]]
                    p_left = np.searchsorted(ts, t, side="left")
                    p_right = np.searchsorted(ts, t, side="right")
                    n_m = float(p_left - np.searchsorted(ts, t - h, side="left"))
                    n_p = float(np.searchsorted(ts, t + h, side="right") - p_right)
                else:
                    n_m = 0.0
                    n_p = 0.0
                L_m = np.log((n_m + alpha * qi) / (N_minus + alpha)) - log_qi
                L_p = np.log((n_p + alpha * qi) / (N_plus + alpha)) - log_qi
                base_col = w_idx * 4
                f_cand[side, base_col + 0] = L_m
                f_cand[side, base_col + 1] = L_p
                f_cand[side, base_col + 2] = L_p - L_m
                f_cand[side, base_col + 3] = np.log1p(n_m + n_p)

        if u < nu:
            ustart, uend = u_ptr[u], u_ptr[u + 1]
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
                W_tau = np.zeros(5, dtype=np.float64)
                num_co = np.zeros((2, 5), dtype=np.float64)
                num_sp = np.zeros((2, 5), dtype=np.float64)

                best_past_dt = 1e18
                best_fut_dt = 1e18
                past_lift1, past_lift2 = 0.0, 0.0
                fut_lift1, fut_lift2 = 0.0, 0.0
                past_sp1, past_sp2 = 0.0, 0.0
                fut_sp1, fut_sp2 = 0.0, 0.0
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
                    jstart, jend = i_ptr[j], i_ptr[j + 1]
                    w1, w2 = 0.0, 0.0
                    for q in range(jstart, jend):
                        up = i_users[q]
                        w1 += in_u1[up]
                        w2 += in_u2[up]
                    v1_co = w1 * inv_cj
                    v2_co = w2 * inv_cj
                    k_co1 = w1 / np.sqrt((c1 + 5.0) * (cj + 5.0))
                    k_co2 = w2 / np.sqrt((c2 + 5.0) * (cj + 5.0))

                    vj_vec = i_unit[j]
                    k_sp1, k_sp2 = 0.0, 0.0
                    for k in range(dim):
                        k_sp1 += v1_vec[k] * vj_vec[k]
                        k_sp2 += v2_vec[k] * vj_vec[k]

                    for m in range(5):
                        wm = np.exp(-dt / taus[m])
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
                    f_cand[0, 20 + m] = np.log((num_co[0, m] / denom_m) + 1e-5)
                    f_cand[1, 20 + m] = np.log((num_co[1, m] / denom_m) + 1e-5)
                    f_cand[0, 25 + m] = num_sp[0, m] / denom_m
                    f_cand[1, 25 + m] = num_sp[1, m] / denom_m
                    u_ctx[idx, 2 + m] = np.log1p(W_tau[m])

                f_cand[0, 30] = np.log((past_lift1 + 1e-6) / (c1 + 1.0))
                f_cand[1, 30] = np.log((past_lift2 + 1e-6) / (c2 + 1.0))
                f_cand[0, 31] = np.log((fut_lift1 + 1e-6) / (c1 + 1.0))
                f_cand[1, 31] = np.log((fut_lift2 + 1e-6) / (c2 + 1.0))
                f_cand[0, 32] = past_sp1
                f_cand[1, 32] = past_sp2
                f_cand[0, 33] = fut_sp1
                f_cand[1, 33] = fut_sp2

                u_ctx[idx, 0] = np.log1p(same_hr)
                u_ctx[idx, 1] = np.log1p(same_day)
                u_ctx[idx, 7] = np.log1p(best_past_dt / 3600.0) if best_past_dt < 1e17 else 10.0
                u_ctx[idx, 8] = np.log1p(best_fut_dt / 3600.0) if best_fut_dt < 1e17 else 10.0
                u_ctx[idx, 12] = 1.0 if best_past_dt < 1e17 else 0.0
                u_ctx[idx, 13] = 1.0 if best_fut_dt < 1e17 else 0.0

        for k in range(34):
            d_out[idx, k] = f_cand[0, k] - f_cand[1, k]
            s_out[idx, k] = f_cand[0, k] + f_cand[1, k]
    return d_out, s_out, u_ctx

@njit(cache=True)
def task2_stage_features(
    q_u, q_i1, q_i2, u_ptr, u_items, u_times, i_ptr, i_times, i_cnt, all_times_sorted, i_unit
):
    """Split user u's timestamped history into S=3 equal-count temporal segments,
    compute segment unit-spectral centroids m_{us} and segment time bounds, and evaluate:
    T_{ui} = sum_{s=0}^2 pi_{us} exp(2.0 <v_i, m_{us}>) f_hat_i(t_bar_{us}) / (g_hat(t_bar_{us}) + eps).
    """
    n = len(q_u)
    d_out = np.zeros((n, 9), dtype=np.float64)
    s_out = np.zeros((n, 9), dtype=np.float64)
    nu = len(u_ptr) - 1
    ni = len(i_cnt)
    n_total = float(len(all_times_sorted))
    dim = i_unit.shape[1]
    W90 = 90 * 86400
    W30 = 30 * 86400
    kappa = 5.0

    for idx in range(n):
        u, i1, i2 = q_u[idx], q_i1[idx], q_i2[idx]
        if u >= nu:
            continue
        ustart, uend = u_ptr[u], u_ptr[u + 1]
        L_raw = uend - ustart
        if L_raw == 0:
            continue
        valid_pos = np.empty(L_raw, dtype=np.int32)
        L_eff = 0
        for p in range(ustart, uend):
            j = u_items[p]
            if j != i1 and j != i2:
                valid_pos[L_eff] = p
                L_eff += 1
        if L_eff == 0:
            continue

        m_us = np.zeros((3, dim), dtype=np.float64)
        pi_us = np.zeros(3, dtype=np.float64)
        t_bar = np.zeros(3, dtype=np.int64)
        t_min_s = np.zeros(3, dtype=np.int64)
        t_max_s = np.zeros(3, dtype=np.int64)
        g_A = np.zeros(3, dtype=np.float64)
        g_B = np.zeros(3, dtype=np.float64)

        for s in range(3):
            s_lo = (s * L_eff) // 3
            s_hi = ((s + 1) * L_eff) // 3
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
            norm_m = np.sqrt(norm_m) + 1e-8
            for k in range(dim):
                m_us[s, k] /= norm_m

            tb = u_times[valid_pos[(s_lo + s_hi) // 2]]
            tmin = u_times[valid_pos[s_lo]]
            tmax = u_times[valid_pos[s_hi - 1]]
            t_bar[s] = tb
            t_min_s[s] = tmin
            t_max_s[s] = tmax

            n_gA = (
                np.searchsorted(all_times_sorted, tb + W90, side="right")
                - np.searchsorted(all_times_sorted, tb - W90, side="left")
            )
            g_A[s] = (float(n_gA) + 1.0) / (n_total + 1.0)
            n_gB = (
                np.searchsorted(all_times_sorted, tmax + W30, side="right")
                - np.searchsorted(all_times_sorted, tmin - W30, side="left")
            )
            g_B[s] = (float(n_gB) + 1.0) / (n_total + 1.0)

        f_cand = np.zeros((2, 9), dtype=np.float64)
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
                exp_sim = np.exp(2.0 * sim)

                if item < ni and ci > 0.0:
                    ts = i_times[i_ptr[item]:i_ptr[item + 1]]
                    n_iA = (
                        np.searchsorted(ts, t_bar[s] + W90, side="right")
                        - np.searchsorted(ts, t_bar[s] - W90, side="left")
                    )
                    f_iA = (float(n_iA) + kappa * g_A[s]) / (ci + kappa)
                    R_A = f_iA / (g_A[s] + 1e-8)

                    n_iB = (
                        np.searchsorted(ts, t_max_s[s] + W30, side="right")
                        - np.searchsorted(ts, t_min_s[s] - W30, side="left")
                    )
                    f_iB = (float(n_iB) + kappa * g_B[s]) / (ci + kappa)
                    R_B = f_iB / (g_B[s] + 1e-8)
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

            f_cand[side, 0] = np.log(T_A + 1e-6)
            f_cand[side, 1] = np.log(T_B + 1e-6)
            f_cand[side, 2] = np.log(T_spec + 1e-6)
            f_cand[side, 3] = max_sim
            f_cand[side, 4] = min_sim
            f_cand[side, 5] = np.log(lift_A + 1e-6)
            f_cand[side, 6] = np.log(lift_B + 1e-6)
            f_cand[side, 7] = np.log(max_lift_A + 1e-6)
            f_cand[side, 8] = max_term_A / (T_A + 1e-8)

        for k in range(9):
            d_out[idx, k] = f_cand[0, k] - f_cand[1, k]
            s_out[idx, k] = f_cand[0, k] + f_cand[1, k]
    return d_out, s_out

def fit_equivariant_pair_embedding(a_fit, nu, ni, u_imp, s_imp, vt_imp, inv_i, steps=20, lr=0.05, reg=0.01):
    """Deterministic mean-negative pair embedding; this is a surrogate, not exact BPR."""
    signs = np.where(np.sum(u_imp ** 3, axis=0) >= 0, 1.0, -1.0)
    u_c = u_imp * signs[None, :]
    v_c = vt_imp.T * signs[None, :]
    k = min(32, u_c.shape[1])
    P = (u_c[:, -k:] * (s_imp[-k:] ** 0.5)).astype(np.float64)
    Q = (v_c[:, -k:] * (s_imp[-k:] ** 0.5) * (inv_i[:, None] ** 0.5)).astype(np.float64)

    u_idx, i_idx = a_fit[:, 0], a_fit[:, 1]
    i_cnt = np.bincount(i_idx, minlength=ni).astype(np.float64)
    pop_prob = i_cnt / max(1.0, float(np.sum(i_cnt)))

    for _ in range(steps):
        q_bar = pop_prob @ Q
        margin = np.sum(P[u_idx] * (Q[i_idx] - q_bar[None, :]), axis=1)
        g = expit(-np.clip(margin, -20.0, 20.0))
        G_mat = csr_matrix((g, (u_idx, i_idx)), shape=(nu, ni))
        grad_P = (G_mat @ Q) - np.asarray(G_mat.sum(axis=1)) * q_bar[None, :] - reg * P
        grad_Q = (G_mat.T @ P) - np.outer(pop_prob, np.sum(g[:, None] * P[u_idx], axis=0)) - reg * Q
        P += lr * (grad_P / (np.linalg.norm(grad_P, axis=0, keepdims=True) + 1e-8))
        Q += lr * (grad_Q / (np.linalg.norm(grad_Q, axis=0, keepdims=True) + 1e-8))

    P_ext = np.vstack([P, np.zeros((1, k), dtype=np.float64)])
    Q_ext = np.vstack([Q, np.zeros((1, k), dtype=np.float64)])
    return P_ext, Q_ext

def stable_unit_rows(values, threshold=1e-7):
    """Zero numerical null-space projections before normalization.

    Two successive normalizations otherwise amplify floating point noise for
    disconnected components absent from the retained spectral subspace.
    """
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > threshold)

def fit_graph_state(task, a_fit, nu, ni, bpr_epochs=25, compute_bt_diag=False, equivariant_mode=False):
    """Fit all graph/spectral/BPR/order_tendency structures on a_fit once, so multiple
    query sets (e.g. va_A, va_B, test, single-row) can be scored against the exact same state.
    """
    u_cnt = np.bincount(a_fit[:, 0], minlength=nu)
    i_cnt = np.bincount(a_fit[:, 1], minlength=ni)
    inv_u = 1.0 / np.sqrt(u_cnt + 5.0)
    inv_i = 1.0 / np.sqrt(i_cnt + 10.0)

    state = {
        "task": task,
        "a_fit": a_fit,
        "nu": nu,
        "ni": ni,
        "u_cnt": u_cnt,
        "i_cnt": i_cnt,
        "inv_u": inv_u,
        "inv_i": inv_i,
    }

    if task in (2, 4):
        u_order = np.lexsort((a_fit[:, 2], a_fit[:, 0]))
        u_ptr = np.r_[0, u_cnt.cumsum()]
        u_items = a_fit[u_order, 1].astype(np.int32)
        u_times = a_fit[u_order, 2]
        u_rats = a_fit[u_order, 3]

        i_order = np.lexsort((a_fit[:, 2], a_fit[:, 1]))
        i_ptr = np.r_[0, i_cnt.cumsum()]
        i_users = a_fit[i_order, 0].astype(np.int32)
        i_times = a_fit[i_order, 2]
        all_times_sorted = np.sort(a_fit[:, 2])

        x_imp = csr_matrix((inv_u[a_fit[:, 0]] * inv_i[a_fit[:, 1]], (a_fit[:, 0], a_fit[:, 1])), shape=(nu, ni))
        u_imp, s_imp, vt_imp = svds(x_imp, k=64, random_state=base.SEED)
        u_vec = np.vstack([(u_imp * s_imp).astype(np.float32), np.zeros((1, 64), np.float32)])
        i_vec = np.vstack([(vt_imp.T * inv_i[:, None]).astype(np.float32), np.zeros((1, 64), np.float32)])

        u_emb = (u_imp * (s_imp ** 0.5)).astype(np.float32)
        u_unit = stable_unit_rows(u_emb)
        i_cent = np.zeros((ni + 1, 64), dtype=np.float32)
        np.add.at(i_cent, a_fit[:, 1], u_unit[a_fit[:, 0]])
        i_unit = stable_unit_rows(i_cent)
        u_unit_ext = np.vstack([u_unit, np.zeros((1, 64), np.float32)])

        p, q = base.train_bpr_pop(a_fit, nu, ni, bpr_epochs)
        p = np.vstack([p, np.zeros((1, 32), np.float32)])
        q = np.vstack([q, np.zeros((1, 32), np.float32)])

        state.update({
            "u_ptr": u_ptr, "u_items": u_items, "u_times": u_times, "u_rats": u_rats,
            "i_ptr": i_ptr, "i_users": i_users, "i_times": i_times,
            "all_times_sorted": all_times_sorted,
            "u_vec": u_vec, "i_vec": i_vec, "u_unit_ext": u_unit_ext, "i_unit": i_unit,
            "p_bpr": p, "q_bpr": q,
        })
        return state

    # Task 5: strictly sequential order (no timestamps used)
    u_order = np.argsort(a_fit[:, 0], kind="mergesort")
    u_ptr = np.r_[0, u_cnt.cumsum()]
    u_items = a_fit[u_order, 1].astype(np.int32)
    u_rats = a_fit[u_order, 3]

    u_pos = np.zeros(len(a_fit), dtype=np.int32)
    u_rel_pos = np.zeros(len(a_fit), dtype=np.float64 if equivariant_mode else np.float32)
    last1_cnt = np.zeros(ni, dtype=np.float64 if equivariant_mode else np.float32)
    last3_cnt = np.zeros(ni, dtype=np.float64 if equivariant_mode else np.float32)
    last5_cnt = np.zeros(ni, dtype=np.float64 if equivariant_mode else np.float32)

    for u in range(nu):
        s, e = u_ptr[u], u_ptr[u + 1]
        L = e - s
        if L > 0:
            p_idx = np.arange(L, dtype=np.int32)
            u_pos[s:e] = p_idx
            u_rel_pos[s:e] = (p_idx + 1.0) / float(L)
            last1_cnt[u_items[e - 1]] += 1.0
            for p_i in range(max(s, e - 3), e):
                last3_cnt[u_items[p_i]] += 1.0
            for p_i in range(max(s, e - 5), e):
                last5_cnt[u_items[p_i]] += 1.0

    row_pos = np.zeros(len(a_fit), dtype=np.int32)
    row_rel_pos = np.zeros(len(a_fit), dtype=np.float64 if equivariant_mode else np.float32)
    row_pos[u_order] = u_pos
    row_rel_pos[u_order] = u_rel_pos

    item_rel_sum = np.bincount(a_fit[:, 1], weights=row_rel_pos, minlength=ni)
    item_rel_mean = ((item_rel_sum + 2.0 * 0.58) / (i_cnt + 2.0)).astype(np.float32)
    item_rel4_sum = np.bincount(a_fit[:, 1], weights=row_rel_pos ** 4, minlength=ni)
    item_rel4_mean = ((item_rel4_sum + 2.0 * 0.24) / (i_cnt + 2.0)).astype(np.float32)

    item_last1_rate = np.log((last1_cnt + 0.3) / (i_cnt + 3.0)).astype(np.float32)
    item_last3_rate = np.log((last3_cnt + 0.6) / (i_cnt + 3.0)).astype(np.float32)
    item_last5_rate = np.log((last5_cnt + 1.0) / (i_cnt + 3.0)).astype(np.float32)

    u_end_era = np.full(nu + 1, 0.60, dtype=np.float32)
    for u in range(nu):
        s, e = u_ptr[u], u_ptr[u + 1]
        if e > s:
            u_end_era[u] = np.mean(item_rel_mean[u_items[max(s, e - 3):e]])

    i_order = np.argsort(a_fit[:, 1], kind="mergesort")
    i_ptr = np.r_[0, i_cnt.cumsum()]
    i_users = a_fit[i_order, 0].astype(np.int32)
    i_pos = row_pos[i_order]

    rec_weights = (0.5 + 0.5 * row_rel_pos) * inv_u[a_fit[:, 0]] * inv_i[a_fit[:, 1]]
    x_imp = csr_matrix((rec_weights, (a_fit[:, 0], a_fit[:, 1])), shape=(nu, ni))
    v0 = 1.0 / np.sqrt(u_cnt + 5.0)
    u_imp, s_imp, vt_imp = svds(x_imp, k=64, v0=v0, random_state=base.SEED)
    order_sv = np.argsort(s_imp)
    u_imp, s_imp, vt_imp = u_imp[:, order_sv], s_imp[order_sv], vt_imp[order_sv, :]

    u_vec = np.vstack([(u_imp * s_imp).astype(np.float64), np.zeros((1, 64), np.float64)])
    i_vec = np.vstack([(vt_imp.T * inv_i[:, None]).astype(np.float64), np.zeros((1, 64), np.float64)])

    u_emb = (u_imp * (s_imp ** 0.5)).astype(np.float64)
    u_unit = stable_unit_rows(u_emb)
    i_cent = np.zeros((ni + 1, 64), dtype=np.float64)
    np.add.at(i_cent, a_fit[:, 1], u_unit[a_fit[:, 0]])
    i_unit = stable_unit_rows(i_cent).astype(np.float32)
    u_unit_ext = np.vstack([u_unit, np.zeros((1, 64), np.float64)])

    v_item = (vt_imp.T * (s_imp ** 0.5) * inv_i[:, None]).astype(np.float64)
    u_decay = np.zeros((nu + 1, 64), dtype=np.float64)
    u_last1 = np.zeros((nu + 1, 64), dtype=np.float64)
    for u in range(nu):
        s, e = u_ptr[u], u_ptr[u + 1]
        if e > s:
            u_last1[u] = v_item[u_items[e - 1]]
            start_k = max(s, e - 15)
            w_sum = 0.0
            for p_idx in range(e - 1, start_k - 1, -1):
                w = 0.75 ** (e - 1 - p_idx)
                u_decay[u] += w * v_item[u_items[p_idx]]
                w_sum += w
            u_decay[u] /= w_sum
    v_item_ext = np.vstack([v_item, np.zeros((1, 64), np.float64)])

    p_eq, q_eq = fit_equivariant_pair_embedding(a_fit, nu, ni, u_imp, s_imp, vt_imp, inv_i)
    era, J, pair_counts, bt_diag = fit_order_tendency(
        u_ptr, u_items, ni, reg=1.0, maxiter=25, compute_diagnostics=compute_bt_diag
    )

    state.update({
        "u_ptr": u_ptr, "u_items": u_items, "u_rats": u_rats,
        "i_ptr": i_ptr, "i_users": i_users, "i_pos": i_pos,
        "item_rel_mean": item_rel_mean, "item_rel4_mean": item_rel4_mean,
        "item_last1_rate": item_last1_rate, "item_last3_rate": item_last3_rate, "item_last5_rate": item_last5_rate,
        "u_end_era": u_end_era,
        "u_vec": u_vec, "i_vec": i_vec, "u_unit_ext": u_unit_ext, "i_unit": i_unit,
        "u_decay": u_decay, "u_last1": u_last1, "v_item_ext": v_item_ext,
        "p_bpr": p_eq, "q_bpr": q_eq,
        "era": era, "J": J, "bt_pair_counts": pair_counts, "bt_diag": bt_diag,
    })
    return state

def extract_query_features(
    state, q_u, q_i1, q_i2, q_t, precomputed_base=None, precomputed_nb=None
):
    """Extract full antisymmetric features d, symmetric context features ctx,
    base features x0, and clean seen_override for any query set (q_u, q_i1, q_i2, q_t).
    """
    task = state["task"]
    a_fit = state["a_fit"]
    nu, ni = state["nu"], state["ni"]
    u_cnt, i_cnt = state["u_cnt"], state["i_cnt"]
    u_ptr, u_items = state["u_ptr"], state["u_items"]

    seen_override, zero_diff, zero_sum, zero_neq = compute_seen_and_zero_flags(
        q_u, q_i1, q_i2, u_ptr, u_items, i_cnt
    )

    # 1. Base features x0
    if precomputed_base is not None:
        x0 = precomputed_base.astype(np.float64)
    else:
        spec_lift = np.sum(state["u_vec"][q_u] * (state["i_vec"][q_i1] - state["i_vec"][q_i2]), axis=1)
        comm_sim = np.sum(state["u_unit_ext"][q_u] * (state["i_unit"][q_i1] - state["i_unit"][q_i2]), axis=1)
        f_pool = base.spectral_history_pooling(q_u, q_i1, q_i2, u_ptr, u_items, state["i_unit"])
        bpr_score = np.sum(state["p_bpr"][q_u] * (state["q_bpr"][q_i1] - state["q_bpr"][q_i2]), axis=1)

        if task == 2:
            f_pair = base.task2_pair_features(
                q_u, q_i1, q_i2, u_ptr, u_items, state["u_times"], state["u_rats"],
                state["i_ptr"], state["i_users"], state["i_times"], u_cnt, i_cnt
            )
            x0 = np.column_stack([f_pair[:, :17], spec_lift, comm_sim, f_pool, bpr_score]).astype(np.float64)
        elif task == 4:
            f_pair = base.task4_pair_features(
                q_u, q_i1, q_i2, q_t, u_ptr, u_items, state["u_times"], state["u_rats"],
                state["i_ptr"], state["i_users"], state["i_times"], u_cnt, i_cnt
            )
            x0 = np.column_stack([f_pair[:, :22], spec_lift, comm_sim, f_pool, bpr_score]).astype(np.float64)
        else:
            f_seq = base.task5_seq_features(
                q_u, q_i1, q_i2, u_ptr, u_items, state["u_rats"],
                state["i_ptr"], state["i_users"], state["i_pos"], u_cnt, i_cnt,
                state["item_rel_mean"], state["item_rel4_mean"],
                state["item_last1_rate"], state["item_last3_rate"], state["item_last5_rate"],
                state["u_end_era"]
            )
            spec_decay = np.sum(state["u_decay"][q_u] * (state["v_item_ext"][q_i1] - state["v_item_ext"][q_i2]), axis=1)
            spec_last1 = np.sum(state["u_last1"][q_u] * (state["v_item_ext"][q_i1] - state["v_item_ext"][q_i2]), axis=1)
            x0 = np.column_stack([
                f_seq[:, :23], spec_lift, comm_sim, f_pool, spec_decay, spec_last1, bpr_score
            ]).astype(np.float64)

    # 2. Rarity-weighted higher-order user neighborhood features (26 antisym, 26 sym)
    if precomputed_nb is not None:
        d_nb = precomputed_nb[0].astype(np.float64)
        s_nb = precomputed_nb[1].astype(np.float64)
    else:
        d_nb, s_nb = build_neighbor_features(a_fit, nu, ni, q_u, q_i1, q_i2, q_t, task)
        d_nb = d_nb.astype(np.float64)
        s_nb = s_nb.astype(np.float64)

    # 3. Task-specific new features (Sections 4.4, 4.5, 5)
    if task == 2:
        d_new, s_new = task2_stage_features(
            q_u, q_i1, q_i2, u_ptr, u_items, state["u_times"],
            state["i_ptr"], state["i_times"], i_cnt, state["all_times_sorted"], state["i_unit"]
        )
        d_full = np.column_stack([x0, d_nb, d_new, zero_diff])
        s_extra = s_new
    elif task == 4:
        d_new, s_new, u_ctx = task4_session_window_features(
            q_u, q_i1, q_i2, q_t, u_ptr, u_items, state["u_times"],
            state["i_ptr"], state["i_users"], state["i_times"],
            u_cnt, i_cnt, state["all_times_sorted"], state["i_unit"]
        )
        d_full = np.column_stack([x0, d_nb, d_new, zero_diff])
        s_extra = np.column_stack([s_new, u_ctx])
    else:
        d_new, s_new, u_ctx = task5_bt_features(
            q_u, q_i1, q_i2, u_ptr, u_items, i_cnt, state["era"], state["J"], reg=1.0
        )
        # Note: x0[:, 22] in Task 5 is already zero_diff = (c1==0)-(c2==0)
        d_full = np.column_stack([x0, d_nb, d_new])
        s_extra = np.column_stack([s_new, u_ctx])

    # 4. Symmetric context features:
    # log1p(u_cnt), log1p(c1)+log1p(c2), |log1p(c1)-log1p(c2)|, min(log1p(c1), log1p(c2)),
    # max(log1p(c1), log1p(c2)), (c1==0)+(c2==0), (c1==0)!=(c2==0), s = f(i1)+f(i2), |d|
    uc_ext = np.r_[u_cnt, 0]
    ic_ext = np.r_[i_cnt, 0]
    lc1 = np.log1p(ic_ext[q_i1])
    lc2 = np.log1p(ic_ext[q_i2])
    base_ctx = np.column_stack([
        np.log1p(uc_ext[q_u]),
        lc1 + lc2,
        np.abs(lc1 - lc2),
        np.minimum(lc1, lc2),
        np.maximum(lc1, lc2),
        zero_sum,
        zero_neq,
    ])
    ctx_full = np.column_stack([base_ctx, s_nb, s_extra, np.abs(d_full)]).astype(np.float64)
    return x0, d_full, ctx_full, seen_override

def expand(d, ctx, sign=1.0):
    return np.column_stack([sign * d, ctx])

def fit_antisymmetric_hgb(d_tr, ctx_tr, leaves=15, reg=10.0, lr=0.06, max_iter=250):
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_leaf_nodes=leaves,
        learning_rate=lr,
        l2_regularization=reg,
        min_samples_leaf=40,
        early_stopping=False,
        random_state=258,
    )
    x_dup = np.vstack([expand(d_tr, ctx_tr, 1.0), expand(d_tr, ctx_tr, -1.0)])
    y_dup = np.r_[np.ones(len(d_tr)), np.zeros(len(d_tr))]
    model.fit(x_dup, y_dup)
    return model

def score_antisymmetric_hgb(model, d, ctx):
    return 0.5 * (
        model.decision_function(expand(d, ctx, 1.0))
        - model.decision_function(expand(d, ctx, -1.0))
    )

def compute_bucket_breakdown(base_scores, cand_scores, i_cnt, q_i1, q_i2):
    """Report accuracy by min(c1, c2) frequency buckets {0, 1, 2-5, 6-20, >20},
    sample counts, and paired flip counts / standard errors.
    """
    ni = len(i_cnt)
    ic_ext = np.r_[i_cnt, 0]
    c1 = ic_ext[q_i1]
    c2 = ic_ext[q_i2]
    min_c = np.minimum(c1, c2)

    # The sampler stores the positive candidate first. Resolve exact ties using
    # a frozen random candidate orientation, rather than granting every tie a
    # correct prediction. This is equivalent to swapping those audit pairs.
    tie_positive_first = np.random.default_rng(258).integers(0, 2, len(base_scores)).astype(bool)
    b_ok = (base_scores > 0) | ((base_scores == 0) & tie_positive_first)
    c_ok = (cand_scores > 0) | ((cand_scores == 0) & tie_positive_first)

    def stats_for_mask(mask):
        cnt = int(np.sum(mask))
        if cnt == 0:
            return {
                "count": 0, "baseline_acc": None, "candidate_acc": None,
                "improvement": None, "gained": 0, "lost": 0, "paired_se": None
            }
        bm = b_ok[mask]
        cm = c_ok[mask]
        diff = cm.astype(np.float64) - bm.astype(np.float64)
        se = float(diff.std(ddof=1) / np.sqrt(cnt)) if cnt > 1 else 0.0
        return {
            "count": cnt,
            "baseline_acc": float(np.mean(bm)),
            "candidate_acc": float(np.mean(cm)),
            "improvement": float(np.mean(diff)),
            "gained": int(np.sum(cm & ~bm)),
            "lost": int(np.sum(bm & ~cm)),
            "paired_se": se,
        }

    buckets = {
        "0": min_c == 0,
        "1": min_c == 1,
        "2-5": (min_c >= 2) & (min_c <= 5),
        "6-20": (min_c >= 6) & (min_c <= 20),
        ">20": min_c > 20,
    }
    return {
        "overall": stats_for_mask(np.ones(len(min_c), dtype=bool)),
        "by_min_c_bucket": {k: stats_for_mask(m) for k, m in buckets.items()},
    }

@njit(cache=True)
def neighborhood(qu,qi1,qi2,qt,uptr,ui,ut,iptr,iu,ipos,it,norm,task):
    out=np.zeros((len(qu),26),np.float32)
    symmetric=np.zeros((len(qu),26),np.float32)
    nusers=len(uptr)-1
    stamps=np.full(nusers,-1,np.int64)
    ov=np.zeros((nusers,5),np.float32)
    lastpos=np.zeros(nusers,np.int32)
    nitems=len(iptr)-1
    for row in range(len(qu)):
        u=qu[row]; i1=qi1[row];i2=qi2[row]
        if u>=nusers: continue
        start=uptr[u];end=uptr[u+1];L=end-start
        if L==0: continue
        for pos in range(start,end):
            j=ui[pos]
            if j==i1 or j==i2: continue
            cj=iptr[j+1]-iptr[j]
            idf=np.log(1+nusers/(cj+5))
            rec=1./np.sqrt(end-pos)
            if task==4: rec=np.exp(-abs(float(ut[pos]-qt[row]))/(86400*365))
            for ptr in range(iptr[j],iptr[j+1]):
                v=iu[ptr]
                if v==u: continue
                if stamps[v]!=row:
                    stamps[v]=row;ov[v,:]=0;lastpos[v]=-1
                ov[v,0]+=1
                ov[v,1]+=idf
                ov[v,2]+=idf*rec
                ov[v,3]+=1./np.sqrt(cj+5)
                ov[v,4]+=rec
                if ipos[ptr]>lastpos[v]:lastpos[v]=ipos[ptr]
        fs=np.zeros((2,26),np.float64)
        for side in range(2):
            item=i1 if side==0 else i2
            if item>=nitems:continue
            count=iptr[item+1]-iptr[item]
            best=np.zeros(10,np.float64)
            nmatch=0
            for ptr in range(iptr[item],iptr[item+1]):
                v=iu[ptr]
                if stamps[v]!=row:continue
                nmatch+=1
                nv=uptr[v+1]-uptr[v]
                cosine=ov[v,0]/np.sqrt((L+3.)*(nv+3.))
                idfcos=ov[v,1]/np.sqrt((norm[u]+8.)*(norm[v]+8.))
                recent=ov[v,2]/np.sqrt((norm[u]+8.)*(norm[v]+8.))
                rare=ov[v,3]/np.sqrt((L+3.)*(nv+3.))
                for k,sim in enumerate((cosine,idfcos,recent,rare)):
                    fs[side,k*4]+=sim
                    fs[side,k*4+1]+=sim*sim
                    fs[side,k*4+2]+=sim*sim*sim
                    if sim>fs[side,k*4+3]:fs[side,k*4+3]=sim
                # Candidate-relative local sequence signal; time kernel for Task4.
                if task==5:
                    dist=ipos[ptr]-lastpos[v]
                    if dist>0:
                        fs[side,16]+=idfcos*idfcos/(dist+1)
                        if dist<=3:fs[side,17]+=idfcos*idfcos
                    elif dist>=-3:
                        fs[side,18]+=idfcos*idfcos
                elif task==4:
                    dt=abs(float(it[ptr]-qt[row]))/86400
                    fs[side,16]+=idfcos*idfcos*np.exp(-dt/7)
                    fs[side,17]+=idfcos*idfcos*np.exp(-dt/90)
                    fs[side,18]+=idfcos*idfcos*np.exp(-dt/365)
                else:
                    fs[side,16]+=idfcos*idfcos/(nv+5.)
                    fs[side,17]+=idfcos*idfcos*ov[v,0]
                    fs[side,18]+=idfcos*idfcos*ov[v,1]
                if idfcos>best[0]:
                    best[0]=idfcos;best.sort()
            fs[side,19]=np.sum(best)
            fs[side,20]=np.sum(best[5:])
            fs[side,21]=np.sum(best[7:])
            fs[side,22]=nmatch/(count+2.)
            fs[side,23]=np.sum(best**2)
            fs[side,24]=np.sum(best**3)
            fs[side,25]=nmatch
            # Log density lifts; max/top-neighbor evidence kept without popularity adjustment.
            for k in range(19):
                if k%4==3 and k<16:
                    fs[side,k]=np.log(fs[side,k]+.001)
                else:
                    fs[side,k]=np.log((fs[side,k]+.001)/(count+3.))
            fs[side,25]=np.log((nmatch+.1)/(count+3.))
        for k in range(26):
            out[row,k]=fs[0,k]-fs[1,k]
            symmetric[row,k]=fs[0,k]+fs[1,k]
    return out,symmetric

def build_neighbor_features(a,nu,ni,qu,qi1,qi2,qt,task):
    uc=np.bincount(a[:,0],minlength=nu);ic=np.bincount(a[:,1],minlength=ni)
    uorder=np.argsort(a[:,0],kind='stable') if task==5 else np.lexsort((a[:,2],a[:,0]))
    uptr=np.r_[0,uc.cumsum()];ui=a[uorder,1];ut=a[uorder,2]
    position=np.empty(len(a),np.int64)
    position[uorder]=np.arange(len(a))-np.repeat(uptr[:-1],uc)
    iorder=np.argsort(a[:,1],kind='stable'); iptr=np.r_[0,ic.cumsum()]
    norm=np.bincount(a[:,0],weights=np.log(1+nu/(ic[a[:,1]]+5)),minlength=nu)
    return neighborhood(qu,qi1,qi2,qt,uptr,ui,ut,iptr,a[iorder,0],position[iorder],a[iorder,2],norm,task)


def verify_task5_invariance(a_full):
    """On a 500-row validation slice, verify that:
    (a) randomly permuting the order of user blocks in fit_rows, and
    (b) randomly permuting user/item integer IDs
    leave the extracted features and predictions invariant (within 1e-5).
    """
    log('[Task 5 Invariance Check] Starting mandatory invariance verification on 500-row validation slice...')
    t0 = time.time()
    sub_mask = a_full[:, 0] < 3000
    a_sub = a_full[sub_mask].copy()
    uniq_u, u_inv = np.unique(a_sub[:, 0], return_inverse=True)
    uniq_i, i_inv = np.unique(a_sub[:, 1], return_inverse=True)
    a_sub[:, 0] = u_inv
    a_sub[:, 1] = i_inv
    nu, ni = (len(uniq_u), len(uniq_i))
    fit_rows, va_A, _ = make_interaction_splits_dual(a_sub, nu, ni, sequential=True, seed=258)
    va_slice = va_A[:500].copy()

    def run_pipeline_on_slice(fit_r, q_u, q_i1, q_i2):
        st = fit_graph_state(5, fit_r, nu, ni, compute_bt_diag=False, equivariant_mode=True)
        _, d_feat, ctx_feat, seen_ov = extract_query_features(st, q_u, q_i1, q_i2, np.zeros(len(q_u), dtype=np.int64))
        rms = np.sqrt(np.mean(d_feat[:250] ** 2, axis=0)) + 1e-08
        w = base.fit_pairwise_logistic(np.clip(d_feat[:250] / rms, -8.0, 8.0), l2=25.0)
        preds = np.clip(d_feat / rms, -8.0, 8.0) @ w + 100.0 * seen_ov
        return (d_feat, ctx_feat, preds)
    d0, c0, p0 = run_pipeline_on_slice(fit_rows, va_slice[:, 0], va_slice[:, 1], va_slice[:, 2])
    rng = np.random.default_rng(2026)
    u_order = np.argsort(fit_rows[:, 0], kind='mergesort')
    u_cnt = np.bincount(fit_rows[:, 0], minlength=nu)
    u_ptr = np.r_[0, u_cnt.cumsum()]
    u_perm_blocks = rng.permutation(nu)
    blocks = [fit_rows[u_order[u_ptr[u]:u_ptr[u + 1]]] for u in u_perm_blocks if u_ptr[u + 1] > u_ptr[u]]
    fit_block_perm = np.vstack(blocks)
    d_blk, c_blk, p_blk = run_pipeline_on_slice(fit_block_perm, va_slice[:, 0], va_slice[:, 1], va_slice[:, 2])
    perm_u = rng.permutation(nu)
    perm_i = rng.permutation(ni)
    fit_id_perm = fit_rows.copy()
    fit_id_perm[:, 0] = perm_u[fit_rows[:, 0]]
    fit_id_perm[:, 1] = perm_i[fit_rows[:, 1]]
    d_id, c_id, p_id = run_pipeline_on_slice(fit_id_perm, perm_u[va_slice[:, 0]], perm_i[va_slice[:, 1]], perm_i[va_slice[:, 2]])
    diff_block_feat = float(max(np.max(np.abs(d0 - d_blk)), np.max(np.abs(c0 - c_blk))))
    diff_block_pred = float(np.max(np.abs(p0 - p_blk)))
    diff_id_feat = float(max(np.max(np.abs(d0 - d_id)), np.max(np.abs(c0 - c_id))))
    diff_id_pred = float(np.max(np.abs(p0 - p_id)))
    assert diff_block_feat < 1e-05 and diff_block_pred < 1e-05, f'Block invariance failed: {diff_block_feat}, {diff_block_pred}'
    assert diff_id_feat < 1e-05 and diff_id_pred < 1e-05, f'ID invariance failed: {diff_id_feat}, {diff_id_pred}'
    res = {'n_validation_slice_rows': 500, 'n_subgraph_users': int(nu), 'n_subgraph_items': int(ni), 'user_block_permutation_max_feature_diff': diff_block_feat, 'user_block_permutation_max_prediction_diff': diff_block_pred, 'user_item_id_permutation_max_feature_diff': diff_id_feat, 'user_item_id_permutation_max_prediction_diff': diff_id_pred, 'passed_within_1e_5': True, 'elapsed_seconds': float(time.time() - t0)}
    log(f'[Task 5 Invariance Check] PASSED: {res}')
    return res


def fit_fixed_pair_combiner(d, context, config):
    rms = np.sqrt(np.mean(d * d, axis=0)) + 1e-8
    linear = base.fit_pairwise_logistic(np.clip(d / rms, -8, 8), l2=10.)
    hgb = fit_antisymmetric_hgb(d, context, leaves=config['leaves'], reg=10.)
    return rms, linear, hgb


def predict_pair_combiner(bundle, d, context, config):
    rms, linear, hgb = bundle
    z_linear = np.clip(d / rms, -8, 8) @ linear
    z_hgb = score_antisymmetric_hgb(hgb, d, context)
    return (1-config['blend'])*z_linear + config['blend']*z_hgb


def oriented_pair_accuracy(scores):
    tie_positive_first = np.random.default_rng(258).integers(0,2,len(scores)).astype(bool)
    return float(np.mean((scores > 0) | ((scores == 0) & tie_positive_first)))


def production_interaction_task(task, a, users, items, args):
    started = time.time()
    nu, ni = len(users), len(items)
    checks = verify_task5_invariance(a) if task == 5 else {}
    fit, va_a, va_b = make_interaction_splits_dual(a, nu, ni, sequential=task == 5)
    state = fit_graph_state(task, fit, nu, ni, equivariant_mode=task == 5)
    queries = (va_a, va_b)
    feature_sets = [extract_query_features(state, v[:,0], v[:,1], v[:,2], v[:,3]) for v in queries]
    x0, d, context, seen = feature_sets[1]
    idx = np.random.default_rng(258).permutation(len(d))
    tr, tune, audit = np.split(idx, [len(d)//2, 3*len(d)//4])
    config = {'leaves':15, 'blend':.5}
    # All three tasks are now locked. Task 5 chose 15/.5 on tune rows in the
    # completed execution; do not repeat that selection during reproduction.
    tuning = [{'selection':'frozen after tune-only validation','leaves':15,'blend':.5}] if task == 5 else []
    bundle = fit_fixed_pair_combiner(d[tr], context[tr], config)
    audits = {}
    for label, v, features in zip(('after_holdout','before_holdout'), queries, feature_sets):
        bx, dx, cx, sx = features
        # Train an identical frozen configuration for each candidate mechanism.
        b = bundle if label == 'before_holdout' else fit_fixed_pair_combiner(dx[tr], cx[tr], config)
        score = predict_pair_combiner(b, dx[audit], cx[audit], config) + 100*sx[audit]
        rms0 = np.sqrt(np.mean(bx[tr]**2, axis=0))+1e-8
        w0 = base.fit_pairwise_logistic(np.clip(bx[tr]/rms0, -8, 8))
        old_score = np.clip(bx[audit]/rms0, -8, 8)@w0+100*sx[audit]
        audits[label] = compute_bucket_breakdown(old_score, score, state['i_cnt'], v[audit,1], v[audit,2])
        audits[label]['exact_ties'] = {'baseline':int(np.sum(old_score==0)), 'candidate':int(np.sum(score==0))}
        # Simultaneous candidate swap must reverse the learned margin exactly.
        reverse = predict_pair_combiner(b, -dx[audit], cx[audit], config)-100*sx[audit]
        checks[label+'_swap_max_error'] = float(np.max(np.abs(score+reverse)))
        assert checks[label+'_swap_max_error'] < 1e-8
    full_bundle = fit_fixed_pair_combiner(d, context, config)
    del state, feature_sets, d, context, x0
    full_state = fit_graph_state(task, a, nu, ni, equivariant_mode=task == 5)
    tests = base.read_test(args.data_dir/f'task_{task}_test.csv')
    qu = np.array([users.get(r['user'],nu) for r in tests], dtype=np.int64)
    q1 = np.array([items.get(r['item1'],ni) for r in tests], dtype=np.int64)
    q2 = np.array([items.get(r['item2'],ni) for r in tests], dtype=np.int64)
    qt = np.array([int(r['timestamp']) for r in tests], dtype=np.int64) if task == 4 else np.zeros(len(tests),dtype=np.int64)
    _, dt, ct, st = extract_query_features(full_state,qu,q1,q2,qt)
    margins = predict_pair_combiner(full_bundle,dt,ct,config)+100*st
    # Query batching must not change normalization or any feature definition.
    _, dr, cr, sr = extract_query_features(full_state,qu[:3],q1[:3],q2[:3],qt[:3])
    short = predict_pair_combiner(full_bundle,dr,cr,config)+100*sr
    checks['single_batch_max_error'] = float(np.max(np.abs(short-margins[:3])))
    assert checks['single_batch_max_error'] < 1e-6
    _, swapped_d, swapped_c, swapped_s = extract_query_features(full_state,qu[:3],q2[:3],q1[:3],qt[:3])
    checks['swapped_query_feature_error'] = float(max(np.max(np.abs(dr+swapped_d)),np.max(np.abs(cr-swapped_c)),np.max(np.abs(sr+swapped_s))))
    assert checks['swapped_query_feature_error'] < 1e-5
    if not np.all(np.isfinite(margins)):
        raise ValueError('Invalid pairwise margins')
    predictions = [r['item1'] if m>=0 else r['item2'] for r,m in zip(tests,margins)]
    base.write_predictions(args.data_dir/f'task_{task}_test.csv',args.output_dir/f'task_{task}_predictions.csv',predictions)
    metrics = {'model':'symmetric_context_hgb_linear_graph','fixed_config':config,
               'audit':audits,'checks':checks,'test_rows':len(tests),'tune_only_selection':tuning,
               'seed':258,'test_exact_ties':int(np.sum(margins==0)),
               'elapsed_seconds':round(time.time()-started,2),
               'negative_sampling':'before-holdout pool for final fit; both pools audited',
               'zero_frequency_hard_override':False}
    base.log(f'Task {task} audit: { {k:v["overall"] for k,v in audits.items()} }')
    return metrics
