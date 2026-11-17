"""Fixed Task 6 terminal-event model; no experiment artifacts are runtime inputs.

Validation uses disjoint users at four historical cutoffs. The locked model
is trained before July 2017 and audited on July users never used for fitting.
Historical eligibility selects still-active users, so these backtest errors are
not estimates of the unseen full leaderboard population error.
"""
from __future__ import annotations
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
import assignment1 as base
SEED = 258
DAY = 86400.0
T_MIN_VALID = base.T_MIN_VALID
T_MAX_VALID = base.T_MAX_VALID
CUTOFF_SPECS = [("2016-01", 1451606400), ("2016-07", 1467331200),
                ("2017-01", 1483228800), ("2017-07", 1498867200)]
HGB_PARAMS = dict(loss="absolute_error", early_stopping=False, random_state=258,
                  max_leaf_nodes=15, l2_regularization=50.0, max_iter=300,
                  learning_rate=0.04, min_samples_leaf=100)


def extract_cutoff_delta_features(histories, item_ts, query_u, query_i, cutoff=T_MAX_VALID):
    n = len(query_u)
    valid_tp = np.zeros(n, dtype=bool)
    u_med = np.zeros(n, dtype=np.float64)
    u_q15 = np.zeros(n, dtype=np.float64)
    u_q25 = np.zeros(n, dtype=np.float64)
    u_rec2 = np.zeros(n, dtype=np.float64)
    u_rec3 = np.zeros(n, dtype=np.float64)
    u_rec5 = np.zeros(n, dtype=np.float64)
    u_last1 = np.zeros(n, dtype=np.float64)
    upper_arr = np.zeros(n, dtype=np.float64)
    lower2 = np.zeros(n, dtype=np.float64)
    na_arr = np.zeros(n, dtype=np.float64)
    ntot_arr = np.zeros(n, dtype=np.float64)
    rem_p15 = np.zeros(n, dtype=np.float64)
    rem_p25 = np.zeros(n, dtype=np.float64)
    rem_p30 = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        u = query_u[idx]
        it = query_i[idx]
        h = histories[u]
        tp = float(h[-1][1])
        valid_tp[idx] = T_MIN_VALID <= tp <= cutoff
        raw_t = [x[1] for x in h]
        clean_d = [float(raw_t[k + 1] - raw_t[k]) for k in range(len(raw_t) - 1) if T_MIN_VALID <= raw_t[k] <= cutoff and T_MIN_VALID <= raw_t[k + 1] <= cutoff]
        if not clean_d:
            clean_d = np.maximum(0.0, np.diff(raw_t)).tolist()
        u_med[idx] = np.median(clean_d)
        u_q15[idx] = np.percentile(clean_d, 15)
        u_q25[idx] = np.percentile(clean_d, 25)
        u_rec2[idx] = np.median(clean_d[-2:])
        u_rec3[idx] = np.median(clean_d[-3:])
        u_rec5[idx] = np.median(clean_d[-5:])
        u_last1[idx] = clean_d[-1]
        if valid_tp[idx]:
            upper_arr[idx] = max(0.0, float(cutoff) - tp)
            ts = item_ts[it] if it < len(item_ts) else []
            n_tot = len(ts)
            ntot_arr[idx] = n_tot
            if n_tot >= 1:
                lower2[idx] = max(0.0, float(ts[0] if n_tot < 20 else ts[int(0.02 * n_tot)]) - tp)
                pos = int(np.searchsorted(ts, tp))
                na = n_tot - pos
                na_arr[idx] = na
                if na >= 1:
                    rem = np.asarray(ts[pos:], dtype=np.float64) - tp
                    rem_p15[idx] = np.percentile(rem, 15)
                    rem_p25[idx] = np.percentile(rem, 25)
                    rem_p30[idx] = np.percentile(rem, 30)
    return (valid_tp, u_med, u_q15, u_q25, u_rec2, u_rec3, u_rec5, u_last1, upper_arr, lower2, na_arr, ntot_arr, rem_p15, rem_p25, rem_p30)

def posterior_median_grid(
    edges: np.ndarray,
    positive_mass: np.ndarray,
    log_item_lift: np.ndarray,
    log_post_survival: np.ndarray,
    zero_prob: float,
    zero_log_lift: float,
    zero_log_survival: float,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Discrete-grid conditional posterior median over [0, H].

    positive_mass[k] = P(edges[k] < D <= edges[k+1]), excluding D=0.
    log_item_lift and log_post_survival are evaluated at interval midpoints.
    """
    if edges[-1] <= 0:
        return 0.0
    nodes = np.r_[0.0, (edges[:-1] + edges[1:]) / 2.0]
    mass = np.r_[zero_prob, positive_mass]
    logm = np.full(len(mass), -np.inf, dtype=np.float64)
    present = mass > 0
    logm[present] = np.log(mass[present])
    if beta != 0.0:
        logm += beta * np.r_[zero_log_lift, log_item_lift]
    if gamma != 0.0:
        logm += gamma * np.r_[zero_log_survival, log_post_survival]
    good = np.isfinite(logm)
    if not good.any():
        raise ValueError("No posterior support: use a documented fallback")
    weights = np.zeros_like(logm)
    weights[good] = np.exp(logm[good] - np.max(logm[good]))
    median_idx = int(np.searchsorted(np.cumsum(weights), 0.5 * weights.sum()))
    median_idx = min(median_idx, len(nodes) - 1)
    return float(nodes[median_idx])

@njit(cache=True)
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865475))

@njit(cache=True)
def _quantile_sorted(sorted_arr: np.ndarray, q: float) -> float:
    n = len(sorted_arr)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_arr[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_arr[lo])
    frac = pos - lo
    return float(sorted_arr[lo] * (1.0 - frac) + sorted_arr[hi] * frac)

@njit(cache=True)
def _eval_posterior_quantiles_numba(
    edges: np.ndarray,
    pos_mass: np.ndarray,
    log_lift_unit: np.ndarray,
    log_surv: np.ndarray,
    zero_prob: float,
    zero_log_lift_unit: float,
    zero_log_surv: float,
    beta_i: float,
    gamma: float,
) -> tuple[float, float, float]:
    k_len = len(pos_mass)
    if edges[k_len] <= 0.0:
        return 0.0, 0.0, 0.0
    logm = np.full(k_len + 1, -1e300, dtype=np.float64)
    if zero_prob > 0.0:
        logm[0] = math.log(zero_prob) + beta_i * zero_log_lift_unit + gamma * zero_log_surv
    for k in range(k_len):
        if pos_mass[k] > 0.0:
            logm[k + 1] = math.log(pos_mass[k]) + beta_i * log_lift_unit[k] + gamma * log_surv[k]
    max_log = -1e300
    for k in range(k_len + 1):
        if logm[k] > max_log:
            max_log = logm[k]
    if max_log <= -1e290:
        return 0.0, 0.0, 0.0
    w_sum = 0.0
    weights = np.zeros(k_len + 1, dtype=np.float64)
    for k in range(k_len + 1):
        if logm[k] > -1e290:
            w = math.exp(logm[k] - max_log)
            weights[k] = w
            w_sum += w
    if w_sum <= 0.0:
        return 0.0, 0.0, 0.0
    q25_t = 0.25 * w_sum
    q50_t = 0.50 * w_sum
    q75_t = 0.75 * w_sum
    cs = 0.0
    q25, q50, q75 = -1.0, -1.0, -1.0
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
    return q25, q50, q75

@njit(cache=True)
def _activity_bucket(hlen: int) -> int:
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

def fit_activity_priors(histories: list[list[tuple[int, int]]], train_users: np.ndarray, cutoff: int) -> dict[str, np.ndarray]:
    """Estimate activity-group log-gap priors strictly from train_users at cutoff."""
    n_buckets = 6
    mu_lists = [[] for _ in range(n_buckets)]
    hi_lists = [[] for _ in range(n_buckets)]
    sig_lists = [[] for _ in range(n_buckets)]
    zero_num = np.zeros(n_buckets, dtype=np.float64)
    zero_den = np.zeros(n_buckets, dtype=np.float64)

    for u in train_users:
        h = histories[int(u)]
        if len(h) < 2:
            continue
        b = _activity_bucket(len(h))
        clean_t = [x[1] for x in h if T_MIN_VALID <= x[1] <= cutoff]
        if len(clean_t) < 2:
            clean_t = [x[1] for x in h]
        gaps = np.maximum(0.0, np.diff(np.asarray(clean_t, dtype=np.float64)))
        if len(gaps) == 0:
            continue
        zero_num[b] += float(np.sum(gaps <= 60.0))
        zero_den[b] += float(len(gaps))
        pos_g = gaps[gaps > 60.0]
        if len(pos_g) == 0:
            pos_g = np.maximum(61.0, gaps)
        z = np.log1p(pos_g)
        mu_lists[b].append(0.5 * float(np.median(z)) + 0.5 * float(np.percentile(z, 75)))
        hi_lists[b].append(float(np.percentile(z, 90)))
        if len(z) >= 2:
            sig_lists[b].append(float(np.std(z)))

    act_mu = np.full(n_buckets, 15.1, dtype=np.float64)
    act_hi = np.full(n_buckets, 16.2, dtype=np.float64)
    act_sig = np.full(n_buckets, 0.90, dtype=np.float64)
    act_pi0 = np.full(n_buckets, 0.0018, dtype=np.float64)

    for b in range(n_buckets):
        if mu_lists[b]:
            act_mu[b] = float(np.median(mu_lists[b]))
        if hi_lists[b]:
            act_hi[b] = float(np.median(hi_lists[b]))
        if sig_lists[b]:
            act_sig[b] = float(np.clip(np.median(sig_lists[b]), 0.65, 1.15))
        if zero_den[b] > 0:
            act_pi0[b] = float((zero_num[b] + 1.0) / (zero_den[b] + 500.0))

    return {
        "act_mu": act_mu,
        "act_hi": act_hi,
        "act_sig": act_sig,
        "act_pi0": act_pi0,
    }

@njit(cache=True)
def _compute_grid_and_rich_features_numba(
    query_u: np.ndarray,
    query_i: np.ndarray,
    u_raw_ptr: np.ndarray,
    u_raw_ts: np.ndarray,
    u_clean_ptr: np.ndarray,
    u_clean_ts: np.ndarray,
    i_ptr: np.ndarray,
    i_ts: np.ndarray,
    global_ts: np.ndarray,
    cutoff: int,
    act_mu: np.ndarray,
    act_hi: np.ndarray,
    act_sig: np.ndarray,
    act_pi0: np.ndarray,
    k_grid: int,
    kappa_main: float,
    beta_tuned: float,
    gamma_tuned: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    quantiles = np.array([0.0, 0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98, 1.0], dtype=np.float64)
    rec_ks = np.array([1, 2, 3, 5, 10, 20], dtype=np.int64)
    win_days = np.array([7.0, 30.0, 90.0, 365.0, 730.0], dtype=np.float64)

    rich_feats = np.zeros((n_q, 101), dtype=np.float64)
    grid_f_rat = np.zeros((n_q, k_grid), dtype=np.float64)
    grid_pos_mass = np.zeros((n_q, k_grid), dtype=np.float64)
    grid_log_surv = np.zeros((n_q, k_grid), dtype=np.float64)
    grid_meta = np.zeros((n_q, 5), dtype=np.float64)

    u_grid = np.linspace(0.0, 1.0, k_grid + 1)
    bw30 = int(30.0 * DAY)
    bw120 = int(120.0 * DAY)

    for idx in range(n_q):
        u = query_u[idx]
        it = query_i[idx]

        r_start, r_end = u_raw_ptr[u], u_raw_ptr[u + 1]
        c_start, c_end = u_clean_ptr[u], u_clean_ptr[u + 1]
        raw_len = r_end - r_start
        clean_len = c_end - c_start

        tp = float(u_raw_ts[r_end - 1])
        t_first = float(u_clean_ts[c_start]) if clean_len >= 1 else float(u_raw_ts[r_start])
        valid_tp = (tp >= T_MIN_VALID) and (tp <= cutoff)
        H = max(0.0, float(cutoff) - tp) if valid_tp else 0.0
        H_days = H / DAY

        if clean_len >= 2:
            n_gaps = clean_len - 1
            gaps_sec = np.empty(n_gaps, dtype=np.float64)
            for g_i in range(n_gaps):
                d_val = float(u_clean_ts[c_start + g_i + 1] - u_clean_ts[c_start + g_i])
                gaps_sec[g_i] = d_val if d_val > 0.0 else 0.0
        elif raw_len >= 2:
            n_gaps = raw_len - 1
            gaps_sec = np.empty(n_gaps, dtype=np.float64)
            for g_i in range(n_gaps):
                d_val = float(u_raw_ts[r_start + g_i + 1] - u_raw_ts[r_start + g_i])
                gaps_sec[g_i] = d_val if d_val > 0.0 else 0.0
        else:
            n_gaps = 1
            gaps_sec = np.zeros(1, dtype=np.float64)

        gaps_days = gaps_sec / DAY
        sorted_gaps_days = np.sort(gaps_days)

        # Basic user gap stats
        g_mean = float(np.mean(gaps_days))
        g_std = float(np.std(gaps_days))
        g_log1p_mean = 0.0
        lt1, lt7, lt30, n_zero60 = 0.0, 0.0, 0.0, 0.0
        pos_count = 0
        for g_i in range(n_gaps):
            gd = gaps_days[g_i]
            gs = gaps_sec[g_i]
            g_log1p_mean += math.log1p(gd)
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

        b_act = _activity_bucket(raw_len)
        pi_u = (n_zero60 + 5.0 * act_pi0[b_act]) / (float(n_gaps) + 5.0)

        col = 0
        rich_feats[idx, col] = float(raw_len); col += 1
        rich_feats[idx, col] = float(clean_len); col += 1
        rich_feats[idx, col] = H_days; col += 1
        rich_feats[idx, col] = max(0.0, (tp - t_first) / DAY); col += 1
        rich_feats[idx, col] = max(0.0, (float(cutoff) - t_first) / DAY); col += 1
        rich_feats[idx, col] = 1.0 if valid_tp else 0.0; col += 1
        rich_feats[idx, col] = g_mean; col += 1
        rich_feats[idx, col] = g_std; col += 1
        rich_feats[idx, col] = g_log1p_mean; col += 1
        rich_feats[idx, col] = lt1; col += 1
        rich_feats[idx, col] = lt7; col += 1
        rich_feats[idx, col] = lt30; col += 1
        rich_feats[idx, col] = pi_u; col += 1

        # 9 quantiles of user gaps in days
        for q_i in range(9):
            rich_feats[idx, col] = _quantile_sorted(sorted_gaps_days, quantiles[q_i])
            col += 1
        u_med_days = rich_feats[idx, col - 5]
        u_q75_days = rich_feats[idx, col - 4]

        # Recent k in [1, 2, 3, 5, 10, 20]: mean, median, std, mean(log1p) -> 24 features
        for rk_i in range(6):
            rk = rec_ks[rk_i]
            start_g = max(0, n_gaps - rk)
            sub_g = gaps_days[start_g:]
            sub_sorted = np.sort(sub_g)
            m_val = float(np.mean(sub_g))
            med_val = _quantile_sorted(sub_sorted, 0.5)
            std_val = float(np.std(sub_g))
            l_val = 0.0
            for s_i in range(len(sub_g)):
                l_val += math.log1p(sub_g[s_i])
            l_val /= float(len(sub_g))
            rich_feats[idx, col] = m_val; col += 1
            rich_feats[idx, col] = med_val; col += 1
            rich_feats[idx, col] = std_val; col += 1
            rich_feats[idx, col] = l_val; col += 1

        u_last1_days = rich_feats[idx, col - 23]
        u_rec3_days = rich_feats[idx, col - 15]
        u_rec5_days = rich_feats[idx, col - 11]

        # User activity counts in last [7, 30, 90, 365, 730] days -> 5 features
        for w_i in range(5):
            thresh = tp - win_days[w_i] * DAY
            cnt_w = 0
            for r_p in range(r_start, r_end):
                if u_raw_ts[r_p] >= thresh:
                    cnt_w += 1
            rich_feats[idx, col] = float(cnt_w)
            col += 1

        # 5 dimensionless user-to-horizon ratios
        denom_h = H_days + 1.0
        rich_feats[idx, col] = u_med_days / denom_h; col += 1
        rich_feats[idx, col] = u_q75_days / denom_h; col += 1
        rich_feats[idx, col] = u_rec3_days / denom_h; col += 1
        rich_feats[idx, col] = u_rec5_days / denom_h; col += 1
        rich_feats[idx, col] = u_last1_days / denom_h; col += 1

        # Item timestamp features -> 31 features
        if it < n_items:
            istart, iend = i_ptr[it], i_ptr[it + 1]
            ci = iend - istart
        else:
            istart, iend = 0, 0
            ci = 0

        rich_feats[idx, col] = float(ci); col += 1
        rich_feats[idx, col] = math.log1p(float(ci)); col += 1

        if ci > 0 and valid_tp:
            its_sub = i_ts[istart:iend]
            # 9 quantiles of (its - tp)/DAY
            for q_i in range(9):
                q_ts = _quantile_sorted(its_sub, quantiles[q_i])
                rich_feats[idx, col] = (q_ts - tp) / DAY
                col += 1
            pos_idx = int(np.searchsorted(its_sub, int(tp)))
            na = ci - pos_idx
            rich_feats[idx, col] = float(na) / float(ci); col += 1
            rich_feats[idx, col] = (float(na) + 1.0) / (float(ci) + 4.0); col += 1
            if na > 0:
                fut_sub = its_sub[pos_idx:]
                for q_i in range(9):
                    q_fut = _quantile_sorted(fut_sub, quantiles[q_i])
                    rich_feats[idx, col] = (q_fut - tp) / DAY
                    col += 1
            else:
                for _ in range(9):
                    rich_feats[idx, col] = np.nan
                    col += 1
            for w_i in range(5):
                w_sec = int(win_days[w_i] * DAY)
                c_fwd = int(np.searchsorted(its_sub, int(tp) + w_sec, side="right")) - pos_idx
                c_bwd = pos_idx - int(np.searchsorted(its_sub, int(tp) - w_sec, side="left"))
                rich_feats[idx, col] = float(c_fwd) / float(ci); col += 1
                rich_feats[idx, col] = float(c_bwd) / float(ci); col += 1
        else:
            for _ in range(30):
                rich_feats[idx, col] = np.nan
                col += 1

        # Positive waiting distribution in log1p(seconds) space
        if pos_count > 0:
            z_pos = np.empty(pos_count, dtype=np.float64)
            p_k = 0
            for g_i in range(n_gaps):
                if gaps_sec[g_i] > 60.0:
                    z_pos[p_k] = math.log1p(gaps_sec[g_i])
                    p_k += 1
        else:
            z_pos = np.empty(n_gaps, dtype=np.float64)
            for g_i in range(n_gaps):
                z_pos[g_i] = math.log1p(max(61.0, gaps_sec[g_i]))

        nz = len(z_pos)
        z_sorted = np.sort(z_pos)
        z_med = _quantile_sorted(z_sorted, 0.50)
        z_q75 = _quantile_sorted(z_sorted, 0.75)
        z_q90 = _quantile_sorted(z_sorted, 0.90)
        rec_n = min(5, nz)
        z_rec_sub = np.sort(z_pos[nz - rec_n :])
        z_rec = _quantile_sorted(z_rec_sub, 0.50)

        mu_all = (float(nz) * (0.5 * z_med + 0.5 * z_q75) + 3.0 * act_mu[b_act]) / (float(nz) + 3.0)
        mu_hi = (float(nz) * z_q90 + 3.0 * act_hi[b_act]) / (float(nz) + 3.0)
        mu_rec = (float(rec_n) * z_rec + 2.0 * mu_all) / (float(rec_n) + 2.0)
        z_std = float(np.std(z_pos)) if nz >= 2 else act_sig[b_act]
        sig = min(1.05, max(0.50, (0.5 * z_std * float(nz) + act_sig[b_act] * 6.0) / (float(nz) + 6.0)))

        if valid_tp and H > 0.0:
            log1p_H = math.log1p(H)
            edges = np.empty(k_grid + 1, dtype=np.float64)
            z_edges = np.empty(k_grid + 1, dtype=np.float64)
            cdf_edges = np.empty(k_grid + 1, dtype=np.float64)
            for k in range(k_grid + 1):
                ze = u_grid[k] * log1p_H
                z_edges[k] = ze
                edges[k] = math.expm1(ze)
                cdf_edges[k] = (
                    0.45 * _norm_cdf((ze - mu_all) / sig)
                    + 0.30 * _norm_cdf((ze - mu_rec) / sig)
                    + 0.25 * _norm_cdf((ze - mu_hi) / sig)
                )

            pos_mass = np.empty(k_grid, dtype=np.float64)
            log_surv = np.empty(k_grid, dtype=np.float64)
            f_rat_arr = np.ones(k_grid, dtype=np.float64)
            log_lift_k1 = np.zeros(k_grid, dtype=np.float64)
            log_lift_main = np.zeros(k_grid, dtype=np.float64)

            cdf_H = (
                0.45 * _norm_cdf((log1p_H - mu_all) / sig)
                + 0.30 * _norm_cdf((log1p_H - mu_rec) / sig)
                + 0.25 * _norm_cdf((log1p_H - mu_hi) / sig)
            )
            zero_surv = (1.0 - pi_u) * max(1e-12, 1.0 - cdf_H)
            zero_log_surv = math.log(zero_surv)

            # Zero-lag lift at t0
            zero_f_rat = 1.0
            if ci > 0:
                its_sub = i_ts[istart:iend]
                t0_int = int(tp)
                lo1_0 = max(T_MIN_VALID, t0_int - bw30)
                hi1_0 = min(cutoff, t0_int + bw30)
                lo2_0 = max(T_MIN_VALID, t0_int - bw120)
                hi2_0 = min(cutoff, t0_int + bw120)
                ic1_0 = float(np.searchsorted(its_sub, hi1_0, side="right") - np.searchsorted(its_sub, lo1_0, side="left"))
                gc1_0 = float(max(1, np.searchsorted(global_ts, hi1_0, side="right") - np.searchsorted(global_ts, lo1_0, side="left")))
                ic2_0 = float(np.searchsorted(its_sub, hi2_0, side="right") - np.searchsorted(its_sub, lo2_0, side="left"))
                gc2_0 = float(max(1, np.searchsorted(global_ts, hi2_0, side="right") - np.searchsorted(global_ts, lo2_0, side="left")))
                zero_f_rat = 0.55 * ((ic1_0 / float(ci)) / (gc1_0 / n_glob)) + 0.45 * ((ic2_0 / float(ci)) / (gc2_0 / n_glob))

            zero_lift_k1 = (float(ci) * zero_f_rat + 1.0) / (float(ci) + 1.0) if ci > 0 else 1.0
            zero_lift_main = (float(ci) * zero_f_rat + kappa_main) / (float(ci) + kappa_main) if ci > 0 else 1.0
            zero_log_lift_k1 = math.log(max(1e-8, zero_lift_k1))
            zero_log_lift_main = math.log(max(1e-8, zero_lift_main))

            for k in range(k_grid):
                dm = cdf_edges[k + 1] - cdf_edges[k]
                pm_k = (1.0 - pi_u) * (dm if dm > 1e-15 else 1e-15)
                pos_mass[k] = pm_k

                mid_d = 0.5 * (edges[k] + edges[k + 1])
                rem_d = max(0.0, H - mid_d)
                z_rem = math.log1p(rem_d)
                cdf_rem = (
                    0.45 * _norm_cdf((z_rem - mu_all) / sig)
                    + 0.30 * _norm_cdf((z_rem - mu_rec) / sig)
                    + 0.25 * _norm_cdf((z_rem - mu_hi) / sig)
                )
                s_k = (1.0 - pi_u) * max(1e-12, 1.0 - cdf_rem)
                log_surv[k] = math.log(s_k)

                if ci > 0:
                    its_sub = i_ts[istart:iend]
                    tn = int(tp + mid_d)
                    lo1 = max(T_MIN_VALID, tn - bw30)
                    hi1 = min(cutoff, tn + bw30)
                    lo2 = max(T_MIN_VALID, tn - bw120)
                    hi2 = min(cutoff, tn + bw120)
                    ic1 = float(np.searchsorted(its_sub, hi1, side="right") - np.searchsorted(its_sub, lo1, side="left"))
                    gc1 = float(max(1, np.searchsorted(global_ts, hi1, side="right") - np.searchsorted(global_ts, lo1, side="left")))
                    ic2 = float(np.searchsorted(its_sub, hi2, side="right") - np.searchsorted(its_sub, lo2, side="left"))
                    gc2 = float(max(1, np.searchsorted(global_ts, hi2, side="right") - np.searchsorted(global_ts, lo2, side="left")))
                    fr = 0.55 * ((ic1 / float(ci)) / (gc1 / n_glob)) + 0.45 * ((ic2 / float(ci)) / (gc2 / n_glob))
                    f_rat_arr[k] = fr
                    l_k1 = (float(ci) * fr + 1.0) / (float(ci) + 1.0)
                    l_main = (float(ci) * fr + kappa_main) / (float(ci) + kappa_main)
                    log_lift_k1[k] = math.log(max(1e-8, l_k1))
                    log_lift_main[k] = math.log(max(1e-8, l_main))

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

            _, med_b0_g1, _ = _eval_posterior_quantiles_numba(
                edges, pos_mass, log_lift_k1, log_surv, pi_u, zero_log_lift_k1, zero_log_surv, 0.0, 1.0
            )
            _, med_b05_g05, _ = _eval_posterior_quantiles_numba(
                edges, pos_mass, log_lift_k1, log_surv, pi_u, zero_log_lift_k1, zero_log_surv, 0.5 * att_k1, 0.5
            )
            q25_b1_g1, med_b1_g1, q75_b1_g1 = _eval_posterior_quantiles_numba(
                edges, pos_mass, log_lift_k1, log_surv, pi_u, zero_log_lift_k1, zero_log_surv, 1.0 * att_k1, 1.0
            )
            _, med_b1_g0, _ = _eval_posterior_quantiles_numba(
                edges, pos_mass, log_lift_k1, log_surv, pi_u, zero_log_lift_k1, zero_log_surv, 1.0 * att_k1, 0.0
            )
            q25_tun, med_tun, q75_tun = _eval_posterior_quantiles_numba(
                edges, pos_mass, log_lift_main, log_surv, pi_u, zero_log_lift_main, zero_log_surv, beta_tuned * att_main, gamma_tuned
            )

            rich_feats[idx, col] = med_b0_g1 / DAY; col += 1
            rich_feats[idx, col] = med_b05_g05 / DAY; col += 1
            rich_feats[idx, col] = med_b1_g1 / DAY; col += 1
            rich_feats[idx, col] = med_b1_g0 / DAY; col += 1
            rich_feats[idx, col] = q25_b1_g1 / DAY; col += 1
            rich_feats[idx, col] = q75_b1_g1 / DAY; col += 1
            rich_feats[idx, col] = (q75_b1_g1 - q25_b1_g1) / DAY; col += 1
            rich_feats[idx, col] = med_tun / DAY; col += 1
            rich_feats[idx, col] = q25_tun / DAY; col += 1
            rich_feats[idx, col] = q75_tun / DAY; col += 1
            rich_feats[idx, col] = (med_b1_g1 - med_b0_g1) / DAY; col += 1
            rich_feats[idx, col] = (med_b1_g1 - med_b1_g0) / DAY; col += 1
            rich_feats[idx, col] = (med_tun / DAY) / denom_h; col += 1
        else:
            fallback_s = u_med_days * DAY
            grid_meta[idx, 0] = 0.0
            grid_meta[idx, 1] = float(ci)
            grid_meta[idx, 2] = pi_u
            grid_meta[idx, 3] = 1.0
            grid_meta[idx, 4] = 0.0
            for _ in range(6):
                rich_feats[idx, col] = u_med_days
                col += 1
            rich_feats[idx, col] = 0.0; col += 1
            for _ in range(3):
                rich_feats[idx, col] = u_med_days
                col += 1
            rich_feats[idx, col] = 0.0; col += 1
            rich_feats[idx, col] = 0.0; col += 1
            rich_feats[idx, col] = 0.5; col += 1

    return rich_feats, grid_f_rat, grid_pos_mass, grid_log_surv, grid_meta

@njit(cache=True)
def eval_cached_posterior_grid(
    grid_f_rat: np.ndarray,
    grid_pos_mass: np.ndarray,
    grid_log_surv: np.ndarray,
    grid_meta: np.ndarray,
    fallback_med_sec: np.ndarray,
    beta: float,
    gamma: float,
    kappa: float,
) -> np.ndarray:
    """Fast evaluation of posterior_median_grid from cached per-query grid arrays."""
    n_q, k_grid = grid_pos_mass.shape
    u_grid = np.linspace(0.0, 1.0, k_grid + 1)
    preds = np.empty(n_q, dtype=np.float64)
    edges = np.empty(k_grid + 1, dtype=np.float64)
    log_lift = np.empty(k_grid, dtype=np.float64)

    for idx in range(n_q):
        H = grid_meta[idx, 0]
        if H <= 0.0:
            preds[idx] = fallback_med_sec[idx]
            continue
        ci = grid_meta[idx, 1]
        pi_u = grid_meta[idx, 2]
        zero_f_rat = grid_meta[idx, 3]
        zero_log_surv = grid_meta[idx, 4]

        log1p_H = math.log1p(H)
        for k in range(k_grid + 1):
            edges[k] = math.expm1(u_grid[k] * log1p_H)

        if ci > 0.0 and beta != 0.0:
            beta_i = beta * (ci / (ci + kappa))
            zero_lift = (ci * zero_f_rat + kappa) / (ci + kappa)
            zero_log_lift = math.log(max(1e-8, zero_lift))
            for k in range(k_grid):
                lk = (ci * grid_f_rat[idx, k] + kappa) / (ci + kappa)
                log_lift[k] = math.log(max(1e-8, lk))
        else:
            beta_i = 0.0
            zero_log_lift = 0.0
            for k in range(k_grid):
                log_lift[k] = 0.0

        _, q50, _ = _eval_posterior_quantiles_numba(
            edges,
            grid_pos_mass[idx],
            log_lift,
            grid_log_surv[idx],
            pi_u,
            zero_log_lift,
            zero_log_surv,
            beta_i,
            gamma,
        )
        preds[idx] = q50
    return preds

def run_numerical_verification_checks(
    histories: list[list[tuple[int, int]]],
    item_ts_list: list[list[int]],
    global_ts: np.ndarray,
    query_u: np.ndarray,
    query_i: np.ndarray,
    cutoff: int,
    priors: dict[str, np.ndarray],
) -> dict:
    """Execute mandatory verification checks from Section 6.2 - 6.4."""
    # 1. Exponential special case: F_u(d) = 1 - exp(-lambda * d), pi_u = 0, beta = 0, gamma = 1
    H_test = 120.0 * DAY
    k_grid = 96
    edges_1s = np.expm1(np.linspace(0.0, np.log1p(H_test), k_grid + 1))
    mid_1s = 0.5 * (edges_1s[:-1] + edges_1s[1:])
    idx_h2_1s = int(np.searchsorted(edges_1s, 0.5 * H_test)) - 1
    grid_res_1s = float(edges_1s[idx_h2_1s + 1] - edges_1s[idx_h2_1s])

    edges_1h = 3600.0 * np.expm1(np.linspace(0.0, np.log1p(H_test / 3600.0), k_grid + 1))
    mid_1h = 0.5 * (edges_1h[:-1] + edges_1h[1:])
    idx_h2_1h = int(np.searchsorted(edges_1h, 0.5 * H_test)) - 1
    grid_res_1h = float(edges_1h[idx_h2_1h + 1] - edges_1h[idx_h2_1h])

    exp_records = []
    for mean_days in [10.0, 30.0, 60.0, 120.0, 300.0, 1000.0]:
        lam = 1.0 / (mean_days * DAY)
        pm_1s = np.exp(-lam * edges_1s[:-1]) - np.exp(-lam * edges_1s[1:])
        ls_1s = -lam * (H_test - mid_1s)
        med_1s = posterior_median_grid(
            edges_1s, pm_1s, np.zeros(k_grid), ls_1s, 0.0, 0.0, -lam * H_test, beta=0.0, gamma=1.0
        )

        pm_1h = np.exp(-lam * edges_1h[:-1]) - np.exp(-lam * edges_1h[1:])
        ls_1h = -lam * (H_test - mid_1h)
        med_1h = posterior_median_grid(
            edges_1h, pm_1h, np.zeros(k_grid), ls_1h, 0.0, 0.0, -lam * H_test, beta=0.0, gamma=1.0
        )

        err_1s = abs(med_1s - 0.5 * H_test)
        err_1h = abs(med_1h - 0.5 * H_test)
        assert err_1s <= grid_res_1s * 1.05, f"Exponential check failed on 1s grid for mean_days={mean_days}"
        assert err_1h <= grid_res_1h * 1.05, f"Exponential check failed on 1h grid for mean_days={mean_days}"
        exp_records.append({
            "mean_wait_days": mean_days,
            "target_half_H_days": 0.5 * H_test / DAY,
            "median_log1p_sec_days": med_1s / DAY,
            "abs_err_log1p_sec_days": err_1s / DAY,
            "within_grid_res_log1p_sec": bool(err_1s <= grid_res_1s),
            "median_log1p_hour_days": med_1h / DAY,
            "abs_err_log1p_hour_days": err_1h / DAY,
            "within_grid_res_log1p_hour": bool(err_1h <= grid_res_1h),
        })

    # 2. Zero-atom special case: pi_u = 0.6, beta = 0, gamma = 0 -> returns 0.0
    pm_unif = np.full(k_grid, 0.4 / k_grid, dtype=np.float64)
    zero_med = posterior_median_grid(
        edges_1s, pm_unif, np.ones(k_grid), np.ones(k_grid), 0.6, 1.0, 1.0, beta=0.0, gamma=0.0
    )
    assert zero_med == 0.0, f"Zero-atom check failed: got {zero_med}"

    # 3. Convergence check between K = 64 and K = 128 across validation queries
    u_raw_ptr, u_raw_ts, u_clean_ptr, u_clean_ts = _pack_user_histories(histories, cutoff)
    i_ptr, i_ts = _pack_item_timestamps(item_ts_list)
    sub_u = query_u[:4000]
    sub_i = query_i[:4000]

    rf_64, _, _, _, _ = _compute_grid_and_rich_features_numba(
        sub_u, sub_i, u_raw_ptr, u_raw_ts, u_clean_ptr, u_clean_ts,
        i_ptr, i_ts, global_ts, cutoff,
        priors["act_mu"], priors["act_hi"], priors["act_sig"], priors["act_pi0"],
        64, 0.8, 1.2, 1.0,
    )
    rf_96, _, _, _, _ = _compute_grid_and_rich_features_numba(
        sub_u, sub_i, u_raw_ptr, u_raw_ts, u_clean_ptr, u_clean_ts,
        i_ptr, i_ts, global_ts, cutoff,
        priors["act_mu"], priors["act_hi"], priors["act_sig"], priors["act_pi0"],
        96, 0.8, 1.2, 1.0,
    )
    rf_128, _, _, _, _ = _compute_grid_and_rich_features_numba(
        sub_u, sub_i, u_raw_ptr, u_raw_ts, u_clean_ptr, u_clean_ts,
        i_ptr, i_ts, global_ts, cutoff,
        priors["act_mu"], priors["act_hi"], priors["act_sig"], priors["act_pi0"],
        128, 0.8, 1.2, 1.0,
    )
    # Column 90 is med_b1_g1 in days, Column 95 is med_tun in days
    diff_64_128_days = np.abs(rf_64[:, 95] - rf_128[:, 95])
    diff_96_128_days = np.abs(rf_96[:, 95] - rf_128[:, 95])
    H_sub_days = np.maximum(1.0, rf_96[:, 2])

    return {
        "exponential_special_case": {
            "passed": True,
            "H_days": H_test / DAY,
            "grid_res_at_H_half_log1p_sec_days": grid_res_1s / DAY,
            "grid_res_at_H_half_log1p_hour_days": grid_res_1h / DAY,
            "results_by_lambda": exp_records,
        },
        "zero_atom_special_case": {
            "passed": bool(zero_med == 0.0),
            "pi_u": 0.6,
            "beta": 0.0,
            "gamma": 0.0,
            "returned_median": zero_med,
        },
        "grid_convergence_K64_K96_K128": {
            "n_queries_checked": int(len(sub_u)),
            "K64_vs_K128_median_abs_diff_days": float(np.median(diff_64_128_days)),
            "K64_vs_K128_median_rel_diff_pct_of_H": float(np.median(diff_64_128_days / H_sub_days) * 100.0),
            "K96_vs_K128_median_abs_diff_days": float(np.median(diff_96_128_days)),
            "K96_vs_K128_median_rel_diff_pct_of_H": float(np.median(diff_96_128_days / H_sub_days) * 100.0),
        },
    }

def _pack_user_histories(
    histories: list[list[tuple[int, int]]], cutoff: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nu = len(histories)
    u_raw_ptr = np.zeros(nu + 1, dtype=np.int64)
    u_clean_ptr = np.zeros(nu + 1, dtype=np.int64)
    raw_list = []
    clean_list = []
    for u, h in enumerate(histories):
        for _, t in h:
            raw_list.append(t)
            if T_MIN_VALID <= t <= cutoff:
                clean_list.append(t)
        u_raw_ptr[u + 1] = len(raw_list)
        u_clean_ptr[u + 1] = len(clean_list)
    return (
        u_raw_ptr,
        np.asarray(raw_list, dtype=np.int64),
        u_clean_ptr,
        np.asarray(clean_list, dtype=np.int64),
    )

def _pack_item_timestamps(item_ts_list: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    ni = len(item_ts_list)
    i_ptr = np.zeros(ni + 1, dtype=np.int64)
    flat = []
    for i, ts in enumerate(item_ts_list):
        if ts:
            flat.extend(ts)
        i_ptr[i + 1] = len(flat)
    return i_ptr, np.asarray(flat, dtype=np.int64)

def build_clean_structures(
    histories: list[list[tuple[int, int]]], ni: int, cutoff: int
) -> tuple[list[list[int]], np.ndarray]:
    """Build sorted item_ts and global_ts strictly from events <= cutoff."""
    item_ts = [[] for _ in range(ni)]
    global_ts = []
    for h in histories:
        for i, t in h:
            if T_MIN_VALID <= t <= cutoff:
                item_ts[i].append(t)
                global_ts.append(t)
    for ts in item_ts:
        if ts:
            ts.sort()
    global_ts_arr = np.sort(np.asarray(global_ts, dtype=np.int64))
    return item_ts, global_ts_arr

def make_terminal_examples(
    allhist: list[list[tuple[int, int]]],
    ni: int,
    cutoff: int,
    user_group: np.ndarray,
    user_mask: np.ndarray | None = None,
) -> dict:
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
    eligible = np.array(
        [len(h_le[u]) >= 3 and allhist[u][-1][1] > cutoff for u in range(nu)],
        dtype=bool,
    )
    if user_mask is not None:
        selected_mask = eligible & user_mask
    else:
        selected_mask = eligible

    qu = np.where(selected_mask)[0].astype(np.int64)
    qi = np.array([h_le[u][-1][0] for u in qu], dtype=np.int64)
    truth_sec = np.array(
        [max(0.0, float(h_le[u][-1][1] - h_le[u][-2][1])) for u in qu],
        dtype=np.float64,
    )
    groups_q = user_group[qu]

    # Remove held-out target event for ALL eligible users at this cutoff
    obs_histories = [h_le[u][:-1] if eligible[u] else h_le[u] for u in range(nu)]
    item_ts, global_ts = build_clean_structures(obs_histories, ni, cutoff)

    # Cutoff is an explicit feature argument: no mutable module globals.
    vf_tuple = extract_cutoff_delta_features(obs_histories, item_ts, qu, qi, cutoff)

    return {
        "cutoff": cutoff,
        "obs_histories": obs_histories,
        "item_ts": item_ts,
        "global_ts": global_ts,
        "query_u": qu,
        "query_i": qi,
        "truth_sec": truth_sec,
        "truth_days": truth_sec / DAY,
        "groups": groups_q,
        "vf_tuple": vf_tuple,
        "vf_array": np.asarray(vf_tuple, dtype=np.float64),
    }

def slice_vf_tuple(vf_tuple: tuple, mask_or_idx: np.ndarray) -> tuple:
    return tuple(arr[mask_or_idx] for arr in vf_tuple)

def concat_vf_tuples(vf_list: list[tuple]) -> tuple:
    return tuple(np.concatenate([vf[k] for vf in vf_list], axis=0) for k in range(len(vf_list[0])))

def build_full_feature_matrix(
    rich_feats: np.ndarray,
    vf_tuple: tuple,
    pred_term_nm_sec: np.ndarray,
    post_med_tuned_sec: np.ndarray,
) -> np.ndarray:
    """Combine rich soft features, scaled base features, terminal NM, and posterior median."""
    valid_tp, u_med, _, _, _, _, _, _, upper_arr, low, na, nt, rem_p15, rem_p25, rem_p30 = vf_tuple
    vf_scaled = np.column_stack([
        valid_tp.astype(np.float64),
        u_med / DAY,
        vf_tuple[2] / DAY,
        vf_tuple[3] / DAY,
        vf_tuple[4] / DAY,
        vf_tuple[5] / DAY,
        vf_tuple[6] / DAY,
        vf_tuple[7] / DAY,
        upper_arr / DAY,
        low / DAY,
        na,
        nt,
        rem_p15 / DAY,
        rem_p25 / DAY,
        rem_p30 / DAY,
    ])
    conf = (na / (na + 3.0)) * ((na / np.maximum(1.0, nt)) ** 0.5)
    it_est_days = (0.35 * rem_p15 + 0.35 * rem_p25 + 0.30 * rem_p30) / DAY
    term_nm_days = pred_term_nm_sec / DAY
    post_tun_days = post_med_tuned_sec / DAY
    H_days = np.maximum(1.0, upper_arr / DAY)

    extra_cols = np.column_stack([
        term_nm_days,
        post_tun_days,
        0.5 * (term_nm_days + post_tun_days),
        term_nm_days - post_tun_days,
        term_nm_days / H_days,
        post_tun_days / H_days,
        it_est_days,
        conf,
    ])
    return np.column_stack([rich_feats, vf_scaled, extra_cols])

def postprocess_terminal_predictions(
    raw_pred_sec: np.ndarray,
    vf_tuple: tuple,
) -> np.ndarray:
    """Clip to [0, H] only when t0 in [T_MIN_VALID, cutoff] is valid; fallback to user median otherwise."""
    valid_tp = vf_tuple[0]
    u_med_sec = vf_tuple[1]
    H_sec = vf_tuple[8]
    clipped = np.clip(raw_pred_sec, 0.0, H_sec)
    return np.where(valid_tp, clipped, np.maximum(0.0, u_med_sec))


def fixed_features(ds, prior_users):
    """Identical rich-feature definitions at every training/test cutoff.

    The rich bank includes a fixed beta=1.2/kappa=.8 posterior. A separate
    selected posterior uses beta=1/kappa=.5. Keeping these definitions separate
    fixes the research prototype's train/test semantic mismatch.
    """
    priors = fit_activity_priors(ds['obs_histories'], prior_users, ds['cutoff'])
    raw_ptr, raw_ts, clean_ptr, clean_ts = _pack_user_histories(ds['obs_histories'], ds['cutoff'])
    ip, it = _pack_item_timestamps(ds['item_ts'])
    rich, fr, pm, ls, meta = _compute_grid_and_rich_features_numba(
        ds['query_u'], ds['query_i'], raw_ptr, raw_ts, clean_ptr, clean_ts,
        ip, it, ds['global_ts'], ds['cutoff'],
        priors['act_mu'], priors['act_hi'], priors['act_sig'], priors['act_pi0'],
        96, .8, 1.2, 1.0)
    post = postprocess_terminal_predictions(
        eval_cached_posterior_grid(fr, pm, ls, meta, ds['vf_tuple'][1], 1., 1., .5),
        ds['vf_tuple'])
    return rich, post


def fit_terminal_params(datasets, params, group=None):
    vfs, ys = [], []
    for ds in datasets:
        mask = np.ones(len(ds['query_u']), dtype=bool) if group is None else ds['groups'] == group
        vfs.append(slice_vf_tuple(ds['vf_tuple'], mask))
        ys.append(ds['truth_sec'][mask])
    vf = concat_vf_tuples(vfs)
    y = np.concatenate(ys)
    fit = minimize(lambda p: np.mean(np.abs(base.apply_delta_params(vf, p)-y))/DAY,
                   params, method='Nelder-Mead', options={'maxiter': 1200})
    if not np.all(np.isfinite(fit.x)):
        raise ValueError('Task 6 fitted non-finite parameters')
    return fit.x


def assemble_terminal(ds, params):
    # The earliest observed review is only an input to one component. The
    # final ensemble has no publication-date lower bound; 0 <= delta <= H.
    term = postprocess_terminal_predictions(base.apply_delta_params(ds['vf_tuple'], params), ds['vf_tuple'])
    x = build_full_feature_matrix(ds['rich'], ds['vf_tuple'], term, ds['post'])
    anchor = .65*term + .35*ds['post']
    return x, anchor, term


def predict_terminal(ds, params, model):
    x, anchor, term = assemble_terminal(ds, params)
    residual = postprocess_terminal_predictions(anchor + model.predict(x)*DAY, ds['vf_tuple'])
    ensemble = postprocess_terminal_predictions(.8*residual+.05*term+.15*ds['post'], ds['vf_tuple'])
    return ensemble, term, residual


def production_delta_task(a, users, items, args):
    """Validate the locked terminal model, refit, and write Task 6 predictions."""
    started = time.time()
    nu, ni = len(users), len(items)
    histories = [[] for _ in range(nu)]
    for u, i, t, _ in a:
        histories[int(u)].append((int(i), int(t)))
    if any(len(h) < 3 for h in histories):
        raise ValueError('Task 6 needs at least three historical events per user')
    if any(np.any(np.diff([t for _, t in h]) < 0) for h in histories):
        raise ValueError('Task 6 histories are not chronological')
    perm = np.random.default_rng(SEED).permutation(nu)
    groups = np.zeros(nu, dtype=np.int64)
    groups[perm[int(.6*nu):int(.8*nu)]] = 1
    groups[perm[int(.8*nu):]] = 2
    train_users = np.where(groups == 0)[0]
    all_users = np.arange(nu, dtype=np.int64)

    # Recreate the old local-training component for a paired baseline. Its
    # audit label is not used for selecting the production terminal model.
    penult = [h[:-1] for h in histories]
    pit, _ = build_clean_structures(penult, ni, T_MAX_VALID)
    pi = np.array([h[-1][0] for h in histories], dtype=np.int64)
    py = np.array([h[-1][1]-h[-2][1] for h in histories], dtype=np.float64)
    pf = extract_cutoff_delta_features(penult, pit, all_users, pi, T_MAX_VALID)
    init = np.array([.13,.05,.08,.05,.12,.04,.05,.50,.65,.35,.26,.45,.35,.70])
    pf_train = slice_vf_tuple(pf, train_users)
    legacy = minimize(lambda p: np.mean(np.abs(base.apply_delta_params(pf_train,p)-py[train_users]))/DAY,
                      init, method='Nelder-Mead', options={'maxiter': 800}).x
    del penult, pit

    datasets = []
    for name, cutoff in CUTOFF_SPECS:
        ds = make_terminal_examples(histories, ni, cutoff, groups)
        ds['name'] = name
        ds['rich'], ds['post'] = fixed_features(ds, train_users)
        datasets.append(ds)
        base.log(f'Task 6 terminal examples {name}: {len(ds["query_u"]):,}')
    early = datasets[:3]
    init_terminal = np.array([1.20,-.15,.25,.35,.80,.05,.40,.86,-.80,.42,.95,-.50,1.10,.90])
    early_params = fit_terminal_params(early, init_terminal, group=0)
    xx, yy = [], []
    for ds in early:
        x, anchor, _ = assemble_terminal(ds, early_params)
        mask = ds['groups'] == 0
        xx.append(x[mask]); yy.append((ds['truth_sec'][mask]-anchor[mask])/DAY)
    model = HistGradientBoostingRegressor(**HGB_PARAMS).fit(np.vstack(xx),np.concatenate(yy))
    del xx, yy

    # Frozen configuration, strictly disjoint users; July also has a later
    # cutoff than all model-fitting examples. Report each cohort separately.
    audit_metrics = {}
    for ds in datasets:
        pred, term, residual = predict_terminal(ds, early_params, model)
        old = postprocess_terminal_predictions(base.apply_delta_params(ds['vf_tuple'], legacy), ds['vf_tuple'])
        mask = ds['groups'] == 2
        y = ds['truth_sec'][mask]
        audit_metrics[ds['name']] = {
            'users': int(mask.sum()),
            'legacy_mae_seconds': float(np.mean(np.abs(old[mask]-y))),
            'terminal_nm_mae_seconds': float(np.mean(np.abs(term[mask]-y))),
            'posterior_mae_seconds': float(np.mean(np.abs(ds['post'][mask]-y))),
            'residual_hgb_mae_seconds': float(np.mean(np.abs(residual[mask]-y))),
            'ensemble_mae_seconds': float(np.mean(np.abs(pred[mask]-y))),
        }
    check_ds = datasets[2]
    priors = fit_activity_priors(check_ds['obs_histories'], train_users, check_ds['cutoff'])
    checks = run_numerical_verification_checks(
        check_ds['obs_histories'], check_ds['item_ts'], check_ds['global_ts'],
        check_ds['query_u'],check_ds['query_i'],check_ds['cutoff'],priors)

    # Final refit: the model choices are locked. Use every terminal pseudo-label
    # and rebuild the activity priors from all visible histories consistently.
    full_params = fit_terminal_params(datasets, early_params)
    xx, yy = [], []
    for ds in datasets:
        ds['rich'], ds['post'] = fixed_features(ds, all_users)
        x, anchor, _ = assemble_terminal(ds, full_params)
        xx.append(x); yy.append((ds['truth_sec']-anchor)/DAY)
    model = HistGradientBoostingRegressor(**HGB_PARAMS).fit(np.vstack(xx), np.concatenate(yy))
    del xx, yy
    test = base.read_test(args.data_dir / 'task_6_test.csv')
    qu = np.array([users[r['user']] for r in test], dtype=np.int64)
    qi = np.array([items.get(r['item'],ni) for r in test], dtype=np.int64)
    item_ts, global_ts = build_clean_structures(histories, ni, T_MAX_VALID)
    test_ds = dict(cutoff=T_MAX_VALID,obs_histories=histories,item_ts=item_ts,
                   global_ts=global_ts,query_u=qu,query_i=qi,
                   vf_tuple=extract_cutoff_delta_features(histories,item_ts,qu,qi,T_MAX_VALID))
    test_ds['rich'], test_ds['post'] = fixed_features(test_ds, all_users)
    predictions, _, _ = predict_terminal(test_ds, full_params, model)
    if not np.all(np.isfinite(predictions)) or np.any(predictions < 0):
        raise ValueError('Task 6 generated invalid predictions')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    base.write_predictions(args.data_dir/'task_6_test.csv',args.output_dir/'task_6_predictions.csv',predictions)
    valid = test_ds['vf_tuple'][0]
    metrics = {
        'validation_protocol': 'fixed user groups; terminal events at historical cutoffs; model fit before July 2017',
        'validation_population_caveat': 'historical eligibility selects users active after each cutoff; MAE is not a full-test estimate',
        'cross_cutoff_audit': audit_metrics,
        'numerical_checks': checks,
        'fixed_config': {'grid_K':96,'posterior_beta':1.,'posterior_gamma':1.,'posterior_kappa':.5,
                         'rich_beta':1.2,'rich_gamma':1.,'rich_kappa':.8,
                         'hgb':HGB_PARAMS,'weights':{'residual_hgb':.8,'terminal_nm':.05,'posterior':.15}},
        'prediction_count':len(predictions),'invalid_time_fallback_count':int((~valid).sum()),
        'test_prediction_quantiles_days':np.quantile(predictions/DAY,[.1,.25,.5,.75,.9]).tolist(),
        'elapsed_seconds':float(time.time()-started),
    }
    base.log(f'Task 6 July user-isolated terminal audit: {audit_metrics["2017-07"]}')
    return metrics
