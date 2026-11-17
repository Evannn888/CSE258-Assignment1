"""Fixed-configuration rating pipeline for Tasks 1 and 3.

Rating losses condition on r > 0; zero-rating rows remain implicit interactions.
This is a deliberate evaluation-target assumption, not proof about test labels.
All validation rows are removed from every base model and implicit graph.
No experimental files, cached arrays, leaderboard labels or external data are used.
SVD++ alpha is the gradient-normalization exponent; alpha=0 keeps implicit factors.
"""
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import assignment1 as base

SEED_PRIMARY = 258


RATING_CONFIGS = {1: {'best_stage_name': 'stage5_plus_conf_gating', 'best_l2': 40.0, 'best_eta': 1.0, 'best_hgb_cfg': {'max_leaf_nodes': 15, 'min_samples_leaf': 200, 'l2_regularization': 100.0, 'max_iter': 150, 'learning_rate': 0.03}, 'best_epochs': {'mf32': 28, 'mf64': 23, 'svdpp32': 3, 'svdpp64': 28, 'alpha32': 0.0, 'alpha64': 0.0}}, 3: {'best_stage_name': 'stage5_plus_conf_gating', 'best_l2': 3500.0, 'best_eta': 1.0, 'best_hgb_cfg': {'max_leaf_nodes': 15, 'min_samples_leaf': 200, 'l2_regularization': 100.0, 'max_iter': 150, 'learning_rate': 0.03}, 'best_epochs': {'mf32': 28, 'mf64': 11, 'svdpp32': 26, 'svdpp64': 28, 'alpha32': 0.0, 'alpha64': 0.0}}}


@njit(cache=True)
def mf_pos_epoch(a_rated, order, p, q, bu, bi, mu, lr, reg_b, reg_pq):
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


class ExplicitPosMF:
    def __init__(self, nu, ni, k, mu, bu_init=None, bi_init=None, seed=SEED_PRIMARY):
        rng = np.random.default_rng(seed)
        self.nu = nu
        self.ni = ni
        self.k = k
        self.mu = float(mu)
        self.p = rng.normal(0.0, 0.02, (nu, k)).astype(np.float32)
        self.q = rng.normal(0.0, 0.02, (ni, k)).astype(np.float32)
        self.bu = bu_init.astype(np.float32).copy() if bu_init is not None else np.zeros(nu, dtype=np.float32)
        self.bi = bi_init.astype(np.float32).copy() if bi_init is not None else np.zeros(ni, dtype=np.float32)

    def predict_components(self, q_u, q_i):
        u_valid = q_u < self.nu
        i_valid = q_i < self.ni
        both_valid = u_valid & i_valid
        u_safe = np.where(u_valid, q_u, 0)
        i_safe = np.where(i_valid, q_i, 0)
        b_u = np.where(u_valid, self.bu[u_safe], 0.0)
        b_i = np.where(i_valid, self.bi[i_safe], 0.0)
        dot = np.where(both_valid, np.sum(self.p[u_safe] * self.q[i_safe], axis=1), 0.0)
        pred = np.clip(self.mu + b_u + b_i + dot, 1.0, 5.0)
        return pred.astype(np.float64), dot.astype(np.float64)

    def predict(self, q_u, q_i):
        return self.predict_components(q_u, q_i)[0]


def train_explicit_pos_mf(a_rated, nu, ni, k=32, lr0=0.012, decay=0.97, reg_b=0.02, reg_pq=0.055,
                          max_epochs=28, bu_init=None, bi_init=None, tune_pos=None, seed=SEED_PRIMARY):
    mu = float(a_rated[:, 3].mean())
    model = ExplicitPosMF(nu, ni, k, mu, bu_init=bu_init, bi_init=bi_init, seed=seed)
    rng = np.random.default_rng(seed)
    best_mse = float('inf')
    best_epoch = max_epochs
    stale = 0
    best_state = None

    for ep in range(1, max_epochs + 1):
        lr = lr0 * (decay ** (ep - 1))
        mf_pos_epoch(a_rated, rng.permutation(len(a_rated)), model.p, model.q,
                     model.bu, model.bi, model.mu, lr, reg_b, reg_pq)
        if tune_pos is not None:
            pred = model.predict(tune_pos[:, 0], tune_pos[:, 1])
            mse = float(np.mean((pred - tune_pos[:, 3]) ** 2))
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
        best_mse = float(np.mean((pred - tune_pos[:, 3]) ** 2))
    return model, best_epoch, best_mse


@njit(cache=True)
def svdpp_user_block_epoch(
    u_order, u_pos_ptr, u_pos_items, u_pos_ratings,
    u_imp_ptr, u_imp_items,
    p, q, y, bu, bi, mu,
    lr_b, lr, reg_b, reg_p, reg_q, reg_y,
    ep_shrink_p, ep_shrink_q, ep_shrink_y, alpha
):
    """Exact User-Block CSR Mini-Batch SVD++ in Numba.
    r_hat_{ui} = mu + b_u + b_i + q_i^T (p_u + h_u),  h_u = |N(u)|^{-1/2} sum_{j in N(u)} y_j
    Accumulates G_u = (1 / Z_u) sum_{i in R+(u)} e_{ui} q_i in float64,
    updates p_u, q_i, y_j AFTER computing all block errors,
    and applies explicit per-epoch L2 shrinkage on p, q, y.
    """
    k = p.shape[1]
    hu = np.zeros(k, dtype=np.float64)
    pu_hu = np.zeros(k, dtype=np.float64)
    Gu = np.zeros(k, dtype=np.float64)

    for idx_u in range(len(u_order)):
        u = u_order[idx_u]
        rs, re = u_pos_ptr[u], u_pos_ptr[u + 1]
        n_pos = re - rs
        if n_pos == 0:
            continue
        ims, ime = u_imp_ptr[u], u_imp_ptr[u + 1]
        n_imp = ime - ims
        inv_sqrt_imp = 1.0 / np.sqrt(float(n_imp)) if n_imp > 0 else 0.0

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

        Z_u = (float(n_pos) ** alpha) if alpha > 0.0 else 1.0
        W_u = float(n_pos) / Z_u
        errs = np.empty(n_pos, dtype=np.float64)
        err_sum = 0.0

        # 1. Compute all errors in user block and accumulate Gu in float64
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

        # 2. Update biases, p_u, q_i, and y_j AFTER computing all block errors
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

    # 3. Explicit per-epoch L2 shrinkage so shared item factors q_i, y_j are not over-penalized by frequency
    if ep_shrink_p > 0.0:
        p *= np.float32(1.0 - lr * ep_shrink_p)
    if ep_shrink_q > 0.0:
        q *= np.float32(1.0 - lr * ep_shrink_q)
    if ep_shrink_y > 0.0:
        y *= np.float32(1.0 - lr * ep_shrink_y)


@njit(cache=True)
def svdpp_predict_components(q_u, q_i, u_imp_ptr, u_imp_items, p, q, y, bu, bi, mu):
    """Returns (clipped_pred, dot_pu_qi, dot_hu_qi)."""
    nu, k = p.shape
    ni = q.shape[0]
    h_all = np.zeros((nu, k), dtype=np.float32)
    for u in range(nu):
        ims, ime = u_imp_ptr[u], u_imp_ptr[u + 1]
        n_imp = ime - ims
        if n_imp > 0:
            inv_s = np.float32(1.0 / np.sqrt(float(n_imp)))
            for p_idx in range(ims, ime):
                j = u_imp_items[p_idx]
                for d in range(k):
                    h_all[u, d] += y[j, d]
            for d in range(k):
                h_all[u, d] *= inv_s

    n = len(q_u)
    pred = np.empty(n, dtype=np.float64)
    dot_p = np.zeros(n, dtype=np.float64)
    dot_h = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        u, i = q_u[idx], q_i[idx]
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
    return pred, dot_p, dot_h


class UserBlockSVDpp:
    def __init__(self, nu, ni, k, mu, bu_init, bi_init, u_imp_ptr, u_imp_items, seed=SEED_PRIMARY):
        rng = np.random.default_rng(seed)
        self.nu = nu
        self.ni = ni
        self.k = k
        self.mu = float(mu)
        self.p = rng.normal(0.0, 0.015, (nu, k)).astype(np.float32)
        self.q = rng.normal(0.0, 0.015, (ni, k)).astype(np.float32)
        self.y = np.zeros((ni, k), dtype=np.float32)
        self.bu = bu_init.astype(np.float32).copy()
        self.bi = bi_init.astype(np.float32).copy()
        self.u_imp_ptr = u_imp_ptr
        self.u_imp_items = u_imp_items

    def predict_components(self, q_u, q_i):
        return svdpp_predict_components(
            q_u, q_i, self.u_imp_ptr, self.u_imp_items,
            self.p, self.q, self.y, self.bu, self.bi, self.mu
        )

    def predict(self, q_u, q_i):
        return self.predict_components(q_u, q_i)[0]


def train_user_block_svdpp(
    u_pos_ptr, u_pos_items, u_pos_ratings,
    u_imp_ptr, u_imp_items,
    nu, ni, mu0, bu0, bi0,
    k=32, alpha=0.0, lr_b0=0.04, lr0=0.012, decay=0.95,
    reg_b=0.05, reg_p=0.06, reg_q=0.06, reg_y=0.05,
    ep_shrink=0.003, max_epochs=25, tune_pos=None, seed=SEED_PRIMARY
):
    model = UserBlockSVDpp(nu, ni, k, mu0, bu0, bi0, u_imp_ptr, u_imp_items, seed=seed)
    rng = np.random.default_rng(seed)
    best_mse = float('inf')
    best_epoch = max_epochs
    stale = 0
    best_state = None

    for ep in range(1, max_epochs + 1):
        d_factor = decay ** (ep - 1)
        svdpp_user_block_epoch(
            rng.permutation(nu).astype(np.int32),
            u_pos_ptr, u_pos_items, u_pos_ratings,
            u_imp_ptr, u_imp_items,
            model.p, model.q, model.y, model.bu, model.bi, model.mu,
            lr_b0 * d_factor, lr0 * d_factor,
            reg_b, reg_p, reg_q, reg_y,
            ep_shrink, ep_shrink, ep_shrink, alpha
        )
        if tune_pos is not None:
            pred = model.predict(tune_pos[:, 0], tune_pos[:, 1])
            mse = float(np.mean((pred - tune_pos[:, 3]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_epoch = ep
                stale = 0
                best_state = (
                    model.p.copy(), model.q.copy(), model.y.copy(),
                    model.bu.copy(), model.bi.copy()
                )
            else:
                stale += 1
                if stale >= 4:
                    break

    if best_state is not None:
        model.p, model.q, model.y, model.bu, model.bi = best_state
    elif tune_pos is not None:
        pred = model.predict(tune_pos[:, 0], tune_pos[:, 1])
        best_mse = float(np.mean((pred - tune_pos[:, 3]) ** 2))
    return model, best_epoch, best_mse


def compute_spectral_features(a_rated, a_implicit, nu, ni, u_pos_cnt, i_pos_cnt,
                              u_imp_cnt, i_imp_cnt, res, res_n, u_std, q_u, q_i, seed=SEED_PRIMARY):
    """SVD on degree-normalized a_implicit, aggregating a_rated residuals (raw and std-normalized)."""
    inv_u = 1.0 / np.sqrt(u_imp_cnt + 10.0)
    inv_i = 1.0 / np.sqrt(i_imp_cnt + 10.0)
    x_imp = csr_matrix(
        (inv_u[a_implicit[:, 0]] * inv_i[a_implicit[:, 1]], (a_implicit[:, 0], a_implicit[:, 1])),
        shape=(nu, ni), dtype=np.float32
    )
    u_imp, s_imp, vt_imp = svds(x_imp, k=48, random_state=seed)

    v_i = (vt_imp.T * (s_imp ** 0.5)).astype(np.float32)
    v_i /= (np.linalg.norm(v_i, axis=1, keepdims=True) + 1e-8)

    u_u = (u_imp * (s_imp ** 0.5)).astype(np.float32)
    u_u /= (np.linalg.norm(u_u, axis=1, keepdims=True) + 1e-8)

    # Raw residual profiles on a_rated
    u_prof_sum = np.zeros((nu, 48), dtype=np.float32)
    np.add.at(u_prof_sum, a_rated[:, 0], res[:, None] * v_i[a_rated[:, 1]])
    u_prof5 = np.vstack([u_prof_sum / (u_pos_cnt[:, None] + 5.0), np.zeros((1, 48), np.float32)])
    u_prof1 = np.vstack([u_prof_sum / (u_pos_cnt[:, None] + 1.5), np.zeros((1, 48), np.float32)])

    i_prof_sum = np.zeros((ni, 48), dtype=np.float32)
    np.add.at(i_prof_sum, a_rated[:, 1], res[:, None] * u_u[a_rated[:, 0]])
    i_prof5 = np.vstack([i_prof_sum / (i_pos_cnt[:, None] + 5.0), np.zeros((1, 48), np.float32)])
    i_prof1 = np.vstack([i_prof_sum / (i_pos_cnt[:, None] + 1.5), np.zeros((1, 48), np.float32)])

    # Normalized residual profiles on a_rated
    u_profn_sum = np.zeros((nu, 48), dtype=np.float32)
    np.add.at(u_profn_sum, a_rated[:, 0], res_n[:, None] * v_i[a_rated[:, 1]])
    u_profn5 = np.vstack([u_profn_sum / (u_pos_cnt[:, None] + 5.0), np.zeros((1, 48), np.float32)])

    i_profn_sum = np.zeros((ni, 48), dtype=np.float32)
    np.add.at(i_profn_sum, a_rated[:, 1], res_n[:, None] * u_u[a_rated[:, 0]])
    i_profn5 = np.vstack([i_profn_sum / (i_pos_cnt[:, None] + 5.0), np.zeros((1, 48), np.float32)])

    v_i_pad = np.vstack([v_i, np.zeros((1, 48), np.float32)])
    u_u_pad = np.vstack([u_u, np.zeros((1, 48), np.float32)])

    vi_q = v_i_pad[q_i]
    uu_q = u_u_pad[q_u]

    feat_ii5 = np.sum(u_prof5[q_u] * vi_q, axis=1)
    feat_ii1 = np.sum(u_prof1[q_u] * vi_q, axis=1)
    feat_uu5 = np.sum(i_prof5[q_i] * uu_q, axis=1)
    feat_uu1 = np.sum(i_prof1[q_i] * uu_q, axis=1)
    feat_iin5 = np.sum(u_profn5[q_u] * vi_q, axis=1) * u_std[q_u]
    feat_uun5 = np.sum(i_profn5[q_i] * uu_q, axis=1) * u_std[q_u]
    spec_cos = np.sum(uu_q * vi_q, axis=1)

    return np.column_stack([feat_ii5, feat_ii1, feat_uu5, feat_uu1, feat_iin5, feat_uun5, spec_cos])


@njit(cache=True)
def _insert_top5(w, t1, t2, t3, t4, t5):
    if w > t1:
        return w, t1, t2, t3, t4
    elif w > t2:
        return t1, w, t2, t3, t4
    elif w > t3:
        return t1, t2, w, t3, t4
    elif w > t4:
        return t1, t2, t3, w, t4
    elif w > t5:
        return t1, t2, t3, t4, w
    return t1, t2, t3, t4, t5


@njit(cache=True)
def fused_knn_and_support(
    q_u, q_i,
    u_ptr, u_items, u_res, u_resn,
    i_ptr, i_users, i_res, i_resn,
    u_pos_cnt, i_pos_cnt, u_imp_cnt, i_imp_cnt
):
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

    f_raw = np.zeros((n, 19), dtype=np.float64)
    f_norm = np.zeros((n, 19), dtype=np.float64)
    f_supp = np.zeros((n, 18), dtype=np.float64)

    present = np.zeros(nu, dtype=np.bool_)
    target_r = np.zeros(nu, dtype=np.float64)
    target_rn = np.zeros(nu, dtype=np.float64)

    dot = np.zeros(nu, dtype=np.float64)
    dot_n = np.zeros(nu, dtype=np.float64)
    su = np.zeros(nu, dtype=np.float64)
    sv = np.zeros(nu, dtype=np.float64)
    su_n = np.zeros(nu, dtype=np.float64)
    sv_n = np.zeros(nu, dtype=np.float64)
    cnt = np.zeros(nu, dtype=np.float64)
    dotw = np.zeros(nu, dtype=np.float64)
    dotw_n = np.zeros(nu, dtype=np.float64)
    cw = np.zeros(nu, dtype=np.float64)

    inv_i_imp = 1.0 / np.sqrt(i_imp_cnt.astype(np.float64) + 10.0)
    inv_u_imp = 1.0 / np.sqrt(u_imp_cnt.astype(np.float64) + 5.0)

    for z in range(n):
        u = q_u[z]
        i = q_i[z]
        if u >= nu or i >= ni:
            continue
        istart, iend = i_ptr[i], i_ptr[i + 1]
        ustart, uend = u_ptr[u], u_ptr[u + 1]
        if istart == iend or ustart == uend:
            continue

        for p in range(istart, iend):
            v = i_users[p]
            present[v] = True
            target_r[v] = i_res[p]
            target_rn[v] = i_resn[p]

        nums = np.zeros(11, dtype=np.float64)
        dens = np.zeros(11, dtype=np.float64)
        nums_n = np.zeros(11, dtype=np.float64)
        dens_n = np.zeros(11, dtype=np.float64)

        ii_sum_w = 0.0
        ii_sum_w2 = 0.0
        ii_cnt = 0.0
        ii_max_sim = 0.0
        ii_t1, ii_t2, ii_t3, ii_t4, ii_t5 = 0.0, 0.0, 0.0, 0.0, 0.0
        ii_we = 0.0
        ii_we2 = 0.0
        ii_struct_w = 0.0
        ii_struct_w2 = 0.0

        norm_i = 1.0 / np.sqrt(float(i_imp_cnt[i]) + 5.0)

        for p in range(ustart, uend):
            j = u_items[p]
            if j == i:
                continue
            ru = float(u_res[p])
            run = float(u_resn[p])
            w_j = inv_i_imp[j]
            norm_j = 1.0 / np.sqrt(float(i_imp_cnt[j]) + 5.0)

            ij = 0.0
            ii_s = 0.0
            jj_s = 0.0
            ij_n = 0.0
            ii_sn = 0.0
            jj_sn = 0.0
            nc = 0.0
            struct_dot = 0.0

            jstart, jend = i_ptr[j], i_ptr[j + 1]
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

            cos = ij / (np.sqrt(ii_s * jj_s) + 1e-8)
            cos_n = ij_n / (np.sqrt(ii_sn * jj_sn) + 1e-8)

            # Use shrink=5 similarity magnitude for support/confidence statistics
            s5 = cos * nc / (nc + 5.0)
            as5 = abs(s5)
            ii_sum_w += as5
            ii_sum_w2 += as5 * as5
            ii_cnt += 1.0
            if as5 > ii_max_sim:
                ii_max_sim = as5
            ii_t1, ii_t2, ii_t3, ii_t4, ii_t5 = _insert_top5(as5, ii_t1, ii_t2, ii_t3, ii_t4, ii_t5)
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

        uu_nums = np.zeros(8, dtype=np.float64)
        uu_dens = np.zeros(8, dtype=np.float64)
        uu_nums_n = np.zeros(8, dtype=np.float64)
        uu_dens_n = np.zeros(8, dtype=np.float64)

        uu_sum_w = 0.0
        uu_sum_w2 = 0.0
        uu_cnt = 0.0
        uu_max_sim = 0.0
        uu_t1, uu_t2, uu_t3, uu_t4, uu_t5 = 0.0, 0.0, 0.0, 0.0, 0.0
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
                cos = dot[v] / (np.sqrt(su[v] * sv[v]) + 1e-8)
                cos_n = dot_n[v] / (np.sqrt(su_n[v] * sv_n[v]) + 1e-8)

                s5 = cos * nc / (nc + 5.0)
                as5 = abs(s5)
                uu_sum_w += as5
                uu_sum_w2 += as5 * as5
                uu_cnt += 1.0
                if as5 > uu_max_sim:
                    uu_max_sim = as5
                uu_t1, uu_t2, uu_t3, uu_t4, uu_t5 = _insert_top5(as5, uu_t1, uu_t2, uu_t3, uu_t4, uu_t5)
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

        # Support / confidence features
        ii_neff = (ii_sum_w * ii_sum_w) / (ii_sum_w2 + 1e-8)
        ii_mean_e = ii_we / (ii_sum_w + 1e-8)
        ii_var_e = max(0.0, (ii_we2 / (ii_sum_w + 1e-8)) - ii_mean_e * ii_mean_e) * (ii_cnt / (ii_cnt + 2.0))
        ii_s_neff = (ii_struct_w * ii_struct_w) / (ii_struct_w2 + 1e-8)

        f_supp[z, 0] = ii_sum_w
        f_supp[z, 1] = ii_neff
        f_supp[z, 2] = ii_cnt
        f_supp[z, 3] = ii_max_sim
        f_supp[z, 4] = ii_t1 / (ii_sum_w + 1e-8)
        f_supp[z, 5] = (ii_t1 + ii_t2 + ii_t3 + ii_t4 + ii_t5) / (ii_sum_w + 1e-8)
        f_supp[z, 6] = ii_var_e
        f_supp[z, 7] = dens[5]  # positive-similarity squared weight sum (shrink=5)
        f_supp[z, 8] = ii_s_neff

        uu_neff = (uu_sum_w * uu_sum_w) / (uu_sum_w2 + 1e-8)
        uu_mean_e = uu_we / (uu_sum_w + 1e-8)
        uu_var_e = max(0.0, (uu_we2 / (uu_sum_w + 1e-8)) - uu_mean_e * uu_mean_e) * (uu_cnt / (uu_cnt + 2.0))
        uu_s_neff = (uu_struct_w * uu_struct_w) / (uu_struct_w2 + 1e-8)

        f_supp[z, 9] = uu_sum_w
        f_supp[z, 10] = uu_neff
        f_supp[z, 11] = uu_cnt
        f_supp[z, 12] = uu_max_sim
        f_supp[z, 13] = uu_t1 / (uu_sum_w + 1e-8)
        f_supp[z, 14] = (uu_t1 + uu_t2 + uu_t3 + uu_t4 + uu_t5) / (uu_sum_w + 1e-8)
        f_supp[z, 15] = uu_var_e
        f_supp[z, 16] = uu_dens[5]
        f_supp[z, 17] = uu_s_neff

    return f_raw, f_norm, f_supp


@njit(cache=True)
def task3_session_and_kernel_features(
    q_ids, q_times, ptr, sorted_times, sorted_res, taus, shrinks, is_user
):
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
    # Per tau:
    #   past_mean, fut_mean, comb_mean, comb_mean_lowshrink,
    #   past_num_tanh, fut_num_tanh,
    #   past_sumw, fut_sumw, comb_sumw,
    #   past_neff, fut_neff, comb_neff  -> 12 cols per tau
    n_kcols = n_taus * 12
    n_scols = 27 if is_user else 1
    out = np.zeros((n, n_kcols + n_scols), dtype=np.float64)

    for idx in range(n):
        e = q_ids[idx]
        t = q_times[idx]
        if e >= len(ptr) - 1:
            continue
        start, end = ptr[e], ptr[e + 1]
        L = end - start
        if L == 0:
            continue

        pos = np.searchsorted(sorted_times[start:end], t) + start
        # Temporal rank quantile rank_u(t) / (L_u + 1)
        out[idx, n_kcols] = float(pos - start + 1) / float(L + 1)

        # Scan window around pos (up to 400 on each side covers all session & medium-term neighbors,
        # and for full user history L is almost always < 800)
        left = max(start, pos - 400)
        right = min(end, pos + 400)

        for m in range(n_taus):
            tau = taus[m]
            sh = shrinks[m]
            w_past, w2_past, r_past = 0.0, 0.0, 0.0
            w_fut, w2_fut, r_fut = 0.0, 0.0, 0.0
            w_all, w2_all, r_all = 0.0, 0.0, 0.0

            for p in range(left, right):
                tp = sorted_times[p]
                rp = sorted_res[p]
                dt = float(abs(t - tp))
                w = np.exp(-dt / tau)
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
            # Bounded numerator so raw unnormalized sums do not blow up linear ridge on huge bursts
            out[idx, base_c + 4] = r_past / (1.0 + 0.2 * w_past)
            out[idx, base_c + 5] = r_fut / (1.0 + 0.2 * w_fut)
            out[idx, base_c + 6] = np.log1p(w_past)
            out[idx, base_c + 7] = np.log1p(w_fut)
            out[idx, base_c + 8] = np.log1p(w_all)
            out[idx, base_c + 9] = (w_past * w_past) / (w2_past + 1e-8)
            out[idx, base_c + 10] = (w_fut * w_fut) / (w2_fut + 1e-8)
            out[idx, base_c + 11] = (w_all * w_all) / (w2_all + 1e-8)

        if not is_user:
            continue

        # Exact session features for user
        s_sec_r, s_sec_c = 0.0, 0.0
        s_min_r, s_min_c = 0.0, 0.0
        s_hr_r, s_hr_r2, s_hr_c = 0.0, 0.0, 0.0
        s_day_r, s_day_r2, s_day_c = 0.0, 0.0, 0.0

        best_prev_dt = 1e18
        best_prev_r = 0.0
        has_prev = 0.0

        best_next_dt = 1e18
        best_next_r = 0.0
        has_next = 0.0

        best_any_dt = 1e18
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
        # Same-second
        out[idx, sc + 0] = s_sec_r / (s_sec_c + 0.5)
        out[idx, sc + 1] = s_sec_r / (s_sec_c + 2.0)
        out[idx, sc + 2] = np.log1p(s_sec_c)
        # Same-minute
        out[idx, sc + 3] = s_min_r / (s_min_c + 0.5)
        out[idx, sc + 4] = s_min_r / (s_min_c + 2.0)
        out[idx, sc + 5] = np.log1p(s_min_c)
        # Same-hour
        out[idx, sc + 6] = s_hr_r / (s_hr_c + 0.6)
        out[idx, sc + 7] = s_hr_r / (s_hr_c + 2.5)
        out[idx, sc + 8] = np.log1p(s_hr_c)
        # Same-day
        out[idx, sc + 9] = s_day_r / (s_day_c + 0.8)
        out[idx, sc + 10] = s_day_r / (s_day_c + 3.0)
        out[idx, sc + 11] = np.log1p(s_day_c)

        # Within-1-hour and within-1-day residual std
        if s_hr_c >= 2.0:
            hr_m = s_hr_r / s_hr_c
            hr_v = max(0.0, (s_hr_r2 / s_hr_c) - hr_m * hr_m)
            out[idx, sc + 12] = np.sqrt(hr_v) * (s_hr_c / (s_hr_c + 2.0))
        if s_day_c >= 2.0:
            day_m = s_day_r / s_day_c
            day_v = max(0.0, (s_day_r2 / s_day_c) - day_m * day_m)
            out[idx, sc + 13] = np.sqrt(day_v) * (s_day_c / (s_day_c + 2.0))

        # Nearest past
        out[idx, sc + 14] = has_prev
        if has_prev > 0.0:
            out[idx, sc + 15] = best_prev_r
            out[idx, sc + 16] = np.log1p(best_prev_dt)
            out[idx, sc + 17] = best_prev_r / (1.0 + np.sqrt(best_prev_dt / 3600.0))
            out[idx, sc + 18] = best_prev_r / (1.0 + best_prev_dt / 86400.0)
        else:
            out[idx, sc + 16] = 20.0

        # Nearest future
        out[idx, sc + 19] = has_next
        if has_next > 0.0:
            out[idx, sc + 20] = best_next_r
            out[idx, sc + 21] = np.log1p(best_next_dt)
            out[idx, sc + 22] = best_next_r / (1.0 + np.sqrt(best_next_dt / 3600.0))
            out[idx, sc + 23] = best_next_r / (1.0 + best_next_dt / 86400.0)
        else:
            out[idx, sc + 21] = 20.0

        # Nearest overall (including dt=0)
        if best_any_dt < 1e17:
            out[idx, sc + 24] = best_any_r / (1.0 + np.sqrt(best_any_dt / 600.0))
            out[idx, sc + 25] = np.log1p(best_any_dt)
        else:
            out[idx, sc + 25] = 20.0

    return out


def fit_user_temporal_drift(a_rated, nu, res, u_pos_cnt):
    """Fits per-user temporal drift on a_rated:
    d_u(t) = sign(t - t_bar_u) * |(t - t_bar_u) / 86400|^0.4
    alpha_u = sum_j d_u(t_uj) e_uj / (lambda + sum_j d_u(t_uj)^2), lambda = 10.0 (and 30.0).
    """
    u_idx = a_rated[:, 0]
    t_vals = a_rated[:, 2].astype(np.float64)
    t_sum = np.bincount(u_idx, weights=t_vals, minlength=nu)
    t_bar = np.where(u_pos_cnt > 0, t_sum / np.maximum(u_pos_cnt, 1), 0.0)

    dt_days = (t_vals - t_bar[u_idx]) / 86400.0
    d_uj = np.sign(dt_days) * (np.abs(dt_days) ** 0.4)

    num = np.bincount(u_idx, weights=d_uj * res.astype(np.float64), minlength=nu)
    den = np.bincount(u_idx, weights=d_uj * d_uj, minlength=nu)

    alpha_10 = num / (den + 10.0)
    alpha_30 = num / (den + 30.0)
    return t_bar, alpha_10, alpha_30


def eval_user_temporal_drift(q_u, q_t, nu, t_bar, alpha_10, alpha_30, u_pos_cnt):
    u_valid = (q_u < nu)
    u_safe = np.where(u_valid, q_u, 0)
    has_pos = u_valid & (u_pos_cnt[u_safe] > 0)
    dt_days = np.where(has_pos, (q_t.astype(np.float64) - t_bar[u_safe]) / 86400.0, 0.0)
    # Clip extreme query-time extrapolate drift to prevent outlier spikes
    d_ut = np.clip(np.sign(dt_days) * (np.abs(dt_days) ** 0.4), -15.0, 15.0)
    a10 = np.where(has_pos, alpha_10[u_safe], 0.0)
    a30 = np.where(has_pos, alpha_30[u_safe], 0.0)
    drift_pred10 = a10 * d_ut
    drift_pred30 = a30 * d_ut
    return np.column_stack([drift_pred10, drift_pred30, d_ut, a10])


def fit_rating_base(fit_rows, nu, ni, task, tune_rows=None, seed=SEED_PRIMARY, fixed_epochs=None, run_svdpp_grid=True):
    """Fits all rating supervision models strictly on a_rated = fit_rows[fit_rows[:, 3] > 0],
    while preserving a_implicit = fit_rows for implicit graph counts, zero rates, spectral SVD, and SVD++ N(u).
    """
    assert fixed_epochs is not None and tune_rows is None and (not run_svdpp_grid), 'Production fits require frozen epochs and no tuning rows'
    t0 = time.time()
    a_implicit = fit_rows
    a_rated = fit_rows[fit_rows[:, 3] > 0]
    tune_pos = None
    if tune_rows is not None:
        tune_pos = tune_rows[tune_rows[:, 3] > 0]
    u_imp_cnt = np.bincount(a_implicit[:, 0], minlength=nu)
    i_imp_cnt = np.bincount(a_implicit[:, 1], minlength=ni)
    u_pos_cnt = np.bincount(a_rated[:, 0], minlength=nu)
    i_pos_cnt = np.bincount(a_rated[:, 1], minlength=ni)
    mu0, bu0, bi0 = base.als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 1.5, 2.5, n_iter=20)
    mu1, bu1, bi1 = base.als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 3.0, 5.0, n_iter=20)
    mu2, bu2, bi2 = base.als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 8.0, 12.0, n_iter=20)
    mu3, bu3, bi3 = base.als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 25.0, 35.0, n_iter=20)
    mu4, bu4, bi4 = base.als_biases_custom(a_rated, nu, ni, u_pos_cnt, i_pos_cnt, 60.0, 80.0, n_iter=20)
    res = (a_rated[:, 3] - (mu2 + bu2[a_rated[:, 0]] + bi2[a_rated[:, 1]])).astype(np.float32)
    prior_var = float(np.mean(res ** 2))
    u_var = np.bincount(a_rated[:, 0], weights=res ** 2, minlength=nu)
    u_std = np.append(np.sqrt((u_var + 5.0 * prior_var) / (u_pos_cnt + 5.0)), np.sqrt(prior_var)).astype(np.float64)
    i_var = np.bincount(a_rated[:, 1], weights=res ** 2, minlength=ni)
    i_std = np.append(np.sqrt((i_var + 5.0 * prior_var) / (i_pos_cnt + 5.0)), np.sqrt(prior_var)).astype(np.float64)
    res_n = (res / (u_std[a_rated[:, 0]] + 0.1)).astype(np.float32)
    u_zero = u_imp_cnt - u_pos_cnt
    i_zero = i_imp_cnt - i_pos_cnt
    p0_global = float(np.sum(u_zero) / max(len(a_implicit), 1))
    pu_z = np.append((u_zero + 2.0 * p0_global) / (u_imp_cnt + 2.0), p0_global)
    pi_z = np.append((i_zero + 10.0 * p0_global) / (i_imp_cnt + 10.0), p0_global)
    u_order = np.lexsort((a_rated[:, 2], a_rated[:, 0]))
    u_pos_ptr = np.r_[0, u_pos_cnt.cumsum()].astype(np.int32)
    u_pos_items = a_rated[u_order, 1].astype(np.int32)
    u_pos_times = a_rated[u_order, 2].astype(np.int64)
    u_pos_ratings = a_rated[u_order, 3].astype(np.float32)
    u_pos_res = res[u_order]
    u_pos_resn = res_n[u_order]
    i_order = np.lexsort((a_rated[:, 2], a_rated[:, 1]))
    i_pos_ptr = np.r_[0, i_pos_cnt.cumsum()].astype(np.int32)
    i_pos_users = a_rated[i_order, 0].astype(np.int32)
    i_pos_times = a_rated[i_order, 2].astype(np.int64)
    i_pos_res = res[i_order]
    i_pos_resn = res_n[i_order]
    u_imp_order = np.lexsort((a_implicit[:, 2], a_implicit[:, 0]))
    u_imp_ptr = np.r_[0, u_imp_cnt.cumsum()].astype(np.int32)
    u_imp_items = a_implicit[u_imp_order, 1].astype(np.int32)
    u_imp_times = a_implicit[u_imp_order, 2].astype(np.int64)
    ep_mf32 = fixed_epochs['mf32'] if fixed_epochs else 28
    mf32, best_ep_mf32, mse_mf32 = train_explicit_pos_mf(a_rated, nu, ni, k=32, lr0=0.012, decay=0.97, reg_b=0.02, reg_pq=0.055, max_epochs=ep_mf32, bu_init=None, bi_init=None, tune_pos=tune_pos if not fixed_epochs else None, seed=seed)
    ep_mf64 = fixed_epochs['mf64'] if fixed_epochs else 25
    mf64, best_ep_mf64, mse_mf64 = train_explicit_pos_mf(a_rated, nu, ni, k=64, lr0=0.015, decay=0.96, reg_b=0.03, reg_pq=0.07, max_epochs=ep_mf64, bu_init=bu2, bi_init=bi2, tune_pos=tune_pos if not fixed_epochs else None, seed=seed + 1)
    svdpp_grid_results = {}
    ep_s32 = fixed_epochs['svdpp32'] if fixed_epochs else 25
    best_alpha32 = fixed_epochs.get('alpha32', 0.0) if fixed_epochs else 0.0
    lr32 = 0.009 if best_alpha32 == 0.0 else 0.025 if best_alpha32 == 0.5 else 0.06
    svdpp32, best_ep_svdpp32, mse_svdpp32 = train_user_block_svdpp(u_pos_ptr, u_pos_items, u_pos_ratings, u_imp_ptr, u_imp_items, nu, ni, mu2, bu2, bi2, k=32, alpha=best_alpha32, lr_b0=0.05, lr0=lr32, decay=0.96, reg_b=0.08, reg_p=0.06, reg_q=0.06, reg_y=0.06, ep_shrink=0.002, max_epochs=ep_s32, tune_pos=None, seed=seed + 32)
    if tune_pos is not None:
        mse_svdpp32 = float(np.mean((svdpp32.predict(tune_pos[:, 0], tune_pos[:, 1]) - tune_pos[:, 3]) ** 2))
    ep_s64 = fixed_epochs['svdpp64'] if fixed_epochs else 25
    best_alpha64 = fixed_epochs.get('alpha64', 0.0) if fixed_epochs else 0.0
    lr64 = 0.009 if best_alpha64 == 0.0 else 0.025 if best_alpha64 == 0.5 else 0.06
    svdpp64, best_ep_svdpp64, mse_svdpp64 = train_user_block_svdpp(u_pos_ptr, u_pos_items, u_pos_ratings, u_imp_ptr, u_imp_items, nu, ni, mu2, bu2, bi2, k=64, alpha=best_alpha64, lr_b0=0.05, lr0=lr64, decay=0.96, reg_b=0.08, reg_p=0.07, reg_q=0.07, reg_y=0.07, ep_shrink=0.002, max_epochs=ep_s64, tune_pos=None, seed=seed + 64)
    if tune_pos is not None:
        mse_svdpp64 = float(np.mean((svdpp64.predict(tune_pos[:, 0], tune_pos[:, 1]) - tune_pos[:, 3]) ** 2))
    t_bar, alpha_10, alpha_30 = (None, None, None)
    if task == 3:
        t_bar, alpha_10, alpha_30 = fit_user_temporal_drift(a_rated, nu, res, u_pos_cnt)
    return {'task': task, 'nu': nu, 'ni': ni, 'seed': seed, 'a_rated': a_rated, 'a_implicit': a_implicit, 'u_imp_cnt': u_imp_cnt, 'i_imp_cnt': i_imp_cnt, 'u_pos_cnt': u_pos_cnt, 'i_pos_cnt': i_pos_cnt, 'biases': ((mu0, bu0, bi0), (mu1, bu1, bi1), (mu2, bu2, bi2), (mu3, bu3, bi3), (mu4, bu4, bi4)), 'res': res, 'res_n': res_n, 'u_std': u_std, 'i_std': i_std, 'pu_z': pu_z, 'pi_z': pi_z, 'u_pos_ptr': u_pos_ptr, 'u_pos_items': u_pos_items, 'u_pos_times': u_pos_times, 'u_pos_ratings': u_pos_ratings, 'u_pos_res': u_pos_res, 'u_pos_resn': u_pos_resn, 'i_pos_ptr': i_pos_ptr, 'i_pos_users': i_pos_users, 'i_pos_times': i_pos_times, 'i_pos_res': i_pos_res, 'i_pos_resn': i_pos_resn, 'u_imp_ptr': u_imp_ptr, 'u_imp_items': u_imp_items, 'u_imp_times': u_imp_times, 'mf32': mf32, 'mf64': mf64, 'svdpp32': svdpp32, 'svdpp64': svdpp64, 'best_epochs': {'mf32': int(best_ep_mf32), 'mf64': int(best_ep_mf64), 'svdpp32': int(best_ep_svdpp32), 'svdpp64': int(best_ep_svdpp64), 'alpha32': float(best_alpha32), 'alpha64': float(best_alpha64)}, 'base_tune_mses': {'mf32_tune_pos_mse': float(mse_mf32), 'mf64_tune_pos_mse': float(mse_mf64), 'svdpp32_tune_pos_mse': float(mse_svdpp32), 'svdpp64_tune_pos_mse': float(mse_svdpp64)}, 'svdpp_grid_results': svdpp_grid_results, 't_bar': t_bar, 'alpha_10': alpha_10, 'alpha_30': alpha_30, 'fit_elapsed_sec': round(time.time() - t0, 2)}


def build_rating_features(state, q_u, q_i, q_t):
    """Builds the complete feature matrix for queries (q_u, q_i, q_t) and returns
    (X, col_groups, diagnostics) to support exact progressive stage ablations.
    """
    task = state['task']
    nu, ni = state['nu'], state['ni']
    n = len(q_u)

    (mu0, bu0, bi0), (mu1, bu1, bi1), (mu2, bu2, bi2), (mu3, bu3, bi3), (mu4, bu4, bi4) = state['biases']
    bu0_e, bi0_e = np.append(bu0, 0.0), np.append(bi0, 0.0)
    bu1_e, bi1_e = np.append(bu1, 0.0), np.append(bi1, 0.0)
    bu2_e, bi2_e = np.append(bu2, 0.0), np.append(bi2, 0.0)
    bu3_e, bi3_e = np.append(bu3, 0.0), np.append(bi3, 0.0)
    bu4_e, bi4_e = np.append(bu4, 0.0), np.append(bi4, 0.0)

    # Clean fallback: for users/items with 0 positive ratings, bu_e[u]==0 / bi_e[i]==0,
    # so base_k automatically falls back to mu_k + bi_k[i] or mu_k + bu_k[u].
    base0 = mu0 + bu0_e[q_u] + bi0_e[q_i]
    base1 = mu1 + bu1_e[q_u] + bi1_e[q_i]
    base2 = mu2 + bu2_e[q_u] + bi2_e[q_i]
    base3 = mu3 + bu3_e[q_u] + bi3_e[q_i]
    base4 = mu4 + bu4_e[q_u] + bi4_e[q_i]

    u_pos_cnt_e = np.append(state['u_pos_cnt'], 0)
    i_pos_cnt_e = np.append(state['i_pos_cnt'], 0)
    u_imp_cnt_e = np.append(state['u_imp_cnt'], 0)
    i_imp_cnt_e = np.append(state['i_imp_cnt'], 0)

    log_upos = np.log1p(u_pos_cnt_e[q_u])
    log_ipos = np.log1p(i_pos_cnt_e[q_i])
    log_uimp = np.log1p(u_imp_cnt_e[q_u])
    log_iimp = np.log1p(i_imp_cnt_e[q_i])

    u_no_pos = (u_pos_cnt_e[q_u] == 0).astype(np.float64)
    i_no_pos = (i_pos_cnt_e[q_i] == 0).astype(np.float64)
    u_no_imp = (u_imp_cnt_e[q_u] == 0).astype(np.float64)
    i_no_imp = (i_imp_cnt_e[q_i] == 0).astype(np.float64)

    u_std_q = state['u_std'][q_u]
    i_std_q = state['i_std'][q_i]
    pu_z_q = state['pu_z'][q_u]
    pi_z_q = state['pi_z'][q_i]

    # Explicit PosMF predictions
    mf32_pred, mf32_dot = state['mf32'].predict_components(q_u, q_i)
    mf64_pred, mf64_dot = state['mf64'].predict_components(q_u, q_i)
    mf32_res = mf32_pred - base2
    mf64_res = mf64_pred - base2

    # Spectral SVD features (graph on a_implicit, residuals on a_rated)
    f_spec = compute_spectral_features(
        state['a_rated'], state['a_implicit'], nu, ni,
        state['u_pos_cnt'], state['i_pos_cnt'], state['u_imp_cnt'], state['i_imp_cnt'],
        state['res'], state['res_n'], state['u_std'], q_u, q_i, seed=state['seed']
    )

    # Structural 2-hop neighborhood features on a_rated (using a_implicit degree normalization)
    f_2hop_raw = base.exact_2hop_uu_ii(
        q_u, q_i, q_t, task == 3,
        state['u_pos_ptr'], state['u_pos_items'], state['u_pos_times'], state['u_pos_res'],
        state['i_pos_ptr'], state['i_pos_users'], state['i_pos_times'], state['i_pos_res'],
        state['u_imp_cnt'], state['i_imp_cnt']
    ).astype(np.float64)
    f_2hop_use = f_2hop_raw[:, :11] if task == 1 else f_2hop_raw

    # Stage 1: Positive-conditional base block (with basic temporal kernels on a_rated for Task 3)
    stage1_list = [
        np.ones(n, dtype=np.float64),
        base0, base1, base2, base3, base4,
        bu2_e[q_u], bi2_e[q_i],
        pu_z_q, pi_z_q, base2 * pu_z_q, base2 * pi_z_q,
        u_no_pos, i_no_pos, u_no_imp, i_no_imp,
        log_upos, log_ipos, log_uimp, log_iimp,
        mf32_pred, mf64_pred, mf32_dot, mf64_dot,
        f_spec,
        f_2hop_use,
        u_std_q * bi2_e[q_i],
        i_std_q * bu2_e[q_u],
        u_std_q * f_2hop_raw[:, 0],
        u_std_q * f_2hop_raw[:, 6],
        u_std_q * mf32_res,
        u_std_q * mf64_res,
        i_std_q * f_2hop_raw[:, 0],
        i_std_q * f_2hop_raw[:, 6],
        i_std_q * mf32_res,
        i_std_q * mf64_res,
    ]
    if task == 3:
        taus_7 = np.array([600.0, 3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0, 3650 * 86400.0])
        u_sh_7 = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
        i_sh_7 = np.array([3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0])
        u_tk7 = base.temporal_kernel_residuals(
            q_u, q_t, state['u_pos_ptr'], state['u_pos_times'],
            state['u_pos_res'].astype(np.float64), taus_7, u_sh_7
        )
        i_tk7 = base.temporal_kernel_residuals(
            q_i, q_t, state['i_pos_ptr'], state['i_pos_times'],
            state['i_pos_res'].astype(np.float64), taus_7, i_sh_7
        )
        stage1_list.extend([u_tk7, i_tk7, u_std_q[:, None] * u_tk7[:, 1:4]])

    x_stage1 = np.column_stack(stage1_list)

    # Stage 2: User-Block CSR Mini-Batch SVD++ features
    svdpp32_pred, svdpp32_dp, svdpp32_dh = state['svdpp32'].predict_components(q_u, q_i)
    svdpp64_pred, svdpp64_dp, svdpp64_dh = state['svdpp64'].predict_components(q_u, q_i)
    svdpp32_res = svdpp32_pred - base2
    svdpp64_res = svdpp64_pred - base2

    x_stage2_only = np.column_stack([
        svdpp32_pred, svdpp64_pred,
        svdpp32_dp, svdpp32_dh,
        svdpp64_dp, svdpp64_dh,
        u_std_q * svdpp32_res, i_std_q * svdpp32_res,
        u_std_q * svdpp64_res, i_std_q * svdpp64_res,
    ])

    # Stage 3: Fused Cosine/Dot 2-Hop kNN + Dispersion-Normalized kNN + Support/Confidence Features
    f_knn_raw, f_knn_norm, f_knn_supp = fused_knn_and_support(
        q_u, q_i,
        state['u_pos_ptr'], state['u_pos_items'], state['u_pos_res'], state['u_pos_resn'],
        state['i_pos_ptr'], state['i_pos_users'], state['i_pos_res'], state['i_pos_resn'],
        state['u_pos_cnt'], state['i_pos_cnt'], state['u_imp_cnt'], state['i_imp_cnt']
    )
    f_knn_norm_rescaled = f_knn_norm * u_std_q[:, None]
    # Also structural 2-hop on normalized residuals rescaled by u_std_q
    f_2hop_norm = base.exact_2hop_uu_ii(
        q_u, q_i, q_t, task == 3,
        state['u_pos_ptr'], state['u_pos_items'], state['u_pos_times'], state['u_pos_resn'],
        state['i_pos_ptr'], state['i_pos_users'], state['i_pos_times'], state['i_pos_resn'],
        state['u_imp_cnt'], state['i_imp_cnt']
    ).astype(np.float64)
    f_2hop_norm_use = (f_2hop_norm[:, :11] if task == 1 else f_2hop_norm) * u_std_q[:, None]

    # Log-transformed support features for linear stability
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

    f_supp_clean = np.column_stack([
        np.log1p(ii_sum_w), np.log1p(ii_neff), np.log1p(ii_cnt),
        ii_max_sim, ii_top1_c, ii_top5_c, ii_var_e, np.log1p(ii_s_neff),
        np.log1p(uu_sum_w), np.log1p(uu_neff), np.log1p(uu_cnt),
        uu_max_sim, uu_top1_c, uu_top5_c, uu_var_e, np.log1p(uu_s_neff),
    ])

    x_stage3_only = np.column_stack([
        f_knn_raw,
        f_knn_norm,
        f_knn_norm_rescaled,
        f_2hop_norm_use,
        f_supp_clean,
    ])

    # Stage 4 (Task 3 Only): Fine-grained session residuals & user temporal drift
    if task == 3:
        taus_8 = np.array([60.0, 600.0, 3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0, 3650 * 86400.0])
        u_sh_8 = np.array([0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
        i_sh_8 = np.array([2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0])

        u_sess_raw = task3_session_and_kernel_features(
            q_u, q_t, state['u_pos_ptr'], state['u_pos_times'],
            state['u_pos_res'].astype(np.float64), taus_8, u_sh_8, True
        )
        u_sess_norm = task3_session_and_kernel_features(
            q_u, q_t, state['u_pos_ptr'], state['u_pos_times'],
            state['u_pos_resn'].astype(np.float64), taus_8, u_sh_8, False
        )
        i_sess_raw = task3_session_and_kernel_features(
            q_i, q_t, state['i_pos_ptr'], state['i_pos_times'],
            state['i_pos_res'].astype(np.float64), taus_8, i_sh_8, False
        )
        # Extract combined normalized kernel residuals for the 8 taus and rescale by u_std_q
        u_norm_comb = np.column_stack([u_sess_norm[:, m * 12 + 2] * u_std_q for m in range(8)])
        # Convert n_eff columns in u_sess_raw and i_sess_raw to log1p(n_eff) for linear stability
        for m in range(8):
            for c_off in (9, 10, 11):
                u_sess_raw[:, m * 12 + c_off] = np.log1p(u_sess_raw[:, m * 12 + c_off])
                i_sess_raw[:, m * 12 + c_off] = np.log1p(i_sess_raw[:, m * 12 + c_off])

        f_drift = eval_user_temporal_drift(
            q_u, q_t, nu, state['t_bar'], state['alpha_10'], state['alpha_30'], state['u_pos_cnt']
        )
        # Implicit temporal rank quantile
        imp_rank_q = np.zeros(n, dtype=np.float64)
        u_imp_ptr = state['u_imp_ptr']
        u_imp_times = state['u_imp_times']
        for idx in range(n):
            u = q_u[idx]
            if u < nu:
                s, e = u_imp_ptr[u], u_imp_ptr[u + 1]
                if e > s:
                    p_pos = np.searchsorted(u_imp_times[s:e], q_t[idx])
                    imp_rank_q[idx] = float(p_pos + 1) / float(e - s + 1)

        x_stage4_only = np.column_stack([
            u_sess_raw,
            u_norm_comb,
            i_sess_raw,
            f_drift,
            imp_rank_q,
        ])
    else:
        x_stage4_only = np.empty((n, 0), dtype=np.float64)

    # Stage 5: Explicit Confidence-Gating Interactions (Section 3.3)
    # 1. f_neigh * log1p(n_eff) and rational gate f_neigh * (n_eff / (n_eff + 5))
    log_ii_neff = np.log1p(ii_neff)
    log_uu_neff = np.log1p(uu_neff)
    gate_ii = ii_neff / (ii_neff + 5.0)
    gate_uu = uu_neff / (uu_neff + 5.0)
    gate_ii_s = ii_s_neff / (ii_s_neff + 10.0)
    gate_uu_s = uu_s_neff / (uu_s_neff + 10.0)

    f_neigh_ii = f_knn_raw[:, 1]  # cosine shrink=5
    f_neigh_uu = f_knn_raw[:, 10]
    f_2hop_ii = f_2hop_raw[:, 0]
    f_2hop_uu = f_2hop_raw[:, 6]

    # 2. (f_mf - f_bias) * log1p(u_pos_cnt) and (f_svdpp - f_bias) * log1p(u_imp_cnt)
    # 3. Pairwise model differences and absolute differences
    gating_cols = [
        f_neigh_ii * log_ii_neff,
        f_neigh_uu * log_uu_neff,
        f_neigh_ii * gate_ii,
        f_neigh_uu * gate_uu,
        f_2hop_ii * np.log1p(ii_s_neff),
        f_2hop_uu * np.log1p(uu_s_neff),
        f_2hop_ii * gate_ii_s,
        f_2hop_uu * gate_uu_s,
        f_neigh_ii / (1.0 + ii_var_e),
        f_neigh_uu / (1.0 + uu_var_e),
        mf32_res * log_upos,
        mf32_res * log_ipos,
        mf64_res * log_upos,
        mf64_res * log_ipos,
        svdpp32_res * log_uimp,
        svdpp32_res * log_iimp,
        svdpp64_res * log_uimp,
        svdpp64_res * log_iimp,
        svdpp32_pred - mf32_pred,
        np.abs(svdpp32_pred - mf32_pred),
        svdpp64_pred - mf64_pred,
        np.abs(svdpp64_pred - mf64_pred),
        np.abs(mf64_res),
        np.abs(svdpp64_res),
        np.abs(f_neigh_ii - f_neigh_uu),
    ]
    if task == 3:
        # Session gating interactions: f_sess * log1p(n_eff_sess)
        for m in range(4):  # 60s, 600s, 3600s, 86400s
            sess_mean = u_sess_raw[:, m * 12 + 2]
            sess_log_neff = u_sess_raw[:, m * 12 + 11]
            gating_cols.append(sess_mean * sess_log_neff)
            gating_cols.append(sess_mean * u_std_q)

    x_stage5_only = np.column_stack(gating_cols)

    # Assemble cumulative feature matrix and stage column slices
    c0 = 52  # Static positive-conditional base without any timestamps
    c1 = x_stage1.shape[1]
    c2 = c1 + x_stage2_only.shape[1]
    c3 = c2 + x_stage3_only.shape[1]
    c4 = c3 + x_stage4_only.shape[1]
    c5 = c4 + x_stage5_only.shape[1]

    X_full = np.column_stack([x_stage1, x_stage2_only, x_stage3_only, x_stage4_only, x_stage5_only])
    col_groups = {}
    if task == 3:
        col_groups['stage0_static_no_time'] = slice(0, c0)
    col_groups.update({
        'stage1_pos_base': slice(0, c1),
        'stage2_plus_svdpp': slice(0, c2),
        'stage3_plus_knn_support': slice(0, c3),
        'stage4_plus_session_drift': slice(0, c4),
        'stage5_plus_conf_gating': slice(0, c5),
    })
    return X_full, col_groups


class StandardizedRidge:
    def __init__(self, l2=10.0):
        self.l2 = float(l2)
        self.mean = None
        self.scale = None
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        # Column 0 is constant 1.0 intercept; standardize columns 1..D using training stats ONLY
        X_feat = X[:, 1:]
        self.mean = np.mean(X_feat, axis=0)
        std = np.std(X_feat, axis=0)
        self.scale = np.where(std > 1e-7, std, 1.0)
        Z = (X_feat - self.mean) / self.scale
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        d = Z.shape[1]
        A = Z.T @ Z + self.l2 * np.eye(d, dtype=np.float64)
        rhs = Z.T @ y_centered
        self.w = np.linalg.solve(A, rhs)
        self.b = y_mean
        return self

    def predict_raw(self, X):
        Z = (X[:, 1:] - self.mean) / self.scale
        return self.b + Z @ self.w

    def predict(self, X):
        return np.clip(self.predict_raw(X), 1.0, 5.0)


def predict_stacked(ridge_mod, hgb_mod, eta, X):
    m0 = ridge_mod.predict_raw(X)
    if eta > 0.0 and hgb_mod is not None:
        m0 = m0 + eta * hgb_mod.predict(X[:, 1:])
    return np.clip(m0, 1.0, 5.0)


def fit_fixed_rating_combiner(X, y, config, seed=SEED_PRIMARY):
    """Train fixed Ridge and HGB on positive labels, with OOF Ridge residuals."""
    pos = y > 0
    Xp, yp = X[pos], y[pos].astype(np.float64)
    ridge = StandardizedRidge(l2=config['best_l2']).fit(Xp, yp)
    hgb = None
    if config['best_eta'] > 0:
        oof = np.empty(len(yp), dtype=np.float64)
        for fit_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=seed).split(Xp):
            fold_ridge = StandardizedRidge(l2=config['best_l2']).fit(Xp[fit_idx], yp[fit_idx])
            oof[val_idx] = fold_ridge.predict_raw(Xp[val_idx])
        hgb = HistGradientBoostingRegressor(
            **config['best_hgb_cfg'], random_state=seed
        ).fit(Xp[:, 1:], yp - oof)
    return ridge, hgb


def production_rating_task(task, a, users, items, args):
    """Generate one rating CSV with frozen settings; return recomputed audit metrics.

    Base training excludes all 50,000 validation rows. A combiner fitted on the
    first 20,000 is evaluated on the final 15,000; no selection is performed here.
    Then the combiner uses all 50,000, and base models are refitted on all rows.
    The audit is conditional-positive MSE plus all-rating sensitivity MSE.
    """
    if task not in RATING_CONFIGS:
        raise ValueError('Rating task must be 1 or 3')
    if len(a) <= 50000:
        raise ValueError('The fixed rating protocol requires more than 50,000 training rows')
    started = time.time()
    config = RATING_CONFIGS[task]
    nu, ni = len(users), len(items)
    order = np.random.default_rng(SEED_PRIMARY).permutation(len(a))
    valid, fit = a[order[:50000]], a[order[50000:]]
    base.log(f'Task {task}: fixed positive-conditional rating pipeline; fitting validation base models')
    state = fit_rating_base(fit, nu, ni, task, seed=SEED_PRIMARY,
                            fixed_epochs=config['best_epochs'], run_svdpp_grid=False)
    X, groups = build_rating_features(state, valid[:, 0], valid[:, 1], valid[:, 2])
    X = X[:, groups[config['best_stage_name']]]
    if not np.all(np.isfinite(X)):
        raise ValueError('Non-finite validation rating features')
    audit_ridge, audit_hgb = fit_fixed_rating_combiner(X[:20000], valid[:20000, 3], config)
    audit_pred = predict_stacked(audit_ridge, audit_hgb, config['best_eta'], X[35000:])
    audit_y = valid[35000:, 3]
    positive = audit_y > 0
    metrics = {
        'model': 'positive_conditional_svdpp_knn_ridge_hgb',
        'target': 'rating conditional on rating > 0; zero rows retained as implicit interactions',
        'seed': SEED_PRIMARY,
        'base_fit_rows': int(len(fit)),
        'base_positive_rows': int(len(state['a_rated'])),
        'meta_fit_rows': 20000,
        'audit_rows': len(audit_y),
        'audit_positive_rows': int(positive.sum()),
        'audit_positive_mse': float(np.mean((audit_pred[positive] - audit_y[positive]) ** 2)),
        'audit_all_rating_mse': float(np.mean((audit_pred - audit_y) ** 2)),
        'n_features': int(X.shape[1]),
        'config': config,
        'svdpp_alpha_meaning': 'exponent of user-block gradient normalizer, not implicit contribution',
        'svdpp_implicit_y_l2_norms': {
            'rank32': float(np.linalg.norm(state['svdpp32'].y)),
            'rank64': float(np.linalg.norm(state['svdpp64'].y)),
        },
    }
    base.log(f'Task {task}: audit positive MSE={metrics["audit_positive_mse"]:.6f}; '
             f'all-rating sensitivity MSE={metrics["audit_all_rating_mse"]:.6f}')
    ridge, hgb = fit_fixed_rating_combiner(X, valid[:, 3], config)
    del X, state, audit_ridge, audit_hgb
    base.log(f'Task {task}: refitting full rating base models')
    full_state = fit_rating_base(a, nu, ni, task, seed=SEED_PRIMARY,
                                 fixed_epochs=config['best_epochs'], run_svdpp_grid=False)
    test_path = Path(args.data_dir) / f'task_{task}_test.csv'
    tests = base.read_test(test_path)
    q_u = np.asarray([users.get(row['user'], nu) for row in tests], dtype=np.int64)
    q_i = np.asarray([items.get(row['item'], ni) for row in tests], dtype=np.int64)
    q_t = (np.asarray([int(row['timestamp']) for row in tests], dtype=np.int64)
           if task == 3 else np.zeros(len(tests), dtype=np.int64))
    X_test, test_groups = build_rating_features(full_state, q_u, q_i, q_t)
    if not np.all(np.isfinite(X_test)):
        raise ValueError('Non-finite test rating features')
    pred = predict_stacked(ridge, hgb, config['best_eta'],
                           X_test[:, test_groups[config['best_stage_name']]])
    if len(pred) != len(tests) or not np.all(np.isfinite(pred)) or np.any((pred < 1) | (pred > 5)):
        raise ValueError('Invalid rating predictions')
    out_path = Path(args.output_dir) / f'task_{task}_predictions.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.write_predictions(test_path, out_path, pred)
    metrics.update({
        'test_rows': int(len(pred)),
        'test_users_without_positive_history': int(np.sum(np.append(full_state['u_pos_cnt'], 0)[q_u] == 0)),
        'test_items_without_positive_history': int(np.sum(np.append(full_state['i_pos_cnt'], 0)[q_i] == 0)),
        'prediction_min': float(pred.min()),
        'prediction_max': float(pred.max()),
        'elapsed_seconds': round(time.time() - started, 2),
    })
    return metrics
