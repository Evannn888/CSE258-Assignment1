"""V3: larger meta-training set and complementary conditional-rating features."""
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import assignment1 as base


V3_CONFIGS = {1: {'kind': 'ridge_hgb', 'reg': 500.0, 'eta': 1.0, 'hgb': {'max_leaf_nodes': 31, 'max_iter': 400, 'learning_rate': 0.04, 'min_samples_leaf': 250, 'l2_regularization': 100.0, 'early_stopping': False, 'random_state': 258}, 'tune_mse': 0.5306373790895934, 'use_stage': True}, 3: {'kind': 'ridge_hgb', 'reg': 500.0, 'eta': 0.5, 'hgb': {'max_leaf_nodes': 63, 'max_iter': 400, 'learning_rate': 0.04, 'min_samples_leaf': 250, 'l2_regularization': 100.0, 'early_stopping': False, 'random_state': 258}, 'tune_mse': 0.6600240009778012}}

@njit(cache=True)
def nk__summarize_neighbors(sim, values, adjusted):
    result = np.zeros(42)
    col = 0
    for variant in range(3):
        for positive in range(2):
            w = sim[variant].copy()
            if positive:
                for j in range(len(w)):
                    if w[j] < 0:
                        w[j] = 0.0
            order = np.argsort(-np.abs(w))
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

@njit(cache=True)
def nk_topk_residual_features(qu, qi, up, ui, ur, ip, iu, ir, ue, ie):
    nu = len(up) - 1
    ni = len(ip) - 1
    out = np.zeros((len(qu), 84))
    present = np.zeros(nu, np.bool_)
    target = np.zeros(nu)
    dot = np.zeros(nu)
    sx = np.zeros(nu)
    sy = np.zeros(nu)
    xx = np.zeros(nu)
    yy = np.zeros(nu)
    count = np.zeros(nu)
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
        sims = np.zeros((3, n_h))
        values = np.zeros(n_h)
        adjusted = np.zeros(n_h)
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
                sims[0, n] = ds / (np.sqrt(ie[i] * ie[j]) + 1e-08) * nc / (nc + 5)
                covariance = ds - xs * ys / nc
                variance = max(0.0, x2 - xs * xs / nc) * max(0.0, y2 - ys * ys / nc)
                sims[1, n] = covariance / (np.sqrt(variance) + 1e-08) * nc / (nc + 10)
                sims[2, n] = np.exp(-(x2 + y2 - 2 * ds) / (nc * 0.5)) * nc / (nc + 10)
                adjusted[n] = ru + (xs - ys) / (nc + 5)
        out[z, :42] = nk__summarize_neighbors(sims, values, adjusted)
        nr = iep - is_
        sims = np.zeros((3, nr))
        values = np.zeros(nr)
        adjusted = np.zeros(nr)
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
                sims[0, n] = ds / (np.sqrt(ue[u] * ue[v]) + 1e-08) * nc / (nc + 5)
                covariance = ds - xs * ys / nc
                variance = max(0.0, x2 - xs * xs / nc) * max(0.0, y2 - ys * ys / nc)
                sims[1, n] = covariance / (np.sqrt(variance) + 1e-08) * nc / (nc + 10)
                sims[2, n] = np.exp(-(x2 + y2 - 2 * ds) / (nc * 0.5)) * nc / (nc + 10)
                adjusted[n] = target[v] + (xs - ys) / (nc + 5)
            present[v] = False
            target[v] = 0.0
            dot[v] = 0.0
            sx[v] = 0.0
            sy[v] = 0.0
            xx[v] = 0.0
            yy[v] = 0.0
            count[v] = 0.0
        out[z, 42:] = nk__summarize_neighbors(sims, values, adjusted)
    return out

def nk_extract_topk_features(state, qu, qi):
    a = state['a_rated']
    r = state['res']
    ue = np.bincount(a[:, 0], weights=r * r, minlength=state['nu'])
    ie = np.bincount(a[:, 1], weights=r * r, minlength=state['ni'])
    return nk_topk_residual_features(qu, qi, state['u_pos_ptr'], state['u_pos_items'], state['u_pos_res'], state['i_pos_ptr'], state['i_pos_users'], state['i_pos_res'], ue, ie)

def ord_fit_ordinal(fit, nu, ni, reg_u=3.0, reg_i=5.0, iterations=20):
    a = fit[fit[:, 3] > 0]
    uu, ii = (a[:, 0], a[:, 1])
    uc = np.bincount(uu, minlength=nu)
    ic = np.bincount(ii, minlength=ni)
    mu = np.zeros(4)
    bu = np.zeros((nu, 4))
    bi = np.zeros((ni, 4))
    for k in range(4):
        target = (a[:, 3] >= k + 2).astype(float)
        prior = target.mean()
        mu[k] = np.log(prior / (1 - prior))
        for it in range(iterations):
            prob = expit(mu[k] + bu[uu, k] + bi[ii, k])
            h = prob * (1 - prob)
            gu = np.bincount(uu, weights=target - prob, minlength=nu) - reg_u * bu[:, k]
            hu = np.bincount(uu, weights=h, minlength=nu) + reg_u
            bu[:, k] += np.clip(gu / hu, -1.0, 1.0)
            prob = expit(mu[k] + bu[uu, k] + bi[ii, k])
            h = prob * (1 - prob)
            gi = np.bincount(ii, weights=target - prob, minlength=ni) - reg_i * bi[:, k]
            hi = np.bincount(ii, weights=h, minlength=ni) + reg_i
            bi[:, k] += np.clip(gi / hi, -1.0, 1.0)
    return dict(mu=mu, bu=bu, bi=bi)

def ord_ordinal_features(state, qu, qi):
    bu = np.vstack([state['bu'], np.zeros((1, 4))])[qu]
    bi = np.vstack([state['bi'], np.zeros((1, 4))])[qi]
    prob = expit(state['mu'][None, :] + bu + bi)
    mean = 1 + prob.sum(axis=1)
    return np.column_stack([mean, prob, bu, bi])

jt_DAY = 86400.0

@njit(cache=True)
def jt_categorical_sweep(y, prediction, ids, effects, counts, penalty):
    sums = np.zeros(len(effects))
    for r in range(len(y)):
        k = ids[r]
        sums[k] += y[r] - prediction[r] + effects[k]
    newer = sums / (counts + penalty)
    for r in range(len(y)):
        k = ids[r]
        prediction[r] += newer[k] - effects[k]
    return newer

@njit(cache=True)
def jt_slope_sweep(y, prediction, users, z, slopes, counts_z2, penalty):
    sums = np.zeros(len(slopes))
    for r in range(len(y)):
        u = users[r]
        sums[u] += z[r] * (y[r] - prediction[r] + slopes[u] * z[r])
    newer = sums / (counts_z2 + penalty)
    for r in range(len(y)):
        u = users[r]
        prediction[r] += (newer[u] - slopes[u]) * z[r]
    return newer

def jt_time_code(ids, times, days):
    bins = np.clip(np.floor((times - 946684800) / (jt_DAY * days)), -10000, 10000).astype(np.int64)
    return ids.astype(np.int64) * 20001 + bins + 10000

def jt_make_group(ids, times, days):
    keys, inverse = np.unique(jt_time_code(ids, times, days), return_inverse=True)
    return (keys, inverse.astype(np.int32), np.bincount(inverse).astype(np.float64))
jt_DEFAULT_CONFIGS = [{'user_days': 1.0, 'item_days': 90.0, 'session_reg': 1.0, 'item_reg': 20.0, 'slope_reg': 20.0}, {'user_days': 1.0, 'item_days': 30.0, 'session_reg': 4.0, 'item_reg': 10.0, 'slope_reg': 80.0}, {'user_days': 7.0, 'item_days': 90.0, 'session_reg': 2.0, 'item_reg': 20.0, 'slope_reg': 20.0}, {'user_days': 30.0, 'item_days': 365.0, 'session_reg': 4.0, 'item_reg': 30.0, 'slope_reg': 80.0}, {'user_days': 90.0, 'item_days': 90.0, 'session_reg': 8.0, 'item_reg': 10.0, 'slope_reg': 20.0}, {'user_days': 365.0, 'item_days': 365.0, 'session_reg': 12.0, 'item_reg': 30.0, 'slope_reg': 80.0}]

def jt_fit_temporal_models(fit_rows, nu, ni, configs=None):
    a = fit_rows[fit_rows[:, 3] > 0]
    u, i, t, y = (a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(np.float64))
    uc = np.bincount(u, minlength=nu).astype(np.float64)
    ic = np.bincount(i, minlength=ni).astype(np.float64)
    tc = np.bincount(u, weights=t.astype(float), minlength=nu) / np.maximum(uc, 1)
    raw = np.sign((t - tc[u]) / jt_DAY) * np.abs((t - tc[u]) / jt_DAY) ** 0.4
    zc = np.bincount(u, weights=raw, minlength=nu) / np.maximum(uc, 1)
    z = raw - zc[u]
    z2 = np.bincount(u, weights=z * z, minlength=nu)
    mu = float(y.mean())
    states = []
    for cfg in configs or jt_DEFAULT_CONFIGS:
        keys_u, gu, cgu = jt_make_group(u, t, cfg['user_days'])
        keys_i, gi, cgi = jt_make_group(i, t, cfg['item_days'])
        keys_long, gl, cgl = jt_make_group(u, t, 365.0)
        bu, bi, slope = (np.zeros(nu), np.zeros(ni), np.zeros(nu))
        eu, ei, el = (np.zeros(len(keys_u)), np.zeros(len(keys_i)), np.zeros(len(keys_long)))
        pred = np.full(len(y), mu)
        for _ in range(25):
            bu = jt_categorical_sweep(y, pred, u, bu, uc, 5.0)
            bi = jt_categorical_sweep(y, pred, i, bi, ic, 8.0)
            slope = jt_slope_sweep(y, pred, u, z, slope, z2, cfg['slope_reg'])
            el = jt_categorical_sweep(y, pred, gl, el, cgl, 15.0)
            ei = jt_categorical_sweep(y, pred, gi, ei, cgi, cfg['item_reg'])
            eu = jt_categorical_sweep(y, pred, gu, eu, cgu, cfg['session_reg'])
        states.append(dict(config=cfg, mu=mu, bu=bu, bi=bi, tc=tc, zc=zc, slope=slope, keys_u=keys_u, effect_u=eu, count_u=cgu, keys_i=keys_i, effect_i=ei, count_i=cgi, keys_long=keys_long, effect_long=el, count_long=cgl))
    return dict(nu=nu, ni=ni, models=states)

def jt_lookup_group(keys, effects, counts, query_keys):
    where = np.searchsorted(keys, query_keys)
    good = where < len(keys)
    safe = np.minimum(where, len(keys) - 1)
    good &= keys[safe] == query_keys
    return (np.where(good, effects[safe], 0.0), np.where(good, counts[safe], 0.0))

def jt_temporal_features(state, qu, qi, qt):
    nu, ni = (state['nu'], state['ni'])
    cols = []
    for s in state['models']:
        cfg = s['config']
        bu = np.r_[s['bu'], 0.0][qu]
        bi = np.r_[s['bi'], 0.0][qi]
        tc = np.r_[s['tc'], 0.0][qu]
        zc = np.r_[s['zc'], 0.0][qu]
        dt = (qt - tc) / jt_DAY
        z = np.clip(np.sign(dt) * np.abs(dt) ** 0.4 - zc, -30, 30)
        trend = np.r_[s['slope'], 0.0][qu] * z
        eu, cu = jt_lookup_group(s['keys_u'], s['effect_u'], s['count_u'], jt_time_code(qu, qt, cfg['user_days']))
        ei, ci = jt_lookup_group(s['keys_i'], s['effect_i'], s['count_i'], jt_time_code(qi, qt, cfg['item_days']))
        el, cl = jt_lookup_group(s['keys_long'], s['effect_long'], s['count_long'], jt_time_code(qu, qt, 365.0))
        pred = s['mu'] + bu + bi + trend + eu + ei + el
        cols.extend([pred, eu, ei, el, trend, np.log1p(cu), np.log1p(ci), eu * np.log1p(cu), ei * np.log1p(ci), eu * (cu / (cu + 2.0))])
    return np.column_stack(cols)

def jt_fit_features(fit_rows, nu, ni):
    return jt_fit_temporal_models(fit_rows, nu, ni)

def jt_extract_features(state, query_users, query_items, query_times):
    return jt_temporal_features(state, query_users, query_items, query_times)

@njit(cache=True)
def lf_mf_epoch(arr, order, p, q, bu, bi, mu, lr, rp, rq, rb, scales):
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

@njit(cache=True)
def lf_fast_svdpp_epoch(uperm, ptr, its, rats, iptr, iits, p, q, y, bu, bi, mu, lr, rp, rq, ry, rb):
    k = p.shape[1]
    h = np.empty(k, np.float64)
    h0 = np.empty(k, np.float64)
    for u in uperm:
        s, e = (ptr[u], ptr[u + 1])
        ns, ne = (iptr[u], iptr[u + 1])
        if e <= s or ne <= ns:
            continue
        norm = 1.0 / np.sqrt(ne - ns)
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

@njit(cache=True)
def lf_implicit_vectors(ptr, its, y):
    out = np.zeros((len(ptr) - 1, y.shape[1]), np.float32)
    for u in range(len(out)):
        s, e = (ptr[u], ptr[u + 1])
        if e > s:
            norm = 1.0 / np.sqrt(e - s)
            for ix in range(s, e):
                for d in range(y.shape[1]):
                    out[u, d] += y[its[ix], d] * norm
    return out

class lf_Model:

    def __init__(self, p, q, bu, bi, mu, scales, iptr=None, iits=None, y=None):
        self.p, self.q, self.bu, self.bi, self.mu, self.scales = (p, q, bu, bi, mu, scales)
        self.iptr, self.iits, self.y = (iptr, iits, y)

    def predict(self, u, i, raw=False):
        uv = u < len(self.p)
        iv = i < len(self.q)
        uu = np.minimum(u, len(self.p) - 1)
        ii = np.minimum(i, len(self.q) - 1)
        pp = self.p if self.y is None else self.p + lf_implicit_vectors(self.iptr, self.iits, self.y)
        dot = np.sum(pp[uu] * self.q[ii], axis=1) * self.scales[uu]
        pred = self.mu + np.where(uv, self.bu[uu], 0) + np.where(iv, self.bi[ii], 0) + np.where(uv & iv, dot, 0)
        return pred if raw else np.clip(pred, 1, 5)

    def state(self):
        return dict(p=self.p, q=self.q, bu=self.bu, bi=self.bi, mu=self.mu, scales=self.scales, iptr=self.iptr, iits=self.iits, y=self.y)

def lf_train(fit, nu, ni, valid, config, seed=258, fixed_epochs=None):
    pos = fit[fit[:, 3] > 0]
    rng = np.random.default_rng(seed)
    uc = np.bincount(pos[:, 0], minlength=nu)
    ic = np.bincount(pos[:, 1], minlength=ni)
    mu, bu, bi = base.als_biases_custom(pos, nu, ni, uc, ic, 3.0, 5.0, 20)
    p = rng.normal(0, config['init'], (nu, config['k'])).astype(np.float32)
    q = rng.normal(0, config['init'], (ni, config['k'])).astype(np.float32)
    scales = np.ones(nu, np.float32)
    if config.get('scale', False):
        resid = pos[:, 3] - mu - bu[pos[:, 0]] - bi[pos[:, 1]]
        vv = np.bincount(pos[:, 0], weights=resid ** 2, minlength=nu)
        scales = np.sqrt((vv + 10 * 0.55) / (uc + 10)).astype(np.float32)
    model = lf_Model(p, q, bu.astype(np.float32), bi.astype(np.float32), mu, scales)
    if config['kind'] == 'svdpp':
        order = np.argsort(pos[:, 0], kind='stable')
        ptr = np.r_[0, uc.cumsum()]
        its = pos[order, 1]
        rats = pos[order, 3]
        iu = np.bincount(fit[:, 0], minlength=nu)
        io = np.argsort(fit[:, 0], kind='stable')
        model.iptr = np.r_[0, iu.cumsum()]
        model.iits = fit[io, 1]
        model.y = np.zeros((ni, config['k']), np.float32)
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
            lf_mf_epoch(pos, rng.permutation(len(pos)), p, q, model.bu, model.bi, mu, lr, config['reg'], config['reg'], config.get('reg_b', 0.02), scales)
        else:
            lf_fast_svdpp_epoch(rng.permutation(nu), ptr, its, rats, model.iptr, model.iits, p, q, model.y, model.bu, model.bi, mu, lr, config['reg'], config['reg'], config['reg_y'], config.get('reg_b', 0.02))
        if tv is not None and (ep % 5 == 0 or ep == maxep):
            pred = model.predict(tv[:, 0], tv[:, 1])
            mse = float(np.mean((pred - tv[:, 3]) ** 2))
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

@njit(cache=True)
def stg__fit_stages(ptr, items, times, rating, res, emb, nu):
    k = emb.shape[1]
    bounds = np.zeros((nu, 8, 2), np.int64)
    stats = np.zeros((nu, 8, 5))
    centers = np.zeros((nu, 8, k))
    for u in range(nu):
        us, ue = (ptr[u], ptr[u + 1])
        n = ue - us
        if n == 0:
            continue
        high = np.mean(rating[us:ue] >= 4)
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
                stats[u, idx, 1] = np.sum(rr) / (nn + 3.0)
                stats[u, idx, 2] = np.median(rr) * nn / (nn + 3.0)
                stats[u, idx, 3] = (np.sum(rating[lo:hi] >= 4) - high * nn) / (nn + 3.0)
                stats[u, idx, 4] = np.sum(rr * rr) / (nn + 3.0)
                for ix in range(lo, hi):
                    for d in range(k):
                        centers[u, idx, d] += emb[items[ix], d]
                norm = np.sqrt(np.sum(centers[u, idx] ** 2))
                if norm > 1e-07:
                    centers[u, idx] /= norm
            offset += parts
    return (bounds, stats, centers)

def stg_fit_stage_model(fit, nu, ni, embeddings=None, biases=None):
    pos = fit[fit[:, 3] > 0]
    uc = np.bincount(pos[:, 0], minlength=nu)
    ic = np.bincount(pos[:, 1], minlength=ni)
    if biases is None:
        biases = base.als_biases_custom(pos, nu, ni, uc, ic, 8.0, 12.0, 20)
    mu, bu, bi = biases
    res = pos[:, 3] - mu - bu[pos[:, 0]] - bi[pos[:, 1]]
    emb = np.zeros((ni, 1), np.float32) if embeddings is None else embeddings.astype(np.float32).copy()
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = np.divide(emb, norm, out=np.zeros_like(emb), where=norm > 1e-07)
    order = np.lexsort((pos[:, 2], pos[:, 0]))
    ptr = np.r_[0, uc.cumsum()]
    bounds, stats, centers = stg__fit_stages(ptr, pos[order, 1], pos[order, 2], pos[order, 3], res[order], emb, nu)
    all_times = np.sort(fit[:, 2])
    io = np.lexsort((fit[:, 2], fit[:, 1]))
    ics = np.bincount(fit[:, 1], minlength=ni)
    return dict(bounds=bounds, stats=stats, centers=centers, emb=emb, iptr=np.r_[0, ics.cumsum()], itimes=fit[io, 2], global_times=all_times)

@njit(cache=True)
def stg_stage_features(state_bounds, stats, centers, emb, iptr, itimes, global_times, qu, qi):
    out = np.zeros((len(qu), 20))
    nu = stats.shape[0]
    ni = len(iptr) - 1
    for z in range(len(qu)):
        u, i = (qu[z], qi[z])
        if u >= nu:
            continue
        is_, ie = (iptr[i], iptr[i + 1]) if i < ni else (0, 0)
        offset = 0
        for parts in (3, 5):
            tw = np.zeros(parts)
            ew = np.zeros(parts)
            cw = np.zeros(parts)
            lift = np.zeros(parts)
            for s in range(parts):
                idx = offset + s
                prior = stats[u, idx, 0]
                if prior <= 0:
                    continue
                lo, hi = state_bounds[u, idx]
                p = max(1e-09, (np.searchsorted(global_times, hi, side='right') - np.searchsorted(global_times, lo)) / len(global_times))
                c = np.searchsorted(itimes[is_:ie], hi, side='right') - np.searchsorted(itimes[is_:ie], lo)
                ll = np.log((c + 3.0 * p) / ((ie - is_ + 3.0) * p))
                ll = max(-6.0, min(6.0, ll))
                lift[s] = ll
                cosine = 0.0
                if i < ni:
                    for d in range(emb.shape[1]):
                        cosine += emb[i, d] * centers[u, idx, d]
                tw[s] = prior * np.exp(ll)
                ew[s] = prior * np.exp(3.0 * cosine)
                cw[s] = prior * np.exp(ll + 3.0 * cosine)
            st, se, sc = (np.sum(tw), np.sum(ew), np.sum(cw))
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
            out[z, b + 7] = np.max(tw) - np.min(tw)
            offset += parts
    return out

def stg_extract_stage_features(state, qu, qi):
    return stg_stage_features(state['bounds'], state['stats'], state['centers'], state['emb'], state['iptr'], state['itimes'], state['global_times'], qu, qi)

def fit_extended(fit,nu,ni,task):
    old_config=base.ratings_RATING_CONFIGS[task]
    old_state=base.ratings_fit_rating_base(fit,nu,ni,task,seed=258,
        fixed_epochs=old_config['best_epochs'],run_svdpp_grid=False)
    regs=[(1.,2.),(3.,5.),(10.,15.)] if task==1 else [(1.,2.)]
    ordinal=[ord_fit_ordinal(fit,nu,ni,ru,ri) for ru,ri in regs]
    joint=jt_fit_features(fit,nu,ni) if task==3 else None
    cfg=dict(name='fastpp64_reg08',kind='svdpp',k=64,init=.05,reg=.08,reg_y=.02,lr=.007,decay=.99,epochs=80)
    latent,_=lf_train(fit,nu,ni,None,cfg,fixed_epochs=80)
    stage=stg_fit_stage_model(fit,nu,ni,old_state['svdpp64'].y,old_state['biases'][2]) if V3_CONFIGS[task].get('use_stage',False) else None
    return dict(old_state=old_state,ordinal=ordinal,joint=joint,latent=latent,stage=stage)


def extract_extended(state,qu,qi,qt):
    x,groups=base.ratings_build_rating_features(state['old_state'],qu,qi,qt)
    x=x[:,groups['stage5_plus_conf_gating']]
    blocks=[x,nk_extract_topk_features(state['old_state'],qu,qi)]
    blocks.extend(ord_ordinal_features(s,qu,qi) for s in state['ordinal'])
    if state['joint'] is not None:blocks.append(jt_extract_features(state['joint'],qu,qi,qt))
    blocks.append(state['latent'].predict(qu,qi,raw=True)[:,None])
    if state.get('stage') is not None:blocks.append(stg_extract_stage_features(state['stage'],qu,qi))
    out=np.column_stack(blocks).astype(np.float32)
    if not np.all(np.isfinite(out)):raise ValueError('Nonfinite v3 rating features')
    return out,int(x.shape[1])


def fit_meta(X,y,config):
    mask=y>0;xp=X[mask].astype(np.float64);yp=y[mask].astype(np.float64)
    ridge=base.ratings_StandardizedRidge(config['reg']).fit(xp,yp)
    hgb=None
    if config['eta']:
        oof=np.empty(len(yp))
        for tr,va in KFold(5,shuffle=True,random_state=258).split(xp):
            rr=base.ratings_StandardizedRidge(config['reg']).fit(xp[tr],yp[tr])
            oof[va]=rr.predict_raw(xp[va])
        hgb=HistGradientBoostingRegressor(**config['hgb']).fit(xp[:,1:],yp-oof)
    return ridge,hgb


def predict_meta(bundle,X,config):
    ridge,hgb=bundle
    pred=ridge.predict_raw(X.astype(np.float64))
    if hgb is not None:pred+=config['eta']*hgb.predict(X[:,1:])
    return np.clip(pred,1,5)


def production_rating_task(task,a,users,items,args):
    """Frozen v3 settings: 200k base holdout, 150k meta-fit, 25k audit."""
    start=time.time();nu,ni=len(users),len(items);config=V3_CONFIGS[task]
    order=np.random.default_rng(258).permutation(len(a))
    valid,fit=a[order[:200000]],a[order[200000:]]
    base.log(f'Task {task}: fitting v3 validation base and ordinal/neighbor/latent feature bank')
    state=fit_extended(fit,nu,ni,task)
    X,width=extract_extended(state,valid[:,0],valid[:,1],valid[:,2]);y=valid[:,3]
    audited=fit_meta(X[:150000],y[:150000],config)
    pred=predict_meta(audited,X[175000:],config);truth=y[175000:];mask=truth>0
    old_config=base.ratings_RATING_CONFIGS[task]
    br,bh=base.ratings_fit_fixed_rating_combiner(X[:20000,:width].astype(np.float64),y[:20000],old_config)
    baseline=base.ratings_predict_stacked(br,bh,old_config['best_eta'],X[175000:,:width].astype(np.float64))
    difference=(pred[mask]-truth[mask])**2-(baseline[mask]-truth[mask])**2
    metrics=dict(model='v3_large_meta_ordinal_neighborhood_fast_svdpp',seed=258,
        audit_positive_mse=float(np.mean((pred[mask]-truth[mask])**2)),
        audit_baseline20k_positive_mse=float(np.mean((baseline[mask]-truth[mask])**2)),
        audit_positive_rows=int(mask.sum()),meta_fit_rows=150000,base_holdout_rows=200000,
        audit_paired_mse_change=float(difference.mean()),audit_paired_standard_error=float(difference.std(ddof=1)/np.sqrt(len(difference))),
        fixed_config=config,n_features=int(X.shape[1]),
        caveat='Historical internal validation only; not a leaderboard estimate')
    base.log(f'Task {task}: frozen-config audit MSE={metrics["audit_positive_mse"]:.6f}, baseline20k={metrics["audit_baseline20k_positive_mse"]:.6f}')
    final_meta=fit_meta(X,y,config)
    del state,X,audited,br,bh
    base.log(f'Task {task}: refitting full v3 feature bank')
    state=fit_extended(a,nu,ni,task)
    tests=base.read_test(args.data_dir/f'task_{task}_test.csv')
    qu=np.array([users.get(r['user'],nu) for r in tests],dtype=np.int64)
    qi=np.array([items.get(r['item'],ni) for r in tests],dtype=np.int64)
    qt=np.array([int(r['timestamp']) for r in tests],dtype=np.int64) if task==3 else np.zeros(len(tests),dtype=np.int64)
    Xt,_=extract_extended(state,qu,qi,qt)
    predictions=predict_meta(final_meta,Xt,config)
    base.write_predictions(args.data_dir/f'task_{task}_test.csv',args.output_dir/f'task_{task}_predictions.csv',predictions)
    metrics.update(test_rows=len(tests),elapsed_seconds=round(time.time()-start,2),
                   prediction_min=float(predictions.min()),prediction_max=float(predictions.max()))
    return metrics

