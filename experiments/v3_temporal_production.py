"""Joint temporal random-effects feature bank, positive labels only.

Frozen six-model development configuration. No cache, test labels, or external
files are needed. Meta-training/selection is the caller's responsibility.
"""
import numpy as np
from numba import njit

DAY = 86400.0

@njit(cache=True)
def categorical_sweep(y, prediction, ids, effects, counts, penalty):
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
def slope_sweep(y, prediction, users, z, slopes, counts_z2, penalty):
    sums = np.zeros(len(slopes))
    for r in range(len(y)):
        u = users[r]
        sums[u] += z[r] * (y[r] - prediction[r] + slopes[u] * z[r])
    newer = sums / (counts_z2 + penalty)
    for r in range(len(y)):
        u = users[r]
        prediction[r] += (newer[u] - slopes[u]) * z[r]
    return newer

def time_code(ids, times, days):
    bins = np.clip(np.floor((times - 946684800) / (DAY * days)), -10000, 10000).astype(np.int64)
    return ids.astype(np.int64) * 20001 + bins + 10000

def make_group(ids, times, days):
    keys, inverse = np.unique(time_code(ids, times, days), return_inverse=True)
    return (keys, inverse.astype(np.int32), np.bincount(inverse).astype(np.float64))

DEFAULT_CONFIGS = [{'user_days': 1.0, 'item_days': 90.0, 'session_reg': 1.0, 'item_reg': 20.0, 'slope_reg': 20.0}, {'user_days': 1.0, 'item_days': 30.0, 'session_reg': 4.0, 'item_reg': 10.0, 'slope_reg': 80.0}, {'user_days': 7.0, 'item_days': 90.0, 'session_reg': 2.0, 'item_reg': 20.0, 'slope_reg': 20.0}, {'user_days': 30.0, 'item_days': 365.0, 'session_reg': 4.0, 'item_reg': 30.0, 'slope_reg': 80.0}, {'user_days': 90.0, 'item_days': 90.0, 'session_reg': 8.0, 'item_reg': 10.0, 'slope_reg': 20.0}, {'user_days': 365.0, 'item_days': 365.0, 'session_reg': 12.0, 'item_reg': 30.0, 'slope_reg': 80.0}]

def fit_temporal_models(fit_rows, nu, ni, configs=None):
    a = fit_rows[fit_rows[:, 3] > 0]
    u, i, t, y = (a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(np.float64))
    uc = np.bincount(u, minlength=nu).astype(np.float64)
    ic = np.bincount(i, minlength=ni).astype(np.float64)
    tc = np.bincount(u, weights=t.astype(float), minlength=nu) / np.maximum(uc, 1)
    raw = np.sign((t - tc[u]) / DAY) * np.abs((t - tc[u]) / DAY) ** 0.4
    zc = np.bincount(u, weights=raw, minlength=nu) / np.maximum(uc, 1)
    z = raw - zc[u]
    z2 = np.bincount(u, weights=z * z, minlength=nu)
    mu = float(y.mean())
    states = []
    for cfg in configs or DEFAULT_CONFIGS:
        keys_u, gu, cgu = make_group(u, t, cfg['user_days'])
        keys_i, gi, cgi = make_group(i, t, cfg['item_days'])
        keys_long, gl, cgl = make_group(u, t, 365.0)
        bu, bi, slope = (np.zeros(nu), np.zeros(ni), np.zeros(nu))
        eu, ei, el = (np.zeros(len(keys_u)), np.zeros(len(keys_i)), np.zeros(len(keys_long)))
        pred = np.full(len(y), mu)
        for _ in range(25):
            bu = categorical_sweep(y, pred, u, bu, uc, 5.0)
            bi = categorical_sweep(y, pred, i, bi, ic, 8.0)
            slope = slope_sweep(y, pred, u, z, slope, z2, cfg['slope_reg'])
            el = categorical_sweep(y, pred, gl, el, cgl, 15.0)
            ei = categorical_sweep(y, pred, gi, ei, cgi, cfg['item_reg'])
            eu = categorical_sweep(y, pred, gu, eu, cgu, cfg['session_reg'])
        states.append(dict(config=cfg, mu=mu, bu=bu, bi=bi, tc=tc, zc=zc, slope=slope, keys_u=keys_u, effect_u=eu, count_u=cgu, keys_i=keys_i, effect_i=ei, count_i=cgi, keys_long=keys_long, effect_long=el, count_long=cgl))
    return dict(nu=nu, ni=ni, models=states)

def lookup_group(keys, effects, counts, query_keys):
    where = np.searchsorted(keys, query_keys)
    good = where < len(keys)
    safe = np.minimum(where, len(keys) - 1)
    good &= keys[safe] == query_keys
    return (np.where(good, effects[safe], 0.0), np.where(good, counts[safe], 0.0))

def temporal_features(state, qu, qi, qt):
    nu, ni = (state['nu'], state['ni'])
    cols = []
    for s in state['models']:
        cfg = s['config']
        bu = np.r_[s['bu'], 0.0][qu]
        bi = np.r_[s['bi'], 0.0][qi]
        tc = np.r_[s['tc'], 0.0][qu]
        zc = np.r_[s['zc'], 0.0][qu]
        dt = (qt - tc) / DAY
        z = np.clip(np.sign(dt) * np.abs(dt) ** 0.4 - zc, -30, 30)
        trend = np.r_[s['slope'], 0.0][qu] * z
        eu, cu = lookup_group(s['keys_u'], s['effect_u'], s['count_u'], time_code(qu, qt, cfg['user_days']))
        ei, ci = lookup_group(s['keys_i'], s['effect_i'], s['count_i'], time_code(qi, qt, cfg['item_days']))
        el, cl = lookup_group(s['keys_long'], s['effect_long'], s['count_long'], time_code(qu, qt, 365.0))
        pred = s['mu'] + bu + bi + trend + eu + ei + el
        cols.extend([pred, eu, ei, el, trend, np.log1p(cu), np.log1p(ci), eu * np.log1p(cu), ei * np.log1p(ci), eu * (cu / (cu + 2.0))])
    return np.column_stack(cols)

def fit_features(fit_rows, nu, ni):
    return fit_temporal_models(fit_rows, nu, ni)


def extract_features(state, query_users, query_items, query_times):
    return temporal_features(state, query_users, query_items, query_times)
