"""Task 6 experiments: supervised conditional-median waiting-time regression.

Only task_6_train.csv supplies labels/features. The evaluation labels are never
used to construct feature histories or tune model settings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
import assignment1 as base

DAY = 86400.
ROOT = Path(__file__).resolve().parents[1]


def extra_features(histories, item_ts, query_u, query_i):
    feats = []
    quantiles = [0, .02, .1, .25, .5, .75, .9, .98, 1]
    for u, i in zip(query_u, query_i):
        raw = np.asarray([x[1] for x in histories[u]], dtype=float)
        tp = raw[-1]
        clean = raw[(raw >= base.T_MIN_VALID) & (raw <= base.T_MAX_VALID)]
        if len(clean) < 2:
            clean = raw
        gaps = np.maximum(0, np.diff(clean)) / DAY
        if len(gaps) == 0:
            gaps = np.array([0.])
        f = [len(raw), len(clean), tp / DAY, raw[0] / DAY,
             (tp - raw[0]) / DAY, max(0., base.T_MAX_VALID - tp) / DAY,
             np.mean(gaps), np.std(gaps), np.mean(np.log1p(gaps)),
             np.mean(gaps < 1), np.mean(gaps < 7), np.mean(gaps < 30)]
        f.extend(np.quantile(gaps, quantiles))
        for k in [1, 2, 3, 5, 10, 20]:
            g = gaps[-k:]
            f.extend([np.mean(g), np.median(g), np.std(g), np.mean(np.log1p(g))])
        for d in [7, 30, 90, 365, 730]:
            f.append(np.sum(raw >= tp - DAY * d))
        its = np.asarray(item_ts[i] if i < len(item_ts) else [], dtype=float)
        f.append(len(its))
        if len(its):
            its = (its - tp) / DAY
            f.extend(np.quantile(its, quantiles))
            future = its[its >= 0]
            f.append(len(future) / len(its))
            f.extend(np.quantile(future, quantiles) if len(future) else [np.nan] * len(quantiles))
            for d in [7, 30, 90, 365, 730]:
                f.extend([np.mean((its >= 0) & (its <= d)), np.mean((its <= 0) & (its >= -d))])
        else:
            f.extend([np.nan] * (2 * len(quantiles) + 11))
        feats.append(f)
    return np.asarray(feats, dtype=np.float64)


def structures(histories, ni):
    item_ts = [[] for _ in range(ni)]
    for h in histories:
        for i, t in h:
            if base.T_MIN_VALID <= t <= base.T_MAX_VALID:
                item_ts[i].append(t)
    for ts in item_ts:
        ts.sort()
    return item_ts


def prepare():
    cache = ROOT / 'experiments/delta_features.npz'
    if cache.exists():
        return dict(np.load(cache))
    a, users, items = base.load(ROOT / 'assignment1_2026/task_6_train.csv')
    histories = [[] for _ in range(len(users))]
    for u, i, t, _ in a:
        histories[u].append((int(i), int(t)))
    hist = [h[:-1] for h in histories]
    qi = np.array([h[-1][0] for h in histories])
    qu = np.arange(len(hist))
    truth = np.array([h[-1][1] - h[-2][1] for h in histories]) / DAY
    item_ts = structures(hist, len(items))
    vf = base.extract_delta_features(hist, item_ts, qu, qi)
    x = extra_features(hist, item_ts, qu, qi)
    base.log(f'Validation feature shape {x.shape}')
    tests = base.read_test(ROOT / 'assignment1_2026/task_6_test.csv')
    tu = np.array([users[r['user']] for r in tests])
    ti = np.array([items.get(r['item'], len(items)) for r in tests])
    full_item_ts = structures(histories, len(items))
    tf = base.extract_delta_features(histories, full_item_ts, tu, ti)
    tx = extra_features(histories, full_item_ts, tu, ti)
    data = dict(x=x, y=truth, tx=tx, vf=np.asarray(vf), tf=np.asarray(tf))
    np.savez_compressed(cache, **data)
    return data


def main():
    data = prepare()
    x, y, tx = data['x'], data['y'], data['tx']
    rng = np.random.default_rng(258)
    ix = rng.permutation(len(y))
    tr, tune, audit = np.split(ix, [int(.6 * len(y)), int(.8 * len(y))])
    init = np.array([0.13, .05, .08, .05, .12, .04, .05, .50, .65, .35, .26, .45, .35, .70])
    vf = (data['vf'][0].astype(bool), *data['vf'][1:])
    opt = minimize(lambda p: np.mean(abs(base.apply_delta_params(tuple(f[tr] for f in vf), p) / DAY - y[tr])), init,
                   method='Nelder-Mead', options={'maxiter': 800})
    baseline = base.apply_delta_params(vf, opt.x) / DAY
    test_baseline = base.apply_delta_params((data['tf'][0].astype(bool), *data['tf'][1:]), opt.x) / DAY
    x = np.column_stack([x, baseline, data['vf'].T / DAY])
    tx = np.column_stack([tx, test_baseline, data['tf'].T / DAY])
    records = []
    best = (float('inf'), None, None)
    base.log('Target days quantiles '+str(np.quantile(y, [0, .1, .5, .9, .99, .999, 1])))
    base.log('Baseline tune %.2f audit %.2f seconds' % (np.mean(abs(baseline[tune]-y[tune]))*DAY, np.mean(abs(baseline[audit]-y[audit]))*DAY))
    for leaves, l2, iters, lr in [(7, 20, 180, .05), (15, 20, 220, .05), (31, 30, 250, .05), (15, 100, 450, .03), (31, 100, 500, .03), (63, 200, 350, .04)]:
        params = dict(loss='absolute_error', max_leaf_nodes=leaves, l2_regularization=l2,
                      max_iter=iters, learning_rate=lr, min_samples_leaf=80, early_stopping=False, random_state=258)
        model = HistGradientBoostingRegressor(**params).fit(x[tr], y[tr])
        p = np.maximum(0, model.predict(x[tune]))
        mae = float(np.mean(abs(p-y[tune]))*DAY)
        rec = dict(params=params, tune_mae_seconds=mae)
        records.append(rec)
        base.log(str(rec))
        if mae < best[0]:
            best = (mae, params, model)
    params, model = best[1:]
    tune_pred = np.maximum(0, model.predict(x[tune]))
    alphas = np.linspace(0, 1, 21)
    alpha = alphas[np.argmin([np.mean(abs(alpha * tune_pred + (1-alpha)*baseline[tune]-y[tune])) for alpha in alphas])]
    pred = np.maximum(0, model.predict(x[audit]))
    result = dict(experiments=records, selected_params=params, blend_model_weight=float(alpha),
                  audit_model_mae_seconds=float(np.mean(abs(pred-y[audit]))*DAY),
                  audit_blend_mae_seconds=float(np.mean(abs(alpha*pred+(1-alpha)*baseline[audit]-y[audit]))*DAY),
                  audit_baseline_mae_seconds=float(np.mean(abs(baseline[audit]-y[audit]))*DAY),
                  train_users=len(tr), tune_users=len(tune), audit_users=len(audit))
    # Refit to all pseudo-next events only after selection and independent audit.
    model = HistGradientBoostingRegressor(**params).fit(x, y)
    test_pred = np.maximum(0, model.predict(tx))
    test_pred = (alpha*test_pred + (1-alpha)*test_baseline)*DAY
    base.write_predictions(ROOT / 'assignment1_2026/task_6_test.csv', ROOT / 'experiments/delta_task_6_predictions.csv', test_pred)
    (ROOT / 'experiments/delta_results.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(ROOT / 'experiments/delta_audit.npz', users=audit, truth=y[audit]*DAY, baseline=baseline[audit]*DAY, model=pred*DAY)
    base.log(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
