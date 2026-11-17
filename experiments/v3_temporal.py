"""Task 3 temporal random-effects experiments; tune rows only select features.

Each model jointly deconfounds user/item static effects and sparse local time
effects. No validation target enters a group table or any historical feature.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
from numba import njit
import assignment1 as base
from experiments import production_ratings as old

ROOT = Path(__file__).resolve().parents[1]
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
    # Huge corrupt dates are explicitly pooled into extreme bins; keys remain
    # bounded and no date/ID numeric meaning is used as a model feature.
    bins = np.clip(np.floor((times-946684800)/(DAY*days)), -10000, 10000).astype(np.int64)
    return ids.astype(np.int64)*20001 + bins + 10000


def make_group(ids, times, days):
    keys, inverse = np.unique(time_code(ids,times,days), return_inverse=True)
    return keys, inverse.astype(np.int32), np.bincount(inverse).astype(np.float64)


DEFAULT_CONFIGS = [
    {'user_days':1., 'item_days':90., 'session_reg':1., 'item_reg':20., 'slope_reg':20.},
    {'user_days':1., 'item_days':30., 'session_reg':4., 'item_reg':10., 'slope_reg':80.},
    {'user_days':7., 'item_days':90., 'session_reg':2., 'item_reg':20., 'slope_reg':20.},
    {'user_days':30., 'item_days':365., 'session_reg':4., 'item_reg':30., 'slope_reg':80.},
    {'user_days':90., 'item_days':90., 'session_reg':8., 'item_reg':10., 'slope_reg':20.},
    {'user_days':365., 'item_days':365., 'session_reg':12., 'item_reg':30., 'slope_reg':80.},
]


def fit_temporal_models(fit_rows, nu, ni, configs=None):
    a = fit_rows[fit_rows[:,3]>0]
    u,i,t,y = a[:,0],a[:,1],a[:,2],a[:,3].astype(np.float64)
    uc = np.bincount(u,minlength=nu).astype(np.float64)
    ic = np.bincount(i,minlength=ni).astype(np.float64)
    tc = np.bincount(u,weights=t.astype(float),minlength=nu)/np.maximum(uc,1)
    # Robust sign-power trend, centered within each user's positive history.
    raw = np.sign((t-tc[u])/DAY)*np.abs((t-tc[u])/DAY)**.4
    zc = np.bincount(u,weights=raw,minlength=nu)/np.maximum(uc,1)
    z = raw-zc[u]
    z2 = np.bincount(u,weights=z*z,minlength=nu)
    mu = float(y.mean())
    states=[]
    for cfg in configs or DEFAULT_CONFIGS:
        keys_u, gu, cgu = make_group(u,t,cfg['user_days'])
        keys_i, gi, cgi = make_group(i,t,cfg['item_days'])
        keys_long, gl, cgl = make_group(u,t,365.)
        bu,bi,slope = np.zeros(nu),np.zeros(ni),np.zeros(nu)
        eu,ei,el = np.zeros(len(keys_u)),np.zeros(len(keys_i)),np.zeros(len(keys_long))
        pred = np.full(len(y),mu)
        for _ in range(25):
            bu=categorical_sweep(y,pred,u,bu,uc,5.)
            bi=categorical_sweep(y,pred,i,bi,ic,8.)
            slope=slope_sweep(y,pred,u,z,slope,z2,cfg['slope_reg'])
            el=categorical_sweep(y,pred,gl,el,cgl,15.)
            ei=categorical_sweep(y,pred,gi,ei,cgi,cfg['item_reg'])
            eu=categorical_sweep(y,pred,gu,eu,cgu,cfg['session_reg'])
        states.append(dict(config=cfg,mu=mu,bu=bu,bi=bi,tc=tc,zc=zc,slope=slope,
            keys_u=keys_u,effect_u=eu,count_u=cgu,keys_i=keys_i,effect_i=ei,count_i=cgi,
            keys_long=keys_long,effect_long=el,count_long=cgl))
    return dict(nu=nu,ni=ni,models=states)


def lookup_group(keys, effects, counts, query_keys):
    where=np.searchsorted(keys,query_keys)
    good=(where<len(keys))
    safe=np.minimum(where,len(keys)-1)
    good &= keys[safe]==query_keys
    return np.where(good,effects[safe],0.),np.where(good,counts[safe],0.)


def temporal_features(state,qu,qi,qt):
    nu,ni=state['nu'],state['ni']
    cols=[]
    for s in state['models']:
        cfg=s['config']
        bu=np.r_[s['bu'],0.][qu]; bi=np.r_[s['bi'],0.][qi]
        tc=np.r_[s['tc'],0.][qu]; zc=np.r_[s['zc'],0.][qu]
        dt=(qt-tc)/DAY
        z=np.clip(np.sign(dt)*np.abs(dt)**.4-zc,-30,30)
        trend=np.r_[s['slope'],0.][qu]*z
        eu,cu=lookup_group(s['keys_u'],s['effect_u'],s['count_u'],time_code(qu,qt,cfg['user_days']))
        ei,ci=lookup_group(s['keys_i'],s['effect_i'],s['count_i'],time_code(qi,qt,cfg['item_days']))
        el,cl=lookup_group(s['keys_long'],s['effect_long'],s['count_long'],time_code(qu,qt,365.))
        pred=s['mu']+bu+bi+trend+eu+ei+el
        cols.extend([pred,eu,ei,el,trend,np.log1p(cu),np.log1p(ci),
            eu*np.log1p(cu),ei*np.log1p(ci),eu*(cu/(cu+2.))])
    return np.column_stack(cols)


def main():
    started=time.time()
    a,users,items=base.load(ROOT/'assignment1_2026/tasks_3_4_train.csv')
    order=np.random.default_rng(258).permutation(len(a))
    valid,fit=a[order[:50000]],a[order[50000:]]
    state=fit_temporal_models(fit,len(users),len(items))
    F=temporal_features(state,valid[:,0],valid[:,1],valid[:,2])
    np.savez_compressed(ROOT/'experiments/v3_temporal_features.npz',F=F,valid=valid)
    print('Prepared joint temporal features',F.shape,'sec',time.time()-started,flush=True)
    cache=ROOT/'experiments/v3_base_3.npz'
    if not cache.exists():
        print('Baseline cache pending; run --evaluate after it is available.',flush=True)
        return
    evaluate(F,valid,cache)


def evaluate(F,valid,cache):
    data=np.load(cache)
    X=data['X']; y=valid[:,3].astype(float)
    tr=np.arange(20000)[y[:20000]>0]
    tu=np.arange(20000,35000)[y[20000:35000]>0]
    records=[]
    for extra in [False,True]:
        xx=np.column_stack([X,F]) if extra else X
        for l2 in [100.,600.,1500.,3500.,10000.]:
            mod=old.StandardizedRidge(l2=l2).fit(xx[tr],y[tr])
            mse=float(np.mean((np.clip(mod.predict_raw(xx[tu]),1,5)-y[tu])**2))
            rec=dict(extra=extra,l2=l2,tune_mse=mse)
            records.append(rec); print(rec,flush=True)
    (ROOT/'experiments/v3_temporal_results.json').write_text(json.dumps(dict(records=records, audit_viewed=False),indent=2))


if __name__=='__main__':
    if '--evaluate' in sys.argv:
        z=np.load(ROOT/'experiments/v3_temporal_features.npz')
        evaluate(z['F'],z['valid'],ROOT/'experiments/v3_base_3.npz')
    else:
        main()
