"""Time-aware explicit MF for Task 3, with tune-only epoch selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time
import numpy as np
from numba import njit
import assignment1 as base
from experiments.v3_temporal import time_code

ROOT=Path(__file__).resolve().parents[1]
DAY=86400.


@njit(cache=True)
def time_mf_epoch(a,order,z,session,itembin,p,q,alpha,bu,bi,sb,ib,drift,mu,lr,reg,sreg):
    for n in order:
        u,i=a[n,0],a[n,1]
        zz=z[n]
        s,b=session[n],itembin[n]
        yhat=mu+bu[u]+bi[i]+sb[s]+ib[b]+drift[u]*zz
        for k in range(p.shape[1]):
            yhat+=(p[u,k]+alpha[u,k]*zz)*q[i,k]
        e=a[n,3]-yhat
        bu[u]+=lr*(e-.025*bu[u])
        bi[i]+=lr*(e-.025*bi[i])
        sb[s]+=lr*(e-sreg*sb[s])
        ib[b]+=lr*(e-.15*ib[b])
        drift[u]+=lr*(e*zz-.15*drift[u])
        for k in range(p.shape[1]):
            pu,qi,au=p[u,k],q[i,k],alpha[u,k]
            p[u,k]+=lr*(e*qi-reg*pu)
            alpha[u,k]+=lr*(e*qi*zz-.3*au)
            q[i,k]+=lr*(e*(pu+au*zz)-reg*qi)


def prepare_time_mf(a,nu,ni,cfg,seed=258):
    a=a[a[:,3]>0]
    u,i,t=a[:,0],a[:,1],a[:,2].astype(float)
    uc=np.bincount(u,minlength=nu)
    ic=np.bincount(i,minlength=ni)
    mu,bu,bi=base.als_biases_custom(a,nu,ni,uc,ic,8.,12.,n_iter=20)
    tm=np.bincount(u,weights=t,minlength=nu)/np.maximum(uc,1)
    ts=np.sqrt(np.bincount(u,weights=(t-tm[u])**2,minlength=nu)/np.maximum(uc,1))+180*DAY
    z=np.clip((t-tm[u])/ts[u],-3,3)
    skeys,session=np.unique(time_code(u,a[:,2],cfg['session_days']),return_inverse=True)
    ikeys,itembin=np.unique(time_code(i,a[:,2],90.),return_inverse=True)
    rng=np.random.default_rng(seed)
    k=cfg['k']
    st=dict(mu=mu,bu=bu.astype(np.float32),bi=bi.astype(np.float32),
       p=rng.normal(0,.05,(nu,k)).astype(np.float32),q=rng.normal(0,.05,(ni,k)).astype(np.float32),
       alpha=np.zeros((nu,k),np.float32),sb=np.zeros(len(skeys),np.float32),ib=np.zeros(len(ikeys),np.float32),
       drift=np.zeros(nu,np.float32),tm=tm,ts=ts,skeys=skeys,ikeys=ikeys,cfg=cfg,nu=nu,ni=ni)
    st['p'][uc==0]=0.; st['q'][ic==0]=0.
    return a,z,session,itembin,st


def lookup(keys,vals,qkeys):
    j=np.searchsorted(keys,qkeys)
    safe=np.minimum(j,len(keys)-1)
    good=(j<len(keys))&(keys[safe]==qkeys)
    return np.where(good,vals[safe],0.)


def predict_time_mf(st,qu,qi,qt):
    p=np.vstack([st['p'],np.zeros((1,st['p'].shape[1]),np.float32)])
    q=np.vstack([st['q'],np.zeros((1,st['q'].shape[1]),np.float32)])
    al=np.vstack([st['alpha'],np.zeros((1,st['alpha'].shape[1]),np.float32)])
    tm=np.r_[st['tm'],0.][qu]; ts=np.r_[st['ts'],180*DAY][qu]
    z=np.clip((qt-tm)/ts,-3,3)
    ses=lookup(st['skeys'],st['sb'],time_code(qu,qt,st['cfg']['session_days']))
    ib=lookup(st['ikeys'],st['ib'],time_code(qi,qt,90.))
    td=np.r_[st['drift'],0.][qu]*z
    dot=np.sum((p[qu]+al[qu]*z[:,None])*q[qi],axis=1)
    pred=st['mu']+np.r_[st['bu'],0.][qu]+np.r_[st['bi'],0.][qi]+dot+ses+ib+td
    return np.column_stack([np.clip(pred,1,5),dot,ses,ib,td])


def train_time_mf(fit,nu,ni,cfg,tune=None,epochs=80,seed=258):
    a,z,session,itembin,st=prepare_time_mf(fit,nu,ni,cfg,seed)
    rng=np.random.default_rng(seed)
    best=float('inf');best_ep=epochs;best_st=None
    mutable=['p','q','alpha','bu','bi','sb','ib','drift']
    for ep in range(1,epochs+1):
        time_mf_epoch(a,rng.permutation(len(a)),z,session,itembin,st['p'],st['q'],st['alpha'],st['bu'],st['bi'],st['sb'],st['ib'],st['drift'],st['mu'],cfg['lr']*(.985**(ep-1)),cfg['reg'],cfg['sreg'])
        if tune is not None and ep%5==0:
            pred=predict_time_mf(st,tune[:,0],tune[:,1],tune[:,2])[:,0]
            mse=float(np.mean((pred-tune[:,3])**2))
            print('TimeMF',cfg,'epoch',ep,'tune',mse,flush=True)
            if mse<best:
                best,best_ep=mse,ep
                best_st={key:st[key].copy() for key in mutable}
            elif ep-best_ep>=15:
                break
    if best_st is not None:st.update(best_st)
    return st,best_ep,best


def main():
    a,users,items=base.load(ROOT/'assignment1_2026/tasks_3_4_train.csv')
    perm=np.random.default_rng(258).permutation(len(a))
    valid,fit=a[perm[:50000]],a[perm[50000:]]
    tune=valid[20000:35000];tune=tune[tune[:,3]>0]
    configs=[dict(k=32,lr=.01,reg=.04,sreg=.2,session_days=1.),
             dict(k=64,lr=.01,reg=.04,sreg=.2,session_days=1.),
             dict(k=64,lr=.012,reg=.08,sreg=.5,session_days=7.)]
    preds=[];records=[]
    for n,cfg in enumerate(configs):
        st,ep,mse=train_time_mf(fit,len(users),len(items),cfg,tune)
        preds.append(predict_time_mf(st,valid[:,0],valid[:,1],valid[:,2]))
        records.append(dict(config=cfg,best_epoch=ep,tune_mse=mse))
        np.savez_compressed(ROOT/'experiments/v3_time_mf_features.npz',F=np.column_stack(preds),valid=valid)
        (ROOT/'experiments/v3_time_mf_results.json').write_text(json.dumps(records,indent=2))
    print('TimeMF completed',records,flush=True)


if __name__=='__main__':main()
