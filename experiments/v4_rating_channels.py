"""Leave-self-out, rating-channel graph encoder with user-center correction."""
import sys,time,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
import assignment1 as base
ROOT=Path(__file__).resolve().parents[1]


@njit(cache=True)
def channel_values(H,Q,W,Y,rows,cols,channels,coef):
    out=np.zeros(len(rows),H.dtype)
    for n in range(len(rows)):
        u,i,r=rows[n],cols[n],channels[n]
        for j in range(Q.shape[1]):out[n]+=(H[u,j]-coef[n]*(W[r,i,j]+Y[i,j]))*Q[i,j]
    return out


def objective(params,As,B,rows,cols,ch,coef,target,penalty):
    Q,W,Y=params['Q'],params['W'],params['Y']
    H=B@Y
    for r in range(5):H+=As[r]@W[r]
    pred=channel_values(H,Q,W,Y,rows,cols,ch,coef);err=pred-target
    E=csr_matrix((err,(rows,cols)),shape=B.shape);G=E@Q
    gw=np.empty_like(W);gq=E.T@H
    for r in range(5):
        mask=ch==r
        sd=np.bincount(cols[mask],weights=err[mask]*coef[mask],minlength=len(Q))
        gw[r]=As[r].T@G-sd[:,None]*Q
        gq-=sd[:,None]*W[r]
    sd=np.bincount(cols,weights=err*coef,minlength=len(Q))
    gy=B.T@G-sd[:,None]*Q;gq-=sd[:,None]*Y
    grads={'Q':gq,'W':gw,'Y':gy}
    loss=.5*float(err.astype(float)@err.astype(float))
    for name,v in params.items():
        grads[name]+=penalty[name]*v
        loss+=.5*float(np.sum(penalty[name]*v.astype(float)**2))
    return loss,grads,H


def check_gradient():
    rng=np.random.default_rng(77);rows=np.array([0,0,1,1,2,2]);cols=np.array([0,1,1,2,0,2]);ch=np.array([0,1,2,3,4,1]);c=np.ones(6)*.4
    B=csr_matrix((c,(rows,cols)),shape=(3,3))
    As=[csr_matrix((c[ch==r],(rows[ch==r],cols[ch==r])),shape=(3,3)) for r in range(5)]
    p={'Q':rng.normal(0,.1,(3,2)),'W':rng.normal(0,.1,(5,3,2)),'Y':rng.normal(0,.1,(3,2))}
    target=rng.normal(size=6);pen={'Q':.2,'W':.3,'Y':.2}
    _,g,_=objective(p,As,B,rows,cols,ch,c,target,pen);worst=0.
    for key,v in p.items():
        for idx in np.ndindex(v.shape):
            value=v[idx];v[idx]=value+1e-6;a=objective(p,As,B,rows,cols,ch,c,target,pen)[0]
            v[idx]=value-1e-6;b=objective(p,As,B,rows,cols,ch,c,target,pen)[0];v[idx]=value
            worst=max(worst,abs((a-b)/2e-6-g[key][idx]))
    assert worst<1e-7,worst
    return worst


def fit_channels(fit,nu,ni,tune,reg=5.,epochs=100,rank=64,seed=258):
    a=fit[fit[:,3]>0];a=a[np.lexsort((a[:,1],a[:,0]))]
    u,i=a[:,0],a[:,1];ch=(a[:,3]-1).astype(np.int32)
    uc=np.bincount(u,minlength=nu);ic=np.bincount(i,minlength=ni)
    mu,_,bi=base.als_biases_custom(a,nu,ni,uc,ic,8.,12.,20)
    raw=a[:,3]-mu-bi[i];sums=np.bincount(u,weights=raw,minlength=nu)
    bu=sums/(uc+8.);loo=(sums[u]-raw)/(uc[u]+7.)
    target=(a[:,3]-mu-bi[i]-loo).astype(np.float32)
    impc=np.bincount(fit[:,0],minlength=nu)
    scale=1/np.sqrt(impc+5.);coef=scale[u].astype(np.float32)
    As=[csr_matrix((coef[ch==r],(u[ch==r],i[ch==r])),shape=(nu,ni)) for r in range(5)]
    B=csr_matrix((scale[fit[:,0]].astype(np.float32),(fit[:,0],fit[:,1])),shape=(nu,ni))
    rng=np.random.default_rng(seed)
    params={'Q':rng.normal(0,.05,(ni,rank)).astype(np.float32),'W':np.zeros((5,ni,rank),np.float32),'Y':np.zeros((ni,rank),np.float32)}
    penalties={'Q':(.1*(ic+5.))[:,None],
       'W':np.stack([(reg*(np.asarray(ar.power(2).sum(axis=0)).ravel()+.1))[:,None] for ar in As]),
       'Y':(reg*(np.asarray(B.power(2).sum(axis=0)).ravel()+.1))[:,None]}
    m={k:np.zeros_like(v) for k,v in params.items()};v={k:np.zeros_like(x) for k,x in params.items()}
    best=1e9;bestp=None;bestep=0;stale=0;records=[]
    for ep in range(1,epochs+1):
        loss,g,H=objective(params,As,B,u,i,ch,coef,target,penalties)
        for key in params:
            m[key]=.9*m[key]+.1*g[key];v[key]=.999*v[key]+.001*g[key]**2
            params[key]-=(.005*.995**(ep-1))*(m[key]/(1-.9**ep))/(np.sqrt(v[key]/(1-.999**ep))+1e-8)
        if ep%5==0:
            H=B@params['Y']
            for r in range(5):H+=As[r]@params['W'][r]
            pred=mu+bu[tune[:,0]]+bi[tune[:,1]]+np.sum(H[tune[:,0]]*params['Q'][tune[:,1]],axis=1)
            mse=float(np.mean((np.clip(pred,1,5)-tune[:,3])**2))
            records.append(dict(epoch=ep,loss=loss/len(u),tune=mse));print(reg,records[-1],flush=True)
            if mse<best:
                best=mse;bestep=ep;bestp={k:x.copy() for k,x in params.items()};stale=0
            else:stale+=1
            if stale>=6:break
    H=B@bestp['Y']
    for r in range(5):H+=As[r]@bestp['W'][r]
    return dict(H=H,Q=bestp['Q'],mu=mu,bu=bu,bi=bi,params=bestp),dict(reg=reg,epoch=bestep,tune=best,history=records)


def main():
    err=check_gradient();print('GRADIENT CHECK',err,flush=True)
    with (ROOT/'experiments/v3_big_state_1.pkl').open('rb') as f:st=pickle.load(f)
    d=np.load(ROOT/'experiments/v3_big_stage_1.npz');val=d['valid'];tune=val[150000:175000];tune=tune[tune[:,3]>0]
    for reg in [5.,20.]:
        model,report=fit_channels(st['fit_rows'],len(st['users']),len(st['items']),tune,reg)
        pred=model['mu']+model['bu'][val[:,0]]+model['bi'][val[:,1]]+np.sum(model['H'][val[:,0]]*model['Q'][val[:,1]],axis=1)
        np.savez_compressed(ROOT/f'experiments/v4_channels_{int(reg)}.npz',pred=pred)
        with (ROOT/f'experiments/v4_channels_{int(reg)}.pkl').open('wb') as f:pickle.dump(model,f)
        (ROOT/f'experiments/v4/channels_{int(reg)}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
