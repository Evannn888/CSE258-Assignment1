"""Raw ordinal session profiles complementary to bias-residual kernels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from numba import njit
import assignment1 as base

ROOT=Path(__file__).resolve().parents[1]


def fit_session_hist(fit,nu):
    a=fit[fit[:,3]>0]
    cnt=np.bincount(a[:,0],minlength=nu)
    order=np.lexsort((a[:,2],a[:,0]));aa=a[order]
    ptr=np.r_[0,cnt.cumsum()]
    prior=np.zeros((nu+1,5))
    np.add.at(prior,(a[:,0],a[:,3]-1),1.)
    global_p=prior.sum(axis=0)/len(a)
    prior=(prior+5*global_p)/(np.r_[cnt,0][:,None]+5.)
    return dict(ptr=ptr,times=aa[:,2],ratings=aa[:,3],prior=prior)


@njit(cache=True)
def _hist_features(qu,qt,ptr,times,ratings,prior):
    taus=np.array([60.,600.,3600.,86400.,7*86400.,30*86400.,365*86400.])
    out=np.zeros((len(qu),7*15+8))
    for n in range(len(qu)):
        u,t=qu[n],qt[n]
        if u>=len(ptr)-1:continue
        s,e=ptr[u],ptr[u+1]
        if e<=s:continue
        p=prior[u]
        umean=0.
        for star in range(5):umean+=(star+1)*p[star]
        pos=np.searchsorted(times[s:e],t)+s
        l,r=max(s,pos-500),min(e,pos+500)
        for k,tau in enumerate(taus):
            hist=np.zeros(5);w2=0.
            for j in range(l,r):
                w=np.exp(-abs(t-times[j])/tau)
                hist[ratings[j]-1]+=w
                w2+=w*w
            support=hist.sum()
            probs=(hist+2*p)/(support+2.)
            mean=0.
            for star in range(5):mean+=(star+1)*probs[star]
            var,ent=0.,0.
            for star in range(5):
                var+=(star+1-mean)**2*probs[star]
                ent-=probs[star]*np.log(max(probs[star],1e-12))
            mode=float(np.argmax(hist)+1) if support>0 else umean
            basecol=15*k
            out[n,basecol]=mean
            out[n,basecol+1]=mean-umean
            out[n,basecol+2]=var
            out[n,basecol+3]=ent
            out[n,basecol+4]=mode-umean
            out[n,basecol+5]=np.log1p(support)
            out[n,basecol+6]=np.log1p(support*support/(w2+1e-8))
            out[n,basecol+7]=(mean-umean)*np.log1p(support)
            for star in range(5):out[n,basecol+8+star]=probs[star]-p[star]
            out[n,basecol+13]=max(probs)
            out[n,basecol+14]=(mode-umean)*support/(support+2.)
        b=7*15
        if pos>s:
            out[n,b]=ratings[pos-1]-umean
            out[n,b+1]=np.log1p(abs(t-times[pos-1]))
        if pos<e:
            out[n,b+2]=ratings[pos]-umean
            out[n,b+3]=np.log1p(abs(t-times[pos]))
        out[n,b+4]=umean
        for star in range(3):out[n,b+5+star]=p[star]
    return out


def session_hist_features(state,qu,qi,qt):
    return _hist_features(qu,qt,state['ptr'],state['times'],state['ratings'],state['prior'])


if __name__=='__main__':
    a,users,items=base.load(ROOT/'assignment1_2026/tasks_3_4_train.csv')
    ix=np.random.default_rng(258).permutation(len(a));v,fit=a[ix[:50000]],a[ix[50000:]]
    s=fit_session_hist(fit,len(users))
    F=session_hist_features(s,v[:,0],v[:,1],v[:,2])
    np.savez_compressed(ROOT/'experiments/v3_session_hist_features.npz',F=F,valid=v)
    print(F.shape,flush=True)
