"""Infer Task 1 rating stage by integrating over its unobserved query time."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,time,json,pickle
from pathlib import Path
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import assignment1 as base
import production_ratings as old
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True)
def _fit_stages(ptr,items,times,rating,res,emb,nu):
 k=emb.shape[1];bounds=np.zeros((nu,8,2),np.int64);stats=np.zeros((nu,8,5));centers=np.zeros((nu,8,k))
 for u in range(nu):
  us,ue=ptr[u],ptr[u+1];n=ue-us
  if n==0:continue
  high=np.mean(rating[us:ue]>=4)
  offset=0
  for parts in (3,5):
   for s in range(parts):
    lo=us+n*s//parts;hi=us+n*(s+1)//parts;idx=offset+s;nn=hi-lo
    if nn==0:continue
    bounds[u,idx,0]=times[lo]-43200;bounds[u,idx,1]=times[hi-1]+43200
    rr=res[lo:hi];stats[u,idx,0]=nn/n
    stats[u,idx,1]=np.sum(rr)/(nn+3.)
    stats[u,idx,2]=np.median(rr)*nn/(nn+3.)
    stats[u,idx,3]=(np.sum(rating[lo:hi]>=4)-high*nn)/(nn+3.)
    stats[u,idx,4]=np.sum(rr*rr)/(nn+3.)
    for ix in range(lo,hi):
     for d in range(k):centers[u,idx,d]+=emb[items[ix],d]
    norm=np.sqrt(np.sum(centers[u,idx]**2))
    if norm>1e-7:centers[u,idx]/=norm
   offset+=parts
 return bounds,stats,centers

def fit_stage_model(fit,nu,ni,embeddings=None,biases=None):
 pos=fit[fit[:,3]>0];uc=np.bincount(pos[:,0],minlength=nu);ic=np.bincount(pos[:,1],minlength=ni)
 if biases is None:biases=base.als_biases_custom(pos,nu,ni,uc,ic,8.,12.,20)
 mu,bu,bi=biases;res=pos[:,3]-mu-bu[pos[:,0]]-bi[pos[:,1]]
 emb=np.zeros((ni,1),np.float32) if embeddings is None else embeddings.astype(np.float32).copy()
 norm=np.linalg.norm(emb,axis=1,keepdims=True);emb=np.divide(emb,norm,out=np.zeros_like(emb),where=norm>1e-7)
 order=np.lexsort((pos[:,2],pos[:,0]));ptr=np.r_[0,uc.cumsum()]
 bounds,stats,centers=_fit_stages(ptr,pos[order,1],pos[order,2],pos[order,3],res[order],emb,nu)
 all_times=np.sort(fit[:,2]);io=np.lexsort((fit[:,2],fit[:,1]));ics=np.bincount(fit[:,1],minlength=ni)
 return dict(bounds=bounds,stats=stats,centers=centers,emb=emb,iptr=np.r_[0,ics.cumsum()],itimes=fit[io,2],global_times=all_times)

@njit(cache=True)
def stage_features(state_bounds,stats,centers,emb,iptr,itimes,global_times,qu,qi):
 out=np.zeros((len(qu),20));nu=stats.shape[0];ni=len(iptr)-1
 for z in range(len(qu)):
  u,i=qu[z],qi[z]
  if u>=nu:continue
  is_,ie=(iptr[i],iptr[i+1]) if i<ni else (0,0)
  offset=0
  for parts in (3,5):
   tw=np.zeros(parts);ew=np.zeros(parts);cw=np.zeros(parts);lift=np.zeros(parts)
   for s in range(parts):
    idx=offset+s;prior=stats[u,idx,0]
    if prior<=0:continue
    lo,hi=state_bounds[u,idx]
    p=max(1e-9,(np.searchsorted(global_times,hi,side='right')-np.searchsorted(global_times,lo))/len(global_times))
    c=np.searchsorted(itimes[is_:ie],hi,side='right')-np.searchsorted(itimes[is_:ie],lo)
    ll=np.log((c+3.*p)/((ie-is_+3.)*p));ll=max(-6.,min(6.,ll));lift[s]=ll
    cosine=0.
    if i<ni:
     for d in range(emb.shape[1]):cosine+=emb[i,d]*centers[u,idx,d]
    tw[s]=prior*np.exp(ll);ew[s]=prior*np.exp(3.*cosine);cw[s]=prior*np.exp(ll+3.*cosine)
   st,se,sc=np.sum(tw),np.sum(ew),np.sum(cw)
   if st<=0:offset+=parts;continue
   tw/=st;ew/=se;cw/=sc
   b=0 if parts==3 else 10
   for s in range(parts):
    idx=offset+s
    out[z,b]+=tw[s]*stats[u,idx,1]
    out[z,b+1]+=tw[s]*stats[u,idx,2]
    out[z,b+2]+=cw[s]*stats[u,idx,1]
    out[z,b+3]+=cw[s]*stats[u,idx,2]
    out[z,b+4]+=ew[s]*stats[u,idx,1]
    out[z,b+5]+=tw[s]*stats[u,idx,3]
    out[z,b+6]+=tw[s]*stats[u,idx,4]
    out[z,b+8]+=tw[s]*(s+.5)/parts
    out[z,b+9]+=tw[s]*lift[s]
   out[z,b+7]=np.max(tw)-np.min(tw)
   offset+=parts
 return out

def extract_stage_features(state,qu,qi):
 return stage_features(state['bounds'],state['stats'],state['centers'],state['emb'],state['iptr'],state['itimes'],state['global_times'],qu,qi)

if __name__=='__main__':
 data=np.load(ROOT/'experiments/v3_base_1.npz');X,y,v,bp=data['X'],data['y'],data['valid'],data['pred']
 with open(ROOT/'experiments/v3_state_1.pkl','rb') as f:oldstate=pickle.load(f)
 start=time.time();state=fit_stage_model(oldstate['a_implicit'],oldstate['nu'],oldstate['ni'],oldstate['svdpp64'].y,oldstate['biases'][2])
 feats=extract_stage_features(state,v[:,0],v[:,1]);np.savez_compressed(ROOT/'experiments/v3_task1_stages.npz',features=feats)
 tr=np.arange(20000)[y[:20000]>0];tu=np.arange(20000,35000)[y[20000:35000]>0]
 rr=old.StandardizedRidge(40).fit(X[tr],y[tr]);raw=rr.predict_raw(X)
 records=[];best=(1e9,None,None)
 for l2 in (40.,400.,2000.):
  xx=np.column_stack([X,feats]);m=old.StandardizedRidge(l2).fit(xx[tr],y[tr]);delta=m.predict_raw(xx)-raw
  for scale in (.5,1.):
   pred=np.clip(bp+scale*delta,1,5);mse=float(np.mean((pred[tu]-y[tu])**2));rec=dict(l2=l2,scale=scale,tune=mse);records.append(rec)
   if mse<best[0]:best=(mse,rec,pred)
 result=dict(baseline_tune=float(np.mean((bp[tu]-y[tu])**2)),best_tune=best[0],selected=best[1],records=records,elapsed_seconds=time.time()-start)
 (ROOT/'experiments/v3_task1_stages.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
