"""Low-rank residual-history SVD, excluding each training target from its profile."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,time,json,argparse
from pathlib import Path
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import assignment1 as base
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True)
def epoch(order,ptr,items,rats,weight,p,q,x,bu,bi,mu,lr,reg):
 k=p.shape[1];h=np.zeros(k,np.float64);G=np.zeros(k,np.float64)
 for u in order:
  s,e=ptr[u],ptr[u+1];n=e-s
  if n==0:continue
  h[:]=0;G[:]=0
  for ix in range(s,e):
   j=items[ix]
   for d in range(k):h[d]+=weight[ix]*x[j,d]
  errs=np.empty(n,np.float64);es=0.
  for ix in range(s,e):
   i=items[ix];pred=mu+bu[u]+bi[i]
   for d in range(k):pred+=q[i,d]*(p[u,d]+h[d]-weight[ix]*x[i,d])
   err=rats[ix]-pred;errs[ix-s]=err;es+=err
   for d in range(k):G[d]+=err*q[i,d]
  # Item encoder gradients exclude the target's own residual input.
  for ix in range(s,e):
   j=items[ix];wj=weight[ix];err=errs[ix-s]
   for d in range(k):
    x[j,d]=(x[j,d]+lr*wj*(G[d]-err*q[j,d])/np.sqrt(n))/(1+lr*reg*np.sqrt(n))
  # q must see the frozen old profile; its self term needs the old x_j.
  # We therefore reconstruct old x_j from its reversible encoder update.
  for ix in range(s,e):
   i=items[ix];err=errs[ix-s];wi=weight[ix]
   for d in range(k):
    oldx=x[i,d]*(1+lr*reg*np.sqrt(n))-lr*wi*(G[d]-err*q[i,d])/np.sqrt(n)
    oldq=q[i,d]
    q[i,d]=(oldq+lr*err*(p[u,d]+h[d]-wi*oldx))/(1+lr*reg)
   bi[i]+=lr*(err-.02*bi[i])
  for d in range(k):p[u,d]=(p[u,d]+lr*G[d])/(1+lr*reg*n)
  bu[u]+=lr*es/(1+lr*.05*n)

@njit(cache=True)
def profiles(ptr,items,weight,x):
 out=np.zeros((len(ptr)-1,x.shape[1]),np.float32)
 for u in range(len(out)):
  for ix in range(ptr[u],ptr[u+1]):
   for d in range(x.shape[1]):out[u,d]+=weight[ix]*x[items[ix],d]
 return out

class AsymModel:
 def __init__(self,p,q,x,bu,bi,mu,ptr,items,weight):
  self.p,self.q,self.x,self.bu,self.bi,self.mu=p,q,x,bu,bi,mu
  self.ptr,self.items,self.weight=ptr,items,weight
 def predict(self,u,i):
  return self.mu+self.bu[u]+self.bi[i]+np.sum((self.p+profiles(self.ptr,self.items,self.weight,self.x))[u]*self.q[i],axis=1)

def train(fit,nu,ni,tune,k=32,reg=.08,lr=.002,epochs=100,seed=258):
 a=fit[fit[:,3]>0];uc=np.bincount(a[:,0],minlength=nu);ic=np.bincount(a[:,1],minlength=ni)
 mu,bu,bi=base.als_biases_custom(a,nu,ni,uc,ic,3.,5.,20)
 oo=np.argsort(a[:,0],kind='stable');ptr=np.r_[0,uc.cumsum()];items=a[oo,1];rats=a[oo,3]
 weight=(a[:,3]-mu-bu[a[:,0]]-bi[a[:,1]])/np.sqrt(uc[a[:,0]]+5.)
 weight=weight[oo].astype(np.float32);rng=np.random.default_rng(seed)
 p=rng.normal(0,.05,(nu,k)).astype(np.float32);q=rng.normal(0,.05,(ni,k)).astype(np.float32);x=np.zeros((ni,k),np.float32)
 model=AsymModel(p,q,x,bu.astype(np.float32),bi.astype(np.float32),mu,ptr,items,weight)
 best=1e9;stale=0;state=None;bestep=0
 for ep in range(1,epochs+1):
  epoch(rng.permutation(nu),ptr,items,rats,weight,p,q,x,model.bu,model.bi,mu,lr*.99**(ep-1),reg)
  if ep%5==0:
   pred=np.clip(model.predict(tune[:,0],tune[:,1]),1,5);mse=float(np.mean((pred-tune[:,3])**2));print('ASYM',k,reg,ep,mse,flush=True)
   if not np.isfinite(mse):break
   if mse<best:best=mse;bestep=ep;stale=0;state=[z.copy() for z in [p,q,x,model.bu,model.bi]]
   else:stale+=1
   if stale>=4:break
 model.p,model.q,model.x,model.bu,model.bi=state
 return model,dict(k=k,reg=reg,lr=lr,epochs=bestep,tune_mse=best)

if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--task',type=int,default=1);args=pa.parse_args()
 a,us,its=base.load(ROOT/'assignment1_2026'/('tasks_1_2_train.csv' if args.task==1 else 'tasks_3_4_train.csv'))
 order=np.random.default_rng(258).permutation(len(a));valid=a[order[:50000]];fit=a[order[50000:]]
 tune=valid[20000:35000];tune=tune[tune[:,3]>0]
 for k,reg in [(32,.08),(64,.08)]:
  start=time.time();model,config=train(fit,len(us),len(its),tune,k=k,reg=reg)
  config['elapsed_seconds']=time.time()-start
  np.savez_compressed(ROOT/f'experiments/v3_latent_models/t{args.task}_asym{k}.npz',val_pred=model.predict(valid[:,0],valid[:,1]),p=model.p,q=model.q,x=model.x,bu=model.bu,bi=model.bi,mu=model.mu,ptr=model.ptr,items=model.items,weight=model.weight,config_json=json.dumps(config))
  print('ASYM FINISHED',config,flush=True)
