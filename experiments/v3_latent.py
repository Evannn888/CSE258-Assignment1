"""Explicit MF and exact user-cached SVD++ experiments; positive labels only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,time,json,argparse,pickle
from pathlib import Path
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import assignment1 as base
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True)
def mf_epoch(arr, order, p,q,bu,bi,mu,lr,rp,rq,rb,scales):
 for idx in order:
  u,i,r=arr[idx,0],arr[idx,1],arr[idx,3]
  dot=0.
  for k in range(p.shape[1]):dot+=p[u,k]*q[i,k]
  sc=scales[u]
  err=r-(mu+bu[u]+bi[i]+sc*dot)
  bu[u]+=lr*(err-rb*bu[u]);bi[i]+=lr*(err-rb*bi[i])
  for k in range(p.shape[1]):
   pu,qi=p[u,k],q[i,k]
   p[u,k]+=lr*(err*sc*qi-rp*pu)
   q[i,k]+=lr*(err*sc*pu-rq*qi)

@njit(cache=True)
def fast_svdpp_epoch(uperm,ptr,its,rats,iptr,iits,p,q,y,bu,bi,mu,lr,rp,rq,ry,rb):
 k=p.shape[1]
 h=np.empty(k,np.float64);h0=np.empty(k,np.float64)
 for u in uperm:
  s,e=ptr[u],ptr[u+1]
  ns,ne=iptr[u],iptr[u+1]
  if e<=s or ne<=ns:continue
  norm=1./np.sqrt(ne-ns)
  for d in range(k):
   z=0.
   for ix in range(ns,ne):z+=y[iits[ix],d]
   h[d]=z*norm;h0[d]=h[d]
  for ix in range(s,e):
   i=its[ix];r=rats[ix]
   pred=mu+bu[u]+bi[i]
   for d in range(k):pred+=q[i,d]*(p[u,d]+h[d])
   err=r-pred
   bu[u]+=lr*(err-rb*bu[u]);bi[i]+=lr*(err-rb*bi[i])
   for d in range(k):
    pu,qi=p[u,d],q[i,d]
    q[i,d]+=lr*(err*(pu+h[d])-rq*qi)
    p[u,d]+=lr*(err*qi-rp*pu)
    h[d]+=lr*(err*qi-ry*h[d])
  shrink=(1.-lr*ry)**(e-s)
  for ix in range(ns,ne):
   j=iits[ix]
   for d in range(k):y[j,d]=shrink*y[j,d]+(h[d]-shrink*h0[d])*norm

@njit(cache=True)
def implicit_vectors(ptr,its,y):
 out=np.zeros((len(ptr)-1,y.shape[1]),np.float32)
 for u in range(len(out)):
  s,e=ptr[u],ptr[u+1]
  if e>s:
   norm=1./np.sqrt(e-s)
   for ix in range(s,e):
    for d in range(y.shape[1]):out[u,d]+=y[its[ix],d]*norm
 return out

class Model:
 def __init__(self,p,q,bu,bi,mu,scales,iptr=None,iits=None,y=None):
  self.p,self.q,self.bu,self.bi,self.mu,self.scales=p,q,bu,bi,mu,scales
  self.iptr,self.iits,self.y=iptr,iits,y
 def predict(self,u,i,raw=False):
  uv=u<len(self.p);iv=i<len(self.q);uu=np.minimum(u,len(self.p)-1);ii=np.minimum(i,len(self.q)-1)
  pp=self.p if self.y is None else self.p+implicit_vectors(self.iptr,self.iits,self.y)
  dot=np.sum(pp[uu]*self.q[ii],axis=1)*self.scales[uu]
  pred=self.mu+np.where(uv,self.bu[uu],0)+np.where(iv,self.bi[ii],0)+np.where(uv&iv,dot,0)
  return pred if raw else np.clip(pred,1,5)
 def state(self):
  return dict(p=self.p,q=self.q,bu=self.bu,bi=self.bi,mu=self.mu,scales=self.scales,iptr=self.iptr,iits=self.iits,y=self.y)

def train(fit,nu,ni,valid,config,seed=258,fixed_epochs=None):
 pos=fit[fit[:,3]>0];rng=np.random.default_rng(seed)
 uc=np.bincount(pos[:,0],minlength=nu);ic=np.bincount(pos[:,1],minlength=ni)
 mu,bu,bi=base.als_biases_custom(pos,nu,ni,uc,ic,3.,5.,20)
 p=rng.normal(0,config['init'],(nu,config['k'])).astype(np.float32)
 q=rng.normal(0,config['init'],(ni,config['k'])).astype(np.float32)
 scales=np.ones(nu,np.float32)
 if config.get('scale',False):
  resid=pos[:,3]-mu-bu[pos[:,0]]-bi[pos[:,1]]
  vv=np.bincount(pos[:,0],weights=resid**2,minlength=nu)
  scales=np.sqrt((vv+10*.55)/(uc+10)).astype(np.float32)
 model=Model(p,q,bu.astype(np.float32),bi.astype(np.float32),mu,scales)
 if config['kind']=='svdpp':
  order=np.argsort(pos[:,0],kind='stable');ptr=np.r_[0,uc.cumsum()]
  its=pos[order,1];rats=pos[order,3]
  iu=np.bincount(fit[:,0],minlength=nu);io=np.argsort(fit[:,0],kind='stable')
  model.iptr=np.r_[0,iu.cumsum()];model.iits=fit[io,1];model.y=np.zeros((ni,config['k']),np.float32)
 best=1e10;best_state=None;best_ep=0;stale=0;hist=[]
 maxep=fixed_epochs or config.get('epochs',100)
 tv=valid[(valid[:,3]>0)] if valid is not None else None
 for ep in range(1,maxep+1):
  lr=config['lr']*(config.get('decay',.99)**(ep-1))
  if config['kind']=='mf':
   mf_epoch(pos,rng.permutation(len(pos)),p,q,model.bu,model.bi,mu,lr,config['reg'],config['reg'],config.get('reg_b',.02),scales)
  else:
   fast_svdpp_epoch(rng.permutation(nu),ptr,its,rats,model.iptr,model.iits,p,q,model.y,model.bu,model.bi,mu,lr,config['reg'],config['reg'],config['reg_y'],config.get('reg_b',.02))
  if tv is not None and (ep%5==0 or ep==maxep):
   pred=model.predict(tv[:,0],tv[:,1]);mse=float(np.mean((pred-tv[:,3])**2));hist.append([ep,mse])
   print(config['name'],ep,round(mse,7),flush=True)
   if mse<best:
    best=mse;best_ep=ep;stale=0
    best_state=[model.p.copy(),model.q.copy(),model.bu.copy(),model.bi.copy(),None if model.y is None else model.y.copy()]
   else:stale+=1
   if fixed_epochs is None and stale>=4:break
 if best_state is not None:model.p,model.q,model.bu,model.bi,model.y=best_state
 return model,dict(best_epoch=best_ep or maxep,best_tune_mse=best,history=hist,config=config)

CONFIGS=[
 dict(name='fastpp64_reg08long',kind='svdpp',k=64,init=.05,reg=.08,reg_y=.02,lr=.007,decay=.995,epochs=220),
 dict(name='fastpp64_reg05long',kind='svdpp',k=64,init=.05,reg=.05,reg_y=.01,lr=.007,decay=.995,epochs=180),
 dict(name='mf16_reg08',kind='mf',k=16,init=.1,reg=.08,lr=.01,decay=.99,epochs=90),
 dict(name='mf64_reg06',kind='mf',k=64,init=.07,reg=.06,lr=.008,decay=.99,epochs=100),
 dict(name='mf64_reg03',kind='mf',k=64,init=.07,reg=.03,lr=.008,decay=.99,epochs=90),
 dict(name='mf96_reg05',kind='mf',k=96,init=.06,reg=.05,lr=.008,decay=.99,epochs=100),
 dict(name='mf64_scaled',kind='mf',k=64,init=.07,reg=.04,lr=.008,decay=.99,scale=True,epochs=100),
 dict(name='fastpp32_reg05',kind='svdpp',k=32,init=.05,reg=.05,reg_y=.01,lr=.007,decay=.99,epochs=80),
 dict(name='fastpp64_reg05',kind='svdpp',k=64,init=.05,reg=.05,reg_y=.01,lr=.007,decay=.99,epochs=80),
 dict(name='fastpp64_reg08',kind='svdpp',k=64,init=.05,reg=.08,reg_y=.02,lr=.007,decay=.99,epochs=80),
]

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--task',type=int,default=1);pa.add_argument('--configs',nargs='*');args=pa.parse_args()
 file='tasks_1_2_train.csv' if args.task==1 else 'tasks_3_4_train.csv'
 arr,users,items=base.load(ROOT/'assignment1_2026'/file)
 order=np.random.default_rng(258).permutation(len(arr));valid=arr[order[:50000]];fit=arr[order[50000:]]
 results={};out=ROOT/'experiments/v3_latent_models';out.mkdir(exist_ok=True)
 for conf in CONFIGS:
  if args.configs and conf['name'] not in args.configs:continue
  path=out/f't{args.task}_{conf["name"]}.npz'
  if path.exists():print('skip existing',path.name,flush=True);continue
  start=time.time();model,res=train(fit,len(users),len(items),valid[20000:35000],conf)
  vp=model.predict(valid[:,0],valid[:,1],raw=True)
  res['elapsed_seconds']=time.time()-start;results[conf['name']]=res
  np.savez_compressed(path,val_pred=vp,**{k:v for k,v in model.state().items() if v is not None},config_json=json.dumps(res))
  (out/f't{args.task}_{conf["name"]}.json').write_text(json.dumps(res,indent=2))
  print('FINISHED',conf['name'],res['best_tune_mse'],res['elapsed_seconds'],flush=True)
 (ROOT/f'experiments/v3_latent_results_{args.task}.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
