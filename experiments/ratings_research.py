import os, sys, time, json
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('NUMBA_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from numba import njit
from sklearn.ensemble import HistGradientBoostingRegressor
import assignment1 as m

@njit(cache=True)
def epoch(a,order,p,q,bu,bi,mu,lr,reg):
 for z in order:
  u,i,r=a[z,0],a[z,1],a[z,3]
  dot=0.
  for k in range(p.shape[1]):dot+=p[u,k]*q[i,k]
  e=r-mu-bu[u]-bi[i]-dot
  bu[u]+=lr*(e-.02*bu[u]); bi[i]+=lr*(e-.02*bi[i])
  for k in range(p.shape[1]):
   pu=p[u,k]; qi=q[i,k]
   p[u,k]+=lr*(e*qi-reg*pu);q[i,k]+=lr*(e*pu-reg*qi)

def train(a,nu,ni,valid,rank,reg,positive=False,n_epochs=100):
 if positive:a=a[a[:,3]>0]
 uc=np.bincount(a[:,0],minlength=nu);ic=np.bincount(a[:,1],minlength=ni)
 mu,bu,bi=m.als_biases_custom(a,nu,ni,uc,ic,5,8)
 rng=np.random.default_rng(258)
 p=rng.normal(0,.08,(nu,rank)).astype(np.float32);q=rng.normal(0,.08,(ni,rank)).astype(np.float32)
 bu=bu.astype(np.float32);bi=bi.astype(np.float32)
 best=100;best_e=0;pred_best=None
 mask=valid[:25000,3]>0 if positive else np.ones(25000,bool)
 for e in range(1,n_epochs+1):
  epoch(a,rng.permutation(len(a)),p,q,bu,bi,mu,.020*(.992**(e-1)),reg)
  if e%5==0:
   pred=np.clip(mu+bu[valid[:,0]]+bi[valid[:,1]]+(p[valid[:,0]]*q[valid[:,1]]).sum(1),0,5)
   err=np.mean((pred[:25000][mask]-valid[:25000,3][mask])**2)
   if err<best:best=err;best_e=e;pred_best=pred.copy()
   if e%10==0:print('MF',rank,reg,positive,e,'select_mse',round(err,6),'best',best_e,flush=True)
   if e-best_e>=20:break
 print('MF BEST',rank,reg,positive,best_e,best,flush=True)
 return pred_best,best_e

def score(name,pred,y):
 mask=y>0
 v={'all':float(np.mean((pred-y)**2)),'positive':float(np.mean((pred[mask]-y[mask])**2)),'zeros':float(np.mean(pred[~mask]**2))}
 print(name,v,flush=True);return v

def main(task):
 cache=Path('experiments')/f'ratings_cache_{task}.npz'
 a,users,items=m.load(Path('assignment1_2026')/('tasks_1_2_train.csv' if task==1 else 'tasks_3_4_train.csv'))
 order=np.random.default_rng(258).permutation(len(a));va=a[order[:50000]];tr=a[order[50000:]]
 nu,ni=len(users),len(items)
 if cache.exists():
  d=np.load(cache);x=d['x']; preds=d['preds']; names=list(d['names']);epochs=list(d['epochs'])
 else:
  mod,_,_=m.train_rating(tr,nu,ni,26,None)
  x,_=m.build_rating_design(tr,nu,ni,va[:,0],va[:,1],va[:,2],mod,task)
  preds=np.empty((50000,0));names=[];epochs=[]
  np.savez(cache,x=x,preds=preds,names=names,epochs=epochs)
 result={}
 w=np.linalg.lstsq(x[:25000],va[:25000,3],rcond=None)[0]
 result['baseline']=score('baseline',np.clip(x[25000:]@w,0,5),va[25000:,3])
 for rank,reg,pos in [(64,.04,False),(64,.08,False),(32,.03,True),(64,.07,True)]:
  name=f'mf_{rank}_{reg}_{pos}'
  if name not in names:
   p,ep=train(tr,nu,ni,va,rank,reg,pos)
   names.append(name);epochs.append(ep);preds=np.column_stack([preds,p]);np.savez(cache,x=x,preds=preds,names=names,epochs=epochs)
  xp=np.column_stack([x,preds])
  w=np.linalg.lstsq(xp[:25000],va[:25000,3],rcond=None)[0]
  result[name]=score('stack '+name,np.clip(xp[25000:]@w,0,5),va[25000:,3])
 xp=np.column_stack([x,preds])
 for leaf in [7,15,31]:
  h=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=leaf,l2_regularization=50,learning_rate=.04,min_samples_leaf=100,random_state=258)
  h.fit(xp[:25000],va[:25000,3])
  result[f'hgb_{leaf}']=score(f'hgb {leaf}',np.clip(h.predict(xp[25000:]),0,5),va[25000:,3])
 Path(f'experiments/ratings_metrics_{task}.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main(int(sys.argv[1]))
