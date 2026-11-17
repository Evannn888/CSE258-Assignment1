"""Cumulative logistic user/item effects for conditional 1--5 star ratings."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,time,json,argparse
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import assignment1 as base
import production_ratings as old
ROOT=Path(__file__).resolve().parents[1]

def fit_ordinal(fit,nu,ni,reg_u=3.,reg_i=5.,iterations=20):
 a=fit[fit[:,3]>0];uu,ii=a[:,0],a[:,1]
 uc=np.bincount(uu,minlength=nu);ic=np.bincount(ii,minlength=ni)
 mu=np.zeros(4);bu=np.zeros((nu,4));bi=np.zeros((ni,4))
 for k in range(4):
  target=(a[:,3]>=k+2).astype(float);prior=target.mean();mu[k]=np.log(prior/(1-prior))
  for it in range(iterations):
   prob=expit(mu[k]+bu[uu,k]+bi[ii,k]);h=prob*(1-prob)
   gu=np.bincount(uu,weights=target-prob,minlength=nu)-reg_u*bu[:,k]
   hu=np.bincount(uu,weights=h,minlength=nu)+reg_u
   bu[:,k]+=np.clip(gu/hu,-1.,1.)
   prob=expit(mu[k]+bu[uu,k]+bi[ii,k]);h=prob*(1-prob)
   gi=np.bincount(ii,weights=target-prob,minlength=ni)-reg_i*bi[:,k]
   hi=np.bincount(ii,weights=h,minlength=ni)+reg_i
   bi[:,k]+=np.clip(gi/hi,-1.,1.)
 return dict(mu=mu,bu=bu,bi=bi)

def ordinal_features(state,qu,qi):
 bu=np.vstack([state['bu'],np.zeros((1,4))])[qu]
 bi=np.vstack([state['bi'],np.zeros((1,4))])[qi]
 prob=expit(state['mu'][None,:]+bu+bi)
 mean=1+prob.sum(axis=1)
 return np.column_stack([mean,prob,bu,bi])

def run(task):
 path=ROOT/f'experiments/v3_base_{task}.npz'
 while not path.exists():time.sleep(1)
 data=np.load(path);X,y,v,bp=data['X'],data['y'],data['valid'],data['pred']
 arr,us,its=base.load(ROOT/'assignment1_2026'/('tasks_1_2_train.csv' if task==1 else 'tasks_3_4_train.csv'))
 order=np.random.default_rng(258).permutation(len(arr));fit=arr[order[50000:]]
 tr=np.arange(20000)[y[:20000]>0];tu=np.arange(20000,35000)[y[20000:35000]>0]
 conf=old.RATING_CONFIGS[task];oldrr=old.StandardizedRidge(conf['best_l2']).fit(X[tr],y[tr]);raw=oldrr.predict_raw(X)
 baseline=float(np.mean((bp[tu]-y[tu])**2));records=[];best=(baseline,None,None,None)
 feats=[]
 for ru,ri in [(1.,2.),(3.,5.),(10.,15.),(30.,40.)]:
  st=time.time();state=fit_ordinal(fit,len(us),len(its),ru,ri)
  f=ordinal_features(state,v[:,0],v[:,1]);feats.append(f)
  stand=float(np.mean((f[tu,0]-y[tu])**2))
  for featuremode in ['prob','full','cumulative']:
   if featuremode=='prob':zz=f[:,:5]
   elif featuremode=='full':zz=f
   else:zz=np.column_stack(feats)
   xx=np.column_stack([X,zz])
   for l2 in [40.,400.,2000.]:
    rr=old.StandardizedRidge(l2).fit(xx[tr],y[tr]);delta=rr.predict_raw(xx)-raw
    for scale in [.5,1.]:
     pred=np.clip(bp+scale*delta,1,5);mse=float(np.mean((pred[tu]-y[tu])**2))
     rec=dict(reg_u=ru,reg_i=ri,featuremode=featuremode,l2=l2,scale=scale,tune=mse,standalone=stand)
     records.append(rec)
     if mse<best[0]:best=(mse,rec,np.column_stack(feats).copy(),pred.copy())
  np.savez_compressed(ROOT/f'experiments/v3_latent_models/t{task}_ordinal_{int(ru)}.npz',features=f,**state)
  print(task,ru,ri,'standalone',stand,'best',best[:2],'secs',time.time()-st,flush=True)
 np.savez_compressed(ROOT/f'experiments/v3_ordinal_{task}.npz',features=best[2],pred=best[3])
 (ROOT/f'experiments/v3_ordinal_{task}.json').write_text(json.dumps(dict(baseline_tune=baseline,best_tune=best[0],selected=best[1],records=records),indent=2))

if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--task',type=int,default=1);args=pa.parse_args();run(args.task)
