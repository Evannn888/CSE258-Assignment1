"""Stable sparse full-batch Adam residual/implicit history encoder.

For observed edges, both explicit and implicit target-self contributions are
subtracted in prediction AND every gradient. No query labels enter profiles.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,time,json,pickle,argparse
from pathlib import Path
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import assignment1 as base
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True)
def edge_values(H,Q,W,Y,rows,cols,ac,bc):
 result=np.zeros(len(rows),dtype=H.dtype)
 for n in range(len(rows)):
  u,i=rows[n],cols[n];z=0.
  for d in range(Q.shape[1]):z+=(H[u,d]-ac[n]*W[i,d]-bc[n]*Y[i,d])*Q[i,d]
  result[n]=z
 return result

@njit(cache=True)
def query_dot(H,Q,rows,cols):
 result=np.zeros(len(rows),np.float64)
 for n in range(len(rows)):
  u,i=rows[n],cols[n]
  if u>=len(H) or i>=len(Q):continue
  z=0.
  for d in range(Q.shape[1]):z+=H[u,d]*Q[i,d]
  result[n]=z
 return result


def loss_grad(params,A,B,rows,cols,ac,bc,target,penalty,E=None,use_p=False):
 """Unnormalized objective; divide loss by N only for reporting.

 The global scaling is immaterial to Adam. Penalty is degree-aware L2 and
 uses the same scale as the sum-of-squares data loss.
 """
 Q,W,Y,P=(params[k] for k in ['Q','W','Y','P'])
 H=A@W+B@Y+P
 pred=edge_values(H,Q,W,Y,rows,cols,ac,bc)
 err=pred-target
 if E is None:
  E=csr_matrix((err,(rows,cols)),shape=A.shape)
 else:E.data[:]=err
 G=E@Q
 self_a=np.bincount(cols,weights=err*ac,minlength=len(Q)).astype(Q.dtype)
 self_b=np.bincount(cols,weights=err*bc,minlength=len(Q)).astype(Q.dtype)
 grads={
  'Q':E.T@H-self_a[:,None]*W-self_b[:,None]*Y,
  'W':A.T@G-self_a[:,None]*Q,
  'Y':B.T@G-self_b[:,None]*Q,
  'P':G if use_p else np.zeros_like(P),
 }
 loss=.5*float(np.dot(err.astype(np.float64),err.astype(np.float64)))
 for key in params:
  if key=='P' and not use_p:continue
  coeff=penalty[key]
  grads[key]+=coeff[:,None]*params[key]
  loss+=.5*float(np.sum(coeff[:,None]*params[key].astype(np.float64)**2))
 return loss,grads,H,pred


def finite_difference_check(seed=258):
 rng=np.random.default_rng(seed)
 denseB=np.array([[1.,1.,0.,1.],[0.,1.,1.,1.],[1.,0.,1.,0.]])
 denseA=np.array([[.3,-.5,0.,.1],[0.,.4,-.1,0.],[.2,0.,-.6,0.]])
 A=csr_matrix(denseA);B=csr_matrix(denseB)
 rows=np.array([0,0,1,1,2]);cols=np.array([0,1,1,2,2]);ac=denseA[rows,cols];bc=denseB[rows,cols]
 target=rng.normal(size=len(rows))
 params={k:rng.normal(0,.1,(3 if k=='P' else 4,3)) for k in ['Q','W','Y','P']}
 penalty={k:np.linspace(.1,.3,len(v)) for k,v in params.items()}
 loss,g,_,_=loss_grad(params,A,B,rows,cols,ac,bc,target,penalty,use_p=True)
 worst=0.;nchecks=0;eps=1e-6
 for key,arr in params.items():
  for r in range(arr.shape[0]):
   for d in range(arr.shape[1]):
    orig=arr[r,d];arr[r,d]=orig+eps;l1=loss_grad(params,A,B,rows,cols,ac,bc,target,penalty,use_p=True)[0]
    arr[r,d]=orig-eps;l0=loss_grad(params,A,B,rows,cols,ac,bc,target,penalty,use_p=True)[0];arr[r,d]=orig
    numeric=(l1-l0)/(2*eps);error=abs(numeric-g[key][r,d]);worst=max(worst,error);nchecks+=1
 assert worst<2e-7,(worst,loss)
 # Changing target-edge source coefficients must not change that edge's output,
 # provided its corresponding input matrix entry is changed consistently.
 z0=loss_grad(params,A,B,rows,cols,ac,bc,target,penalty,use_p=True)[3]
 aa=A.toarray();aa[0,0]+=3.;newac=aa[rows,cols]
 z1=loss_grad(params,csr_matrix(aa),B,rows,cols,newac,bc,target,penalty,use_p=True)[3]
 assert abs(z1[0]-z0[0])<1e-10
 return dict(passed=True,coordinates=nchecks,max_absolute_gradient_error=worst,self_term_invariance=True)


class EncoderModel:
 def __init__(self,params,A,B,biases,config):
  self.params,self.A,self.B,self.biases,self.config=params,A,B,biases,config
 def features(self,qu,qi):
  Q,W,Y,P=(self.params[k] for k in ['Q','W','Y','P'])
  H=self.A@W+self.B@Y+P
  score=query_dot(H,Q,qu,qi)
  mu,bu,bi=self.biases
  bue=np.r_[bu,0.];bie=np.r_[bi,0.]
  raw=mu+bue[qu]+bie[qi]+score
  hn=np.r_[np.linalg.norm(H,axis=1),0.];qn=np.r_[np.linalg.norm(Q,axis=1),0.]
  return np.column_stack([raw,score,hn[qu],qn[qi]])
 def predict(self,qu,qi):return np.clip(self.features(qu,qi)[:,0],1,5)


def build_problem(fit,nu,ni,biases=None,degree_power=0.,use_explicit=True):
 a=fit[fit[:,3]>0];order=np.lexsort((a[:,1],a[:,0]));a=a[order]
 rows,cols=a[:,0].astype(np.int32),a[:,1].astype(np.int32)
 uc=np.bincount(rows,minlength=nu).astype(np.float64);ic=np.bincount(cols,minlength=ni).astype(np.float64)
 iu=np.bincount(fit[:,0],minlength=nu);ii=np.bincount(fit[:,1],minlength=ni)
 if biases is None:biases=base.als_biases_custom(a,nu,ni,uc,ic,8.,12.,20)
 mu,bu,bi=biases;target=(a[:,3]-mu-bu[rows]-bi[cols]).astype(np.float32)
 colscale=(ii+10.)**(-degree_power)
 ac=(target/np.sqrt(uc[rows]+5.)*colscale[cols]).astype(np.float32)
 bc=(colscale[cols]/np.sqrt(iu[rows]+5.)).astype(np.float32)
 if not use_explicit:ac=np.zeros_like(ac)
 A=csr_matrix((ac,(rows,cols)),shape=(nu,ni));A.sort_indices()
 all_b=(colscale[fit[:,1]]/np.sqrt(iu[fit[:,0]]+5.)).astype(np.float32)
 B=csr_matrix((all_b,(fit[:,0],fit[:,1])),shape=(nu,ni));B.sort_indices()
 ptr=np.r_[0,np.bincount(rows,minlength=nu).cumsum()].astype(np.int32)
 E=csr_matrix((np.zeros(len(rows),np.float32),cols.copy(),ptr),shape=(nu,ni))
 return dict(A=A,B=B,E=E,rows=rows,cols=cols,ac=ac,bc=bc,target=target,biases=biases,uc=uc,ic=ic)


def fit_encoder(fit,nu,ni,config,tune=None,biases=None,seed=258,fixed_epochs=None,progress=None):
 prob=build_problem(fit,nu,ni,biases,config.get('degree_power',0.),config.get('use_explicit',True));A,B=prob['A'],prob['B']
 rng=np.random.default_rng(seed);k=config['k'];dtype=np.float32
 params={'Q':rng.normal(0,.05,(ni,k)).astype(dtype),'W':np.zeros((ni,k),dtype),'Y':np.zeros((ni,k),dtype),'P':np.zeros((nu,k),dtype)}
 params['Q'][prob['ic']==0]=0.
 # Encoder penalties track source energy, decoder penalties target degree.
 penalties={
  'Q':(config['reg_q']*(prob['ic']+5.)).astype(dtype),
  'W':(config['reg_enc']*(np.asarray(A.power(2).sum(axis=0)).ravel()+.1)).astype(dtype),
  'Y':(config['reg_enc']*(np.asarray(B.power(2).sum(axis=0)).ravel()+.1)).astype(dtype),
  'P':(config.get('reg_p',.05)*(prob['uc']+5.)).astype(dtype),
 }
 m={name:np.zeros_like(v) for name,v in params.items()};v={name:np.zeros_like(val) for name,val in params.items()}
 best=1e10;best_params=None;best_ep=0;stale=0;history=[];started=time.time()
 model=EncoderModel(params,A,B,prob['biases'],config)
 maxep=fixed_epochs or config.get('epochs',200)
 for epoch in range(1,maxep+1):
  loss,grad,H,pred=loss_grad(params,A,B,prob['rows'],prob['cols'],prob['ac'],prob['bc'],prob['target'],penalties,prob['E'],config.get('use_p',False))
  if not np.isfinite(loss) or any(not np.all(np.isfinite(g)) for g in grad.values()):raise FloatingPointError('Non-finite encoder loss or gradient')
  lr=config['lr']*.995**(epoch-1)
  for key in params:
   if key=='P' and not config.get('use_p',False):continue
   m[key]*=.9;m[key]+=.1*grad[key]
   v[key]*=.999;v[key]+=.001*(grad[key]*grad[key])
   denom=np.sqrt(v[key]/(1-.999**epoch))+1e-8
   params[key]-=lr*(m[key]/(1-.9**epoch))/denom
  if epoch%5==0 or epoch==maxep:
   entry=dict(epoch=epoch,train_objective=loss/len(prob['target']),elapsed_seconds=time.time()-started)
   if tune is not None:
    tp=model.predict(tune[:,0],tune[:,1]);mse=float(np.mean((tp-tune[:,3])**2));entry['tune_mse']=mse
    if mse<best:
     best=mse;best_ep=epoch;best_params={name:z.copy() for name,z in params.items()};stale=0
    else:stale+=1
   history.append(entry)
   if progress is not None:progress(entry)
   if tune is not None and fixed_epochs is None and stale>=8:break
 if best_params is not None:model.params=best_params
 return model,dict(config=config,best_epoch=best_ep or maxep,best_tune_mse=None if tune is None else best,history=history,elapsed_seconds=time.time()-started)

CONFIGS=[
 dict(name='implicit64_conservative',k=64,reg_enc=20.,reg_q=.2,lr=.003,epochs=180,degree_power=0.,use_p=False,use_explicit=False),
 dict(name='encoder64',k=64,reg_enc=2.,reg_q=.05,lr=.015,epochs=180,degree_power=0.,use_p=False),
 dict(name='encoder128',k=128,reg_enc=4.,reg_q=.05,lr=.015,epochs=180,degree_power=.25,use_p=False),
 dict(name='encoder64_user',k=64,reg_enc=2.,reg_q=.05,reg_p=.05,lr=.015,epochs=180,degree_power=0.,use_p=True),
]

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--check-only',action='store_true');pa.add_argument('--configs',nargs='*');args=pa.parse_args()
 checks=finite_difference_check();print('NUMERICAL',checks,flush=True)
 (ROOT/'experiments/v4_encoder_gradient_check.json').write_text(json.dumps(checks,indent=2))
 if args.check_only:return
 with open(ROOT/'experiments/v3_big_state_1.pkl','rb') as f:state=pickle.load(f)
 z=np.load(ROOT/'experiments/v3_big_stage_1.npz');valid=z['valid'];tune=valid[150000:175000];tune=tune[tune[:,3]>0]
 out=ROOT/'experiments/v4_encoder';out.mkdir(exist_ok=True)
 for config in CONFIGS:
  if args.configs and config['name'] not in args.configs:continue
  if (out/f'{config["name"]}.npz').exists():continue
  print('START',config,flush=True)
  model,result=fit_encoder(state['fit_rows'],len(state['users']),len(state['items']),config,tune,state['old_state']['biases'][2],progress=lambda rec:print(config['name'],rec,flush=True))
  features=model.features(valid[:,0],valid[:,1]).astype(np.float32)
  np.savez_compressed(out/f'{config["name"]}.npz',features=features,pred=np.clip(features[:,0],1,5),config_json=json.dumps(result))
  with open(out/f'{config["name"]}_model.pkl','wb') as f:pickle.dump(model,f,protocol=5)
  (out/f'{config["name"]}.json').write_text(json.dumps(result,indent=2))
  print('FINISHED',config['name'],result['best_tune_mse'],result['best_epoch'],flush=True)


def load_encoder_model(path):
 class Loader(pickle.Unpickler):
  def find_class(self,module,name):
   if module=='__main__' and name=='EncoderModel':return EncoderModel
   return super().find_class(module,name)
 with open(path,'rb') as stream:return Loader(stream).load()

if __name__=='__main__':main()
