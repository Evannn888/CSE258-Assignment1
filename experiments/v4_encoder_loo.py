"""Query-specific leave-one-out user centering for a sparse linear encoder."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,time,json,pickle,argparse
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import assignment1 as base
from v4_encoder import query_dot
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True)
def loo_edge_values(H0,HW,Q,W,Y,rows,cols,bu_loo,ac,ic,offset):
 result=np.empty(len(rows),dtype=H0.dtype)
 for n in range(len(rows)):
  u,i=rows[n],cols[n];z=offset[n]
  for d in range(Q.shape[1]):z+=(H0[u,d]-bu_loo[n]*HW[u,d]-ac[n]*W[i,d]-ic[n]*Y[i,d])*Q[i,d]
  result[n]=z
 return result


def loo_loss_grad(params,prob,penalty):
 Q,W,Y=(params[k] for k in ['Q','W','Y'])
 A,C,I=prob['A'],prob['C'],prob['I']
 H0=A@W+I@Y;HW=C@W
 pred=loo_edge_values(H0,HW,Q,W,Y,prob['rows'],prob['cols'],prob['bu_loo'],prob['ac'],prob['ic'],prob['offset'])
 err=pred-prob['target'];E=prob['E'];EB=prob['EB'];E.data[:]=err;EB.data[:]=err*prob['bu_loo']
 G=E@Q;GB=EB@Q
 sa=np.bincount(prob['cols'],weights=err*prob['ac'],minlength=len(Q)).astype(Q.dtype)
 si=np.bincount(prob['cols'],weights=err*prob['ic'],minlength=len(Q)).astype(Q.dtype)
 gradients={
  'Q':E.T@H0-EB.T@HW-sa[:,None]*W-si[:,None]*Y,
  'W':A.T@G-C.T@GB-sa[:,None]*Q,
  'Y':I.T@G-si[:,None]*Q,
 }
 loss=.5*float(np.dot(err.astype(np.float64),err.astype(np.float64)))
 for name in params:
  gradients[name]+=penalty[name][:,None]*params[name]
  loss+=.5*float(np.sum(penalty[name][:,None]*params[name].astype(np.float64)**2))
 return loss,gradients,pred


def loo_gradient_check(seed=258):
 rng=np.random.default_rng(seed);rows=np.array([0,0,1,1,2]);cols=np.array([0,1,1,2,2]);nu,ni=3,4
 aa=rng.normal(0,.3,(nu,ni));aa[rng.random((nu,ni))<.4]=0
 cc=(aa!=0).astype(float)*.2;ii=(rng.random((nu,ni))<.7).astype(float)*.3
 bu=rng.normal(size=len(rows));ac=aa[rows,cols]-bu*cc[rows,cols];ic=ii[rows,cols]
 ptr=np.r_[0,np.bincount(rows,minlength=nu).cumsum()]
 prob=dict(A=csr_matrix(aa),C=csr_matrix(cc),I=csr_matrix(ii),rows=rows,cols=cols,bu_loo=bu,ac=ac,ic=ic,
           offset=rng.normal(size=len(rows)),target=rng.normal(size=len(rows)),
           E=csr_matrix((np.zeros(len(rows)),cols.copy(),ptr),shape=(nu,ni)),EB=csr_matrix((np.zeros(len(rows)),cols.copy(),ptr),shape=(nu,ni)))
 params={k:rng.normal(0,.1,(ni,3)) for k in ['Q','W','Y']};penalty={k:np.linspace(.1,.3,ni) for k in params}
 loss,g,_=loo_loss_grad(params,prob,penalty);worst=0;count=0;eps=1e-6
 for key,p in params.items():
  for ix in np.ndindex(p.shape):
   x=p[ix];p[ix]=x+eps;l1=loo_loss_grad(params,prob,penalty)[0];p[ix]=x-eps;l0=loo_loss_grad(params,prob,penalty)[0];p[ix]=x
   error=abs((l1-l0)/(2*eps)-g[key][ix]);worst=max(worst,error);count+=1
 assert worst<2e-7,(worst,loss)
 # The user LOO mean for target i is independent of r_ui itself.
 d=np.array([.2,-.6,.9]);lam=8.;before=(d.sum()-d[0])/(len(d)-1+lam)
 d[0]+=4.;after=(d.sum()-d[0])/(len(d)-1+lam)
 assert abs(before-after)<1e-12
 # Forward prediction must also be unchanged by perturbing its own star value.
 fit=np.array([[0,0,1,2],[0,1,2,4],[1,1,1,3],[1,2,2,5],[2,2,1,4],[2,3,2,2]],dtype=np.int64)
 fixed_biases=(3.,np.zeros(3),np.zeros(4))
 original=build_loo_problem(fit,3,4,fixed_biases)
 changed=fit.copy();changed[0,3]+=1
 modified=build_loo_problem(changed,3,4,fixed_biases)
 p0=loo_loss_grad(params,original,penalty)[2]
 p1=loo_loss_grad(params,modified,penalty)[2]
 self_error=float(abs(p0[0]-p1[0]));assert self_error<1e-6,self_error
 return dict(passed=True,coordinates=count,max_absolute_gradient_error=float(worst),loo_center_target_invariant=True,self_rating_forward_max_error=self_error)


def build_loo_problem(fit,nu,ni,biases=None,user_shrink=8.):
 pos=fit[fit[:,3]>0];oo=np.lexsort((pos[:,1],pos[:,0]));pos=pos[oo]
 rows,cols=pos[:,0].astype(np.int32),pos[:,1].astype(np.int32)
 uc=np.bincount(rows,minlength=nu);icount=np.bincount(cols,minlength=ni);iu=np.bincount(fit[:,0],minlength=nu)
 if biases is None:biases=base.als_biases_custom(pos,nu,ni,uc,icount,8.,12.,20)
 mu,_,bi=biases;d=pos[:,3]-mu-bi[cols];S=np.bincount(rows,weights=d,minlength=nu)
 bu_full=S/(uc+user_shrink)
 bu_loo=((S[rows]-d)/(uc[rows]-1+user_shrink)).astype(np.float32)
 norm=1/np.sqrt(uc[rows]+5.)
 aa=(d*norm).astype(np.float32);cc=norm.astype(np.float32)
 isc=(1/np.sqrt(iu[rows]+5.)).astype(np.float32)
 A=csr_matrix((aa,(rows,cols)),shape=(nu,ni));C=csr_matrix((cc,(rows,cols)),shape=(nu,ni))
 iv=(1/np.sqrt(iu[fit[:,0]]+5.)).astype(np.float32);I=csr_matrix((iv,(fit[:,0],fit[:,1])),shape=(nu,ni))
 ptr=np.r_[0,uc.cumsum()].astype(np.int32)
 E=csr_matrix((np.zeros(len(rows),np.float32),cols.copy(),ptr),shape=(nu,ni));EB=E.copy()
 return dict(A=A,C=C,I=I,rows=rows,cols=cols,uc=uc,icount=icount,bu_full=bu_full.astype(np.float32),bu_loo=bu_loo,
             ac=((d-bu_loo)*norm).astype(np.float32),ic=isc,offset=(mu+bi[cols]+bu_loo).astype(np.float32),
             target=pos[:,3].astype(np.float32),E=E,EB=EB,biases=(mu,bi),user_shrink=user_shrink)


class LOOEncoderModel:
 def __init__(self,params,problem,config):
  self.params,self.problem,self.config=params,problem,config
 def features(self,qu,qi):
  Q,W,Y=(self.params[k] for k in ['Q','W','Y']);p=self.problem
  H=p['A']@W+p['I']@Y-p['bu_full'][:,None]*(p['C']@W)
  dot=query_dot(H,Q,qu,qi);mu,bi=p['biases'];bue=np.r_[p['bu_full'],0.];bie=np.r_[bi,0.]
  raw=mu+bue[qu]+bie[qi]+dot
  hn=np.r_[np.linalg.norm(H,axis=1),0.];qn=np.r_[np.linalg.norm(Q,axis=1),0.]
  return np.column_stack([raw,dot,hn[qu],qn[qi]])
 def predict(self,qu,qi):return np.clip(self.features(qu,qi)[:,0],1,5)


def fit_encoder_loo(fit,nu,ni,config,tune=None,biases=None,seed=258,fixed_epochs=None,progress=None):
 prob=build_loo_problem(fit,nu,ni,biases,config.get('user_shrink',8.));rng=np.random.default_rng(seed);k=config['k']
 params={'Q':rng.normal(0,.05,(ni,k)).astype(np.float32),'W':np.zeros((ni,k),np.float32),'Y':np.zeros((ni,k),np.float32)}
 params['Q'][prob['icount']==0]=0.
 # Source penalty is based on fully centered energy, not raw user means.
 center_data=(prob['target']-prob['biases'][0]-prob['biases'][1][prob['cols']]-prob['bu_full'][prob['rows']])/np.sqrt(prob['uc'][prob['rows']]+5.)
 enc_energy=np.bincount(prob['cols'],weights=center_data**2,minlength=ni)
 penalties={'Q':(config['reg_q']*(prob['icount']+5.)).astype(np.float32),
            'W':(config['reg_enc']*(enc_energy+.1)).astype(np.float32),
            'Y':(config['reg_enc']*(np.asarray(prob['I'].power(2).sum(axis=0)).ravel()+.1)).astype(np.float32)}
 m={name:np.zeros_like(x) for name,x in params.items()};v={name:np.zeros_like(x) for name,x in params.items()}
 best=1e9;beststate=None;bestep=0;stale=0;hist=[];start=time.time();model=LOOEncoderModel(params,prob,config)
 for ep in range(1,(fixed_epochs or config['epochs'])+1):
  loss,g,pred=loo_loss_grad(params,prob,penalties)
  if not np.isfinite(loss) or any(not np.all(np.isfinite(a)) for a in g.values()):raise FloatingPointError('LOO encoder non-finite')
  lr=config['lr']*.995**(ep-1)
  for key in params:
   m[key]*=.9;m[key]+=.1*g[key];v[key]*=.999;v[key]+=.001*g[key]**2
   params[key]-=lr*(m[key]/(1-.9**ep))/(np.sqrt(v[key]/(1-.999**ep))+1e-8)
  if ep%5==0:
   rec=dict(epoch=ep,train_objective=loss/len(prob['target']),elapsed_seconds=time.time()-start)
   if tune is not None:
    pred=model.predict(tune[:,0],tune[:,1]);mse=float(np.mean((pred-tune[:,3])**2));rec['tune_mse']=mse
    if mse<best:best=mse;bestep=ep;stale=0;beststate={key:a.copy() for key,a in params.items()}
    else:stale+=1
   hist.append(rec)
   if progress is not None:progress(rec)
   if tune is not None and fixed_epochs is None and stale>=8:break
 if beststate is not None:model.params=beststate
 return model,dict(config=config,best_epoch=bestep or fixed_epochs or config['epochs'],best_tune_mse=None if tune is None else best,history=hist,elapsed_seconds=time.time()-start)

CONFIGS=[dict(name='loo64_moderate',k=64,reg_enc=2.,reg_q=.05,lr=.005,epochs=180,user_shrink=8.),
         dict(name='loo64_strong',k=64,reg_enc=20.,reg_q=.2,lr=.003,epochs=180,user_shrink=8.)]

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--check-only',action='store_true');args=pa.parse_args()
 check=loo_gradient_check();print('LOO NUMERICAL',check,flush=True)
 (ROOT/'experiments/v4_encoder_loo_gradient_check.json').write_text(json.dumps(check,indent=2))
 if args.check_only:return
 with open(ROOT/'experiments/v3_big_state_1.pkl','rb') as f:state=pickle.load(f)
 data=np.load(ROOT/'experiments/v3_big_stage_1.npz');valid=data['valid'];tune=valid[150000:175000];tune=tune[tune[:,3]>0]
 out=ROOT/'experiments/v4_encoder';out.mkdir(exist_ok=True)
 for config in CONFIGS:
  if (out/f'{config["name"]}.npz').exists():continue
  print('LOO START',config,flush=True)
  model,result=fit_encoder_loo(state['fit_rows'],len(state['users']),len(state['items']),config,tune,state['old_state']['biases'][2],progress=lambda rec:print(config['name'],rec,flush=True))
  feat=model.features(valid[:,0],valid[:,1]).astype(np.float32)
  np.savez_compressed(out/f'{config["name"]}.npz',features=feat,pred=np.clip(feat[:,0],1,5),config_json=json.dumps(result))
  with open(out/f'{config["name"]}_model.pkl','wb') as f:pickle.dump(model,f,protocol=5)
  (out/f'{config["name"]}.json').write_text(json.dumps(result,indent=2))
  print('LOO FINISHED',config['name'],result['best_tune_mse'],result['best_epoch'],flush=True)


def load_encoder_loo_model(path):
 class Loader(pickle.Unpickler):
  def find_class(self,module,name):
   if module=='__main__' and name=='LOOEncoderModel':return LOOEncoderModel
   return super().find_class(module,name)
 with open(path,'rb') as stream:return Loader(stream).load()

if __name__=='__main__':main()
