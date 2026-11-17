import os, sys, json
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from numba import njit
import assignment1 as m

@njit(cache=True)
def knn(qu,qi,up,ui,ur,ip,iu,ir,uc,ic):
 n=len(qu);out=np.zeros((n,34),np.float64)
 present=np.zeros(len(uc),np.bool_); target=np.zeros(len(uc),np.float64)
 dot=np.zeros(len(uc),np.float64);su=np.zeros(len(uc),np.float64);sv=np.zeros(len(uc),np.float64);cnt=np.zeros(len(uc),np.float64)
 dotw=np.zeros(len(uc),np.float64);cw=np.zeros(len(uc),np.float64)
 for z in range(n):
  u=qu[z];i=qi[z]
  for p in range(ip[i],ip[i+1]):
   v=iu[p];present[v]=True;target[v]=ir[p]
  nums=np.zeros(16);dens=np.zeros(16)
  for p in range(up[u],up[u+1]):
   j=ui[p];ru=ur[p]
   if j==i:continue
   ij=0.;ii=0.;jj=0.;nc=0.
   for q in range(ip[j],ip[j+1]):
    v=iu[q]
    if present[v]:
     rv=ir[q];ri=target[v]
     dot[v]+=ru*rv;su[v]+=ru*ru;sv[v]+=rv*rv;cnt[v]+=1
     w=1./np.sqrt(ic[j]+10);dotw[v]+=ru*rv*w;cw[v]+=w
     ij+=ri*rv;ii+=ri*ri;jj+=rv*rv;nc+=1
   if nc<2:continue
   cos=ij/(np.sqrt(ii*jj)+1e-8)
   for k in range(4):
    shrink=(2.,5.,10.,30.)[k]
    s=cos*nc/(nc+shrink)
    nums[k]+=s*ru;dens[k]+=abs(s)
    ps=max(0.,s);nums[4+k]+=ps*ps*ru;dens[4+k]+=ps*ps
   s=ij/(nc+10.)
   nums[8]+=s*ru;dens[8]+=abs(s)
  for p in range(ip[i],ip[i+1]):
   v=iu[p];r=target[v];nc=cnt[v]
   if nc>=2:
    cos=dot[v]/(np.sqrt(su[v]*sv[v])+1e-8)
    for k in range(4):
     shrink=(2.,5.,10.,30.)[k]
     s=cos*nc/(nc+shrink)
     out[z,9+k]+=s*r;out[z,25+k]+=abs(s)
     ps=max(0.,s);out[z,13+k]+=ps*ps*r;out[z,29+k]+=ps*ps
    s=dot[v]/(nc+10)
    nums[9]+=s*r;dens[9]+=abs(s)
    s=dotw[v]/(cw[v]+2)
    nums[10]+=s*r;dens[10]+=abs(s)
   present[v]=False;target[v]=0.;dot[v]=0.;su[v]=0.;sv[v]=0.;cnt[v]=0.;dotw[v]=0.;cw[v]=0.
  for k in range(9):out[z,k]=nums[k]/(dens[k]+1.)
  for k in range(8):out[z,9+k]=out[z,9+k]/(out[z,25+k]+1.)
  out[z,17]=nums[9]/(dens[9]+1.);out[z,18]=nums[10]/(dens[10]+1.)
 return out[:,:19]

def design(a,nu,ni,qu,qi):
 uc=np.bincount(a[:,0],minlength=nu);ic=np.bincount(a[:,1],minlength=ni)
 mu,bu,bi=m.als_biases_custom(a,nu,ni,uc,ic,8,12)
 res=(a[:,3]-mu-bu[a[:,0]]-bi[a[:,1]]).astype(np.float32)
 uo=np.argsort(a[:,0],kind='stable');io=np.argsort(a[:,1],kind='stable')
 up=np.r_[0,uc.cumsum()];ip=np.r_[0,ic.cumsum()]
 f=knn(qu,qi,up,a[uo,1],res[uo],ip,a[io,0],res[io],uc,ic)
 # normalized residual variants may disentangle each user's rating dispersion.
 var=np.bincount(a[:,0],weights=res**2,minlength=nu)
 std=np.sqrt((var+5*.75)/(uc+5))
 resn=res/(std[a[:,0]]+.1)
 fn=knn(qu,qi,up,a[uo,1],resn[uo],ip,a[io,0],resn[io],uc,ic)
 return np.column_stack([f,fn,fn*std[qu,None]])

def main(task):
 a,u,i=m.load(Path('assignment1_2026')/('tasks_1_2_train.csv' if task==1 else 'tasks_3_4_train.csv'))
 order=np.random.default_rng(258).permutation(len(a));va=a[order[:50000]];tr=a[order[50000:]]
 fn=Path(f'experiments/ratings_knn_{task}.npy')
 if fn.exists():f=np.load(fn)
 else:f=design(tr,len(u),len(i),va[:,0],va[:,1]);np.save(fn,f)
 d=np.load(f'experiments/ratings_cache_{task}.npz');x=d['x'];p=d['preds']
 for name,xp in [('knn',np.column_stack([x,f])),('all',np.column_stack([x,p,f]))]:
  w=np.linalg.lstsq(xp[:25000],va[:25000,3],rcond=None)[0]
  pred=np.clip(xp[25000:]@w,0,5);y=va[25000:,3]
  print(name,'all',np.mean((pred-y)**2),'pos',np.mean((pred[y>0]-y[y>0])**2),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
