"""Local spectral ridge regression instead of residual mean pooling."""
import json,pickle,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import production_ratings as old
ROOT=Path(__file__).resolve().parents[1]


@njit(cache=True)
def local_coefficients(ptr, indices, residual, embedding, penalties):
    n=len(ptr)-1;k=embedding.shape[1]
    out=np.zeros((n+1,len(penalties),k),np.float64)
    for u in range(n):
        s,e=ptr[u],ptr[u+1]
        if s==e:continue
        H=embedding[indices[s:e]].astype(np.float64)
        r=residual[s:e].astype(np.float64)
        if e-s < k:
            C=H@H.T
            for l in range(len(penalties)):
                A=C.copy()
                for j in range(len(C)):A[j,j]+=penalties[l]
                out[u,l]=H.T@np.linalg.solve(A,r)
        else:
            C=H.T@H; b=H.T@r
            for l in range(len(penalties)):
                A=C.copy()
                for j in range(k):A[j,j]+=penalties[l]
                out[u,l]=np.linalg.solve(A,b)
    return out


def fit_extract(state, qu, qi, rank=64):
    a=state['a_implicit'];nu,ni=state['nu'],state['ni']
    invu=1/np.sqrt(state['u_imp_cnt']+10.)
    invi=1/np.sqrt(state['i_imp_cnt']+10.)
    X=csr_matrix((invu[a[:,0]]*invi[a[:,1]],(a[:,0],a[:,1])),shape=(nu,ni))
    U,s,Vt=svds(X,k=rank,random_state=258)
    features=[];fitted=[]
    penalties=np.array([.3,1.,5.,20.])
    for exponent in [0.,.5]:
        V=(Vt.T*s**exponent)
        norms=np.linalg.norm(V,axis=1,keepdims=True)
        V=np.divide(V,norms,out=np.zeros_like(V),where=norms>1e-8)
        coefficients=local_coefficients(state['u_pos_ptr'],state['u_pos_items'],state['u_pos_res'],V,penalties)
        V=np.vstack([V,np.zeros((1,rank))])
        features.append(np.einsum('nlk,nk->nl',coefficients[qu],V[qi]))
        fitted.append((coefficients,V))
    return np.column_stack(features),fitted


def run(task):
    with (ROOT/f'experiments/v3_state_{task}.pkl').open('rb') as f:state=pickle.load(f)
    d=np.load(ROOT/f'experiments/v3_base_{task}.npz');X,y,v=d['X'],d['y'],d['valid']
    columns=[]
    for rank in [32,64]:
        st=time.time();z,_=fit_extract(state,v[:,0],v[:,1],rank)
        columns.append(z);print(task,rank,'features seconds',time.time()-st,flush=True)
    Z=np.column_stack(columns)
    tr=np.arange(20000)[y[:20000]>0];tu=np.arange(20000,35000)[y[20000:35000]>0]
    records=[]
    for l2 in [10.,100.,1000.]:
        r=old.StandardizedRidge(l2=l2).fit(np.column_stack([X[tr],Z[tr]]),y[tr])
        p=np.clip(r.predict_raw(np.column_stack([X[tu],Z[tu]])),1,5)
        mse=float(np.mean((p-y[tu])**2));records.append(dict(l2=l2,tune=mse))
        print(task,records[-1],flush=True)
    np.savez_compressed(ROOT/f'experiments/v3_spectral_{task}.npz',features=Z)
    (ROOT/f'experiments/v3/spectral_{task}.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    for task in [1,3]:run(task)
