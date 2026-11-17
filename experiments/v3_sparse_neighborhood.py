"""Observed-rating-only supervised sparse explicit/implicit neighborhoods."""
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import production_ratings

ROOT=Path(__file__).resolve().parents[1]


@njit(cache=True)
def select_sparse_neighbors(up,ui,ip,iu,ic,k=60,cap=80):
    ni=len(ic)
    neighbors=np.full((ni,k),-1,dtype=np.int32)
    support=np.zeros((ni,k),dtype=np.float32)
    counts=np.zeros(ni,dtype=np.int32)
    touched=np.empty(ni,dtype=np.int32)
    for i in range(ni):
        nt=0
        for p in range(ip[i],ip[i+1]):
            u=iu[p];s=up[u];e=up[u+1]
            step=max(1,(e-s+cap-1)//cap)
            for h in range(s,e,step):
                j=ui[h]
                if j==i:continue
                if counts[j]==0:touched[nt]=j;nt+=1
                counts[j]+=1
        candidates=np.empty(nt,dtype=np.int32)
        scores=np.empty(nt,dtype=np.float64)
        nc=0
        for z in range(nt):
            j=touched[z]
            if counts[j]>=3:
                candidates[nc]=j
                scores[nc]=counts[j]/np.sqrt(ic[j]+5.)
                nc+=1
        ordering=np.argsort(-scores[:nc])
        for h in range(min(k,nc)):
            j=candidates[ordering[h]]
            neighbors[i,h]=j;support[i,h]=counts[j]
        for z in range(nt):counts[touched[z]]=0
    return neighbors,support


@njit(cache=True)
def sparse_neighborhood_epoch(order,up,ui,ur,neighbors,w,c,lr,reg_w,reg_c):
    ni=neighbors.shape[0]
    present=np.zeros(ni,np.bool_);res=np.zeros(ni,np.float64)
    for u in order:
        s=up[u];e=up[u+1]
        for p in range(s,e):present[ui[p]]=True;res[ui[p]]=ur[p]
        for p in range(s,e):
            i=ui[p];count=0;raw=0.
            for h in range(neighbors.shape[1]):
                j=neighbors[i,h]
                if j>=0 and j!=i and present[j]:
                    count+=1
                    raw+=res[j]*w[i,h]+c[i,h]
            if count==0:continue
            inv=1./np.sqrt(count)
            error=ur[p]-raw*inv
            for h in range(neighbors.shape[1]):
                j=neighbors[i,h]
                if j>=0 and j!=i and present[j]:
                    w[i,h]+=lr*(error*res[j]*inv-reg_w*w[i,h])
                    c[i,h]+=lr*(error*inv-reg_c*c[i,h])
        for p in range(s,e):present[ui[p]]=False;res[ui[p]]=0.


@njit(cache=True)
def sparse_neighborhood_predict(qu,qi,up,ui,ur,neighbors,w,c):
    ni=neighbors.shape[0];nu=len(up)-1
    order=np.argsort(qu)
    features=np.zeros((len(qu),7),np.float64)
    present=np.zeros(ni,np.bool_);res=np.zeros(ni,np.float64)
    current=-1
    for pos in order:
        u=qu[pos];i=qi[pos]
        if u>=nu or i>=ni:continue
        if u!=current:
            if current>=0:
                for p in range(up[current],up[current+1]):present[ui[p]]=False;res[ui[p]]=0.
            for p in range(up[u],up[u+1]):present[ui[p]]=True;res[ui[p]]=ur[p]
            current=u
        count=0;rw=0.;rc=0.
        for h in range(neighbors.shape[1]):
            j=neighbors[i,h]
            if j>=0 and j!=i and present[j]:
                count+=1;rw+=res[j]*w[i,h];rc+=c[i,h]
        if count:
            for h in range(3):
                power=(.25,.5,.75)[h]
                denominator=count**power
                features[pos,h*2]=rw/denominator
                features[pos,h*2+1]=rc/denominator
            features[pos,6]=np.log1p(count)
    return features


def fit_sparse_neighborhood(state,neighbors=None,epochs=10,reg=.1,lr=.025,seed=258,valid=None):
    if neighbors is None:
        neighbors,_=select_sparse_neighbors(state['u_pos_ptr'],state['u_pos_items'],state['i_pos_ptr'],state['i_pos_users'],state['i_pos_cnt'])
    weights=np.zeros(neighbors.shape,np.float32);implicit=np.zeros_like(weights)
    model=dict(neighbors=neighbors,weights=weights,implicit=implicit,reg=reg,epochs=epochs)
    rng=np.random.default_rng(seed)
    trajectory=[]
    for epoch in range(1,epochs+1):
        sparse_neighborhood_epoch(rng.permutation(state['nu']),state['u_pos_ptr'],state['u_pos_items'],state['u_pos_res'],neighbors,
                                  weights,implicit,lr*(.98**(epoch-1)),reg,reg)
        if valid is not None and epoch in [2,5,10,15,20]:
            f=extract_sparse_neighborhood_features(model,state,valid[:,0],valid[:,1])
            trajectory.append((epoch,f))
            print('epoch',epoch,'reg',reg,'featurestd',f.std(axis=0),flush=True)
    return model,trajectory


def extract_sparse_neighborhood_features(model,state,qu,qi):
    return sparse_neighborhood_predict(qu,qi,state['u_pos_ptr'],state['u_pos_items'],state['u_pos_res'],model['neighbors'],model['weights'],model['implicit'])


def main():
    import argparse
    import v3_neighbors
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,default=1)
    args=parser.parse_args();task=args.task
    with (ROOT/f'experiments/v3_state_{task}.pkl').open('rb') as f:state=pickle.load(f)
    valid=np.load(ROOT/f'experiments/v3_base_{task}.npz')['valid']
    start=time.time()
    neighbors,support=select_sparse_neighbors(state['u_pos_ptr'],state['u_pos_items'],state['i_pos_ptr'],state['i_pos_users'],state['i_pos_cnt'])
    print('GRAPH',task,'seconds',time.time()-start,'edges',int((neighbors>=0).sum()),flush=True)
    outputs=[]
    for reg in [.1,1.]:
        model,trajectory=fit_sparse_neighborhood(state,neighbors,epochs=10,reg=reg,valid=valid)
        features=np.column_stack([entry[1] for entry in trajectory])
        tag=f'sparse_reg{reg}'
        result=v3_neighbors.residual_ridge_tune(task,features,tag)
        outputs.append(result)
        np.savez_compressed(ROOT/f'experiments/v3_sparse_{task}_reg{reg}.npz',features=features,neighbors=neighbors,
                            weights=model['weights'],implicit=model['implicit'],support=support)
    print('TOTAL_SECONDS',time.time()-start,flush=True)


if __name__=='__main__':main()
