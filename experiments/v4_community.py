"""Genre-conditioned rating style from a fit-only implicit spectral graph.

No audit selection. Community statistics are label supervised on fit rows only;
all embedding coordinates are obtained without held-out ratings or query edges.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans
from scipy.special import softmax
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import production_ratings

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments'


def atomic_npz(path,**values):
    tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f:np.savez(f,**values)
    os.replace(tmp,path)


def stable_unit_rows(x):
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norms,out=np.zeros_like(x),where=norms>1e-7)


def fit_implicit_embedding(a,nu,ni,rank=256):
    uc=np.bincount(a[:,0],minlength=nu)
    ic=np.bincount(a[:,1],minlength=ni)
    values=1./np.sqrt((uc[a[:,0]]+5.)*(ic[a[:,1]]+10.))
    matrix=csr_matrix((values,(a[:,0],a[:,1])),shape=(nu,ni))
    U,S,VT=svds(matrix,k=rank+1,random_state=258)
    ordering=np.argsort(S)
    U,S,VT=U[:,ordering],S[ordering],VT[ordering]
    return dict(U=U.astype(np.float32),V=VT.T.astype(np.float32),S=S,
                user_counts=uc,item_counts=ic)


def get_embedding():
    path=EXP/'v4_community_embeddings_256.npz'
    if path.exists():return dict(np.load(path))
    with (EXP/'v3_big_state_1.pkl').open('rb') as f:bank=pickle.load(f)
    state=bank['old_state']
    start=time.time()
    emb=fit_implicit_embedding(bank['fit_rows'],state['nu'],state['ni'])
    atomic_npz(path,**emb)
    print('EMBEDDING',path,'shape',emb['U'].shape,emb['V'].shape,'seconds',time.time()-start,flush=True)
    return emb


def select_community_vectors(embedding,rank):
    # Exclude the largest singular component, which mainly reflects activity.
    s=embedding['S'][-rank-1:-1]
    u=stable_unit_rows(embedding['U'][:,-rank-1:-1]*np.sqrt(s)[None,:])
    i=stable_unit_rows(embedding['V'][:,-rank-1:-1]*np.sqrt(s)[None,:])
    return u,i


def _cluster(values,k):
    good=np.linalg.norm(values,axis=1)>1e-7
    model=MiniBatchKMeans(n_clusters=k,random_state=258,n_init=1,max_iter=40,
                         batch_size=2048,init_size=4096,max_no_improvement=5)
    model.fit(values[good])
    centers=stable_unit_rows(model.cluster_centers_)
    scores=values@centers.T
    top=np.argpartition(scores,-3,axis=1)[:,-3:]
    selected=np.take_along_axis(scores,top,axis=1)
    labels=np.argmax(scores,axis=1)
    labels[~good]=k
    top[~good]=k;selected[~good]=0.
    probability=softmax(12.*selected,axis=1)
    probability[~good]=0.
    return dict(labels=np.r_[labels,k],top=np.vstack([top,np.full((1,3),k)]),
                probability=np.vstack([probability,np.zeros((1,3))]),centers=centers)


def fit_community_statistics(old_state,embedding,rank=128,k=128):
    uvec,ivec=select_community_vectors(embedding,rank)
    us=_cluster(uvec,k);its=_cluster(ivec,k)
    state=dict(rank=rank,k=k,user_clusters=us,item_clusters=its,
               nu=old_state['nu'],ni=old_state['ni'])
    return update_community_statistics(state,old_state)


def update_community_statistics(state,old_state):
    a=old_state['a_rated'];res=old_state['res'].astype(np.float64)
    std=old_state['u_std']
    u=a[:,0];i=a[:,1]
    uc=state['user_clusters']['labels'][u]
    ic=state['item_clusters']['labels'][i]
    k=state['k']+1;nu=state['nu'];ni=state['ni']
    normalized=res/(std[u]+.1)
    for tag,groups,size in [('user_itemcluster',u*k+ic,(nu+1)*k),
                            ('item_usercluster',i*k+uc,(ni+1)*k),
                            ('cocluster',uc*k+ic,k*k)]:
        state[tag+'_count']=np.bincount(groups,minlength=size).reshape((-1,k)).astype(np.float32)
        state[tag+'_sum']=np.bincount(groups,weights=res,minlength=size).reshape((-1,k)).astype(np.float32)
        state[tag+'_sum_norm']=np.bincount(groups,weights=normalized,minlength=size).reshape((-1,k)).astype(np.float32)
        state[tag+'_sumsq']=np.bincount(groups,weights=res**2,minlength=size).reshape((-1,k)).astype(np.float32)
    state['user_std']=std
    return state


def extract_community_features(state,qu,qi):
    us=state['user_clusters'];its=state['item_clusters'];uc=us['labels'][qu];ic=its['labels'][qi]
    selectors=[('user_itemcluster',qu,ic),('item_usercluster',qi,uc),('cocluster',uc,ic)]
    std=state['user_std'][qu]
    cols=[]
    for tag,row,col in selectors:
        count=state[tag+'_count'][row,col]
        total=state[tag+'_sum'][row,col]
        norm=state[tag+'_sum_norm'][row,col]
        square=state[tag+'_sumsq'][row,col]
        for shrink in [1.,5.,20.]:
            cols.extend([total/(count+shrink),norm/(count+shrink)*(std+.1)])
        mean=total/(count+1.)
        cols.extend([np.log1p(count),np.maximum(0.,square/(count+1.)-mean**2)])
    for tag,query,clusters in [('user_itemcluster',qu,its),('item_usercluster',qi,us)]:
        target=qi if tag=='user_itemcluster' else qu
        top=clusters['top'][target];prob=clusters['probability'][target]
        count=state[tag+'_count'][query[:,None],top]
        total=state[tag+'_sum'][query[:,None],top]
        norm=state[tag+'_sum_norm'][query[:,None],top]
        for shrink in [1.,5.,20.]:
            cols.append(np.sum(prob*total/(count+shrink),axis=1))
            cols.append(np.sum(prob*norm/(count+shrink),axis=1)*(std+.1))
        cols.append(np.sum(prob*np.log1p(count),axis=1))
    result=np.column_stack(cols).astype(np.float32)
    result[(qu>=state['nu']) | (qi>=state['ni'])]=0.
    return result


def add_soft_community_weights(state,embedding):
    uvec,ivec=select_community_vectors(embedding,state['rank'])
    weights={}
    for tag,vec,centers in [('user',uvec,state['user_clusters']['centers']),
                            ('item',ivec,state['item_clusters']['centers'])]:
        score=vec@centers.T
        score-=score.max(axis=1,keepdims=True)
        good=np.linalg.norm(vec,axis=1)>1e-7
        for kappa in [12,24]:
            w=np.exp(kappa*score).astype(np.float32)
            w[~good]=0.
            weights[f'{tag}{kappa}']=np.pad(w,((0,1),(0,1)))
    state['soft_weights']=weights
    return state


def extract_soft_community_features(state,qu,qi):
    result=np.zeros((len(qu),28),np.float32)
    std=state['user_std'][qu]+.1
    for start in range(0,len(qu),4096):
        end=min(len(qu),start+4096);a=qu[start:end];b=qi[start:end]
        columns=[]
        for tag,row,target,kind in [('user_itemcluster',a,b,'item'),('item_usercluster',b,a,'user')]:
            count=state[tag+'_count'][row]
            total=state[tag+'_sum'][row]
            norm=state[tag+'_sum_norm'][row]
            for kappa in [12,24]:
                w=state['soft_weights'][f'{kind}{kappa}'][target]
                den=np.sum(w*count,axis=1)
                num=np.sum(w*total,axis=1)
                nnum=np.sum(w*norm,axis=1)
                for shrink in [.5,2.,10.]:
                    columns.extend([num/(den+shrink),nnum/(den+shrink)*std[start:end]])
                columns.append(np.log1p(den))
        result[start:end]=np.column_stack(columns)
    result[(qu>=state['nu']) | (qi>=state['ni'])]=0.
    return result


def compare_with_frozen(features,tag):
    """Augment Ridge while retaining the frozen old nonlinear correction."""
    data=np.load(EXP/'v3_big_stage_1.npz');X=data['X'];y=data['y']
    with (EXP/'v3_big_combiner_1.pkl').open('rb') as f:bundle=pickle.load(f)
    old_ridge=bundle['ridge'];old_hgb=bundle['hgb'];eta=bundle['config']['eta']
    tr=np.arange(150000)[y[:150000]>0]
    tu=np.arange(150000,175000)[y[150000:175000]>0]
    correction=eta*old_hgb.predict(X[:,1:]) if old_hgb is not None else np.zeros(len(y))
    oldpred=np.clip(old_ridge.predict_raw(X.astype(float))+correction,1,5)
    baseline=float(np.mean((oldpred[tu]-y[tu])**2))
    xx=np.column_stack([X,features])
    records=[];best=(baseline,None,None)
    for reg in [500.,3000.,15000.]:
        model=production_ratings.StandardizedRidge(reg).fit(xx[tr].astype(float),y[tr])
        raw=model.predict_raw(xx[tu].astype(float))
        pred=np.clip(raw+correction[tu],1,5)
        mse=float(np.mean((pred-y[tu])**2))
        rec=dict(reg=reg,tune_mse=mse);records.append(rec)
        if mse<best[0]:best=(mse,rec,model)
    result=dict(tag=tag,baseline_tune=baseline,best_tune=best[0],selected=best[1],trials=records)
    (EXP/f'v4_community_{tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)
    return result


def retrain_selected_meta(tag='r256_k256',reg=3000.):
    from sklearn.model_selection import KFold
    from sklearn.ensemble import HistGradientBoostingRegressor
    data=np.load(EXP/'v3_big_stage_1.npz');F=np.load(EXP/f'v4_community_features_{tag}.npz')['features']
    X=np.column_stack([data['X'],F]).astype(np.float32);y=data['y']
    tr=np.arange(150000)[y[:150000]>0]
    tu=np.arange(150000,175000)[y[150000:175000]>0]
    xx=X[tr].astype(float);yy=y[tr]
    oof=np.empty(len(tr));start=time.time()
    for a,b in KFold(5,shuffle=True,random_state=258).split(xx):
        rr=production_ratings.StandardizedRidge(reg).fit(xx[a],yy[a]);oof[b]=rr.predict_raw(xx[b])
    ridge=production_ratings.StandardizedRidge(reg).fit(xx,yy)
    config=dict(max_leaf_nodes=31,max_iter=400,learning_rate=.04,min_samples_leaf=250,
                l2_regularization=100.,early_stopping=False,random_state=258)
    hgb=HistGradientBoostingRegressor(**config).fit(X[tr,1:],yy-oof)
    pred=np.clip(ridge.predict_raw(X.astype(float))+hgb.predict(X[:,1:]),1,5)
    result=dict(tag=tag,reg=reg,hgb=config,tune_mse=float(np.mean((pred[tu]-y[tu])**2)),
                baseline_tune=.5306373790895934,seconds=time.time()-start,audit_viewed=False)
    atomic_npz(EXP/f'v4_community_retrained_{tag}.npz',pred=pred)
    (EXP/f'v4_community_retrained_{tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    with (EXP/f'v4_community_combiner_{tag}.pkl').open('wb') as f:pickle.dump(dict(ridge=ridge,hgb=hgb,config=result),f,protocol=5)
    print('RETRAINED',json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--embeddings-only',action='store_true');args=parser.parse_args()
    emb=get_embedding()
    if args.embeddings_only:return
    with (EXP/'v3_big_state_1.pkl').open('rb') as f:bank=pickle.load(f)
    d=np.load(EXP/'v3_big_stage_1.npz');v=d['valid'];old=bank['old_state']
    for rank,k in [(128,64),(128,128),(256,128),(256,256)]:
        start=time.time();tag=f'r{rank}_k{k}'
        state=fit_community_statistics(old,emb,rank=rank,k=k)
        F=extract_community_features(state,v[:,0],v[:,1])
        assert np.isfinite(F).all()
        atomic_npz(EXP/f'v4_community_features_{tag}.npz',features=F)
        with (EXP/f'v4_community_state_{tag}.pkl').open('wb') as f:pickle.dump(state,f,protocol=5)
        print('PREPARED',tag,F.shape,'seconds',time.time()-start,flush=True)
        compare_with_frozen(F,tag)


if __name__=='__main__':main()
