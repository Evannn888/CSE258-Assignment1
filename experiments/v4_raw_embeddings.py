"""A frozen, core-fit spectral coordinate bank for nonlinear residual learning.

The spectral basis is never refitted at prediction time; this avoids feeding a
tree model new coordinates after an arbitrary SVD sign/rotation change.
"""
import sys,pickle,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import production_ratings as old
ROOT=Path(__file__).resolve().parents[1]


def embedding_features(U,s,V,qu,qi,rank):
    # Remove the leading popularity mode; retain the next strongest modes.
    order=np.argsort(s)[-rank-1:-1]
    u=U[:,order]*np.sqrt(s[order]);v=V[:,order]*np.sqrt(s[order])
    for x in [u,v]:
        norm=np.linalg.norm(x,axis=1,keepdims=True)
        np.divide(x,norm,out=x,where=norm>1e-8);x[norm[:,0]<=1e-8]=0
    u=np.vstack([u,np.zeros((1,rank))])[qu]
    v=np.vstack([v,np.zeros((1,rank))])[qi]
    return np.column_stack([u,v,u*v,np.abs(u-v)]).astype(np.float32)


def main():
    path=ROOT/'experiments/v4_community_embeddings_256.npz'
    while not path.exists():time.sleep(1)
    data=np.load(path)
    print('embedding keys',[(k,data[k].shape) for k in data.files],flush=True)
    d=np.load(ROOT/'experiments/v3_big_stage_1.npz');X,y,q=d['X'],d['y'],d['valid']
    nu,ni=30909,51007
    U=next(data[k] for k in data.files if data[k].ndim==2 and data[k].shape[0]==nu)
    V=next((data[k] for k in data.files if data[k].ndim==2 and data[k].shape[0]==ni),None)
    if V is None:V=next(data[k].T for k in data.files if data[k].ndim==2 and data[k].shape[1]==ni)
    s=next(data[k] for k in data.files if data[k].ndim==1 and len(data[k])==U.shape[1])
    tr=np.arange(150000);tr=tr[y[tr]>0]
    tu=np.arange(150000,175000);tu=tu[y[tu]>0]
    ridge=old.StandardizedRidge(500).fit(X[tr].astype(float),y[tr])
    raw=ridge.predict_raw(X.astype(float));oof=np.empty(len(tr))
    for a,b in KFold(5,shuffle=True,random_state=258).split(tr):
        r=old.StandardizedRidge(500).fit(X[tr[a]].astype(float),y[tr[a]])
        oof[b]=r.predict_raw(X[tr[b]].astype(float))
    baseline=np.load(ROOT/'experiments/v3_big_pred_1.npz')['pred']
    best=(float(np.mean((baseline[tu]-y[tu])**2)),None,None,baseline)
    records=[]
    for rank in [64,128]:
        F=embedding_features(U,s,V,q[:,0],q[:,1],rank)
        np.savez_compressed(ROOT/f'experiments/v4_raw_features_{rank}.npz',features=F)
        features=np.column_stack([X,F])
        for leaves in [31,63]:
            params=dict(max_leaf_nodes=leaves,max_iter=500,learning_rate=.04,min_samples_leaf=250,
                        l2_regularization=100.,early_stopping=False,random_state=258)
            m=HistGradientBoostingRegressor(**params).fit(features[tr,1:],y[tr]-oof)
            corr=m.predict(features[:,1:])
            for eta in [.5,1.]:
                pred=np.clip(raw+eta*corr,1,5);mse=float(np.mean((pred[tu]-y[tu])**2))
                rec=dict(rank=rank,params=params,eta=eta,tune_mse=mse)
                records.append(rec);print(rec,flush=True)
                if mse<best[0]:best=(mse,rec,m,pred)
    with (ROOT/'experiments/v4_raw_combiner.pkl').open('wb') as f:pickle.dump(dict(config=best[1],model=best[2],ridge=ridge),f)
    np.savez_compressed(ROOT/'experiments/v4_raw_pred.npz',pred=best[3])
    (ROOT/'experiments/v4/raw_report.json').write_text(json.dumps(dict(baseline_tune=float(np.mean((baseline[tu]-y[tu])**2)),best_tune=best[0],selected=best[1],trials=records),indent=2)+'\n')


if __name__=='__main__':main()
