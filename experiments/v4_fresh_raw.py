"""Locked spectral-coordinate candidate on new, globally isolated cohorts."""
import sys,pickle,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import assignment1 as base
import production_ratings as old
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments'


def fit_reference(fit,nu,ni,rank=128):
    uc=np.bincount(fit[:,0],minlength=nu);ic=np.bincount(fit[:,1],minlength=ni)
    w=1/np.sqrt((uc[fit[:,0]]+10.)*(ic[fit[:,1]]+10.))
    X=csr_matrix((w,(fit[:,0],fit[:,1])),shape=(nu,ni))
    U,s,Vt=svds(X,k=257,random_state=258)
    order=np.argsort(s)[-rank-1:-1]
    u=U[:,order]*np.sqrt(s[order]);v=Vt.T[:,order]*np.sqrt(s[order])
    for z in [u,v]:
        norm=np.linalg.norm(z,axis=1,keepdims=True)
        np.divide(z,norm,out=z,where=norm>1e-8);z[norm[:,0]<=1e-8]=0
    return dict(U=np.vstack([u,np.zeros((1,rank))]),V=np.vstack([v,np.zeros((1,rank))]))


def reference_features(state,qu,qi):
    u,v=state['U'][qu],state['V'][qi]
    return np.column_stack([u,v,u*v,np.abs(u-v)]).astype(np.float32)


def fit_combiner(X,y):
    keep=y>0;xx=X[keep].astype(float);yy=y[keep]
    ridge=old.StandardizedRidge(500.).fit(xx[:,:315],yy)
    oof=np.empty(len(yy))
    for tr,va in KFold(5,shuffle=True,random_state=258).split(xx):
        r=old.StandardizedRidge(500.).fit(xx[tr,:315],yy[tr]);oof[va]=r.predict_raw(xx[va,:315])
    hgb=HistGradientBoostingRegressor(max_leaf_nodes=63,max_iter=500,learning_rate=.04,
        min_samples_leaf=250,l2_regularization=100.,early_stopping=False,random_state=258).fit(xx[:,1:],yy-oof)
    return ridge,hgb


def predict_combiner(model,X):
    r,h=model
    return np.clip(r.predict_raw(X[:,:315].astype(float))+h.predict(X[:,1:]),1,5)


def main():
    a,users,items=base.load(ROOT/'assignment1_2026/tasks_1_2_train.csv')
    control=np.load(EXP/'v4_sparse_control.npz')
    meta,tune,audit=control['meta_indices'],control['tune_indices'],control['audit_indices']
    safe=np.r_[tune,audit];mask=np.ones(len(a),bool);mask[np.r_[meta,safe]]=False
    ref=fit_reference(a[mask],len(users),len(items))
    with (EXP/'v4_raw_fresh_reference.pkl').open('wb') as f:pickle.dump(ref,f,protocol=5)
    fm=reference_features(ref,a[meta,0],a[meta,1]);fs=reference_features(ref,a[safe,0],a[safe,1])
    yd=a[meta,3].astype(float);yt=a[tune,3];keep=yt>0
    dense=np.zeros_like(control['Xmeta']);query=[]
    for k in range(3):
        z=np.load(EXP/f'v4_oof_fold{k}.npz');dense[z['meta_positions']]=z['Xmeta'];query.append(z['Xsafe'])
    records=[];predictions={}
    for name,xm,xslist in [('sparse_raw',control['Xmeta'],[control['Xsafe']]),('oof_raw',dense,query)]:
        print('FIT',name,flush=True)
        combined=np.column_stack([xm,fm]);model=fit_combiner(combined,yd)
        preds=np.vstack([predict_combiner(model,np.column_stack([xs,fs])) for xs in xslist])
        prediction=preds.mean(axis=0);mse=float(np.mean((prediction[:len(tune)][keep]-yt[keep])**2))
        records.append(dict(name=name,tune_mse=mse,individual_tune=[float(np.mean((p[:len(tune)][keep]-yt[keep])**2)) for p in preds]))
        print(records[-1],flush=True);predictions[name]=prediction
        with (EXP/f'v4_{name}_fresh_combiner.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
    baseline=np.load(EXP/'v4_control_predictions.npz')['pred'];oof=np.load(EXP/'v4_oof_predictions.npz')['pred']
    predictions['v3_control']=baseline;predictions['v3_oof']=oof
    best=(float('inf'),None)
    for name,p in predictions.items():
        score=float(np.mean((p[:len(tune)][keep]-yt[keep])**2))
        if score<best[0]:best=(score,dict(name=name,raw_weight=1.))
    # Only one small, predeclared blending check on the held-out tuning labels.
    for w in [.5,.75]:
        p=w*predictions['oof_raw']+(1-w)*oof
        score=float(np.mean((p[:len(tune)][keep]-yt[keep])**2));records.append(dict(name='oof_raw_v3_blend',raw_weight=w,tune_mse=score))
        if score<best[0]:best=(score,dict(name='oof_raw_v3_blend',raw_weight=w))
    np.savez_compressed(EXP/'v4_fresh_all_predictions.npz',**predictions,tune_indices=tune,audit_indices=audit)
    report=dict(trials=records,selected=best[1],best_tune=best[0],audit_viewed=False,
                frozen_hgb=dict(leaves=63,iterations=500,lr=.04),rank=128,seed=2029)
    (EXP/'v4/fresh_raw_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('LOCKED',report,flush=True)


if __name__=='__main__':main()
