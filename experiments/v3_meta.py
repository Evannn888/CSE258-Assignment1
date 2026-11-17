"""Tune fixed-feature residual stacking without inspecting audit performance."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import production_ratings as old

ROOT=Path(__file__).resolve().parents[1]


def run(task):
    path=ROOT/f'experiments/v3_base_{task}.npz'
    while not path.exists():
        time.sleep(1)
    data=np.load(path)
    X,y=data['X'],data['y'].astype(float)
    tr=np.arange(20000)[y[:20000]>0]
    tu=np.arange(20000,35000)[y[20000:35000]>0]
    Xtr,ytr=X[tr],y[tr]
    config=old.RATING_CONFIGS[task]
    ridge=old.StandardizedRidge(config['best_l2']).fit(Xtr,ytr)
    oof=np.empty(len(tr))
    for a,b in KFold(5,shuffle=True,random_state=258).split(Xtr):
        m=old.StandardizedRidge(config['best_l2']).fit(Xtr[a],ytr[a])
        oof[b]=m.predict_raw(Xtr[b])
    raw=ridge.predict_raw(X)
    baseline=data['pred']
    records=[]
    best=(np.mean((baseline[tu]-y[tu])**2),'v2',baseline,None)
    for leaves,iterations,minimum,reg in [(15,400,100,30),(31,500,100,100),(63,400,100,100),(15,800,200,100)]:
        params=dict(max_leaf_nodes=leaves,max_iter=iterations,min_samples_leaf=minimum,l2_regularization=reg,
                    learning_rate=.035,early_stopping=False,random_state=258)
        model=HistGradientBoostingRegressor(**params).fit(Xtr[:,1:],ytr-oof)
        residual=model.predict(X[:,1:])
        for eta in [.5,1.,1.5]:
            pred=np.clip(raw+eta*residual,1,5)
            mse=float(np.mean((pred[tu]-y[tu])**2))
            rec=dict(model='residual_hgb',params=params,eta=eta,tune_mse=mse)
            records.append(rec); print(task,rec,flush=True)
            if mse<best[0]:best=(mse,'residual_hgb',pred,rec)
    params=dict(n_estimators=160,min_samples_leaf=30,max_features=.7,n_jobs=2,random_state=258)
    model=ExtraTreesRegressor(**params).fit(Xtr[:,1:],ytr-oof)
    residual=model.predict(X[:,1:])
    for eta in [.5,1.]:
        pred=np.clip(raw+eta*residual,1,5)
        mse=float(np.mean((pred[tu]-y[tu])**2))
        rec=dict(model='residual_extra_trees',params=params,eta=eta,tune_mse=mse)
        records.append(rec); print(task,rec,flush=True)
        if mse<best[0]:best=(mse,'residual_extra_trees',pred,rec)
    np.savez_compressed(ROOT/f'experiments/v3_meta_{task}.npz',pred=best[2])
    result=dict(task=task,baseline_tune=float(np.mean((baseline[tu]-y[tu])**2)),best_tune=best[0],selected=best[3],trials=records)
    (ROOT/f'experiments/v3/meta_{task}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SELECTED',result,flush=True)


if __name__=='__main__':
    for task in [1,3]:run(task)
