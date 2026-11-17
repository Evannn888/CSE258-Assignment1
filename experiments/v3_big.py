"""Large-meta rating candidate: 200k holdout, tune-only selection, no audit view.

Fits all base and additional feature models on the remaining interaction rows.
The final 25k holdout labels are saved but never scored here.
"""
from pathlib import Path
import os,sys,json,pickle,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
import assignment1 as base
import production_ratings as old
import v3_neighbors as neighbors
import v3_ordinal as ordinal
import v3_latent as latent
import v3_temporal_production as temporal

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments'


def log(message):
    print(time.strftime('%H:%M:%S'),message,flush=True)


def atomic_json(path,value):
    temp=Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def atomic_pickle(path,value):
    temp=Path(str(path)+'.tmp')
    with temp.open('wb') as f:pickle.dump(value,f,protocol=5)
    os.replace(temp,path)


def atomic_npz(path,**values):
    temp=Path(str(path)+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    os.replace(temp,path)


def fit_extra_features(fit,nu,ni,task,with_fast=True,seed=258):
    ordinal_configs=[(1.,2.),(3.,5.),(10.,15.)] if task==1 else [(1.,2.)]
    states=[]
    for reg_u,reg_i in ordinal_configs:
        states.append(ordinal.fit_ordinal(fit,nu,ni,reg_u,reg_i))
    joint=temporal.fit_features(fit,nu,ni) if task==3 else None
    pp=None
    if with_fast:
        cfg=next(dict(c) for c in latent.CONFIGS if c['name']=='fastpp64_reg08')
        pp,_=latent.train(fit,nu,ni,None,cfg,fixed_epochs=80,seed=seed)
    return dict(ordinal=states,joint=joint,fastpp=pp)


def extract_feature_blocks(old_state,extra_state,qu,qi,qt):
    X,groups=old.build_rating_features(old_state,qu,qi,qt)
    X=X[:,groups['stage5_plus_conf_gating']]
    blocks=[X]
    blocks.append(neighbors.extract_topk_features(old_state,qu,qi))
    blocks.extend(ordinal.ordinal_features(s,qu,qi) for s in extra_state['ordinal'])
    if extra_state['joint'] is not None:
        blocks.append(temporal.extract_features(extra_state['joint'],qu,qi,qt))
    if extra_state['fastpp'] is not None:
        blocks.append(extra_state['fastpp'].predict(qu,qi,raw=True)[:,None])
    widths=[int(b.shape[1]) for b in blocks]
    full=np.column_stack(blocks).astype(np.float32)
    if not np.all(np.isfinite(full)):raise ValueError('Nonfinite large-meta features')
    return full,widths,int(X.shape[1])


def fit_extended(fit,nu,ni,task,seed=258,with_fast=True):
    """Fixed feature learners; no held-out labels or research caches required."""
    cfg=old.RATING_CONFIGS[task]
    st=old.fit_rating_base(fit,nu,ni,task,seed=seed,
        fixed_epochs=cfg['best_epochs'],run_svdpp_grid=False)
    extras=fit_extra_features(fit,nu,ni,task,with_fast=with_fast,seed=seed)
    return dict(task=task,old_state=st,extra_state=extras)


def extract_extended(state,qu,qi,qt):
    """Return float32 full feature matrix; column zero remains the intercept."""
    X,widths,base_width=extract_feature_blocks(state['old_state'],state['extra_state'],qu,qi,qt)
    state['block_widths']=widths
    state['base_width']=base_width
    return X


def prepare_task(task,skip_fast=False):
    path=EXP/f'v3_big_{task}.npz'
    state_path=EXP/f'v3_big_state_{task}.pkl'
    if path.exists() and state_path.exists():
        log(f'Task {task}: loading completed atomic large-meta cache')
        return path
    started=time.time()
    fn='tasks_1_2_train.csv' if task==1 else 'tasks_3_4_train.csv'
    a,users,items=base.load(ROOT/'assignment1_2026'/fn)
    order=np.random.default_rng(258).permutation(len(a))
    valid,fit=a[order[:200000]],a[order[200000:]]
    nu,ni=len(users),len(items)
    log(f'Task {task}: fitting old base on {len(fit):,} rows')
    cfg=old.RATING_CONFIGS[task]
    state=old.fit_rating_base(fit,nu,ni,task,seed=258,fixed_epochs=cfg['best_epochs'],run_svdpp_grid=False)
    log(f'Task {task}: fitting ordinal/joint/fast SVD++ extras')
    extra=fit_extra_features(fit,nu,ni,task,with_fast=not skip_fast)
    log(f'Task {task}: extracting all 200,000 query features')
    X,widths,base_width=extract_feature_blocks(state,extra,valid[:,0],valid[:,1],valid[:,2])
    y=valid[:,3].astype(np.float64)
    log(f'Task {task}: evaluating unchanged 20k-meta baseline at new tune window')
    ridge,hgb=old.fit_fixed_rating_combiner(X[:20000,:base_width].astype(np.float64),y[:20000],cfg)
    baselinepred=old.predict_stacked(ridge,hgb,cfg['best_eta'],X[:,:base_width].astype(np.float64))
    bundle=dict(task=task,fit_rows=fit,users=users,items=items,old_state=state,extra_state=extra,
                base_width=base_width,block_widths=widths,split_seed=258)
    atomic_pickle(state_path,bundle)
    atomic_npz(path,X=X,y=y,valid=valid,baselinepred=baselinepred,base_width=np.asarray(base_width),block_widths=np.asarray(widths))
    info=dict(task=task,n_holdout=len(valid),n_fit=len(fit),feature_shape=list(X.shape),block_widths=widths,
              preparation_seconds=time.time()-started,skip_fast=skip_fast)
    atomic_json(EXP/f'v3_big_prepare_{task}.json',info)
    log(f'Task {task}: atomic feature/state cache complete: {info}')
    return path


def fit_large_meta(task,path):
    started=time.time()
    data=np.load(path)
    X=data['X'];y=data['y'];bp=data['baselinepred']
    tr=np.arange(150000);tr=tr[y[tr]>0]
    tune=np.arange(150000,175000);tune=tune[y[tune]>0]
    # The audit rows are deliberately not inspected or scored here.
    tune_y=y[tune]
    baseline=float(np.mean((bp[tune]-tune_y)**2))
    records=[];best=(float('inf'),None,None)
    log(f'Task {task}: baseline20k tune MSE={baseline:.8f}; meta positives={len(tr):,}')
    for reg in [500.,3000.,15000.]:
        ridge=old.StandardizedRidge(reg).fit(X[tr].astype(np.float64),y[tr])
        pred=ridge.predict(X[tune].astype(np.float64))
        mse=float(np.mean((pred-tune_y)**2))
        records.append(dict(kind='ridge',reg=reg,tune_mse=mse))
        log(f'Task {task}: Ridge reg={reg} tune={mse:.8f}')
        if mse<best[0]:best=(mse,reg,ridge)
    best_mse,best_reg,ridge=best
    log(f'Task {task}: preparing five-fold Ridge residuals with reg={best_reg}')
    xp=X[tr].astype(np.float64);yp=y[tr]
    oof=np.empty(len(tr),dtype=np.float64)
    for fold,(fit_idx,val_idx) in enumerate(KFold(n_splits=5,shuffle=True,random_state=258).split(xp)):
        rr=old.StandardizedRidge(best_reg).fit(xp[fit_idx],yp[fit_idx])
        oof[val_idx]=rr.predict_raw(xp[val_idx])
        log(f'Task {task}: OOF Ridge fold {fold+1}/5 done')
    del xp
    raw_tune=ridge.predict_raw(X[tune].astype(np.float64))
    selected=dict(kind='ridge',reg=best_reg,eta=0.,hgb=None)
    best_hgb=None
    for leaves in [31,63]:
        cfg=dict(max_leaf_nodes=leaves,max_iter=400,learning_rate=.04,min_samples_leaf=250,l2_regularization=100.,early_stopping=False,random_state=258)
        log(f'Task {task}: fitting residual HGB leaves={leaves}')
        hgb=HistGradientBoostingRegressor(**cfg).fit(X[tr,1:],yp-oof)
        correction=hgb.predict(X[tune,1:])
        for eta in [.5,1.]:
            pred=np.clip(raw_tune+eta*correction,1,5)
            mse=float(np.mean((pred-tune_y)**2))
            rec=dict(kind='ridge_hgb',reg=best_reg,eta=eta,hgb=cfg,tune_mse=mse)
            records.append(rec);log(f'Task {task}: HGB leaves={leaves} eta={eta} tune={mse:.8f}')
            if mse<best_mse:
                best_mse=mse;best_hgb=hgb;selected=rec.copy()
    log(f'Task {task}: configuration locked; predicting all holdout rows without scoring audit')
    raw=ridge.predict_raw(X.astype(np.float64))
    pp=np.clip(raw+(selected['eta']*best_hgb.predict(X[:,1:]) if best_hgb is not None else 0.),1,5)
    bundle=dict(ridge=ridge,hgb=best_hgb,config=selected)
    atomic_pickle(EXP/f'v3_big_combiner_{task}.pkl',bundle)
    atomic_npz(EXP/f'v3_big_pred_{task}.npz',pred=pp)
    report=dict(task=task,baseline20k_tune_mse=baseline,selected_tune_mse=best_mse,
        selected=selected,trials=records,meta_positive_rows=len(tr),tune_positive_rows=len(tune),
        n_features=X.shape[1],audit_viewed=False,fit_seconds=time.time()-started)
    atomic_json(EXP/f'v3_big_results_{task}.json',report)
    log(f'Task {task}: complete, tune={best_mse:.8f}, audit unviewed')
    return report


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--tasks',nargs='+',type=int,default=[1,3]);pa.add_argument('--skip-fast',action='store_true');pa.add_argument('--prepare-only',action='store_true')
    args=pa.parse_args()
    for task in args.tasks:
        path=prepare_task(task,args.skip_fast)
        if not args.prepare_only:fit_large_meta(task,path)


if __name__=='__main__':main()
