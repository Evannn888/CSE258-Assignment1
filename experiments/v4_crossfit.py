"""Task 1 dense-base OOF meta experiment with globally isolated tune/audit.

The frozen v3 reference source lives beside this file. All artifacts are v4_*
and audit MSE is deliberately never computed here. Primary validation averages
the three complete meta predictions, not the nonlinear model's input features.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys, json, time, gc, pickle, hashlib, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import v4_reference_assignment1 as v3

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments'
SEED=2029


def log(message):
    print(time.strftime('%H:%M:%S'),message,flush=True)


def atomic_npz(path,**data):
    tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    os.replace(tmp,path)


def atomic_pickle(path,value):
    tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,protocol=5)
    os.replace(tmp,path)


def atomic_json(path,value):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def make_split(a):
    order=np.random.default_rng(SEED).permutation(len(a))
    # Audit labels are never used by this script's fitting or scoring routines.
    audit=order[:25000]
    tune=order[25000:50000]
    meta=order[50000:200000]
    return meta,tune,audit


def query_features(state,rows):
    # Task 1 has no query timestamp; do not pass hidden query times accidentally.
    X,_=v3.ratings_extract_extended(state,rows[:,0],rows[:,1],np.zeros(len(rows),np.int64))
    assert X.shape[1]==315 and np.all(np.isfinite(X))
    return X.astype(np.float32,copy=False)


def prepare(a,users,items):
    meta,tune,audit=make_split(a)
    safe=np.r_[tune,audit]
    nu,ni=len(users),len(items)
    for k,positions in enumerate(np.array_split(np.arange(len(meta)),3)):
        path=EXP/f'v4_oof_fold{k}.npz'
        if path.exists():
            log(f'Fold {k}: reuse complete cache')
            continue
        started=time.time()
        held=meta[positions]
        mask=np.ones(len(a),bool);mask[safe]=False;mask[held]=False
        assert not mask[safe].any() and not mask[held].any()
        fit=a[mask]
        log(f'Fold {k}: fit {len(fit):,}/{len(a):,} interactions ({100*len(fit)/len(a):.2f}%)')
        state=v3.ratings_fit_extended(fit,nu,ni,1)
        log(f'Fold {k}: extract {len(held)+len(safe):,} held/query features')
        rows=a[np.r_[held,safe]]
        X=query_features(state,rows)
        Xm,Xs=X[:len(held)],X[len(held):]
        atomic_npz(path,Xmeta=Xm,Xsafe=Xs,meta_positions=positions,meta_indices=held,
                   tune_indices=tune,audit_indices=audit,fit_count=np.asarray(len(fit)),
                   elapsed_seconds=np.asarray(time.time()-started))
        del state,fit,rows,X,Xm,Xs
        gc.collect()
        log(f'Fold {k}: saved atomically in {time.time()-started:.1f}s')
    # Control removes all 200k rows, reproducing v3's 89.3%-dense base protocol
    # on exactly the same seed2029 targets. Only training density changes.
    path=EXP/'v4_sparse_control.npz'
    if not path.exists():
        started=time.time()
        mask=np.ones(len(a),bool);mask[np.r_[meta,safe]]=False
        log(f'Sparse control: fit {mask.sum():,}/{len(a):,} interactions')
        state=v3.ratings_fit_extended(a[mask],nu,ni,1)
        X=query_features(state,a[np.r_[meta,safe]])
        atomic_npz(path,Xmeta=X[:len(meta)],Xsafe=X[len(meta):],meta_indices=meta,
                   tune_indices=tune,audit_indices=audit,fit_count=np.asarray(mask.sum()),
                   elapsed_seconds=np.asarray(time.time()-started))
        del state,X
        gc.collect()
        log(f'Sparse control saved in {time.time()-started:.1f}s')
    return meta,tune,audit


def score_meta(a,meta,tune,audit):
    cfg=json.loads(json.dumps(v3.ratings_V3_CONFIGS[1]))
    cfg.pop('tune_mse',None)  # historical seed258 score is not a hyperparameter
    ym=a[meta,3].astype(np.float64)
    yt=a[tune,3].astype(np.float64)
    keep=yt>0
    # Build one out-of-fold feature row for each meta label. Audit/tune labels
    # have been excluded from every base learner, including their graph edges.
    Xmeta=np.zeros((len(meta),315),np.float32)
    Xsafe=[]
    for k in range(3):
        z=np.load(EXP/f'v4_oof_fold{k}.npz')
        Xmeta[z['meta_positions']]=z['Xmeta']
        Xsafe.append(z['Xsafe'])
    log('Fitting frozen v3 meta model on dense-base OOF rows')
    model=v3.ratings_fit_meta(Xmeta,ym,cfg)
    individual=np.vstack([v3.ratings_predict_meta(model,x,cfg) for x in Xsafe])
    primary=individual.mean(axis=0)
    mean_features=v3.ratings_predict_meta(model,np.mean(np.stack(Xsafe),axis=0),cfg)
    dense_tune=float(np.mean((primary[:len(tune)][keep]-yt[keep])**2))
    feature_mean_tune=float(np.mean((mean_features[:len(tune)][keep]-yt[keep])**2))
    per_fold=[float(np.mean((p[:len(tune)][keep]-yt[keep])**2)) for p in individual]
    atomic_pickle(EXP/'v4_oof_combiner.pkl',dict(model=model,config=cfg))
    # Labels for audit are withheld from prediction artifacts; root can join
    # original data indices after the configuration is frozen for final audit.
    atomic_npz(EXP/'v4_oof_predictions.npz',tune_indices=tune,audit_indices=audit,
               pred=primary,individual_pred=individual,mean_feature_pred=mean_features)
    del Xmeta,model,individual
    gc.collect()
    log('Fitting same frozen meta model on sparse-control feature rows')
    z=np.load(EXP/'v4_sparse_control.npz')
    control_model=v3.ratings_fit_meta(z['Xmeta'],ym,cfg)
    control_pred=v3.ratings_predict_meta(control_model,z['Xsafe'],cfg)
    sparse_tune=float(np.mean((control_pred[:len(tune)][keep]-yt[keep])**2))
    atomic_pickle(EXP/'v4_control_combiner.pkl',dict(model=control_model,config=cfg))
    atomic_npz(EXP/'v4_control_predictions.npz',tune_indices=tune,audit_indices=audit,pred=control_pred)
    differences=(primary[:len(tune)][keep]-yt[keep])**2-(control_pred[:len(tune)][keep]-yt[keep])**2
    result=dict(seed=SEED,meta_rows=len(meta),meta_positive_rows=int((ym>0).sum()),
                tune_rows=len(tune),tune_positive_rows=int(keep.sum()),audit_rows=len(audit),
                audit_viewed=False,primary_aggregation='mean of complete predictions from each fold',
                dense_oof_tune_mse=dense_tune,sparse_control_tune_mse=sparse_tune,
                dense_per_fold_tune_mse=per_fold,dense_single_mean_tune_mse=float(np.mean(per_fold)),
                ensemble_gain_over_mean_single=float(np.mean(per_fold)-dense_tune),
                mean_feature_tune_mse=feature_mean_tune,
                paired_tune_change=float(differences.mean()),
                paired_tune_standard_error=float(differences.std(ddof=1)/np.sqrt(len(differences))),
                config=cfg,source_sha256=hashlib.sha256((EXP/'v4_reference_assignment1.py').read_bytes()).hexdigest(),
                limitations=['Meta rows are correlated through other-fold base training labels.',
                    'OOF Ridge residuals condition on fixed cross-fitted features; this is not full nested cross-fitting.',
                    'A single full-data base for final test prediction differs from the 3-model validation ensemble.',
                    'Full-fit density is still higher than the 94.6% fold base density.'])
    atomic_json(EXP/'v4_oof_results.json',result)
    log(json.dumps(result,indent=2))


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--prepare-only',action='store_true');pa.add_argument('--score-only',action='store_true');args=pa.parse_args()
    a,users,items=v3.load(ROOT/'assignment1_2026/tasks_1_2_train.csv')
    meta,tune,audit=make_split(a)
    if not args.score_only:prepare(a,users,items)
    if not args.prepare_only:score_meta(a,meta,tune,audit)


if __name__=='__main__':main()
