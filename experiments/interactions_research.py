"""Leakage-free symmetric nonlinear comparison of existing interaction features.
Run from repo root with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2.
"""
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import assignment1 as a1
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'

def context(a,nu,ni,qu,qi1,qi2):
    uc=np.r_[np.bincount(a[:,0],minlength=nu),0]
    ic=np.r_[np.bincount(a[:,1],minlength=ni),0]
    c1=np.log1p(ic[qi1]); c2=np.log1p(ic[qi2])
    return np.column_stack([np.log1p(uc[qu]),c1+c2,np.abs(c1-c2),np.minimum(c1,c2),np.maximum(c1,c2)])

def get_cache(task):
    path=OUT/f'interactions_cache_{task}.npz'
    if path.exists(): return dict(np.load(path))
    fname={2:'tasks_1_2_train.csv',4:'tasks_3_4_train.csv',5:'task_5_train.csv'}[task]
    a,u,i=a1.load(ROOT/'assignment1_2026'/fname); nu,ni=len(u),len(i)
    tr,va=a1.make_interaction_split(a,nu,ni,sequential=task==5)
    x,se,bp=a1.build_interaction_design(task,tr,nu,ni,va[:,0],va[:,1],va[:,2],va[:,3],25)
    co=context(tr,nu,ni,va[:,0],va[:,1],va[:,2])
    tests=a1.read_test(ROOT/'assignment1_2026'/f'task_{task}_test.csv')
    qu=np.array([u.get(r['user'],nu) for r in tests]); qi1=np.array([i.get(r['item1'],ni) for r in tests]); qi2=np.array([i.get(r['item2'],ni) for r in tests])
    qt=np.array([int(r['timestamp']) if task==4 else 0 for r in tests])
    xt,st,bt=a1.build_interaction_design(task,a,nu,ni,qu,qi1,qi2,qt,25)
    ct=context(a,nu,ni,qu,qi1,qi2)
    d=dict(x=x,seen=se,bpr=bp,context=co,xt=xt,st=st,bt=bt,ct=ct,va=va)
    np.savez_compressed(path,**d)
    return d

def expand(x,c,sign=1): return np.column_stack([sign*x,c])
def fit_hgb(x,c,leaves,reg):
    model=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=leaves,learning_rate=.06,l2_regularization=reg,min_samples_leaf=40,early_stopping=False,random_state=258)
    model.fit(np.vstack([expand(x,c),expand(x,c,-1)]),np.r_[np.ones(len(x)),np.zeros(len(x))])
    return model

def margin(model,x,c): return .5*(model.decision_function(expand(x,c))-model.decision_function(expand(x,c,-1)))
def acc(z): return float(np.mean((z>0)+.5*(z==0)))

def research(task):
    d=get_cache(task); x=d['x']; co=d['context']; seen=d['seen']; n=len(x)
    # Independent tune and audit quarters; model choices never use audit labels.
    idx=np.random.default_rng(258).permutation(n); tr,tu,ev=np.split(idx,[n//2,3*n//4])
    rms=np.sqrt(np.mean(x[tr]**2,axis=0))+1e-8
    w=a1.fit_pairwise_logistic(np.clip(x[tr]/rms,-8,8))
    baseline=np.clip(x/rms,-8,8)@w
    rows=[]; best=None
    for leaves,reg in [(7,10),(15,10),(15,100),(31,100)]:
        model=fit_hgb(x[tr],co[tr],leaves,reg)
        hgb=margin(model,x,co)
        for blend in [0,.5,1]:
            z=(1-blend)*baseline+blend*hgb
            row=dict(leaves=leaves,reg=reg,blend=blend,tune_acc=acc(z[tu]+100*seen[tu]))
            rows.append(row); print('TUNE',task,row,flush=True)
            if best is None or row['tune_acc']>best[0]['tune_acc']: best=row,z,model
    selected,z,model=best
    metrics=dict(task=task,n_train=len(tr),n_tune=len(tu),n_audit=len(ev),selection=selected,baseline_audit=acc(baseline[ev]+100*seen[ev]),candidate_audit=acc(z[ev]+100*seen[ev]),baseline_tune=acc(baseline[tu]+100*seen[tu]),tuning=rows,disagreements=int(np.sum((z[ev]>0)!=(baseline[ev]>0))))
    # Compare source implementation on identical historical holdout, for diagnostics.
    oldhalf=n//2; rmsold=np.sqrt(np.mean(x[:oldhalf]**2,axis=0))+1e-8
    oldw=a1.fit_pairwise_logistic(np.clip(x[:oldhalf]/rmsold,-8,8))
    metrics['legacy_original_holdout']=acc(np.clip(x[oldhalf:]/rmsold,-8,8)@oldw+100*seen[oldhalf:])
    print('AUDIT',json.dumps(metrics),flush=True)
    (OUT/f'interactions_results_{task}.json').write_text(json.dumps(metrics,indent=2))
    # Full-fit outputs use fixed training transformations and candidate-order antisymmetry.
    rmsfull=np.sqrt(np.mean(x*x,axis=0))+1e-8
    wf=a1.fit_pairwise_logistic(np.clip(x/rmsfull,-8,8))
    zlog=np.clip(d['xt']/rmsfull,-8,8)@wf
    mf=fit_hgb(x,co,selected['leaves'],selected['reg'])
    ztree=margin(mf,d['xt'],d['ct'])
    final=(1-selected['blend'])*zlog+selected['blend']*ztree+100*d['st']
    tests=a1.read_test(ROOT/'assignment1_2026'/f'task_{task}_test.csv')
    pred=[r['item1'] if z>=0 else r['item2'] for r,z in zip(tests,final)]
    a1.write_predictions(ROOT/'assignment1_2026'/f'task_{task}_test.csv',OUT/f'interactions_task_{task}_predictions.csv',pred)
    np.savez_compressed(OUT/f'interactions_audit_{task}.npz',indices=ev,baseline=baseline[ev],candidate=z[ev],seen=seen[ev],test_margin=final)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=int,nargs='+',default=[2,4,5]);args=p.parse_args()
    for task in args.tasks: research(task)
