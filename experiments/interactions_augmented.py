"""Predefined user-neighbor feature extension and train/tune/audit comparison."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import interactions_research as ir
import interactions_neighbors as inn
import numpy as np

def run(task):
    d=ir.get_cache(task); nd=inn.cache(task)
    x0=d['x']; x=np.column_stack([x0,nd['x']]); c=np.column_stack([d['context'],nd['c']])
    xt=np.column_stack([d['xt'],nd['xt']]);ct=np.column_stack([d['ct'],nd['ct']])
    n=len(x);idx=np.random.default_rng(258).permutation(n);tr,tu,ev=np.split(idx,[n//2,3*n//4]);seen=d['seen']
    r0=np.sqrt(np.mean(x0[tr]**2,axis=0))+1e-8;w0=ir.a1.fit_pairwise_logistic(np.clip(x0[tr]/r0,-8,8));base=np.clip(x0/r0,-8,8)@w0
    rms=np.sqrt(np.mean(x[tr]**2,axis=0))+1e-8
    configs=[];best=None
    for reg in [10,100]:
        w=ir.a1.fit_pairwise_logistic(np.clip(x[tr]/rms,-8,8),l2=reg);zl=np.clip(x/rms,-8,8)@w
        for blend in [0,.5,1]:
            z=(1-blend)*base+blend*zl
            row=dict(kind='linear',linear_reg=reg,blend=blend,tune=ir.acc(z[tu]+100*seen[tu]));configs.append(row);print('TUNE',task,row,flush=True)
            if best is None or row['tune']>best[0]['tune']: best=row,z
    # Compare two prespecified nonlinear capacities on the combined features.
    zl=np.clip(x/rms,-8,8)@ir.a1.fit_pairwise_logistic(np.clip(x[tr]/rms,-8,8),l2=10)
    for leaves,reg in [(7,10),(15,10),(15,100),(31,100)]:
        model=ir.fit_hgb(x[tr],c[tr],leaves,reg);zh=ir.margin(model,x,c)
        for blend in [.5,1]:
            z=(1-blend)*zl+blend*zh
            row=dict(kind='hgb',leaves=leaves,reg=reg,blend=blend,tune=ir.acc(z[tu]+100*seen[tu]));configs.append(row);print('TUNE',task,row,flush=True)
            if best is None or row['tune']>best[0]['tune']:best=row,z
    selected,z=best
    b=(base[ev]+100*seen[ev])>0;p=(z[ev]+100*seen[ev])>0
    diff=p.astype(float)-b.astype(float);se=float(diff.std(ddof=1)/np.sqrt(len(ev)))
    result=dict(task=task,selected=selected,baseline_audit=float(b.mean()),candidate_audit=float(p.mean()),improvement=float(diff.mean()),paired_standard_error=se,n_audit=len(ev),gained=int(np.sum(p&~b)),lost=int(np.sum(b&~p)),configs=configs)
    print('AUDIT',json.dumps(result),flush=True)
    (ir.OUT/f'interactions_augmented_results_{task}.json').write_text(json.dumps(result,indent=2))
    rms0=np.sqrt(np.mean(x0*x0,axis=0))+1e-8;w0=ir.a1.fit_pairwise_logistic(np.clip(x0/rms0,-8,8));zt0=np.clip(d['xt']/rms0,-8,8)@w0
    rms=np.sqrt(np.mean(x*x,axis=0))+1e-8
    if selected['kind']=='linear':
        w=ir.a1.fit_pairwise_logistic(np.clip(x/rms,-8,8),l2=selected['linear_reg']);zt1=np.clip(xt/rms,-8,8)@w
        zt=(1-selected['blend'])*zt0+selected['blend']*zt1
    else:
        w=ir.a1.fit_pairwise_logistic(np.clip(x/rms,-8,8),l2=10);zt1=np.clip(xt/rms,-8,8)@w
        model=ir.fit_hgb(x,c,selected['leaves'],selected['reg']);zt2=ir.margin(model,xt,ct)
        zt=(1-selected['blend'])*zt1+selected['blend']*zt2
    zt+=100*d['st'];tests=ir.a1.read_test(ir.ROOT/'assignment1_2026'/f'task_{task}_test.csv');pred=[r['item1'] if z>=0 else r['item2'] for r,z in zip(tests,zt)]
    ir.a1.write_predictions(ir.ROOT/'assignment1_2026'/f'task_{task}_test.csv',ir.OUT/f'interactions_augmented_task_{task}_predictions.csv',pred)
    np.savez_compressed(ir.OUT/f'interactions_augmented_audit_{task}.npz',indices=ev,baseline=base[ev],candidate=z[ev],seen=seen[ev],test_margin=zt)

if __name__=='__main__':
    for task in map(int,sys.argv[1:] or [2,4,5]):run(task)
