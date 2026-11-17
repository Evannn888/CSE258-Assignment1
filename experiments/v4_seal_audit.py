"""Score the new audit once, after all candidate settings are locked."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import assignment1 as base
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments'


def main():
    report=json.loads((EXP/'v4/fresh_raw_report.json').read_text());choice=report['selected']
    p=np.load(EXP/'v4_fresh_all_predictions.npz')
    if choice['name'].endswith('blend'):
        w=choice['raw_weight'];pred=w*p['oof_raw']+(1-w)*p['v3_oof']
    else:pred=p[choice['name']]
    a,_,_=base.load(ROOT/'assignment1_2026/tasks_1_2_train.csv')
    idx=p['audit_indices'];truth=a[idx,3];keep=truth>0;users=a[idx[keep],0]
    new=pred[len(p['tune_indices']):][keep];old=p['v3_control'][len(p['tune_indices']):][keep]
    v3oof=p['v3_oof'][len(p['tune_indices']):][keep];truth=truth[keep]
    diff=(new-truth)**2-(old-truth)**2
    uniq,inv=np.unique(users,return_inverse=True);sums=np.bincount(inv,weights=diff);counts=np.bincount(inv)
    rng=np.random.default_rng(2029);boot=[]
    for _ in range(1000):
        draws=rng.integers(0,len(uniq),len(uniq));boot.append(sums[draws].sum()/counts[draws].sum())
    result=dict(choice=choice,seed=2029,audit_positive_rows=len(truth),audit_users=len(uniq),
       audit_v3_control_mse=float(np.mean((old-truth)**2)),audit_v3_oof_mse=float(np.mean((v3oof-truth)**2)),
       audit_v4_mse=float(np.mean((new-truth)**2)),paired_change=float(diff.mean()),
       cluster_bootstrap_se=float(np.std(boot,ddof=1)),cluster_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
       target=.5035,target_met_on_local_audit=bool(np.mean((new-truth)**2)<.5035),
       configuration_locked_before_audit=True)
    (EXP/'v4/frozen_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(EXP/'v4_locked_predictions.npz',pred=pred,tune_indices=p['tune_indices'],audit_indices=idx)
    print(result)


if __name__=='__main__':main()
