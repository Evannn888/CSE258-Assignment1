"""One final Task 1 stage ablation, selected only on the tuning window."""
import sys,json,pickle,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import v3_big as big
import v3_task1_stages as stages
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments'


def main():
    original=json.loads((EXP/'v3_big_results_1.json').read_text())
    names=['v3_big_results_1.json','v3_big_pred_1.npz','v3_big_combiner_1.pkl']
    for name in names:shutil.copy2(EXP/name,EXP/(name+'.before_stage'))
    with (EXP/'v3_big_state_1.pkl').open('rb') as f:bank=pickle.load(f)
    old=bank['old_state']
    state=stages.fit_stage_model(bank['fit_rows'],old['nu'],old['ni'],old['svdpp64'].y,old['biases'][2])
    data=np.load(EXP/'v3_big_1.npz');v=data['valid']
    F=stages.extract_stage_features(state,v[:,0],v[:,1]).astype(np.float32)
    with (EXP/'v3_big_stage_state_1.pkl').open('wb') as f:pickle.dump(state,f,protocol=5)
    path=EXP/'v3_big_stage_1.npz'
    big.atomic_npz(path,X=np.column_stack([data['X'],F]),y=data['y'],valid=v,
                   baselinepred=data['baselinepred'])
    report=big.fit_large_meta(1,path)
    use=report['selected_tune_mse'] < original['selected_tune_mse']-.0003
    decision=dict(use_stage=use,before_tune=original['selected_tune_mse'],stage_tune=report['selected_tune_mse'])
    if not use:
        for name in names:shutil.copy2(EXP/(name+'.before_stage'),EXP/name)
    chosen=json.loads((EXP/'v3_big_results_1.json').read_text())
    chosen['selected']['use_stage']=use
    (EXP/'v3_big_results_1.json').write_text(json.dumps(chosen,indent=2)+'\n')
    (EXP/'v3/stage_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print('STAGE DECISION',decision,flush=True)


if __name__=='__main__':main()
