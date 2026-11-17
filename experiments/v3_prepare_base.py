"""Cache fixed v2 validation features for independent v3 workstreams."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assignment1 as base
import production_ratings as ratings

ROOT = Path(__file__).resolve().parents[1]


def prepare(task, seed=258):
    data_path = ROOT / 'assignment1_2026' / ('tasks_1_2_train.csv' if task == 1 else 'tasks_3_4_train.csv')
    a, users, items = base.load(data_path)
    order = np.random.default_rng(seed).permutation(len(a))
    val, fit = a[order[:50000]], a[order[50000:]]
    config = ratings.RATING_CONFIGS[task]
    state = ratings.fit_rating_base(fit, len(users),len(items),task,
                                    seed=seed,fixed_epochs=config['best_epochs'],run_svdpp_grid=False)
    X, groups = ratings.build_rating_features(state,val[:,0],val[:,1],val[:,2])
    X = X[:,groups[config['best_stage_name']]]
    ridge,hgb = ratings.fit_fixed_rating_combiner(X[:20000],val[:20000,3],config,seed)
    pred = ratings.predict_stacked(ridge,hgb,config['best_eta'],X)
    path = ROOT / f'experiments/v3_base_{task}.npz'
    state_path=ROOT/f'experiments/v3_state_{task}.pkl'
    state_temp=state_path.with_suffix('.pending.pkl')
    with state_temp.open('wb') as f:
        pickle.dump(state,f,protocol=pickle.HIGHEST_PROTOCOL)
    state_temp.replace(state_path)
    temporary=path.with_suffix('.pending.npz')
    np.savez_compressed(temporary,X=X,y=val[:,3],valid=val,pred=pred,
                        groups_json=json.dumps({k:np.arange(X.shape[1])[v].tolist() for k,v in groups.items()}))
    temporary.replace(path)
    print(f'Cached {path}, X={X.shape}; tune MSE='+str(np.mean((pred[20000:35000][val[20000:35000,3]>0]-val[20000:35000,3][val[20000:35000,3]>0])**2)),flush=True)


if __name__ == '__main__':
    for t in [1,3]:
        prepare(t)
