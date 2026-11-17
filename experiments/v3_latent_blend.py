import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import production_ratings as old
ROOT=Path(__file__).resolve().parents[1]
def run(task):
 data=np.load(ROOT/f'experiments/v3_base_{task}.npz');X,y,bp=data['X'],data['y'],data['pred']
 tr=np.arange(20000)[y[:20000]>0];tu=np.arange(20000,35000)[y[20000:35000]>0]
 conf=old.RATING_CONFIGS[task];oldridge=old.StandardizedRidge(conf['best_l2']).fit(X[tr],y[tr]);raw=oldridge.predict_raw(X)
 baseline=float(np.mean((bp[tu]-y[tu])**2));best=(baseline,[],None,bp);records=[]
 files=sorted((ROOT/'experiments/v3_latent_models').glob(f't{task}_*.npz'))
 bank=[];names=[]
 for f in files:
  z=np.load(f)
  if 'val_pred' not in z:continue
  bank.append(z['val_pred']);names.append(f.stem)
 for which in [[i] for i in range(len(bank))]+[list(range(len(bank)))]:
  preds=np.column_stack([bank[i] for i in which]);xx=np.column_stack([X,preds]);ns=[names[i] for i in which]
  for l2 in [10.,40.,100.,400.]:
   rr=old.StandardizedRidge(l2).fit(xx[tr],y[tr]);delta=rr.predict_raw(xx)-raw
   for scale in [.5,1.]:
    p=np.clip(bp+scale*delta,1,5);mse=float(np.mean((p[tu]-y[tu])**2));rec=dict(names=ns,l2=l2,scale=scale,tune=mse);records.append(rec)
    if mse<best[0]:best=(mse,ns,rec,p)
 print(task,'baseline',baseline,'best',best[:3],flush=True)
 (ROOT/f'experiments/v3_latent_blend_{task}.json').write_text(json.dumps(dict(baseline=baseline,selected=best[2],best_tune=best[0],records=records),indent=2))
 np.savez_compressed(ROOT/f'experiments/v3_latent_blend_{task}.npz',pred=best[3])
if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else 1)
