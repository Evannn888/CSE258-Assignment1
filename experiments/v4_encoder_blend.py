"""Tune-only comparison of encoder features against the existing large-meta bank."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import production_ratings as old
ROOT=Path(__file__).resolve().parents[1]

def main():
 data=np.load(ROOT/'experiments/v3_big_stage_1.npz');X,y=data['X'].astype(np.float64),data['y'];bp=np.load(ROOT/'experiments/v3_big_pred_1.npz')['pred']
 tr=np.arange(150000)[y[:150000]>0];tu=np.arange(150000,175000)[y[150000:175000]>0]
 oldridge=old.StandardizedRidge(500.).fit(X[tr],y[tr]);raw=oldridge.predict_raw(X)
 baseline=float(np.mean((bp[tu]-y[tu])**2));records=[];bank=[];names=[];best=(baseline,None,bp)
 for file in sorted((ROOT/'experiments/v4_encoder').glob('*.npz')):
  z=np.load(file);bank.append(z['features']);names.append(file.stem)
 for which in [[i] for i in range(len(bank))]+[list(range(len(bank)))]:
  f=np.column_stack([bank[i] for i in which]);xx=np.column_stack([X,f]);models=[names[i] for i in which]
  for reg in [500.,2000.]:
   ridge=old.StandardizedRidge(reg).fit(xx[tr],y[tr]);delta=ridge.predict_raw(xx)-raw
   for scale in [.5,1.]:
    pred=np.clip(bp+scale*delta,1,5);mse=float(np.mean((pred[tu]-y[tu])**2));rec=dict(models=models,ridge_reg=reg,delta_scale=scale,tune_mse=mse);records.append(rec)
    if mse<best[0]:best=(mse,rec,pred)
 result=dict(baseline_tune=baseline,best_tune=best[0],selected=best[1],records=records)
 (ROOT/'experiments/v4_encoder_blend.json').write_text(json.dumps(result,indent=2));np.savez_compressed(ROOT/'experiments/v4_encoder_blend.npz',pred=best[2]);print(result,flush=True)

if __name__=='__main__':main()
